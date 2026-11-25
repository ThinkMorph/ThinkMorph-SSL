"""Standalone inference CLI that reuses training config settings.

Usage:
    # Use latest checkpoint (auto-detected from output_dir):
    bagel-infer --config examples/bagel_interleaved/config.yaml

    # Specify a checkpoint:
    bagel-infer --config examples/bagel_interleaved/config.yaml \
        --checkpoint ./output/bagel_interleaved_reasoning/checkpoint-100

    # Override image/text inputs:
    bagel-infer --config examples/bagel_interleaved/config.yaml \
        --image path/to/image.png --text "Your question here"

    # Save output images to a directory:
    bagel-infer --config examples/bagel_interleaved/config.yaml \
        --output_dir ./inference_outputs
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Suppress deprecation warnings from lpips/torchvision (pretrained API)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
from accelerate import init_empty_weights  # noqa: E402
from loguru import logger  # noqa: E402
from PIL import Image  # noqa: E402

from data.processor.bagel_processor import BagelDataProcessor  # noqa: E402
from data.processor.config import ProcessorConfig  # noqa: E402
from merger import FSDP2Merger  # noqa: E402
from models.bagel.bagel import Bagel, BagelConfig  # noqa: E402
from models.bagel.inference_alignment import (  # noqa: E402
    apply_training_alignment_defaults,
)
from models.bagel.inferencer import InterleaveInferencer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Bagel standalone inference")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml (same one used for training)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to DCP checkpoint directory. If not set, uses latest from output_dir.",
    )
    parser.add_argument(
        "--checkpoint_type",
        type=str,
        default="regular",
        choices=["regular", "ema"],
        help="Checkpoint type to load (default: regular)",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="*",
        default=None,
        help="Override input image path(s). If not set, uses probe samples from config.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Override input text/question. If not set, uses probe samples from config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_outputs",
        help="Directory to save output images (default: ./inference_outputs)",
    )
    parser.add_argument(
        "--runtime_max_latent_size",
        type=int,
        default=None,
        help=(
            "Optional runtime override for model.max_latent_size used by inference-side "
            "latent position IDs (for train/infer mismatch ablation)."
        ),
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        nargs="*",
        default=None,
        help="Ground truth image path(s) for quality comparison (LPIPS, SSIM, PSNR).",
    )
    parser.add_argument(
        "--prefill",
        type=str,
        default=None,
        help=(
            "Inject custom assistant reasoning text before image generation. "
            "Skips model text generation and directly triggers image generation "
            "with this text as context. Useful for testing how reasoning affects "
            "the generated image."
        ),
    )
    return parser.parse_args()


def resolve_checkpoint(config: dict, checkpoint_arg: str | None) -> Path:
    """Resolve checkpoint path from args or config output_dir."""
    if checkpoint_arg:
        path = Path(checkpoint_arg)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    output_dir = Path(config["trainer_args"]["output_dir"])
    if not output_dir.exists():
        raise FileNotFoundError(f"output_dir not found: {output_dir}")

    # Use FSDP2Merger's resolution logic (finds latest checkpoint-*)
    merger = FSDP2Merger()
    return merger._resolve_checkpoint_path(output_dir)


def load_model(
    checkpoint_path: Path, model_config: dict, checkpoint_type: str
) -> Bagel:
    """Load Bagel model from DCP checkpoint."""
    logger.info(f"Loading model config from {checkpoint_path}")
    bagel_config = BagelConfig.from_pretrained(checkpoint_path)

    # Apply overwrite_config from config.yaml
    for key, value in (model_config.get("overwrite_config") or {}).items():
        setattr(bagel_config, key, value)
    for key, value in (model_config.get("extra_kwargs") or {}).items():
        if hasattr(bagel_config, key):
            setattr(bagel_config, key, value)

    logger.info("Creating model template for DCP loading")
    with init_empty_weights():
        template = Bagel(bagel_config)

    logger.info(f"Loading weights from DCP checkpoint: {checkpoint_path}")
    merger = FSDP2Merger(checkpoint_type=checkpoint_type)
    state_dict = merger._load_full_state_dict(checkpoint_path, template.state_dict())

    logger.info("Building model and loading state dict")
    model = Bagel(bagel_config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        logger.warning(
            f"Model load has {len(missing_keys)} missing key(s); first 5: {missing_keys[:5]}"
        )
    if unexpected_keys:
        logger.warning(
            f"Model load has {len(unexpected_keys)} unexpected key(s); first 5: {unexpected_keys[:5]}"
        )
    model = model.to(dtype=torch.bfloat16, device="cuda").eval()
    return model


def build_processor(processor_config: dict) -> BagelDataProcessor:
    """Build BagelDataProcessor from config."""
    proc_config = ProcessorConfig(**processor_config)
    processor = BagelDataProcessor(proc_config)
    processor.build()
    return processor


def build_inferencer(
    model: Bagel, processor: BagelDataProcessor
) -> InterleaveInferencer:
    """Build InterleaveInferencer from model and processor."""
    return InterleaveInferencer(
        model=model,
        vae_model=model.vae_model,
        tokenizer=processor.processor,
        vae_transform=processor.vae_image_transform,
        vit_transform=processor.vit_image_transform,
        new_token_ids=processor.new_token_ids,
    )


def prepare_inputs_from_config(
    probe_config: dict,
) -> list[dict]:
    """Extract probe samples from inference_probe_config."""
    samples = probe_config.get("samples", [])
    samples_path = probe_config.get("samples_path")
    if samples_path and not samples:
        import json

        with open(samples_path) as f:
            for line in f:
                samples.append(json.loads(line))
    return samples


def prepare_input_list(
    sample: dict,
) -> list[str | Image.Image]:
    """Convert a probe sample dict to the input_lists format for inferencer."""
    input_terms = []
    for term in sample.get("inputs", []):
        if term.get("type") == "text":
            input_terms.append(term["text"])
        elif term.get("type") in {"image", "image_path"}:
            path = term.get("path", term.get("image"))
            path = os.path.abspath(os.path.expanduser(str(path)))
            input_terms.append(Image.open(path).convert("RGB"))
        else:
            raise ValueError(f"Unsupported input type: {term.get('type')}")
    return input_terms


def save_outputs(outputs: list, output_dir: Path, sample_id: str):
    """Save inference outputs (text printed, images saved to disk)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_idx = 0
    for item in outputs:
        if isinstance(item, str):
            print(f"  [Text]: {item}")
        elif isinstance(item, Image.Image):
            save_path = output_dir / f"{sample_id}_image_{image_idx}.png"
            item.convert("RGB").save(save_path)
            print(f"  [Image]: saved to {save_path}")
            image_idx += 1
        elif isinstance(item, dict):
            img = item.get("image")
            if img is not None:
                if isinstance(img, Image.Image):
                    pass
                elif torch.is_tensor(img):
                    img = img.detach().float().cpu()
                    if img.shape[0] in (1, 3):
                        img = img.permute(1, 2, 0)
                    img = (img.clamp(0, 1) * 255).to(torch.uint8).numpy()
                    img = Image.fromarray(img)
                else:
                    print(f"  [Image]: unhandled type {type(img)}, skipping")
                    continue
                save_path = output_dir / f"{sample_id}_image_{image_idx}.png"
                img.convert("RGB").save(save_path)
                print(f"  [Image]: saved to {save_path}")
                image_idx += 1
        elif torch.is_tensor(item):
            img = item.detach().float().cpu()
            if img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)
            img = (img.clamp(0, 1) * 255).to(torch.uint8).numpy()
            img = Image.fromarray(img)
            save_path = output_dir / f"{sample_id}_image_{image_idx}.png"
            img.save(save_path)
            print(f"  [Image]: saved to {save_path}")
            image_idx += 1


def _pil_to_tensor(
    img: Image.Image, target_size: tuple[int, int] | None = None
) -> torch.Tensor:
    """Convert PIL image to (1, C, H, W) float [0, 1] tensor."""
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _align_gt_image_for_metrics(
    img: Image.Image, processor: BagelDataProcessor | None = None
) -> Image.Image:
    aligned = img.convert("RGB")
    if processor is None:
        return aligned
    vae_transform = getattr(processor, "vae_image_transform", None)
    resize_transform = getattr(vae_transform, "resize_transform", None)
    if not callable(resize_transform):
        return aligned
    try:
        resized = resize_transform(aligned)
    except Exception as exc:
        logger.warning(f"Failed to align GT image for metrics: {exc}")
        return aligned
    if isinstance(resized, Image.Image):
        return resized.convert("RGB")
    return aligned


def _output_to_pil(item) -> Image.Image | None:
    """Extract a PIL image from an inference output item."""
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    img = item
    if isinstance(item, dict):
        img = item.get("image")
        if img is None:
            return None
    if torch.is_tensor(img):
        img = img.detach().float().cpu()
        if img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)
        img = (img.clamp(0, 1) * 255).to(torch.uint8).numpy()
        return Image.fromarray(img).convert("RGB")
    return None


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    mse = ((img1 - img2) ** 2).mean()
    if mse <= 0:
        return 100.0
    return (10.0 * torch.log10(1.0 / mse)).item()


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    c1, c2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(img1, 11, 1, 5)
    mu2 = F.avg_pool2d(img2, 11, 1, 5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1 * img1, 11, 1, 5) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, 11, 1, 5) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 11, 1, 5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().item()


def compute_gt_metrics(
    gen_images: list[Image.Image],
    gt_paths: list[str],
    output_dir: Path | None = None,
    sample_id: str = "sample",
    processor: BagelDataProcessor | None = None,
) -> dict:
    """Compare generated images against ground truth, print and return metrics."""
    gt_count = len(gt_paths)
    gen_count = len(gen_images)
    if gt_count == 0:
        return {}

    lpips_fn = None
    try:
        import lpips as lpips_lib

        lpips_fn = lpips_lib.LPIPS(net="alex", verbose=False).eval()
    except ImportError:
        logger.warning(
            "lpips package not available; skipping LPIPS. Install: pip install lpips"
        )
    except Exception as exc:
        logger.warning(f"Failed to init LPIPS: {exc}")

    num_pairs = min(gen_count, gt_count)
    all_metrics = []

    print(f"\n{'─' * 60}")
    print(
        f"Ground Truth Comparison ({num_pairs} pair(s), gen={gen_count}, gt={gt_count})"
    )
    print(f"{'─' * 60}")

    for i in range(num_pairs):
        gt_path = gt_paths[i]
        try:
            raw_gt_img = Image.open(gt_path).convert("RGB")
        except Exception as exc:
            print(f"  Pair {i}: failed to load GT {gt_path}: {exc}")
            continue
        gt_img = _align_gt_image_for_metrics(raw_gt_img, processor=processor)

        gen_img = gen_images[i]
        gen_size = (gen_img.height, gen_img.width)
        gt_size = (gt_img.height, gt_img.width)
        raw_gt_size = (raw_gt_img.height, raw_gt_img.width)

        if raw_gt_size != gt_size:
            print(
                f"  Pair {i}: aligning raw GT {raw_gt_size[1]}x{raw_gt_size[0]} "
                f"to training size {gt_size[1]}x{gt_size[0]}"
            )

        if gen_size != gt_size:
            compare_size = (
                min(gen_size[0], gt_size[0]),
                min(gen_size[1], gt_size[1]),
            )
            print(
                f"  Pair {i}: resolution mismatch gen={gen_size[1]}x{gen_size[0]} "
                f"vs aligned_gt={gt_size[1]}x{gt_size[0]}, comparing at {compare_size[1]}x{compare_size[0]}"
            )
        else:
            compare_size = None

        gt_tensor = _pil_to_tensor(gt_img, target_size=compare_size)
        gen_tensor = _pil_to_tensor(gen_img, target_size=compare_size)

        print(f"  Pair {i} diagnostics:")
        print(
            f"    gen: size={gen_size[1]}x{gen_size[0]}, "
            f"tensor={list(gen_tensor.shape)}, "
            f"range=[{gen_tensor.min():.4f}, {gen_tensor.max():.4f}], "
            f"mean={gen_tensor.mean():.4f}"
        )
        print(
            f"    gt:  size={gt_size[1]}x{gt_size[0]}, "
            f"tensor={list(gt_tensor.shape)}, "
            f"range=[{gt_tensor.min():.4f}, {gt_tensor.max():.4f}], "
            f"mean={gt_tensor.mean():.4f}"
        )
        abs_diff = (gen_tensor - gt_tensor).abs()
        print(
            f"    diff: mean={abs_diff.mean():.4f}, "
            f"max={abs_diff.max():.4f}, "
            f"median={abs_diff.median():.4f}"
        )

        pair = {}

        try:
            pair["psnr"] = compute_psnr(gen_tensor, gt_tensor)
        except Exception as exc:
            pair["psnr"] = None
            logger.warning(f"PSNR failed for pair {i}: {exc}")

        try:
            pair["ssim"] = compute_ssim(gen_tensor, gt_tensor)
        except Exception as exc:
            pair["ssim"] = None
            logger.warning(f"SSIM failed for pair {i}: {exc}")

        if lpips_fn is not None:
            try:
                with torch.no_grad():
                    score = lpips_fn(gen_tensor * 2 - 1, gt_tensor * 2 - 1)
                pair["lpips"] = score.item()
            except Exception as exc:
                pair["lpips"] = None
                logger.warning(f"LPIPS failed for pair {i}: {exc}")
        else:
            pair["lpips"] = None

        parts = []
        if pair.get("lpips") is not None:
            parts.append(f"LPIPS={pair['lpips']:.4f}")
        if pair.get("ssim") is not None:
            parts.append(f"SSIM={pair['ssim']:.4f}")
        if pair.get("psnr") is not None:
            parts.append(f"PSNR={pair['psnr']:.2f}dB")
        print(f"  Pair {i}: {', '.join(parts) or 'no metrics'}")
        print(f"    GT: {gt_path}")

        if output_dir is not None:
            gt_save = output_dir / f"{sample_id}_gt_{i}.png"
            gt_img.save(gt_save)
            print(f"    GT saved: {gt_save}")

            diff_vis = (abs_diff.squeeze(0).permute(1, 2, 0) * 5.0).clamp(0, 1)
            diff_vis = (diff_vis * 255).to(torch.uint8).numpy()
            diff_save = output_dir / f"{sample_id}_diff_{i}.png"
            Image.fromarray(diff_vis).save(diff_save)
            print(f"    Diff saved: {diff_save} (5x amplified)")

        all_metrics.append(pair)

    if all_metrics:

        def _avg(key):
            vals = [p[key] for p in all_metrics if p.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        avg_lpips = _avg("lpips")
        avg_ssim = _avg("ssim")
        avg_psnr = _avg("psnr")
        avg_parts = []
        if avg_lpips is not None:
            avg_parts.append(f"LPIPS={avg_lpips:.4f}")
        if avg_ssim is not None:
            avg_parts.append(f"SSIM={avg_ssim:.4f}")
        if avg_psnr is not None:
            avg_parts.append(f"PSNR={avg_psnr:.2f}dB")
        print(f"  Average: {', '.join(avg_parts)}")
        print(f"  Count match: {gen_count}/{gt_count}")

    return {
        "gt_count": gt_count,
        "gen_count": gen_count,
        "per_pair": all_metrics,
    }


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    logger.info(f"Using checkpoint: {checkpoint_path}")

    model = load_model(checkpoint_path, config["model_config"], args.checkpoint_type)
    if args.runtime_max_latent_size is not None:
        old_value = int(getattr(model, "max_latent_size", -1))
        model.max_latent_size = int(args.runtime_max_latent_size)
        logger.warning(
            f"Runtime override: model.max_latent_size {old_value} -> {model.max_latent_size}"
        )

    processor = build_processor(config["dataset_config"]["processor_config"])
    if args.runtime_max_latent_size is None:
        model_max = getattr(model, "max_latent_size", None)
        proc_max = getattr(processor, "max_latent_size", None)
        if (
            model_max is not None
            and proc_max is not None
            and int(model_max) != int(proc_max)
        ):
            new_max = int(proc_max)
            logger.warning(
                "Detected max_latent_size mismatch; aligning model to processor for inference: "
                f"{int(model_max)} -> {new_max}"
            )
            model.max_latent_size = new_max
            if hasattr(model, "config"):
                setattr(model.config, "max_latent_size", new_max)
            # Rebuild latent_pos_embed to match the new grid size.
            if hasattr(model, "latent_pos_embed"):
                from models.bagel.bagel import PositionEmbedding

                hidden_size = model.latent_pos_embed.pos_embed.shape[-1]
                device = model.latent_pos_embed.pos_embed.device
                dtype = model.latent_pos_embed.pos_embed.dtype
                model.latent_pos_embed = PositionEmbedding(new_max, hidden_size).to(
                    device=device, dtype=dtype
                )
                logger.warning(
                    f"Rebuilt latent_pos_embed with max_latent_size={new_max}"
                )

    inferencer = build_inferencer(model, processor)

    probe_config = config["trainer_args"].get("inference_probe_config", {})
    infer_kwargs = dict(probe_config.get("infer_kwargs", {}))
    if "timestep_shift" not in infer_kwargs:
        infer_kwargs["timestep_shift"] = float(
            getattr(model.config, "timestep_shift", 1.0)
        )
        logger.info(
            "infer_kwargs.timestep_shift is not set; using model.config.timestep_shift="
            f"{infer_kwargs['timestep_shift']}"
        )
    apply_training_alignment_defaults(infer_kwargs, processor, config["dataset_config"])

    output_dir = Path(args.output_dir)

    if args.image is not None or args.text is not None:
        inputs = []
        for img_path in args.image or []:
            inputs.append({"type": "image", "path": img_path})
        if args.text:
            inputs.append({"type": "text", "text": args.text})
        gt = [{"path": os.path.abspath(p)} for p in (args.ground_truth or [])]
        samples = [
            {
                "id": "cli_input",
                "inputs": inputs,
                "kwargs": {},
                "ground_truth": gt,
            }
        ]
    else:
        samples = prepare_inputs_from_config(probe_config)
        if not samples:
            logger.error(
                "No probe samples found in config and no --image/--text provided."
            )
            return

    for sample in samples:
        sample_id = sample.get("id", sample.get("name", "sample"))
        kwargs = dict(infer_kwargs)
        kwargs.update(sample.get("kwargs", {}))

        input_list = prepare_input_list(sample)
        logger.info(
            f"Running inference for sample '{sample_id}' "
            f"({len(input_list)} inputs, kwargs: {kwargs})"
        )

        if args.prefill is not None:
            kwargs["prefill_assistant_text"] = args.prefill

        with torch.no_grad():
            outputs = inferencer.interleave_inference(input_lists=input_list, **kwargs)

        diagnostics = getattr(inferencer, "last_interleave_diagnostics", {})
        print(f"\n{'=' * 60}")
        print(f"Sample: {sample_id}")
        print(f"Stop reason: {diagnostics.get('stop_reason', 'unknown')}")
        print(f"Text segments: {diagnostics.get('text_segments', 0)}")
        print(f"Images generated: {diagnostics.get('image_outputs', 0)}")
        predicted_sizes = diagnostics.get("predicted_image_sizes", [])
        if predicted_sizes:
            for i, entry in enumerate(predicted_sizes):
                if isinstance(entry, dict):
                    print(f"  Image {i} size: {entry['size']} ({entry['source']})")
                else:
                    print(f"  Image {i} size: {entry}")
        else:
            print("  No size tokens decoded")
        print(f"{'=' * 60}")
        save_outputs(outputs, output_dir, sample_id)

        gt_entries = sample.get("ground_truth", [])
        gt_paths = []
        for gt in gt_entries:
            if isinstance(gt, str):
                gt_paths.append(os.path.abspath(gt))
            elif isinstance(gt, dict):
                gt_paths.append(os.path.abspath(gt.get("path", gt.get("image", ""))))
        if gt_paths:
            gen_images = [
                pil for item in outputs if (pil := _output_to_pil(item)) is not None
            ]
            compute_gt_metrics(
                gen_images,
                gt_paths,
                output_dir,
                sample_id,
                processor=processor,
            )

    print(f"\nDone. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
