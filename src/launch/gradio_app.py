import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import yaml
from loguru import logger
from PIL import Image

from models.bagel.inference_alignment import (
    apply_training_alignment_defaults,
    merge_missing_infer_defaults,
)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Bagel Gradio inference server")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the fixed checkpoint directory to serve.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/bagel_interleaved/config.yaml",
        help="Training config used to recover processor and inference defaults.",
    )
    parser.add_argument(
        "--checkpoint_type",
        type=str,
        default="regular",
        choices=["regular", "ema"],
        help="Checkpoint type to load (default: regular).",
    )
    parser.add_argument(
        "--runtime_max_latent_size",
        type=int,
        default=None,
        help="Optional runtime override for model.max_latent_size.",
    )
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args(argv)


def build_input_list_from_request(
    text: str | None,
    image_paths: str | Sequence[str] | None,
) -> list[str | Image.Image]:
    inputs: list[str | Image.Image] = []

    if isinstance(image_paths, str):
        ordered_paths: Iterable[str] = [image_paths]
    else:
        ordered_paths = image_paths or []

    for image_path in ordered_paths:
        image = Image.open(image_path).convert("RGB")
        inputs.append(image.copy())
        image.close()

    normalized_text = (text or "").strip()
    if normalized_text:
        inputs.append(normalized_text)

    if not inputs:
        raise ValueError("Please provide at least one image or some text.")

    return inputs


def build_infer_kwargs(
    *,
    processor: Any,
    dataset_config: Mapping[str, Any] | None,
    infer_defaults: Mapping[str, Any] | None,
    think: bool,
    runtime_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    infer_kwargs = dict(runtime_overrides or {})
    infer_kwargs["think"] = bool(think)

    merge_missing_infer_defaults(infer_kwargs, infer_defaults or {})
    apply_training_alignment_defaults(infer_kwargs, processor, dataset_config)

    if infer_kwargs.get("use_action_token_control") is not True:
        infer_kwargs["use_action_token_control"] = True

    return infer_kwargs


def _output_to_pil(item: Any) -> Image.Image | None:
    if isinstance(item, Image.Image):
        return item.convert("RGB")

    image = item
    if isinstance(item, dict):
        image = item.get("image")
        if image is None:
            return None

    if torch.is_tensor(image):
        tensor = image.detach().float().cpu()
        if tensor.dim() == 4:
            tensor = tensor[0]
        if tensor.dim() != 3:
            return None
        if tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        array = (tensor.clamp(0, 1) * 255).to(torch.uint8).numpy()
        return Image.fromarray(array).convert("RGB")

    return None


def format_inference_result(
    outputs: Sequence[Any],
    diagnostics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    text_parts: list[str] = []
    images: list[Image.Image] = []

    for item in outputs:
        if isinstance(item, str):
            text_parts.append(item)
            continue

        if isinstance(item, dict):
            maybe_text = item.get("text")
            if isinstance(maybe_text, str) and maybe_text.strip():
                text_parts.append(maybe_text)

        maybe_image = _output_to_pil(item)
        if maybe_image is not None:
            images.append(maybe_image)

    return {
        "text": "\n\n".join(part for part in text_parts if part.strip()),
        "images": images,
        "diagnostics": json.dumps(
            dict(diagnostics or {}), ensure_ascii=False, indent=2
        ),
    }


class GradioInferenceService:
    def __init__(
        self,
        *,
        inferencer: Any,
        processor: Any,
        dataset_config: Mapping[str, Any] | None,
        infer_defaults: Mapping[str, Any] | None,
    ):
        self.inferencer = inferencer
        self.processor = processor
        self.dataset_config = dict(dataset_config or {})
        self.infer_defaults = dict(infer_defaults or {})

    def run(
        self,
        text: str,
        image_paths: Sequence[str] | None,
        think: bool,
        max_think_token_n: int,
        do_sample: bool,
        text_temperature: float,
        cfg_text_scale: float,
        cfg_img_scale: float,
        num_timesteps: int,
        timestep_shift: float,
        image_height: int,
        image_width: int,
        max_turn_text_tokens: int,
        max_auto_images: int,
    ):
        input_lists = build_input_list_from_request(text=text, image_paths=image_paths)
        runtime_overrides = {
            "max_think_token_n": int(max_think_token_n),
            "do_sample": bool(do_sample),
            "text_temperature": float(text_temperature),
            "cfg_text_scale": float(cfg_text_scale),
            "cfg_img_scale": float(cfg_img_scale),
            "num_timesteps": int(num_timesteps),
            "timestep_shift": float(timestep_shift),
            "image_shapes": (int(image_height), int(image_width)),
            "max_turn_text_tokens": int(max_turn_text_tokens),
            "max_auto_images": int(max_auto_images),
        }
        infer_kwargs = build_infer_kwargs(
            processor=self.processor,
            dataset_config=self.dataset_config,
            infer_defaults=self.infer_defaults,
            think=think,
            runtime_overrides=runtime_overrides,
        )

        with torch.no_grad():
            outputs = self.inferencer.interleave_inference(
                input_lists=input_lists,
                **infer_kwargs,
            )

        diagnostics = getattr(self.inferencer, "last_interleave_diagnostics", {})
        formatted = format_inference_result(outputs, diagnostics)
        return formatted["text"], formatted["images"], formatted["diagnostics"]


def _load_runtime_components(args) -> tuple[GradioInferenceService, Path]:
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    from launch.inference_cli import build_inferencer, build_processor, load_model

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
            logger.warning(
                "Detected max_latent_size mismatch; aligning model to processor for inference: "
                f"{int(model_max)} -> {int(proc_max)}"
            )
            model.max_latent_size = int(proc_max)
            if hasattr(model, "config"):
                setattr(model.config, "max_latent_size", int(proc_max))

    inferencer = build_inferencer(model, processor)
    probe_config = dict(
        config.get("trainer_args", {}).get("inference_probe_config", {}) or {}
    )
    infer_defaults = dict(probe_config.get("infer_kwargs", {}) or {})
    if "timestep_shift" not in infer_defaults:
        infer_defaults["timestep_shift"] = float(
            getattr(model.config, "timestep_shift", 1.0)
        )

    service = GradioInferenceService(
        inferencer=inferencer,
        processor=processor,
        dataset_config=config.get("dataset_config", {}),
        infer_defaults=infer_defaults,
    )
    return service, checkpoint_path


def _build_demo(service: GradioInferenceService, checkpoint_path: Path):
    import gradio as gr

    with gr.Blocks(title="Bagel Gradio Inference") as demo:
        gr.Markdown(
            "\n".join(
                [
                    "# Bagel Gradio Inference",
                    f"Serving checkpoint: `{checkpoint_path}`",
                    "Single-request UI with text, images, interleave reasoning, and action-token image generation.",
                ]
            )
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Text",
                    lines=8,
                    placeholder="Enter your prompt or question.",
                )
                image_input = gr.File(
                    label="Images",
                    file_types=["image"],
                    file_count="multiple",
                    type="filepath",
                )
                think_input = gr.Checkbox(
                    label="Interleave Reasoning",
                    value=True,
                )
                with gr.Accordion("Advanced", open=False):
                    max_think_token_n = gr.Slider(
                        label="Max Think Tokens",
                        minimum=1,
                        maximum=4096,
                        step=1,
                        value=1024,
                    )
                    do_sample = gr.Checkbox(label="Do Sample", value=False)
                    text_temperature = gr.Slider(
                        label="Text Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        step=0.05,
                        value=0.3,
                    )
                    cfg_text_scale = gr.Slider(
                        label="CFG Text Scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=float(service.infer_defaults.get("cfg_text_scale", 4.0)),
                    )
                    cfg_img_scale = gr.Slider(
                        label="CFG Image Scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=float(service.infer_defaults.get("cfg_img_scale", 2.0)),
                    )
                    num_timesteps = gr.Slider(
                        label="Num Timesteps",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=int(service.infer_defaults.get("num_timesteps", 50)),
                    )
                    timestep_shift = gr.Slider(
                        label="Timestep Shift",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=float(service.infer_defaults.get("timestep_shift", 3.0)),
                    )
                    with gr.Row():
                        image_height = gr.Number(
                            label="Image Height",
                            value=int(
                                service.infer_defaults.get("image_shapes", [512, 512])[
                                    0
                                ]
                            ),
                            precision=0,
                        )
                        image_width = gr.Number(
                            label="Image Width",
                            value=int(
                                service.infer_defaults.get("image_shapes", [512, 512])[
                                    1
                                ]
                            ),
                            precision=0,
                        )
                    max_turn_text_tokens = gr.Slider(
                        label="Max Turn Text Tokens",
                        minimum=1,
                        maximum=4096,
                        step=1,
                        value=int(
                            service.infer_defaults.get("max_turn_text_tokens", 1024)
                        ),
                    )
                    max_auto_images = gr.Slider(
                        label="Max Auto Images",
                        minimum=1,
                        maximum=8,
                        step=1,
                        value=4,
                    )
                submit = gr.Button("Run Inference", variant="primary")

            with gr.Column(scale=2):
                text_output = gr.Textbox(label="Text Output", lines=14)
                image_output = gr.Gallery(label="Generated Images", height="auto")
                diagnostics_output = gr.Textbox(label="Diagnostics", lines=18)

        submit.click(
            fn=service.run,
            inputs=[
                text_input,
                image_input,
                think_input,
                max_think_token_n,
                do_sample,
                text_temperature,
                cfg_text_scale,
                cfg_img_scale,
                num_timesteps,
                timestep_shift,
                image_height,
                image_width,
                max_turn_text_tokens,
                max_auto_images,
            ],
            outputs=[text_output, image_output, diagnostics_output],
        )

    demo.queue(default_concurrency_limit=1)
    return demo


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    service, checkpoint_path = _load_runtime_components(args)
    demo = _build_demo(service, checkpoint_path)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
