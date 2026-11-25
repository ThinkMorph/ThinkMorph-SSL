from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image as PILImage
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)

from models import MONKEY_PATCHER
from models.bagel.inference_alignment import apply_training_alignment_defaults
from trainer.fsdp2.components import InferenceProbeSchedule
from utils.distributed import dist_barrier, object_collective_device


class BagelProbeMixin:
    def _setup_inference_probes(self):
        if not self.enable_inference_probes:
            return
        cfg = self.inference_probe_config
        enabled = bool(cfg.get("enabled", False))
        interval = int(cfg.get("interval_steps", 0) or 0)
        if not enabled or interval <= 0:
            return

        self.inference_probe_interval_steps = interval
        self.inference_probe_start_step = int(
            cfg.get("start_step", self.inference_probe_interval_steps)
        )
        self.inference_probe_max_samples = int(cfg.get("max_samples", 0) or 0)
        self.inference_probe_default_kwargs = dict(cfg.get("infer_kwargs", {}) or {})
        self.inference_probe_compare_ema = bool(cfg.get("compare_ema", False))
        self._apply_probe_training_alignment_defaults()
        self.inference_probe_use_rank0_unsharded = bool(
            cfg.get("use_rank0_unsharded", True)
        )
        samples = []
        samples_path = cfg.get("samples_path")
        if samples_path:
            samples.extend(self._load_probe_samples_from_path(samples_path))
        inline_samples = cfg.get("samples", [])
        if inline_samples:
            if not isinstance(inline_samples, list):
                raise ValueError("inference_probe_config.samples must be a list")
            samples.extend(inline_samples)

        normalized = []
        for idx, sample in enumerate(samples):
            normalized.append(self._normalize_probe_sample(sample, idx))

        if not normalized:
            logger.warning(
                "inference_probe_config is enabled but no valid samples were provided. Disabling probes."
            )
            return

        self.inference_probe_samples = normalized
        if self.inference_probe_max_samples > 0:
            self.inference_probe_samples = self.inference_probe_samples[
                : self.inference_probe_max_samples
            ]
        self.inference_probe_enabled = True
        self.probe_schedule = InferenceProbeSchedule(
            enabled=True,
            interval_steps=self.inference_probe_interval_steps,
            start_step=self.inference_probe_start_step,
        )
        logger.info(
            f"Inference probes enabled: {len(self.inference_probe_samples)} sample(s), "
            f"interval={self.inference_probe_interval_steps}, start_step={self.inference_probe_start_step}"
        )

    @staticmethod
    def _is_dist_initialized() -> bool:
        return dist.is_available() and dist.is_initialized()

    def _world_size(self) -> int:
        return dist.get_world_size() if self._is_dist_initialized() else 1

    def _rank(self) -> int:
        return dist.get_rank() if self._is_dist_initialized() else 0

    def _apply_probe_training_alignment_defaults(self) -> None:
        """Keep probe inference behavior aligned with current training settings.

        Explicit `inference_probe_config.infer_kwargs` always takes precedence.
        This only fills missing probe kwargs from dataset/processor training config.
        """
        apply_training_alignment_defaults(
            self.inference_probe_default_kwargs,
            self.processing_class,
            self.train_dataset,
        )

    def _load_probe_samples_from_path(self, samples_path: str) -> List[Dict[str, Any]]:
        samples_path = os.path.abspath(os.path.expanduser(samples_path))
        if not os.path.exists(samples_path):
            raise FileNotFoundError(
                f"inference_probe_config.samples_path not found: {samples_path}"
            )
        if samples_path.endswith(".jsonl"):
            rows = []
            with open(samples_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return rows

        with open(samples_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if isinstance(data.get("samples"), list):
                return data["samples"]
            raise ValueError(
                "Probe JSON must be a list or a dict containing a 'samples' list"
            )
        raise ValueError("Unsupported probe file content type")

    def _normalize_probe_sample(self, sample: Any, idx: int) -> Dict[str, Any]:
        if isinstance(sample, str):
            return {
                "id": f"probe_{idx}",
                "inputs": [{"type": "text", "text": sample}],
                "kwargs": {},
                "ground_truth": [],
            }

        if not isinstance(sample, dict):
            raise ValueError(f"Probe sample #{idx} must be a dict or str")

        sample_id = str(sample.get("id", sample.get("name", f"probe_{idx}")))
        kwargs = dict(sample.get("kwargs", sample.get("infer_kwargs", {}) or {}))

        inputs = sample.get("inputs")
        if inputs is None:
            inputs = []
            text = sample.get("text", sample.get("question"))
            if isinstance(text, str) and text.strip():
                inputs.append({"type": "text", "text": text.strip()})
            images = sample.get("images", [])
            if isinstance(images, str):
                images = [images]
            for image_path in images:
                inputs.append({"type": "image", "path": image_path})

        if not isinstance(inputs, list) or len(inputs) == 0:
            raise ValueError(
                f"Probe sample '{sample_id}' must provide a non-empty 'inputs' list"
            )

        normalized_inputs = []
        for term in inputs:
            if isinstance(term, str):
                normalized_inputs.append({"type": "text", "text": term})
                continue
            if not isinstance(term, dict):
                raise ValueError(
                    f"Probe sample '{sample_id}' contains invalid input item: {term}"
                )

            term_type = term.get("type")
            if term_type == "text":
                text = term.get("text")
                if not isinstance(text, str):
                    raise ValueError(
                        f"Probe sample '{sample_id}' has text input without string 'text'"
                    )
                normalized_inputs.append({"type": "text", "text": text})
                continue

            image_path = None
            if term_type in {"image", "image_path"}:
                image_path = term.get("path", term.get("image"))
            elif term_type == "image_url":
                image_url = term.get("image_url")
                if isinstance(image_url, dict):
                    image_path = image_url.get("url")
            if image_path is None:
                raise ValueError(
                    f"Probe sample '{sample_id}' has unsupported input item: {term}"
                )
            image_path = os.path.abspath(os.path.expanduser(str(image_path)))
            if not os.path.exists(image_path):
                raise FileNotFoundError(
                    f"Probe sample '{sample_id}' image not found: {image_path}"
                )
            normalized_inputs.append({"type": "image", "path": image_path})

        # Parse optional ground_truth images for quality metrics
        ground_truth = []
        raw_gt = sample.get("ground_truth", [])
        if isinstance(raw_gt, list):
            for gt_idx, gt_entry in enumerate(raw_gt):
                if isinstance(gt_entry, str):
                    gt_path = gt_entry
                elif isinstance(gt_entry, dict):
                    gt_path = gt_entry.get("path", gt_entry.get("image"))
                else:
                    raise ValueError(
                        f"Probe sample '{sample_id}' ground_truth[{gt_idx}] must be a dict or str"
                    )
                if gt_path is None:
                    raise ValueError(
                        f"Probe sample '{sample_id}' ground_truth[{gt_idx}] missing 'path'"
                    )
                gt_path = os.path.abspath(os.path.expanduser(str(gt_path)))
                if not os.path.exists(gt_path):
                    raise FileNotFoundError(
                        f"Probe sample '{sample_id}' ground_truth image not found: {gt_path}"
                    )
                ground_truth.append({"path": gt_path})

        return {
            "id": sample_id,
            "inputs": normalized_inputs,
            "kwargs": kwargs,
            "ground_truth": ground_truth,
        }

    def _should_run_inference_probe(self) -> bool:
        return self.probe_schedule.should_run(self.global_step)

    @staticmethod
    def _get_probe_variants(
        use_ema: bool, compare_ema: bool
    ) -> List[Tuple[Optional[str], bool]]:
        variants: List[Tuple[Optional[str], bool]] = [("regular", False)]
        if use_ema and compare_ema:
            variants.append(("ema", True))
        elif use_ema:
            variants = [(None, True)]
        return variants

    def _get_probe_inferencer(self):
        if self._probe_inferencer is not None:
            return self._probe_inferencer

        if self.processing_class is None:
            raise ValueError("processing_class is required for inference probes")

        required_attrs = [
            "processor",
            "vae_image_transform",
            "vit_image_transform",
            "new_token_ids",
        ]
        for attr in required_attrs:
            if not hasattr(self.processing_class, attr):
                raise ValueError(
                    f"processing_class is missing required attribute '{attr}' for inference probes"
                )
        if not hasattr(self.model, "vae_model"):
            raise ValueError("model must provide vae_model for inference probes")

        from models.bagel.inferencer import InterleaveInferencer

        self._probe_inferencer = InterleaveInferencer(
            model=self.model,
            vae_model=self.model.vae_model,
            tokenizer=self.processing_class.processor,
            vae_transform=self.processing_class.vae_image_transform,
            vit_transform=self.processing_class.vit_image_transform,
            new_token_ids=self.processing_class.new_token_ids,
        )
        return self._probe_inferencer

    def _unwrap_fsdp_model(self) -> nn.Module:
        candidate = self.fsdp2_model
        seen_ids = set()
        while True:
            next_candidate = None
            for attr_name in ("module", "_forward_module", "_orig_mod"):
                next_candidate = getattr(candidate, attr_name, None)
                if next_candidate is not None:
                    break
            if (
                next_candidate is None
                or next_candidate is candidate
                or id(next_candidate) in seen_ids
            ):
                break
            seen_ids.add(id(next_candidate))
            candidate = next_candidate
        return candidate

    def _build_rank0_probe_model(self, full_state_dict: Dict[str, Any]) -> nn.Module:
        from accelerate import init_empty_weights

        logger.info("Rank0 probe: building non-sharded model from config")
        probe_template = self._unwrap_fsdp_model()
        config = probe_template.config
        setattr(
            config,
            "_attn_implementation",
            getattr(config, "_attn_implementation", "flash_attention_2"),
        )
        model_class = type(probe_template)
        with init_empty_weights():
            model = model_class(config)
        logger.info("Rank0 probe: model instance created")

        patch_kwargs = {"use_rmpad": self.args.use_rmpad, "patch_type": []}
        if self.args.use_liger_kernel:
            patch_kwargs["patch_type"].append("liger")
        if patch_kwargs["patch_type"]:
            logger.info(f"Rank0 probe: applying patches {patch_kwargs['patch_type']}")
            MONKEY_PATCHER.apply_monkey_patch_to_instance(model, **patch_kwargs)

        logger.info("Rank0 probe: loading consolidated state dict")
        missing_keys, unexpected_keys = model.load_state_dict(
            full_state_dict, assign=True, strict=False
        )
        logger.info("Rank0 probe: state dict load finished")
        if missing_keys:
            logger.warning(
                f"Rank0 probe model missing {len(missing_keys)} key(s) while loading consolidated state dict."
            )
        if unexpected_keys:
            logger.warning(
                f"Rank0 probe model encountered {len(unexpected_keys)} unexpected key(s) while loading consolidated state dict."
            )

        logger.info("Rank0 probe: moving model to device")
        model = model.to(self.fsdp2_model.device)
        model.eval()
        logger.info("Rank0 probe: model ready")
        return model

    def _broadcast_state_dict_from_rank0(
        self,
        state_dict_on_rank0: Dict[str, Any],
        rank: int,
        world_size: int,
    ) -> Dict[str, torch.Tensor]:
        """Broadcast state dict from rank0 to all ranks via per-tensor GPU broadcast.

        FSDP2 ``get_model_state_dict(full_state_dict=True)`` only returns the
        full state dict on rank 0.  Instead of pickling the entire dict (which
        caused CPU OOM with 4 processes), this sends each tensor individually
        through GPU ``dist.broadcast``.
        """
        # Step 1: broadcast metadata (key names + shapes + dtypes) – tiny payload
        if rank == 0:
            metadata = [
                {"key": k, "shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in state_dict_on_rank0.items()
            ]
            total_bytes = sum(
                v.numel() * v.element_size() for v in state_dict_on_rank0.values()
            )
            logger.info(
                f"Broadcasting state dict: {len(metadata)} tensors, "
                f"{total_bytes / 1e9:.1f} GB"
            )
        else:
            metadata = None

        meta_list = [metadata]
        dist.broadcast_object_list(
            meta_list,
            src=0,
            device=object_collective_device(),
        )
        metadata = meta_list[0]

        _dtype_map = {
            "torch.float32": torch.float32,
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.int64": torch.int64,
            "torch.int32": torch.int32,
            "torch.int16": torch.int16,
            "torch.int8": torch.int8,
            "torch.uint8": torch.uint8,
            "torch.bool": torch.bool,
        }

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        result: Dict[str, torch.Tensor] = {}

        for i, meta in enumerate(metadata):
            key = meta["key"]
            shape = tuple(meta["shape"])
            dtype = _dtype_map.get(meta["dtype"], torch.float32)

            if rank == 0:
                # pop() to progressively free rank0 CPU memory
                gpu_tensor = state_dict_on_rank0.pop(key).to(device=device, dtype=dtype)
            else:
                gpu_tensor = torch.empty(shape, dtype=dtype, device=device)

            dist.broadcast(gpu_tensor, src=0)
            # Keep on GPU — avoids 4×58 GB CPU accumulation that caused OOM
            result[key] = gpu_tensor

            if (i + 1) % 100 == 0:
                if rank == 0:
                    logger.info(f"Broadcast progress: {i + 1}/{len(metadata)} tensors")

        logger.info(f"[rank {rank}] State dict broadcast complete")
        return result

    @staticmethod
    def _offload_ema_to_cpu(ema) -> list:
        """Move all EMA state tensors to CPU to free GPU memory for probes."""
        if not ema.is_enabled() or not ema.initialized:
            return []
        moved = []
        for name, ema_val in ema._ema_params.items():
            local = ema._local_tensor(ema_val)
            if isinstance(local, torch.Tensor) and local.is_cuda:
                orig_device = local.device
                local.data = local.data.to("cpu", non_blocking=True)
                moved.append((local, orig_device))
        if moved and torch.cuda.is_available():
            torch.cuda.synchronize()
        return moved

    @staticmethod
    def _reload_ema_to_gpu(moved: list):
        """Restore EMA state tensors back to their original GPU devices."""
        for local, orig_device in moved:
            local.data = local.data.to(orig_device, non_blocking=True)
        if moved and torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _offload_optimizer_to_cpu(optimizer: torch.optim.Optimizer) -> list:
        """Move all optimizer state tensors to CPU to free GPU memory for probes."""
        moved = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p not in optimizer.state:
                    continue
                for key, val in optimizer.state[p].items():
                    if isinstance(val, torch.Tensor) and val.is_cuda:
                        optimizer.state[p][key] = val.to("cpu", non_blocking=True)
                        moved.append((p, key, val.device))
        torch.cuda.synchronize()
        return moved

    @staticmethod
    def _reload_optimizer_to_gpu(optimizer: torch.optim.Optimizer, moved: list):
        """Restore optimizer state tensors back to their original GPU devices."""
        for p, key, orig_device in moved:
            if p in optimizer.state and key in optimizer.state[p]:
                optimizer.state[p][key] = optimizer.state[p][key].to(
                    orig_device, non_blocking=True
                )
        torch.cuda.synchronize()

    def _run_rank0_unsharded_probes(self, rank: int):
        import gc

        probe_model = None
        inferencer = None
        full_state_dict = None
        use_ema = self.ema.is_enabled() and self.ema.initialized
        offloaded = []
        ema_offloaded = []
        try:
            # Only offload optimizer on rank0 to save CPU RAM
            if rank == 0 and hasattr(self, "optimizer") and self.optimizer is not None:
                logger.info("Rank0 probe: offloading optimizer states to CPU")
                offloaded = self._offload_optimizer_to_cpu(self.optimizer)
                torch.cuda.empty_cache()
                logger.info(
                    f"Rank0 probe: offloaded {len(offloaded)} optimizer tensors"
                )
            variants = self._get_probe_variants(
                use_ema=use_ema, compare_ema=self.inference_probe_compare_ema
            )
            for variant_name, variant_use_ema in variants:
                variant_label = variant_name or "ema"
                if variant_use_ema:
                    self.ema.swap_with_model()
                try:
                    full_state_dict = get_model_state_dict(
                        self.fsdp2_model,
                        options=StateDictOptions(
                            full_state_dict=True, cpu_offload=True
                        ),
                    )
                finally:
                    if variant_use_ema:
                        self.ema.swap_with_model()

                if rank != 0:
                    continue

                # Free GPU memory before building the full probe model:
                # offload EMA shard-local copies and release gradients.
                if self.ema.is_enabled() and self.ema.initialized:
                    logger.info("Rank0 probe: offloading EMA states to CPU")
                    ema_offloaded = self._offload_ema_to_cpu(self.ema)
                    logger.info(
                        f"Rank0 probe: offloaded {len(ema_offloaded)} EMA tensors"
                    )
                self.fsdp2_model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()

                logger.info(
                    f"Running inference probes at step {self.global_step} via rank0 "
                    f"unsharded model ({variant_label} weights)"
                )

                from models.bagel.inferencer import InterleaveInferencer

                probe_model = self._build_rank0_probe_model(full_state_dict)
                inferencer = InterleaveInferencer(
                    model=probe_model,
                    vae_model=probe_model.vae_model,
                    tokenizer=self.processing_class.processor,
                    vae_transform=self.processing_class.vae_image_transform,
                    vit_transform=self.processing_class.vit_image_transform,
                    new_token_ids=self.processing_class.new_token_ids,
                )
                logger.info(
                    f"Rank0 probe: inferencer ready, starting sample inference ({variant_label})"
                )

                outputs_by_sample = []
                with torch.no_grad():
                    for sample in self.inference_probe_samples:
                        probe_kwargs = dict(self.inference_probe_default_kwargs)
                        probe_kwargs.update(sample.get("kwargs", {}))
                        input_terms = self._prepare_probe_inputs(sample)
                        outputs = inferencer.interleave_inference(
                            input_lists=input_terms, **probe_kwargs
                        )
                        diagnostics = dict(
                            getattr(inferencer, "last_interleave_diagnostics", {}) or {}
                        )
                        entry = {
                            "sample_id": sample["id"],
                            "outputs": outputs,
                            "diagnostics": diagnostics,
                            "ground_truth": sample.get("ground_truth", []),
                        }
                        gt_comparison = self._compute_gt_comparison(
                            entry, sample.get("ground_truth", [])
                        )
                        if gt_comparison:
                            entry["gt_comparison"] = gt_comparison
                        outputs_by_sample.append(entry)
                self._log_probe_outputs_to_wandb(
                    rank, outputs_by_sample, namespace=variant_name
                )
                probe_metrics, _summaries = self._build_probe_metrics(
                    outputs_by_sample, namespace=variant_name
                )
                if probe_metrics and hasattr(self, "tracking"):
                    self.tracking.log(probe_metrics, step=self.global_step)
                del inferencer
                del probe_model
                inferencer = None
                probe_model = None
                full_state_dict = None

                # Restore EMA before next variant iteration
                if ema_offloaded:
                    logger.info("Rank0 probe: reloading EMA states to GPU")
                    self._reload_ema_to_gpu(ema_offloaded)
                    ema_offloaded = []
        finally:
            del inferencer
            del probe_model
            del full_state_dict
            gc.collect()
            torch.cuda.empty_cache()
            if ema_offloaded:
                logger.info("Rank0 probe: reloading EMA states to GPU (cleanup)")
                self._reload_ema_to_gpu(ema_offloaded)
            if offloaded and hasattr(self, "optimizer") and self.optimizer is not None:
                logger.info("Rank0 probe: reloading optimizer states to GPU")
                self._reload_optimizer_to_gpu(self.optimizer, offloaded)
                logger.info("Rank0 probe: optimizer states restored")
            dist_barrier()  # sync all ranks after probe

    def _prepare_probe_inputs(self, sample: Dict[str, Any]):
        input_terms = []
        for term in sample["inputs"]:
            if term["type"] == "text":
                input_terms.append(term["text"])
                continue
            if term["type"] == "image":
                with PILImage.open(term["path"]) as img:
                    input_terms.append(img.convert("RGB"))
                continue
            raise ValueError(
                f"Unsupported probe input type '{term['type']}' in sample {sample['id']}"
            )
        return input_terms

    def _to_wandb_image(self, wandb_client, image_obj):
        def _normalize_to_uint8_hwc(arr: np.ndarray) -> Optional[np.ndarray]:
            if arr.ndim == 2:
                arr = arr[..., None]
            if arr.ndim != 3:
                return None

            if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] not in (1, 3, 4):
                return None

            if np.issubdtype(arr.dtype, np.floating):
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
                # Typical model output is [0, 1]. If values look like that, scale.
                if arr.max(initial=0.0) <= 1.5:
                    arr = np.clip(arr, 0.0, 1.0) * 255.0
                else:
                    arr = np.clip(arr, 0.0, 255.0)
                arr = arr.astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            if arr.shape[-1] == 1:
                arr = arr[..., 0]
            return arr

        try:
            if isinstance(image_obj, PILImage.Image):
                image = image_obj.convert("RGB")
                return wandb_client.Image(image)

            if torch.is_tensor(image_obj):
                arr = image_obj.detach().cpu().numpy()
                arr = _normalize_to_uint8_hwc(arr)
                if arr is None:
                    return None
                image = PILImage.fromarray(arr)
                return wandb_client.Image(image)

            if isinstance(image_obj, np.ndarray):
                arr = _normalize_to_uint8_hwc(image_obj)
                if arr is None:
                    return None
                image = PILImage.fromarray(arr)
                return wandb_client.Image(image)
        except Exception as e:
            logger.warning(
                f"Failed to convert probe image for W&B: {type(e).__name__}: {e}"
            )
            return None

        return None

    def _flatten_probe_outputs(self, output_item):
        flattened = []
        if isinstance(output_item, str):
            flattened.append(("text", output_item))
            return flattened

        if isinstance(output_item, dict):
            text = output_item.get("text")
            if isinstance(text, str):
                flattened.append(("text", text))
            if "image" in output_item and output_item["image"] is not None:
                flattened.append(("image", output_item["image"]))
            return flattened

        if isinstance(output_item, PILImage.Image) or torch.is_tensor(output_item):
            flattened.append(("image", output_item))
            return flattened

        flattened.append(("text", str(output_item)))
        return flattened

    @staticmethod
    def _compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute PSNR between two (1, C, H, W) float tensors in [0, 1]."""
        mse = ((img1 - img2) ** 2).mean()
        if mse <= 0:
            return 100.0
        return (10.0 * torch.log10(1.0 / mse)).item()

    @staticmethod
    def _compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute SSIM between two (1, C, H, W) float tensors in [0, 1]."""
        C1 = 0.01**2
        C2 = 0.03**2
        # Use avg_pool2d with kernel_size=11 as a simple window
        mu1 = F.avg_pool2d(img1, 11, 1, 5)
        mu2 = F.avg_pool2d(img2, 11, 1, 5)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.avg_pool2d(img1 * img1, 11, 1, 5) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, 11, 1, 5) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 11, 1, 5) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean().item()

    @staticmethod
    def _get_image_size(image_obj) -> Optional[tuple]:
        """Return (H, W) of a probe image, or None if unknown."""
        if isinstance(image_obj, PILImage.Image):
            return (image_obj.height, image_obj.width)
        if torch.is_tensor(image_obj) and image_obj.dim() == 3:
            if image_obj.shape[0] in (1, 3):
                return (image_obj.shape[1], image_obj.shape[2])
            if image_obj.shape[-1] in (1, 3):
                return (image_obj.shape[0], image_obj.shape[1])
        if isinstance(image_obj, np.ndarray) and image_obj.ndim in (2, 3):
            if image_obj.ndim == 2:
                return (image_obj.shape[0], image_obj.shape[1])
            if image_obj.shape[0] in (1, 3) and image_obj.shape[-1] not in (
                1,
                3,
            ):
                return (image_obj.shape[1], image_obj.shape[2])
            return (image_obj.shape[0], image_obj.shape[1])
        return None

    def _image_to_tensor(self, image_obj, target_size=None) -> Optional[torch.Tensor]:
        """Convert a probe image (PIL, tensor, ndarray) to (1, C, H, W) float [0, 1]."""
        pil_img = None
        if isinstance(image_obj, PILImage.Image):
            pil_img = image_obj.convert("RGB")
        elif torch.is_tensor(image_obj):
            arr = image_obj.detach().cpu().float()
            if arr.dim() == 3:
                if arr.shape[0] in (1, 3):
                    # (C, H, W) -> normalize
                    if arr.max() > 1.5:
                        arr = arr / 255.0
                    arr = arr.clamp(0, 1).unsqueeze(0)
                    if target_size is not None:
                        arr = F.interpolate(
                            arr,
                            size=target_size,
                            mode="bilinear",
                            align_corners=False,
                        )
                    return arr
                elif arr.shape[-1] in (1, 3):
                    arr = arr.permute(2, 0, 1)
                    if arr.max() > 1.5:
                        arr = arr / 255.0
                    arr = arr.clamp(0, 1).unsqueeze(0)
                    if target_size is not None:
                        arr = F.interpolate(
                            arr,
                            size=target_size,
                            mode="bilinear",
                            align_corners=False,
                        )
                    return arr
            return None
        elif isinstance(image_obj, np.ndarray):
            arr = image_obj
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.ndim == 3:
                if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                    arr = np.transpose(arr, (1, 2, 0))
                pil_img = PILImage.fromarray(
                    arr.astype(np.uint8)
                    if arr.max() > 1.5
                    else (arr * 255).astype(np.uint8)
                )
                pil_img = pil_img.convert("RGB")
            else:
                return None

        if pil_img is not None:
            if target_size is not None:
                pil_img = pil_img.resize(
                    (target_size[1], target_size[0]), PILImage.BILINEAR
                )
            arr_np = np.array(pil_img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr_np).permute(2, 0, 1).unsqueeze(0)
            return tensor

        return None

    def _align_gt_image_for_probe_metrics(
        self, gt_img: PILImage.Image
    ) -> PILImage.Image:
        """Resize GT to the same VAE-space size used during training, when available."""
        aligned = gt_img.convert("RGB")
        processing_class = getattr(self, "processing_class", None)
        vae_transform = getattr(processing_class, "vae_image_transform", None)
        resize_transform = getattr(vae_transform, "resize_transform", None)
        if not callable(resize_transform):
            return aligned
        try:
            resized = resize_transform(aligned)
        except Exception as e:
            logger.warning(f"Failed to align GT image for probe metrics: {e}")
            return aligned
        if isinstance(resized, PILImage.Image):
            return resized.convert("RGB")
        return aligned

    def _compute_gt_comparison(
        self, sample_result: Dict[str, Any], ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare generated images against ground truth images."""
        if not ground_truth:
            return {}

        # Extract generated images from outputs
        gen_images = []
        for raw_output in sample_result.get("outputs", []):
            for output_type, payload in self._flatten_probe_outputs(raw_output):
                if output_type == "image":
                    gen_images.append(payload)

        gt_count = len(ground_truth)
        gen_count = len(gen_images)
        if gen_count == 0:
            return {
                "gt_count": gt_count,
                "gen_count": 0,
                "count_match_ratio": 0.0,
                "lpips_avg": None,
                "ssim_avg": None,
                "psnr_avg": None,
                "per_pair": [],
            }

        count_match_ratio = gen_count / gt_count if gt_count > 0 else 0.0
        num_pairs = min(gen_count, gt_count)

        # Load GT images
        gt_images = []
        for gt_entry in ground_truth[:num_pairs]:
            try:
                with PILImage.open(gt_entry["path"]) as img:
                    raw_gt = img.convert("RGB")
                    gt_images.append(
                        {
                            "raw": raw_gt,
                            "aligned": self._align_gt_image_for_probe_metrics(raw_gt),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to load GT image {gt_entry['path']}: {e}")
                gt_images.append(None)

        # Try to load LPIPS
        lpips_fn = None
        try:
            import warnings

            import lpips as lpips_lib

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module="torchvision"
                )
                lpips_fn = lpips_lib.LPIPS(net="alex", verbose=False).eval()
        except ImportError:
            logger.warning(
                "lpips package not available; skipping LPIPS metric. "
                "Install with: pip install lpips"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize LPIPS: {e}")

        per_pair = []
        for i in range(num_pairs):
            gt_entry = gt_images[i]
            if gt_entry is None:
                per_pair.append({"lpips": None, "ssim": None, "psnr": None})
                continue
            raw_gt_img = gt_entry["raw"]
            gt_img = gt_entry["aligned"]

            gen_size = self._get_image_size(gen_images[i])  # (H, W) or None
            gt_size = (gt_img.height, gt_img.width)
            raw_gt_size = (raw_gt_img.height, raw_gt_img.width)
            if raw_gt_size != gt_size:
                logger.info(
                    f"GT pair {i}: aligning raw GT {raw_gt_size} to training size {gt_size}"
                )
            if gen_size is not None and gen_size != gt_size:
                compare_size = (
                    min(gen_size[0], gt_size[0]),
                    min(gen_size[1], gt_size[1]),
                )
                logger.info(
                    f"GT pair {i}: resolution mismatch gen={gen_size} vs aligned_gt={gt_size}, "
                    f"comparing at {compare_size}"
                )
            else:
                compare_size = None
            gt_tensor = self._image_to_tensor(gt_img, target_size=compare_size)
            gen_tensor = self._image_to_tensor(gen_images[i], target_size=compare_size)

            if gt_tensor is None or gen_tensor is None:
                per_pair.append({"lpips": None, "ssim": None, "psnr": None})
                continue

            pair_metrics: Dict[str, Optional[float]] = {}

            # PSNR
            try:
                pair_metrics["psnr"] = self._compute_psnr(gen_tensor, gt_tensor)
            except Exception:
                pair_metrics["psnr"] = None

            # SSIM
            try:
                pair_metrics["ssim"] = self._compute_ssim(gen_tensor, gt_tensor)
            except Exception:
                pair_metrics["ssim"] = None

            # LPIPS (expects tensors in [-1, 1])
            if lpips_fn is not None:
                try:
                    gen_lpips = gen_tensor * 2.0 - 1.0
                    gt_lpips = gt_tensor * 2.0 - 1.0
                    with torch.no_grad():
                        score = lpips_fn(gen_lpips, gt_lpips)
                    pair_metrics["lpips"] = score.item()
                except Exception:
                    pair_metrics["lpips"] = None
            else:
                pair_metrics["lpips"] = None

            per_pair.append(pair_metrics)

        # Compute averages
        def _avg(key):
            vals = [p[key] for p in per_pair if p.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        return {
            "gt_count": gt_count,
            "gen_count": gen_count,
            "count_match_ratio": count_match_ratio,
            "lpips_avg": _avg("lpips"),
            "ssim_avg": _avg("ssim"),
            "psnr_avg": _avg("psnr"),
            "per_pair": per_pair,
        }

    def _summarize_probe_sample_result(
        self, sample_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        image_outputs = 0
        for raw_output in sample_result.get("outputs", []):
            for output_type, _payload in self._flatten_probe_outputs(raw_output):
                if output_type == "image":
                    image_outputs += 1

        summary = {
            "sample_id": sample_result.get("sample_id", "unknown"),
            "image_outputs": int(image_outputs),
        }

        # Add GT comparison fields if available
        gt_comp = sample_result.get("gt_comparison", {})
        if gt_comp:
            summary["lpips_avg"] = gt_comp.get("lpips_avg")
            summary["ssim_avg"] = gt_comp.get("ssim_avg")
            summary["psnr_avg"] = gt_comp.get("psnr_avg")

        return summary

    def _build_probe_metrics(
        self,
        outputs_by_sample: List[Dict[str, Any]],
        namespace: Optional[str] = None,
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        summaries = [
            self._summarize_probe_sample_result(sample_result)
            for sample_result in outputs_by_sample
        ]
        if not summaries:
            return {}, []

        num_samples = len(summaries)
        total_image_outputs = sum(item["image_outputs"] for item in summaries)

        prefix = "infer_probe" if not namespace else f"infer_probe/{namespace}"
        metrics: Dict[str, float] = {
            f"{prefix}/avg_image_outputs": float(total_image_outputs)
            / float(num_samples),
        }

        # GT quality metrics
        gt_samples = [
            s
            for s in summaries
            if s.get("lpips_avg") is not None or s.get("ssim_avg") is not None
        ]
        if gt_samples:
            gt_lpips_vals = [
                s["lpips_avg"] for s in gt_samples if s.get("lpips_avg") is not None
            ]
            gt_ssim_vals = [
                s["ssim_avg"] for s in gt_samples if s.get("ssim_avg") is not None
            ]
            gt_psnr_vals = [
                s["psnr_avg"] for s in gt_samples if s.get("psnr_avg") is not None
            ]

            if gt_lpips_vals:
                metrics[f"{prefix}/gt_lpips_avg"] = sum(gt_lpips_vals) / len(
                    gt_lpips_vals
                )
            if gt_ssim_vals:
                metrics[f"{prefix}/gt_ssim_avg"] = sum(gt_ssim_vals) / len(gt_ssim_vals)
            if gt_psnr_vals:
                metrics[f"{prefix}/gt_psnr_avg"] = sum(gt_psnr_vals) / len(gt_psnr_vals)

        return metrics, summaries

    def _group_probe_outputs_into_turns(
        self, raw_outputs: list
    ) -> List[Dict[str, Any]]:
        """Group a flat list of outputs into logical turns.

        Each turn pairs the accumulated text preceding an image with that
        image.  Trailing text with no subsequent image becomes a final
        text-only turn.
        """
        turns: List[Dict[str, Any]] = []
        pending_text: List[str] = []

        for raw_output in raw_outputs:
            flattened = self._flatten_probe_outputs(raw_output)
            for output_type, payload in flattened:
                if output_type == "text":
                    pending_text.append(payload)
                elif output_type == "image":
                    turns.append(
                        {
                            "text": "\n".join(pending_text),
                            "image": payload,
                        }
                    )
                    pending_text = []

        if pending_text:
            turns.append({"text": "\n".join(pending_text), "image": None})

        return turns

    def _log_probe_outputs_to_wandb(
        self,
        rank: int,
        outputs_by_sample: List[Dict[str, Any]],
        namespace: Optional[str] = None,
    ):
        if rank != 0:
            return
        if not hasattr(self, "tracking"):
            return
        if "wandb" not in self.tracking.logger:
            return

        wandb_client = self.tracking.logger["wandb"]

        table = wandb_client.Table(
            columns=[
                "sample_id",
                "turn",
                "text",
                "generated_image",
                "ground_truth_image",
                "gt_metrics",
            ]
        )
        generated_gallery = []
        gt_gallery = []

        for sample_result in outputs_by_sample:
            sample_id = sample_result["sample_id"]
            gt_list = sample_result.get("ground_truth", [])
            gt_comp = sample_result.get("gt_comparison", {})
            per_pair = gt_comp.get("per_pair", [])

            turns = self._group_probe_outputs_into_turns(sample_result["outputs"])
            gen_image_idx = 0
            for turn_idx, turn in enumerate(turns):
                turn_text = turn["text"]
                turn_gen_image = None
                turn_gt_image = None
                turn_gt_metrics = ""

                if turn["image"] is not None:
                    wandb_image = self._to_wandb_image(wandb_client, turn["image"])
                    if wandb_image is not None:
                        turn_gen_image = wandb_image
                        generated_gallery.append(wandb_image)

                    if gen_image_idx < len(gt_list):
                        try:
                            with PILImage.open(
                                gt_list[gen_image_idx]["path"]
                            ) as gt_img:
                                gt_wandb = self._to_wandb_image(
                                    wandb_client, gt_img.convert("RGB")
                                )
                                if gt_wandb is not None:
                                    turn_gt_image = gt_wandb
                                    gt_gallery.append(gt_wandb)
                        except Exception:
                            pass

                        if gen_image_idx < len(per_pair):
                            pair = per_pair[gen_image_idx]
                            parts = []
                            if pair.get("lpips") is not None:
                                parts.append(f"LPIPS={pair['lpips']:.4f}")
                            if pair.get("ssim") is not None:
                                parts.append(f"SSIM={pair['ssim']:.4f}")
                            if pair.get("psnr") is not None:
                                parts.append(f"PSNR={pair['psnr']:.2f}")
                            turn_gt_metrics = ", ".join(parts)

                    gen_image_idx += 1

                if turn_text or turn_gen_image is not None:
                    table.add_data(
                        sample_id,
                        turn_idx,
                        turn_text or "",
                        turn_gen_image,
                        turn_gt_image,
                        turn_gt_metrics,
                    )

        prefix = "infer_probe" if not namespace else f"infer_probe/{namespace}"
        log_payload = {f"{prefix}/results": table}
        if generated_gallery:
            log_payload[f"{prefix}/generated_images"] = generated_gallery
        if gt_gallery:
            log_payload[f"{prefix}/gt_images"] = gt_gallery
        wandb_client.log(log_payload, step=self.global_step)

    def run_inference_probes(self, rank: int):
        if not self.enable_inference_probes:
            return
        if not self._should_run_inference_probe():
            return

        if dist.get_world_size() > 1 and self.inference_probe_use_rank0_unsharded:
            try:
                self._run_rank0_unsharded_probes(rank)
            except Exception as e:
                logger.exception(
                    f"[rank {rank}] Failed to run inference probes at step {self.global_step}: "
                    f"{type(e).__name__}: {e!r}"
                )
            return

        inferencer = self._get_probe_inferencer()
        use_ema = self.ema.is_enabled() and self.ema.initialized

        was_training = self.fsdp2_model.training
        self.fsdp2_model.eval()

        try:
            variants = self._get_probe_variants(
                use_ema=use_ema, compare_ema=self.inference_probe_compare_ema
            )
            for variant_name, variant_use_ema in variants:
                variant_label = variant_name or "ema"
                outputs_by_sample = []
                if variant_use_ema:
                    self.ema.swap_with_model()
                logger.info(
                    f"Running inference probes at step {self.global_step} ({variant_label} weights)"
                )
                try:
                    with torch.no_grad():
                        for sample in self.inference_probe_samples:
                            probe_kwargs = dict(self.inference_probe_default_kwargs)
                            probe_kwargs.update(sample.get("kwargs", {}))
                            input_terms = self._prepare_probe_inputs(sample)
                            outputs = inferencer.interleave_inference(
                                input_lists=input_terms, **probe_kwargs
                            )
                            diagnostics = dict(
                                getattr(inferencer, "last_interleave_diagnostics", {})
                                or {}
                            )
                            entry = {
                                "sample_id": sample["id"],
                                "outputs": outputs,
                                "diagnostics": diagnostics,
                                "ground_truth": sample.get("ground_truth", []),
                            }
                            gt_comparison = self._compute_gt_comparison(
                                entry, sample.get("ground_truth", [])
                            )
                            if gt_comparison:
                                entry["gt_comparison"] = gt_comparison
                            outputs_by_sample.append(entry)
                    self._log_probe_outputs_to_wandb(
                        rank, outputs_by_sample, namespace=variant_name
                    )
                    probe_metrics, _summaries = self._build_probe_metrics(
                        outputs_by_sample, namespace=variant_name
                    )
                    if rank == 0 and probe_metrics and hasattr(self, "tracking"):
                        self.tracking.log(probe_metrics, step=self.global_step)
                finally:
                    if variant_use_ema:
                        self.ema.swap_with_model()
        except Exception as e:
            logger.exception(
                f"[rank {rank}] Failed to run inference probes at step {self.global_step}: "
                f"{type(e).__name__}: {e!r}"
            )
        finally:
            if was_training:
                self.fsdp2_model.train()
