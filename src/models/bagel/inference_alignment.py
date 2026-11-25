from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


def _get_dataset_prompt_settings(dataset_like: Any) -> tuple[Optional[bool], Optional[str]]:
    if dataset_like is None:
        return None, None

    if isinstance(dataset_like, Mapping):
        ds_cfg = dict(dataset_like)
        ds_kwargs = ds_cfg.get("dataset_kwargs", {}) or {}
        ds_use_prompt = ds_kwargs.get("use_action_control_system_prompt")
        if ds_use_prompt is None:
            ds_use_prompt = ds_cfg.get("use_action_control_system_prompt")
        ds_prompt = ds_kwargs.get("action_control_system_prompt")
        if ds_prompt is None:
            ds_prompt = ds_cfg.get("action_control_system_prompt")
        return ds_use_prompt, ds_prompt

    ds_use_prompt = getattr(dataset_like, "use_action_control_system_prompt", None)
    ds_prompt = getattr(dataset_like, "action_control_system_prompt", None)
    return ds_use_prompt, ds_prompt


def apply_training_alignment_defaults(
    infer_kwargs: Dict[str, Any],
    processor: Any,
    dataset_like: Any = None,
) -> None:
    """Fill missing inference kwargs from training-time processor/dataset settings."""
    image_action_token = getattr(processor, "image_action_token", None)
    supervise_action = getattr(processor, "supervise_image_action_token", None)
    if "use_action_token_control" not in infer_kwargs:
        if supervise_action is not None:
            infer_kwargs["use_action_token_control"] = bool(supervise_action)
        elif isinstance(image_action_token, str) and image_action_token.strip():
            infer_kwargs["use_action_token_control"] = True

    if (
        infer_kwargs.get("use_action_token_control", False)
        and "image_action_token" not in infer_kwargs
        and isinstance(image_action_token, str)
        and image_action_token.strip()
    ):
        infer_kwargs["image_action_token"] = image_action_token

    ds_use_prompt, ds_prompt = _get_dataset_prompt_settings(dataset_like)
    if (
        "use_action_control_system_prompt" not in infer_kwargs
        and ds_use_prompt is not None
    ):
        infer_kwargs["use_action_control_system_prompt"] = bool(ds_use_prompt)
    if (
        infer_kwargs.get("use_action_control_system_prompt", False)
        and "action_control_system_prompt" not in infer_kwargs
        and isinstance(ds_prompt, str)
        and ds_prompt.strip()
    ):
        infer_kwargs["action_control_system_prompt"] = ds_prompt

    feedback_vae = getattr(processor, "assistant_image_as_vae_condition", None)
    feedback_vit = getattr(processor, "assistant_image_as_vit_condition", None)
    if (
        "feedback_generated_image_as_vae_condition" not in infer_kwargs
        and feedback_vae is not None
    ):
        infer_kwargs["feedback_generated_image_as_vae_condition"] = bool(feedback_vae)
    if (
        "feedback_generated_image_as_vit_condition" not in infer_kwargs
        and feedback_vit is not None
    ):
        infer_kwargs["feedback_generated_image_as_vit_condition"] = bool(feedback_vit)
    input_image_vae = getattr(processor, "user_image_as_vae_condition", None)
    if (
        "input_image_as_vae_condition" not in infer_kwargs
        and input_image_vae is not None
    ):
        infer_kwargs["input_image_as_vae_condition"] = bool(input_image_vae)
    if "feedback_generated_image_to_context" not in infer_kwargs:
        if feedback_vae is not None or feedback_vit is not None:
            infer_kwargs["feedback_generated_image_to_context"] = bool(
                bool(feedback_vae) or bool(feedback_vit)
            )
    # register_action_token is carried via new_token_ids automatically,
    # but we propagate it here for logging/transparency in infer_kwargs.
    register_action_token = getattr(processor, "register_action_token", None)
    if register_action_token is not None and "register_action_token" not in infer_kwargs:
        # Not consumed directly by the inferencer (it reads from new_token_ids),
        # but useful for logging and config inspection.
        pass

    if "enable_ref_image" not in infer_kwargs:
        infer_enable_ref = getattr(processor, "enable_ref_image", None)
        if infer_enable_ref is not None:
            infer_kwargs["enable_ref_image"] = bool(infer_enable_ref)
    if "enable_size_tokens" not in infer_kwargs:
        infer_enable_size = getattr(processor, "enable_size_tokens", None)
        if infer_enable_size is not None:
            infer_kwargs["enable_size_tokens"] = bool(infer_enable_size)
    if "text_can_attend_vae_condition" not in infer_kwargs:
        text_can_attend_vae_condition = getattr(
            processor, "text_can_attend_vae_condition", None
        )
        if text_can_attend_vae_condition is not None:
            infer_kwargs["text_can_attend_vae_condition"] = bool(
                text_can_attend_vae_condition
            )
    if "inline_image_latents" not in infer_kwargs:
        inline_image_latents = getattr(processor, "inline_image_latents", None)
        if inline_image_latents is not None:
            infer_kwargs["inline_image_latents"] = bool(inline_image_latents)
    if "ref_target_can_attend_vae_condition" not in infer_kwargs:
        ref_target_can_attend_vae_condition = getattr(
            processor, "ref_target_can_attend_vae_condition", None
        )
        if ref_target_can_attend_vae_condition is not None:
            infer_kwargs["ref_target_can_attend_vae_condition"] = bool(
                ref_target_can_attend_vae_condition
            )
    if "no_ref_target_can_attend_vae_condition" not in infer_kwargs:
        no_ref_target_can_attend_vae_condition = getattr(
            processor, "no_ref_target_can_attend_vae_condition", None
        )
        if no_ref_target_can_attend_vae_condition is not None:
            infer_kwargs["no_ref_target_can_attend_vae_condition"] = bool(
                no_ref_target_can_attend_vae_condition
            )


def merge_missing_infer_defaults(
    infer_kwargs: Dict[str, Any], defaults: Mapping[str, Any]
) -> None:
    for key, value in defaults.items():
        if key not in infer_kwargs:
            infer_kwargs[key] = copy.deepcopy(value)


def extract_training_inference_context(
    config: Mapping[str, Any],
    default_processor_name: str,
) -> Dict[str, Any]:
    dataset_config = copy.deepcopy(dict(config.get("dataset_config") or {}))
    trainer_args = dict(config.get("trainer_args") or {})
    probe_config = dict(trainer_args.get("inference_probe_config") or {})

    processor_config = copy.deepcopy(dict(dataset_config.get("processor_config") or {}))
    if not processor_config.get("processor_name"):
        processor_config["processor_name"] = default_processor_name
    if not processor_config.get("processor_type"):
        processor_config["processor_type"] = "bagel"

    probe_infer_kwargs = copy.deepcopy(dict(probe_config.get("infer_kwargs") or {}))
    return {
        "dataset_config": dataset_config,
        "processor_config": processor_config,
        "probe_infer_kwargs": probe_infer_kwargs,
    }


def load_training_inference_context(
    config_path: str,
    default_processor_name: str,
) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        config = json.loads(text)
    else:
        config = yaml.safe_load(text)
    if not isinstance(config, Mapping):
        raise ValueError(f"Training config must decode to a mapping: {path}")
    return extract_training_inference_context(config, default_processor_name)
