from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, field_validator

from .processor import ProcessorConfig


class StageConfig(BaseModel):
    """Configuration for a single training stage in multi-stage training."""

    dataset_folders: list[str]
    max_steps: int


class DatasetConfig(BaseModel):
    extra_kwargs: Optional[Dict[str, Any]] = None
    dataset_type: str
    dataset_format: Literal["jsonl", "parquet"]
    processor_config: Union[dict[str, Any], ProcessorConfig]

    # Dataset configuration
    dataset_path: Optional[str] = None  # Optional - used for external files
    dataset_folders: Optional[list[str]] = (
        None  # Task folders (auto-discovers training/training.jsonl)
    )
    jsonl_mode: Optional[Literal["default", "reasoning_columns"]] = "default"
    jsonl_images_root: Optional[str] = None
    jsonl_reasoning_images_root: Optional[str] = None

    @field_validator("dataset_folders", mode="before")
    @classmethod
    def coerce_dataset_folders(cls, v):
        """Accept a single string as a one-element list."""
        if isinstance(v, str):
            return [v]
        return v

    use_action_control_system_prompt: Optional[bool] = False
    action_control_system_prompt: Optional[str] = None
    wrap_assistant_reasoning_with_think: Optional[bool] = False
    shuffle: bool = True
    data_seed: Optional[int] = 42
    eval_dataset_path: Optional[str] = None
    # Override the jsonl filename when loading from dataset_folders
    # (default: "train.jsonl"; set to "val.jsonl" for eval datasets)
    folder_jsonl_filename: Optional[str] = None
    # When auto-discovering jsonl files, also load val.jsonl alongside train and
    # merge it into the training stream (shuffle + packing handled normally).
    # No-op when folder_jsonl_filename is set explicitly.
    include_val_in_training: bool = False

    # Packing configuration
    packing_mode: Optional[Literal["off", "greedy", "dynamic"]] = "off"
    # Legacy compat: accept old packing/dynamic_packing bools
    packing: Optional[bool] = None
    dynamic_packing: Optional[bool] = None
    # Greedy packing
    packing_length: Optional[int] = 32000
    filter_overlong: Optional[bool] = True
    filter_missing_text: Optional[bool] = False
    # When true (reasoning_columns mode only), drop samples that don't have ALL
    # of: non-empty reasoning_thought_0, non-empty reasoning_image_0, and a
    # non-empty answer (gt_answer or answer). Use to restrict training to
    # samples with a complete image-generation reasoning trace + final answer.
    require_reasoning_image_0_and_answer: Optional[bool] = False
    # Dynamic packing (only max_num_tokens is required; others auto-derive)
    max_num_tokens: Optional[int] = 16384
    expected_num_tokens: Optional[int] = None  # defaults to max_num_tokens
    max_num_tokens_per_sample: Optional[int] = None  # defaults to max_num_tokens
    prefer_buffer_before: Optional[int] = None  # defaults to max_num_tokens // 2
    max_buffer_size: Optional[int] = 50

    # When true, samples whose estimated token count would exceed
    # max_num_tokens_per_sample have their reasoning (assistant) images
    # downscaled in-place so the sample fits the budget. User input images
    # are NEVER shrunk. Estimation is worst-case (no dropout) so the actual
    # processed length is always <= the estimate.
    adaptive_image_size_to_fit_budget: Optional[bool] = True
    # Multiplicative safety margin applied to the budget during selection
    # (estimated_tokens * margin <= budget). Slightly >1 protects against
    # tokenizer-vs-estimator mismatch.
    adaptive_image_size_safety_margin: Optional[float] = 1.05
    # Floor for adaptive shrinking of reasoning (assistant) images in pixels
    # (longest side). The adapter never shrinks below this; if at floor the
    # estimate STILL exceeds the budget, the sample is admitted anyway
    # (the iterable's drop check is bypassed for that sample).
    adaptive_image_size_floor: Optional[int] = 512

    # Multi-stage training configuration
    stages: Optional[list[StageConfig]] = None

    # Sharding configuration
    stride_sharding: Optional[bool] = False

    def resolve_packing_mode(self) -> str:
        """Return the effective packing mode, handling legacy bool fields."""
        if self.packing_mode != "off":
            return self.packing_mode
        # Legacy compat: derive from old bool fields
        if self.packing:
            if self.dynamic_packing:
                return "dynamic"
            return "greedy"
        return "off"

    @field_validator("stages")
    @classmethod
    def validate_stages(cls, v):
        """Validate that each stage has positive max_steps and non-empty dataset_folders."""
        if v is not None:
            for i, stage in enumerate(v):
                if stage.max_steps <= 0:
                    raise ValueError(
                        f"Stage {i} max_steps must be positive, got {stage.max_steps}"
                    )
                if not stage.dataset_folders:
                    raise ValueError(f"Stage {i} dataset_folders must not be empty")
        return v

    @field_validator("packing_length")
    @classmethod
    def validate_positive_values(cls, v, info):
        """Validate that numeric video and packing parameters are positive."""
        if v is not None and v <= 0:
            field_name = info.field_name
            raise ValueError(f"{field_name} must be positive, got {v}")
        return v
