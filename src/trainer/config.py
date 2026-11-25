from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional

import transformers

from data import DatasetConfig
from models import ModelConfig


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    log_level_replica: str = "info"
    log_on_each_node: bool = False
    use_muon: Optional[bool] = False
    freeze_modules: Optional[List[str]] = None
    use_rmpad: Optional[bool] = False
    fsdp2: Optional[bool] = False
    # FSDP2-only settings used by custom trainers; avoids HF FSDP warnings.
    fsdp2_config: Optional[Dict[str, Any]] = None
    reduce_dtype: Optional[str] = "bfloat16"
    output_dtype: Optional[str] = "bfloat16"
    use_tm_activation_checkpoint: Optional[bool] = False
    print_batch_input_steps: Optional[int] = -1
    debug_batch_input: Optional[bool] = (
        False  # Enable detailed debug info for batch input decoding
    )
    # Only print batch-debug information on this global rank.
    # Set to -1 to allow all ranks.
    debug_batch_input_rank: Optional[int] = 0
    enable_profiler: Optional[bool] = False
    profiler_config: Optional[Dict[str, Any]] = None

    # Parallelism
    ep_degree: Optional[int] = 1
    sp_ulysses_degree: Optional[int] = 1

    # --- EMA (Exponential Moving Average) ---
    ema_enabled: Optional[bool] = False
    ema_decay: Optional[float] = 0.9999
    ema_update_every: Optional[int] = 1
    ema_start_step: Optional[int] = 0
    ema_requires_grad_only: Optional[bool] = True
    # Optional name-based filtering for which parameters participate in EMA.
    # Example:
    #   ema_param_filter:
    #     mode: "substring"  # or "regex"
    #     include: ["language_model"]   # only include matching params
    #     exclude: ["lm_head"]          # exclude matching params
    ema_param_filter: Optional[Dict[str, Any]] = None
    ema_resume_from_ema: Optional[bool] = False
    # Master dtype for the EMA shadow buffer. Default "model" stores the shadow
    # in the same dtype as the live model parameter (preserves prior behavior).
    # Set to "fp32" to keep the shadow in float32 so that small (1-decay) updates
    # do not get rounded to zero under bf16 precision (recommended for decay
    # >= ~0.999 when training in bf16). Accepted: "model", "fp32"/"float32",
    # "bf16"/"bfloat16".
    ema_master_dtype: Optional[str] = "model"

    # --- Eval Server Configuration ---
    eval_config: Optional[Dict[str, Any]] = None
    # --- In-training fixed-sample inference probe configuration ---
    # Example:
    # inference_probe_config:
    #   enabled: true
    #   interval_steps: 500
    #   samples_path: path/to/probes.jsonl
    #   samples: [...]
    #   infer_kwargs: {...}
    inference_probe_config: Optional[Dict[str, Any]] = None

    # --- Validation loss evaluation on eval dataset ---
    # Example:
    # val_eval_config:
    #   enabled: true
    #   interval_steps: 100
    #   start_step: 100
    #   max_batches: 50
    val_eval_config: Optional[Dict[str, Any]] = None

    # --- VLMEvalKit benchmark evaluation during training ---
    # Example:
    # bench_eval_config:
    #   enabled: true
    #   interval_steps: 500
    #   start_step: 500
    #   datasets: ["MMBench_DEV_EN_V11"]
    #   use_all_ranks: true
    #   understanding_output: true
    #   work_dir: "./bench_eval_outputs"
    #   infer_kwargs:
    #     think: true
    #     do_sample: true
    #     text_temperature: 0.3
    #     max_think_token_n: 4096
    #   judge_kwargs: {}
    bench_eval_config: Optional[Dict[str, Any]] = None

    compile_model: Optional[bool] = False
    compile_loss: Optional[bool] = False
    activation_offloading: Optional[bool] = False

    checkpoint_async_mode: Optional[str] = "disabled"
    checkpoint_keep_latest_k: Optional[int] = 0

    gc_enabled: Optional[bool] = False
    gc_freq: Optional[int] = 200


@dataclass
class TrainerConfig:
    trainer_type: Literal["hf_trainer", "fsdp2_trainer"]
    dataset_config: DatasetConfig
    trainer_args: TrainingArguments
    model_config: ModelConfig
    extra_kwargs: Dict[str, Any] = None

    def to_dict(self):
        trainer_args_dict = self.trainer_args.to_dict()
        model_config_dict = self.model_config.model_dump()
        dataset_config_dict = self.dataset_config.model_dump()
        final_dict = asdict(self)
        final_dict["trainer_args"] = trainer_args_dict
        final_dict["model_config"] = model_config_dict
        final_dict["dataset_config"] = dataset_config_dict
        return final_dict
