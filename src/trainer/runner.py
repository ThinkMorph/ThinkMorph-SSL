import os
import pathlib
import random
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

import parallel.process_group_manager as pgm
from data.registry import DATASET_MAPPING
from models.registry import create_model_from_config, create_model_from_pretrained
from models import MONKEY_PATCHER
from models.utils import setup_flops_counter
from parallel.sequence_parallel.ulysses import (
    set_ulysses_sequence_parallel_group,
)

from utils.train_utils import TrainUtilities
from .config import TrainerConfig
from .registry import TRAINER_REGISTER


def configure_tf32_precision(tf32_enabled: bool | None) -> None:
    if tf32_enabled is None:
        return

    enabled = bool(tf32_enabled)
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    if enabled:
        torch.set_float32_matmul_precision("high")


class TrainRunner:
    """
    This is a base train runner to wrap all other trainer or your training logic
    """

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        configure_tf32_precision(getattr(self.config.trainer_args, "tf32", None))
        self.set_random_seed(self.config.trainer_args.seed)
        self.train_dataset_config = config.dataset_config
        self.eval_dataset_config = self._init_eval_dataset_config(config)
        self.model_config = config.model_config

        # Multi-stage: auto-compute total max_steps from stages
        if self.train_dataset_config.stages:
            total_steps = sum(s.max_steps for s in self.train_dataset_config.stages)
            self.config.trainer_args.max_steps = total_steps
            logger.info(
                f"Multi-stage training: {len(self.train_dataset_config.stages)} stages, "
                f"total max_steps={total_steps}"
            )

    def build(self):
        logger.info("TrainRunner.build(): start")
        if dist.is_initialized():
            self.create_sp_dis_group()
            logger.info("TrainRunner.build(): sequence/data process groups ready")
        logger.info("TrainRunner.build(): building model...")
        self.model = self._build_model()
        logger.info("TrainRunner.build(): model ready")
        logger.info("TrainRunner.build(): building eval dataset...")
        self.eval_dataset = self._maybe_build_eval_dataset()
        logger.info("TrainRunner.build(): building train dataset...")
        self.train_dataset = self._build_train_dataset()
        logger.info("TrainRunner.build(): datasets ready, checking tokenizer resize...")
        self._resize_embeddings_if_needed()
        logger.info("TrainRunner.build(): applying monkey patches...")
        self._apply_monkey_patch()
        logger.info("TrainRunner.build(): building trainer...")
        self.trainer = self._build_trainer()
        logger.info("TrainRunner.build(): done")

    def _init_eval_dataset_config(self, config: TrainerConfig):
        # Explicit eval_dataset_path takes priority
        if config.dataset_config.eval_dataset_path is not None:
            eval_dataset_config = deepcopy(config.dataset_config)
            eval_dataset_config.packing = False
            eval_dataset_config.packing_mode = "off"
            eval_dataset_config.dataset_path = config.dataset_config.eval_dataset_path
            eval_dataset_config.dataset_folders = None
            eval_dataset_config.stages = None
            return eval_dataset_config

        # Auto-discover val.jsonl for loss evaluation only when loss_enabled
        val_eval_cfg = getattr(config.trainer_args, "val_eval_config", None)
        if not val_eval_cfg or not bool(val_eval_cfg.get("enabled", False)):
            return None
        # Val loss eval requires explicit loss_enabled flag.
        # Val inference eval (accuracy) reads val.jsonl directly via ValInferMixin.
        if not bool(val_eval_cfg.get("loss_enabled", False)):
            return None

        # Priority: val_eval_config.val_folders > dataset_folders > stages
        val_folders = val_eval_cfg.get("val_folders")
        if not val_folders:
            folders = config.dataset_config.dataset_folders
            if not folders and config.dataset_config.stages:
                seen = set()
                folders = []
                for stage in config.dataset_config.stages:
                    for f in stage.dataset_folders:
                        if f not in seen:
                            seen.add(f)
                            folders.append(f)
            val_folders = folders

        if not val_folders:
            return None

        eval_dataset_config = deepcopy(config.dataset_config)
        eval_dataset_config.packing = False
        eval_dataset_config.packing_mode = "off"
        eval_dataset_config.dataset_path = None
        eval_dataset_config.dataset_folders = val_folders
        eval_dataset_config.stages = None
        eval_dataset_config.shuffle = False
        eval_dataset_config.folder_jsonl_filename = "val.jsonl"
        logger.info(
            f"Auto-discovered val dataset from {len(val_folders)} folder(s): "
            f"{val_folders}"
        )
        return eval_dataset_config

    def _maybe_build_eval_dataset(self):
        if self.eval_dataset_config is None:
            logger.info("No eval dataset configured, skipping eval dataset build.")
            return None
        return self._build_eval_dataset()

    def _build_model(self):
        model_kwargs = self.model_config.extra_kwargs or {}
        load_from_pretrained_path = self.model_config.load_from_pretrained_path
        load_from_config = self.model_config.load_from_config
        if load_from_pretrained_path is not None:
            model = self._load_model_from_pretrained(
                load_from_pretrained_path, model_kwargs
            )
        elif load_from_config is not None:
            model = self._load_model_from_config(load_from_config, model_kwargs)
        else:
            raise ValueError(
                "No model name or pretrained path provided. Please provide one of them."
            )

        if self.model_config.overwrite_config:
            for key, value in self.model_config.overwrite_config.items():
                setattr(model.config, key, value)
                if getattr(model, key, None) is not None:
                    setattr(model, key, value)
                logger.info(f"Overwrite {key} to {value}")

        setup_flops_counter(model.config)
        logger.info(f"Model Structure: {model}")
        logger.info(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9} B")
        return model

    @staticmethod
    def _resolve_dcp_checkpoint(path: str) -> pathlib.Path | None:
        """Resolve a DCP checkpoint path, returning None if not a DCP checkpoint.

        Handles both direct checkpoint dirs (containing .metadata) and parent
        dirs containing checkpoint-* subdirectories (picks the latest).
        """
        p = pathlib.Path(path)
        if not p.is_dir():
            return None
        # Direct DCP checkpoint directory
        if (p / ".metadata").exists():
            return p
        # Parent directory containing checkpoint-* subdirs
        checkpoint_folders = sorted(
            p.glob("checkpoint-*"),
            key=lambda x: int(x.name.split("-")[-1]),
        )
        if checkpoint_folders and (checkpoint_folders[-1] / ".metadata").exists():
            return checkpoint_folders[-1]
        return None

    def _load_model_from_pretrained(self, load_from_pretrained_path, model_kwargs):
        model_class = create_model_from_pretrained(load_from_pretrained_path)
        # When using FSDP2 mixed precision, load in float32 to keep master weights in
        # full precision (FSDP2 handles bfloat16 casting for forward/backward).
        if getattr(self.config.trainer_args, "fsdp2", False):
            load_dtype = torch.float32
        else:
            load_dtype = torch.bfloat16 if self.config.trainer_args.bf16 else None

        dcp_path = self._resolve_dcp_checkpoint(load_from_pretrained_path)
        if dcp_path is not None:
            return self._load_model_from_dcp(
                str(dcp_path), model_class, load_dtype, model_kwargs
            )

        return model_class.from_pretrained(
            load_from_pretrained_path,
            attn_implementation=self.model_config.attn_implementation,
            torch_dtype=load_dtype,
            **model_kwargs,
        )

    def _load_model_from_dcp(
        self, checkpoint_path, model_class, load_dtype, model_kwargs
    ):
        """Load model weights directly from a DCP checkpoint directory."""
        from merger.fsdp2 import FSDP2Merger
        from transformers import AutoConfig
        from accelerate import init_empty_weights

        logger.info(
            f"Detected DCP checkpoint at {checkpoint_path}, loading directly..."
        )
        config = AutoConfig.from_pretrained(checkpoint_path)

        # Resolve the concrete model class from the auto class
        concrete_class = model_class._model_mapping[type(config)]

        # Create model on meta device to get state dict template
        with init_empty_weights():
            template = concrete_class(config)
        template_sd = template.state_dict()

        # Load weights from DCP
        merger = FSDP2Merger(checkpoint_type="regular")
        full_sd = merger._load_full_state_dict(
            pathlib.Path(checkpoint_path), template_sd
        )
        logger.info(f"Loaded {len(full_sd)} keys from DCP checkpoint")

        # Instantiate real model and load weights
        model = concrete_class(config)
        if load_dtype is not None:
            model = model.to(dtype=load_dtype)
        model.load_state_dict(full_sd, assign=True)
        if load_dtype is not None:
            model = model.to(dtype=load_dtype)

        logger.info(f"Successfully loaded model from DCP checkpoint: {checkpoint_path}")
        return model

    def _load_model_from_config(self, load_from_config, model_kwargs):
        model_type, init_config = self._resolve_model_init_config(load_from_config)
        model_class, m_config = create_model_from_config(model_type, init_config)
        return model_class.from_config(m_config, **model_kwargs)

    def _resolve_model_init_config(self, load_from_config):
        model_type = load_from_config.get("model_type", None)
        init_config = load_from_config.get("config", None)
        if init_config is None:
            # If no nested config, use the load_from_config dict directly (excluding model_type).
            init_config = {
                k: v for k, v in load_from_config.items() if k != "model_type"
            }
        return model_type, init_config

    def _resize_embeddings_if_needed(self):
        """Resize model embeddings if new tokens were added to the tokenizer.

        This must run after dataset building (which adds special tokens like
        action tokens to the tokenizer) but before FSDP wrapping so that
        embed_tokens and lm_head dimensions match the tokenizer vocab size.
        """
        processor = getattr(self.train_dataset, "processor", None)
        if processor is None:
            return
        self._align_latent_position_range_if_needed(processor)
        num_new_tokens = getattr(processor, "num_new_tokens", 0)
        if num_new_tokens == 0:
            return
        tokenizer = getattr(processor, "processor", None)
        if tokenizer is None:
            return
        new_vocab_size = len(tokenizer)
        language_model = getattr(self.model, "language_model", None)
        if language_model is None:
            return
        old_vocab_size = language_model.config.vocab_size
        if new_vocab_size <= old_vocab_size:
            return
        logger.info(
            f"Resizing token embeddings: {num_new_tokens} new token(s), "
            f"vocab {old_vocab_size} -> {new_vocab_size}"
        )
        language_model.resize_token_embeddings(new_vocab_size)
        # Sync vocab_size in configs so checkpoint save/load and inference
        # probe model reconstruction use the correct size (matches ThinkMorph).
        language_model.config.vocab_size = new_vocab_size
        if hasattr(self.model, "config") and hasattr(self.model.config, "llm_config"):
            self.model.config.llm_config.vocab_size = new_vocab_size

    def _align_latent_position_range_if_needed(self, processor) -> None:
        """Keep model/data latent position-id range aligned.

        For Bagel visual generation, latent position ids are created with
        `max_latent_size` on both data and model paths. If they diverge,
        training and inference quality can collapse despite seemingly normal loss.

        When the values differ we must also rebuild ``latent_pos_embed`` so that
        its internal positional table matches the new grid size.  Simply updating
        the scalar ``max_latent_size`` without rebuilding the embedding causes
        position-ID indexing errors (e.g. stereo/repeated-crop artifacts).
        """
        model_max = getattr(self.model, "max_latent_size", None)
        proc_max = getattr(processor, "max_latent_size", None)
        if model_max is None or proc_max is None:
            return
        if int(model_max) == int(proc_max):
            return

        new_max = int(proc_max)
        logger.warning(
            "Aligning model.max_latent_size with processor.max_latent_size: "
            f"{int(model_max)} -> {new_max}"
        )
        self.model.max_latent_size = new_max
        if hasattr(self.model, "config"):
            setattr(self.model.config, "max_latent_size", new_max)

        # Rebuild latent_pos_embed so the positional table matches the new size.
        from models.bagel.bagel import PositionEmbedding

        if hasattr(self.model, "latent_pos_embed"):
            hidden_size = self.model.latent_pos_embed.pos_embed.shape[-1]
            device = self.model.latent_pos_embed.pos_embed.device
            dtype = self.model.latent_pos_embed.pos_embed.dtype
            self.model.latent_pos_embed = PositionEmbedding(new_max, hidden_size).to(
                device=device, dtype=dtype
            )
            logger.warning(f"Rebuilt latent_pos_embed with max_latent_size={new_max}")

    def _apply_monkey_patch(self):
        kwargs = {"use_rmpad": self.config.trainer_args.use_rmpad, "patch_type": []}
        if self.config.trainer_args.use_liger_kernel:
            kwargs["patch_type"].append("liger")
            # Overwrite the use_liger_kernel to False as we already apply the liger kernel by ourselves
            self.config.trainer_args.use_liger_kernel = False

        if self.model_config.monkey_patch_kwargs:
            mp_kwargs = dict(self.model_config.monkey_patch_kwargs)
            patch_type = mp_kwargs.pop("patch_type", [])
            if isinstance(patch_type, str):
                patch_type = [patch_type]
            kwargs["patch_type"].extend(patch_type)
            kwargs.update(mp_kwargs)
            # Keep ordering stable while avoiding duplicate patch application.
            kwargs["patch_type"] = list(dict.fromkeys(kwargs["patch_type"]))
        try:
            MONKEY_PATCHER.apply_monkey_patch_to_instance(self.model, **kwargs)
        except Exception as e:
            logger.error(f"Error applying monkey patch: {e}. Skip monkey patch.")

    def _build_train_dataset(self):
        config = self.train_dataset_config
        if config.stages:
            config = config.model_copy(deep=True)
            config.dataset_folders = config.stages[0].dataset_folders
            logger.info(
                f"Multi-stage: building initial dataset from stage 0 folders: "
                f"{config.dataset_folders}"
            )
        return self._build_dataset(config)

    def _build_eval_dataset(self):
        return self._build_dataset(self.eval_dataset_config)

    def _build_dataset(self, dataset_config):
        dataset_cls = DATASET_MAPPING[dataset_config.dataset_type]
        dataset = dataset_cls(dataset_config)
        dataset.build()
        return dataset

    def _build_stage_dataset(self, stage_config):
        """Build a dataset for a specific training stage."""
        config = self.train_dataset_config.model_copy(deep=True)
        config.dataset_folders = stage_config.dataset_folders
        return self._build_dataset(config)

    def set_random_seed(self, random_seed: int = 42):
        # Setting random seed for all
        if random_seed is None:
            random_seed = 42
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        deterministic = bool(
            getattr(self.config.trainer_args, "full_determinism", False)
        )
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
        np.random.seed(random_seed)
        logger.info(
            "Set random seed to {} (full_determinism={})",
            random_seed,
            deterministic,
        )
        return random_seed

    def create_sp_dis_group(self):
        world_size = dist.get_world_size()
        # rank = dist.get_rank()
        sp_ulysses_degree = self.config.trainer_args.sp_ulysses_degree
        sp_degree = sp_ulysses_degree * 1  # ring attn always 1, kept for clarity

        total_group_size = sp_degree
        assert world_size % total_group_size == 0, (
            f"world_size={world_size} must be divisible by total_group_size={total_group_size}"
        )

        set_ulysses_sequence_parallel_group(pgm.process_group_manager.cp_group)

    def _build_trainer(self):
        trainer_cls = TRAINER_REGISTER[self.config.trainer_type]
        trainer = trainer_cls(**self._build_trainer_kwargs())
        return trainer

    def _build_trainer_kwargs(self):
        kwargs = {
            "model": self.model,
            "args": self.config.trainer_args,
            "data_collator": self.train_dataset.get_collator(),
            "train_dataset": self.train_dataset,
            "eval_dataset": self.eval_dataset,
            "processing_class": self.train_dataset.processor,
        }
        if self.train_dataset_config.stages:
            kwargs["dataset_stages"] = self.train_dataset_config.stages
            kwargs["build_stage_dataset"] = self._build_stage_dataset
        return kwargs

    def run(self, **kwargs):
        # Optional EMA (currently implemented in fsdp2_trainer). Keep default behavior unchanged.
        ema_trainer_type = ["fsdp2_trainer", "bagel_fsdp2_trainer"]
        if getattr(self.config.trainer_args, "ema_enabled", False):
            if self.config.trainer_type not in ema_trainer_type:
                logger.warning(
                    f"EMA is enabled (ema_enabled=True) but trainer_type={self.config.trainer_type!r} "
                    "does not implement EMA yet. EMA will be ignored."
                )
            else:
                logger.info(
                    "EMA enabled: "
                    f"decay={getattr(self.config.trainer_args, 'ema_decay', None)}, "
                    f"update_every={getattr(self.config.trainer_args, 'ema_update_every', None)}, "
                    f"start_step={getattr(self.config.trainer_args, 'ema_start_step', None)}, "
                    f"resume_from_ema={getattr(self.config.trainer_args, 'ema_resume_from_ema', None)}"
                )

        logger.info(
            "Runtime toggles: compile_model={}, compile_loss={}, activation_offloading={}, checkpoint_async_mode={}, checkpoint_keep_latest_k={}, gc_enabled={}, gc_freq={}",
            getattr(self.config.trainer_args, "compile_model", False),
            getattr(self.config.trainer_args, "compile_loss", False),
            getattr(self.config.trainer_args, "activation_offloading", False),
            getattr(self.config.trainer_args, "checkpoint_async_mode", "disabled"),
            getattr(self.config.trainer_args, "checkpoint_keep_latest_k", 0),
            getattr(self.config.trainer_args, "gc_enabled", False),
            getattr(self.config.trainer_args, "gc_freq", 0),
        )

        # Note: freeze_modules handling is now done inside the trainer's prepare_model method
        # before FSDP wrapping, which allows us to also pass frozen modules as ignored_states
        # to avoid FSDP collective mismatch issues with variable-size inputs (e.g., VAE).

        if self._should_resume_from_checkpoint():
            self.trainer.train(resume_from_checkpoint=True)
        else:
            self.trainer.train()
        # Save the state for hf_trainer
        if hasattr(self.trainer, "save_state"):
            self.trainer.save_state()
            self.safe_save_model_for_hf_trainer(
                self.trainer, self.config.trainer_args.output_dir
            )

    def _should_resume_from_checkpoint(self):
        checkpoint_glob = pathlib.Path(self.config.trainer_args.output_dir).glob(
            "checkpoint-*"
        )
        return any(checkpoint_glob)

    def safe_save_model_for_hf_trainer(self, trainer, output_dir: str):
        """Collects the state dict and dump to disk."""
        trainer.accelerator.wait_for_everyone()
        torch.cuda.synchronize()
        check_only_save_mm_adapter = self.config.trainer_args.only_save_mm_adapter
        logger.info(f"Only save projectors: {check_only_save_mm_adapter}")

        if check_only_save_mm_adapter:
            # Only save Adapter
            keys_to_match = ["multi_modal_projector", "audio_modal_projector"]

            weight_to_save = TrainUtilities.get_mm_adapter_state_maybe_zero_3(
                trainer.model.named_parameters(), keys_to_match
            )
            trainer.model.config.save_pretrained(output_dir)

            current_folder = output_dir.split("/")[-1]
            parent_folder = os.path.dirname(output_dir)
            if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
                if current_folder.startswith("checkpoint-"):
                    mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                    os.makedirs(mm_projector_folder, exist_ok=True)
                    torch.save(
                        weight_to_save,
                        os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                    )
                else:
                    torch.save(
                        weight_to_save, os.path.join(output_dir, "mm_projector.bin")
                    )
            return
        if trainer.deepspeed:
            trainer.save_model(output_dir)
            return
        if self.config.trainer_args.fsdp2:
            # For fsdp we merge the shards into a single checkpoint after the training is done
            if trainer.processing_class is not None:
                trainer.processing_class.save_pretrained(output_dir)
            return

        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
