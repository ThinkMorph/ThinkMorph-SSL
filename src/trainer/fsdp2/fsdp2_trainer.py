import contextlib
import gc
import json
import os
import random
import re
import shutil
import time
from functools import partial
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import send_to_device
from loguru import logger
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.utils.data import Dataset, DistributedSampler, IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler
from transformers.trainer_utils import seed_worker

import models.utils as model_utils
import parallel.process_group_manager as pgm
from eval.backends import EvalServerBackend
from trainer.config import TrainingArguments
from trainer.fsdp2.components import (
    CheckpointManager,
    CheckpointPolicy,
    DataStatsTracker,
    DebugBatchDumper,
    GCController,
    calculate_training_metrics,
)
from trainer.registry import TRAINER_REGISTER
from utils.distributed import dist_barrier
from utils.ema_utils import EMAHelper
from utils.fsdp2_utils import (
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    get_constant_schedule,
    get_cosine_schedule_with_warmup,
    get_wsd_schedule_with_warmup,
)
from utils.log_display import LogDisplay
from utils.profiler import StepProfiler
from utils.tracking import Tracking

DatasetType = Union[Dataset, IterableDataset]


@TRAINER_REGISTER.register("fsdp2_trainer")
class FSDP2SFTTrainer:
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: DatasetType,
        eval_dataset: DatasetType = None,
        processing_class=None,
        data_collator=None,
        dataset_stages=None,
        build_stage_dataset=None,
    ) -> None:
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class
        self.data_collator = data_collator

        # Multi-stage training
        self.dataset_stages = dataset_stages
        self.build_stage_dataset = build_stage_dataset
        self.current_stage_idx = 0
        self.stage_boundaries = []
        if dataset_stages:
            cumulative = 0
            for stage in dataset_stages:
                cumulative += stage.max_steps
                self.stage_boundaries.append(cumulative)
            logger.info(
                f"Multi-stage training enabled: {len(dataset_stages)} stages, "
                f"boundaries={self.stage_boundaries}"
            )
        self.default_backend = []
        if "wandb" in self.args.report_to:
            self.default_backend.append("wandb")
        self.default_backend.append("console")

        # Optional per-step PyTorch profiler configuration
        self.enable_profiler = self.args.enable_profiler
        self.profiler_config = self.args.profiler_config
        self.profiler_dir = os.path.join(self.args.output_dir, "profiler")
        profiler_rank = dist.get_rank() if self._is_dist_initialized() else 0
        self.step_profiler = StepProfiler(
            enable=self.enable_profiler,
            directory=self.profiler_dir,
            profiler_config=self.profiler_config,
            rank=profiler_rank,
        )
        self.accumulated_grad_steps = 0
        self._last_debug_print_step = -1

        # Optional EMA (fully opt-in)
        self.ema = EMAHelper(self.args)

        # Optional Eval Server Backend (only on rank 0)
        self.eval_backend = None
        if (
            self._rank() == 0
            and self.args.eval_config is not None
            and self.args.eval_strategy != "no"
        ):
            self.eval_backend = EvalServerBackend(
                url=self.args.eval_config.get("server_url"),
                poll_interval=self.args.eval_config.get("poll_interval", 20.0),
                eval_config=self.args.eval_config,
            )
            assert self.args.eval_steps == self.args.save_steps, (
                "eval_steps must be equal to save_steps"
            )

        self.checkpoint_manager = CheckpointManager(
            output_dir=self.args.output_dir,
            policy=CheckpointPolicy(
                keep_latest_k=int(self.args.checkpoint_keep_latest_k or 0),
                async_mode=str(self.args.checkpoint_async_mode or "disabled"),
            ),
        )
        self.gc_controller = GCController(
            enabled=bool(self.args.gc_enabled),
            gc_freq=int(self.args.gc_freq or 0),
        )

        self.loss_forward = self._default_loss_forward
        self.activation_offloading_enabled = bool(self.args.activation_offloading)
        self._activation_offload_warned = False
        self._retention_precedence_warned = False

        # --- Validation loss evaluation ---
        self._val_eval_enabled = False
        self._val_eval_schedule = None
        self._val_eval_max_batches = 0
        self._val_eval_dataloader = None
        self._setup_val_eval()
        self.debug_batch_dumper = DebugBatchDumper(
            enabled=bool(getattr(self.args, "debug_batch_input", False)),
            interval_steps=int(getattr(self.args, "print_batch_input_steps", -1) or -1),
        )

    def _setup_inference_probes(self):
        return

    def _setup_val_eval(self):
        cfg = getattr(self.args, "val_eval_config", None)
        if not cfg or not isinstance(cfg, dict):
            return
        if not bool(cfg.get("enabled", False)):
            return
        if not bool(cfg.get("loss_enabled", False)):
            return
        if self.eval_dataset is None:
            logger.warning(
                "val_eval_config.loss_enabled is set but no eval_dataset could be built. "
                "Disabling val loss eval."
            )
            return
        interval = int(cfg.get("interval_steps", 0) or 0)
        if interval <= 0:
            return
        from trainer.fsdp2.components import InferenceProbeSchedule

        start_step = int(cfg.get("start_step", interval))
        self._val_eval_max_batches = int(cfg.get("max_batches", 50) or 50)
        self._val_eval_schedule = InferenceProbeSchedule(
            enabled=True,
            interval_steps=interval,
            start_step=start_step,
        )
        self._val_eval_enabled = True
        logger.info(
            f"Val eval enabled: interval={interval}, start_step={start_step}, "
            f"max_batches={self._val_eval_max_batches}"
        )

    def _prepare_val_dataloader(self):
        if self._val_eval_dataloader is not None:
            return self._val_eval_dataloader
        if self.eval_dataset is None:
            return None
        self._val_eval_dataloader = self.prepare_dataloader(
            self.eval_dataset, is_training=False
        )
        return self._val_eval_dataloader

    def run_val_eval(self, rank: int):
        if not self._val_eval_enabled:
            return
        if self._val_eval_schedule is None:
            return
        if not self._val_eval_schedule.should_run(self.global_step):
            return

        val_dataloader = self._prepare_val_dataloader()
        if val_dataloader is None:
            return

        was_training = self.fsdp2_model.training
        self.fsdp2_model.eval()

        total_loss = torch.tensor(0.0, device=self.args.device)
        total_batches = 0

        # Subclass-specific accumulators (e.g. CE/MSE for Bagel)
        extra_accum = self._val_eval_init_extra_accumulators()

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    if batch_idx >= self._val_eval_max_batches:
                        break
                    batch = send_to_device(batch, self.fsdp2_model.device)
                    batch = self.prepare_batch(batch)
                    import contextlib

                    with contextlib.ExitStack() as stack:
                        stack.enter_context(self._autocast_context())
                        outputs = self.loss_forward(batch)
                    loss_dict = self.compute_loss_dict(outputs)
                    loss = loss_dict["loss"]
                    if self._world_size() > 1:
                        loss = loss.mean()
                    total_loss += loss.detach()
                    total_batches += 1
                    self._val_eval_accumulate_extra(extra_accum, loss_dict)
        except Exception as e:
            logger.exception(
                f"[rank {rank}] Val eval failed at step {self.global_step}: "
                f"{type(e).__name__}: {e!r}"
            )
        finally:
            if was_training:
                self.fsdp2_model.train()

        if total_batches == 0:
            return

        avg_loss = total_loss / total_batches
        if self._world_size() > 1:
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)

        metrics = {"val/loss": avg_loss.item(), "val/batches": total_batches}
        metrics.update(self._val_eval_reduce_extra(extra_accum, total_batches))

        if rank == 0 and hasattr(self, "tracking"):
            self.tracking.log(metrics, step=self.global_step)
            logger.info(
                f"Val eval step {self.global_step}: loss={metrics['val/loss']:.4f} "
                f"({total_batches} batches)"
            )

    def _val_eval_init_extra_accumulators(self):
        """Override in subclass to track extra val metrics (e.g. CE/MSE)."""
        return {}

    def _val_eval_accumulate_extra(self, accum, loss_dict):
        """Override in subclass to accumulate extra val metrics."""
        pass

    def _val_eval_reduce_extra(self, accum, total_batches):
        """Override in subclass to reduce extra val metrics."""
        return {}

    @staticmethod
    def _is_dist_initialized() -> bool:
        return dist.is_available() and dist.is_initialized()

    def _world_size(self) -> int:
        return dist.get_world_size() if self._is_dist_initialized() else 1

    def _rank(self) -> int:
        return dist.get_rank() if self._is_dist_initialized() else 0

    def run_inference_probes(self, rank: int):
        return

    def run_bench_eval(self, rank: int):
        return

    def run_val_infer(self, rank: int):
        return

    def prepare_dataloader(self, dataset: DatasetType, is_training: bool = True):
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": self.args.train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if isinstance(dataset, IterableDataset):
            if self.args.group_by_length:
                logger.warning("group_by_length=True is ignored for IterableDataset.")
            sampler = None
        elif self.args.group_by_length:
            sampler = DistributedLengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=dataset,
                lengths=dataset.modality_length,
                model_input_name=None,
                num_replicas=pgm.process_group_manager.dp_world_size,
                rank=pgm.process_group_manager.dp_rank,
            )
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=pgm.process_group_manager.dp_world_size,
                rank=pgm.process_group_manager.dp_rank,
            )
        dataloader_params["sampler"] = sampler
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        if is_training:
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=pgm.process_group_manager.dp_rank,
            )
        dataloader = StatefulDataLoader(dataset, **dataloader_params)
        return dataloader

    def prepare_model(self):
        if self.args.bf16:
            param_dtype = torch.bfloat16
        else:
            param_dtype = torch.float16

        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Freeze modules listed in freeze_modules before FSDP wrapping
        freeze_modules = getattr(self.args, "freeze_modules", None) or []
        for module_name in freeze_modules:
            module = getattr(self.model, module_name, None)
            if module is None:
                logger.warning(
                    f"freeze_modules: '{module_name}' not found on model, skipping"
                )
                continue
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
            logger.info(
                f"Frozen module '{module_name}': all parameters set to requires_grad=False"
            )

        reduce_dtype = (
            getattr(torch, self.args.reduce_dtype) if self.args.reduce_dtype else None
        )
        output_dtype = (
            getattr(torch, self.args.output_dtype) if self.args.output_dtype else None
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            output_dtype=output_dtype,
        )
        fsdp2_cfg = self._get_fsdp2_config()

        fsdp_kwargs = {
            "reshard_after_forward": fsdp2_cfg.get("reshard_after_forward", True),
            "mp_policy": mp_policy,
        }

        transformer_cls_names_to_wrap = fsdp2_cfg.get(
            "transformer_layer_cls_to_wrap", None
        )
        full_state = self.model.state_dict()
        logger.info("Applying FSDP2 to model")
        apply_fsdp2(self.model, fsdp_kwargs, transformer_cls_names_to_wrap)
        logger.info("Loading full state dict to model")
        fsdp2_load_full_state_dict(self.model, full_state)
        logger.info("FSDP2 applied to model")
        self.fsdp2_model = self.model

        if getattr(self.args, "use_tm_activation_checkpoint", False):
            import functools
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                apply_activation_checkpointing,
                checkpoint_wrapper,
            )
            from models.bagel.qwen2_navit import (
                Qwen2DecoderLayer,
                Qwen2MoEDecoderLayer,
                Qwen2MoTDecoderLayer,
            )
            from models.bagel.siglip_navit import SiglipEncoderLayer
            from models.bagel.bagel import MLPconnector

            target_classes = (
                Qwen2DecoderLayer,
                SiglipEncoderLayer,
                MLPconnector,
                Qwen2MoEDecoderLayer,
                Qwen2MoTDecoderLayer,
            )
            apply_activation_checkpointing(
                self.fsdp2_model,
                checkpoint_wrapper_fn=functools.partial(
                    checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                ),
                check_fn=lambda m: isinstance(m, target_classes),
            )
            logger.info("Applied ThinkMorph-style activation checkpointing")

        if bool(self.args.compile_model):
            try:
                self.fsdp2_model = torch.compile(self.fsdp2_model)
                self.model = self.fsdp2_model
                logger.info("Enabled torch.compile for model")
            except Exception as e:
                logger.warning(f"Failed to compile model; fallback to eager mode: {e}")

        if bool(self.args.compile_loss):
            try:
                self.loss_forward = torch.compile(self._default_loss_forward)
                logger.info("Enabled torch.compile for loss forward")
            except Exception as e:
                logger.warning(
                    f"Failed to compile loss forward; fallback to eager mode: {e}"
                )

        del full_state
        gc.collect()
        torch.cuda.empty_cache()

    def _get_fsdp2_config(self) -> Dict[str, Any]:
        # Backward compatibility: still honor legacy `fsdp_config` if present.
        cfg = getattr(self.args, "fsdp2_config", None)
        if cfg is None:
            cfg = getattr(self.args, "fsdp_config", None)
        return dict(cfg or {})

    def prepare_optimizer(self):
        if getattr(self.args, "use_muon", False):
            self._prepare_muon_optimizer()
        else:
            trainable_params = [
                p for p in self.fsdp2_model.parameters() if p.requires_grad
            ]
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )

    def _prepare_muon_optimizer(self):
        from utils.muon_utils import Muon

        # Patterns whose matched params should use AdamW instead of Muon.
        adam_patterns = [
            "emb",
            "norm",
            "lm_head",
            "bias",
            "wte",
            "wpe",
            "output",
            "a_proj",
            "b_proj",
            "conv1d",
            "rotary",
        ]

        muon_wd, muon_no_wd = [], []
        adam_wd, adam_no_wd = [], []

        for name, param in self.fsdp2_model.named_parameters():
            if not param.requires_grad:
                continue
            use_muon = param.ndim == 2 and not any(p in name for p in adam_patterns)
            no_decay = param.ndim < 2 or any(p in name for p in ["bias", "norm", "emb"])
            if use_muon:
                (muon_no_wd if no_decay else muon_wd).append(param)
            else:
                (adam_no_wd if no_decay else adam_wd).append(param)

        param_groups = []
        if muon_wd:
            param_groups.append(
                {
                    "params": muon_wd,
                    "use_muon": True,
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                    "momentum": 0.95,
                }
            )
        if muon_no_wd:
            param_groups.append(
                {
                    "params": muon_no_wd,
                    "use_muon": True,
                    "lr": self.args.learning_rate,
                    "weight_decay": 0.0,
                    "momentum": 0.95,
                }
            )
        if adam_wd:
            param_groups.append(
                {
                    "params": adam_wd,
                    "use_muon": False,
                    "lr": self.args.learning_rate,
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                    "weight_decay": self.args.weight_decay,
                }
            )
        if adam_no_wd:
            param_groups.append(
                {
                    "params": adam_no_wd,
                    "use_muon": False,
                    "lr": self.args.learning_rate,
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                    "weight_decay": 0.0,
                }
            )

        n_muon = sum(len(g["params"]) for g in param_groups if g["use_muon"])
        n_adam = sum(len(g["params"]) for g in param_groups if not g["use_muon"])
        logger.info(
            f"Muon optimizer: {n_muon} params via Muon, {n_adam} params via AdamW"
        )

        self.optimizer = Muon(
            param_groups,
            defaults=dict(lr=self.args.learning_rate),
        )

    def prepare_scheduler(
        self,
        num_warmup_steps: int,
        num_training_steps: int,
    ):
        self.args.lr_scheduler_kwargs = self.args.lr_scheduler_kwargs or {}
        if self.args.lr_scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **self.args.lr_scheduler_kwargs,
            )
        elif self.args.lr_scheduler_type == "wsd":
            self.scheduler = get_wsd_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **self.args.lr_scheduler_kwargs,
            )
        elif self.args.lr_scheduler_type == "constant":
            self.scheduler = get_constant_schedule(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                **self.args.lr_scheduler_kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {self.args.lr_scheduler_type}"
            )

    def _default_loss_forward(self, batch):
        outputs = self.model(**batch)
        return outputs

    def _autocast_context(self):
        if not torch.cuda.is_available():
            return contextlib.nullcontext()
        cast_dtype = torch.bfloat16 if self.args.bf16 else torch.float16
        return torch.autocast(device_type="cuda", dtype=cast_dtype)

    def _activation_offload_context(self):
        if not self.activation_offloading_enabled or not torch.cuda.is_available():
            return contextlib.nullcontext()
        graph_module = getattr(torch.autograd, "graph", None)
        if graph_module is None or not hasattr(graph_module, "save_on_cpu"):
            if not self._activation_offload_warned:
                logger.warning(
                    "activation_offloading is enabled but torch.autograd.graph.save_on_cpu is unavailable; fallback to normal forward"
                )
                self._activation_offload_warned = True
            return contextlib.nullcontext()
        return torch.autograd.graph.save_on_cpu(pin_memory=True)

    def prepare_batch(self, batch):
        return batch

    def forward_step(self, batch):
        with contextlib.ExitStack() as stack:
            stack.enter_context(self._autocast_context())
            stack.enter_context(self._activation_offload_context())
            outputs = self.loss_forward(batch)
        return outputs

    def _effective_checkpoint_keep_latest_k(self, total_limit: int | None) -> int:
        configured = int(self.args.checkpoint_keep_latest_k or 0)
        fallback = int(total_limit or 0)
        if configured > 0:
            if (
                fallback > 0
                and fallback != configured
                and not self._retention_precedence_warned
            ):
                logger.warning(
                    "Both checkpoint_keep_latest_k={} and save_total_limit={} are set; using checkpoint_keep_latest_k",
                    configured,
                    fallback,
                )
                self._retention_precedence_warned = True
            return configured
        return fallback

    def _is_save_only_model(self) -> bool:
        return bool(getattr(self.args, "save_only_model", False))

    @staticmethod
    def _checkpoint_meta_path(output_path: str) -> str:
        return os.path.join(output_path, "checkpoint_meta.json")

    def _write_checkpoint_meta(self, output_path: str, save_only_model: bool) -> None:
        meta = {"save_only_model": bool(save_only_model)}
        with open(self._checkpoint_meta_path(output_path), "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def _read_checkpoint_meta(self, output_path: str) -> dict[str, Any]:
        meta_path = self._checkpoint_meta_path(output_path)
        if not os.path.exists(meta_path):
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            logger.warning(
                "Failed to parse checkpoint metadata at {}; fallback to full-state load",
                meta_path,
            )
        return {}

    def compute_loss_dict(self, outputs):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return {"loss": loss}

    def backward_step(self, loss):
        loss.backward()

    def reduce_step_metrics(self, loss_dict, grad_norm=None):
        lr = self.scheduler.get_last_lr()[0]
        loss_item = torch.tensor(float(loss_dict["loss"]), device=self.args.device)
        if self._world_size() > 1:
            torch.distributed.all_reduce(loss_item, op=torch.distributed.ReduceOp.AVG)
        metrics = {
            "train/loss": loss_item.item(),
            "train/lr": lr,
        }
        if grad_norm is not None:
            metrics["train/grad_norm"] = float(grad_norm.detach().float().item())
        return metrics

    def on_after_optimizer_step(self, outputs, loss_dict):
        return None

    def training_step(self, batch):
        self.fsdp2_model.train()
        batch = self.prepare_batch(batch)
        if self.accumulated_grad_steps == 0:
            self.optimizer.zero_grad()
        outputs = self.forward_step(batch)
        raw_loss_dict = self.compute_loss_dict(outputs)
        loss = raw_loss_dict["loss"]
        if self._world_size() > 1:
            loss = loss.mean()
        loss = loss / self.args.gradient_accumulation_steps
        self.backward_step(loss)
        loss_item = loss.detach().item() * self.args.gradient_accumulation_steps
        self.accumulated_grad_steps += 1
        should_update = (
            self.accumulated_grad_steps >= self.args.gradient_accumulation_steps
        )
        grad_norm = None
        did_step = False
        if should_update:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.fsdp2_model.parameters(), self.args.max_grad_norm
            )
            # if grad_norm is not finite, skip the update
            if not torch.isfinite(grad_norm):
                logger.warning("grad_norm is not finite: {}", grad_norm)
                self.optimizer.zero_grad()
            else:
                self.optimizer.step()
                did_step = True
                self.ema.update(step=self.global_step + 1)

            self.scheduler.step()
            self.accumulated_grad_steps = 0
            self.on_after_optimizer_step(outputs, raw_loss_dict)

        metric_loss_dict = dict(raw_loss_dict)
        metric_loss_dict["loss"] = loss_item
        metric_loss_dict["did_step"] = did_step
        metrics = self.reduce_step_metrics(metric_loss_dict, grad_norm=grad_norm)
        return metrics

    def validation_step(self, output_dir, step: int):
        if self.eval_backend is not None:
            checkpoint_type = "regular" if not self.ema.is_enabled() else "ema"
            checkpoint_path = os.path.abspath(output_dir)
            eval_output_dir = os.path.join(checkpoint_path, "eval")
            self.eval_backend.submit_eval(
                checkpoint_path,
                step,
                eval_output_dir,
                checkpoint_type=checkpoint_type,
            )

    def _check_eval_results(self, rank: int, wait_until_complete: bool = False):
        if self.eval_backend is None:
            return
        if wait_until_complete:
            logger.info("Waiting for pending evaluation jobs to complete...")
            while len(self.eval_backend.pending_evals) > 0:
                for (
                    eval_step,
                    metrics,
                ) in self.eval_backend.check_and_get_completed():
                    if rank == 0:
                        metrics["global_step"] = eval_step
                        self.tracking.log(metrics)
                time.sleep(self.eval_backend.poll_interval)
            logger.info("All evaluation jobs completed")
        else:
            for (
                eval_step,
                metrics,
            ) in self.eval_backend.check_and_get_completed():
                if rank == 0:
                    metrics["global_step"] = eval_step
                    self.tracking.log(metrics)

    def _should_barrier_after_checkpoint_save(self) -> bool:
        if self._world_size() <= 1:
            return False
        is_async_enabled = False
        if hasattr(self.checkpoint_manager, "_is_async_enabled"):
            is_async_enabled = bool(self.checkpoint_manager._is_async_enabled())
        else:
            mode = getattr(
                getattr(self.checkpoint_manager, "policy", None), "async_mode", None
            )
            is_async_enabled = str(mode or "disabled").lower() in {
                "async",
                "enabled",
                "true",
                "1",
            }
        return not is_async_enabled

    def train(self, resume_from_checkpoint: bool = False):
        self.prepare_model()
        train_dataloader = self.prepare_dataloader(self.train_dataset, is_training=True)
        self.train_dataloader = train_dataloader
        self.prepare_optimizer()

        # Validate config for IterableDataset and Dataset
        self.prepare_and_validate_config()

        warmup_steps = (
            int(self.total_steps * self.args.warmup_ratio)
            if self.args.warmup_ratio > 0
            else self.args.warmup_steps
        )
        self.prepare_scheduler(warmup_steps, self.total_steps)
        rank = self._rank()
        world_size = self._world_size()
        # Initialize tracking
        if rank == 0:
            self.tracking = Tracking(
                project_name=os.environ.get("WANDB_PROJECT", self.args.project),
                experiment_name=os.environ.get("WANDB_NAME", self.args.run_name),
                default_backend=self.default_backend,
                config=self.args,
            )
            # Upload merged config.yaml to wandb for easy inspection.
            config_yaml_path = os.path.join(self.args.output_dir, "config.yaml")
            if os.path.exists(config_yaml_path) and "wandb" in self.tracking.logger:
                try:
                    import wandb as _wandb

                    _wandb.save(
                        config_yaml_path,
                        base_path=self.args.output_dir,
                        policy="now",
                    )
                except Exception as e:
                    logger.warning(f"Failed to wandb.save(config.yaml): {e}")

        self.total_tokens = 0
        self.data_stats_tracker = DataStatsTracker()
        loaded_checkpoint_dir: Optional[str] = None
        if resume_from_checkpoint:
            latest_checkpoint_path = self.checkpoint_manager.latest_checkpoint()
            if latest_checkpoint_path is None:
                raise ValueError(
                    f"resume_from_checkpoint=True but no checkpoint found in {self.args.output_dir}"
                )
            latest_checkpoint = os.path.basename(latest_checkpoint_path)
            loaded_checkpoint_dir = latest_checkpoint_path
            self.load_checkpoints(
                loaded_checkpoint_dir,
                int(latest_checkpoint.split("-")[1]),
            )
            start_epoch = int(latest_checkpoint.split("-")[1]) / self.steps_per_epoch
            # start_epoch is a float, we need to convert it to an integer
            start_epoch = int(start_epoch)
            self.global_step = int(latest_checkpoint.split("-")[1])
            need_update_pbar = True
        else:
            start_epoch = 0
            self.global_step = 0
            need_update_pbar = False

        # Multi-stage: if resumed into a later stage, rebuild the correct dataset
        if (
            self.dataset_stages
            and self.current_stage_idx > 0
            and self.build_stage_dataset is not None
        ):
            stage = self.dataset_stages[self.current_stage_idx]
            logger.info(
                f"Resuming into stage {self.current_stage_idx}, "
                f"rebuilding dataset from folders: {stage.dataset_folders}"
            )
            new_dataset = self.build_stage_dataset(stage)
            self.train_dataset = new_dataset
            self.train_dataloader = self.prepare_dataloader(
                new_dataset, is_training=True
            )

        # Initialize EMA after model weights are loaded (and optionally restore from checkpoint).
        self.ema.maybe_init(
            model=self.fsdp2_model, checkpoint_dir=loaded_checkpoint_dir
        )

        logger.info(
            f"Training with {self.args.num_train_epochs} epochs with {self.total_steps} steps"
        )
        self.step_profiler.start()

        # Probe scheduler decides whether to run; step=0 is intentionally skipped for FSDP2.
        self.run_inference_probes(rank)

        curr_epoch = start_epoch

        # Rich-aware progress display on rank 0 only. In TTY mode it renders a
        # live bar whose loss/grad/lr fields update in place; in non-TTY
        # (sbatch) it emits one compact line per step via loguru.
        if rank == 0:
            log_display = LogDisplay(
                total_steps=self.total_steps,
                initial_step=self.global_step,
            )
        else:
            log_display = None
        while not self.should_stop():
            if hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(curr_epoch)

            # LogDisplay's initial_step already reflects resume; no extra catch-up needed.
            if need_update_pbar:
                need_update_pbar = False

            for step, batch in enumerate(self.train_dataloader):
                if self.should_stop():
                    break
                # send batch to device
                batch = send_to_device(batch, self.fsdp2_model.device)
                # Pop data-stats metadata before training_step so the compiled
                # model never sees these non-tensor keys.
                self.data_stats_tracker.pop_and_update(batch)
                start_time = time.perf_counter()
                train_metrics = self.training_step(batch)
                self.step_profiler.step()
                if self.step_profiler.should_save(self.global_step + 1):
                    self.step_profiler.stop_and_save()
                    self.step_profiler.stop_trace()
                end_time = time.perf_counter()
                delta_time = end_time - start_time

                # Calculate flops per rank
                seq_len = (
                    batch.get("attention_mask", torch.zeros((1, 1)))
                    .sum(dim=1)
                    .detach()
                    .cpu()
                    .tolist()
                )
                flops, promised_flops = model_utils.flops_counter.estimate_flops(
                    seq_len, delta_time=delta_time
                )
                device = self.fsdp2_model.device
                flops_tensor = torch.tensor(flops, device=device)
                sp_size = pgm.process_group_manager.cp_world_size

                # Calculate training metrics (MFU, token stats, throughput)
                perf_metrics, self.total_tokens = calculate_training_metrics(
                    flops_tensor=flops_tensor,
                    sp_size=sp_size,
                    promised_flops=promised_flops,
                    device=device,
                    seq_len=seq_len,
                    total_tokens=self.total_tokens,
                    delta_time=delta_time,
                    world_size=world_size,
                )
                train_metrics.update(perf_metrics)
                self.print_batch_input(batch)

                is_accumulation_complete = self.accumulated_grad_steps == 0

                if self.steps_per_epoch is not None and self.steps_per_epoch > 0:
                    epoch_progress = f"{self.global_step / self.steps_per_epoch:.2f}"
                    train_metrics["train/epoch"] = float(epoch_progress)
                if is_accumulation_complete:
                    # Aggregate data stats across ranks and merge into metrics.
                    # All ranks must participate in the collective all_reduce,
                    # but only rank 0 receives the dict to log.
                    data_metrics = self.data_stats_tracker.emit(
                        device=self.fsdp2_model.device,
                        is_rank0=(rank == 0),
                        include_histograms=True,
                    )
                    if rank == 0:
                        train_metrics.update(data_metrics)
                        self.tracking.log(train_metrics, step=self.global_step)
                        if log_display is not None:
                            log_display.update(self.global_step + 1, train_metrics)
                    self.global_step += 1
                    self.run_inference_probes(rank)
                    self.run_val_eval(rank)
                    self.run_val_infer(rank)
                    self.run_bench_eval(rank)

                    # Multi-stage: save checkpoint and switch dataset at stage boundary
                    if self._should_switch_stage():
                        output_dir = os.path.join(
                            self.args.output_dir,
                            f"checkpoint-{self.global_step}",
                        )
                        self.save_checkpoints(
                            output_dir,
                            self.global_step,
                            total_limit=self.args.save_total_limit,
                        )
                        self._switch_to_next_stage()
                        break

                    if self.should_save:
                        output_dir = os.path.join(
                            self.args.output_dir,
                            f"checkpoint-{self.global_step}",
                        )
                        self.save_checkpoints(
                            output_dir,
                            self.global_step,
                            total_limit=self.args.save_total_limit,
                        )
                        if self.eval_backend is not None:
                            self.checkpoint_manager.finalize_pending_save()
                        self.validation_step(output_dir, self.global_step)

                    if (
                        self.args.torch_empty_cache_steps is not None
                        and self.global_step % self.args.torch_empty_cache_steps == 0
                    ):
                        self.empty_cache()
                    self.gc_controller.maybe_collect(self.global_step)
                self._check_eval_results(rank)
            curr_epoch += 1

        if log_display is not None:
            log_display.close()
        # Save the final checkpoint
        output_dir = os.path.join(
            self.args.output_dir, f"checkpoint-{self.global_step}"
        )
        self.save_checkpoints(
            output_dir, self.global_step, total_limit=self.args.save_total_limit
        )
        self.checkpoint_manager.finalize_pending_save()
        self.validation_step(output_dir, self.global_step)
        # Wait for all pending eval jobs to complete
        if self.eval_backend is not None:
            self._check_eval_results(rank, wait_until_complete=True)

        # Flush wandb with a hard timeout so training doesn't hang at shutdown.
        if rank == 0 and hasattr(self, "tracking"):
            self.tracking.close(
                timeout=float(os.environ.get("WANDB_CLOSE_TIMEOUT", "120"))
            )

    def evaluate(self):
        raise NotImplementedError("Evaluation is not implemented")

    def remove_old_checkpoints(self, output_path: str, total_limit: int = None):
        if total_limit is None:
            return
        # get all checkpoints in output_path
        checkpoints = [f for f in os.listdir(output_path) if f.startswith("checkpoint")]
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        if len(checkpoints) > total_limit:
            for checkpoint in checkpoints[:-total_limit]:
                logger.info(f"Removing checkpoint {checkpoint}")
                shutil.rmtree(os.path.join(output_path, checkpoint))

    def save_checkpoints(self, output_path: str, step: int, total_limit: int = None):
        if self._is_save_only_model():
            state_dict = {
                "model": get_model_state_dict(self.fsdp2_model),
                "extra_state": {
                    "lr_scheduler_state": self.scheduler.state_dict(),
                    "rng": self.get_rng_state(),
                    "total_tokens": self.total_tokens,
                    "accumulated_grad_steps": self.accumulated_grad_steps,
                    "current_stage_idx": self.current_stage_idx,
                    "data_stats": self.data_stats_tracker.state_dict(),
                },
                "dataloader_state": self.train_dataloader.state_dict(),
            }
            logger.info(
                "save_only_model=True: saving model weights + training metadata (no optimizer)"
            )
        else:
            state_dict = {
                "model": get_model_state_dict(self.fsdp2_model),
                "optimizer": get_optimizer_state_dict(self.fsdp2_model, self.optimizer),
                "extra_state": {
                    "lr_scheduler_state": self.scheduler.state_dict(),
                    "rng": self.get_rng_state(),
                    "total_tokens": self.total_tokens,
                    "accumulated_grad_steps": self.accumulated_grad_steps,
                    "current_stage_idx": self.current_stage_idx,
                    "data_stats": self.data_stats_tracker.state_dict(),
                },
                "dataloader_state": self.train_dataloader.state_dict(),
            }
        if self.ema.is_enabled() and self.ema.initialized:
            state_dict["ema_model"] = self.ema.state_dict_for_save(self.fsdp2_model)

        self.checkpoint_manager.save(state_dict, output_path)
        logger.info(f"Saved DCP checkpoint to {output_path} at step {step}")

        if self._rank() == 0:
            self.processing_class.save_pretrained(output_path)
            self.model.config.save_pretrained(output_path)
            self._write_checkpoint_meta(
                output_path, save_only_model=self._is_save_only_model()
            )
            keep_latest_k = self._effective_checkpoint_keep_latest_k(total_limit)
            if keep_latest_k > 0:
                self.checkpoint_manager.policy.keep_latest_k = keep_latest_k
                self.checkpoint_manager.purge_old_checkpoints()
        if self._should_barrier_after_checkpoint_save():
            dist_barrier()

    @property
    def should_save(self):
        return self.global_step % self.args.save_steps == 0 and self.global_step > 0

    def load_checkpoints(self, output_path: str, step: int):
        checkpoint_meta = self._read_checkpoint_meta(output_path)
        checkpoint_is_model_only = bool(checkpoint_meta.get("save_only_model", False))
        state_dict = {"model": get_model_state_dict(self.fsdp2_model)}
        if not checkpoint_is_model_only:
            state_dict["optimizer"] = get_optimizer_state_dict(
                self.fsdp2_model, self.optimizer
            )
        # Always try to load training metadata (saved even with save_only_model)
        state_dict["extra_state"] = {}
        state_dict["dataloader_state"] = {}
        if self.ema.is_enabled():
            state_dict["ema_model"] = get_model_state_dict(self.fsdp2_model)

        self.checkpoint_manager.load(state_dict, output_path)

        set_model_state_dict(self.fsdp2_model, state_dict["model"])
        if not checkpoint_is_model_only and "optimizer" in state_dict:
            set_optimizer_state_dict(
                self.fsdp2_model, self.optimizer, state_dict["optimizer"]
            )
        else:
            logger.info(
                "Loaded model-only checkpoint; optimizer/scheduler/rng/dataloader states were not restored"
            )

        extra_state = state_dict.get("extra_state", {})
        self.total_tokens = extra_state.get("total_tokens", 0)
        if "total_tokens" not in extra_state:
            logger.warning(
                "Checkpoint extra_state is missing 'total_tokens'; defaulting to 0."
            )

        self.accumulated_grad_steps = extra_state.get("accumulated_grad_steps", 0)
        self.data_stats_tracker.load_state_dict(extra_state.get("data_stats"))

        if self.dataset_stages:
            self.current_stage_idx = extra_state.get("current_stage_idx", 0)
            logger.info(
                f"Restored multi-stage training at stage {self.current_stage_idx}"
            )

        rng_state = extra_state.get("rng")
        if rng_state is not None:
            self.load_rng_state(rng_state)
        else:
            logger.warning(
                "Checkpoint extra_state is missing 'rng'; skipping RNG restore."
            )

        scheduler_state = extra_state.get("lr_scheduler_state")
        if scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        else:
            logger.warning(
                "Checkpoint extra_state is missing 'lr_scheduler_state'; skipping scheduler restore."
            )

        dataloader_state = state_dict.get("dataloader_state")
        if dataloader_state:
            self.train_dataloader.load_state_dict(dataloader_state)
        else:
            logger.warning(
                "Checkpoint is missing 'dataloader_state' or it is empty; skipping dataloader restore."
            )
        logger.info(f"Loaded DCP checkpoint from {output_path} at step {step}")

    def get_rng_state(self):
        return {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

    def prepare_and_validate_config(self):
        if isinstance(self.train_dataset, IterableDataset):
            is_iterable_dataset = True
        else:
            is_iterable_dataset = False

        if is_iterable_dataset:
            assert self.args.max_steps > 0, "max_steps must be set for IterableDataset"
            if self.args.num_train_epochs > 1:
                logger.warning("num_train_epochs will be ignored for IterableDataset")
                self.args.num_train_epochs = 1
            if self.dataset_stages:
                self.total_steps = self.stage_boundaries[-1]
            else:
                self.total_steps = self.args.max_steps
            self.steps_per_epoch = self.total_steps
        else:
            self.steps_per_epoch = (
                len(self.train_dataloader) + self.args.gradient_accumulation_steps - 1
            ) // self.args.gradient_accumulation_steps
            self.total_steps = (
                self.steps_per_epoch * self.args.num_train_epochs
                if self.args.max_steps < 0
                else self.args.max_steps
            )

    def should_stop(self):
        if self.global_step >= self.total_steps and self.total_steps > 0:
            return True
        return False

    def _should_switch_stage(self):
        if not self.dataset_stages:
            return False
        if self.current_stage_idx >= len(self.dataset_stages) - 1:
            return False
        return self.global_step >= self.stage_boundaries[self.current_stage_idx]

    def _switch_to_next_stage(self):
        self.current_stage_idx += 1
        stage = self.dataset_stages[self.current_stage_idx]
        logger.info(
            f"Switching to training stage {self.current_stage_idx}: "
            f"folders={stage.dataset_folders}, "
            f"steps {self.global_step} -> {self.stage_boundaries[self.current_stage_idx]}"
        )
        new_dataset = self.build_stage_dataset(stage)
        self.train_dataset = new_dataset
        self.train_dataloader = self.prepare_dataloader(new_dataset, is_training=True)
        if self._is_dist_initialized():
            dist_barrier()

    def load_rng_state(self, rng_state):
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

    def empty_cache(self):
        gc.collect()
        torch.cuda.empty_cache()

    def print_batch_input(self, batch):
        if not self.debug_batch_dumper.should_dump(self.global_step):
            return
        # Global step stays constant across gradient-accumulation micro-batches.
        # Print at most once per global step to avoid duplicate dumps.
        if self.global_step == self._last_debug_print_step:
            return
        debug_rank = getattr(self.args, "debug_batch_input_rank", 0)
        if debug_rank is not None and debug_rank >= 0 and dist.is_initialized():
            if dist.get_rank() != int(debug_rank):
                return
        try:
            self._print_batch_input_impl(batch)
            self._last_debug_print_step = self.global_step
        except Exception as e:
            logger.error(f"Error printing batch input: {e}")

    @staticmethod
    def _truncate_preserving_special_tokens(
        text: str, head_chars: int, tail_chars: int
    ) -> str:
        """Truncate decoded text without splitting special tokens like <|...|>."""
        if not text:
            return ""

        # Keep behavior close to previous implementation when text is short.
        min_gap = 20
        if len(text) <= head_chars + tail_chars + min_gap:
            return text

        special_re = re.compile(r"(<\|[^<>|]+?\|>)")
        parts = special_re.split(text)

        def _take_head(char_budget: int) -> str:
            out = []
            remaining = max(char_budget, 0)
            for part in parts:
                if not part:
                    continue
                is_special = special_re.fullmatch(part) is not None
                if is_special:
                    out.append(part)
                    remaining = max(remaining - len(part), 0)
                    continue
                if remaining <= 0:
                    break
                if len(part) <= remaining:
                    out.append(part)
                    remaining -= len(part)
                else:
                    out.append(part[:remaining])
                    break
            return "".join(out)

        def _take_tail(char_budget: int) -> str:
            out = []
            remaining = max(char_budget, 0)
            for part in reversed(parts):
                if not part:
                    continue
                is_special = special_re.fullmatch(part) is not None
                if is_special:
                    out.append(part)
                    remaining = max(remaining - len(part), 0)
                    continue
                if remaining <= 0:
                    break
                if len(part) <= remaining:
                    out.append(part)
                    remaining -= len(part)
                else:
                    out.append(part[-remaining:])
                    break
            return "".join(reversed(out))

        head = _take_head(head_chars)
        tail = _take_tail(tail_chars)
        return f"{head} ...[{len(text)} chars]... {tail}"

    def _format_turnwise_text(
        self,
        decoded_text: str,
        marker: str,
        head_chars: int,
        tail_chars: int,
    ) -> list[str]:
        segments = decoded_text.split(marker)
        turns = []
        turn_idx = 0
        for i, seg in enumerate(segments):
            if i == 0 and not seg.strip():
                continue
            turn = f"{marker}{seg}" if i > 0 else seg
            turn = self._truncate_preserving_special_tokens(
                turn,
                head_chars=head_chars,
                tail_chars=tail_chars,
            )
            turns.append(f"[turn {turn_idx}] {turn}")
            turn_idx += 1
        return turns

    def _format_token_context_windows(
        self,
        token_ids: torch.Tensor,
        tokenizer,
        target_token_id: int,
        context_tokens: int = 24,
        label: str = "token",
        max_hits: int = 8,
    ) -> list[str]:
        if (
            token_ids is None
            or not torch.is_tensor(token_ids)
            or token_ids.numel() == 0
        ):
            return []

        positions = (
            (token_ids == int(target_token_id)).nonzero(as_tuple=False).flatten()
        )
        if positions.numel() == 0:
            return []

        lines = [f"--- {label} contexts (+/-{int(context_tokens)} tokens) ---"]
        total = int(token_ids.numel())
        hit_positions = positions.tolist()[: max(int(max_hits), 0)]
        for hit_idx, pos in enumerate(hit_positions):
            start = max(int(pos) - int(context_tokens), 0)
            end = min(int(pos) + int(context_tokens) + 1, total)
            prefix = tokenizer.decode(token_ids[start:pos], skip_special_tokens=False)
            target = tokenizer.decode(
                token_ids[pos : pos + 1], skip_special_tokens=False
            )
            suffix = tokenizer.decode(
                token_ids[pos + 1 : end], skip_special_tokens=False
            )
            lines.append(f"[hit {hit_idx} @ {pos}] {prefix}<<<{target}>>>{suffix}")
        if positions.numel() > len(hit_positions):
            lines.append(
                f"... {int(positions.numel()) - len(hit_positions)} more {label} hit(s) omitted"
            )
        return lines

    def _print_batch_input_impl(self, batch, head_chars=120, tail_chars=80):
        tokenizer = self.processing_class.processor
        lines = [f"{'=' * 60} [Step {self.global_step}] Batch Debug {'=' * 60}"]

        # --- overview ---
        seq_len = batch.get("sequence_length", None)
        lines.append(
            f"sequence_length: {int(seq_len) if torch.is_tensor(seq_len) else seq_len}"
        )

        sample_lens = batch.get("sample_lens", None)
        if sample_lens is not None:
            if torch.is_tensor(sample_lens):
                sample_lens = sample_lens.flatten().tolist()
            lines.append(f"sample_lens ({len(sample_lens)} samples): {sample_lens}")

        sample_meta = batch.get("sample_meta", None)
        if sample_meta:
            preview = sample_meta[: min(len(sample_meta), 8)]
            lines.append(
                f"sample_meta ({len(sample_meta)} samples, first {len(preview)}): {preview}"
            )

        # --- token counts ---
        ce_idx = batch.get("ce_loss_indexes", None)
        mse_idx = batch.get("mse_loss_indexes", None)
        ce_count = len(ce_idx) if ce_idx is not None and torch.is_tensor(ce_idx) else 0
        mse_count = (
            len(mse_idx) if mse_idx is not None and torch.is_tensor(mse_idx) else 0
        )
        lines.append(f"ce_loss tokens: {ce_count}  |  mse_loss tokens: {mse_count}")
        lines.append(
            "loss target spaces: ce_loss=text labels, mse_loss=image latent tokens"
        )

        vae_shapes = batch.get("patchified_vae_latent_shapes", None)
        if vae_shapes is not None:
            if torch.is_tensor(vae_shapes):
                shapes_list = [tuple(s.tolist()) for s in vae_shapes]
            else:
                shapes_list = vae_shapes
            lines.append(f"vae_latent_shapes ({len(shapes_list)}): {shapes_list}")

        # --- text sequence (truncated per segment) ---
        text_ids = batch.get("packed_text_ids", None)
        if text_ids is not None and torch.is_tensor(text_ids):
            full_decoded = tokenizer.decode(text_ids, skip_special_tokens=False)
            marker = "<|im_start|>"
            lines.append("--- text sequence (turn-wise, truncated) ---")
            lines.extend(
                self._format_turnwise_text(
                    full_decoded,
                    marker=marker,
                    head_chars=head_chars,
                    tail_chars=tail_chars,
                )
            )

        # --- labels (what the model is trained to predict) ---
        label_ids = batch.get("packed_label_ids", None)
        if (
            label_ids is not None
            and torch.is_tensor(label_ids)
            and label_ids.numel() > 0
        ):
            label_decoded = tokenizer.decode(label_ids, skip_special_tokens=False)
            marker = "<|im_start|>"
            lines.append("--- labels (turn-wise, truncated) ---")
            lines.extend(
                self._format_turnwise_text(
                    label_decoded,
                    marker=marker,
                    head_chars=head_chars,
                    tail_chars=tail_chars,
                )
            )

        # --- check action token presence ---
        if text_ids is not None and torch.is_tensor(text_ids):
            vis_start_id = getattr(self.processing_class, "start_of_image", None)
            vis_end_id = getattr(self.processing_class, "end_of_image", None)
            if vis_start_id is not None and vis_end_id is not None:
                vis_start_count = (text_ids == vis_start_id).sum().item()
                vis_end_count = (text_ids == vis_end_id).sum().item()
                lines.append(
                    "vision boundary tokens in text_ids: "
                    f"start={vis_start_count}, end={vis_end_count}"
                )
                lines.append(
                    "note: payload between boundaries is not textual; "
                    "image tokens live in packed_vae_token_indexes/"
                    "packed_vit_token_indexes"
                )

            action_id = getattr(self.processing_class, "image_action_token_id", None)
            if action_id is not None:
                action_count = (text_ids == action_id).sum().item()
                lines.append(
                    f"action token (id={action_id}) in text_ids: {action_count}"
                )
                if action_count > 0:
                    lines.extend(
                        self._format_token_context_windows(
                            token_ids=text_ids,
                            tokenizer=tokenizer,
                            target_token_id=action_id,
                            context_tokens=24,
                            label="action token/text_ids",
                        )
                    )
                if label_ids is not None and torch.is_tensor(label_ids):
                    label_action_count = (label_ids == action_id).sum().item()
                    lines.append(f"action token in labels: {label_action_count}")
                    if label_action_count > 0:
                        lines.extend(
                            self._format_token_context_windows(
                                token_ids=label_ids,
                                tokenizer=tokenizer,
                                target_token_id=action_id,
                                context_tokens=24,
                                label="action token/labels",
                            )
                        )
                    ce_reweighting_on = bool(
                        getattr(self.model.config, "ce_loss_reweighting", False)
                    )
                    lines.append(f"ce_loss_reweighting enabled: {ce_reweighting_on}")
                    action_weight = getattr(
                        self.processing_class,
                        "action_token_ce_loss_weight",
                        None,
                    )
                    if action_weight is not None:
                        lines.append(
                            "configured action_token_ce_loss_weight: "
                            f"{float(action_weight):.6f}"
                        )
                    ce_w = batch.get("ce_loss_weights", None)
                    if ce_w is not None and torch.is_tensor(ce_w):
                        act_mask = label_ids == action_id
                        if act_mask.any():
                            action_avg_w = float(ce_w[act_mask].float().mean().item())
                            non_act_mask = ~act_mask
                            if non_act_mask.any():
                                base_avg_w = float(
                                    ce_w[non_act_mask].float().mean().item()
                                )
                            else:
                                base_avg_w = 0.0
                            ratio = (
                                action_avg_w / base_avg_w
                                if base_avg_w > 0
                                else float("inf")
                            )
                            lines.append(
                                "ce_loss_weights avg: "
                                f"action={action_avg_w:.6f}, "
                                f"non_action={base_avg_w:.6f}, ratio={ratio:.3f}x"
                            )

        lines.append("=" * 130)
        logger.info("\n".join(lines))
