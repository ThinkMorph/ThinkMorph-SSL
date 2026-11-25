from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset

from trainer.config import TrainingArguments
from trainer.fsdp2.bagel_probe_mixin import BagelProbeMixin
from trainer.fsdp2.bench_eval_mixin import BenchEvalMixin
from trainer.fsdp2.components import InferenceProbeSchedule
from trainer.fsdp2.fsdp2_trainer import FSDP2SFTTrainer
from trainer.fsdp2.val_infer_mixin import ValInferMixin
from trainer.registry import TRAINER_REGISTER

DatasetType = Union[Dataset, IterableDataset]


def _as_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if torch.is_tensor(x):
        if x.numel() == 0:
            return None
        return float(x.detach().float().mean().item())
    try:
        return float(x)
    except Exception:
        return None


@TRAINER_REGISTER.register("bagel_fsdp2_trainer")
class BagelFSDP2Trainer(
    ValInferMixin, BenchEvalMixin, BagelProbeMixin, FSDP2SFTTrainer
):
    """
    Bagel-specific FSDP2 SFT trainer that logs Bagel loss components (CE/MSE) to W&B/console via Tracking.

    It intentionally reuses the base FSDP2SFTTrainer training loop behavior (optimizer, scheduler, EMA),
    but extracts Bagel's `loss_dict` / token counts from model outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: DatasetType,
        eval_dataset: DatasetType = None,
        processing_class=None,
        data_collator=None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            data_collator=data_collator,
            **kwargs,
        )
        self.inference_probe_enabled = False
        self.inference_probe_interval_steps = 0
        self.inference_probe_start_step = 0
        self.inference_probe_max_samples = 0
        self.inference_probe_default_kwargs = {}
        self.inference_probe_samples = []
        self.inference_probe_use_rank0_unsharded = True
        self._probe_inferencer = None
        self.probe_schedule = InferenceProbeSchedule(
            enabled=False,
            interval_steps=0,
            start_step=0,
        )
        self.enable_inference_probes = True
        self.inference_probe_config = self.args.inference_probe_config or {}
        self._setup_inference_probes()
        self._setup_bench_eval()
        self._setup_val_infer()

    def compute_loss_dict(self, outputs):
        if not isinstance(outputs, dict):
            return super().compute_loss_dict(outputs)

        loss = outputs["loss"]
        loss_dict = outputs.get("loss_dict", {}) or {}
        return {
            "loss": loss,
            "ce": loss_dict.get("ce", None),
            "mse": loss_dict.get("mse", None),
            "ce_weight": outputs.get("ce_weight", None),
            "mse_weight": outputs.get("mse_weight", None),
            "total_ce_tokens": outputs.get("total_ce_tokens", None),
            "total_mse_tokens": outputs.get("total_mse_tokens", None),
        }

    def reduce_step_metrics(self, loss_dict, grad_norm=None):
        metrics = super().reduce_step_metrics(loss_dict, grad_norm=grad_norm)
        device = self.args.device

        ce_val = _as_float(loss_dict.get("ce", None))
        mse_val = _as_float(loss_dict.get("mse", None))

        if ce_val is not None:
            ce_tensor = torch.tensor(ce_val, device=device)
            if self._world_size() > 1:
                dist.all_reduce(ce_tensor, op=dist.ReduceOp.AVG)
            metrics["train/ce"] = ce_tensor.item()

        if mse_val is not None:
            mse_tensor = torch.tensor(mse_val, device=device)
            if self._world_size() > 1:
                dist.all_reduce(mse_tensor, op=dist.ReduceOp.AVG)
            metrics["train/mse"] = mse_tensor.item()

        ce_w = _as_float(loss_dict.get("ce_weight", None))
        mse_w = _as_float(loss_dict.get("mse_weight", None))
        if ce_w is None:
            ce_w = _as_float(getattr(self.model.config, "ce_weight", 1.0))
        if mse_w is None:
            mse_w = _as_float(getattr(self.model.config, "mse_weight", 1.0))
        if ce_w is not None:
            metrics["train/ce_weight"] = ce_w
        if mse_w is not None:
            metrics["train/mse_weight"] = mse_w
        if ce_val is not None and ce_w is not None:
            metrics["train/ce_weighted"] = ce_val * ce_w
        if mse_val is not None and mse_w is not None:
            metrics["train/mse_weighted"] = mse_val * mse_w

        ce_tokens = loss_dict.get("total_ce_tokens", None)
        mse_tokens = loss_dict.get("total_mse_tokens", None)
        if torch.is_tensor(ce_tokens):
            metrics["train/ce_tokens"] = ce_tokens.detach().float().item()
        if torch.is_tensor(mse_tokens):
            metrics["train/mse_tokens"] = mse_tokens.detach().float().item()

        return metrics

    # --- Val eval extra metrics (CE/MSE) ---

    def _val_eval_init_extra_accumulators(self):
        return {
            "ce": torch.tensor(0.0, device=self.args.device),
            "mse": torch.tensor(0.0, device=self.args.device),
            "ce_count": 0,
            "mse_count": 0,
        }

    def _val_eval_accumulate_extra(self, accum, loss_dict):
        ce_val = _as_float(loss_dict.get("ce", None))
        mse_val = _as_float(loss_dict.get("mse", None))
        if ce_val is not None:
            accum["ce"] += ce_val
            accum["ce_count"] += 1
        if mse_val is not None:
            accum["mse"] += mse_val
            accum["mse_count"] += 1

    def _val_eval_reduce_extra(self, accum, total_batches):
        metrics = {}
        if accum["ce_count"] > 0:
            avg_ce = accum["ce"] / accum["ce_count"]
            if self._world_size() > 1:
                dist.all_reduce(avg_ce, op=dist.ReduceOp.AVG)
            metrics["val/ce"] = avg_ce.item()
        if accum["mse_count"] > 0:
            avg_mse = accum["mse"] / accum["mse_count"]
            if self._world_size() > 1:
                dist.all_reduce(avg_mse, op=dist.ReduceOp.AVG)
            metrics["val/mse"] = avg_mse.item()
        return metrics
