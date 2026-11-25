import re
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from loguru import logger
from torch.distributed.checkpoint.state_dict import get_model_state_dict


class EMAHelper:
    """
    Lightweight Exponential Moving Average (EMA) helper designed to work with:
    - Vanilla nn.Module parameters (torch.Tensor)
    - FSDP2 / sharded parameters that expose a shard-local `_local_tensor`

    Key design goals:
    - Fully opt-in (no behavior change unless ema_enabled=True)
    - Shard-local updates (no all-gather / no distributed communication)
    - Generic for HF-style models (operates on model.named_parameters())
    """

    # Mapping for ema_master_dtype config value -> torch dtype. None means
    # "match the model param dtype" which preserves the original behavior.
    _MASTER_DTYPE_MAP = {
        "model": None,
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }

    def __init__(self, args) -> None:
        # Extract all config once - args won't change at runtime
        self._enabled = bool(getattr(args, "ema_enabled", False))
        self._decay = float(getattr(args, "ema_decay", 0.9999))
        self._update_every = int(getattr(args, "ema_update_every", 1))
        self._start_step = int(getattr(args, "ema_start_step", 0))
        self._requires_grad_only = bool(getattr(args, "ema_requires_grad_only", True))
        self._resume_from_ema = bool(getattr(args, "ema_resume_from_ema", False))

        master_dtype_str = str(
            getattr(args, "ema_master_dtype", "model") or "model"
        ).lower()
        if master_dtype_str not in self._MASTER_DTYPE_MAP:
            raise ValueError(
                f"ema_master_dtype must be one of "
                f"{list(self._MASTER_DTYPE_MAP)}; got {master_dtype_str!r}"
            )
        # None => keep model dtype (original behavior, byte-identical when EMA off)
        self._master_dtype: torch.dtype | None = self._MASTER_DTYPE_MAP[master_dtype_str]

        param_filter = getattr(args, "ema_param_filter", None) or {}
        self._param_filter_mode = str(
            param_filter.get("mode", "substring") or "substring"
        ).lower()
        self._param_include = param_filter.get("include")
        self._param_exclude = param_filter.get("exclude")

        # Disable if config is invalid
        if not (0.0 < self._decay < 1.0) or self._update_every <= 0:
            self._enabled = False

        self._initialized: bool = False
        self._warmup_done: bool = False
        self._ema_params: Dict[str, Any] = {}
        self._ema_param_pairs: List[Tuple[Any, nn.Parameter]] = []
        # Used only when _master_dtype != model dtype: holds bf16 model snapshot
        # during a swap_with_model() / swap_back pair so that the higher-precision
        # EMA shadow is not destroyed by a bf16 round-trip.
        self._swap_backup: List[torch.Tensor] | None = None

    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def initialized(self) -> bool:
        return self._initialized

    def maybe_init(self, model: nn.Module, checkpoint_dir: str | None) -> None:
        if self._initialized or not self._enabled:
            return

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        # Load EMA checkpoint from DCP if present
        ema_state_dict = None
        if checkpoint_dir is not None:
            try:
                temp_sd = {"ema_model": get_model_state_dict(model)}
                dcp.load(temp_sd, checkpoint_id=checkpoint_dir)
                ema_state_dict = temp_sd["ema_model"]
                logger.info(f"Loaded EMA state from DCP checkpoint: {checkpoint_dir}")
            except Exception as e:
                logger.warning(f"No EMA in DCP checkpoint ({e}). Init from model.")

        for name, param in model.named_parameters():
            if not self._should_track_param(name, param):
                continue

            if ema_state_dict and name in ema_state_dict:
                ema_val = ema_state_dict[name]
            else:
                ema_val = param.detach().clone()
            # Promote shadow to master dtype if it differs from the param dtype.
            # When _master_dtype is None this is a no-op, preserving the original
            # behavior for any caller that has not opted into fp32 EMA.
            if self._master_dtype is not None:
                local = self._local_tensor(ema_val)
                if local.dtype != self._master_dtype:
                    ema_val = ema_val.to(self._master_dtype)
            self._ema_params[name] = ema_val
            self._ema_param_pairs.append((ema_val, param))

        self._initialized = True
        # If EMA was restored from checkpoint, warmup already happened in the
        # previous run — skip the re-init on the first update() call.
        if ema_state_dict is not None:
            self._warmup_done = True

        if self._resume_from_ema and ema_state_dict is not None:
            self.copy_to_model()
            if rank == 0:
                logger.info(
                    "Applied EMA weights to live model (ema_resume_from_ema=True)."
                )

    def update(self, step: int) -> None:
        if not self._enabled or not self._initialized:
            return
        if step < self._start_step:
            return

        # At start_step, re-initialize EMA from current model weights so that
        # newly added tokens (e.g. action tokens) that were randomly initialized
        # at training start have had time to converge before EMA begins tracking.
        if not self._warmup_done:
            for ema_val, param in self._ema_param_pairs:
                self._local_tensor(ema_val).copy_(self._local_tensor(param))
            self._warmup_done = True
            return

        if self._update_every > 1 and (step % self._update_every != 0):
            return

        one_minus_decay = 1.0 - self._decay
        for ema_val, param in self._ema_param_pairs:
            ema_local = self._local_tensor(ema_val)
            model_local = self._local_tensor(param)
            # Cast model param to shadow dtype before the in-place add so that
            # tiny (1-decay) updates aren't rounded away by bf16. When dtypes
            # already match (default "model" mode) this branch is skipped and
            # behavior is identical to the original implementation.
            if model_local.dtype != ema_local.dtype:
                model_local = model_local.to(ema_local.dtype)
            ema_local.mul_(self._decay).add_(model_local, alpha=one_minus_decay)

    def copy_to_model(self) -> None:
        if not self._initialized:
            return
        for ema_val, param in self._ema_param_pairs:
            ema_local = self._local_tensor(ema_val)
            model_local = self._local_tensor(param)
            model_local.copy_(ema_local)

    def swap_with_model(self) -> None:
        """Swap EMA weights and live model weights in-place.

        After calling this, the model holds EMA weights and the EMA buffer holds
        the original training weights.  Call again to restore.

        When ema_master_dtype matches the model dtype (default), this is a true
        in-place swap. When the shadow is in a higher-precision dtype (e.g.
        fp32 with bf16 model), a true swap would silently downcast the shadow
        to bf16 and back, destroying its higher precision; in that case we
        instead snapshot the model into a per-call backup buffer, copy EMA into
        the model, and restore from the backup on the second call so the EMA
        shadow is preserved untouched.
        """
        if not self._initialized:
            return

        if self._master_dtype is None:
            for ema_val, param in self._ema_param_pairs:
                ema_local = self._local_tensor(ema_val)
                model_local = self._local_tensor(param)
                tmp = model_local.clone()
                model_local.copy_(ema_local)
                ema_local.copy_(tmp)
            return

        # Higher-precision shadow path: cache model, swap EMA in. Second call
        # restores the cached model and leaves the shadow untouched.
        if self._swap_backup is None:
            self._swap_backup = []
            for ema_val, param in self._ema_param_pairs:
                ema_local = self._local_tensor(ema_val)
                model_local = self._local_tensor(param)
                self._swap_backup.append(model_local.clone())
                model_local.copy_(ema_local)
        else:
            for (_, param), saved in zip(
                self._ema_param_pairs, self._swap_backup, strict=True
            ):
                model_local = self._local_tensor(param)
                model_local.copy_(saved)
            self._swap_backup = None

    def state_dict_for_save(self, model: nn.Module) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError(
                "EMA is not initialized; cannot build EMA state_dict for save."
            )

        sd = get_model_state_dict(model)
        for name, ema_val in self._ema_params.items():
            if name in sd:
                sd[name] = ema_val
        return sd

    @staticmethod
    def _local_tensor(t: Any) -> torch.Tensor:
        if isinstance(t, nn.Parameter):
            t = t.data
        # FSDP2 shards expose _local_tensor for shard access without all-gather
        if hasattr(t, "_local_tensor"):
            return t._local_tensor  # noqa: SLF001
        return t

    def _name_match(self, name: str, patterns: List[str]) -> bool:
        if self._param_filter_mode == "regex":
            return any(re.search(p, name) is not None for p in patterns)
        return any(p in name for p in patterns)

    def _should_track_param(self, name: str, param: nn.Parameter) -> bool:
        if self._requires_grad_only and not param.requires_grad:
            return False

        if not torch.is_floating_point(self._local_tensor(param)):
            return False

        if self._param_include and not self._name_match(name, self._param_include):
            return False
        if self._param_exclude and self._name_match(name, self._param_exclude):
            return False

        return True
