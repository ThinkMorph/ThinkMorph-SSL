from __future__ import annotations

import os
import shutil
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from loguru import logger
from utils.distributed import dist_barrier


@dataclass
class CheckpointPolicy:
    keep_latest_k: int = 0
    async_mode: str = "disabled"


class CheckpointManager:
    def __init__(
        self,
        *,
        output_dir: str,
        policy: Optional[CheckpointPolicy] = None,
    ) -> None:
        self.output_dir = output_dir
        self.policy = policy or CheckpointPolicy()
        self._pending_save: Optional[Future] = None

    def _is_async_enabled(self) -> bool:
        mode = str(self.policy.async_mode or "disabled").lower()
        return mode in {"async", "enabled", "true", "1"}

    @staticmethod
    def _is_missing_cpu_backend_async_save_error(error: AssertionError) -> bool:
        return "A CPU backend must be enabled for async save" in str(error)

    def _wait_for_pending_save(self) -> None:
        if self._pending_save is None:
            return
        self._pending_save.result()
        self._pending_save = None
        self.purge_old_checkpoints()

    def finalize_pending_save(self) -> None:
        self._wait_for_pending_save()

    def save(self, state_dict: Dict[str, Any], checkpoint_path: str) -> None:
        self._wait_for_pending_save()
        os.makedirs(checkpoint_path, exist_ok=True)
        if dist.is_available() and dist.is_initialized():
            dist_barrier()
        if self._is_async_enabled() and hasattr(dcp, "async_save"):
            try:
                self._pending_save = dcp.async_save(
                    state_dict, checkpoint_id=checkpoint_path
                )
                return
            except AssertionError as error:
                if not self._is_missing_cpu_backend_async_save_error(error):
                    raise
                logger.warning(
                    "Async checkpoint save is unavailable because the default process "
                    "group has no CPU backend; falling back to synchronous save. "
                    "To re-enable async save, initialize the process group with "
                    "'cpu:gloo,cuda:nccl'."
                )

        dcp.save(state_dict, checkpoint_id=checkpoint_path)
        if dist.is_available() and dist.is_initialized():
            dist_barrier()
        self.purge_old_checkpoints()

    def load(self, state_dict: Dict[str, Any], checkpoint_path: str) -> None:
        self._wait_for_pending_save()
        dcp.load(state_dict, checkpoint_id=checkpoint_path)

    def latest_checkpoint(self) -> Optional[str]:
        if not os.path.isdir(self.output_dir):
            return None
        checkpoints = [
            f
            for f in os.listdir(self.output_dir)
            if f.startswith("checkpoint-") and f.split("-")[-1].isdigit()
        ]
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        return os.path.join(self.output_dir, checkpoints[-1])

    def purge_old_checkpoints(self) -> None:
        keep_latest_k = int(self.policy.keep_latest_k or 0)
        if keep_latest_k <= 0:
            return
        if not os.path.isdir(self.output_dir):
            return
        checkpoints = [
            f
            for f in os.listdir(self.output_dir)
            if f.startswith("checkpoint-") and f.split("-")[-1].isdigit()
        ]
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        stale = checkpoints[:-keep_latest_k]
        for ckpt in stale:
            shutil.rmtree(os.path.join(self.output_dir, ckpt), ignore_errors=True)
