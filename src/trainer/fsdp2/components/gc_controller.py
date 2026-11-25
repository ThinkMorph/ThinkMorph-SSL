from __future__ import annotations

import gc
from dataclasses import dataclass

import torch


@dataclass
class GCController:
    enabled: bool = False
    gc_freq: int = 200

    def maybe_collect(self, step: int) -> None:
        if not self.enabled:
            return
        if self.gc_freq <= 0:
            return
        if step <= 0 or step % self.gc_freq != 0:
            return
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
