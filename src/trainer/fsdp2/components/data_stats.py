"""Per-sample data statistics tracker for wandb logging.

Extracts image count/size info from each batch (populated by bagel_processor),
accumulates running totals across ranks, and emits wandb-friendly metrics
(scalars + histograms) at logging time.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore


def _unwrap_nested_int_list(value: Any) -> List[int]:
    """Collator wraps per-sample ints as [[a, b, c]]. Flatten to [a, b, c]."""
    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return list(value[0])
    if isinstance(value, list):
        return list(value)
    return []


class DataStatsTracker:
    """Accumulates per-rank data stats and all-reduces cross-rank totals on emit.

    Scalars (cumulative totals) are all-reduced. Per-batch means/extremes and
    histograms use the local rank-0 window - statistically representative when
    the iterable dataset stride-shards samples across ranks.
    """

    SCALAR_TOTAL_KEYS = (
        "total_samples",
        "total_user_images",
        "total_gen_images",
    )

    def __init__(self):
        self.totals: Dict[str, int] = {k: 0 for k in self.SCALAR_TOTAL_KEYS}
        # Per-window buffers (cleared after each emit).
        self.window: Dict[str, List[int]] = {
            "batch_samples": [],
            "batch_user_images": [],
            "batch_gen_images": [],
            "user_image_h": [],
            "user_image_w": [],
            "gen_image_h": [],
            "gen_image_w": [],
        }

    def pop_and_update(self, batch: Dict[str, Any]) -> None:
        """Pop data stats keys from batch (so model forward never sees them)
        and update local counters."""
        num_user = _unwrap_nested_int_list(batch.pop("num_user_images", None))
        num_gen = _unwrap_nested_int_list(batch.pop("num_gen_images", None))
        user_h = batch.pop("user_image_h", []) or []
        user_w = batch.pop("user_image_w", []) or []
        gen_h = batch.pop("gen_image_h", []) or []
        gen_w = batch.pop("gen_image_w", []) or []

        num_samples = len(num_user) if num_user else len(num_gen)
        self.totals["total_samples"] += num_samples
        self.totals["total_user_images"] += sum(num_user)
        self.totals["total_gen_images"] += sum(num_gen)

        self.window["batch_samples"].append(num_samples)
        self.window["batch_user_images"].append(sum(num_user))
        self.window["batch_gen_images"].append(sum(num_gen))
        self.window["user_image_h"].extend(int(v) for v in user_h)
        self.window["user_image_w"].extend(int(v) for v in user_w)
        self.window["gen_image_h"].extend(int(v) for v in gen_h)
        self.window["gen_image_w"].extend(int(v) for v in gen_w)

    def _all_reduced_totals(self, device: torch.device) -> Dict[str, int]:
        """Sum scalar totals across ranks. No-op when world_size == 1."""
        values = [self.totals[k] for k in self.SCALAR_TOTAL_KEYS]
        if dist.is_initialized() and dist.get_world_size() > 1:
            tensor = torch.tensor(values, dtype=torch.long, device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            values = tensor.tolist()
        return dict(zip(self.SCALAR_TOTAL_KEYS, values))

    def emit(
        self,
        device: torch.device,
        is_rank0: bool,
        include_histograms: bool = True,
    ) -> Dict[str, Any]:
        """Return dict of metrics ready to feed to tracking.log.

        Non-rank-0 callers receive an empty dict after participating in the
        collective all_reduce.
        """
        totals = self._all_reduced_totals(device)
        if not is_rank0:
            self._reset_window()
            return {}

        metrics: Dict[str, Any] = {
            "data/total_samples_seen": int(totals["total_samples"]),
            "data/total_user_images_seen": int(totals["total_user_images"]),
            "data/total_gen_images_seen": int(totals["total_gen_images"]),
        }

        def _scalar_stats(name: str, values: List[int]) -> None:
            if not values:
                return
            metrics[f"data/{name}_mean"] = statistics.fmean(values)
            metrics[f"data/{name}_max"] = max(values)
            metrics[f"data/{name}_min"] = min(values)

        for key in (
            "batch_samples",
            "batch_user_images",
            "batch_gen_images",
            "user_image_h",
            "user_image_w",
            "gen_image_h",
            "gen_image_w",
        ):
            _scalar_stats(key, self.window[key])

        user_pixels = [
            h * w
            for h, w in zip(self.window["user_image_h"], self.window["user_image_w"])
        ]
        gen_pixels = [
            h * w
            for h, w in zip(self.window["gen_image_h"], self.window["gen_image_w"])
        ]
        _scalar_stats("user_image_pixels", user_pixels)
        _scalar_stats("gen_image_pixels", gen_pixels)

        if include_histograms and wandb is not None:
            if self.window["user_image_h"]:
                metrics["data/user_image_h_hist"] = wandb.Histogram(
                    self.window["user_image_h"]
                )
                metrics["data/user_image_w_hist"] = wandb.Histogram(
                    self.window["user_image_w"]
                )
                metrics["data/user_image_pixels_hist"] = wandb.Histogram(user_pixels)
            if self.window["gen_image_h"]:
                metrics["data/gen_image_h_hist"] = wandb.Histogram(
                    self.window["gen_image_h"]
                )
                metrics["data/gen_image_w_hist"] = wandb.Histogram(
                    self.window["gen_image_w"]
                )
                metrics["data/gen_image_pixels_hist"] = wandb.Histogram(gen_pixels)
            if self.window["batch_samples"]:
                metrics["data/batch_samples_hist"] = wandb.Histogram(
                    self.window["batch_samples"]
                )
                metrics["data/batch_user_images_hist"] = wandb.Histogram(
                    self.window["batch_user_images"]
                )
                metrics["data/batch_gen_images_hist"] = wandb.Histogram(
                    self.window["batch_gen_images"]
                )

        self._reset_window()
        return metrics

    def _reset_window(self) -> None:
        for key in self.window:
            self.window[key].clear()

    # ---- checkpoint integration -------------------------------------------------

    def state_dict(self) -> Dict[str, int]:
        return dict(self.totals)

    def load_state_dict(self, state: Optional[Dict[str, int]]) -> None:
        if not state:
            return
        for k in self.SCALAR_TOTAL_KEYS:
            if k in state:
                self.totals[k] = int(state[k])
