"""Rich-based training progress display.

Emits a compact per-step `logger.info` line every update (captured cleanly
by loguru sinks / tee to the log file), and - when a controlling terminal
is available - also drives a live rich.Progress bar on it.

Terminal resolution order (first writable wins):
  1. /dev/tty  - works even when stdout/stderr is piped through `tee`,
                 so the log file stays ANSI-free while the terminal still
                 gets a live bar.
  2. sys.stdout / sys.stderr if isatty().

Overrides:
  DISABLE_RICH_DISPLAY=1 - skip the bar entirely.
  FORCE_RICH_DISPLAY=1   - route the bar to sys.stdout even if not a TTY.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Dict, Optional

from loguru import logger


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _open_tty_stream():
    """Open a writable stream for the rich bar, or return None.

    Prefers /dev/tty so `torchrun ... | tee log` still gets a live bar on
    the terminal while the tee'd file stays free of ANSI control chars.
    """
    if _env_truthy("DISABLE_RICH_DISPLAY"):
        return None
    try:
        return open("/dev/tty", "w", buffering=1)
    except (OSError, FileNotFoundError):
        pass
    if _env_truthy("FORCE_RICH_DISPLAY"):
        return sys.stdout
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.isatty():
                return stream
        except Exception:
            continue
    return None


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None:
        return "-"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _fmt_int(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _first_metric_value(metrics: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return None


class LogDisplay:
    """Minimal progress-bar + metric display. Thread-safe for single updater."""

    def __init__(
        self,
        total_steps: int,
        enabled: Optional[bool] = None,
        initial_step: int = 0,
    ):
        self.total_steps = int(total_steps)
        # Prefer the Console already wired into loguru's RichHandler. Sharing
        # state is what lets rich.Live coordinate: log lines scroll ABOVE
        # the bar instead of each refresh drawing a fresh line.
        shared_console = None
        try:
            from utils.logging_utils import get_shared_console

            shared_console = get_shared_console()
        except Exception:
            shared_console = None

        self._shared_console = (
            shared_console
            if (shared_console is not None and shared_console.is_terminal)
            else None
        )

        if enabled is False:
            # Force-disable: ignore the shared console too so that use_rich
            # stays False even when setup_distributed_logging() has been called.
            self._shared_console = None
            self._tty_stream = None
        elif self._shared_console is not None:
            self._tty_stream = None  # Console already has its own file
        else:
            self._tty_stream = _open_tty_stream()

        self.use_rich = self._shared_console is not None or self._tty_stream is not None
        self._owns_tty = self._tty_stream not in (None, sys.stdout, sys.stderr)
        self._progress = None
        self._task_id = None
        self._lock = threading.Lock()
        self._last_step_logged = initial_step
        if self.use_rich:
            try:
                from rich.console import Console
                from rich.progress import (
                    BarColumn,
                    MofNCompleteColumn,
                    Progress,
                    TaskProgressColumn,
                    TextColumn,
                    TimeElapsedColumn,
                    TimeRemainingColumn,
                )

                if self._shared_console is not None:
                    console = self._shared_console
                else:
                    console = Console(file=self._tty_stream, force_terminal=True)
                self._progress = Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=30),
                    MofNCompleteColumn(),
                    TaskProgressColumn(),
                    TextColumn("|"),
                    TimeElapsedColumn(),
                    TextColumn("ETA"),
                    TimeRemainingColumn(),
                    TextColumn("|"),
                    TextColumn("{task.fields[metrics]}"),
                    refresh_per_second=4,
                    transient=False,
                    console=console,
                )
                self._progress.start()
                self._task_id = self._progress.add_task(
                    "training",
                    total=self.total_steps,
                    completed=initial_step,
                    metrics="",
                )
                if self._shared_console is not None:
                    _stream_name = "shared loguru Console (stderr)"
                else:
                    _stream_name = getattr(
                        self._tty_stream, "name", str(self._tty_stream)
                    )
                logger.info(f"LogDisplay: rich bar enabled on {_stream_name}")
            except Exception as e:
                logger.warning(f"Rich display unavailable, falling back: {e}")
                self._close_tty()
                self._progress = None
                self._task_id = None
                self.use_rich = False
        else:
            logger.info(
                "LogDisplay: rich bar disabled (no writable TTY / /dev/tty); "
                "per-step logger.info only. Set FORCE_RICH_DISPLAY=1 to force."
            )

    def update(self, step: int, metrics: Dict[str, Any]) -> None:
        """Advance the bar to `step` and refresh the metric fields."""
        loss = _fmt(metrics.get("train/loss"), ".4f")
        # BAGEL subclass emits train/ce and train/mse (component losses, pre-weight).
        ce_loss = _fmt(_first_metric_value(metrics, "train/ce", "train/ce_loss"), ".4f")
        mse_loss = _fmt(
            _first_metric_value(metrics, "train/mse", "train/mse_loss"), ".4f"
        )
        grad = _fmt(metrics.get("train/grad_norm"), ".3f")
        lr = _fmt(metrics.get("train/lr"), ".2e")
        samples = _fmt_int(metrics.get("data/total_samples_seen"))
        tokens = _fmt_int(
            _first_metric_value(metrics, "total_tokens", "train/total_tokens")
        )
        mfu = metrics.get("train/mfu")
        mfu_s = f" mfu={_fmt(mfu, '.2f')}" if mfu is not None else ""

        metric_line = (
            f"loss={loss}"
            + (f" ce={ce_loss}" if ce_loss != "-" else "")
            + (f" mse={mse_loss}" if mse_loss != "-" else "")
            + f" grad={grad} lr={lr} samples={samples} tokens={tokens}"
            + mfu_s
        )

        with self._lock:
            # Always emit the compact one-liner via loguru so file sinks
            # (tee, loguru file handlers) capture per-step metrics cleanly,
            # even when rich is driving a live bar on /dev/tty.
            logger.info(f"[step {step}/{self.total_steps}] {metric_line}")
            if self._progress is not None and self._task_id is not None:
                self._progress.update(
                    self._task_id, completed=step, metrics=metric_line
                )
            self._last_step_logged = step

    def print(self, message: str) -> None:
        """Print a free-form message without disrupting the bar (tty only).
        In non-tty mode, forwards to loguru.info."""
        with self._lock:
            if self._progress is not None:
                try:
                    self._progress.console.log(message)
                    return
                except Exception:
                    pass
            logger.info(message)

    def _close_tty(self) -> None:
        if self._owns_tty and self._tty_stream is not None:
            try:
                self._tty_stream.close()
            except Exception:
                pass
        self._tty_stream = None
        self._owns_tty = False

    def close(self) -> None:
        with self._lock:
            if self._progress is not None:
                try:
                    self._progress.stop()
                except Exception:
                    pass
                self._progress = None
                self._task_id = None
            self._close_tty()
