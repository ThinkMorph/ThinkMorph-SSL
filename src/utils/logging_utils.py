import os
from typing import Any, Dict, Optional

import torch.distributed as dist
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

# Shared Console instance so log_display's rich.Progress can coordinate
# cursor state with loguru's RichHandler (same stream, same Live group) -
# otherwise Progress redraws on new lines instead of in place.
_SHARED_CONSOLE: Optional[Console] = None


def get_shared_console() -> Optional[Console]:
    """Return the Console shared with loguru's RichHandler, or None if
    `setup_distributed_logging` has not been called yet."""
    return _SHARED_CONSOLE


def distributed_filter(record: Dict[str, Any]) -> bool:
    """
    Filter function for distributed training.
    Only allows logs from rank 0 when distributed training is initialized.
    """
    if os.environ.get("LOG_ALL_RANKS", "").lower() in ("1", "true", "yes"):
        return True
    # Before dist is initialized, torchrun already provides RANK in env.
    # Use it to avoid duplicated startup logs from non-zero ranks.
    rank = os.environ.get("RANK")
    if rank is not None:
        try:
            return int(rank) == 0
        except ValueError:
            pass
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def setup_distributed_logging():
    """
    Setup loguru logger with distributed training filter.
    Call this function once at the beginning of your program.

    Sinks:
    - RichHandler on stderr: pretty console output. When stderr is a TTY,
      log_display's rich progress bar coexists cleanly above it.
    - File sink (optional): if env TRAIN_LOG_FILE is set, also write a
      plain (no-ANSI) copy of every log line to that file - replaces the
      shell-level `| tee` so the log file stays ANSI-free while stderr is
      free for rich to own the terminal.
    """
    global _SHARED_CONSOLE
    logger.remove()

    # A single Console drives both loguru's RichHandler and log_display's
    # Progress bar. Shared state lets rich.Live coordinate: log lines scroll
    # cleanly ABOVE the persistent bar instead of punching through it.
    _SHARED_CONSOLE = Console(stderr=True)
    logger.add(
        RichHandler(
            console=_SHARED_CONSOLE,
            rich_tracebacks=True,
            show_path=True,
            omit_repeated_times=False,
        ),
        format="{message}",
        filter=distributed_filter,
        level="DEBUG",
    )

    train_log_file = os.environ.get("TRAIN_LOG_FILE")
    if train_log_file:
        os.makedirs(os.path.dirname(os.path.abspath(train_log_file)), exist_ok=True)
        logger.add(
            train_log_file,
            format=(
                "[{time:YYYY-MM-DD HH:mm:ss}] {level: <8} {name}:{function}:{line} "
                "| {message}"
            ),
            filter=distributed_filter,
            level="DEBUG",
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )
