import threading
from typing import List, Union

from loguru import logger

import wandb


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = [
        "wandb",
    ]

    def __init__(
        self,
        project_name,
        experiment_name,
        default_backend: Union[str, List[str]] = "console",
        config=None,
    ):
        if isinstance(default_backend, str):
            default_backend = [default_backend]

        self.logger = {}
        self._closed = False

        if "wandb" in default_backend:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
            )
            self.logger["wandb"] = wandb
            wandb.define_metric("eval/*", step_metric="global_step")

    def log(self, data, step=None):
        if "wandb" in self.logger:
            wandb_data = {}
            for key, value in data.items():
                if isinstance(value, str):
                    continue
                else:
                    wandb_data[key] = value
            self.logger["wandb"].log(wandb_data, step=step)

        # Console: do NOT dump the full dict (becomes huge with histograms +
        # data/* fields). The LogDisplay in the trainer handles per-step
        # progress; here we keep only non-train/* or eval scalars for
        # unexpected callers. Per-step train metrics pass silently.
        log_data = dict(data)
        if step is None and "global_step" in log_data:
            step = log_data.pop("global_step")

        # Filter out bulky types and per-step train metrics for console output.
        compact = {}
        for k, v in log_data.items():
            # Drop wandb.Histogram (no readable repr) and internal large payloads.
            if type(v).__name__ == "Histogram":
                continue
            if k.startswith("train/") or k.startswith("data/"):
                continue
            compact[k] = v
        if compact:
            logger.info("{}, step={}", compact, step)

    def close(self, timeout: float = 60.0) -> None:
        """Finish the wandb run with a hard timeout to avoid hanging at shutdown.

        Wandb's default finish waits for the cloud uploader indefinitely if the
        network is slow. We run finish() in a thread and give up after `timeout`
        seconds, letting the caller proceed. Any un-synced data stays under
        wandb/run-*/ and can be resynced with `wandb sync` later.
        """
        if self._closed:
            return
        self._closed = True
        if "wandb" not in self.logger:
            return

        done = threading.Event()

        def _finish():
            try:
                self.logger["wandb"].finish(exit_code=0, quiet=True)
            except Exception as e:
                logger.warning(f"wandb.finish raised: {e}")
            finally:
                done.set()

        t = threading.Thread(target=_finish, daemon=True, name="wandb-finish")
        t.start()
        if not done.wait(timeout=timeout):
            logger.warning(
                f"wandb.finish did not complete within {timeout}s; "
                f"continuing shutdown. Resync later with 'wandb sync'."
            )

    def __del__(self):
        # Best-effort close; the trainer should call .close() explicitly.
        try:
            self.close(timeout=10.0)
        except Exception:
            pass
