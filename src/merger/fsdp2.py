"""FSDP2 checkpoint merger implementation using PyTorch Distributed Checkpoint (DCP)."""

import os
import shutil
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from accelerate import init_empty_weights
from loguru import logger
from transformers import AutoConfig, AutoProcessor

from merger.base import CheckpointMerger
from models.registry import create_model_from_pretrained

CheckpointType = Literal["regular", "ema"]

# DCP state dict keys for each checkpoint type
STATE_DICT_KEY_MAP: dict[CheckpointType, str] = {
    "regular": "model",
    "ema": "ema_model",
}


class FSDP2Merger(CheckpointMerger):
    """Merger for FSDP2 DCP checkpoints.

    This class handles loading DCP (Distributed Checkpoint Protocol) checkpoints
    and exporting them as consolidated HuggingFace-compatible checkpoints for
    evaluation or inference.

    Args:
        checkpoint_type: Type of checkpoint to merge - "regular" for the main
                        model weights, "ema" for exponential moving average weights

    Example:
        >>> from pathlib import Path
        >>> from merger import FSDP2Merger
        >>> merger = FSDP2Merger(checkpoint_type="regular")
        >>> merger.merge(Path("checkpoint-1000"))
    """

    def __init__(self, checkpoint_type: CheckpointType = "regular") -> None:
        self.checkpoint_type = checkpoint_type
        self._state_dict_key = STATE_DICT_KEY_MAP[checkpoint_type]

    def _resolve_checkpoint_path(self, path: Path) -> Path:
        """Resolve checkpoint path, handling parent directories with multiple checkpoints.

        If path is a parent directory containing checkpoint-* subdirectories,
        returns the latest checkpoint. Otherwise returns the path as-is.

        Args:
            path: Input path (may be checkpoint directory or parent directory)

        Returns:
            Resolved checkpoint directory path

        Raises:
            ValueError: If no checkpoints found
        """
        # Check if path contains DCP metadata (indicating it's a DCP checkpoint)
        if (path / ".metadata").exists():
            return path

        # Check if path contains checkpoint subdirectories
        checkpoint_folders = list(path.glob("checkpoint-*"))
        if not checkpoint_folders:
            raise ValueError(
                f"No DCP checkpoint (.metadata) or checkpoint-* subdirectories found in {path}"
            )

        # Sort by checkpoint number and use the latest
        checkpoint_folders.sort(key=lambda x: int(x.name.split("-")[-1]))
        latest_checkpoint = checkpoint_folders[-1]
        return latest_checkpoint

    def _load_full_state_dict(
        self, checkpoint_path: Path, model_state_dict: dict[str, torch.Tensor]
    ) -> dict:
        """Load a full (unsharded) state dict from a DCP checkpoint.

        DCP requires pre-populated tensor templates to know what to load.
        The model_state_dict provides these templates (shape, dtype).

        This works in both distributed and non-distributed contexts.
        In non-distributed context, it initializes a temporary single-process group.

        Args:
            checkpoint_path: Path to DCP checkpoint directory
            model_state_dict: Model state dict to use as template for DCP loading.
                             Tensors will be overwritten in-place with checkpoint values.

        Returns:
            Full consolidated state dict with loaded weights
        """
        needs_pg = not (dist.is_available() and dist.is_initialized())
        if needs_pg:
            # Initialize a single-process group for standalone usage
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

        try:
            state_dict = {self._state_dict_key: model_state_dict}
            dcp.load(
                state_dict,
                checkpoint_id=str(checkpoint_path),
                no_dist=needs_pg,
            )
            return state_dict[self._state_dict_key]
        finally:
            if needs_pg:
                dist.destroy_process_group()

    def _maybe_tie_weights(self, model: torch.nn.Module, config: object) -> None:
        """Re-tie weights if the model declares weight tying.

        FSDP saves tied parameters (e.g. ``lm_head`` <-> ``embed_tokens``) as
        independent shards, so after ``load_state_dict(..., assign=True)`` they
        become separate tensors and ``save_pretrained`` would write both.
        """
        tied_keys = getattr(model, "_tied_weights_keys", None)
        tie_word_embeddings = (
            getattr(config, "tie_word_embeddings", False)
            or getattr(
                getattr(config, "text_config", None), "tie_word_embeddings", False
            )
            or getattr(
                getattr(config, "llm_config", None), "tie_word_embeddings", False
            )
        )
        if not (tied_keys and tie_word_embeddings):
            return

        # Check actual embedding weights (works for both list and dict _tied_weights_keys)
        in_emb = model.get_input_embeddings()
        out_emb = model.get_output_embeddings()
        if (
            in_emb is not None
            and out_emb is not None
            and in_emb.weight is not None
            and out_emb.weight is not None
            and not torch.equal(in_emb.weight, out_emb.weight)
        ):
            logger.warning(
                "Input and output embeddings differ after load_state_dict; "
                "skipping tie_weights() to preserve checkpoint values."
            )
            return

        logger.info("Re-tying weights (tie_word_embeddings=True).")
        model.tie_weights()

    def merge(
        self,
        checkpoint_path: Path,
        output_path: Path | None = None,
        model_cls: type | None = None,
        config: object | None = None,
    ) -> Path:
        """Merge DCP checkpoint into a single consolidated HuggingFace checkpoint.

        Args:
            checkpoint_path: Path to DCP checkpoint directory or parent directory
                           containing checkpoint-* subdirectories
            output_path: Where to save merged checkpoint. If None, saves to checkpoint_path directly
            model_cls: Model class to instantiate. If None, infers from checkpoint_path
            config: Model config. If None, loads from checkpoint_path

        Returns:
            Path to the merged checkpoint directory

        Raises:
            ValueError: If no DCP checkpoint is found
        """
        # Resolve checkpoint path (handles parent directories with checkpoint-* subdirs)
        original_checkpoint_path = checkpoint_path
        checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)

        if output_path is None:
            output_path = original_checkpoint_path

        logger.info(
            f"Merging DCP checkpoint: {checkpoint_path} "
            f"(type={self.checkpoint_type}, key={self._state_dict_key})"
        )

        # Infer model class and config if not provided
        if model_cls is None:
            model_cls = create_model_from_pretrained(checkpoint_path)
        if config is None:
            config = AutoConfig.from_pretrained(checkpoint_path)

        # Create model on meta device to get state dict template
        logger.info("Creating model template for DCP loading...")
        with init_empty_weights():
            model = model_cls(config)
        model_state_dict = model.state_dict()

        # Load full state dict from DCP using the model template
        logger.info("Loading full state dict from DCP checkpoint...")
        full_state_dict = self._load_full_state_dict(checkpoint_path, model_state_dict)
        logger.info(f"Loaded {len(full_state_dict)} keys from DCP checkpoint")

        # Load into model
        model.load_state_dict(full_state_dict, assign=True)
        self._maybe_tie_weights(model, config)

        # Save in HuggingFace format
        processor = AutoProcessor.from_pretrained(checkpoint_path)
        processor.save_pretrained(output_path)
        config.save_pretrained(output_path)
        model.save_pretrained(output_path)

        # Copy extra config files that AutoProcessor may not emit for custom processors
        for extra_file in ["processor_config.json"]:
            src = checkpoint_path / extra_file
            dst = output_path / extra_file
            if src.exists() and src.resolve() != dst.resolve():
                shutil.copy2(src, dst)

        logger.info(f"Merged checkpoint saved to {output_path}")
        return output_path
