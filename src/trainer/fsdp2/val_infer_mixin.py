"""Mixin for validation inference evaluation during training.

Reads val.jsonl from dataset folders, runs inference on sampled questions,
and computes answer accuracy against ground truth.
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from loguru import logger
from PIL import Image as PILImage
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)

from trainer.fsdp2.components import InferenceProbeSchedule
from utils.distributed import dist_barrier


def _parse_image_list(raw):
    """Parse an image list field (may be JSON string or list)."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        return [raw] if raw else []
    return []


def _extract_answer_tag(text: str) -> str:
    """Extract content from <answer>...</answer> tags, or return full text."""
    if not isinstance(text, str):
        return str(text).strip()
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_choice(pred: str) -> str:
    """Extract the chosen option letter (A-D) from a model prediction.

    Checks in order: <answer>X</answer>, leading "X:" pattern (after
    stripping <think> blocks), (X) pattern, standalone letter.
    Returns uppercase letter or empty string.
    """
    if not pred:
        return ""
    # 1) <answer>X</answer>
    m = re.search(r"<answer>\s*([A-Da-d])\b", pred)
    if m:
        return m.group(1).upper()
    # Strip <think>...</think> blocks to find the actual answer part
    cleaned = re.sub(r"<think>.*?</think>", "", pred, flags=re.DOTALL).strip()
    if not cleaned:
        return ""
    # 2) Starts with "X:" — e.g. "B: Front right"
    m = re.match(r"([A-Da-d])\s*[:.)]\s", cleaned)
    if m:
        return m.group(1).upper()
    # 3) (X) or [X] pattern
    m = re.search(r"[(\[]([A-Da-d])[)\]]", cleaned)
    if m:
        return m.group(1).upper()
    # 4) Standalone letter (entire cleaned text is just one letter)
    m = re.match(r"([A-Da-d])\s*$", cleaned)
    if m:
        return m.group(1).upper()
    return ""


class ValInferMixin:
    """Mixin that adds validation inference evaluation to a trainer.

    Reads val.jsonl, runs inference on sampled questions, compares answers
    against ground truth, and logs accuracy to W&B.
    """

    def _setup_val_infer(self):
        cfg = getattr(self.args, "val_eval_config", None)
        if not cfg or not isinstance(cfg, dict):
            self._val_infer_enabled = False
            return
        if not bool(cfg.get("enabled", False)):
            self._val_infer_enabled = False
            return
        if not bool(cfg.get("inference_enabled", False)):
            self._val_infer_enabled = False
            return

        interval = int(cfg.get("interval_steps", 0) or 0)
        if interval <= 0:
            self._val_infer_enabled = False
            return

        start_step = int(cfg.get("start_step", interval))
        self._val_infer_schedule = InferenceProbeSchedule(
            enabled=True,
            interval_steps=interval,
            start_step=start_step,
        )
        self._val_infer_max_samples = int(cfg.get("max_inference_samples", 100))
        self._val_infer_kwargs = dict(cfg.get("infer_kwargs", {}) or {})
        self._val_infer_use_all_ranks = bool(cfg.get("use_all_ranks", True))

        # Determine which folders to use
        val_folders = cfg.get("val_folders")
        if not val_folders:
            # Auto-discover from stages or dataset_folders
            ds_cfg = (
                self.train_dataset.config
                if hasattr(self.train_dataset, "config")
                else None
            )
            if ds_cfg:
                val_folders = getattr(ds_cfg, "dataset_folders", None)
                if not val_folders and getattr(ds_cfg, "stages", None):
                    seen = set()
                    val_folders = []
                    for stage in ds_cfg.stages:
                        for f in stage.dataset_folders:
                            if f not in seen:
                                seen.add(f)
                                val_folders.append(f)

        if not val_folders:
            logger.warning(
                "val_eval_config.inference_enabled but no val_folders found."
            )
            self._val_infer_enabled = False
            return

        self._val_infer_folders = list(val_folders)

        # Apply training alignment defaults to infer kwargs
        if not self._val_infer_kwargs:
            self._val_infer_kwargs = dict(
                getattr(self, "inference_probe_default_kwargs", {})
            )
        try:
            from models.bagel.inference_alignment import (
                apply_training_alignment_defaults,
            )

            apply_training_alignment_defaults(
                self._val_infer_kwargs,
                self.processing_class,
                self.train_dataset,
            )
        except Exception:
            pass

        self._val_infer_enabled = True
        # Load and cache samples (done once at setup)
        self._val_infer_samples = self._load_and_sample_val_data(
            self._val_infer_folders, self._val_infer_max_samples
        )
        if not self._val_infer_samples:
            logger.warning("No val inference samples loaded. Disabling.")
            self._val_infer_enabled = False
            return

        logger.info(
            f"Val infer eval enabled: {len(self._val_infer_samples)} samples from "
            f"{len(self._val_infer_folders)} folder(s), interval={interval}"
        )

    def _load_and_sample_val_data(
        self, folders: List[str], max_samples: int
    ) -> List[Dict[str, Any]]:
        """Load val.jsonl from folders and sample evenly.

        Uses a fixed-seed local RNG so the same samples are selected
        regardless of prior random state (checkpoint resume, etc.).
        """
        per_folder_data = []
        for folder in folders:
            folder = folder.rstrip("/")
            val_path = f"{folder}/val.jsonl"
            try:
                rows = self._read_jsonl_file(val_path, folder)
                if rows:
                    per_folder_data.append(rows)
                    logger.info(f"Val infer: loaded {len(rows)} rows from {val_path}")
            except Exception as e:
                logger.warning(f"Val infer: failed to load {val_path}: {e}")

        if not per_folder_data:
            return []

        # Fixed-seed RNG: same samples across restarts and experiments
        # with identical val_folders + max_inference_samples
        rng = random.Random(42)

        # Distribute max_samples evenly across folders
        n_folders = len(per_folder_data)
        base_per_folder = max_samples // n_folders
        remainder = max_samples % n_folders

        sampled = []
        for i, rows in enumerate(per_folder_data):
            n = base_per_folder + (1 if i < remainder else 0)
            n = min(n, len(rows))
            sampled.extend(rng.sample(rows, n))

        rng.shuffle(sampled)
        return sampled

    def _read_jsonl_file(self, val_path: str, images_root: str) -> List[Dict[str, Any]]:
        """Read a val.jsonl file and parse into inference-ready samples."""
        # Load via DataUtilities (same path as training data loading)
        try:
            from utils import DataUtilities

            rows = DataUtilities.load_jsonlines(val_path)
        except Exception as e:
            logger.warning(
                f"Val infer: DataUtilities.load_jsonlines failed for {val_path}: {e}"
            )
            # Fallback: direct file read
            if not os.path.exists(val_path):
                logger.warning(f"Val infer: {val_path} not found locally either")
                return []
            rows = []
            with open(val_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))

        # Convert to inference samples
        samples = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            question = (row.get("question") or "").strip()
            if not question:
                continue
            gt_answer = row.get("gt_answer") or row.get("answer") or ""
            gt_answer = str(gt_answer).strip()
            if not gt_answer:
                continue

            images = _parse_image_list(row.get("images"))
            resolved_images = []
            for img_path in images:
                if not os.path.isabs(img_path):
                    img_path = os.path.join(images_root, img_path)
                resolved_images.append(img_path)

            samples.append(
                {
                    "question": question,
                    "images": resolved_images,
                    "gt_answer": gt_answer,
                    "sample_id": row.get("sample_id", ""),
                }
            )

        return samples

    def run_val_infer(self, rank: int):
        if not self._val_infer_enabled:
            return
        if not self._val_infer_schedule.should_run(self.global_step):
            return

        world_size = self._world_size()
        use_all_ranks = self._val_infer_use_all_ranks and world_size > 1

        logger.info(
            f"[rank {rank}] Val infer eval at step {self.global_step}: "
            f"{len(self._val_infer_samples)} samples"
            f"{' (distributed)' if use_all_ranks else ' (rank0 only)'}"
        )

        probe_model = None
        inferencer = None
        try:
            use_ema = self.ema.is_enabled() and self.ema.initialized
            if use_ema:
                self.ema.swap_with_model()
            try:
                full_state_dict = get_model_state_dict(
                    self.fsdp2_model,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )
            finally:
                if use_ema:
                    self.ema.swap_with_model()

            if use_all_ranks:
                # Broadcast state dict from rank0 to all ranks via GPU
                full_state_dict = self._broadcast_state_dict_from_rank0(
                    full_state_dict, rank, world_size
                )
            else:
                if rank != 0:
                    del full_state_dict
                    dist_barrier()
                    return

            probe_model = self._build_rank0_probe_model(full_state_dict)
            del full_state_dict
            torch.cuda.empty_cache()

            from models.bagel.inferencer import InterleaveInferencer

            inferencer = InterleaveInferencer(
                model=probe_model,
                vae_model=probe_model.vae_model,
                tokenizer=self.processing_class.processor,
                vae_transform=self.processing_class.vae_image_transform,
                vit_transform=self.processing_class.vit_image_transform,
                new_token_ids=self.processing_class.new_token_ids,
            )

            # Distribute samples across ranks
            if use_all_ranks:
                my_indices = list(range(rank, len(self._val_infer_samples), world_size))
            else:
                my_indices = list(range(len(self._val_infer_samples)))

            # Run inference
            my_results = []
            n_my = len(my_indices)
            with torch.no_grad():
                for i, idx in enumerate(my_indices):
                    sample = self._val_infer_samples[idx]
                    if i % 5 == 0 or i == n_my - 1:
                        logger.info(f"[rank {rank}] Val infer: {i + 1}/{n_my} samples")
                    try:
                        input_list = self._build_val_infer_input(sample)
                        outputs = inferencer.interleave_inference(
                            input_list,
                            understanding_output=True,
                            **self._val_infer_kwargs,
                        )
                        pred_text = str(outputs[0]) if outputs else ""
                        pred_answer = _extract_answer_tag(pred_text)
                        gt_answer = sample["gt_answer"]
                        my_results.append(
                            {
                                "pred": pred_answer,
                                "pred_raw": pred_text,
                                "gt": gt_answer,
                                "question": sample["question"][:120],
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            f"[rank {rank}] Val infer sample {idx} failed: {e}"
                        )
                        my_results.append(
                            {
                                "pred": "",
                                "pred_raw": "",
                                "gt": sample["gt_answer"],
                                "question": sample["question"][:120],
                            }
                        )
                    torch.cuda.empty_cache()

            # Gather results and log metrics
            if use_all_ranks:
                all_results = self._gather_val_infer_results(
                    my_results, rank, world_size
                )
                if rank == 0 and all_results:
                    self._log_val_infer_metrics(all_results)
            else:
                if my_results:
                    self._log_val_infer_metrics(my_results)

        except Exception as e:
            logger.exception(
                f"[rank {rank}] Val infer eval failed at step {self.global_step}: "
                f"{type(e).__name__}: {e!r}"
            )
        finally:
            probe_model = None
            inferencer = None
            torch.cuda.empty_cache()
            if world_size > 1:
                dist_barrier()

    def _load_image(self, img_path: str) -> "PILImage.Image":
        """Load an image from a local path."""
        return PILImage.open(img_path).convert("RGB")

    def _build_val_infer_input(self, sample: Dict[str, Any]) -> list:
        """Build InterleaveInferencer input from a val sample."""
        input_list = []
        for img_path in sample["images"]:
            try:
                img = self._load_image(img_path)
                input_list.append(img)
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
        input_list.append(sample["question"])
        return input_list

    def _gather_val_infer_results(
        self, my_results: list, rank: int, world_size: int
    ) -> Optional[list]:
        """Gather val infer results from all ranks to rank 0."""
        import pickle

        buffer = pickle.dumps(my_results)
        size = torch.tensor(len(buffer), dtype=torch.long, device="cuda")

        all_sizes = [
            torch.tensor(0, dtype=torch.long, device="cuda") for _ in range(world_size)
        ]
        dist.all_gather(all_sizes, size)
        max_size = max(s.item() for s in all_sizes)

        padded = torch.zeros(max_size, dtype=torch.uint8, device="cuda")
        padded[: len(buffer)] = torch.frombuffer(
            bytearray(buffer), dtype=torch.uint8
        ).cuda()

        all_buffers = [
            torch.zeros(max_size, dtype=torch.uint8, device="cuda")
            for _ in range(world_size)
        ]
        dist.all_gather(all_buffers, padded)

        if rank != 0:
            return None

        all_results = []
        for buf, sz in zip(all_buffers, all_sizes):
            data_bytes = buf[: sz.item()].cpu().numpy().tobytes()
            all_results.extend(pickle.loads(data_bytes))
        return all_results

    def _log_val_infer_metrics(self, results: list):
        """Compute accuracy metrics and log to W&B."""
        total = len(results)
        if total == 0:
            return

        # Exact match (case-insensitive, stripped)
        exact_match = sum(
            1 for r in results if r["pred"].lower().strip() == r["gt"].lower().strip()
        )

        # Choice match: extract the chosen option (A/B/C/D) from the
        # prediction and compare with GT.  For non-single-letter GT,
        # falls back to substring match.
        choice_match = 0
        for r in results:
            gt = r["gt"].strip()
            pred_raw = r.get("pred_raw", r["pred"])
            if len(gt) == 1 and gt.isalpha():
                if _extract_choice(pred_raw) == gt.upper():
                    choice_match += 1
            else:
                if gt.lower() in r["pred"].lower():
                    choice_match += 1

        metrics = {
            "val_infer/exact_match_acc": exact_match / total,
            "val_infer/choice_match_acc": choice_match / total,
            "val_infer/total_samples": total,
        }

        if hasattr(self, "tracking"):
            self.tracking.log(metrics, step=self.global_step)
        logger.info(
            f"Val infer step {self.global_step}: "
            f"exact_match={exact_match}/{total} ({metrics['val_infer/exact_match_acc']:.2%}), "
            f"choice_match={choice_match}/{total} ({metrics['val_infer/choice_match_acc']:.2%})"
        )

        # Log a few example predictions for inspection
        n_examples = min(5, total)
        for i in range(n_examples):
            r = results[i]
            raw = r.get("pred_raw", r["pred"])
            # Truncate raw output for readability
            raw_short = raw[:500] + "..." if len(raw) > 500 else raw
            logger.info(
                f"  [example {i + 1}] Q: {r.get('question', '?')}\n"
                f"    GT: {r['gt']}\n"
                f"    Pred(answer): {r['pred']}\n"
                f"    Pred(raw): {raw_short}"
            )
