"""Mixin for VLMEvalKit benchmark evaluation during training.

Runs VLMEvalKit benchmarks on the in-memory model without saving checkpoints.
Supports multi-rank data-parallel evaluation for speed.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
from typing import Optional

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)

from trainer.fsdp2.components import InferenceProbeSchedule
from utils.distributed import dist_barrier


class BenchEvalMixin:
    """Mixin that adds VLMEvalKit benchmark evaluation to a trainer.

    Requires the trainer to also inherit from BagelProbeMixin (for model
    building utilities) and FSDP2SFTTrainer (for optimizer/model access).
    """

    def _setup_bench_eval(self):
        cfg = getattr(self.args, "bench_eval_config", None)
        if not cfg or not isinstance(cfg, dict):
            self._bench_eval_enabled = False
            return
        if not bool(cfg.get("enabled", False)):
            self._bench_eval_enabled = False
            return

        interval = int(cfg.get("interval_steps", 0) or 0)
        if interval <= 0:
            self._bench_eval_enabled = False
            return

        datasets = cfg.get("datasets", [])
        if not datasets:
            logger.warning(
                "bench_eval_config is enabled but no datasets specified. "
                "Disabling bench eval."
            )
            self._bench_eval_enabled = False
            return

        start_step = int(cfg.get("start_step", interval))
        self._bench_eval_schedule = InferenceProbeSchedule(
            enabled=True,
            interval_steps=interval,
            start_step=start_step,
        )
        self._bench_eval_datasets = list(datasets)
        self._bench_eval_use_all_ranks = bool(cfg.get("use_all_ranks", True))
        self._bench_eval_understanding = bool(cfg.get("understanding_output", True))
        self._bench_eval_work_dir = str(cfg.get("work_dir", "./bench_eval_outputs"))
        self._bench_eval_infer_kwargs = dict(cfg.get("infer_kwargs", {}) or {})
        # Fallback to probe kwargs when bench eval infer_kwargs is not specified
        if not self._bench_eval_infer_kwargs:
            self._bench_eval_infer_kwargs = dict(
                getattr(self, "inference_probe_default_kwargs", {})
            )
        self._bench_eval_judge_kwargs = dict(cfg.get("judge_kwargs", {}) or {})
        self._bench_eval_enabled = True
        # Apply training alignment defaults to bench eval kwargs
        self._apply_bench_eval_alignment_defaults()

        logger.info(
            f"Bench eval enabled: datasets={self._bench_eval_datasets}, "
            f"interval={interval}, start_step={start_step}, "
            f"use_all_ranks={self._bench_eval_use_all_ranks}"
        )

    def _apply_bench_eval_alignment_defaults(self):
        """Align bench eval inference kwargs with training settings."""
        try:
            from models.bagel.inference_alignment import (
                apply_training_alignment_defaults,
            )

            apply_training_alignment_defaults(
                self._bench_eval_infer_kwargs,
                self.processing_class,
                self.train_dataset,
            )
        except Exception as e:
            logger.warning(f"Failed to apply bench eval alignment defaults: {e}")

    def _ensure_vlmeval_importable(self):
        """Add VLMEvalKit to sys.path if not already importable."""
        try:
            import vlmeval  # noqa: F401

            return True
        except ImportError:
            pass
        # Search for VLMEvalKit_Thinkmorph relative to the project root
        candidates = [
            os.path.join(os.getcwd(), "VLMEvalKit_Thinkmorph"),
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                "..",
                "VLMEvalKit_Thinkmorph",
            ),
        ]
        for candidate in candidates:
            candidate = os.path.abspath(candidate)
            if os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.insert(0, candidate)
                try:
                    import vlmeval  # noqa: F401

                    logger.info(f"Added VLMEvalKit to sys.path: {candidate}")
                    return True
                except ImportError:
                    sys.path.remove(candidate)
        logger.error(
            "Cannot import vlmeval. Ensure VLMEvalKit_Thinkmorph is in the "
            "project root or on PYTHONPATH."
        )
        return False

    def run_bench_eval(self, rank: int):
        if not self._bench_eval_enabled:
            return
        if not self._bench_eval_schedule.should_run(self.global_step):
            return

        if not self._ensure_vlmeval_importable():
            return

        logger.info(
            f"[rank {rank}] Starting bench eval at step {self.global_step} "
            f"for datasets: {self._bench_eval_datasets}"
        )

        world_size = self._world_size()
        use_all_ranks = self._bench_eval_use_all_ranks and world_size > 1

        try:
            if use_all_ranks:
                self._run_bench_eval_all_ranks(rank, world_size)
            else:
                self._run_bench_eval_rank0(rank, world_size)
        except Exception as e:
            logger.exception(
                f"[rank {rank}] Bench eval failed at step {self.global_step}: "
                f"{type(e).__name__}: {e!r}"
            )
        finally:
            torch.cuda.empty_cache()
            if world_size > 1:
                dist_barrier()

    def _run_bench_eval_all_ranks(self, rank: int, world_size: int):
        """Run benchmark evaluation with all ranks in data-parallel mode."""
        probe_model = None
        try:
            # All ranks gather full state dict
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

            # Broadcast state dict from rank0 to all ranks via GPU
            full_state_dict = self._broadcast_state_dict_from_rank0(
                full_state_dict, rank, world_size
            )

            # Each rank builds its own unsharded model
            probe_model = self._build_rank0_probe_model(full_state_dict)
            del full_state_dict
            torch.cuda.empty_cache()

            # Create adapter
            adapter = self._build_bench_eval_adapter(probe_model)

            # Run evaluation for each dataset
            for dataset_name in self._bench_eval_datasets:
                try:
                    self._eval_single_dataset_distributed(
                        adapter, dataset_name, rank, world_size
                    )
                except Exception as e:
                    logger.error(
                        f"[rank {rank}] Bench eval failed for {dataset_name}: "
                        f"{type(e).__name__}: {e!r}"
                    )
        finally:
            del probe_model
            torch.cuda.empty_cache()

    def _run_bench_eval_rank0(self, rank: int, world_size: int):
        """Run benchmark evaluation on rank 0 only."""
        probe_model = None
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

            if rank != 0:
                del full_state_dict
                return

            probe_model = self._build_rank0_probe_model(full_state_dict)
            del full_state_dict
            torch.cuda.empty_cache()

            adapter = self._build_bench_eval_adapter(probe_model)

            for dataset_name in self._bench_eval_datasets:
                try:
                    self._eval_single_dataset_rank0(adapter, dataset_name)
                except Exception as e:
                    logger.error(
                        f"Bench eval failed for {dataset_name}: "
                        f"{type(e).__name__}: {e!r}"
                    )
        finally:
            del probe_model
            torch.cuda.empty_cache()

    def _build_bench_eval_adapter(self, probe_model):
        from trainer.fsdp2.vlmeval_adapter import ThinkMorphInMemory

        save_dir = os.path.join(
            self._bench_eval_work_dir,
            f"step_{self.global_step}",
        )
        return ThinkMorphInMemory(
            model=probe_model,
            vae_model=probe_model.vae_model,
            tokenizer=self.processing_class.processor,
            vae_transform=self.processing_class.vae_image_transform,
            vit_transform=self.processing_class.vit_image_transform,
            new_token_ids=self.processing_class.new_token_ids,
            infer_kwargs=self._bench_eval_infer_kwargs,
            understanding_output=self._bench_eval_understanding,
            save_dir=save_dir if not self._bench_eval_understanding else None,
        )

    def _eval_single_dataset_distributed(
        self, adapter, dataset_name: str, rank: int, world_size: int
    ):
        """Evaluate a single dataset with samples distributed across ranks."""
        from vlmeval.dataset import build_dataset

        # All ranks build the dataset (it caches data locally)
        dataset = build_dataset(dataset_name)
        if dataset is None:
            logger.error(f"Failed to build dataset: {dataset_name}")
            return

        adapter.set_dump_image(dataset.dump_image)

        data = dataset.data
        total_samples = len(data)
        # Distribute samples: rank i handles indices [i, i+ws, i+2*ws, ...]
        my_indices = list(range(rank, total_samples, world_size))

        logger.info(
            f"[rank {rank}] Bench eval {dataset_name}: "
            f"{len(my_indices)}/{total_samples} samples"
        )

        # Run inference on my subset
        my_predictions = {}
        for i, sample_idx in enumerate(my_indices):
            row = data.iloc[sample_idx]
            idx = row["index"]
            try:
                struct = dataset.build_prompt(row)
                response = adapter.generate(message=struct, dataset=dataset_name)
                my_predictions[idx] = response
            except Exception as e:
                logger.warning(
                    f"[rank {rank}] Bench eval sample {idx} failed: "
                    f"{type(e).__name__}: {e}"
                )
                my_predictions[idx] = f"Failed: {type(e).__name__}: {e}"
            if (i + 1) % 10 == 0:
                logger.info(
                    f"[rank {rank}] Bench eval {dataset_name}: "
                    f"{i + 1}/{len(my_indices)} done"
                )
            torch.cuda.empty_cache()

        # Gather predictions to rank 0
        all_predictions = self._gather_predictions(my_predictions, rank, world_size)

        # Rank 0 evaluates
        if rank == 0 and all_predictions is not None:
            self._evaluate_and_log(dataset, dataset_name, all_predictions)

    def _eval_single_dataset_rank0(self, adapter, dataset_name: str):
        """Evaluate a single dataset on rank 0 only."""
        from vlmeval.dataset import build_dataset

        dataset = build_dataset(dataset_name)
        if dataset is None:
            logger.error(f"Failed to build dataset: {dataset_name}")
            return

        adapter.set_dump_image(dataset.dump_image)

        data = dataset.data
        total_samples = len(data)
        logger.info(f"Bench eval {dataset_name}: {total_samples} samples (rank0 only)")

        predictions = {}
        for i in range(total_samples):
            row = data.iloc[i]
            idx = row["index"]
            try:
                struct = dataset.build_prompt(row)
                response = adapter.generate(message=struct, dataset=dataset_name)
                predictions[idx] = response
            except Exception as e:
                logger.warning(
                    f"Bench eval sample {idx} failed: {type(e).__name__}: {e}"
                )
                predictions[idx] = f"Failed: {type(e).__name__}: {e}"
            if (i + 1) % 10 == 0:
                logger.info(f"Bench eval {dataset_name}: {i + 1}/{total_samples} done")
            torch.cuda.empty_cache()

        self._evaluate_and_log(dataset, dataset_name, predictions)

    def _gather_predictions(
        self, my_predictions: dict, rank: int, world_size: int
    ) -> Optional[dict]:
        """Gather prediction dicts from all ranks to rank 0."""
        # Serialize predictions
        buffer = pickle.dumps(my_predictions)
        size = torch.tensor(len(buffer), dtype=torch.long, device="cuda")

        # Gather sizes
        all_sizes = [
            torch.tensor(0, dtype=torch.long, device="cuda") for _ in range(world_size)
        ]
        dist.all_gather(all_sizes, size)

        if rank == 0:
            max_size = max(s.item() for s in all_sizes)
        else:
            max_size = max(s.item() for s in all_sizes)

        # Pad buffer to max_size
        padded = torch.zeros(max_size, dtype=torch.uint8, device="cuda")
        padded[: len(buffer)] = torch.frombuffer(
            bytearray(buffer), dtype=torch.uint8
        ).cuda()

        # Gather all buffers
        all_buffers = [
            torch.zeros(max_size, dtype=torch.uint8, device="cuda")
            for _ in range(world_size)
        ]
        dist.all_gather(all_buffers, padded)

        if rank != 0:
            return None

        # Deserialize and merge
        all_predictions = {}
        for i, (buf, sz) in enumerate(zip(all_buffers, all_sizes)):
            data_bytes = buf[: sz.item()].cpu().numpy().tobytes()
            preds = pickle.loads(data_bytes)
            all_predictions.update(preds)

        return all_predictions

    def _evaluate_and_log(self, dataset, dataset_name: str, predictions: dict):
        """Build result file, run evaluation, log metrics to W&B, and clean up."""
        import pandas as pd

        # Build result DataFrame in VLMEvalKit format
        data = dataset.data.copy()
        data["prediction"] = [str(predictions.get(idx, "")) for idx in data["index"]]
        if "image" in data.columns:
            data = data.drop(columns=["image"])

        # Use a temp file for VLMEvalKit evaluate() (it reads from disk)
        tmp_dir = tempfile.mkdtemp(prefix="bench_eval_")
        result_file = os.path.join(tmp_dir, f"bench_eval_{dataset_name}.tsv")
        data.to_csv(result_file, sep="\t", index=False)

        # Run evaluation
        metrics = {}
        try:
            eval_result = dataset.evaluate(result_file, **self._bench_eval_judge_kwargs)
            if isinstance(eval_result, dict):
                metrics = eval_result
            elif isinstance(eval_result, pd.DataFrame):
                for col in eval_result.columns:
                    try:
                        val = eval_result[col].iloc[0]
                        if isinstance(val, (int, float)):
                            metrics[col] = float(val)
                    except Exception:
                        pass
        except Exception as e:
            logger.error(
                f"Bench eval evaluation failed for {dataset_name}: "
                f"{type(e).__name__}: {e}"
            )
        finally:
            # Clean up temp files
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Log to W&B
        if metrics and hasattr(self, "tracking"):
            prefixed = {
                f"bench_eval/{dataset_name}/{k}": v
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            if prefixed:
                self.tracking.log(prefixed, step=self.global_step)
                logger.info(
                    f"Bench eval {dataset_name} step {self.global_step}: {prefixed}"
                )
            else:
                logger.info(
                    f"Bench eval {dataset_name} step {self.global_step}: "
                    f"no numeric metrics to log"
                )
        elif not metrics:
            logger.info(
                f"Bench eval {dataset_name} step {self.global_step}: "
                f"evaluation returned no metrics"
            )
