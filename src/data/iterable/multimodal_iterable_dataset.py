import math
import random
import traceback

import torch.distributed as dist
from datasets import Dataset as HFDataset
from loguru import logger
from torch.utils.data import get_worker_info

import parallel.process_group_manager as pgm
from data.iterable.base_iterable_dataset import BaseIterableDataset
from data.multimodal_mixin import MultiModalDataLoadingMixin
from utils import DataUtilities

class MultiModalIterableDataset(BaseIterableDataset, MultiModalDataLoadingMixin):
    """
    MultiModalDataset provides concrete implementation for handling multimodal data
    including images, audio, and videos read from local JSONL or Parquet files.

    This class inherits from BaseDataset and implements all the abstract methods
    with full functionality for data loading, processing, and packing.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.cur_idx = 0
        self._resume_cur_idx = 0
        if not dist.is_initialized():
            logger.info(
                "Distributed environment not initialized, setting rank and world size to 0 and 1"
            )
            self.rank = 0
            self.world_size = 1
        else:
            logger.info(
                "Distributed environment initialized, setting rank and world size to dist.get_rank() and dist.get_world_size()"
            )
            # Try to use data parallel rank if available, otherwise fall back to global rank
            if (
                pgm is not None
                and hasattr(pgm, "process_group_manager")
                and pgm.process_group_manager is not None
            ):
                self.rank = pgm.process_group_manager.dp_rank
                self.world_size = pgm.process_group_manager.dp_world_size
            else:
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()

    def _build_from_config(self):
        """Load and prepare data from the configuration."""
        if self.config.dataset_format == "jsonl":
            folders = getattr(self.config, "dataset_folders", None)
            if folders:
                self.data_list = self._load_from_folders(folders)
            else:
                self.data_list = DataUtilities.load_jsonlines(self.config.dataset_path)
                for line_idx, row in enumerate(self.data_list):
                    if not isinstance(row, dict):
                        continue
                    row.setdefault("_jsonl_line_idx", line_idx)
                    row.setdefault("_jsonl_source_path", self.config.dataset_path)
        elif self.config.dataset_format == "parquet":
            self.data_list = HFDataset.from_parquet(self.config.dataset_path)
            self.data_folder = [""] * len(self.data_list)
        else:
            raise NotImplementedError(
                f"Unsupported dataset_format={self.config.dataset_format}. "
                "Only 'jsonl' and 'parquet' are supported."
            )

        if not isinstance(self.data_list, HFDataset):
            # Serialize dict-valued fields to JSON strings to avoid PyArrow
            # schema inference failures on dicts with inconsistent nested types
            # (e.g. a field that is sometimes a list, sometimes a scalar).
            import json as _json

            for row in self.data_list:
                for k, v in row.items():
                    if isinstance(v, dict):
                        row[k] = _json.dumps(v, ensure_ascii=False)
            self.data_list = HFDataset.from_list(self.data_list)
        if getattr(self, "data_folder", None) is None:
            self.data_folder = [""] * len(self.data_list)

        if self.config.shuffle:
            logger.info("Shuffle Dataset ...")
            data_index = [i for i in range(len(self.data_list))]
            # Make sure the shuffle is the same across all dp ranks
            random.seed(self.config.data_seed)
            random.shuffle(data_index)
            if isinstance(self.data_list, HFDataset):
                self.data_list = self.data_list.select(data_index)
            else:
                self.data_list = [self.data_list[i] for i in data_index]
            if getattr(self, "data_folder", None) is not None:
                self.data_folder = [self.data_folder[i] for i in data_index]

    def _load_from_folders(self, folders: list[str]) -> list[dict]:
        """Load and merge JSONL data from multiple task folders.

        When ``folder_jsonl_filename`` is set in the config, that exact name is
        used.  Otherwise the method auto-discovers the JSONL file by trying
        these candidates in order:

        1. ``{folder}/train.jsonl``
        2. ``{folder}/training.jsonl``
        3. ``{folder}/train/train.jsonl``
        4. ``{folder}/training/training.jsonl``

        Each candidate is tried in turn; a missing file is skipped.

        A per-row ``_images_root`` field is injected so downstream code resolves
        image paths correctly.
        """
        all_rows: list[dict] = []
        explicit_filename = getattr(self.config, "folder_jsonl_filename", None)
        include_val = (
            bool(getattr(self.config, "include_val_in_training", False))
            and not explicit_filename
        )

        def try_load(path):
            return DataUtilities.load_jsonlines(path)

        def discover(candidates, accept_empty):
            """Try candidates in order. Return (rows, path, images_root) for the
            first successful load, or (None, None, None) if every candidate is
            missing (and, when accept_empty is False, also if every candidate is
            empty)."""
            for jsonl_path, images_root in candidates:
                try:
                    rows = try_load(jsonl_path)
                except Exception:
                    logger.debug(
                        f"JSONL not found at {jsonl_path}, trying next candidate"
                    )
                    continue
                if not rows and not accept_empty:
                    logger.debug(f"JSONL {jsonl_path} is empty, trying next candidate")
                    continue
                return rows, jsonl_path, images_root
            return None, None, None

        for folder in folders:
            folder = folder.rstrip("/")

            if explicit_filename:
                train_candidates = [(f"{folder}/{explicit_filename}", folder)]
            else:
                train_candidates = [
                    (f"{folder}/train.jsonl", folder),
                    (f"{folder}/training.jsonl", folder),
                    (f"{folder}/train/train.jsonl", f"{folder}/train"),
                    (f"{folder}/training/training.jsonl", f"{folder}/training"),
                ]

            # When include_val is on, an empty train should fall back to val,
            # so don't accept empty train. Without include_val, preserve the
            # legacy behavior where an empty train.jsonl silently yields 0 rows.
            train_rows, train_path, train_images_root = discover(
                train_candidates, accept_empty=not include_val
            )
            if train_rows is not None:
                logger.info(f"Loaded {len(train_rows)} rows from {train_path}")

            val_rows, val_path, val_images_root = None, None, None
            if include_val:
                val_candidates = [
                    (f"{folder}/val.jsonl", folder),
                    (f"{folder}/validation.jsonl", folder),
                    (f"{folder}/val/val.jsonl", f"{folder}/val"),
                    (f"{folder}/validation/validation.jsonl", f"{folder}/validation"),
                ]
                val_rows, val_path, val_images_root = discover(
                    val_candidates, accept_empty=False
                )
                if val_rows is not None:
                    logger.info(f"Loaded {len(val_rows)} val rows from {val_path}")

            if train_rows is None and val_rows is None:
                tried = [c[0] for c in train_candidates]
                if include_val:
                    tried += [c[0] for c in val_candidates]
                raise FileNotFoundError(
                    f"No non-empty JSONL file found for folder '{folder}'. "
                    f"Tried: {tried}"
                )

            if include_val and train_rows is None:
                logger.warning(
                    f"include_val_in_training=True: no train jsonl found for "
                    f"'{folder}', using val only ({val_path})"
                )
            elif include_val and val_rows is None:
                logger.warning(
                    f"include_val_in_training=True: no val jsonl found for "
                    f"'{folder}', using train only ({train_path})"
                )

            for rows, path, images_root in (
                (train_rows, train_path, train_images_root),
                (val_rows, val_path, val_images_root),
            ):
                if rows is None:
                    continue
                for line_idx, row in enumerate(rows):
                    if not isinstance(row, dict):
                        continue
                    row["_images_root"] = images_root
                    row.setdefault("_jsonl_line_idx", line_idx)
                    row.setdefault("_jsonl_source_path", path)
                all_rows.extend(rows)

        logger.info(f"Total {len(all_rows)} rows from {len(folders)} folder(s)")
        return all_rows

    def get_one_sample(self, index, data_folder=None, data_list=None):
        """Get a sample from the dataset by index."""
        if data_folder is None:
            data_folder = self.data_folder[index]
        if data_list is None:
            data_list = self.data_list
        if self.config.dataset_format == "jsonl":
            data_dict = self.load_from_json(data_list[index])
        elif self.config.dataset_format == "parquet":
            data_dict = self.load_from_hf(data_list[index])
        else:
            raise NotImplementedError(
                f"Unsupported dataset_format={self.config.dataset_format}. "
                "Only 'jsonl' and 'parquet' are supported."
            )
        return data_dict

    def _get_rank_indices(self, total_size: int) -> list[int]:
        if total_size <= 0:
            return []

        if getattr(self.config, "stride_sharding", False):
            return list(range(self.rank, total_size, self.world_size))

        rank_mod_size = total_size % self.world_size
        per_rank_size = [
            (total_size // self.world_size) + 1
            if i < rank_mod_size
            else (total_size // self.world_size)
            for i in range(self.world_size)
        ]
        start_index = sum(per_rank_size[: self.rank])
        end_index = start_index + per_rank_size[self.rank]
        return list(range(start_index, end_index))

    def _get_worker_indices(
        self, total_size: int, worker_info
    ) -> tuple[list[int], int]:
        if worker_info is None:
            return list(range(total_size)), 0

        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        if total_size <= 0:
            return [], worker_id

        if getattr(self.config, "stride_sharding", False):
            effective_workers = min(num_workers, total_size)
            worker_slot = worker_id % effective_workers
            return list(range(worker_slot, total_size, effective_workers)), worker_id

        per_worker = int(math.ceil(total_size / float(num_workers)))
        iter_start = worker_id * per_worker
        iter_end = min(iter_start + per_worker, total_size)
        return list(range(iter_start, iter_end)), worker_id

    def _select_rows(self, indices: list[int]):
        curr_data_list = self.data_list.select(indices)
        curr_data_folder = [self.data_folder[i] for i in indices]
        return curr_data_list, curr_data_folder

    def state_dict(self):
        return {"cur_idx": self.cur_idx}

    def load_state_dict(self, state_dict):
        self._resume_cur_idx = state_dict.get("cur_idx", 0)

    def __iter__(self):
        worker_info = get_worker_info()

        assert isinstance(self.data_list, HFDataset), (
            "Data list must be a HuggingFace dataset for IterableDataset"
        )

        rank_indices = self._get_rank_indices(len(self.data_list))
        worker_local_indices, worker_id = self._get_worker_indices(
            len(rank_indices), worker_info
        )
        selected_indices = [rank_indices[i] for i in worker_local_indices]
        curr_data_list, curr_data_folder = self._select_rows(selected_indices)

        packing_mode = self.config.resolve_packing_mode()
        if packing_mode == "dynamic":
            # ThinkMorph-style dynamic packing with overflow buffer
            self.cur_idx = self._resume_cur_idx
            self._resume_cur_idx = 0
            max_num_tokens = getattr(self.config, "max_num_tokens", 36864)
            expected_num_tokens = (
                getattr(self.config, "expected_num_tokens", None) or max_num_tokens
            )
            max_num_tokens_per_sample = (
                getattr(self.config, "max_num_tokens_per_sample", None)
                or max_num_tokens
            )
            prefer_buffer_before = (
                getattr(self.config, "prefer_buffer_before", None)
                or max_num_tokens // 2
            )
            max_buffer_size = getattr(self.config, "max_buffer_size", 50)

            packed_batch = []
            packed_length = 0
            _dropped_overlong_count = 0
            overflow_buffer = []

            def _make_meta(idx, from_buf):
                if from_buf:
                    return {
                        "dataset_index": -1,
                        "worker_id": int(worker_id),
                        "rank": int(self.rank),
                    }
                meta = {
                    "dataset_index": int(selected_indices[idx]),
                    "worker_id": int(worker_id),
                    "rank": int(self.rank),
                }
                row = curr_data_list[idx]
                if isinstance(row, dict):
                    if row.get("_jsonl_line_idx") is not None:
                        meta["jsonl_line_idx"] = int(row["_jsonl_line_idx"])
                    if row.get("_jsonl_source_path"):
                        meta["jsonl_source_path"] = row["_jsonl_source_path"]
                return meta

            def _next_sample():
                while self.cur_idx < len(curr_data_list):
                    try:
                        data_dict = self.get_one_sample(
                            self.cur_idx,
                            curr_data_folder[self.cur_idx],
                            curr_data_list,
                        )
                    except Exception as e:
                        traceback.print_exc()
                        logger.error(f"Error getting one sample: {e}, skip")
                        self.cur_idx += 1
                        continue
                    idx = self.cur_idx
                    self.cur_idx += 1
                    if data_dict is None:
                        continue
                    if "sample_meta" not in data_dict:
                        data_dict["sample_meta"] = [_make_meta(idx, False)]
                    return data_dict
                return None

            while self.cur_idx < len(curr_data_list) or overflow_buffer:
                # Mandatory-first: pack one sample at start of each batch
                if packed_length == 0:
                    data_dict = _next_sample()
                    if data_dict is None:
                        break
                    data_length = data_dict["input_ids"].shape[0]
                    if data_length > max_num_tokens_per_sample and not data_dict.get(
                        "_adaptive_keep_overlong"
                    ):
                        _dropped_overlong_count += 1
                        logger.warning(
                            f"Dropped sample with {data_length} tokens "
                            f"(max_num_tokens_per_sample={max_num_tokens_per_sample}). "
                            f"Total dropped so far: {_dropped_overlong_count}"
                        )
                        continue
                    data_dict.pop("_adaptive_keep_overlong", None)
                    packed_batch.append(data_dict)
                    packed_length += data_length
                    if packed_length >= expected_num_tokens:
                        yield packed_batch
                        packed_batch = []
                        packed_length = 0
                        continue

                # Regular loop: buffer-first or draw new
                if packed_length < prefer_buffer_before and len(overflow_buffer) > 0:
                    data_dict = overflow_buffer.pop(0)
                    from_buffer = True
                else:
                    data_dict = _next_sample()
                    if data_dict is None:
                        break
                    from_buffer = False

                data_length = data_dict["input_ids"].shape[0]

                if data_length > max_num_tokens_per_sample and not data_dict.get(
                    "_adaptive_keep_overlong"
                ):
                    _dropped_overlong_count += 1
                    logger.warning(
                        f"Dropped sample with {data_length} tokens "
                        f"(max_num_tokens_per_sample={max_num_tokens_per_sample}). "
                        f"Total dropped so far: {_dropped_overlong_count}"
                    )
                    continue
                data_dict.pop("_adaptive_keep_overlong", None)

                if packed_length + data_length > max_num_tokens:
                    if not from_buffer and len(overflow_buffer) < max_buffer_size:
                        overflow_buffer.append(data_dict)
                    else:
                        if len(packed_batch) > 0:
                            yield packed_batch
                            packed_batch = []
                            packed_length = 0
                        # Instead of dropping, start a new batch with this sample
                        packed_batch.append(data_dict)
                        packed_length = data_length
                        if packed_length >= expected_num_tokens:
                            yield packed_batch
                            packed_batch = []
                            packed_length = 0
                    continue

                packed_batch.append(data_dict)
                packed_length += data_length

                if packed_length >= expected_num_tokens:
                    yield packed_batch
                    packed_batch = []
                    packed_length = 0

            # Drain remaining overflow buffer
            while len(overflow_buffer) > 0:
                data_dict = overflow_buffer.pop(0)
                data_length = data_dict["input_ids"].shape[0]

                if packed_length + data_length > max_num_tokens:
                    if len(packed_batch) > 0:
                        yield packed_batch
                        packed_batch = []
                        packed_length = 0

                packed_batch.append(data_dict)
                packed_length += data_length

                if packed_length >= expected_num_tokens:
                    yield packed_batch
                    packed_batch = []
                    packed_length = 0

            # Flush remaining batch
            if len(packed_batch) > 0:
                yield packed_batch

        elif packing_mode == "greedy":
            # Original greedy packing
            self.cur_idx = self._resume_cur_idx
            self._resume_cur_idx = 0
            buffer = []
            buffer_length = 0
            packing_length = self.config.packing_length

            # Iterate through the dataset once per epoch
            while self.cur_idx < len(curr_data_list):
                try:
                    data_dict = self.get_one_sample(
                        self.cur_idx,
                        curr_data_folder[self.cur_idx],
                        curr_data_list,
                    )
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error getting one sample: {e}, skip this sample")
                    self.cur_idx += 1
                    continue
                if data_dict is None:
                    self.cur_idx += 1
                    continue
                if "sample_meta" not in data_dict:
                    sample_row = curr_data_list[self.cur_idx]
                    sample_meta = {
                        "dataset_index": int(selected_indices[self.cur_idx]),
                        "worker_id": int(worker_id),
                        "rank": int(self.rank),
                    }
                    if isinstance(sample_row, dict):
                        if sample_row.get("_jsonl_line_idx") is not None:
                            sample_meta["jsonl_line_idx"] = int(
                                sample_row["_jsonl_line_idx"]
                            )
                        if sample_row.get("_jsonl_source_path"):
                            sample_meta["jsonl_source_path"] = sample_row[
                                "_jsonl_source_path"
                            ]
                    data_dict["sample_meta"] = [sample_meta]
                input_ids = data_dict["input_ids"]
                data_length = input_ids.shape[0]
                self.cur_idx += 1

                # Drop overlong sample if filtering is enabled (bypassed when
                # the adaptive image-sizing layer asked to keep it).
                if (
                    data_length > packing_length
                    and self.config.filter_overlong
                    and not data_dict.get("_adaptive_keep_overlong")
                ):
                    continue
                data_dict.pop("_adaptive_keep_overlong", None)

                # If current sample cannot fit into current buffer, yield the buffer first
                if buffer_length > 0 and buffer_length + data_length > packing_length:
                    yield buffer
                    buffer = []
                    buffer_length = 0

                # If the sample is still longer than packing_length (and not filtered),
                # yield it as its own batch to avoid stalling
                if data_length > packing_length:
                    yield [data_dict]
                    continue

                # Append to buffer
                buffer.append(data_dict)
                buffer_length += data_length

            # Flush remaining buffer
            if len(buffer) > 0:
                yield buffer
        else:
            self.cur_idx = self._resume_cur_idx
            self._resume_cur_idx = 0
            while self.cur_idx < len(curr_data_list):
                try:
                    data_dict = self.get_one_sample(
                        self.cur_idx,
                        curr_data_folder[self.cur_idx],
                        curr_data_list,
                    )
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error getting one sample: {e}, skip this sample")
                    self.cur_idx += 1
                    continue
                if data_dict is not None:
                    if "sample_meta" not in data_dict:
                        sample_row = curr_data_list[self.cur_idx]
                        sample_meta = {
                            "dataset_index": int(selected_indices[self.cur_idx]),
                            "worker_id": int(worker_id),
                            "rank": int(self.rank),
                        }
                        if isinstance(sample_row, dict):
                            if sample_row.get("_jsonl_line_idx") is not None:
                                sample_meta["jsonl_line_idx"] = int(
                                    sample_row["_jsonl_line_idx"]
                                )
                            if sample_row.get("_jsonl_source_path"):
                                sample_meta["jsonl_source_path"] = sample_row[
                                    "_jsonl_source_path"
                                ]
                        data_dict["sample_meta"] = [sample_meta]
                    yield data_dict
                self.cur_idx += 1

    def load_from_json(self, data, data_folder=None):
        """
        Default implementation for loading from JSON format.
        Subclasses should override this method to provide specific implementations.

        Args:
            data: The JSON data to process
            data_folder: Optional folder path for data files

        Returns:
            Processed data dictionary
        """
        raise NotImplementedError("Subclasses must implement load_from_json")

    def load_from_hf(self, data):
        """
        Default implementation for loading from HuggingFace dataset format.
        Subclasses should override this method to provide specific implementations.

        Args:
            data: The HuggingFace dataset data to process

        Returns:
            Processed data dictionary
        """
        raise NotImplementedError("Subclasses must implement load_from_hf")

    def get_collator(self):
        """
        Get the appropriate collator for this dataset.
        Subclasses should override this method to provide specific implementations.

        Returns:
            A collator instance suitable for this dataset type
        """
        raise NotImplementedError("Subclasses must implement get_collator")
