#!/usr/bin/env python3
"""Merge regular or EMA FSDP2 weights into one evaluation checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from accelerate import init_empty_weights
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader
from transformers import AutoConfig, AutoProcessor

from merger import FSDP2Merger
from models.registry import create_model_from_pretrained


STATE_ROOTS = {"regular": "model", "ema": "ema_model"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge an FSDP2 DCP checkpoint and fill weights absent from the "
            "selected state (for example frozen VAE weights in EMA) from the base model."
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("merged_dir", type=Path)
    parser.add_argument("base_model", type=Path)
    parser.add_argument(
        "--checkpoint-type",
        choices=sorted(STATE_ROOTS),
        default="regular",
    )
    return parser.parse_args()


def checkpoint_model_keys(checkpoint: Path, checkpoint_type: str) -> set[str]:
    root = STATE_ROOTS[checkpoint_type] + "."
    metadata = FileSystemReader(checkpoint).read_metadata()
    return {
        key[len(root) :]
        for key in metadata.state_dict_metadata
        if key.startswith(root)
    }


def weight_map(model_path: Path) -> dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as handle:
            return dict(json.load(handle)["weight_map"])
    single = model_path / "model.safetensors"
    if single.exists():
        with safe_open(single, framework="pt", device="cpu") as handle:
            return {key: single.name for key in handle.keys()}
    shards = sorted(model_path.glob("model-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No model safetensors found in {model_path}")
    mapping: dict[str, str] = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            mapping.update({key: shard.name for key in handle.keys()})
    return mapping


def load_base_weights(model_path: Path, keys: set[str]) -> dict:
    mapping = weight_map(model_path)
    unavailable = keys - mapping.keys()
    if unavailable:
        examples = ", ".join(sorted(unavailable)[:5])
        raise KeyError(
            f"Base model is missing {len(unavailable)} required weights: {examples}"
        )

    by_shard: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_shard[mapping[key]].append(key)

    loaded = {}
    for shard_name, shard_keys in sorted(by_shard.items()):
        shard_path = model_path / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                loaded[key] = handle.get_tensor(key)
    return loaded


def copy_supplementary_files(base_model: Path, output: Path) -> None:
    for name in ("ae.safetensors", "llm_config.json", "vit_config.json"):
        source = base_model / name
        target = output / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)


def merge(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint.resolve()
    base_model = args.base_model.resolve()
    output = args.merged_dir.resolve()
    if not (checkpoint / ".metadata").exists():
        raise FileNotFoundError(f"DCP metadata not found: {checkpoint / '.metadata'}")
    if not base_model.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {base_model}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    model_cls = create_model_from_pretrained(checkpoint)
    config = AutoConfig.from_pretrained(checkpoint)
    with init_empty_weights():
        model = model_cls.from_config(config)
    template = model.state_dict()

    dcp_keys = checkpoint_model_keys(checkpoint, args.checkpoint_type)
    unknown_dcp_keys = dcp_keys - template.keys()
    if unknown_dcp_keys:
        examples = ", ".join(sorted(unknown_dcp_keys)[:5])
        raise KeyError(
            f"DCP has {len(unknown_dcp_keys)} weights absent from the model: {examples}"
        )
    missing_keys = template.keys() - dcp_keys
    print(
        f"model keys={len(template)}, DCP {args.checkpoint_type} keys={len(dcp_keys)}, "
        f"base-filled keys={len(missing_keys)}",
        flush=True,
    )

    merger = FSDP2Merger(checkpoint_type=args.checkpoint_type)
    dcp_template = {key: template[key] for key in dcp_keys}
    state_dict = merger._load_full_state_dict(checkpoint, dcp_template)
    if missing_keys:
        state_dict.update(load_base_weights(base_model, set(missing_keys)))
    model.load_state_dict(state_dict, assign=True)
    merger._maybe_tie_weights(model, config)

    temp_output = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temp_output.exists():
        shutil.rmtree(temp_output)
    try:
        temp_output.mkdir(parents=True)
        processor = AutoProcessor.from_pretrained(checkpoint)
        processor.save_pretrained(temp_output)
        config.save_pretrained(temp_output)
        model.save_pretrained(temp_output, max_shard_size="100GB")
        copy_supplementary_files(base_model, temp_output)
        if not (temp_output / "model.safetensors").is_file():
            raise RuntimeError("Merge did not produce model.safetensors")
        output.parent.mkdir(parents=True, exist_ok=True)
        temp_output.rename(output)
    except Exception:
        shutil.rmtree(temp_output, ignore_errors=True)
        raise
    print(f"Merged checkpoint saved to {output}", flush=True)


def main() -> None:
    merge(parse_args())


if __name__ == "__main__":
    main()
