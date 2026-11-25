#!/usr/bin/env python3
"""Merge verified subset exports into one release directory.

Each input is a directory written by prepare_data_release.py export. The output has the same
layout (train.jsonl, val.jsonl when kept, images/, MANIFEST.json), so verify, pack and restore
work unchanged. Rows keep their text, image references and public id; each row gains a
`subset` field (the given label) and a `source_split` field (the split it came from). Images
are content-addressed, so identical files from different subsets are stored once. An input
written as LABEL=EXPORT_DIR@IDS_FILE leaves out the rows whose public id is listed in IDS_FILE
(one id per line); only images referenced by the kept rows are included. With
--merge-val-into-train, rows from val files go to train.jsonl (for corpora whose val rows
were used in training).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import re
import time

from prepare_data_release import digest, image_list

IMAGE = re.compile(r"reasoning_image_\d+$")


def merge(output: Path, name: str, stage: str, inputs: list[tuple[str, Path, Path | None]], merge_val: bool) -> dict:
    start = time.monotonic()
    if output.exists():
        raise ValueError("Output already exists; choose a new release directory")
    if stage not in {"pretext", "instruct"}:
        raise ValueError("stage must be pretext or instruct")
    labels = [label for label, _, _ in inputs]
    if len(set(labels)) != len(labels):
        raise ValueError("Subset labels must be unique")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    handles, rows, images, excluded, sources, merged_from = {}, {}, {}, {}, [], []
    try:
        for label, source, ids_file in inputs:
            manifest = json.loads((source / "MANIFEST.json").read_text())
            if manifest.get("stage") != stage:
                raise ValueError(f"{label}: stage {manifest.get('stage')} != {stage}")
            listed = {e["path"]: e for e in manifest["files"] if e["path"].startswith("images/")}
            drop = set()
            if ids_file is not None:
                drop = {line.strip() for line in ids_file.read_text().splitlines() if line.strip()}
            dropped = 0
            for split in sorted(manifest["rows"]):
                target = "train" if merge_val else split
                if target not in handles:
                    handles[target] = (output / f"{target}.jsonl").open("w", encoding="utf-8", newline="\n")
                count = kept = 0
                with (source / f"{split}.jsonl").open(encoding="utf-8") as stream:
                    for line in stream:
                        row = json.loads(line)
                        count += 1
                        if row.get("id") in drop:
                            dropped += 1
                            continue
                        if "subset" in row or "source_split" in row:
                            raise ValueError(f"{label}: rows already carry subset/source_split")
                        for ref in image_list(row["images"]) + [v for k, v in row.items() if IMAGE.fullmatch(k) and v]:
                            entry = listed.get(ref)
                            if entry is None:
                                raise ValueError(f"{label}: {ref} is not in its manifest")
                            known = images.get(ref)
                            if known is None:
                                src = source / ref
                                if src.stat().st_size != entry["bytes"]:
                                    raise ValueError(f"{label}: {ref} has the wrong size")
                                try:
                                    os.link(src, output / ref)
                                except OSError:
                                    shutil.copyfile(src, output / ref)
                                images[ref] = {"path": ref, "sha256": entry["sha256"], "bytes": entry["bytes"]}
                            elif known["sha256"] != entry["sha256"]:
                                raise ValueError(f"{label}: {ref} differs from an earlier subset")
                        row["subset"] = label
                        row["source_split"] = split
                        handles[target].write(json.dumps(row, ensure_ascii=False) + "\n")
                        kept += 1
                if count != manifest["rows"][split]:
                    raise ValueError(f"{label}: {split} has {count} rows, manifest says {manifest['rows'][split]}")
                rows[target] = rows.get(target, 0) + kept
            if dropped != len(drop):
                raise ValueError(f"{label}: {len(drop)} ids listed, {dropped} rows left out")
            for key, value in manifest.get("excluded_field_counts", {}).items():
                excluded[key] = excluded.get(key, 0) + value
            sources += [dict(entry, subset=label) for entry in manifest.get("source_annotations", [])]
            merged_from.append({"subset": label, "manifest_sha256": digest(source / "MANIFEST.json"),
                                "rows": manifest["rows"], "rows_left_out": dropped,
                                "left_out_ids_sha256": digest(ids_file) if ids_file is not None else None,
                                "redistribution": manifest.get("redistribution")})
    finally:
        for handle in handles.values():
            handle.close()
    files = [{"path": f"{split}.jsonl", "sha256": digest(output / f"{split}.jsonl"),
              "bytes": (output / f"{split}.jsonl").stat().st_size} for split in sorted(rows)]
    files += [images[key] for key in sorted(images)]
    manifest = {
        "schema_version": 1, "subset": name, "stage": stage, "release_ready": False,
        "training_annotation_identity_verified": False, "training_media_identity_verified": False,
        "training_media_manifest_sha256": None, "profile_sha256": None,
        "source_annotations": sources, "rows": rows, "excluded_field_counts": excluded,
        "redistribution": {item["subset"]: item["redistribution"] for item in merged_from},
        "merged_from": merged_from,
        "record": {"who": None, "when_utc": datetime.now(timezone.utc).isoformat(), "gpus": 0,
                   "wall_seconds": round(time.monotonic() - start, 3),
                   "from": "merged_from (verified subset exports)",
                   "how": {"script": "scripts/merge_data_release.py", "script_sha256": digest(Path(__file__)),
                           "merge_val_into_train": merge_val},
                   "out": ".", "notes": "Rows unchanged except the added subset and source_split fields."},
        "files": files,
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True, help="Name of the merged corpus")
    parser.add_argument("--stage", required=True, choices=["pretext", "instruct"])
    parser.add_argument("--merge-val-into-train", action="store_true")
    parser.add_argument("inputs", nargs="+", help="LABEL=EXPORT_DIR[@IDS_FILE], in output order")
    args = parser.parse_args()
    pairs = []
    for item in args.inputs:
        label, sep, rest = item.partition("=")
        path, _, ids = rest.partition("@")
        if not sep or not label or not path:
            parser.error("Inputs must be LABEL=EXPORT_DIR[@IDS_FILE]")
        pairs.append((label, Path(path), Path(ids) if ids else None))
    try:
        result = merge(args.output, args.name, args.stage, pairs, args.merge_val_into_train)
    except (ValueError, OSError) as exc:
        print(str(exc) if isinstance(exc, ValueError) else type(exc).__name__, file=sys.stderr)
        return 1
    print(json.dumps({"subset": result["subset"], "rows": result["rows"],
                      "images": sum(1 for f in result["files"] if f["path"].startswith("images/"))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
