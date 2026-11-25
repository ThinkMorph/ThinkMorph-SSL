#!/usr/bin/env python3
"""Reorganize a merged release (merge_data_release.py output) into one folder per task.

Input: the merged release (train.jsonl, val.jsonl when kept, images/, MANIFEST.json; rows carry
`subset` and `source_split`) and a task map: JSONL lines with `id` and `task`, where a task is a
relative folder name such as `counting` or `spatial/counting`.

Output layout:

    MANIFEST.json
    images/<sha256>.<ext>          shared, content-addressed (unchanged files)
    <task>/train.jsonl             one folder per task; val.jsonl where the input has val rows

Each row drops `subset` and `source_split` and gains `task`. Its image references become
relative to its own folder (`../images/...`, `../../images/...`), which is how the training
loader resolves media for a dataset folder. Text and ids are unchanged. MANIFEST.json counts rows
per JSONL path; verify() in prepare_data_release.py checks this layout. With --move-images the
input images/ folder is renamed into the output (no copy; the input is no longer a complete
release afterwards); otherwise images are hard-linked (copied when linking fails). With
--drop-empty-text-fields, `question`, `options`, `answer` and `gt_answer` are left out of a row
when they are null or blank; the training loader reads a missing field and an empty one the
same way (no question text, no options, no answer).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

from prepare_data_release import digest, image_list

IMAGE = re.compile(r"reasoning_image_\d+$")
TASK = re.compile(r"[a-z0-9_]+(/[a-z0-9_]+)*")
TEXT_FIELDS = ("question", "options", "answer", "gt_answer")


def organize(source: Path, output: Path, task_map: Path, move_images: bool,
             drop_empty_text_fields: bool = False) -> dict:
    start = time.monotonic()
    if output.exists():
        raise ValueError("Output already exists; choose a new release directory")
    manifest = json.loads((source / "MANIFEST.json").read_text())
    listed = {e["path"]: e for e in manifest["files"] if e["path"].startswith("images/")}
    tasks = {}
    with task_map.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rec = json.loads(line)
                if not TASK.fullmatch(rec["task"]) or rec["task"] == "images":
                    raise ValueError(f"Unsafe task name {rec['task']!r}")
                tasks[rec["id"]] = rec["task"]
    output.mkdir(parents=True)
    handles, rows, used, dropped = {}, {}, set(), {}
    try:
        for split in sorted(manifest["rows"]):
            count = 0
            with (source / f"{split}.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    count += 1
                    task = tasks.get(row["id"])
                    if task is None:
                        raise ValueError(f"Row {row['id'][:12]} has no task in the map")
                    up = "../" * (task.count("/") + 1)
                    out = {"id": row["id"], "task": task}
                    for key, value in row.items():
                        if key in ("id", "subset", "source_split", "task"):
                            continue
                        if (drop_empty_text_fields and key in TEXT_FIELDS
                                and (value is None or (isinstance(value, str) and not value.strip()))):
                            dropped[key] = dropped.get(key, 0) + 1
                            continue
                        if key == "images":
                            refs = image_list(value)
                            out[key] = [up + ref for ref in refs]
                        elif IMAGE.fullmatch(key) and value:
                            refs = [value]
                            out[key] = up + value
                        else:
                            out[key] = value
                            continue
                        for ref in refs:
                            if ref not in listed:
                                raise ValueError(f"{ref} is not in the input manifest")
                            used.add(ref)
                    path = f"{task}/{split}.jsonl"
                    if path not in handles:
                        (output / task).mkdir(parents=True, exist_ok=True)
                        handles[path] = (output / path).open("w", encoding="utf-8", newline="\n")
                    handles[path].write(json.dumps(out, ensure_ascii=False) + "\n")
                    rows[path] = rows.get(path, 0) + 1
            if count != manifest["rows"][split]:
                raise ValueError(f"{split} has {count} rows, manifest says {manifest['rows'][split]}")
    finally:
        for handle in handles.values():
            handle.close()
    if used != set(listed):
        raise ValueError(f"{len(set(listed) - used)} listed images are not referenced by any row")
    if move_images:
        os.rename(source / "images", output / "images")
    else:
        (output / "images").mkdir()
        for ref in sorted(listed):
            try:
                os.link(source / ref, output / ref)
            except OSError:
                shutil.copyfile(source / ref, output / ref)
    files = [{"path": p, "sha256": digest(output / p), "bytes": (output / p).stat().st_size} for p in sorted(rows)]
    files += [listed[ref] for ref in sorted(listed)]
    per_task = {}
    for path, n in rows.items():
        task, split = path.rsplit("/", 1)
        per_task.setdefault(task, {})[split.removesuffix(".jsonl")] = n
    result = {
        "schema_version": 1, "subset": manifest["subset"], "stage": manifest["stage"],
        "release_ready": manifest["release_ready"], "layout": "tasks",
        "rows": dict(sorted(rows.items())), "tasks": dict(sorted(per_task.items())),
        "empty_text_fields_left_out": dict(sorted(dropped.items())),
        "organized_from": {"manifest_sha256": digest(source / "MANIFEST.json"),
                           "task_map_sha256": digest(task_map)},
        "record": {"when_utc": datetime.now(timezone.utc).isoformat(), "gpus": 0,
                   "wall_seconds": round(time.monotonic() - start, 3),
                   "how": {"script": "scripts/organize_release_by_task.py",
                           "script_sha256": digest(Path(__file__)), "move_images": move_images,
                           "drop_empty_text_fields": drop_empty_text_fields}},
        "files": files,
    }
    (output / "MANIFEST.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="merged release directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--task-map", type=Path, required=True, help="JSONL with id and task per row")
    parser.add_argument("--move-images", action="store_true")
    parser.add_argument("--drop-empty-text-fields", action="store_true")
    args = parser.parse_args()
    try:
        result = organize(args.source, args.output, args.task_map, args.move_images, args.drop_empty_text_fields)
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"organize failed: {type(error).__name__}: {error}")
    print(json.dumps({"subset": result["subset"], "tasks": len(result["tasks"]),
                      "rows": sum(result["rows"].values()), "images": len(result["files"]) - len(result["rows"])}))


if __name__ == "__main__":
    main()
