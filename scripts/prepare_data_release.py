#!/usr/bin/env python3
"""Normalize a locally staged paper subset without changing its supervision.

This is a release-format exporter, not a reconstruction of the original data
generation pipelines. It never downloads data or silently drops invalid rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import posixpath
import re
import shutil
import sys
import time


TEXT_FIELDS = {"question", "options", "answer", "gt_answer"}
THOUGHT = re.compile(r"reasoning_thought_\d+$")
IMAGE = re.compile(r"reasoning_image_\d+$")
REFERENCE = re.compile(r"reasoning_ref_image_\d+$")
SHA256 = re.compile(r"[a-f0-9]{64}$")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def contained(root: Path, relative: str) -> Path:
    """Reject absolute paths, traversal, URL paths, and escaping symlinks."""
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("Image paths must be nonempty relative strings")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or "://" in relative:
        raise ValueError("Image path is not portable")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError("Image is missing or escapes its declared media root")
    return target


def image_list(value):
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value
        value = decoded if isinstance(decoded, list) else [value]
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v.strip() for v in value):
        raise ValueError("images must contain at least one image path")
    return [v.strip() for v in value]


def normalize_row(row: dict, stage: str, move_image, subset: str, line: int):
    if not isinstance(row, dict):
        raise ValueError("Each row must be an object")
    if any(key.startswith("_") for key in row):
        raise ValueError("Resolve loader-internal fields before export")
    output = {key: value for key, value in row.items()
              if key in TEXT_FIELDS or THOUGHT.fullmatch(key) or REFERENCE.fullmatch(key)}
    if any(value is not None and not isinstance(value, str) for key, value in output.items()
           if key != "options"):
        raise ValueError("Text and reference fields must be strings or null")
    # Preserve exact text and both answer fields; the loader prefers gt_answer.
    output["images"] = [move_image(path) for path in image_list(row.get("images"))]
    for key, value in row.items():
        if IMAGE.fullmatch(key):
            if value is not None and not isinstance(value, str):
                raise ValueError("Reasoning images must be paths or null")
            output[key] = move_image(value.strip()) if value and value.strip() else value
    if not any(IMAGE.fullmatch(k) and v for k, v in output.items()):
        raise ValueError("A release trace must contain a target reasoning image")
    if stage == "instruct":
        if not (str(row.get("question") or "").strip() or str(row.get("options") or "").strip()):
            raise ValueError("Instruction trace has no question/options")
        required = ("reasoning_thought_0", "reasoning_image_0")
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in required):
            raise ValueError("Instruction trace lacks the required first reasoning step")
        if not any(isinstance(row.get(k), str) and row[k].strip() for k in ("gt_answer", "answer")):
            raise ValueError("Instruction trace has no textual answer")
    elif str(row.get("question") or "").strip() or str(row.get("options") or "").strip():
        raise ValueError("Instruction-free pretext export has nonempty question/options")
    for key, value in output.items():
        if REFERENCE.fullmatch(key) and value:
            if not re.fullmatch(r"(input|reasoning_image)_\d+", value):
                raise ValueError("Image references must use symbolic image indices")
            if value.startswith("input_"):
                valid = int(value.split("_")[-1]) < len(output["images"])
            else:
                valid = bool(output.get(value))
            if not valid:
                raise ValueError("Image reference target is absent")
    # Public IDs are stable and do not expose internal source identifiers.
    original = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    output["id"] = hashlib.sha256((subset + "\n" + str(line) + "\n" + original).encode()).hexdigest()
    # Metadata is excluded by default because source rows can contain local paths.
    # Its original content remains bound by the source annotation checksum.
    return output


def export(profile_path: Path, media_root: Path, output: Path):
    start = time.monotonic()
    profile = json.loads(profile_path.read_text())
    if profile.get("stage") not in {"pretext", "instruct"}:
        raise ValueError("stage must be pretext or instruct")
    if not profile.get("subset") or not profile.get("annotations"):
        raise ValueError("subset and annotations are required")
    if output.exists():
        raise ValueError("Output already exists; choose a new release directory")
    media_root = media_root.resolve()
    media_manifest = {}
    media_record = profile.get("training_media_manifest")
    if media_record:
        media_path = (profile_path.parent / media_record["path"]).resolve()
        if digest(media_path) != media_record.get("sha256"):
            raise ValueError("Training media manifest hash mismatch")
        media_manifest = json.loads(media_path.read_text())
    sources = []
    for entry in profile["annotations"]:
        path = (profile_path.parent / entry["path"]).resolve()
        if not SHA256.fullmatch(entry.get("sha256", "")) or digest(path) != entry["sha256"]:
            raise ValueError("Source annotation hash is absent or mismatched")
        if entry.get("split") not in {"train", "val"}:
            raise ValueError("Each annotation needs train or val split")
        # Media root may differ for training/ and val/ layout.
        entry_root = media_root / entry.get("media_subdir", "")
        if not entry_root.resolve().is_relative_to(media_root):
            raise ValueError("Annotation media root escapes the supplied root")
        sources.append((entry, path, entry_root.resolve()))
    output.mkdir(parents=True)
    (output / "images").mkdir()
    counts = Counter()
    stripped = Counter()
    files = {}
    cached_paths = {}
    handles = {}
    try:
        for entry, path, entry_root in sources:
            split = entry["split"]
            if split not in handles:
                handles[split] = (output / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n")
            seen = 0

            def move_image(relative):
                source = contained(entry_root, relative)
                if source in cached_paths:
                    return cached_paths[source]
                sha = digest(source)
                source_key = source.relative_to(media_root).as_posix()
                if media_record:
                    expected = media_manifest.get(source_key)
                    if not expected or expected.get("sha256") != sha or expected.get("bytes") != source.stat().st_size:
                        raise ValueError("Media does not match the training inventory")
                suffix = source.suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    raise ValueError("Unsupported release image extension")
                target_rel = f"images/{sha}{suffix}"
                target = output / target_rel
                if target_rel not in files:
                    shutil.copyfile(source, target)
                    if digest(target) != sha:
                        raise ValueError("Image changed during export")
                    files[target_rel] = {"path": target_rel, "sha256": sha, "bytes": target.stat().st_size}
                cached_paths[source] = target_rel
                return target_rel

            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        normalized = normalize_row(row, profile["stage"], move_image,
                                                   profile["subset"] + ":" + split + ":" + entry["sha256"],
                                                   line_number)
                    except (ValueError, TypeError, KeyError) as exc:
                        # No row values, credentials, or internal image paths in errors.
                        raise ValueError(f"{split} line {line_number}: {type(exc).__name__}: {exc}") from None
                    for key in row.keys() - normalized.keys():
                        stripped[key] += 1
                    # The loader stringifies dict-valued options. Sorting nested
                    # keys would therefore alter the actual question text.
                    handles[split].write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    counts[split] += 1
                    seen += 1
            if digest(path) != entry["sha256"]:
                raise ValueError("Source annotation changed during export")
            if entry.get("expected_rows") is not None and seen != entry["expected_rows"]:
                raise ValueError("Source annotation row count mismatch")
    finally:
        for handle in handles.values():
            handle.close()
    for split in counts:
        path = output / f"{split}.jsonl"
        files[path.name] = {"path": path.name, "sha256": digest(path), "bytes": path.stat().st_size}
    if not counts or not sum(counts.values()):
        raise ValueError("Empty release")
    training_verified = all(entry.get("training_sha256") == entry["sha256"] for entry, _, _ in sources)
    rights = profile.get("redistribution", {})
    media_verified = bool(media_record and media_manifest)
    ready = training_verified and media_verified and rights.get("status") == "cleared" and bool(rights.get("source_terms"))
    manifest = {
        "schema_version": 1, "subset": profile["subset"], "stage": profile["stage"],
        "release_ready": ready, "training_annotation_identity_verified": training_verified,
        "training_media_identity_verified": media_verified,
        "training_media_manifest_sha256": media_record.get("sha256") if media_record else None,
        "profile_sha256": digest(profile_path),
        "source_annotations": [{"sha256": e["sha256"], "split": e["split"],
                                "training_sha256": e.get("training_sha256")} for e, _, _ in sources],
        "rows": dict(counts), "excluded_field_counts": dict(stripped),
        "redistribution": rights,
        "record": {"who": profile.get("producer"), "when_utc": datetime.now(timezone.utc).isoformat(),
                   "gpus": 0, "wall_seconds": round(time.monotonic() - start, 3),
                   "from": "source_annotations and profile_sha256",
                   "how": {"script": "scripts/prepare_data_release.py", "script_sha256": digest(Path(__file__))},
                   "out": ".", "notes": "No filtering, deduplication, answer rewriting, or image recompression. Source metadata is excluded."},
        "files": [files[key] for key in sorted(files)],
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(root: Path):
    manifest = json.loads((root / "MANIFEST.json").read_text())
    paths = set()
    for entry in manifest["files"]:
        if entry["path"] in paths:
            raise ValueError("Duplicate file in manifest")
        paths.add(entry["path"])
        path = contained(root, entry["path"])
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError("Release file failed integrity check")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != paths | {"MANIFEST.json"}:
        raise ValueError("Unlisted or missing files in release")
    for key, expected in manifest["rows"].items():
        # Split layout: key "train" -> train.jsonl at the root, image paths relative to the root.
        # Task layout: key "<task>/train.jsonl", image paths relative to that file's folder.
        relative = key if key.endswith(".jsonl") else f"{key}.jsonl"
        base = posixpath.dirname(relative)
        if relative not in paths:
            raise ValueError("Counted annotation file is not in the manifest")
        count = 0
        with contained(root, relative).open() as stream:
            for line in stream:
                row = json.loads(line)
                for value in image_list(row["images"]) + [v for k, v in row.items() if IMAGE.fullmatch(k) and v]:
                    target = posixpath.normpath(posixpath.join(base, value)) if base else value
                    if target not in paths or not target.startswith("images/"):
                        raise ValueError("Annotation references an unlisted image")
                count += 1
        if count != expected:
            raise ValueError("Release row count mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("export")
    build.add_argument("--profile", type=Path, required=True)
    build.add_argument("--media-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        result = export(args.profile, args.media_root, args.output) if args.command == "export" else verify(args.directory)
    except (ValueError, OSError) as exc:
        # OSError paths can contain private storage details; do not expose them.
        print(str(exc) if isinstance(exc, ValueError) else type(exc).__name__, file=sys.stderr)
        return 1
    print(json.dumps({k: result[k] for k in ("subset", "rows", "release_ready")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
