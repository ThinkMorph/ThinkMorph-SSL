#!/usr/bin/env python3
"""Create deterministic data shards and restore them with checksum validation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from prepare_data_release import digest, verify


def pack(source: Path, output: Path, max_bytes: int):
    manifest = verify(source)
    if max_bytes < 10240:
        raise ValueError("Shard target must be at least 10240 bytes")
    if output.exists():
        raise ValueError("Output already exists")
    output.mkdir(parents=True)
    paths = ["MANIFEST.json", *[x["path"] for x in manifest["files"]]]
    archive = None
    shards = []
    current_size = 0
    current_name = None
    try:
        for relative in paths:
            path = source / relative
            size = path.stat().st_size
            if archive is None or (current_size and current_size + size + 512 > max_bytes):
                if archive is not None:
                    archive.close()
                current_name = f"data-{len(shards):05d}.tar"
                shards.append(current_name)
                archive = tarfile.open(output / current_name, "w", format=tarfile.PAX_FORMAT)
                current_size = 0
            info = tarfile.TarInfo(relative)
            info.size, info.mode, info.mtime = size, 0o644, 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)
            current_size += ((size + 511) // 512 + 1) * 512
    finally:
        if archive is not None:
            archive.close()
    index = {"schema_version": 1, "subset": manifest["subset"],
             "release_ready": manifest["release_ready"],
             "manifest_sha256": digest(source / "MANIFEST.json"),
             "shard_target_bytes": max_bytes,
             "files": [{"path": name, "bytes": (output / name).stat().st_size,
                        "sha256": digest(output / name)} for name in shards]}
    (output / "SHARDS.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def restore(index_path: Path, shard_root: Path, output: Path):
    index = json.loads(index_path.read_text())
    if output.exists():
        raise ValueError("Output already exists")
    # Check every shard before extracting any of them.
    checked = []
    names = set()
    for entry in index["files"]:
        name = entry["path"]
        if Path(name).name != name or name in names:
            raise ValueError("Unsafe or duplicate shard name")
        names.add(name)
        path = shard_root / name
        if path.is_symlink() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError("Shard integrity check failed")
        checked.append(path)
    output.mkdir(parents=True)
    seen = set()
    for path in checked:
        with tarfile.open(path, "r|") as archive:
            for member in archive:
                relative = Path(member.name)
                if not member.isfile() or relative.is_absolute() or ".." in relative.parts or "\\" in member.name:
                    raise ValueError("Unsafe archive entry")
                if member.name in seen or not member.name:
                    raise ValueError("Duplicate or empty archive entry")
                seen.add(member.name)
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as stream, target.open("xb") as dest:
                    shutil.copyfileobj(stream, dest, length=1024 * 1024)
    if digest(output / "MANIFEST.json") != index["manifest_sha256"]:
        raise ValueError("Restored manifest hash mismatch")
    result = verify(output)
    if result["release_ready"] != index["release_ready"] or result["subset"] != index["subset"]:
        raise ValueError("Index and manifest disagree")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--max-mib", type=int, default=512)
    r = sub.add_parser("restore")
    r.add_argument("index", type=Path)
    r.add_argument("shard_root", type=Path)
    r.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "pack":
        result = pack(args.source, args.output, args.max_mib * 1024 * 1024)
        print(json.dumps({"subset": result["subset"], "shards": len(result["files"])}))
    else:
        result = restore(args.index, args.shard_root, args.output)
        print(json.dumps({"subset": result["subset"], "rows": result["rows"]}))


if __name__ == "__main__":
    main()
