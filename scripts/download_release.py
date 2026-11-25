#!/usr/bin/env python3
"""Fetch checksummed release files one at a time from an HTTPS base URL.

For a gated Hugging Face repository, accept its terms on the website and set HF_TOKEN; the
token is sent only to huggingface.co and is dropped when a download redirects to another host.
"""
import argparse
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request
import urllib.error
import sys
import tempfile
import re
from prepare_data_release import digest

HF_HOSTS = {"huggingface.co", "hf.co"}


class DropAuthOnHostChange(urllib.request.HTTPRedirectHandler):
    """Never forward the token to another host (for example a storage CDN)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).hostname != urllib.parse.urlsplit(req.full_url).hostname:
            new.remove_header("Authorization")
        return new


def request_for(url, token):
    request = urllib.request.Request(url)
    if token and urllib.parse.urlsplit(url).hostname in HF_HOSTS:
        request.add_header("Authorization", f"Bearer {token}")
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path, help="Reviewed local SHARDS.json or model file manifest")
    parser.add_argument("--base-url", required=True,
                        help="HTTPS directory, preferably pinned to a repository commit (gated HF repos: set HF_TOKEN)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = urllib.parse.urlsplit(args.base_url)
    if base.scheme != "https" or base.query or base.fragment or base.username or base.password:
        parser.error("Use a public unsigned HTTPS directory URL")
    token = os.environ.get("HF_TOKEN") or None
    if token:
        urllib.request.install_opener(urllib.request.build_opener(DropAuthOnHostChange))
    manifest = json.loads(args.index.read_text())
    for entry in manifest["files"]:
        if not re.fullmatch(r"[a-f0-9]{64}", entry.get("sha256", "")):
            parser.error("Every file needs a SHA256 checksum")
        if not isinstance(entry.get("bytes"), int) or entry["bytes"] < 0:
            parser.error("Every file needs a nonnegative byte count")
    args.output.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in entry["path"]:
            parser.error("Manifest contains an unsafe path")
        target = args.output / relative
        if not target.resolve().is_relative_to(args.output.resolve()):
            parser.error("Download path escapes output")
        if target.exists():
            if target.stat().st_size != entry["bytes"] or digest(target) != entry["sha256"]:
                parser.error("An existing file differs; use a new output directory")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        url = args.base_url.rstrip("/") + "/" + urllib.parse.quote(entry["path"], safe="/")
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".download-", delete=False) as dest:
                temporary = Path(dest.name)
                size = 0
                with urllib.request.urlopen(request_for(url, token), timeout=120) as stream:
                    while block := stream.read(1024 * 1024):
                        size += len(block)
                        if size > entry["bytes"]:
                            raise ValueError("Download exceeds declared size")
                        dest.write(block)
            if size != entry["bytes"] or digest(temporary) != entry["sha256"]:
                raise ValueError("Download checksum mismatch")
            if target.exists():
                raise ValueError("Destination appeared during download")
            temporary.rename(target)
        except Exception as exc:
            # A URL-bearing exception may contain sensitive redirect parameters.
            print("Download failed: " + type(exc).__name__, file=sys.stderr)
            return 1
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print("All release files verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
