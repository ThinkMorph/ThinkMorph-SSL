import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import download_release


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.payload = b"synthetic shard"
        self.output = self.root / "downloads"
        self.index = self.root / "SHARDS.json"
        self.record = {"path": "data-00000.tar", "bytes": len(self.payload),
                       "sha256": hashlib.sha256(self.payload).hexdigest()}
        self.write_index()

    def tearDown(self):
        self.temp.cleanup()

    def write_index(self):
        self.index.write_text(json.dumps({"files": [self.record]}))

    def run_main(self, stream=None, base="https://example.org/pinned-revision"):
        argv = ["download_release.py", str(self.index), "--base-url", base,
                "--output", str(self.output)]
        with patch.object(sys, "argv", argv), patch.object(
            download_release.urllib.request, "urlopen",
            return_value=io.BytesIO(self.payload) if stream is None else stream
        ) as request, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = download_release.main()
            return result, request.call_count

    def test_download_checks_bytes_and_hash(self):
        result, calls = self.run_main()
        self.assertEqual((result, calls), (0, 1))
        self.assertEqual((self.output / self.record["path"]).read_bytes(), self.payload)
        self.assertEqual(len(list(self.output.iterdir())), 1)

    def test_checksum_failure_cleans_only_its_temporary(self):
        result, _ = self.run_main(io.BytesIO(b"changed content"))
        self.assertEqual(result, 1)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_verified_existing_file_is_not_downloaded_again(self):
        self.output.mkdir()
        (self.output / self.record["path"]).write_bytes(self.payload)
        self.assertEqual(self.run_main(), (0, 0))

    def test_different_existing_file_is_not_overwritten(self):
        self.output.mkdir()
        path = self.output / self.record["path"]
        path.write_bytes(b"keep me")
        with self.assertRaises(SystemExit):
            self.run_main()
        self.assertEqual(path.read_bytes(), b"keep me")

    def test_traversal_is_rejected(self):
        self.record["path"] = "../escape"
        self.write_index()
        with self.assertRaises(SystemExit):
            self.run_main()
        self.assertFalse((self.root / "escape").exists())

    def test_signed_base_url_is_not_accepted_as_public_release(self):
        with self.assertRaises(SystemExit):
            self.run_main(base="https://example.org/data?signature=example")


    def test_token_is_sent_only_to_hugging_face(self):
        with patch.dict("os.environ", {"HF_TOKEN": "hf_example"}):
            argv = ["download_release.py", str(self.index), "--base-url",
                    "https://huggingface.co/datasets/org/repo/resolve/abc", "--output", str(self.output)]
            with patch.object(sys, "argv", argv), patch.object(
                download_release.urllib.request, "urlopen", return_value=io.BytesIO(self.payload)
            ) as request, patch.object(download_release.urllib.request, "install_opener"), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(download_release.main(), 0)
            self.assertEqual(request.call_args[0][0].get_header("Authorization"), "Bearer hf_example")
        other = download_release.request_for("https://example.org/x", "hf_example")
        self.assertIsNone(other.get_header("Authorization"))

    def test_token_is_dropped_on_redirect_to_another_host(self):
        handler = download_release.DropAuthOnHostChange()
        start = download_release.request_for("https://huggingface.co/datasets/o/r/resolve/c/f", "hf_example")
        same = handler.redirect_request(start, None, 302, "Found", {}, "https://huggingface.co/other")
        moved = handler.redirect_request(start, None, 302, "Found", {}, "https://cdn.example.org/f")
        self.assertEqual(same.get_header("Authorization"), "Bearer hf_example")
        self.assertIsNone(moved.get_header("Authorization"))


if __name__ == "__main__":
    unittest.main()
