"""Small offline fixtures test preservation, integrity, and safe restoration."""
import base64
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from prepare_data_release import contained, digest, export, normalize_row, verify
from shard_data_release import pack, restore
from merge_data_release import merge
from organize_release_by_task import organize


class DataReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = self.root / "source.png"
        self.image.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a5gAAAABJRU5ErkJggg=="))
        self.row = {"id": "internal-id", "question": "Which side?\nKeep this text.",
                    "options": {"B": "right", "A": "left"}, "images": ["source.png"],
                    "reasoning_thought_0": " Inspect the scene. ",
                    "reasoning_image_0": "source.png",
                    "reasoning_ref_image_0": "input_0",
                    "reasoning_thought_1": "The object is left.",
                    "gt_answer": " A ", "answer": "legacy answer",
                    "metadata": {"original_image": "/private/internal/path"}}
        self.annotation = self.root / "source.jsonl"
        self.annotation.write_text(json.dumps(self.row) + "\n")
        self.profile = self.root / "profile.json"
        self.settings = {"subset": "fixture", "stage": "instruct", "producer": "offline test",
                         "annotations": [{"path": "source.jsonl", "split": "train",
                                          "sha256": digest(self.annotation), "expected_rows": 1}],
                         "redistribution": {"status": "pending", "source_terms": []}}
        self.save_profile()

    def tearDown(self):
        self.temp.cleanup()

    def save_profile(self):
        self.profile.write_text(json.dumps(self.settings))

    def test_export_preserves_supervision_and_deduplicates_media(self):
        out = self.root / "out"
        result = export(self.profile, self.root, out)
        row = json.loads((out / "train.jsonl").read_text())
        for key in ("question", "options", "gt_answer", "answer", "reasoning_thought_0", "reasoning_thought_1"):
            self.assertEqual(row[key], self.row[key])
        self.assertEqual(str(row["options"]), str(self.row["options"]))
        self.assertEqual(row["images"][0], row["reasoning_image_0"])
        self.assertEqual(row["reasoning_ref_image_0"], "input_0")
        self.assertNotIn("metadata", row)
        self.assertNotEqual(row["id"], self.row["id"])
        self.assertEqual(len(list((out / "images").iterdir())), 1)
        self.assertFalse(result["release_ready"])
        self.assertEqual(verify(out)["rows"], {"train": 1})

    def test_merge_keeps_rows_adds_labels_and_drops_listed_rows(self):
        first = self.root / "first"
        export(self.profile, self.root, first)
        second_image = self.root / "other.png"
        second_image.write_bytes(self.image.read_bytes() + b"\0")  # different bytes, still a file
        other = dict(self.row, images=["other.png"], reasoning_image_0="other.png", question="Other?")
        self.annotation.write_text(json.dumps(self.row) + "\n" + json.dumps(other) + "\n")
        self.settings["annotations"][0].update(sha256=digest(self.annotation), expected_rows=2)
        self.settings["subset"] = "fixture-two"
        self.save_profile()
        second = self.root / "second"
        export(self.profile, self.root, second)
        rows_second = [json.loads(line) for line in (second / "train.jsonl").read_text().splitlines()]
        drop = self.root / "drop.txt"
        drop.write_text(rows_second[1]["id"] + "\n")
        out = self.root / "merged"
        merge(out, "fixture-all", "instruct", [("a", first, None), ("b", second, drop)], merge_val=False)
        manifest = verify(out)
        merged = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
        self.assertEqual(manifest["rows"], {"train": 2})
        self.assertEqual([r["subset"] for r in merged], ["a", "b"])
        self.assertEqual({r["source_split"] for r in merged}, {"train"})
        self.assertEqual(merged[1]["question"], self.row["question"])
        self.assertEqual(len(list((out / "images").iterdir())), 1)  # shared image stored once, dropped row's image left out
        with self.assertRaises(ValueError):
            merge(self.root / "bad", "x", "instruct", [("a", first, None), ("a", second, None)], merge_val=False)

    def test_organize_by_task_keeps_rows_and_round_trips(self):
        first = self.root / "first"
        export(self.profile, self.root, first)
        other_image = self.root / "other.png"
        other_image.write_bytes(self.image.read_bytes() + b"\0")
        other = dict(self.row, images=["other.png"], reasoning_image_0="other.png", question="Other?", options="")
        self.annotation.write_text(json.dumps(other) + "\n")
        self.settings["annotations"][0].update(sha256=digest(self.annotation), expected_rows=1)
        self.settings["subset"] = "fixture-two"
        self.save_profile()
        second = self.root / "second"
        export(self.profile, self.root, second)
        merged = self.root / "merged"
        merge(merged, "fixture-all", "instruct", [("a_1k", first, None), ("b_2k", second, None)], merge_val=False)
        rows = [json.loads(line) for line in (merged / "train.jsonl").read_text().splitlines()]
        task_map = self.root / "tasks.jsonl"
        task_map.write_text("".join(json.dumps({"id": r["id"], "task": t}) + "\n"
                                    for r, t in zip(rows, ["spatial/counting", "perception/search"])))
        kept = self.root / "by_task_kept"
        organize(merged, kept, task_map, move_images=False)
        self.assertEqual(json.loads((kept / "perception/search/train.jsonl").read_text())["options"], "")
        out = self.root / "by_task"
        organize(merged, out, task_map, move_images=False, drop_empty_text_fields=True)
        manifest = verify(out)
        self.assertNotIn("options", json.loads((out / "perception/search/train.jsonl").read_text()))
        self.assertEqual(manifest["empty_text_fields_left_out"], {"options": 1})
        self.assertEqual(manifest["rows"], {"perception/search/train.jsonl": 1, "spatial/counting/train.jsonl": 1})
        row = json.loads((out / "spatial/counting/train.jsonl").read_text())
        self.assertEqual(row["task"], "spatial/counting")
        self.assertNotIn("subset", row)
        self.assertNotIn("source_split", row)
        self.assertEqual(row["question"], self.row["question"])
        self.assertEqual(row["id"], rows[0]["id"])
        self.assertTrue(row["images"][0].startswith("../../images/"))
        self.assertTrue((out / "spatial/counting" / row["images"][0]).is_file())
        pack(out, self.root / "shards", 10240)
        self.assertEqual(restore(self.root / "shards" / "SHARDS.json", self.root / "shards", self.root / "back")["rows"],
                         manifest["rows"])
        bad = dict(row, images=["../../../source.png"])
        (out / "spatial/counting/train.jsonl").write_text(json.dumps(bad) + "\n")
        entry = next(e for e in manifest["files"] if e["path"] == "spatial/counting/train.jsonl")
        entry.update(sha256=digest(out / entry["path"]), bytes=(out / entry["path"]).stat().st_size)
        (out / "MANIFEST.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            verify(out)
        with self.assertRaises(ValueError):
            task_map.write_text(json.dumps({"id": rows[0]["id"], "task": "../escape"}) + "\n")
            organize(merged, self.root / "bad", task_map, move_images=False)

    def test_release_readiness_requires_annotation_media_and_terms(self):
        self.settings["annotations"][0]["training_sha256"] = digest(self.annotation)
        self.settings["redistribution"] = {"status": "cleared", "source_terms": ["Synthetic test fixture"]}
        self.save_profile()
        self.assertFalse(export(self.profile, self.root, self.root / "without_media")["release_ready"])
        media = self.root / "media.json"
        media.write_text(json.dumps({"source.png": {"sha256": digest(self.image), "bytes": self.image.stat().st_size}}))
        self.settings["training_media_manifest"] = {"path": "media.json", "sha256": digest(media)}
        self.save_profile()
        self.assertTrue(export(self.profile, self.root, self.root / "with_media")["release_ready"])

    def test_source_annotation_drift_rejected(self):
        self.annotation.write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "hash"):
            export(self.profile, self.root, self.root / "bad")

    def test_training_media_drift_rejected(self):
        media = self.root / "media.json"
        media.write_text(json.dumps({"source.png": {"sha256": "0" * 64, "bytes": self.image.stat().st_size}}))
        self.settings["training_media_manifest"] = {"path": "media.json", "sha256": digest(media)}
        self.save_profile()
        with self.assertRaisesRegex(ValueError, "training inventory"):
            export(self.profile, self.root, self.root / "bad")

    def test_empty_instruction_answer_not_silently_dropped(self):
        self.row["answer"], self.row["gt_answer"] = "", ""
        with self.assertRaisesRegex(ValueError, "answer"):
            normalize_row(self.row, "instruct", lambda p: p, "fixture", 1)

    def test_pretext_without_question_is_valid(self):
        self.row["question"], self.row["answer"], self.row["gt_answer"] = "", "", ""
        self.row.pop("options")
        row = normalize_row(self.row, "pretext", lambda p: p, "fixture", 1)
        self.assertEqual(row["question"], "")

    def test_pretext_options_cannot_silently_add_a_question(self):
        self.row["question"] = ""
        with self.assertRaisesRegex(ValueError, "question/options"):
            normalize_row(self.row, "pretext", lambda p: p, "fixture", 1)

    def test_symbolic_reference_must_resolve(self):
        self.row["reasoning_ref_image_0"] = "input_7"
        with self.assertRaisesRegex(ValueError, "absent"):
            normalize_row(self.row, "instruct", lambda p: p, "fixture", 1)

    def test_path_traversal_and_escaping_symlink_rejected(self):
        with self.assertRaises(ValueError):
            contained(self.root, "../outside.png")
        (self.root / "link.png").symlink_to("/etc/hosts")
        with self.assertRaises(ValueError):
            contained(self.root, "link.png")

    def test_corrupted_export_and_unlisted_file_rejected(self):
        out = self.root / "out"
        export(self.profile, self.root, out)
        (out / "extra.txt").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "Unlisted"):
            verify(out)
        (out / "extra.txt").unlink()
        (out / "train.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "integrity"):
            verify(out)

    def test_deterministic_shards_and_verified_restore(self):
        out = self.root / "out"
        export(self.profile, self.root, out)
        a, b = self.root / "shards_a", self.root / "shards_b"
        ia, ib = pack(out, a, 10240), pack(out, b, 10240)
        self.assertEqual(ia, ib)
        recovered = self.root / "recovered"
        restore(a / "SHARDS.json", a, recovered)
        self.assertEqual((out / "train.jsonl").read_bytes(), (recovered / "train.jsonl").read_bytes())

    def test_corrupted_shard_rejected_before_extraction(self):
        out, shards = self.root / "out", self.root / "shards"
        export(self.profile, self.root, out)
        index = pack(out, shards, 10240)
        (shards / index["files"][0]["path"]).write_bytes(b"broken")
        destination = self.root / "restore"
        with self.assertRaisesRegex(ValueError, "integrity"):
            restore(shards / "SHARDS.json", shards, destination)
        self.assertFalse(destination.exists())

    def test_unsafe_archive_rejected_even_with_matching_checksum(self):
        shard = self.root / "evil.tar"
        with tarfile.open(shard, "w") as archive:
            item = tarfile.TarInfo("../escaped.txt")
            item.size = 1
            archive.addfile(item, io.BytesIO(b"x"))
        index = {"files": [{"path": shard.name, "bytes": shard.stat().st_size, "sha256": digest(shard)}]}
        path = self.root / "index.json"
        path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            restore(path, self.root, self.root / "restored")
        self.assertFalse((self.root / "escaped.txt").exists())


if __name__ == "__main__":
    unittest.main()
