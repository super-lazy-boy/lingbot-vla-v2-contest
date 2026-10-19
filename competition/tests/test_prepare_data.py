import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location("prepare_data", TOOLS / "prepare_data.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.payload = b"fixed clean archive fixture"
        self.entry = {"task": "adjust_bottle", "path": "dataset/adjust_bottle/demo_clean.zip",
                      "size_bytes": len(self.payload), "sha256": hashlib.sha256(self.payload).hexdigest()}
        self.manifest = {"revision": "a" * 40}

    def tearDown(self):
        self.temp.cleanup()

    def test_committed_manifest_contains_only_fifty_pinned_clean_archives(self):
        manifest = prepare.load_manifest(prepare.DEFAULT_MANIFEST)
        self.assertEqual(len(manifest["files"]), 50)
        self.assertEqual(manifest["episodes_per_task"], 50)
        base = prepare.DEFAULT_MANIFEST.parent
        expected = "".join(entry["sha256"] + "  raw/" + entry["task"] + "/demo_clean.zip\n"
                           for entry in manifest["files"])
        self.assertEqual((base / "SHA256SUMS").read_text(encoding="utf-8"), expected)
        self.assertEqual((base / "tasks.txt").read_text(encoding="utf-8").splitlines(),
                         [entry["task"] for entry in manifest["files"]])

    def test_manifest_rejects_randomized_or_unsafe_paths(self):
        manifest = json.loads(prepare.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        for path in ("dataset/adjust_bottle/demo_randomized.zip", "../../escape.zip"):
            invalid = copy.deepcopy(manifest)
            invalid["files"][0]["path"] = path
            target = self.root / "invalid.json"
            target.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare.load_manifest(target)

    def test_hash_rejects_same_size_corruption(self):
        target = self.root / "bad.zip"
        target.write_bytes(b"x" * len(self.payload))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            prepare.verify_file(target, self.entry)

    def test_complete_download_is_verified_then_renamed(self):
        with patch.object(prepare, "urlopen", return_value=Response(self.payload)):
            prepare.download_file(self.manifest, self.root, self.entry, 10, 0)
        target = prepare.archive_path(self.root, self.entry)
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertFalse(target.with_suffix(".zip.partial").exists())

    def test_resume_requires_matching_content_range(self):
        partial = self.root / "test.partial"
        partial.write_bytes(self.payload[:5])
        headers = {"Content-Range": f"bytes 5-{len(self.payload)-1}/{len(self.payload)}"}
        with patch.object(prepare, "urlopen", return_value=Response(self.payload[5:], 206, headers)) as opener:
            prepare.fetch_once("https://example.invalid", partial, len(self.payload), 10)
            self.assertEqual(opener.call_args.args[0].get_header("Range"), "bytes=5-")
        self.assertEqual(partial.read_bytes(), self.payload)

    def test_server_ignoring_range_restarts_instead_of_appending(self):
        partial = self.root / "test.partial"
        partial.write_bytes(self.payload[:5])
        with patch.object(prepare, "urlopen", return_value=Response(self.payload)):
            prepare.fetch_once("https://example.invalid", partial, len(self.payload), 10)
        self.assertEqual(partial.read_bytes(), self.payload)

    def test_wrong_resume_range_does_not_modify_partial(self):
        partial = self.root / "test.partial"
        partial.write_bytes(self.payload[:5])
        headers = {"Content-Range": f"bytes 0-{len(self.payload)-1}/{len(self.payload)}"}
        with patch.object(prepare, "urlopen", return_value=Response(self.payload, 206, headers)):
            with self.assertRaisesRegex(ValueError, "Content-Range"):
                prepare.fetch_once("https://example.invalid", partial, len(self.payload), 10)
        self.assertEqual(partial.read_bytes(), self.payload[:5])

    def test_short_response_preserves_partial_for_resume(self):
        partial = self.root / "test.partial"
        with patch.object(prepare, "urlopen", return_value=Response(self.payload[:5])):
            with self.assertRaises(OSError):
                prepare.fetch_once("https://example.invalid", partial, len(self.payload), 10)
        self.assertEqual(partial.read_bytes(), self.payload[:5])

    def test_disk_space_check_includes_reserve(self):
        with patch.object(prepare.shutil, "disk_usage") as usage:
            usage.return_value.free = 100
            with self.assertRaisesRegex(ValueError, "Not enough disk space"):
                prepare.check_disk_space(self.root, 101, 0)

    def test_existing_verified_file_does_not_access_network(self):
        target = prepare.archive_path(self.root, self.entry)
        target.parent.mkdir(parents=True)
        target.write_bytes(self.payload)
        with patch.object(prepare, "urlopen") as opener:
            prepare.download_file(self.manifest, self.root, self.entry, 10, 0)
            opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
