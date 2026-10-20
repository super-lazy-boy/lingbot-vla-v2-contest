import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location("extract_data", TOOLS / "extract_data.py")
extract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extract)


def make_manifest():
    files = []
    total = 0
    for i in range(50):
        task = f"task_{i:02d}"
        payload = b"x"
        files.append({"task": task, "path": f"dataset/{task}/demo_clean.zip",
                      "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
        total += len(payload)
    return {"schema_version": 1, "repo_id": "TianxingChen/RoboTwin2.0",
            "setting": "demo_clean", "expected_tasks": 50, "episodes_per_task": 50,
            "expected_training_episodes": 2500, "revision": "a" * 40,
            "files": files, "total_bytes": total}


class ExtractDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "raw" / "task_00").mkdir(parents=True)
        self.zip_path = self.root / "raw" / "task_00" / "demo_clean.zip"
        self.manifest = make_manifest()

    def write_zip(self, entries):
        with zipfile.ZipFile(self.zip_path, "w") as zf:
            for name, data, is_link in entries:
                info = zipfile.ZipInfo(name)
                if is_link:
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                zf.writestr(info, data)

    def dest(self):
        return self.root / "extracted" / "task_00"

    def test_rejects_absolute_path_entry(self):
        self.write_zip([("/abs/evil.txt", b"x", False)])
        with self.assertRaisesRegex(ValueError, "absolute|outside|unsafe"):
            extract.extract_archive(self.zip_path, self.dest())

    def test_rejects_parent_traversal_entry(self):
        self.write_zip([("../evil.txt", b"x", False)])
        with self.assertRaisesRegex(ValueError, "traversal|outside|unsafe"):
            extract.extract_archive(self.zip_path, self.dest())

    def test_rejects_nested_parent_traversal_entry(self):
        self.write_zip([("nested/../../evil.txt", b"x", False)])
        with self.assertRaisesRegex(ValueError, "traversal|outside|unsafe"):
            extract.extract_archive(self.zip_path, self.dest())

    def test_rejects_escaping_symlink_entry(self):
        self.write_zip([("link.txt", b"../../etc/passwd", True)])
        with self.assertRaisesRegex(ValueError, "symlink target escapes"):
            extract.extract_archive(self.zip_path, self.dest())

    def test_rejects_absolute_target_symlink_entry(self):
        self.write_zip([("link.txt", b"/etc/passwd", True)])
        with self.assertRaisesRegex(ValueError, "symlink target escapes"):
            extract.extract_archive(self.zip_path, self.dest())

    def test_allows_symlink_pointing_inside_tree(self):
        self.write_zip([
            ("dataset/adjust_bottle/real.json", b"{}", False),
            ("dataset/adjust_bottle/scene_info.json", b"real.json", True),
        ])
        extract.extract_archive(self.zip_path, self.dest())
        link = self.dest() / "dataset/adjust_bottle/scene_info.json"
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.read_bytes(), b"{}")

    def test_regular_file_with_high_mode_bits_not_mistaken_for_symlink(self):
        # Real dataset zips store st_mode (S_IFREG|0o644) << 16; bit 31 is set
        # there and must not trigger the Windows reparse-point check.
        payload = b'{"episode_0": {}}'
        with zipfile.ZipFile(self.zip_path, "w") as zf:
            info = zipfile.ZipInfo("dataset/adjust_bottle/scene_info.json")
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            zf.writestr(info, payload)
        extract.extract_archive(self.zip_path, self.dest())
        out = self.dest() / "dataset/adjust_bottle/scene_info.json"
        self.assertTrue(out.is_file())
        self.assertEqual(out.read_bytes(), payload)

    def test_extracts_regular_archive(self):
        self.write_zip([
            ("dataset/task_00/demo_clean/", b"", False),
            ("dataset/task_00/demo_clean/episode_0.h5", b"hdf5-bytes", False),
        ])
        count = extract.extract_archive(self.zip_path, self.dest())
        out = self.dest() / "dataset/task_00/demo_clean/episode_0.h5"
        self.assertTrue(out.is_file())
        self.assertEqual(out.read_bytes(), b"hdf5-bytes")
        self.assertGreaterEqual(count, 1)

    def test_overwrites_existing_files(self):
        self.write_zip([("episode_0.h5", b"old", False)])
        extract.extract_archive(self.zip_path, self.dest())
        self.write_zip([("episode_0.h5", b"new", False)])
        extract.extract_archive(self.zip_path, self.dest())
        self.assertEqual((self.dest() / "episode_0.h5").read_bytes(), b"new")

    def test_cli_rejects_unknown_task(self):
        manifest_path = self.root / "data_manifest.json"
        manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        # Convention from prepare_data.py: main() raises, the __main__ block
        # converts ValueError/OSError/RuntimeError into a nonzero exit code.
        with self.assertRaisesRegex(ValueError, "Unknown tasks"):
            extract.main(["--root", str(self.root), "--manifest", str(manifest_path),
                          "--task", "not_a_task"])


if __name__ == "__main__":
    unittest.main()
