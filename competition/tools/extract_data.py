#!/usr/bin/env python3
"""Extract pinned clean archives with zip-slip safety checks.

Rejects absolute paths, parent traversal (``..``), and symlink entries before
writing anything. Pure stdlib; run it with any Python 3.10+ interpreter.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import zipfile


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from prepare_data import archive_path, load_manifest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def validate_entry_name(name: str) -> str:
    """Return the normalized safe name, or raise ValueError for unsafe entries."""
    if name.startswith("/") or name.startswith("\\"):
        raise ValueError(f"absolute path entry: {name!r}")
    if ":" in name.split("/")[0]:
        raise ValueError(f"drive-relative entry: {name!r}")
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]
    if ".." in parts:
        raise ValueError(f"parent traversal entry: {name!r}")
    return "/".join(parts)


def is_symlink(info: zipfile.ZipInfo) -> bool:
    """Detect symlinks stored as Unix mode or the DOS reparse-point attribute.

    The Windows attribute check applies only to non-Unix entries: Unix
    entries put the full ``st_mode`` in the high 16 bits, so bits like
    0x80000000 are ordinary permission bits and must not be read as the
    reparse-point marker.
    """
    if info.create_system == 3:  # Unix
        return (info.external_attr >> 16) & 0o170000 == 0o120000
    return (info.external_attr >> 16) & 0x400 != 0  # DOS FILE_ATTRIBUTE_REPARSE_POINT


def extract_archive(zip_path: Path, destination: Path) -> int:
    """Extract one archive into destination. Returns the number of files written.

    Symlinks are allowed only when their target resolves inside the
    destination tree (the official archives use in-tree symlinks such as
    ``scene_info.json``); absolute targets and targets that escape the tree
    are rejected.
    """
    if not zip_path.is_file():
        raise ValueError(f"Missing archive: {zip_path}")
    destination.mkdir(parents=True, exist_ok=True)
    dest_resolved = destination.resolve()
    symlinks = []
    count = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = validate_entry_name(info.filename)
            if not name:
                continue
            if is_symlink(info):
                symlinks.append((name, info))
                continue
            target = (destination / name).resolve()
            if not target.is_relative_to(dest_resolved):
                raise ValueError(f"entry escapes destination: {info.filename!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(target, "wb") as sink:
                while chunk := source.read(4 * 1024 * 1024):
                    sink.write(chunk)
            count += 1
        for name, info in symlinks:
            link_path = destination / name
            target_text = archive.read(info).decode("utf-8")
            if os.path.isabs(target_text):
                raise ValueError(f"symlink target escapes destination: {info.filename!r}")
            resolved = (link_path.parent / target_text).resolve()
            if not resolved.is_relative_to(dest_resolved):
                raise ValueError(f"symlink target escapes destination: {info.filename!r}")
            link_path.parent.mkdir(parents=True, exist_ok=True)
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            os.symlink(target_text, link_path)
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "local_data" / "robotwin2",
                        help="Workspace root holding raw/ and extracted/ (default matches prepare_data)")
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "competition" / "data_manifest.json")
    parser.add_argument("--task", action="append", help="Select one task; repeat to select several")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    files = manifest["files"]
    if args.task:
        requested = set(args.task)
        unknown = requested - {entry["task"] for entry in files}
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")
        files = [entry for entry in files if entry["task"] in requested]
    root = args.root.expanduser().resolve()
    for entry in files:
        destination = root / "extracted" / entry["task"]
        count = extract_archive(archive_path(root, entry), destination)
        print(f"EXTRACTED {entry['task']} ({count} files)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
