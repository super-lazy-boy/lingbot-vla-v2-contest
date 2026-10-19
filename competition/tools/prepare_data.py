#!/usr/bin/env python3
"""Inspect/download the pinned competition clean archives using Python's stdlib.

No Torch, Conda, Hugging Face client, or model weights are needed. This downloads
raw ZIPs only; conversion to LeRobot and training run on the target computer.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "competition" / "data_manifest.json"
DATASET_ID = "TianxingChen/RoboTwin2.0"
GIB = 1024 ** 3


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("repo_id") != DATASET_ID:
        raise ValueError("Unsupported dataset manifest")
    if (manifest.get("setting") != "demo_clean" or manifest.get("expected_tasks") != 50
            or manifest.get("episodes_per_task") != 50
            or manifest.get("expected_training_episodes") != 2500):
        raise ValueError("Manifest must describe the 50-task, 2500-episode clean competition subset")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", "")):
        raise ValueError("Dataset revision must be a pinned 40-character commit")
    files = manifest.get("files", [])
    tasks = set()
    if len(files) != 50:
        raise ValueError("Competition manifest must contain exactly 50 clean archives")
    for entry in files:
        task = entry.get("task", "")
        if not re.fullmatch(r"[a-z0-9_]+", task) or task in tasks:
            raise ValueError(f"Invalid or duplicate task: {task!r}")
        tasks.add(task)
        if entry.get("path") != f"dataset/{task}/demo_clean.zip":
            raise ValueError(f"Only the clean archive is allowed for {task}")
        if not isinstance(entry.get("size_bytes"), int) or entry["size_bytes"] <= 0:
            raise ValueError(f"Invalid file size for {task}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
            raise ValueError(f"Missing SHA-256 for {task}")
    if manifest.get("total_bytes") != sum(entry["size_bytes"] for entry in files):
        raise ValueError("Manifest total_bytes does not match its file entries")
    return manifest


def archive_path(root: Path, entry: dict) -> Path:
    return root / "raw" / entry["task"] / "demo_clean.zip"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, entry: dict) -> None:
    if not path.is_file():
        raise ValueError(f"Missing archive: {path}")
    if path.stat().st_size != entry["size_bytes"]:
        raise ValueError(f"Incorrect file size: {path}")
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


def remaining_bytes(root: Path, files: list[dict]) -> int:
    remaining = 0
    for entry in files:
        path = archive_path(root, entry)
        if path.is_file() and path.stat().st_size == entry["size_bytes"]:
            continue  # Download/verify still checks the content hash.
        partial = path.with_suffix(".zip.partial")
        present = partial.stat().st_size if partial.is_file() else 0
        remaining += max(0, entry["size_bytes"] - present)
    return remaining


def check_disk_space(root: Path, needed: int, reserve_gib: float) -> int:
    existing = root.resolve()
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < needed + reserve_gib * GIB:
        raise ValueError(
            f"Not enough disk space: free={free / GIB:.2f} GiB, "
            f"download={needed / GIB:.2f} GiB, reserve={reserve_gib:.2f} GiB. "
            "This budget excludes extraction and LeRobot conversion."
        )
    return free


def fetch_once(url: str, partial: Path, expected_size: int, timeout: int) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError(f"Partial file is larger than the pinned archive: {partial}")
    if offset == expected_size:
        return
    headers = {"User-Agent": "lingbot-contest-preparation/1.0", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with urlopen(Request(url, headers=headers), timeout=timeout) as response:
        status = response.status
        if status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or int(match[1]) != offset or int(match[3]) != expected_size:
                raise ValueError("Server returned an unexpected Content-Range; resume was rejected")
            if int(match[2]) < offset or int(match[2]) >= expected_size:
                raise ValueError("Server returned an invalid range end")
            mode = "ab" if offset else "wb"
        elif status == 200:
            offset = 0  # A server ignoring Range must restart, never append.
            mode = "wb"
        else:
            raise ValueError(f"Unexpected HTTP status: {status}")
        received = offset
        with partial.open(mode) as stream:
            while chunk := response.read(1024 * 1024):
                if received + len(chunk) > expected_size:
                    raise ValueError("Server response exceeds the pinned archive size")
                stream.write(chunk)
                received += len(chunk)
    if received != expected_size:
        raise OSError(f"Interrupted download: {received}/{expected_size} bytes; run again to resume")


def download_file(manifest: dict, root: Path, entry: dict, timeout: int, retries: int) -> str:
    destination = archive_path(root, entry)
    if destination.exists():
        verify_file(destination, entry)
        return f"VERIFIED {entry['task']} (already present)"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".zip.partial")
    url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{manifest['revision']}/{entry['path']}"
    for attempt in range(retries + 1):
        try:
            fetch_once(url, partial, entry["size_bytes"], timeout)
            verify_file(partial, entry)
            partial.replace(destination)
            return f"DOWNLOADED {entry['task']} (SHA-256 verified)"
        except (OSError, URLError, ValueError) as error:
            # A complete but corrupt cache cannot be resumed. Preserve it for inspection.
            if partial.exists() and partial.stat().st_size >= entry["size_bytes"]:
                bad = partial.with_name(partial.name + f".corrupt-{time.time_ns()}")
                partial.replace(bad)
            if attempt == retries:
                raise RuntimeError(f"{entry['task']}: {error}") from error
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError("Unreachable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "download", "verify"))
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "local_data" / "robotwin2")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--task", action="append", help="Select one task; repeat to select several")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--reserve-gib", type=float, default=5)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8 or args.timeout <= 0 or args.retries < 0 or args.reserve_gib < 0:
        parser.error("workers must be 1..8; timeout > 0; retries and reserve-gib >= 0")
    manifest = load_manifest(args.manifest)
    files = manifest["files"]
    if args.task:
        requested = set(args.task)
        unknown = requested - {entry["task"] for entry in files}
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")
        files = [entry for entry in files if entry["task"] in requested]
    root = args.root.expanduser().resolve()
    total = sum(entry["size_bytes"] for entry in files)
    needed = remaining_bytes(root, files)
    print(f"Dataset: {DATASET_ID}@{manifest['revision']}", flush=True)
    print(f"Clean archives: {len(files)}; compressed size: {total / 1e9:.2f} GB ({total / GIB:.2f} GiB)", flush=True)
    print(f"Destination: {root}; bytes remaining: {needed:,}", flush=True)
    if args.action == "status":
        try:
            free = check_disk_space(root, needed, args.reserve_gib)
            print(f"Disk space: OK ({free / GIB:.2f} GiB free)")
        except ValueError as error:
            print(f"Disk space: {error}")
        print("Status checks do not download, extract, or convert files.")
        return 0
    if args.action == "verify":
        failures = []
        for entry in files:
            try:
                verify_file(archive_path(root, entry), entry)
            except ValueError as error:
                failures.append(str(error))
        if failures:
            raise ValueError("\n".join(failures))
        print(f"All {len(files)} selected archives passed SHA-256 verification.")
        return 0
    check_disk_space(root, needed, args.reserve_gib)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_file, manifest, root, entry, args.timeout, args.retries) for entry in files]
        failures = []
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as error:
                failures.append(str(error))
                print(f"ERROR {error}", file=sys.stderr, flush=True)
    if failures:
        raise ValueError(f"{len(failures)} download(s) failed; rerun the same command to resume")
    print("Raw clean data is ready. LeRobot conversion is a separate target-machine step.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
