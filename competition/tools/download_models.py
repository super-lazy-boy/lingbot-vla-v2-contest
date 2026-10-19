#!/usr/bin/env python3
"""Print the pinned model plan; download on the target machine with --download."""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QWEN_CONFIG_PATTERNS = ["*.json", "*.txt", "*.jinja", "*.model", "*.tiktoken"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Model directory on the target SSD")
    parser.add_argument("--download", action="store_true", help="Actually download; otherwise print plan only")
    parser.add_argument("--qwen-full", action="store_true", help="Also download redundant Qwen weights")
    parser.add_argument("--model", choices=("all", "base", "qwen", "moge"), default="all")
    args = parser.parse_args()
    lock = json.loads((PROJECT_ROOT / "competition/sources.lock.json").read_text(encoding="utf-8"))
    models = dict(zip(("base", "qwen", "moge"), lock["models"]))
    for name, item in models.items():
        if args.model not in ("all", name):
            continue
        destination = args.root.expanduser().resolve() / item["directory"]
        print(f"{name}: {item['repo_id']}@{item['revision']} -> {destination}", flush=True)
        patterns = QWEN_CONFIG_PATTERNS if name == "qwen" and not args.qwen_full else None
        if patterns:
            print("  Qwen config/tokenizer/processor only; VLM weights are in the VLA checkpoint.")
        if args.download:
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                raise SystemExit("Install huggingface_hub in the target environment before --download.")
            snapshot_download(repo_id=item["repo_id"], revision=item["revision"],
                              local_dir=str(destination), allow_patterns=patterns, max_workers=3)
    if not args.download:
        print("Plan only. No files downloaded and no environment changed.")


if __name__ == "__main__":
    main()
