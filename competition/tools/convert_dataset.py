#!/usr/bin/env python3
"""Convert new-format RoboTwin 2.0 HDF5 (data_format_version v1.0) to LeRobot v2.1.

The pinned RoboTwin conversion scripts expect the legacy raw layout
(``joint_action``/``observation``), but the pinned dataset release stores
``action``/``state``/``vision`` with marked standard-RGB JPEG buffers, so this
converter replaces the official ``process_data_pi0.sh`` + ``generate.sh`` chain:

- 14-dim state/action: left arm 6, left gripper 1, right arm 6, right gripper 1
  (matches ``configs/robot_configs/robotwin.yaml`` and the contest motors list);
- cameras: cam_head -> cam_high, cam_left_wrist, cam_right_wrist;
  cam_third_view is dropped; frames decode via ``image_codec.decode_image_bit``
  (RGB for both legacy and marked buffers) and resize to 640x480;
- one episode instruction: ``additional_info/instructions`` JSON, first entry;
- output: LeRobot v2.1 with videos, fps=50 metadata (frame i maps to frame i).

Requires the lingbot-train environment (lerobot, av, h5py, cv2, numpy).
Set XDG_CACHE_HOME to control the LeRobot dataset home before running.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from image_codec import decode_image_bit  # noqa: E402

CAMERA_MAP = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}

# (hdf5 field, width) in the exact 14-dim order the robot config consumes:
# left arm 6 + left gripper 1 + right arm 6 + right gripper 1.
STATE_KEYS = [
    ("left_arm_joint_states", 6),
    ("left_ee_joint_states", 1),
    ("right_arm_joint_states", 6),
    ("right_ee_joint_states", 1),
]

MOTORS = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
    "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
    "right_wrist_angle", "right_wrist_rotate", "right_gripper",
]

TARGET_WIDTH, TARGET_HEIGHT = 640, 480


def concat_14dim(root: h5py.Group, prefix: str) -> np.ndarray:
    """Assemble the 14-dim vector from the new-format field layout."""
    parts = []
    for key, width in STATE_KEYS:
        array = root[f"{prefix}/{key}"][:]
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f"Unexpected {prefix}/{key} shape: {array.shape}")
        parts.append(array)
    return np.concatenate(parts, axis=1).astype(np.float32)


def read_instructions(root: h5py.Group) -> list[str]:
    """Parse the episode instruction list stored as JSON bytes."""
    raw = root["additional_info/instructions"][()]
    if isinstance(raw, (bytes, np.bytes_)):
        raw = bytes(raw).decode("utf-8")
    instructions = json.loads(raw)
    if not isinstance(instructions, list) or not instructions:
        raise ValueError("instructions must be a non-empty JSON list")
    return instructions


def episode_to_frames(ep_path: Path, instruction: str):
    """Yield per-frame dicts for one episode in the official converter's shape.

    Images are decoded to RGB, resized to 640x480, and returned as uint8
    (H, W, 3) arrays — the same representation the LeRobot video writer and
    the training dataloader (torchcodec/pyav) agree on.
    """
    with h5py.File(ep_path, "r") as root:
        state = concat_14dim(root, "state")
        action = concat_14dim(root, "action")
        if state.shape != action.shape:
            raise ValueError(f"state/action length mismatch in {ep_path}")
        images = {}
        for source, target in CAMERA_MAP.items():
            decoded = decode_image_bit(root[f"vision/{source}/colors"][:])
            images[target] = [cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                              for frame in decoded]
    num_frames = state.shape[0]
    for i in range(num_frames):
        frame = {
            "observation.state": state[i],
            "action": action[i],
            "task": instruction,
        }
        for camera, frames in images.items():
            frame[f"observation.images.{camera}"] = frames[i]
        yield frame


def lerobot_features() -> dict:
    """Feature dict mirroring the official converter's create_empty_dataset."""
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
        "action": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
    }
    for camera in CAMERA_MAP.values():
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (3, TARGET_HEIGHT, TARGET_WIDTH),
            "names": ["channels", "height", "width"],
        }
    return features


def find_episodes(raw_root: Path, task: str) -> list[Path]:
    """Sorted list of episode HDF5s for one extracted task directory."""
    episodes = sorted((raw_root / task).rglob("data/episode_*.hdf5"))
    if not episodes:
        raise ValueError(f"No data/episode_*.hdf5 found under {raw_root / task}")
    return episodes


def build_dataset(episodes: list[Path], repo_id: str) -> None:
    """Write all episodes into a fresh LeRobot v2.1 dataset (videos)."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type="aloha-agilex",
        features=lerobot_features(),
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=10,
        image_writer_threads=5,
        video_backend=None,
    )
    for ep_path in episodes:
        with h5py.File(ep_path, "r") as root:
            instruction = read_instructions(root)[0]
        for frame in episode_to_frames(ep_path, instruction):
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f"WROTE episode {ep_path.parent.name}/{ep_path.name} ({instruction!r})", flush=True)
    print(f"Dataset ready: {repo_id} with {len(episodes)} episodes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Root holding extracted/<task>/ (datasets/robotwin2/extracted)")
    parser.add_argument("--task", required=True, help="Single task name, e.g. adjust_bottle")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset name, e.g. robotwin_clean_adjust_bottle")
    args = parser.parse_args(argv)
    episodes = find_episodes(args.raw_root, args.task)
    print(f"{args.task}: {len(episodes)} episodes", flush=True)
    build_dataset(episodes, args.repo_id)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
