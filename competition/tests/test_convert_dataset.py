import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

try:
    import cv2  # noqa: F401
    import h5py  # noqa: F401
    import numpy as np
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if HAVE_DEPS:
    SPEC = importlib.util.spec_from_file_location("convert_dataset", TOOLS / "convert_dataset.py")
    convert = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(convert)
    CODEC_SPEC = importlib.util.spec_from_file_location("image_codec", TOOLS / "image_codec.py")
    codec = importlib.util.module_from_spec(CODEC_SPEC)
    CODEC_SPEC.loader.exec_module(codec)


def gradient_frame(h=240, w=320):
    """Smooth RGB gradient: JPEG lossy error stays small, a channel swap is huge."""
    y, x = np.mgrid[0:h, 0:w]
    frame = np.zeros((h, w, 3), np.uint8)
    frame[..., 0] = (x * 255 // max(w - 1, 1)).astype(np.uint8)  # R varies along x
    frame[..., 1] = (y * 255 // max(h - 1, 1)).astype(np.uint8)  # G varies along y
    frame[..., 2] = 128                                          # B constant
    return frame


@unittest.skipUnless(HAVE_DEPS, "requires cv2/h5py (tools-venv or lingbot-train)")
class ConvertDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def make_episode_hdf5(self, steps=3, h=240, w=320, instructions=None):
        path = self.root / "episode_0000000.hdf5"
        rng = np.random.RandomState(0)
        with h5py.File(path, "w") as f:
            for prefix in ("state", "action"):
                for side in ("left", "right"):
                    f.create_dataset(f"{prefix}/{side}_arm_joint_states", data=rng.randn(steps, 6).astype(np.float32))
                    f.create_dataset(f"{prefix}/{side}_ee_joint_states", data=rng.randn(steps, 1).astype(np.float32))
            colors = {}
            for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                frames = [gradient_frame(h, w) for _ in range(steps)]
                encoded = [codec.encode_image_bit(frame) for frame in frames]
                max_len = max(len(buf) for buf in encoded)
                col = np.array([buf.ljust(max_len, b"\0") for buf in encoded], dtype=f"S{max_len}")
                f.create_dataset(f"vision/{cam}/colors", data=col)
                colors[cam] = frames
            if instructions is None:
                instructions = ["Pick the bottle with ridges", "Lift the mug"]
            f.create_dataset("additional_info/instructions", data=json.dumps(instructions).encode("utf-8"))
        return path, colors

    def test_decode_marked_jpeg_returns_rgb(self):
        frame = gradient_frame()
        decoded = codec.decode_image_bit(codec.encode_image_bit(frame))
        self.assertLess(np.abs(decoded.astype(int) - frame.astype(int)).mean(), 4)

    def test_decode_legacy_jpeg_returns_rgb(self):
        # Legacy: RGB array handed straight to cv2.imencode, which reads it as BGR.
        frame = gradient_frame()
        ok, buf = cv2.imencode(".jpg", frame)
        self.assertTrue(ok)
        decoded = codec.decode_image_bit(buf.tobytes())
        self.assertLess(np.abs(decoded.astype(int) - frame.astype(int)).mean(), 4)

    def test_concat_14dim_order(self):
        path, _ = self.make_episode_hdf5(steps=5)
        with h5py.File(path, "r") as f:
            state = convert.concat_14dim(f, "state")
            action = convert.concat_14dim(f, "action")
        self.assertEqual(state.shape, (5, 14))
        self.assertEqual(action.shape, (5, 14))
        # columns 0-5 = left arm, 6 = left gripper, 7-12 = right arm, 13 = right gripper
        with h5py.File(path, "r") as f:
            np.testing.assert_array_equal(state[:, :6], f["state/left_arm_joint_states"][:])
            np.testing.assert_array_equal(state[:, 6:7], f["state/left_ee_joint_states"][:])
            np.testing.assert_array_equal(state[:, 7:13], f["state/right_arm_joint_states"][:])
            np.testing.assert_array_equal(state[:, 13:14], f["state/right_ee_joint_states"][:])

    def test_read_instructions(self):
        path, _ = self.make_episode_hdf5(instructions=["instr one", "instr two"])
        with h5py.File(path, "r") as f:
            self.assertEqual(convert.read_instructions(f), ["instr one", "instr two"])

    def test_episode_to_frames_maps_cameras_and_resizes(self):
        path, originals = self.make_episode_hdf5(steps=2)
        frames = convert.episode_to_frames(path, instruction="instr one")
        frames = list(frames)
        self.assertEqual(len(frames), 2)
        frame = frames[0]
        self.assertEqual(frame["task"], "instr one")
        self.assertEqual(frame["observation.state"].shape, (14,))
        self.assertEqual(frame["action"].shape, (14,))
        img = frame["observation.images.cam_high"]
        self.assertEqual(img.shape, (480, 640, 3))
        # Smooth gradient survives decode+resize with small lossy error;
        # a channel swap would show up as a huge difference in R and B.
        src = originals["cam_head"][0][120, 160]
        got = img[240, 320]
        self.assertLess(np.abs(src.astype(int) - got.astype(int)).max(), 12)
        # channel order structure: R must still vary along width, B must stay flat
        self.assertGreater(img[0, -1, 0], img[0, 0, 0] + 100)
        self.assertLess(np.abs(img[:, :, 2].astype(int) - 128).max(), 8)

    def test_camera_name_mapping(self):
        self.assertEqual(convert.CAMERA_MAP["cam_head"], "cam_high")
        self.assertEqual(convert.CAMERA_MAP["cam_left_wrist"], "cam_left_wrist")
        self.assertEqual(convert.CAMERA_MAP["cam_right_wrist"], "cam_right_wrist")
        self.assertNotIn("cam_third_view", convert.CAMERA_MAP)


if __name__ == "__main__":
    unittest.main()
