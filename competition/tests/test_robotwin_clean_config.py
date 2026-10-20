import json
from pathlib import Path
import unittest

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "vla" / "robotwin_clean" / "robotwin_clean.yaml"


@unittest.skipUnless(HAVE_YAML, "requires pyyaml")
class RobotwinCleanConfigTests(unittest.TestCase):
    def test_clean_config_pins_clean_norm_stats(self):
        """The committed config must default to the clean-only norm stats.

        Without this, training silently falls back to the robot config's
        official stats (clean+randomized), which the contest forbids.
        """
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        stats_path = cfg["data"]["norm_stats_file"]
        stats = REPO / stats_path
        self.assertTrue(stats.is_file(), f"norm stats missing: {stats}")
        stats_json = json.loads(stats.read_text(encoding="utf-8"))
        self.assertEqual(stats_json.get("count"), 548893)
        self.assertNotIn("randomized", json.dumps(stats_json))
