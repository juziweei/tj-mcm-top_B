import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm_b.official import (  # noqa: E402
    OfficialEnvironmentAdapter,
    require_latest_practice_run,
)


class FakeClient:
    def measure(self, position, channel):
        return {
            "accepted": True,
            "virtual_time_s": 8.0,
            "measure_result": "direction",
            "svd_deg": 42.5,
        }

    def clear(self, position, channel):
        return {
            "accepted": True,
            "virtual_time_s": 13.0,
            "clear_result": "success",
        }


class OfficialAdapterTests(unittest.TestCase):
    def test_converts_official_responses_and_tracks_state(self):
        environment = OfficialEnvironmentAdapter(FakeClient())
        measurement = environment.measure((3.0, 4.0), 2)
        self.assertEqual(measurement.result, "direction")
        self.assertEqual(measurement.bearing_deg, 42.5)
        self.assertEqual(measurement.action_duration_s, 8.0)
        clearing = environment.clear((5.0, 6.0), 2)
        self.assertEqual(clearing.result, "success")
        self.assertEqual(clearing.action_duration_s, 5.0)
        self.assertEqual(environment.cleared_channels, {2})

    def test_practice_guard_rejects_formal_journal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            journal = Path(temp_dir) / "behavior-runs" / "run-test" / "behavior.journal.jsonl"
            journal.parent.mkdir(parents=True)
            journal.write_text(
                json.dumps({"record_type": "lifecycle", "event": "formal_authorized"}) + "\n"
                + json.dumps({"record_type": "lifecycle", "event": "api_opened"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                require_latest_practice_run(temp_dir)

    def test_practice_guard_accepts_open_practice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            journal = Path(temp_dir) / "behavior-runs" / "run-test" / "behavior.journal.jsonl"
            journal.parent.mkdir(parents=True)
            journal.write_text(
                json.dumps({"record_type": "lifecycle", "event": "practice_authorized"}) + "\n"
                + json.dumps({"record_type": "lifecycle", "event": "api_opened"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(require_latest_practice_run(temp_dir), journal)


if __name__ == "__main__":
    unittest.main()
