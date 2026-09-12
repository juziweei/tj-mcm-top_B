from contextlib import closing
import json
from pathlib import Path
import sys
import tempfile
import unittest
import sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm_b.official import (  # noqa: E402
    OfficialEnvironmentAdapter,
    latest_practice_statistics_id,
    require_latest_practice_run,
    wait_for_practice_statistics,
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
    def test_reads_only_the_new_matching_practice_statistics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "practice.sqlite3"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    """CREATE TABLE practice_statistics_tasks (
                    id INTEGER PRIMARY KEY, team_no TEXT, problem_no INTEGER,
                    practice_run_no INTEGER, case_code TEXT, end_reason TEXT,
                    cleared_jammer_count INTEGER, measure_accepted_count INTEGER,
                    virtual_time_us INTEGER, program_run_duration_ms INTEGER,
                    channel_switch_count INTEGER, clear_failure_count INTEGER,
                    jammer_count INTEGER, state TEXT)"""
                )
                connection.execute(
                    "INSERT INTO practice_statistics_tasks VALUES "
                    "(1, 'team', 4, 7, 'case', 'user_exit', 16, 200, "
                    "5000000, 900, 150, 2, 16, 'confirmed')"
                )
                connection.commit()
            self.assertEqual(latest_practice_statistics_id(database), 1)
            self.assertIsNone(
                wait_for_practice_statistics(
                    database,
                    after_id=1,
                    team_no="team",
                    problem_no=4,
                    timeout_s=0.0,
                )
            )
            row = wait_for_practice_statistics(
                database,
                after_id=0,
                team_no="team",
                problem_no=4,
                timeout_s=0.0,
            )
            self.assertIsNotNone(row)
            self.assertEqual(row["cleared_jammer_count"], row["jammer_count"])

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
