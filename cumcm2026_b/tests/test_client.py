import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm_b.client import SimulatorClient, SimulatorProtocolError  # noqa: E402


class SimulatorClientTests(unittest.TestCase):
    def test_measure_serializes_required_fields(self):
        captured = []

        def transport(url, body, timeout):
            captured.append((url, json.loads(body), timeout))
            return 200, json.dumps(
                {
                    "accepted": True,
                    "virtual_time_s": 5,
                    "measure_result": "direction",
                    "svd_deg": 45.0,
                }
            ).encode()

        client = SimulatorClient("team-1", transport=transport)
        response = client.measure((3.0, 4.0), 7, request_id="fixed-id")
        self.assertEqual(response["svd_deg"], 45.0)
        self.assertEqual(captured[0][0], "http://127.0.0.1:2026/measure")
        self.assertEqual(captured[0][1]["request_id"], "fixed-id")
        self.assertEqual(captured[0][1]["position"], {"x": 3.0, "y": 4.0})
        self.assertEqual(captured[0][1]["channel"], 7)

    def test_timeout_retry_reuses_identical_request(self):
        bodies = []

        def transport(url, body, timeout):
            bodies.append(body)
            if len(bodies) == 1:
                raise TimeoutError("simulated timeout")
            return 200, b'{"accepted":true,"virtual_time_s":0,"exit_reason":"user_exit"}'

        client = SimulatorClient("team-1", transport=transport, timeout_retries=1)
        client.exit(request_id="exit-fixed")
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[0], bodies[1])

    def test_unaccepted_response_raises(self):
        def transport(url, body, timeout):
            return 200, b'{"accepted":false,"reason":"not_entered"}'

        client = SimulatorClient("team-1", transport=transport)
        with self.assertRaises(SimulatorProtocolError):
            client.enter()

    def test_jsonl_log_contains_request_and_response(self):
        def transport(url, body, timeout):
            return 200, b'{"accepted":true,"virtual_time_s":0,"max_virtual_duration_s":360000,"max_real_duration_s":1200,"remaining_real_duration_s":1200}'

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "actions.jsonl"
            client = SimulatorClient("team-1", transport=transport, log_path=log_path)
            client.enter(request_id="enter-fixed")
            record = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(record["path"], "/enter")
            self.assertEqual(record["request"]["request_id"], "enter-fixed")
            self.assertTrue(record["response"]["accepted"])


if __name__ == "__main__":
    unittest.main()

