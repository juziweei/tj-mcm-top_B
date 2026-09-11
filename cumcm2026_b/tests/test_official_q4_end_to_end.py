import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm_b.simulator import InterferenceEnvironment, generate_sources  # noqa: E402


class OfficialQ4EndToEndTests(unittest.TestCase):
    def test_actual_cli_over_official_http_shape_clears_mixed_case(self):
        simulation = InterferenceEnvironment(
            generate_sources(95_000_123, count=10, directional_probability=0.6)
        )
        requests: list[tuple[str, dict]] = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def do_POST(self):  # noqa: N802
                length = int(self.headers["Content-Length"])
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                requests.append((self.path, payload))
                response = {"accepted": True, "real_timestamp_ms": 1}
                if self.path == "/enter":
                    response.update(
                        virtual_time_s=0.0,
                        max_virtual_duration_s=360000.0,
                        max_real_duration_s=1200.0,
                        remaining_real_duration_s=1200.0,
                    )
                elif self.path == "/measure":
                    point = (
                        float(payload["position"]["x"]),
                        float(payload["position"]["y"]),
                    )
                    observation = simulation.measure(point, int(payload["channel"]))
                    response.update(
                        virtual_time_s=observation.virtual_time_s,
                        measure_result=observation.result,
                    )
                    if observation.bearing_deg is not None:
                        response["svd_deg"] = observation.bearing_deg
                elif self.path == "/clear":
                    point = (
                        float(payload["position"]["x"]),
                        float(payload["position"]["y"]),
                    )
                    observation = simulation.clear(point, int(payload["channel"]))
                    response.update(
                        virtual_time_s=observation.virtual_time_s,
                        clear_result=observation.result,
                    )
                elif self.path == "/exit":
                    response.update(
                        virtual_time_s=simulation.virtual_time_s,
                        exit_reason="user_exit",
                    )
                else:
                    self.send_error(404)
                    return
                encoded = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        root = Path(__file__).resolve().parents[1]
        try:
            with tempfile.TemporaryDirectory() as directory:
                temp = Path(directory)
                journal = temp / "behavior-runs" / "run-test" / "behavior.journal.jsonl"
                journal.parent.mkdir(parents=True)
                journal.write_text(
                    json.dumps({"record_type": "lifecycle", "event": "practice_authorized"})
                    + "\n"
                    + json.dumps({"record_type": "lifecycle", "event": "api_opened"})
                    + "\n",
                    encoding="utf-8",
                )
                log = temp / "client.jsonl"
                summary = temp / "summary.json"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(root / "run_official_q4.py"),
                        "--robot-id", "test-team",
                        "--simulator-data-dir", str(temp),
                        "--base-url", f"http://127.0.0.1:{server.server_port}",
                        "--log", str(log),
                        "--summary", str(summary),
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=True,
                )
                self.assertTrue(completed.stdout.strip())
                result = json.loads(summary.read_text(encoding="utf-8"))
                self.assertEqual(result["cleared_count"], simulation.source_count)
                self.assertEqual(result["unresolved_channels"], [])
                self.assertEqual(requests[0][0], "/enter")
                self.assertEqual(requests[-1][0], "/exit")
                request_ids = [payload["request_id"] for _, payload in requests]
                self.assertEqual(len(request_ids), len(set(request_ids)))
                self.assertTrue(
                    all(payload["arena_id"] == "default" for _, payload in requests)
                )
                self.assertEqual(
                    len(log.read_text(encoding="utf-8").splitlines()), len(requests)
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
