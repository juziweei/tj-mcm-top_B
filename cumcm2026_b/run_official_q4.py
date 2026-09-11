"""Run the sparse Q4 policy against an official practice session."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cumcm_b.client import SimulatorClient  # noqa: E402
from cumcm_b.official import OfficialEnvironmentAdapter, require_latest_practice_run  # noqa: E402
from cumcm_b.q4_search import SparseDirectionalSearch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--simulator-data-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    journal = require_latest_practice_run(args.simulator_data_dir)
    client = SimulatorClient(
        args.robot_id,
        base_url=args.base_url,
        timeout_s=5.0,
        timeout_retries=2,
        log_path=args.log,
    )
    wall_started = time.perf_counter()
    enter_response = client.enter()
    environment = OfficialEnvironmentAdapter(client)
    result = SparseDirectionalSearch().run(environment)
    exit_response = client.exit()
    wall_time_s = time.perf_counter() - wall_started
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": 4,
        "mode": "official_practice",
        "practice_journal": str(journal),
        "robot_id": args.robot_id,
        "enter": enter_response,
        "exit": exit_response,
        "wall_time_s": wall_time_s,
        "virtual_time_s": environment.virtual_time_s,
        "cleared_count": environment.cleared_count,
        "cleared_channels": sorted(environment.cleared_channels),
        "measure_actions": result.measure_actions,
        "clear_actions": result.clear_actions,
        "travel_distance_m": result.travel_distance_m,
        "discovery_travel_distance_m": result.discovery_travel_distance_m,
        "pursuit_travel_distance_m": result.pursuit_travel_distance_m,
        "discovery_positions": result.discovery_positions,
        "unresolved_channels": list(result.unresolved_channels),
        "posterior_all_sources_detected": result.posterior_all_sources_detected,
        "stopped_by_probability": result.stopped_by_probability,
        "mean_virtual_time_per_cleared_source_s": result.mean_clear_time_s,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
