"""Run the deterministic Q3 baseline against an official practice session."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cumcm_b.baseline import OmniGeometryBaseline  # noqa: E402
from cumcm_b.client import SimulatorClient  # noqa: E402
from cumcm_b.official import (  # noqa: E402
    OfficialEnvironmentAdapter,
    latest_practice_statistics_id,
    require_latest_practice_run,
    wait_for_practice_statistics,
)
from cumcm_b.q4_search import SparseDirectionalSearch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--simulator-data-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--maximum-localization-steps", type=int, default=8)
    parser.add_argument(
        "--strategy", choices=("joint", "baseline"), default="joint"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    journal = require_latest_practice_run(args.simulator_data_dir)
    statistics_database = args.simulator_data_dir / "practice-statistics-queue.sqlite3"
    statistics_watermark = latest_practice_statistics_id(statistics_database)
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
    if args.strategy == "joint":
        policy = SparseDirectionalSearch(
            stop_probability=0.998,
            belief_directional_probability=0.0,
            centroid_clear_max_radius_m=75.0,
            use_enclosing_circle_target=True,
        )
    else:
        policy = OmniGeometryBaseline(
            maximum_localization_steps=args.maximum_localization_steps
        )
    result = policy.run(environment)
    exit_response = client.exit()
    wall_elapsed = time.perf_counter() - wall_started
    official_statistics = wait_for_practice_statistics(
        statistics_database,
        after_id=statistics_watermark,
        team_no=args.robot_id,
        problem_no=3,
    )
    official_all_cleared = (
        None
        if official_statistics is None
        else official_statistics["cleared_jammer_count"]
        == official_statistics["jammer_count"]
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": 3,
        "mode": "official_practice",
        "practice_journal": str(journal),
        "robot_id": args.robot_id,
        "strategy": args.strategy,
        "official_statistics": official_statistics,
        "official_all_cleared": official_all_cleared,
        "enter": enter_response,
        "exit": exit_response,
        "wall_time_s": wall_elapsed,
        "virtual_time_s": environment.virtual_time_s,
        "cleared_count": environment.cleared_count,
        "cleared_channels": sorted(environment.cleared_channels),
        "measure_actions": result.measure_actions,
        "clear_actions": result.clear_actions,
        "travel_distance_m": getattr(result, "travel_distance_m", None),
        "discovery_measure_actions": getattr(
            result, "discovery_measure_actions", None
        ),
        "pursuit_measure_actions": getattr(result, "pursuit_measure_actions", None),
        "posterior_all_sources_detected": getattr(
            result, "posterior_all_sources_detected", None
        ),
        "mean_virtual_time_per_cleared_source_s": (
            environment.virtual_time_s / environment.cleared_count
            if environment.cleared_count
            else None
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
