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
from cumcm_b.official import (  # noqa: E402
    OfficialEnvironmentAdapter,
    latest_practice_statistics_id,
    require_latest_practice_run,
    wait_for_practice_statistics,
)
from cumcm_b.q4_search import SparseDirectionalSearch, q4_profile_options  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--simulator-data-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=(
            "proven-fast",
            "conservative",
            "balanced",
            "aggressive",
            "source-reliable",
            "source-efficient",
            "source-98",
        ),
        default="conservative",
    )
    args = parser.parse_args()

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
    # ``proven-fast`` reproduces the original SparseDirectionalSearch defaults
    # that achieved 355.19 virtual seconds/source in official practice run 002.
    policy_options = (
        {} if args.profile == "proven-fast" else q4_profile_options(args.profile)
    )
    result = SparseDirectionalSearch(**policy_options).run(environment)
    exit_response = client.exit()
    wall_time_s = time.perf_counter() - wall_started
    official_statistics = wait_for_practice_statistics(
        statistics_database,
        after_id=statistics_watermark,
        team_no=args.robot_id,
        problem_no=4,
    )
    official_all_cleared = (
        None
        if official_statistics is None
        else official_statistics["cleared_jammer_count"]
        == official_statistics["jammer_count"]
    )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": 4,
        "mode": "official_practice",
        "practice_journal": str(journal),
        "robot_id": args.robot_id,
        "profile": args.profile,
        "policy_options": policy_options,
        "official_statistics": official_statistics,
        "official_all_cleared": official_all_cleared,
        "enter": enter_response,
        "exit": exit_response,
        "wall_time_s": wall_time_s,
        "virtual_time_s": environment.virtual_time_s,
        "cleared_count": environment.cleared_count,
        "cleared_channels": sorted(environment.cleared_channels),
        "measure_actions": result.measure_actions,
        "discovery_measure_actions": result.discovery_measure_actions,
        "pursuit_measure_actions": result.pursuit_measure_actions,
        "clear_actions": result.clear_actions,
        "travel_distance_m": result.travel_distance_m,
        "discovery_travel_distance_m": result.discovery_travel_distance_m,
        "pursuit_travel_distance_m": result.pursuit_travel_distance_m,
        "discovery_positions": result.discovery_positions,
        "unresolved_channels": list(result.unresolved_channels),
        "posterior_all_sources_detected": result.posterior_all_sources_detected,
        "posterior_expected_remaining_sources": (
            result.posterior_expected_remaining_sources
        ),
        "posterior_expected_missed_source_fraction": (
            result.posterior_expected_missed_source_fraction
        ),
        "stopped_by_probability": result.stopped_by_probability,
        "stopped_by_source_risk": result.stopped_by_source_risk,
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
