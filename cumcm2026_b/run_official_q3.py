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
    require_latest_practice_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--simulator-data-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--maximum-localization-steps", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    baseline = OmniGeometryBaseline(
        maximum_localization_steps=args.maximum_localization_steps
    )
    result = baseline.run(environment)
    exit_response = client.exit()
    wall_elapsed = time.perf_counter() - wall_started

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": 3,
        "mode": "official_practice",
        "practice_journal": str(journal),
        "robot_id": args.robot_id,
        "enter": enter_response,
        "exit": exit_response,
        "wall_time_s": wall_elapsed,
        "virtual_time_s": environment.virtual_time_s,
        "cleared_count": environment.cleared_count,
        "cleared_channels": sorted(environment.cleared_channels),
        "measure_actions": result.measure_actions,
        "clear_actions": result.clear_actions,
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
