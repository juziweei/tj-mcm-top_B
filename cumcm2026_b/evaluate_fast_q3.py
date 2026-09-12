"""Evaluate the interleaved Q3 policy on independent surrogate seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cumcm_b.fast_q3 import InterleavedOmniSearch  # noqa: E402
from cumcm_b.simulator import InterferenceEnvironment, generate_sources  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=70_000_000)
    parser.add_argument("--miss-budget-sources", type=float, default=0.015)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for offset in range(args.episodes):
        episode_wall_started = time.perf_counter()
        sources = generate_sources(
            args.seed_start + offset, directional_probability=0.0
        )
        environment = InterferenceEnvironment(sources)
        result = InterleavedOmniSearch(
            miss_budget_sources=args.miss_budget_sources
        ).run(environment)
        rows.append(
            {
                "seed": args.seed_start + offset,
                "sources": len(sources),
                "cleared": result.cleared_count,
                "virtual_time_s": result.virtual_time_s,
                "mean_time_s": result.mean_clear_time_s,
                "measures": result.measure_actions,
                "clears": result.clear_actions,
                "travel_m": result.travel_distance_m,
                "scan_positions": result.discovery_scan_positions,
                "wall_time_s": time.perf_counter() - episode_wall_started,
            }
        )

    total_sources = sum(row["sources"] for row in rows)
    total_cleared = sum(row["cleared"] for row in rows)
    complete = sum(row["sources"] == row["cleared"] for row in rows)
    mean_times = [row["mean_time_s"] for row in rows]
    report = {
        "episodes": len(rows),
        "source_clear_rate": total_cleared / total_sources,
        "complete_case_rate": complete / len(rows),
        "mean_episode_mean_time_s": statistics.fmean(mean_times),
        "median_episode_mean_time_s": statistics.median(mean_times),
        "p95_episode_mean_time_s": sorted(mean_times)[int(0.95 * (len(mean_times) - 1))],
        "mean_travel_m": statistics.fmean(row["travel_m"] for row in rows),
        "mean_episode_wall_time_s": statistics.fmean(
            row["wall_time_s"] for row in rows
        ),
        "p95_episode_wall_time_s": sorted(row["wall_time_s"] for row in rows)[
            int(0.95 * (len(rows) - 1))
        ],
        "max_episode_wall_time_s": max(row["wall_time_s"] for row in rows),
        "failures": [row for row in rows if row["sources"] != row["cleared"]],
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
