"""Paired Q3 comparison between phase-separated and joint belief routing."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cumcm_b.baseline import OmniGeometryBaseline  # noqa: E402
from cumcm_b.q4_search import SparseDirectionalSearch  # noqa: E402
from cumcm_b.simulator import InterferenceEnvironment, generate_sources  # noqa: E402


class TravelEnvironment(InterferenceEnvironment):
    def __init__(self, sources):
        super().__init__(sources)
        self.travel_m = 0.0

    def _move_to(self, target):
        self.travel_m += math.dist(self.position, target)
        return super()._move_to(target)


def run_one(seed: int) -> dict:
    sources = generate_sources(seed, directional_probability=0.0)
    baseline_env = TravelEnvironment(sources)
    baseline = OmniGeometryBaseline().run(baseline_env)
    joint_env = TravelEnvironment(sources)
    joint = SparseDirectionalSearch(
        stop_probability=0.998,
        belief_directional_probability=0.0,
        centroid_clear_max_radius_m=75.0,
        use_enclosing_circle_target=True,
    ).run(joint_env)
    return {
        "seed": seed,
        "sources": len(sources),
        "baseline_cleared": baseline.cleared_count,
        "baseline_time_s": baseline.virtual_time_s,
        "baseline_travel_m": baseline_env.travel_m,
        "baseline_measures": baseline.measure_actions,
        "joint_cleared": joint.cleared_count,
        "joint_time_s": joint.virtual_time_s,
        "joint_travel_m": joint.travel_distance_m,
        "joint_measures": joint.measure_actions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=150_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_one, range(args.seed_start, args.seed_start + args.episodes)))

    def summarize(prefix: str):
        times = [row[f"{prefix}_time_s"] / row[f"{prefix}_cleared"] for row in rows]
        return {
            "complete_case_rate": sum(
                row[f"{prefix}_cleared"] == row["sources"] for row in rows
            ) / len(rows),
            "source_clear_rate": sum(row[f"{prefix}_cleared"] for row in rows)
            / sum(row["sources"] for row in rows),
            "mean_time_s_per_source": statistics.fmean(times),
            "p95_time_s_per_source": sorted(times)[int(0.95 * (len(times) - 1))],
            "mean_travel_m": statistics.fmean(row[f"{prefix}_travel_m"] for row in rows),
            "mean_measures": statistics.fmean(row[f"{prefix}_measures"] for row in rows),
        }

    paired_savings = [
        (row["baseline_time_s"] - row["joint_time_s"]) / row["baseline_time_s"]
        for row in rows
    ]
    report = {
        "episodes": len(rows),
        "baseline": summarize("baseline"),
        "joint": summarize("joint"),
        "paired_mean_relative_time_saving": statistics.fmean(paired_savings),
        "joint_faster_case_rate": sum(value > 0.0 for value in paired_savings) / len(rows),
        "wall_time_s": time.perf_counter() - started,
        "failures": [
            row for row in rows
            if row["baseline_cleared"] != row["sources"]
            or row["joint_cleared"] != row["sources"]
        ],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
