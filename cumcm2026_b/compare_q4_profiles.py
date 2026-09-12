"""Paired comparison of two Q4 evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[int(probability * (len(ordered) - 1))]


def _bootstrap_mean_interval(
    values: list[float], *, samples: int = 10_000, seed: int = 2026
) -> tuple[float, float]:
    generator = random.Random(seed)
    size = len(values)
    means = [
        statistics.fmean(values[generator.randrange(size)] for _ in range(size))
        for _ in range(samples)
    ]
    return _percentile(means, 0.025), _percentile(means, 0.975)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline_rows = baseline.get("rows")
    candidate_rows = candidate.get("rows")
    if not baseline_rows or not candidate_rows:
        raise SystemExit("both reports must be generated with --include-rows")
    baseline_by_seed = {int(row["seed"]): row for row in baseline_rows}
    candidate_by_seed = {int(row["seed"]): row for row in candidate_rows}
    if baseline_by_seed.keys() != candidate_by_seed.keys():
        raise SystemExit("paired reports do not contain identical seeds")
    pairs = [
        (baseline_by_seed[seed], candidate_by_seed[seed])
        for seed in sorted(baseline_by_seed)
    ]

    baseline_time = [row[0]["virtual_time_s"] / row[0]["sources"] for row in pairs]
    candidate_time = [row[1]["virtual_time_s"] / row[1]["sources"] for row in pairs]
    paired_savings = [old - new for old, new in zip(baseline_time, candidate_time)]
    relative_savings = [
        (old - new) / old for old, new in zip(baseline_time, candidate_time)
    ]
    time_ci = _bootstrap_mean_interval(paired_savings)
    relative_ci = _bootstrap_mean_interval(relative_savings)
    baseline_sources = sum(row[0]["sources"] for row in pairs)
    candidate_sources = sum(row[1]["sources"] for row in pairs)
    report = {
        "episodes": len(pairs),
        "baseline": {
            "source_clear_rate": sum(row[0]["cleared"] for row in pairs)
            / baseline_sources,
            "complete_case_rate": statistics.fmean(
                row[0]["cleared"] == row[0]["sources"] for row in pairs
            ),
            "mean_virtual_time_s_per_present_source": statistics.fmean(
                baseline_time
            ),
            "p95_virtual_time_s_per_present_source": _percentile(
                baseline_time, 0.95
            ),
            "mean_wall_time_s": statistics.fmean(
                row[0]["wall_time_s"] for row in pairs
            ),
        },
        "candidate": {
            "source_clear_rate": sum(row[1]["cleared"] for row in pairs)
            / candidate_sources,
            "complete_case_rate": statistics.fmean(
                row[1]["cleared"] == row[1]["sources"] for row in pairs
            ),
            "mean_virtual_time_s_per_present_source": statistics.fmean(
                candidate_time
            ),
            "p95_virtual_time_s_per_present_source": _percentile(
                candidate_time, 0.95
            ),
            "mean_wall_time_s": statistics.fmean(
                row[1]["wall_time_s"] for row in pairs
            ),
        },
        "paired": {
            "mean_savings_s_per_present_source": statistics.fmean(paired_savings),
            "mean_savings_95pct_bootstrap_ci_s": time_ci,
            "mean_relative_savings": statistics.fmean(relative_savings),
            "mean_relative_savings_95pct_bootstrap_ci": relative_ci,
            "candidate_faster_case_rate": statistics.fmean(
                saving > 0.0 for saving in paired_savings
            ),
        },
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
