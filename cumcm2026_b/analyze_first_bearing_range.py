"""Audit the one-bearing range proxy against hidden source distance."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cumcm_b.baseline import ChannelTrack  # noqa: E402
from cumcm_b.q4_search import SparseDirectionalSearch  # noqa: E402
from cumcm_b.simulator import (  # noqa: E402
    InterferenceEnvironment,
    generate_sources,
)


class FirstBearingEnvironment(InterferenceEnvironment):
    def __init__(self, sources, estimator):
        super().__init__(sources)
        self._estimator = estimator
        self._audit_tracks = {channel: ChannelTrack(channel) for channel in range(1, 21)}
        self.rows: list[dict] = []

    def measure(self, position, channel):
        result = super().measure(position, channel)
        track = self._audit_tracks[channel]
        track.update(result)
        if result.result == "direction" and len(track.measurements) <= 3:
            source = self._sources[channel]
            estimate = self._estimator(track)
            circle = track.enclosing_circle
            true_range = math.dist(position, source.position)
            estimated_range = math.dist(position, estimate)
            self.rows.append(
                {
                    "bearing_count": len(track.measurements),
                    "directional": source.directional,
                    "true_range_m": true_range,
                    "estimated_range_m": estimated_range,
                    "signed_error_m": estimated_range - true_range,
                    "absolute_error_m": math.dist(estimate, source.position),
                    "belief_radius_m": None if circle is None else circle.radius,
                    "circle_center_absolute_error_m": (
                        None
                        if circle is None
                        else math.dist(circle.center, source.position)
                    ),
                }
            )
        return result


def _episode(arguments):
    seed, source_count = arguments
    sources = generate_sources(seed, count=source_count, directional_probability=0.5)
    controller = SparseDirectionalSearch()
    environment = FirstBearingEnvironment(sources, controller._estimated_target)
    result = controller.run(environment)
    return result.cleared_count == len(sources), environment.rows


def _quantile(values, fraction):
    return sorted(values)[int(fraction * (len(values) - 1))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=100_000_000)
    parser.add_argument("--source-count", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        episodes = list(
            executor.map(
                _episode,
                [
                    (args.seed_start + offset, args.source_count)
                    for offset in range(args.episodes)
                ],
            )
        )
    rows = [row for _, episode_rows in episodes for row in episode_rows]
    one_bearing_rows = [row for row in rows if row["bearing_count"] == 1]
    true_ranges = [row["true_range_m"] for row in one_bearing_rows]
    estimates = [row["estimated_range_m"] for row in one_bearing_rows]
    errors = [row["signed_error_m"] for row in one_bearing_rows]
    mean_true = statistics.fmean(true_ranges)
    mean_estimate = statistics.fmean(estimates)
    covariance = statistics.fmean(
        (truth - mean_true) * (estimate - mean_estimate)
        for truth, estimate in zip(true_ranges, estimates)
    )
    correlation = covariance / (
        statistics.pstdev(true_ranges) * statistics.pstdev(estimates)
    )
    two_bearing_rows = [row for row in rows if row["bearing_count"] == 2]
    radius_thresholds = {}
    for threshold in (40.0, 75.0, 125.0, 200.0, 400.0):
        selected = [
            row
            for row in two_bearing_rows
            if row["belief_radius_m"] is not None
            and row["belief_radius_m"] <= threshold
        ]
        if selected:
            radius_thresholds[str(threshold)] = {
                "selected": len(selected),
                "selected_fraction": len(selected) / max(1, len(two_bearing_rows)),
                "within_20m_rate": statistics.fmean(
                    row["absolute_error_m"] <= 20.0 for row in selected
                ),
                "absolute_error_mean_m": statistics.fmean(
                    row["absolute_error_m"] for row in selected
                ),
            }
    report = {
        "episodes": args.episodes,
        "complete_rate": statistics.fmean(complete for complete, _ in episodes),
        "first_bearings": len(one_bearing_rows),
        "true_range_mean_m": mean_true,
        "estimated_range_mean_m": mean_estimate,
        "signed_error_mean_m": statistics.fmean(errors),
        "absolute_error_mean_m": statistics.fmean(
            row["absolute_error_m"] for row in one_bearing_rows
        ),
        "absolute_error_p50_m": _quantile(
            [row["absolute_error_m"] for row in one_bearing_rows], 0.5
        ),
        "absolute_error_p95_m": _quantile(
            [row["absolute_error_m"] for row in one_bearing_rows], 0.95
        ),
        "pearson_correlation": correlation,
        "error_by_bearing_count": {
            str(count): {
                "observations": len(selected),
                "absolute_error_mean_m": statistics.fmean(
                    row["absolute_error_m"] for row in selected
                ),
                "absolute_error_p50_m": _quantile(
                    [row["absolute_error_m"] for row in selected], 0.5
                ),
                "absolute_error_p95_m": _quantile(
                    [row["absolute_error_m"] for row in selected], 0.95
                ),
                "within_20m_rate": statistics.fmean(
                    row["absolute_error_m"] <= 20.0 for row in selected
                ),
                "circle_center_absolute_error_mean_m": statistics.fmean(
                    row["circle_center_absolute_error_m"] for row in selected
                ),
                "circle_center_absolute_error_p50_m": _quantile(
                    [row["circle_center_absolute_error_m"] for row in selected], 0.5
                ),
                "circle_center_absolute_error_p95_m": _quantile(
                    [row["circle_center_absolute_error_m"] for row in selected], 0.95
                ),
                "circle_center_within_20m_rate": statistics.fmean(
                    row["circle_center_absolute_error_m"] <= 20.0
                    for row in selected
                ),
            }
            for count in (1, 2, 3)
            if (selected := [row for row in rows if row["bearing_count"] == count])
        },
        "two_bearing_radius_thresholds": radius_thresholds,
        "rows": rows,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
