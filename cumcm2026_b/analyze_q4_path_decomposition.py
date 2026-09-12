"""Decompose Q4 travel into clear-order cost and between-clear detours."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cumcm_b.q4_search import SparseDirectionalSearch  # noqa: E402
from cumcm_b.route_oracle import fast_open_route  # noqa: E402
from cumcm_b.simulator import (  # noqa: E402
    ClearObservation,
    InterferenceEnvironment,
    MeasureObservation,
    Source,
    generate_sources,
)


@dataclass(frozen=True)
class ActionRecord:
    kind: str
    channel: int
    start: tuple[float, float]
    target: tuple[float, float]
    movement_m: float
    result: str


class TracingEnvironment(InterferenceEnvironment):
    def __init__(self, sources: Sequence[Source]):
        super().__init__(sources)
        self.records: list[ActionRecord] = []

    def measure(self, position, channel) -> MeasureObservation:
        start = self.position
        result = super().measure(position, channel)
        self.records.append(
            ActionRecord(
                "measure",
                channel,
                start,
                position,
                math.dist(start, position),
                result.result,
            )
        )
        return result

    def clear(self, position, channel) -> ClearObservation:
        start = self.position
        result = super().clear(position, channel)
        self.records.append(
            ActionRecord(
                "clear",
                channel,
                start,
                position,
                math.dist(start, position),
                result.result,
            )
        )
        return result


def _episode(arguments: tuple[int, int | None, float]) -> dict:
    seed, source_count, directional_probability = arguments
    sources = generate_sources(
        seed,
        count=source_count,
        directional_probability=directional_probability,
    )
    by_channel = {source.channel: source for source in sources}
    environment = TracingEnvironment(sources)
    result = SparseDirectionalSearch().run(environment)

    last_clear = (0.0, 0.0)
    segment_travel = 0.0
    segment_measures = 0
    segment_failed_clears = 0
    segments: list[dict] = []
    clear_positions: list[tuple[float, float]] = []
    for record in environment.records:
        segment_travel += record.movement_m
        segment_measures += record.kind == "measure"
        segment_failed_clears += record.kind == "clear" and record.result != "success"
        if record.kind == "clear" and record.result == "success":
            chord = math.dist(last_clear, record.target)
            source = by_channel[record.channel]
            segments.append(
                {
                    "channel": record.channel,
                    "directional": source.directional,
                    "travel_m": segment_travel,
                    "endpoint_chord_m": chord,
                    "detour_m": segment_travel - chord,
                    "measures": segment_measures,
                    "failed_clears": segment_failed_clears,
                    "clear_error_m": math.dist(record.target, source.position),
                }
            )
            clear_positions.append(record.target)
            last_clear = record.target
            segment_travel = 0.0
            segment_measures = 0
            segment_failed_clears = 0

    execution_clear_route_m = sum(
        math.dist(first, second)
        for first, second in zip(
            [(0.0, 0.0), *clear_positions[:-1]], clear_positions
        )
    )
    source_oracle = fast_open_route([source.position for source in sources])
    return {
        "seed": seed,
        "sources": len(sources),
        "cleared": result.cleared_count,
        "virtual_time_s": result.virtual_time_s,
        "travel_m": result.travel_distance_m,
        "execution_clear_route_m": execution_clear_route_m,
        "source_oracle_route_m": source_oracle.distance_m,
        "between_clear_detour_m": result.travel_distance_m - execution_clear_route_m,
        "clear_order_excess_m": execution_clear_route_m - source_oracle.distance_m,
        "post_last_clear_travel_m": segment_travel,
        "segments": segments,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=98_000_000)
    parser.add_argument("--source-count", type=int, default=16)
    parser.add_argument("--directional-probability", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    work = [
        (
            args.seed_start + offset,
            args.source_count,
            args.directional_probability,
        )
        for offset in range(args.episodes)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(_episode, work))
    all_segments = [
        {"seed": row["seed"], **segment}
        for row in rows
        for segment in row["segments"]
    ]
    total_sources = sum(row["sources"] for row in rows)
    aggregate = {
        "episodes": len(rows),
        "complete_rate": statistics.fmean(
            row["cleared"] == row["sources"] for row in rows
        ),
        "mean_virtual_time_s_per_source": sum(
            row["virtual_time_s"] for row in rows
        )
        / total_sources,
        "mean_total_travel_m_per_source": sum(row["travel_m"] for row in rows)
        / total_sources,
        "mean_source_oracle_route_m_per_source": sum(
            row["source_oracle_route_m"] for row in rows
        )
        / total_sources,
        "mean_between_clear_detour_m_per_source": sum(
            row["between_clear_detour_m"] for row in rows
        )
        / total_sources,
        "mean_clear_order_excess_m_per_source": sum(
            row["clear_order_excess_m"] for row in rows
        )
        / total_sources,
        "travel_excess_share_between_clear_detour": (
            sum(row["between_clear_detour_m"] for row in rows)
            / sum(
                row["travel_m"] - row["source_oracle_route_m"] for row in rows
            )
        ),
        "segments": len(all_segments),
        "directional_segment_mean_detour_m": statistics.fmean(
            segment["detour_m"]
            for segment in all_segments
            if segment["directional"]
        ),
        "omnidirectional_segment_mean_detour_m": statistics.fmean(
            segment["detour_m"]
            for segment in all_segments
            if not segment["directional"]
        ),
        "mean_measures_per_clear_segment": statistics.fmean(
            segment["measures"] for segment in all_segments
        ),
        "mean_failed_clears_per_segment": statistics.fmean(
            segment["failed_clears"] for segment in all_segments
        ),
        "largest_detours": sorted(
            all_segments, key=lambda segment: segment["detour_m"], reverse=True
        )[:20],
    }
    report = {"aggregate": aggregate, "episodes": rows}
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
