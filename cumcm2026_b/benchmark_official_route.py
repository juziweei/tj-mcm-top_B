"""Reconstruct an official practice run and compare it with an exact route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cumcm_b.route_oracle import (  # noqa: E402
    euclidean_distance,
    exact_open_route,
    mst_lower_bound,
    route_distance,
)


def read_actions(path: Path) -> list[dict]:
    actions: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("path") in {"/measure", "/clear"}:
                actions.append(row)
    return actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("client_log", type=Path)
    args = parser.parse_args()

    actions = read_actions(args.client_log)
    position = (0.0, 0.0)
    current_channel = 1
    travel_m = 0.0
    measures = 0
    switches = 0
    successful_clears = 0
    failed_clears = 0
    clear_positions = []
    last_clear_position = (0.0, 0.0)
    last_cleared_channel = None
    segment_travel_m = 0.0
    segment_measures = 0
    segment_switches = 0
    segment_failed_clears = 0
    clear_segments = []
    for row in actions:
        request = row["request"]
        target = (
            float(request["position"]["x"]),
            float(request["position"]["y"]),
        )
        movement = euclidean_distance(position, target)
        travel_m += movement
        segment_travel_m += movement
        position = target
        channel = int(request["channel"])
        if row["path"] == "/measure":
            measures += 1
            segment_measures += 1
            switched = channel != current_channel
            switches += switched
            segment_switches += switched
            current_channel = channel
        else:
            if row["response"]["clear_result"] == "success":
                successful_clears += 1
                clear_positions.append(target)
                direct_m = euclidean_distance(last_clear_position, target)
                clear_segments.append(
                    {
                        "clear_sequence_index": successful_clears,
                        "from_cleared_channel": last_cleared_channel,
                        "cleared_channel": channel,
                        "travel_m": segment_travel_m,
                        "endpoint_chord_m": direct_m,
                        "detour_m": segment_travel_m - direct_m,
                        "measures": segment_measures,
                        "channel_switches": segment_switches,
                        "failed_clears": segment_failed_clears,
                    }
                )
                last_clear_position = target
                last_cleared_channel = channel
                segment_travel_m = 0.0
                segment_measures = 0
                segment_switches = 0
                segment_failed_clears = 0
            else:
                failed_clears += 1
                segment_failed_clears += 1

    route = exact_open_route(clear_positions)
    mst = mst_lower_bound(clear_positions)
    execution_clear_route_m = route_distance(
        clear_positions, range(len(clear_positions))
    )
    reconstructed_time = (
        travel_m / 5.0
        + measures * 5.0
        + switches
        + successful_clears * 5.0
        + failed_clears * 3.0
    )
    observed_position_oracle_time = route.distance_m / 5.0 + successful_clears * 5.0
    report = {
        "scope": "successful official clear positions; each is within 20 m of its source",
        "actions": {
            "measures": measures,
            "channel_switches": switches,
            "successful_clears": successful_clears,
            "failed_clears": failed_clears,
        },
        "time_decomposition_s": {
            "movement": travel_m / 5.0,
            "measurement": measures * 5.0,
            "channel_switch": float(switches),
            "successful_clear": successful_clears * 5.0,
            "failed_clear": failed_clears * 3.0,
            "reconstructed_total": reconstructed_time,
        },
        "travel_m": {
            "actual": travel_m,
            "mst_lower_bound": mst,
            "exact_open_route": route.distance_m,
            "avoidable_vs_exact": travel_m - route.distance_m,
            "actual_over_exact": travel_m / route.distance_m,
            "execution_clear_order": execution_clear_route_m,
            "excess_from_clear_order": execution_clear_route_m - route.distance_m,
            "inter_clear_search_and_localization_detour": (
                travel_m - execution_clear_route_m
            ),
        },
        "observed_position_oracle": {
            "total_time_s": observed_position_oracle_time,
            "mean_time_s_per_source": observed_position_oracle_time
            / successful_clears,
            "actual_over_oracle_time": reconstructed_time
            / observed_position_oracle_time,
            "clear_order_zero_based": list(route.order),
        },
        "largest_inter_clear_detours": sorted(
            clear_segments, key=lambda segment: segment["detour_m"], reverse=True
        )[:8],
        "post_last_clear": {
            "travel_m": segment_travel_m,
            "measures": segment_measures,
            "channel_switches": segment_switches,
            "failed_clears": segment_failed_clears,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
