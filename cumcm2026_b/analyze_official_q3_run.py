"""Decompose an official Q3 client log into physical time components."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cumcm_b.route_oracle import exact_open_route  # noqa: E402


def analyze(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    position = (0.0, 0.0)
    channel = 1
    travel_m = 0.0
    measure_count = 0
    switch_count = 0
    clear_count = 0
    clear_failures = 0
    clear_points = []
    first_clear_time_s = None
    max_leg_m = 0.0
    final_virtual_time_s = 0.0
    for row in rows:
        operation = row.get("path")
        if operation not in {"/measure", "/clear"}:
            continue
        request = row["request"]
        target = (float(request["position"]["x"]), float(request["position"]["y"]))
        leg = math.dist(position, target)
        travel_m += leg
        max_leg_m = max(max_leg_m, leg)
        position = target
        response = row["response"]
        final_virtual_time_s = float(response["virtual_time_s"])
        if operation == "/measure":
            measure_count += 1
            requested_channel = int(request["channel"])
            switch_count += int(requested_channel != channel)
            channel = requested_channel
        else:
            if response["clear_result"] == "success":
                clear_count += 1
                clear_points.append(target)
                if first_clear_time_s is None:
                    first_clear_time_s = final_virtual_time_s
            else:
                clear_failures += 1
    travel_time_s = travel_m / 5.0
    measure_time_s = 5.0 * measure_count
    switch_time_s = float(switch_count)
    clear_time_s = 3.0 * (clear_count + clear_failures) + 2.0 * clear_count
    reconstructed = travel_time_s + measure_time_s + switch_time_s + clear_time_s
    oracle = exact_open_route(clear_points)
    oracle_time_s = oracle.distance_m / 5.0 + 5.0 * clear_count
    return {
        "log": str(path),
        "sources_cleared": clear_count,
        "virtual_time_s": final_virtual_time_s,
        "mean_time_s_per_source": final_virtual_time_s / max(1, clear_count),
        "travel_m": travel_m,
        "travel_time_s": travel_time_s,
        "measure_actions": measure_count,
        "measure_time_s": measure_time_s,
        "channel_switches": switch_count,
        "switch_time_s": switch_time_s,
        "clear_attempts": clear_count + clear_failures,
        "clear_failures": clear_failures,
        "clear_time_s": clear_time_s,
        "first_clear_time_s": first_clear_time_s,
        "max_leg_m": max_leg_m,
        "reconstruction_error_s": final_virtual_time_s - reconstructed,
        "clear_point_oracle_distance_m": oracle.distance_m,
        "clear_point_oracle_time_s": oracle_time_s,
        "actual_to_oracle_ratio": final_virtual_time_s / max(oracle_time_s, 1e-9),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"runs": [analyze(path) for path in args.logs]}
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
