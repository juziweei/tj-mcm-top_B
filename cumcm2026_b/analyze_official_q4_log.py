"""Decompose an official Q4 client log without hidden simulator state."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cumcm_b.route_oracle import exact_open_route  # noqa: E402


def analyze(log_path: Path, summary_path: Path) -> dict[str, object]:
    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    position = (0.0, 0.0)
    total_travel_m = 0.0
    travel_at_last_clear_m = 0.0
    clear_positions: list[tuple[float, float]] = []
    clear_channels: list[int] = []
    first_detection_s: dict[int, float] = {}
    successful_clear_s: dict[int, float] = {}

    for record in records:
        path = record.get("path")
        if path not in {"/measure", "/clear"}:
            continue
        request = record["request"]
        target = (
            float(request["position"]["x"]),
            float(request["position"]["y"]),
        )
        total_travel_m += math.dist(position, target)
        position = target
        response = record["response"]
        channel = int(request["channel"])
        virtual_time_s = float(response["virtual_time_s"])
        if (
            path == "/measure"
            and response.get("measure_result") in {"direction", "near"}
        ):
            first_detection_s.setdefault(channel, virtual_time_s)
        if path == "/clear" and response.get("clear_result") == "success":
            clear_positions.append(target)
            clear_channels.append(channel)
            successful_clear_s[channel] = virtual_time_s
            travel_at_last_clear_m = total_travel_m

    execution_clear_route_m = sum(
        math.dist(first, second)
        for first, second in zip(
            [(0.0, 0.0), *clear_positions[:-1]], clear_positions
        )
    )
    clear_oracle_m = (
        exact_open_route(clear_positions, start=(0.0, 0.0)).distance_m
        if clear_positions
        else 0.0
    )
    last_clear_s = max(successful_clear_s.values(), default=0.0)
    virtual_time_s = float(summary["virtual_time_s"])
    cleared_count = len(clear_positions)
    official_sources = int(summary["official_statistics"]["jammer_count"])
    return {
        "log": str(log_path),
        "sources": official_sources,
        "cleared": cleared_count,
        "virtual_time_s": virtual_time_s,
        "virtual_time_s_per_true_source": virtual_time_s / official_sources,
        "total_travel_m": total_travel_m,
        "travel_time_share": total_travel_m / 5.0 / virtual_time_s,
        "execution_clear_route_m": execution_clear_route_m,
        "clear_position_oracle_m": clear_oracle_m,
        "clear_order_excess_m": execution_clear_route_m - clear_oracle_m,
        "between_clear_detour_m": travel_at_last_clear_m - execution_clear_route_m,
        "post_last_clear_travel_m": total_travel_m - travel_at_last_clear_m,
        "last_first_detection_s": max(first_detection_s.values(), default=0.0),
        "last_successful_clear_s": last_clear_s,
        "post_last_clear_virtual_time_s": virtual_time_s - last_clear_s,
        "clear_channels_in_order": clear_channels,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.log, args.summary)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
