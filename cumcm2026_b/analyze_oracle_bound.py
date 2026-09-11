"""Estimate physical lower bounds for Q3 from omniscient source locations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cumcm_b.simulator import generate_sources  # noqa: E402


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mst_length(points):
    """Euclidean MST length, a rigorous lower bound on any clearing path."""

    count = len(points)
    used = [False] * count
    best = [math.inf] * count
    best[0] = 0.0
    total = 0.0
    for _ in range(count):
        node = min((i for i in range(count) if not used[i]), key=best.__getitem__)
        used[node] = True
        total += best[node]
        for other in range(count):
            if not used[other]:
                best[other] = min(best[other], distance(points[node], points[other]))
    return total


def improve_open_path(points, order):
    """2-opt an origin-fixed, free-end path to estimate the omniscient route."""

    order = list(order)
    changed = True
    while changed:
        changed = False
        for left in range(1, len(order) - 1):
            for right in range(left + 1, len(order)):
                a = points[order[left - 1]]
                b = points[order[left]]
                old = distance(a, b)
                new = distance(a, points[order[right]])
                if right + 1 < len(order):
                    c = points[order[right]]
                    d = points[order[right + 1]]
                    old += distance(c, d)
                    new += distance(b, d)
                if new + 1e-9 < old:
                    order[left : right + 1] = reversed(order[left : right + 1])
                    changed = True
    return order


def path_length(points, order):
    return sum(distance(points[a], points[b]) for a, b in zip(order, order[1:]))


def approximate_open_route(points, rng, restarts=8):
    source_ids = list(range(1, len(points)))
    orders = []
    remaining = set(source_ids)
    nearest = [0]
    while remaining:
        nxt = min(remaining, key=lambda node: distance(points[nearest[-1]], points[node]))
        nearest.append(nxt)
        remaining.remove(nxt)
    orders.append(nearest)
    orders.append([0, *sorted(source_ids, key=lambda i: math.atan2(points[i][1], points[i][0]))])
    for _ in range(max(0, restarts - len(orders))):
        shuffled = list(source_ids)
        rng.shuffle(shuffled)
        orders.append([0, *shuffled])
    improved = [improve_open_path(points, order) for order in orders]
    return min(path_length(points, order) for order in improved)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed-start", type=int, default=80_000_000)
    args = parser.parse_args()
    rng = random.Random(2026)
    rows = []
    for offset in range(args.episodes):
        sources = generate_sources(
            args.seed_start + offset, directional_probability=0.0, error_modes=0
        )
        points = [(0.0, 0.0), *(source.position for source in sources)]
        lower_distance = mst_length(points)
        route_distance = approximate_open_route(points, rng)
        count = len(sources)
        rows.append(
            {
                "count": count,
                "mst_lower_s_per_source": lower_distance / 5.0 / count + 5.0,
                "oracle_route_s_per_source": route_distance / 5.0 / count + 5.0,
            }
        )
    lower = [row["mst_lower_s_per_source"] for row in rows]
    route = [row["oracle_route_s_per_source"] for row in rows]
    report = {
        "episodes": len(rows),
        "assumption": "omniscient positions, zero detection and switch time",
        "mst_lower_bound_mean_s_per_source": statistics.fmean(lower),
        "mst_lower_bound_median_s_per_source": statistics.median(lower),
        "mst_lower_bound_p95_s_per_source": sorted(lower)[int(0.95 * (len(lower) - 1))],
        "fraction_with_rigorous_lower_bound_above_100s": sum(x > 100.0 for x in lower) / len(lower),
        "approx_oracle_route_mean_s_per_source": statistics.fmean(route),
        "approx_oracle_route_median_s_per_source": statistics.median(route),
        "approx_oracle_route_p95_s_per_source": sorted(route)[int(0.95 * (len(route) - 1))],
        "fraction_approx_oracle_route_above_100s": sum(x > 100.0 for x in route) / len(route),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
