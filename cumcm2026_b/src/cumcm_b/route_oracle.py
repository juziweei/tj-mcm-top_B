"""Exact small-instance route bounds for policy benchmarking.

The official task contains at most 16 sources.  That is small enough to solve
the origin-fixed, free-end Euclidean Hamiltonian path exactly with Held--Karp.
The solver is deliberately kept separate from the controller: it is an oracle
used to measure avoidable travel, not a source of hidden information at run
time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .geometry import Point


@dataclass(frozen=True)
class OpenRoute:
    distance_m: float
    order: tuple[int, ...]


def exact_open_route_costs(
    start_costs: Sequence[float], transition_costs: Sequence[Sequence[float]]
) -> OpenRoute:
    """Solve an exact free-end path for an arbitrary directed cost matrix."""

    count = len(start_costs)
    if count == 0:
        return OpenRoute(0.0, ())
    if count > 16:
        raise ValueError("exact Held--Karp route is limited to 16 targets")
    if len(transition_costs) != count or any(
        len(row) != count for row in transition_costs
    ):
        raise ValueError("transition_costs must be a square matrix")
    if any(cost < 0.0 or not math.isfinite(cost) for cost in start_costs):
        raise ValueError("start costs must be finite and non-negative")

    state_count = (1 << count) * count
    costs = [math.inf] * state_count
    parents = [-1] * state_count
    for last, cost in enumerate(start_costs):
        costs[((1 << last) * count) + last] = cost

    full_mask = (1 << count) - 1
    for mask in range(1, full_mask + 1):
        remaining = full_mask ^ mask
        present_bits = mask
        while present_bits:
            last_bit = present_bits & -present_bits
            last = last_bit.bit_length() - 1
            base_cost = costs[mask * count + last]
            if math.isfinite(base_cost):
                candidate_bits = remaining
                while candidate_bits:
                    next_bit = candidate_bits & -candidate_bits
                    nxt = next_bit.bit_length() - 1
                    next_mask = mask | next_bit
                    next_index = next_mask * count + nxt
                    proposed = base_cost + transition_costs[last][nxt]
                    if proposed < costs[next_index]:
                        costs[next_index] = proposed
                        parents[next_index] = last
                    candidate_bits ^= next_bit
            present_bits ^= last_bit

    last = min(range(count), key=lambda index: costs[full_mask * count + index])
    optimum = costs[full_mask * count + last]
    reverse_order: list[int] = []
    mask = full_mask
    while last >= 0:
        reverse_order.append(last)
        parent = parents[mask * count + last]
        mask ^= 1 << last
        last = parent
    return OpenRoute(optimum, tuple(reversed(reverse_order)))


def euclidean_distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def route_distance(
    targets: Sequence[Point], order: Sequence[int], *, start: Point = (0.0, 0.0)
) -> float:
    position = start
    total = 0.0
    for index in order:
        total += euclidean_distance(position, targets[index])
        position = targets[index]
    return total


def exact_open_route(
    targets: Sequence[Point], *, start: Point = (0.0, 0.0)
) -> OpenRoute:
    """Return the exact shortest path from ``start`` visiting every target.

    The route does not return to the start.  Indices in ``order`` address the
    input ``targets`` sequence.  The implementation is intended for the
    problem's hard limit of 16 sources; larger inputs are rejected instead of
    silently returning a heuristic answer under an "exact" name.
    """

    count = len(targets)
    if count == 0:
        return OpenRoute(0.0, ())
    if count > 16:
        raise ValueError("exact Held--Karp route is limited to 16 targets")

    pairwise = [
        [euclidean_distance(first, second) for second in targets]
        for first in targets
    ]
    start_costs = [euclidean_distance(start, target) for target in targets]
    return exact_open_route_costs(start_costs, pairwise)


def fast_open_route(
    targets: Sequence[Point], *, start: Point = (0.0, 0.0)
) -> OpenRoute:
    """Return a deterministic nearest-neighbour plus 2-opt open route."""

    if not targets:
        return OpenRoute(0.0, ())
    remaining = set(range(len(targets)))
    order: list[int] = []
    position = start
    while remaining:
        nxt = min(
            remaining,
            key=lambda index: (euclidean_distance(position, targets[index]), index),
        )
        order.append(nxt)
        remaining.remove(nxt)
        position = targets[nxt]

    changed = True
    while changed:
        changed = False
        points = [start, *(targets[index] for index in order)]
        for left in range(1, len(points) - 1):
            for right in range(left + 1, len(points)):
                old = euclidean_distance(points[left - 1], points[left])
                new = euclidean_distance(points[left - 1], points[right])
                if right + 1 < len(points):
                    old += euclidean_distance(points[right], points[right + 1])
                    new += euclidean_distance(points[left], points[right + 1])
                if new + 1e-9 < old:
                    order[left - 1 : right] = reversed(order[left - 1 : right])
                    changed = True
                    break
            if changed:
                break
    return OpenRoute(route_distance(targets, order, start=start), tuple(order))


def mst_lower_bound(
    targets: Sequence[Point], *, start: Point = (0.0, 0.0)
) -> float:
    """Return a rigorous Euclidean lower bound on any complete visit path."""

    points = [start, *targets]
    if len(points) <= 1:
        return 0.0
    used = [False] * len(points)
    best = [math.inf] * len(points)
    best[0] = 0.0
    total = 0.0
    for _ in points:
        node = min(
            (index for index in range(len(points)) if not used[index]),
            key=best.__getitem__,
        )
        used[node] = True
        total += best[node]
        for other in range(len(points)):
            if not used[other]:
                best[other] = min(
                    best[other], euclidean_distance(points[node], points[other])
                )
    return total
