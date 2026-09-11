"""Robust experiment design for the second bearing measurement."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import fmean
from typing import Iterable, Sequence

from .geometry import Point, bearing_feasible_polygon, convex_diameter


@dataclass(frozen=True)
class SecondPointEvaluation:
    point: Point
    missed_fraction: float
    worst_diameter: float
    mean_diameter: float
    travel_distance: float
    evaluated_outcomes: int

    @property
    def robust_key(self) -> tuple[float, float, float, float]:
        """Lexicographic key used by the robust selector."""

        return (
            self.missed_fraction,
            self.worst_diameter,
            self.mean_diameter,
            self.travel_distance,
        )


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_in_ccw_convex_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    if len(polygon) < 3:
        return False
    return all(
        _cross(polygon[index], polygon[(index + 1) % len(polygon)], point) >= -1e-8
        for index in range(len(polygon))
    )


def _polygon_centroid(polygon: Sequence[Point]) -> Point:
    if not polygon:
        raise ValueError("centroid is undefined for an empty polygon")
    if len(polygon) < 3:
        return (
            fmean(point[0] for point in polygon),
            fmean(point[1] for point in polygon),
        )
    area_twice = 0.0
    x_sum = 0.0
    y_sum = 0.0
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        cross = first[0] * second[1] - second[0] * first[1]
        area_twice += cross
        x_sum += (first[0] + second[0]) * cross
        y_sum += (first[1] + second[1]) * cross
    if abs(area_twice) <= 1e-12:
        return (
            fmean(point[0] for point in polygon),
            fmean(point[1] for point in polygon),
        )
    return (x_sum / (3.0 * area_twice), y_sum / (3.0 * area_twice))


def sample_convex_polygon(
    polygon: Sequence[Point],
    *,
    spacing: float = 150.0,
    max_points: int = 96,
) -> list[Point]:
    """Return deterministic hypotheses spanning a convex feasible region."""

    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if max_points < 1:
        raise ValueError("max_points must be positive")
    if not polygon:
        return []

    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_y = min(point[1] for point in polygon)
    max_y = max(point[1] for point in polygon)
    samples: list[Point] = [_polygon_centroid(polygon)]
    samples.extend(polygon)

    x = math.floor(min_x / spacing) * spacing
    while x <= max_x + 1e-9:
        y = math.floor(min_y / spacing) * spacing
        while y <= max_y + 1e-9:
            point = (x, y)
            if _point_in_ccw_convex_polygon(point, polygon):
                samples.append(point)
            y += spacing
        x += spacing

    unique: list[Point] = []
    seen: set[tuple[int, int]] = set()
    for point in samples:
        key = (round(point[0] * 1e6), round(point[1] * 1e6))
        if key not in seen:
            seen.add(key)
            unique.append(point)
    if len(unique) <= max_points:
        return unique

    # Preserve the centroid and cover the remaining deterministic sequence
    # approximately uniformly instead of retaining only one end of the wedge.
    selected = [unique[0]]
    stride = (len(unique) - 1) / (max_points - 1) if max_points > 1 else 0.0
    for index in range(1, max_points):
        selected.append(unique[1 + round((index - 1) * stride)])
    return selected


def grid_points_in_disk(
    *,
    radius: float = 1800.0,
    spacing: float = 200.0,
) -> list[Point]:
    """Generate deterministic candidate robot locations inside the target disk."""

    if radius <= 0 or spacing <= 0:
        raise ValueError("radius and spacing must be positive")
    bound = math.floor(radius / spacing)
    points: list[Point] = []
    for ix in range(-bound, bound + 1):
        for iy in range(-bound, bound + 1):
            point = (ix * spacing, iy * spacing)
            if point[0] * point[0] + point[1] * point[1] <= radius * radius + 1e-8:
                points.append(point)
    return points


def evaluate_second_point(
    first_feasible_polygon: Sequence[Point],
    first_observer: Point,
    candidate: Point,
    source_hypotheses: Sequence[Point],
    *,
    bearing_error_deg: float = 1.0,
    second_error_scenarios_deg: Sequence[float] = (-1.0, 0.0, 1.0),
    guaranteed_receive_radius: float = 1000.0,
    near_radius: float = 5.0,
) -> SecondPointEvaluation:
    """Evaluate a second measurement point over source and error scenarios.

    A source hypothesis farther than ``guaranteed_receive_radius`` is counted as
    a missed observation.  Likewise, a source within ``near_radius`` would yield
    ``near`` rather than a bearing and is therefore not counted as a successful
    second direction measurement.
    """

    if not first_feasible_polygon:
        raise ValueError("first feasible polygon must be non-empty")
    if not source_hypotheses:
        raise ValueError("at least one source hypothesis is required")
    if guaranteed_receive_radius <= near_radius:
        raise ValueError("receive radius must exceed near radius")

    diameters: list[float] = []
    missed = 0
    measurement_count = 0
    for source in source_hypotheses:
        distance = math.hypot(source[0] - candidate[0], source[1] - candidate[1])
        if distance > guaranteed_receive_radius or distance <= near_radius:
            missed += len(second_error_scenarios_deg)
            continue
        true_bearing = math.degrees(
            math.atan2(source[1] - candidate[1], source[0] - candidate[0])
        )
        for measurement_error in second_error_scenarios_deg:
            measured_bearing = true_bearing + measurement_error
            posterior = bearing_feasible_polygon(
                first_feasible_polygon,
                candidate,
                measured_bearing,
                bearing_error_deg,
            )
            if not posterior:
                # This can occur at discretization boundaries; treating it as a
                # miss is safer than rewarding an inconsistent empty posterior.
                missed += 1
                continue
            diameter = convex_diameter(posterior).distance
            diameters.append(diameter)
            measurement_count += 1

    total = len(source_hypotheses) * len(second_error_scenarios_deg)
    if diameters:
        worst = max(diameters)
        mean = fmean(diameters)
    else:
        worst = math.inf
        mean = math.inf
    return SecondPointEvaluation(
        point=candidate,
        missed_fraction=missed / total,
        worst_diameter=worst,
        mean_diameter=mean,
        travel_distance=math.hypot(
            candidate[0] - first_observer[0], candidate[1] - first_observer[1]
        ),
        evaluated_outcomes=measurement_count,
    )


def select_second_measurement_points(
    first_feasible_polygon: Sequence[Point],
    first_observer: Point,
    candidates: Iterable[Point],
    *,
    source_hypotheses: Sequence[Point] | None = None,
    hypothesis_spacing: float = 150.0,
    max_hypotheses: int = 96,
    top_n: int = 10,
    bearing_error_deg: float = 1.0,
    guaranteed_receive_radius: float = 1000.0,
) -> list[SecondPointEvaluation]:
    """Rank candidate second points using robust lexicographic performance."""

    if top_n < 1:
        raise ValueError("top_n must be positive")
    hypotheses = list(source_hypotheses or sample_convex_polygon(
        first_feasible_polygon,
        spacing=hypothesis_spacing,
        max_points=max_hypotheses,
    ))
    evaluations = [
        evaluate_second_point(
            first_feasible_polygon,
            first_observer,
            candidate,
            hypotheses,
            bearing_error_deg=bearing_error_deg,
            guaranteed_receive_radius=guaranteed_receive_radius,
        )
        for candidate in candidates
    ]
    evaluations.sort(key=lambda evaluation: evaluation.robust_key)
    return evaluations[:top_n]
