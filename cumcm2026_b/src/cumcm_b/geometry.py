"""Robust planar geometry for bearing-only localization.

Angles follow the problem convention: zero degrees points east and positive
angles rotate counter-clockwise.  Polygons are represented by vertices in
counter-clockwise order.  The target disk is approximated by a regular polygon;
all subsequent bearing updates are exact half-plane clips of that polygon.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence

Point = tuple[float, float]
_EPS = 1e-9


@dataclass(frozen=True)
class DiameterResult:
    distance: float
    first: Point
    second: Point


@dataclass(frozen=True)
class Circle:
    center: Point
    radius: float


def _cross(a: Point, b: Point, c: Point) -> float:
    """Return cross((b-a), (c-a))."""

    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _distance_sq(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def disk_polygon(radius: float = 1800.0, vertices: int = 720) -> list[Point]:
    """Return a counter-clockwise regular-polygon approximation of a disk."""

    if radius <= 0:
        raise ValueError("radius must be positive")
    if vertices < 3:
        raise ValueError("vertices must be at least three")
    return [
        (
            radius * math.cos(2.0 * math.pi * i / vertices),
            radius * math.sin(2.0 * math.pi * i / vertices),
        )
        for i in range(vertices)
    ]


def clip_polygon_halfplane(
    polygon: Sequence[Point],
    a: float,
    b: float,
    c: float,
    *,
    tolerance: float = _EPS,
) -> list[Point]:
    """Clip a convex polygon by the closed half-plane ``a*x+b*y <= c``."""

    if not polygon:
        return []
    norm_sq = a * a + b * b
    if norm_sq <= tolerance * tolerance:
        raise ValueError("half-plane normal must be non-zero")

    def value(point: Point) -> float:
        return a * point[0] + b * point[1] - c

    output: list[Point] = []
    previous = polygon[-1]
    previous_value = value(previous)
    previous_inside = previous_value <= tolerance

    for current in polygon:
        current_value = value(current)
        current_inside = current_value <= tolerance
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > tolerance:
                ratio = previous_value / denominator
                output.append(
                    (
                        previous[0] + ratio * (current[0] - previous[0]),
                        previous[1] + ratio * (current[1] - previous[1]),
                    )
                )
        if current_inside:
            output.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside

    return _deduplicate_adjacent(output, tolerance)


def _deduplicate_adjacent(points: Sequence[Point], tolerance: float) -> list[Point]:
    if not points:
        return []
    tolerance_sq = tolerance * tolerance
    result = [points[0]]
    for point in points[1:]:
        if _distance_sq(point, result[-1]) > tolerance_sq:
            result.append(point)
    if len(result) > 1 and _distance_sq(result[0], result[-1]) <= tolerance_sq:
        result.pop()
    return result


def bearing_feasible_polygon(
    polygon: Sequence[Point],
    observer: Point,
    bearing_deg: float,
    error_deg: float = 1.0,
) -> list[Point]:
    """Intersect ``polygon`` with a bearing wedge.

    The wedge contains rays whose counter-clockwise angles lie between
    ``bearing_deg-error_deg`` and ``bearing_deg+error_deg``.  The supported error
    must be strictly smaller than 90 degrees, which is the only case needed by
    the problem and avoids the non-convex wide-wedge case.
    """

    if not 0.0 <= error_deg < 90.0:
        raise ValueError("error_deg must lie in [0, 90)")
    angle = math.radians(bearing_deg)
    error = math.radians(error_deg)
    lower = (math.cos(angle - error), math.sin(angle - error))
    upper = (math.cos(angle + error), math.sin(angle + error))
    ox, oy = observer

    # Left of the lower ray: cross(lower, q-observer) >= 0.
    clipped = clip_polygon_halfplane(
        polygon,
        lower[1],
        -lower[0],
        lower[1] * ox - lower[0] * oy,
    )
    # Right of the upper ray: cross(upper, q-observer) <= 0.
    return clip_polygon_halfplane(
        clipped,
        -upper[1],
        upper[0],
        -upper[1] * ox + upper[0] * oy,
    )


def localization_polygon(
    measurements: Iterable[tuple[Point, float]],
    *,
    error_deg: float = 1.0,
    target_radius: float = 1800.0,
    disk_vertices: int = 720,
) -> list[Point]:
    """Compute the common feasible region of several bearing measurements."""

    polygon = disk_polygon(target_radius, disk_vertices)
    for observer, bearing_deg in measurements:
        polygon = bearing_feasible_polygon(
            polygon, observer, bearing_deg, error_deg
        )
        if not polygon:
            break
    return polygon


def convex_diameter(polygon: Sequence[Point]) -> DiameterResult:
    """Compute the Euclidean diameter of a CCW convex polygon.

    Uses rotating calipers and therefore runs in O(n) time after the polygon is
    available in convex cyclic order.
    """

    n = len(polygon)
    if n == 0:
        raise ValueError("diameter is undefined for an empty polygon")
    if n == 1:
        return DiameterResult(0.0, polygon[0], polygon[0])
    if n == 2:
        return DiameterResult(
            math.sqrt(_distance_sq(polygon[0], polygon[1])),
            polygon[0],
            polygon[1],
        )

    best_sq = -1.0
    best_pair = (polygon[0], polygon[1])
    j = 1

    def update(first: Point, second: Point) -> None:
        nonlocal best_sq, best_pair
        distance_sq = _distance_sq(first, second)
        if distance_sq > best_sq:
            best_sq = distance_sq
            best_pair = (first, second)

    for i in range(n):
        next_i = (i + 1) % n
        while True:
            next_j = (j + 1) % n
            current_area = abs(_cross(polygon[i], polygon[next_i], polygon[j]))
            next_area = abs(_cross(polygon[i], polygon[next_i], polygon[next_j]))
            if next_area > current_area + _EPS:
                j = next_j
            else:
                break
        update(polygon[i], polygon[j])
        update(polygon[next_i], polygon[j])

    return DiameterResult(math.sqrt(best_sq), best_pair[0], best_pair[1])


def _contains(circle: Circle, point: Point, tolerance: float = 1e-7) -> bool:
    return _distance_sq(circle.center, point) <= (circle.radius + tolerance) ** 2


def _diameter_circle(a: Point, b: Point) -> Circle:
    center = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    return Circle(center, math.sqrt(_distance_sq(a, b)) / 2.0)


def _circumcircle(a: Point, b: Point, c: Point) -> Circle | None:
    determinant = 2.0 * (
        a[0] * (b[1] - c[1])
        + b[0] * (c[1] - a[1])
        + c[0] * (a[1] - b[1])
    )
    if abs(determinant) <= _EPS:
        return None
    a_sq = a[0] * a[0] + a[1] * a[1]
    b_sq = b[0] * b[0] + b[1] * b[1]
    c_sq = c[0] * c[0] + c[1] * c[1]
    ux = (
        a_sq * (b[1] - c[1])
        + b_sq * (c[1] - a[1])
        + c_sq * (a[1] - b[1])
    ) / determinant
    uy = (
        a_sq * (c[0] - b[0])
        + b_sq * (a[0] - c[0])
        + c_sq * (b[0] - a[0])
    ) / determinant
    center = (ux, uy)
    return Circle(center, math.sqrt(_distance_sq(center, a)))


def _circle_from_two_boundary_points(
    points: Sequence[Point], first: Point, second: Point
) -> Circle:
    diameter = _diameter_circle(first, second)
    if all(_contains(diameter, point) for point in points):
        return diameter

    left: Circle | None = None
    right: Circle | None = None
    baseline_cross_sign = lambda point: _cross(first, second, point)
    for point in points:
        if _contains(diameter, point):
            continue
        candidate = _circumcircle(first, second, point)
        if candidate is None:
            continue
        side = baseline_cross_sign(point)
        center_side = baseline_cross_sign(candidate.center)
        if side > 0.0:
            if left is None or center_side > baseline_cross_sign(left.center):
                left = candidate
        elif side < 0.0:
            if right is None or center_side < baseline_cross_sign(right.center):
                right = candidate

    candidates = [circle for circle in (left, right) if circle is not None]
    if not candidates:
        return diameter
    return min(candidates, key=lambda circle: circle.radius)


def _circle_from_one_boundary_point(
    points: Sequence[Point], boundary: Point
) -> Circle:
    circle = Circle(boundary, 0.0)
    for index, point in enumerate(points):
        if _contains(circle, point):
            continue
        if circle.radius <= _EPS:
            circle = _diameter_circle(boundary, point)
        else:
            circle = _circle_from_two_boundary_points(
                points[: index + 1], boundary, point
            )
    return circle


def minimum_enclosing_circle(
    points: Sequence[Point], *, seed: int = 2026
) -> Circle:
    """Return the smallest circle enclosing all points.

    This is the randomized incremental algorithm with a fixed default seed, so
    runs are reproducible while retaining expected linear complexity.
    """

    if not points:
        raise ValueError("minimum enclosing circle needs at least one point")
    shuffled = list(points)
    random.Random(seed).shuffle(shuffled)
    circle: Circle | None = None
    for index, point in enumerate(shuffled):
        if circle is None or not _contains(circle, point):
            circle = _circle_from_one_boundary_point(
                shuffled[: index + 1], point
            )
    assert circle is not None
    return circle

