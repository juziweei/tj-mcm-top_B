"""Build reproducible numerical examples and figures for Questions 1 and 2."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cumcm_b.geometry import (
    bearing_feasible_polygon,
    clip_polygon_halfplane,
    convex_diameter,
    disk_polygon,
    minimum_enclosing_circle,
)
from cumcm_b.planning import (
    evaluate_second_point,
    grid_points_in_disk,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts" / "q12_paper"
TARGET_RADIUS = 1800.0
MAX_RECEIVE_RADIUS = 1500.0
MIN_GUARANTEED_RADIUS = 1000.0
BEARING_ERROR_DEG = 1.0


def clip_to_convex(subject: list[tuple[float, float]], clipper: list[tuple[float, float]]):
    """Intersect two counter-clockwise convex polygons."""

    result = list(subject)
    for index, first in enumerate(clipper):
        second = clipper[(index + 1) % len(clipper)]
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        # A CCW polygon is the intersection of the half-planes left of its edges.
        result = clip_polygon_halfplane(
            result,
            dy,
            -dx,
            dy * first[0] - dx * first[1],
        )
        if not result:
            break
    return result


def circle_polygon(center: tuple[float, float], radius: float, vertices: int = 360):
    cx, cy = center
    return [
        (
            cx + radius * math.cos(2.0 * math.pi * i / vertices),
            cy + radius * math.sin(2.0 * math.pi * i / vertices),
        )
        for i in range(vertices)
    ]


def feasible_region(measurements: list[tuple[tuple[float, float], float]]):
    polygon = disk_polygon(TARGET_RADIUS, 720)
    for observer, bearing in measurements:
        polygon = clip_to_convex(
            polygon, circle_polygon(observer, MAX_RECEIVE_RADIUS, 360)
        )
        polygon = bearing_feasible_polygon(
            polygon, observer, bearing, BEARING_ERROR_DEG
        )
    return polygon


def bearing(observer: tuple[float, float], source: tuple[float, float]) -> float:
    return math.degrees(
        math.atan2(source[1] - observer[1], source[0] - observer[0])
    ) % 360.0


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    return abs(
        sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(polygon, polygon[1:] + polygon[:1])
        )
    ) / 2.0


def _font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def _canvas(title: str, xlim=(-1900.0, 1900.0), ylim=(-1900.0, 1900.0)):
    width, height = 1700, 1450
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    available_left, available_top = 220.0, 120.0
    available_width, available_height = 1360.0, 1170.0
    scale = min(
        available_width / (xlim[1] - xlim[0]),
        available_height / (ylim[1] - ylim[0]),
    )
    plot_width = (xlim[1] - xlim[0]) * scale
    plot_height = (ylim[1] - ylim[0]) * scale
    left = available_left + (available_width - plot_width) / 2.0
    top = available_top + (available_height - plot_height) / 2.0
    plot_right = left + plot_width
    plot_bottom = top + plot_height

    def transform(point):
        x, y = point
        px = left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (plot_right - left)
        py = plot_bottom - (y - ylim[0]) / (ylim[1] - ylim[0]) * (plot_bottom - top)
        return px, py

    # Light grid and axes.
    for value in range(-1500, 1501, 500):
        x0, y0 = transform((value, ylim[0]))
        x1, y1 = transform((value, ylim[1]))
        draw.line((x0, y0, x1, y1), fill="#E5E7EB", width=2)
        x0, y0 = transform((xlim[0], value))
        x1, y1 = transform((xlim[1], value))
        draw.line((x0, y0, x1, y1), fill="#E5E7EB", width=2)
    draw.rectangle((left, top, plot_right, plot_bottom), outline="#6B7280", width=3)
    draw.text((width / 2, 35), title, fill="#111827", font=_font(30, True), anchor="ma")
    draw.text((width / 2, height - 58), "x coordinate (m)", fill="#111827", font=_font(23), anchor="mm")
    draw.text((20, 88), "y coordinate (m)", fill="#111827", font=_font(21), anchor="la")
    for value in range(-1500, 1501, 500):
        px, py = transform((value, ylim[0]))
        draw.text((px, plot_bottom + 20), str(value), fill="#4B5563", font=_font(18), anchor="ma")
        px, py = transform((xlim[0], value))
        draw.text((left - 18, py), str(value), fill="#4B5563", font=_font(18), anchor="rm")
    return image, draw, transform


def _ellipse_box(transform, center, radius):
    cx, cy = transform(center)
    rx = abs(transform((center[0] + radius, center[1]))[0] - cx)
    ry = abs(transform((center[0], center[1] + radius))[1] - cy)
    return (cx - rx, cy - ry, cx + rx, cy + ry)


def _dashed(draw, start, end, fill, width=3, dash=16, gap=10):
    x0, y0 = start
    x1, y1 = end
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    position = 0.0
    while position < length:
        stop = min(position + dash, length)
        draw.line(
            (x0 + ux * position, y0 + uy * position, x0 + ux * stop, y0 + uy * stop),
            fill=fill,
            width=width,
        )
        position += dash + gap


def _viridis(value: float, low: float, high: float):
    palette = [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ]
    t = 0.0 if high <= low else min(1.0, max(0.0, (value - low) / (high - low)))
    scaled = t * (len(palette) - 1)
    index = min(len(palette) - 2, int(scaled))
    local = scaled - index
    a, b = palette[index], palette[index + 1]
    return tuple(round(a[k] * (1 - local) + b[k] * local) for k in range(3)) + (210,)


def plot_q1(metrics: dict):
    source = (500.0, 400.0)
    observers = [(-500.0, 0.0), (400.0, -600.0), (-200.0, 1000.0)]
    errors = [0.6, -0.7, 0.3]
    measurements = [
        (observer, bearing(observer, source) + error)
        for observer, error in zip(observers, errors)
    ]
    polygon = feasible_region(measurements)
    diameter = convex_diameter(polygon)
    enclosing = minimum_enclosing_circle(polygon)

    metrics["q1_example"] = {
        "true_source": source,
        "observers": observers,
        "measurement_errors_deg": errors,
        "measured_bearings_deg": [round(value, 6) for _, value in measurements],
        "vertices": len(polygon),
        "area_m2": polygon_area(polygon),
        "diameter_m": diameter.distance,
        "diameter_endpoints": [diameter.first, diameter.second],
        "minimum_enclosing_circle_center": enclosing.center,
        "minimum_enclosing_circle_radius_m": enclosing.radius,
        "covered_by_diameter_circle": enclosing.radius <= diameter.distance / 2.0 + 1e-6,
    }

    image, draw, transform = _canvas(
        "Question 1  Feasible set from bounded bearing measurements",
        xlim=(-1900.0, 1900.0),
        ylim=(-1500.0, 1900.0),
    )
    draw.ellipse(_ellipse_box(transform, (0, 0), TARGET_RADIUS), outline="#9CA3AF", width=3)
    pixels = [transform(point) for point in polygon]
    draw.polygon(pixels, fill=(147, 197, 253, 145), outline="#2563EB", width=4)
    colors = ["#0F766E", "#7C3AED", "#C2410C"]
    for idx, ((ox, oy), measured) in enumerate(measurements, start=1):
        px, py = transform((ox, oy))
        draw.ellipse((px - 9, py - 9, px + 9, py + 9), fill=colors[idx - 1])
        draw.text((px + 16, py - 28), f"S{idx}", fill="#111827", font=_font(22))
        for offset in (-BEARING_ERROR_DEG, BEARING_ERROR_DEG):
            angle = math.radians(measured + offset)
            _dashed(
                draw,
                transform((ox, oy)),
                transform((ox + 1650 * math.cos(angle), oy + 1650 * math.sin(angle))),
                colors[idx - 1],
                width=3,
            )
    sx, sy = transform(source)
    draw.regular_polygon((sx, sy, 16), n_sides=5, rotation=-90, fill="#111827")
    draw.line((*transform(diameter.first), *transform(diameter.second)), fill="#DC2626", width=7)
    draw.ellipse(_ellipse_box(transform, enclosing.center, enclosing.radius), outline="#D97706", width=4)
    # Zoom inset so the localization region and its two size measures remain legible.
    inset = (270, 850, 760, 1240)
    draw.rounded_rectangle(inset, radius=12, fill=(255, 255, 255, 238), outline="#9CA3AF", width=3)
    draw.text((inset[0] + 18, inset[1] + 14), "Localization-region detail", fill="#111827", font=_font(20, True))
    local_min_x = min(point[0] for point in polygon) - 12.0
    local_max_x = max(point[0] for point in polygon) + 12.0
    local_min_y = min(point[1] for point in polygon) - 12.0
    local_max_y = max(point[1] for point in polygon) + 12.0
    local_scale = min(
        (inset[2] - inset[0] - 45) / (local_max_x - local_min_x),
        (inset[3] - inset[1] - 80) / (local_max_y - local_min_y),
    )
    def local_transform(point):
        x, y = point
        return (
            inset[0] + 22 + (x - local_min_x) * local_scale,
            inset[3] - 22 - (y - local_min_y) * local_scale,
        )
    draw.polygon([local_transform(point) for point in polygon], fill=(147, 197, 253, 180), outline="#2563EB", width=4)
    draw.line((*local_transform(diameter.first), *local_transform(diameter.second)), fill="#DC2626", width=6)
    lc = local_transform(enclosing.center)
    lr = enclosing.radius * local_scale
    draw.ellipse((lc[0] - lr, lc[1] - lr, lc[0] + lr, lc[1] + lr), outline="#D97706", width=4)
    ls = local_transform(source)
    draw.regular_polygon((ls[0], ls[1], 11), n_sides=5, rotation=-90, fill="#111827")
    draw.rounded_rectangle((1030, 1040, 1555, 1210), radius=12, fill=(255, 255, 255, 230), outline="#D1D5DB", width=2)
    draw.text((1052, 1062), "Blue area: feasible region", fill="#1F2937", font=_font(20))
    draw.text((1052, 1104), f"Red segment: D = {diameter.distance:.1f} m", fill="#DC2626", font=_font(20))
    draw.text((1052, 1146), f"Orange circle: R = {enclosing.radius:.1f} m", fill="#B45309", font=_font(20))
    image.save(OUT / "q1_localization.png")


def plot_q2(metrics: dict):
    first_observer = (0.0, 0.0)
    first_bearing = 35.0
    first_region = feasible_region([(first_observer, first_bearing)])
    # The initial wedge is narrow, so a Cartesian grid can undersample it.
    # A polar lattice spans both range and angular uncertainty uniformly.
    hypotheses = []
    for source_range in np.arange(25.0, 1500.0, 50.0):
        for angular_offset in (-1.0, -0.5, 0.0, 0.5, 1.0):
            angle = math.radians(first_bearing + angular_offset)
            hypotheses.append(
                (
                    source_range * math.cos(angle),
                    source_range * math.sin(angle),
                )
            )
    candidates = grid_points_in_disk(radius=TARGET_RADIUS, spacing=100.0)
    evaluations = [
        evaluate_second_point(
            first_region,
            first_observer,
            candidate,
            hypotheses,
            bearing_error_deg=BEARING_ERROR_DEG,
            guaranteed_receive_radius=MIN_GUARANTEED_RADIUS,
        )
        for candidate in candidates
    ]
    evaluations.sort(key=lambda item: item.robust_key)
    top = evaluations[:10]
    best = top[0]
    metrics["q2_example"] = {
        "first_observer": first_observer,
        "first_bearing_deg": first_bearing,
        "hypotheses": len(hypotheses),
        "candidate_grid_spacing_m": 100.0,
        "evaluated_candidates": len(evaluations),
        "best": {
            "point": best.point,
            "missed_fraction": best.missed_fraction,
            "worst_diameter_m": best.worst_diameter,
            "mean_diameter_m": best.mean_diameter,
            "travel_distance_m": best.travel_distance,
        },
        "top10": [
            {
                "rank": rank,
                "x_m": item.point[0],
                "y_m": item.point[1],
                "missed_fraction": item.missed_fraction,
                "worst_diameter_m": item.worst_diameter,
                "mean_diameter_m": item.mean_diameter,
                "travel_distance_m": item.travel_distance,
            }
            for rank, item in enumerate(top, start=1)
        ],
    }

    zero_miss = [item for item in evaluations if item.missed_fraction <= 1e-12]
    finite_worst = np.array([item.worst_diameter for item in zero_miss])
    threshold = float(np.quantile(finite_worst, 0.20))
    candidate_region = [item for item in zero_miss if item.worst_diameter <= threshold]
    metrics["q2_example"]["zero_miss_candidates"] = len(zero_miss)
    metrics["q2_example"]["recommended_region_candidates"] = len(candidate_region)
    metrics["q2_example"]["recommended_worst_diameter_threshold_m"] = threshold

    image, draw, transform = _canvas(
        "Question 2  Robust candidate region for the second measurement"
    )
    draw.ellipse(_ellipse_box(transform, (0, 0), TARGET_RADIUS), outline="#9CA3AF", width=3)
    draw.polygon([transform(point) for point in first_region], fill=(219, 234, 254, 190), outline="#60A5FA", width=3)
    if zero_miss:
        low = min(item.worst_diameter for item in zero_miss)
        high = max(item.worst_diameter for item in zero_miss)
        for item in zero_miss:
            px, py = transform(item.point)
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=_viridis(item.worst_diameter, low, high))
    if candidate_region:
        for item in candidate_region:
            px, py = transform(item.point)
            draw.ellipse((px - 10, py - 10, px + 10, py + 10), outline="#F97316", width=3)
    px, py = transform((0, 0))
    draw.rectangle((px - 10, py - 10, px + 10, py + 10), fill="#111827")
    bx, by = transform(best.point)
    draw.regular_polygon((bx, by, 18), n_sides=5, rotation=-90, fill="#DC2626")
    draw.text((bx + 20, by - 35), f"best ({best.point[0]:.0f}, {best.point[1]:.0f})", fill="#991B1B", font=_font(20, True))
    draw.rounded_rectangle((1000, 1005, 1560, 1185), radius=12, fill=(255, 255, 255, 230), outline="#D1D5DB", width=2)
    draw.text((1020, 1025), "Blue wedge: initial feasible region", fill="#1F2937", font=_font(19))
    draw.text((1020, 1065), "Colored dots: guaranteed reception", fill="#1F2937", font=_font(19))
    draw.text((1020, 1105), "Orange rings: best 20% robust set", fill="#C2410C", font=_font(19))
    draw.text((1020, 1145), "Red star: lexicographic optimum", fill="#991B1B", font=_font(19))
    image.save(OUT / "q2_candidate_region.png")

    # Local geometry experiment: hold the second-source range fixed and vary
    # the crossing angle. This isolates the conditioning effect of geometry.
    nominal_source = (1000.0, 0.0)
    initial = feasible_region([((0.0, 0.0), 0.0)])
    angles = np.arange(10.0, 91.0, 5.0)
    diameters = []
    for alpha in angles:
        second = (
            nominal_source[0] - 700.0 * math.cos(math.radians(alpha)),
            nominal_source[1] + 700.0 * math.sin(math.radians(alpha)),
        )
        posterior = bearing_feasible_polygon(
            initial, second, bearing(second, nominal_source), BEARING_ERROR_DEG
        )
        diameters.append(convex_diameter(posterior).distance)
    metrics["q2_crossing_angle_sensitivity"] = [
        {"crossing_angle_deg": float(angle), "diameter_m": float(value)}
        for angle, value in zip(angles, diameters)
    ]

    image = Image.new("RGB", (1700, 980), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    left, top, right, bottom = 155, 115, 1580, 820
    draw.rectangle((left, top, right, bottom), outline="#6B7280", width=3)
    draw.text((850, 40), "Localization improves as the crossing angle approaches 90 degrees", fill="#111827", font=_font(29, True), anchor="ma")
    min_y, max_y = min(diameters) * 0.92, max(diameters) * 1.05
    def xy(angle, value):
        return (
            left + (angle - 10.0) / 80.0 * (right - left),
            bottom - (value - min_y) / (max_y - min_y) * (bottom - top),
        )
    for angle in range(10, 91, 10):
        px, _ = xy(angle, min_y)
        draw.line((px, top, px, bottom), fill="#E5E7EB", width=2)
        draw.text((px, bottom + 20), str(angle), fill="#4B5563", font=_font(18), anchor="ma")
    for value in np.linspace(min_y, max_y, 6):
        _, py = xy(10.0, float(value))
        draw.line((left, py, right, py), fill="#E5E7EB", width=2)
        draw.text((left - 18, py), f"{value:.0f}", fill="#4B5563", font=_font(18), anchor="rm")
    points = [xy(float(a), float(d)) for a, d in zip(angles, diameters)]
    draw.line(points, fill="#2563EB", width=6, joint="curve")
    for px, py in points:
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill="#2563EB")
    best_index = int(np.argmin(diameters))
    bx, by = points[best_index]
    draw.ellipse((bx - 11, by - 11, bx + 11, by + 11), fill="#DC2626")
    draw.text((bx - 10, by - 48), f"minimum {diameters[best_index]:.1f} m", fill="#991B1B", font=_font(20, True), anchor="ra")
    draw.text((850, 920), "Crossing angle between two lines of sight (deg)", fill="#111827", font=_font(22), anchor="mm")
    draw.text((left, 82), "Posterior diameter (m)", fill="#111827", font=_font(20), anchor="la")
    image.save(OUT / "q2_angle_sensitivity.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics: dict = {
        "constants": {
            "target_radius_m": TARGET_RADIUS,
            "bearing_error_deg": BEARING_ERROR_DEG,
            "maximum_receive_radius_m": MAX_RECEIVE_RADIUS,
            "minimum_guaranteed_receive_radius_m": MIN_GUARANTEED_RADIUS,
        }
    }
    plot_q1(metrics)
    plot_q2(metrics)
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
