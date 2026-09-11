import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm_b.geometry import (  # noqa: E402
    bearing_feasible_polygon,
    convex_diameter,
    disk_polygon,
    localization_polygon,
    minimum_enclosing_circle,
)
from cumcm_b.planning import evaluate_second_point  # noqa: E402


class BearingWedgeTests(unittest.TestCase):
    def test_east_bearing_keeps_positive_x_axis(self):
        square = [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)]
        feasible = bearing_feasible_polygon(square, (0.0, 0.0), 0.0, 1.0)
        self.assertTrue(feasible)
        self.assertTrue(all(point[0] >= -1e-7 for point in feasible))
        self.assertLess(max(abs(point[1]) for point in feasible), 0.18)

    def test_wraparound_bearing_contains_expected_ray(self):
        disk = disk_polygon(100.0, 360)
        feasible = bearing_feasible_polygon(disk, (0.0, 0.0), 359.5, 1.0)
        expected = (80.0 * math.cos(math.radians(359.5)), 80.0 * math.sin(math.radians(359.5)))
        # The expected point lies inside every half-plane, so the clipped polygon
        # must extend approximately to the same radius along that ray.
        self.assertGreater(max(math.hypot(x, y) for x, y in feasible), 99.0)
        self.assertGreater(expected[0], 0.0)

    def test_two_bearings_retain_true_target(self):
        target = (600.0, 800.0)
        first = math.degrees(math.atan2(target[1], target[0]))
        second_observer = (800.0, 0.0)
        second = math.degrees(
            math.atan2(target[1] - second_observer[1], target[0] - second_observer[0])
        )
        polygon = localization_polygon(
            [((0.0, 0.0), first), (second_observer, second)],
            error_deg=1.0,
            disk_vertices=1440,
        )
        self.assertTrue(polygon)
        # A point is in a CCW convex polygon iff it lies left of every edge.
        for index, a in enumerate(polygon):
            b = polygon[(index + 1) % len(polygon)]
            cross = (b[0] - a[0]) * (target[1] - a[1]) - (b[1] - a[1]) * (target[0] - a[0])
            self.assertGreaterEqual(cross, -1e-5)


class DiameterAndCircleTests(unittest.TestCase):
    def test_square_diameter(self):
        square = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        result = convex_diameter(square)
        self.assertAlmostEqual(result.distance, math.sqrt(8.0), places=9)

    def test_segment_is_covered_by_diameter_circle(self):
        segment = [(0.0, 0.0), (4.0, 0.0)]
        diameter = convex_diameter(segment)
        circle = minimum_enclosing_circle(segment)
        self.assertAlmostEqual(circle.radius, diameter.distance / 2.0, places=9)

    def test_equilateral_triangle_needs_larger_radius(self):
        height = math.sqrt(3.0)
        triangle = [(0.0, 0.0), (2.0, 0.0), (1.0, height)]
        diameter = convex_diameter(triangle)
        circle = minimum_enclosing_circle(triangle)
        self.assertAlmostEqual(diameter.distance, 2.0, places=9)
        self.assertAlmostEqual(circle.radius, 2.0 / math.sqrt(3.0), places=8)
        self.assertGreater(circle.radius, diameter.distance / 2.0)


class SecondPointPlanningTests(unittest.TestCase):
    def test_perpendicular_baseline_improves_localization(self):
        first_polygon = localization_polygon(
            [((0.0, 0.0), 0.0)],
            error_deg=1.0,
            target_radius=1200.0,
            disk_vertices=720,
        )
        hypotheses = [(700.0, 0.0), (900.0, 0.0), (1100.0, 0.0)]
        repeated = evaluate_second_point(
            first_polygon,
            (0.0, 0.0),
            (0.0, 0.0),
            hypotheses,
            guaranteed_receive_radius=1500.0,
        )
        displaced = evaluate_second_point(
            first_polygon,
            (0.0, 0.0),
            (700.0, 700.0),
            hypotheses,
            guaranteed_receive_radius=1500.0,
        )
        self.assertLess(displaced.worst_diameter, repeated.worst_diameter)

    def test_out_of_range_hypotheses_are_counted(self):
        first_polygon = localization_polygon(
            [((0.0, 0.0), 0.0)],
            target_radius=1200.0,
        )
        evaluation = evaluate_second_point(
            first_polygon,
            (0.0, 0.0),
            (-1000.0, 0.0),
            [(1000.0, 0.0)],
            guaranteed_receive_radius=1000.0,
        )
        self.assertEqual(evaluation.missed_fraction, 1.0)
        self.assertTrue(math.isinf(evaluation.worst_diameter))


if __name__ == "__main__":
    unittest.main()
