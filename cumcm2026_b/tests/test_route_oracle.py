from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm_b.route_oracle import (  # noqa: E402
    exact_open_route,
    exact_open_route_costs,
    fast_open_route,
    route_distance,
)


class ExactOpenRouteTests(unittest.TestCase):
    def test_empty_route(self):
        self.assertEqual(exact_open_route([]).distance_m, 0.0)
        self.assertEqual(exact_open_route([]).order, ())

    def test_free_end_route_is_exact_on_a_line(self):
        targets = [(2.0, 0.0), (1.0, 0.0), (3.0, 0.0)]
        route = exact_open_route(targets)
        self.assertAlmostEqual(route.distance_m, 3.0)
        self.assertEqual(set(route.order), {0, 1, 2})
        self.assertAlmostEqual(route_distance(targets, route.order), 3.0)

    def test_nonzero_start(self):
        targets = [(1.0, 1.0), (2.0, 1.0)]
        route = exact_open_route(targets, start=(1.0, 0.0))
        self.assertAlmostEqual(route.distance_m, 2.0)
        self.assertTrue(math.isfinite(route.distance_m))

    def test_directed_costs_can_change_order(self):
        route = exact_open_route_costs(
            [10.0, 1.0],
            [[0.0, 1.0], [100.0, 0.0]],
        )
        self.assertEqual(route.order, (0, 1))
        self.assertEqual(route.distance_m, 11.0)

    def test_fast_route_visits_every_target(self):
        targets = [(float(index), float(index % 3)) for index in range(20)]
        route = fast_open_route(targets)
        self.assertEqual(set(route.order), set(range(20)))
        self.assertAlmostEqual(
            route.distance_m, route_distance(targets, route.order)
        )


if __name__ == "__main__":
    unittest.main()
