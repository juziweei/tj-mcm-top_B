import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm_b.baseline import (  # noqa: E402
    OmniGeometryBaseline,
    guaranteed_discovery_waypoints,
)
from cumcm_b.simulator import InterferenceEnvironment, generate_sources  # noqa: E402


class DiscoveryCoverageTests(unittest.TestCase):
    def test_waypoints_cover_target_disk_at_minimum_radius(self):
        waypoints = guaranteed_discovery_waypoints()
        worst_distance = 0.0
        for radial_index in range(37):
            radius = 1800.0 * radial_index / 36
            for angle_index in range(360):
                angle = math.radians(angle_index)
                point = (radius * math.cos(angle), radius * math.sin(angle))
                nearest = min(
                    math.hypot(point[0] - waypoint[0], point[1] - waypoint[1])
                    for waypoint in waypoints
                )
                worst_distance = max(worst_distance, nearest)
        self.assertLessEqual(worst_distance, 1000.0)


class OmniBaselineTests(unittest.TestCase):
    def test_baseline_clears_fixed_omni_instance(self):
        sources = generate_sources(
            17,
            count=10,
            directional_probability=0.0,
            error_modes=3,
        )
        environment = InterferenceEnvironment(sources)
        result = OmniGeometryBaseline(maximum_localization_steps=8).run(environment)
        self.assertEqual(result.source_count, 10)
        self.assertEqual(result.cleared_count, 10)
        self.assertEqual(result.cleared_fraction, 1.0)
        self.assertGreater(result.virtual_time_s, 0.0)


if __name__ == "__main__":
    unittest.main()

