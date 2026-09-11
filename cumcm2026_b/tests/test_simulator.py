from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm_b.simulator import (  # noqa: E402
    InterferenceEnvironment,
    Source,
    generate_sources,
)


class SimulatorRuleTests(unittest.TestCase):
    def test_measurement_time_includes_move_and_switch(self):
        source = Source(2, (100.0, 0.0), 1000.0)
        environment = InterferenceEnvironment([source])
        observation = environment.measure((3.0, 4.0), 2)
        self.assertEqual(observation.result, "direction")
        self.assertAlmostEqual(observation.action_duration_s, 7.0)
        self.assertAlmostEqual(observation.virtual_time_s, 7.0)

    def test_same_position_has_fixed_bearing_error(self):
        source = generate_sources(7, count=10)[0]
        environment = InterferenceEnvironment([source])
        # Use the source location shifted toward the origin to remain visible for
        # both omni- and directional-generated cases by replacing the type.
        source = Source(
            channel=source.channel,
            position=(500.0, 100.0),
            receive_radius=source.receive_radius,
            directional=False,
            coverage_deg=360.0,
            error_coefficients=source.error_coefficients,
        )
        environment = InterferenceEnvironment([source])
        first = environment.measure((0.0, 0.0), source.channel)
        second = environment.measure((0.0, 0.0), source.channel)
        self.assertEqual(first.result, "direction")
        self.assertEqual(second.result, "direction")
        self.assertEqual(first.bearing_deg, second.bearing_deg)

    def test_directional_source_can_be_cleared_outside_beam(self):
        source = Source(
            channel=3,
            position=(0.0, 0.0),
            receive_radius=1000.0,
            directional=True,
            heading_deg=0.0,
            coverage_deg=180.0,
        )
        environment = InterferenceEnvironment([source])
        measurement = environment.measure((-10.0, 0.0), 3)
        self.assertEqual(measurement.result, "no_signal")
        clearing = environment.clear((-10.0, 0.0), 3)
        self.assertEqual(clearing.result, "success")
        self.assertTrue(environment.all_cleared)
        # Moving zero metres; success costs 3 seconds localization + 2 clearing.
        self.assertAlmostEqual(clearing.action_duration_s, 5.0)

    def test_directional_coverage_includes_ninety_degree_boundary(self):
        source = Source(
            channel=3,
            position=(0.0, 0.0),
            receive_radius=1000.0,
            directional=True,
            heading_deg=0.0,
            coverage_deg=180.0,
        )
        environment = InterferenceEnvironment([source])
        observation = environment.measure((0.0, 100.0), 3)
        self.assertEqual(observation.result, "direction")

    def test_failed_and_repeated_clear_cost_three_seconds(self):
        source = Source(4, (0.0, 0.0), 1000.0)
        environment = InterferenceEnvironment([source])
        successful = environment.clear((0.0, 0.0), 4)
        repeated = environment.clear((0.0, 0.0), 4)
        self.assertEqual(successful.result, "success")
        self.assertEqual(repeated.result, "no_target_in_range")
        self.assertAlmostEqual(successful.action_duration_s, 5.0)
        self.assertAlmostEqual(repeated.action_duration_s, 3.0)
        self.assertEqual(environment.current_channel, 1)

    def test_generation_respects_channel_and_geometry_constraints(self):
        sources = generate_sources(2026, count=16, directional_probability=0.4)
        self.assertEqual(len(sources), 16)
        self.assertEqual(len({source.channel for source in sources}), 16)
        for source in sources:
            self.assertLessEqual(
                source.position[0] ** 2 + source.position[1] ** 2,
                1800.0 ** 2,
            )
            self.assertGreaterEqual(source.receive_radius, 1000.0)
            self.assertLessEqual(source.receive_radius, 1500.0)
            self.assertEqual(source.coverage_deg, 180.0 if source.directional else 360.0)


if __name__ == "__main__":
    unittest.main()
