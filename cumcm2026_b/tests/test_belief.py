import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm_b.belief import ChannelBelief, SourceParticle  # noqa: E402


def particle(x, y, *, exists=True, directional=False, heading=0.0, weight=1.0):
    return SourceParticle(
        exists=exists,
        position=(x, y),
        receive_radius=1200.0,
        directional=directional,
        heading_deg=heading,
        weight=weight,
    )


class ChannelBeliefTests(unittest.TestCase):
    def test_direction_selects_matching_bearing_particles(self):
        belief = ChannelBelief(
            [
                particle(500.0, 0.0, weight=0.5),
                particle(0.0, 500.0, weight=0.5),
            ],
            resample_ratio=0.01,
        )
        belief.update_measurement((0.0, 0.0), "direction", 0.0)
        summary = belief.summary()
        self.assertGreater(summary.mean_position[0], 499.0)
        self.assertLess(abs(summary.mean_position[1]), 1.0)

    def test_no_signal_reduces_visible_existence_probability(self):
        belief = ChannelBelief(
            [
                particle(500.0, 0.0, exists=True, weight=0.5),
                particle(0.0, 0.0, exists=False, weight=0.5),
            ],
            resample_ratio=0.01,
        )
        belief.update_measurement((0.0, 0.0), "no_signal")
        self.assertLess(belief.summary().existence_probability, 1e-4)

    def test_directional_visibility_is_reflected_in_no_signal(self):
        belief = ChannelBelief(
            [
                particle(0.0, 0.0, directional=True, heading=0.0, weight=0.5),
                particle(0.0, 0.0, directional=True, heading=180.0, weight=0.5),
            ],
            resample_ratio=0.01,
        )
        belief.update_measurement((500.0, 0.0), "no_signal")
        # Heading 0 predicts visibility and is rejected; heading 180 survives.
        directional_particles = belief.particles
        surviving = max(directional_particles, key=lambda item: item.weight)
        self.assertAlmostEqual(surviving.heading_deg, 180.0)

    def test_successful_clear_marks_channel(self):
        belief = ChannelBelief(
            [particle(10.0, 0.0, weight=0.8), particle(100.0, 0.0, weight=0.2)],
            resample_ratio=0.01,
        )
        belief.update_clear((0.0, 0.0), "success")
        summary = belief.summary()
        self.assertTrue(summary.cleared)
        self.assertEqual(belief.clear_probability((10.0, 0.0)), 0.0)

    def test_prior_summary_is_finite(self):
        belief = ChannelBelief.prior(1000, seed=42)
        summary = belief.summary()
        self.assertTrue(0.5 < summary.existence_probability < 0.8)
        self.assertTrue(math.isfinite(summary.spatial_spread))


if __name__ == "__main__":
    unittest.main()

