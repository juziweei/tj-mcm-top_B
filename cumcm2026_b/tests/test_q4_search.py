import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm_b.q4_search import (  # noqa: E402
    SparseDirectionalSearch,
    directional_discovery_candidates,
    directional_discovery_lattice,
)
from cumcm_b.simulator import InterferenceEnvironment, generate_sources  # noqa: E402


class DirectionalDiscoveryTests(unittest.TestCase):
    def test_lattice_spacing_and_extent_support_the_geometric_proof(self):
        points = directional_discovery_lattice()
        self.assertIn((0.0, 0.0), points)
        self.assertLessEqual(700.0 * math.sqrt(2.0), 1000.0)
        self.assertGreaterEqual(2800.0, 1800.0 + 700.0 * math.sqrt(2.0))

    def test_refined_candidates_preserve_guaranteed_lattice(self):
        self.assertTrue(
            set(directional_discovery_lattice()).issubset(
                directional_discovery_candidates()
            )
        )

    def test_phase_refinement_enriches_candidate_pool(self):
        coarse = directional_discovery_candidates()
        phased = directional_discovery_candidates(refinement_phase_divisions=2)
        self.assertTrue(set(coarse).issubset(phased))
        self.assertGreater(len(phased), len(coarse))

    def test_clears_fixed_mixed_cases(self):
        for seed in (41, 42, 43, 44, 45):
            sources = generate_sources(seed, directional_probability=0.65)
            environment = InterferenceEnvironment(sources)
            result = SparseDirectionalSearch().run(environment)
            self.assertEqual(result.cleared_count, len(sources), (seed, result))
            self.assertAlmostEqual(
                result.travel_distance_m,
                result.discovery_travel_distance_m
                + result.pursuit_travel_distance_m,
                places=6,
            )

    def test_clears_source_just_beyond_last_signal_point(self):
        seed = 90_000_009
        sources = generate_sources(seed, directional_probability=0.5)
        environment = InterferenceEnvironment(sources)
        result = SparseDirectionalSearch().run(environment)
        self.assertEqual(result.cleared_count, len(sources), (seed, result))
        self.assertEqual(result.unresolved_channels, ())

    def test_completion_posterior_increases_with_spatially_diverse_scans(self):
        policy = SparseDirectionalSearch(belief_particle_count=512)
        first = policy._posterior_all_detected(12, [(0.0, 0.0)])
        diverse = policy._posterior_all_detected(
            12,
            [
                (0.0, 0.0),
                (1200.0, 0.0),
                (-1200.0, 0.0),
                (0.0, 1200.0),
                (0.0, -1200.0),
            ],
        )
        self.assertGreater(diverse, first)

    def test_heading_miss_fraction_integrates_all_orientations(self):
        miss = SparseDirectionalSearch._heading_miss_fraction
        self.assertAlmostEqual(miss([], 90.0), 1.0)
        self.assertAlmostEqual(miss([0.0], 90.0), 0.5)
        self.assertAlmostEqual(miss([0.0, 180.0], 90.0), 0.0)
        self.assertAlmostEqual(miss([350.0, 170.0], 90.0), 0.0)


if __name__ == "__main__":
    unittest.main()
