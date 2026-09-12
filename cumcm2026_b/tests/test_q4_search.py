import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm_b.q4_search import (  # noqa: E402
    Q4DecisionContext,
    SparseDirectionalSearch,
    directional_discovery_candidates,
    directional_discovery_lattice,
    q4_profile_options,
)
from cumcm_b.simulator import InterferenceEnvironment, generate_sources  # noqa: E402


class DirectionalDiscoveryTests(unittest.TestCase):
    def test_profiles_keep_the_reliability_choice_explicit(self):
        self.assertEqual(q4_profile_options("conservative")["stop_probability"], 0.998)
        self.assertEqual(q4_profile_options("balanced")["stop_probability"], 0.97)
        self.assertEqual(q4_profile_options("aggressive")["stop_probability"], 0.95)
        self.assertEqual(
            q4_profile_options("source-reliable")["source_miss_risk_budget"],
            0.03,
        )
        self.assertEqual(
            q4_profile_options("source-reliable")[
                "any_source_remaining_probability_budget"
            ],
            0.05,
        )
        self.assertEqual(
            q4_profile_options("source-efficient")["source_miss_risk_budget"],
            0.04,
        )
        self.assertEqual(
            q4_profile_options("source-efficient")[
                "any_source_remaining_probability_budget"
            ],
            0.10,
        )
        self.assertEqual(
            q4_profile_options("source-98")["source_miss_risk_budget"],
            0.04,
        )
        self.assertEqual(
            q4_profile_options("source-98")[
                "any_source_remaining_probability_budget"
            ],
            0.25,
        )
        for name in (
            "conservative",
            "balanced",
            "aggressive",
            "source-reliable",
            "source-efficient",
            "source-98",
        ):
            options = q4_profile_options(name)
            self.assertEqual(options["centroid_clear_max_radius_m"], 75.0)
            self.assertTrue(options["use_enclosing_circle_target"])

    def test_observational_decision_hook_preserves_default_policy(self):
        sources = generate_sources(91_234_567, count=10, directional_probability=0.5)
        baseline = SparseDirectionalSearch().run(InterferenceEnvironment(sources))
        decisions: list[Q4DecisionContext] = []

        def observe(context: Q4DecisionContext):
            decisions.append(context)
            return None

        observed = SparseDirectionalSearch(decision_override=observe).run(
            InterferenceEnvironment(sources)
        )
        self.assertTrue(decisions)
        self.assertEqual(
            (baseline.cleared_count, baseline.virtual_time_s),
            (observed.cleared_count, observed.virtual_time_s),
        )
        self.assertEqual(
            [context.decision_index for context in decisions],
            list(range(len(decisions))),
        )

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

    def test_source_level_posterior_risk_falls_with_miss_probability(self):
        high_miss = SparseDirectionalSearch._source_count_posterior_summary(12, 0.5)
        low_miss = SparseDirectionalSearch._source_count_posterior_summary(12, 0.05)
        self.assertGreater(low_miss[0], high_miss[0])
        self.assertLess(low_miss[1], high_miss[1])
        self.assertLess(low_miss[2], high_miss[2])
        self.assertEqual(
            SparseDirectionalSearch._source_count_posterior_summary(16, 1.0),
            (1.0, 0.0, 0.0),
        )
        self.assertEqual(
            SparseDirectionalSearch._source_count_posterior_summary(9, 0.0),
            (0.0, 1.0, 0.1),
        )

    def test_source_risk_budget_replaces_whole_case_stopping_objective(self):
        source_risk_policy = SparseDirectionalSearch(
            belief_particle_count=512,
            source_miss_risk_budget=0.02,
        )
        self.assertTrue(source_risk_policy._stopping_objective_satisfied(10, 0.01))
        self.assertFalse(source_risk_policy._stopping_objective_satisfied(10, 0.1))

        whole_case_policy = SparseDirectionalSearch(
            belief_particle_count=512,
            stop_probability=0.998,
        )
        self.assertFalse(whole_case_policy._stopping_objective_satisfied(10, 0.01))

    def test_any_remaining_chance_constraint_protects_the_last_source(self):
        source_only = SparseDirectionalSearch(
            belief_particle_count=512,
            source_miss_risk_budget=0.04,
        )
        chance_constrained = SparseDirectionalSearch(
            belief_particle_count=512,
            source_miss_risk_budget=0.04,
            any_source_remaining_probability_budget=0.2,
        )
        self.assertTrue(source_only._stopping_objective_satisfied(15, 0.05))
        self.assertFalse(
            chance_constrained._stopping_objective_satisfied(15, 0.05)
        )
        self.assertTrue(
            chance_constrained._stopping_objective_satisfied(15, 0.01)
        )

    def test_heading_miss_fraction_integrates_all_orientations(self):
        miss = SparseDirectionalSearch._heading_miss_fraction
        self.assertAlmostEqual(miss([], 90.0), 1.0)
        self.assertAlmostEqual(miss([0.0], 90.0), 0.5)
        self.assertAlmostEqual(miss([0.0, 180.0], 90.0), 0.0)
        self.assertAlmostEqual(miss([350.0, 170.0], 90.0), 0.0)


if __name__ == "__main__":
    unittest.main()
