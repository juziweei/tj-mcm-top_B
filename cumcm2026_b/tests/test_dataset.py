import math
import unittest

import numpy as np

from cumcm_b.dataset import FEATURE_NAMES, directional_scan_waypoints, generate_episode


class DatasetGenerationTests(unittest.TestCase):
    def test_episode_has_grouped_expert_labels_and_finite_features(self):
        episode = generate_episode(
            seed=10_000_000,
            episode_id=10_000_000,
            decision_id_start=10_000_000_000,
            max_steps=8,
        )
        self.assertEqual(episode.features.shape[1], len(FEATURE_NAMES))
        self.assertTrue(np.isfinite(episode.features).all())
        decisions, first = np.unique(episode.decision_id, return_index=True)
        chosen_per_decision = np.add.reduceat(episode.chosen.astype(np.int64), first)
        self.assertGreater(len(decisions), 0)
        self.assertTrue(np.all(chosen_per_decision == 1))
        self.assertEqual(len(episode.observations), len(decisions))

    def test_hidden_source_coordinates_are_not_feature_names(self):
        forbidden = {"source_x", "source_y", "receive_radius", "heading_deg"}
        self.assertTrue(forbidden.isdisjoint(FEATURE_NAMES))

    def test_q3_short_range_single_bearing_regression_completes(self):
        episode = generate_episode(
            seed=63_000_148,
            episode_id=63_000_148,
            decision_id_start=63_000_148_000,
            max_steps=12_000,
            q_variant_override=3,
            record_arrays=False,
        )
        self.assertEqual(episode.summary["cleared_count"], episode.summary["source_count"])
        self.assertLess(episode.summary["decisions"], 12_000)

    def test_q4_teacher_uses_sparse_proven_discovery_lattice(self):
        points = directional_scan_waypoints()
        self.assertEqual(points[0], (0.0, 0.0))
        self.assertLessEqual(len(points), 50)
        self.assertLessEqual(700.0 * math.sqrt(2.0), 1000.0)
        self.assertGreaterEqual(2800.0, 1800.0 + 700.0 * math.sqrt(2.0))

    def test_q4_near_observation_regression_clears_immediately(self):
        episode = generate_episode(
            seed=64_000_031,
            episode_id=64_000_031,
            decision_id_start=64_000_031_000,
            max_steps=3_000,
            q_variant_override=4,
            record_arrays=False,
        )
        self.assertEqual(episode.summary["cleared_count"], episode.summary["source_count"])
        self.assertLess(episode.summary["decisions"], 3_000)


if __name__ == "__main__":
    unittest.main()
