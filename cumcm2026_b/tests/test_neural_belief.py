from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cumcm_b.neural_belief import TOKEN_FEATURE_NAMES, observation_token, time_bin_index
from generate_belief_dataset import generate_one


class NeuralBeliefSchemaTests(unittest.TestCase):
    def test_token_contains_only_finite_observable_values(self):
        token = observation_token(
            previous_position=(0.0, 0.0), target=(100.0, -50.0), channel=7,
            is_measure=True, switched=True, result="direction", bearing_deg=123.0,
            action_duration_s=28.0, virtual_time_s=1200.0, cleared_count=4, q_variant=4,
        )
        self.assertEqual(token.shape, (len(TOKEN_FEATURE_NAMES),))
        self.assertTrue(np.isfinite(token).all())

    def test_time_bins_include_overflow(self):
        np.testing.assert_array_equal(time_bin_index([1.0, 30.0, 8000.0]), [0, 1, 9])

    def test_generated_episode_labels_match_source_count(self):
        episode = generate_one((141_000_001, 4))
        rows = len(episode["tokens"])
        self.assertEqual(episode["tokens"].shape, (rows, len(TOKEN_FEATURE_NAMES)))
        self.assertEqual(len(episode["remaining_count"]), rows)
        np.testing.assert_array_equal(
            episode["remaining_count"],
            episode["source_count"] - episode["cleared_count_after"],
        )
        self.assertTrue((episode["known_unresolved_after"] >= 0).all())
        self.assertTrue(np.isfinite(episode["tokens"]).all())


if __name__ == "__main__":
    unittest.main()
