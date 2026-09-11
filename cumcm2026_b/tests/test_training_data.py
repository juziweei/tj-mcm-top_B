from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm_b.training_data import iter_decision_batches  # noqa: E402


class GroupedTrainingDataTests(unittest.TestCase):
    def test_batches_keep_complete_decision_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory) / "train"
            split.mkdir()
            decision_id = np.asarray([10, 10, 11, 11, 11, 12], dtype=np.int64)
            chosen = np.asarray([0, 1, 1, 0, 0, 1], dtype=np.uint8)
            rows = len(decision_id)
            np.savez(
                split / "part-00000.npz",
                features=np.zeros((rows, 47), dtype=np.float32),
                chosen=chosen,
                target_clear_gain=np.zeros(rows, dtype=np.float32),
                target_information_gain=np.zeros(rows, dtype=np.float32),
                target_duration_s=np.ones(rows, dtype=np.float32),
                q_variant=np.full(rows, 3, dtype=np.uint8),
                decision_id=decision_id,
            )
            batches = list(
                iter_decision_batches(
                    directory,
                    decisions_per_batch=2,
                    shuffle=False,
                )
            )
            self.assertEqual([batch.group_count for batch in batches], [2, 1])
            self.assertEqual([len(batch.features) for batch in batches], [5, 1])
            for batch in batches:
                selected = np.add.reduceat(
                    batch.chosen.astype(np.int64), batch.group_offsets[:-1]
                )
                self.assertTrue(np.all(selected == 1))


if __name__ == "__main__":
    unittest.main()
