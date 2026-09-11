from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm_b.fast_q3 import InterleavedOmniSearch  # noqa: E402
from cumcm_b.simulator import InterferenceEnvironment, generate_sources  # noqa: E402


class InterleavedOmniSearchTests(unittest.TestCase):
    def test_clears_fixed_omni_cases_without_true_count(self):
        for seed in (31, 32, 33):
            sources = generate_sources(seed, directional_probability=0.0)
            environment = InterferenceEnvironment(sources)
            result = InterleavedOmniSearch().run(environment)
            self.assertEqual(result.cleared_count, len(sources), seed)


if __name__ == "__main__":
    unittest.main()
