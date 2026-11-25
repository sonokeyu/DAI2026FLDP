"""Independent checks for the multi-stage evaluator and PSR ranks."""
from itertools import product
import unittest

import numpy as np

from multistage_pomdp import Model, candidates, worst_policy_tv
from multistage_psr import STABILITY, prefix_state
from run_multistage_ours import design_law


class MultiStageChecks(unittest.TestCase):
    def test_worst_policy_tv_against_direct_enumeration(self):
        first, second = Model(4, (0,)), Model(4, (1,))
        brute = 0.
        for actions in product(range(2), repeat=4):
            total = 0.
            for first_observation, second_observation in product(range(4), repeat=2):
                trace = (first_observation, actions[first_observation],
                         second_observation, 0)
                total += abs(first.trajectory_probability(trace)
                             - second.trajectory_probability(trace)) / 2
            brute = max(brute, total)
        self.assertAlmostEqual(worst_policy_tv(first, second)[0], brute)

    def test_rank_and_design_law(self):
        self.assertAlmostEqual(STABILITY, 1.75)
        models = candidates(3)
        block = len(models) // 3
        for index, expected_rank in enumerate((1, 2, 4)):
            model = models[(index + 1) * block - 1]
            rows = [prefix_state(model, (observation, action))
                    for observation, action in product(range(4), range(2))]
            self.assertEqual(np.linalg.matrix_rank(np.array(rows)), expected_rank)
            self.assertAlmostEqual(sum(design_law(model, {}).values()), 1.)


if __name__ == "__main__":
    unittest.main()
