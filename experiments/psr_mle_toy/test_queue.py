"""Independent checks for the 64-state PSR experiment."""
from collections import Counter
from itertools import product
import unittest

import numpy as np

from queue_pomdp import CORES, QueueModel, STATE_COUNT, STABILITY, TRUE_WEIGHTS
from queue_psr import feature
from run_queue_ours import design_law, fit_mle, worst_policy_tv


class QueueChecks(unittest.TestCase):
    def test_core_and_reachable_rank(self):
        self.assertEqual(STATE_COUNT, 64)
        self.assertEqual([np.linalg.matrix_rank(CORES[length])
                          for length in (1, 2, 3)], [4, 16, 64])
        self.assertAlmostEqual(STABILITY, 1.75**3)
        model = QueueModel(TRUE_WEIGHTS)
        reachable = [feature(model, (o1, a1, o2, a2))[0]
                     for o1, a1, o2, a2 in product(range(4), range(2),
                                                     range(4), range(2))]
        self.assertEqual(np.linalg.matrix_rank(np.asarray(reachable)), 16)
        self.assertAlmostEqual(sum(design_law(model, {}).values()), 1.)

    def test_stage_operator_factors_through_core(self):
        model = QueueModel(TRUE_WEIGHTS)
        stage = 2
        current = CORES[model.core_length(stage)]
        following = CORES[model.core_length(stage + 1)]
        for observation, action in product(range(4), range(2)):
            direct = np.column_stack([
                following @ model.step(stage, np.eye(STATE_COUNT)[state],
                                       observation, action)
                for state in range(STATE_COUNT)])
            residual = direct @ np.linalg.pinv(current) @ current - direct
            self.assertLess(np.max(np.abs(residual)), 1e-12)

    def test_tv_witness_and_exact_law_mle(self):
        truth = QueueModel(TRUE_WEIGHTS)
        wrong = QueueModel((0.,) * 12)
        distance, policy = worst_policy_tv(truth, wrong)
        direct = 0.
        for observations in product(range(4), repeat=4):
            history = ()
            for observation in observations:
                observed = history + (observation,)
                history = observed + (policy[observed],)
            direct += abs(truth.trajectory_probability(history)
                          - wrong.trajectory_probability(history)) / 2
        self.assertAlmostEqual(distance, direct)

        counts = Counter()
        for trace in product(range(4), range(2), repeat=4):
            counts[trace] = round(100_000_000
                                  * truth.trajectory_probability(trace) / 16)
        fitted, _, _ = fit_mle(counts)
        for actual, estimated in zip(truth.weights, fitted.weights):
            self.assertAlmostEqual(actual, estimated, delta=.003)


if __name__ == "__main__":
    unittest.main()
