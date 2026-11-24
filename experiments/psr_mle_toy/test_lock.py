"""Small checks for the exact lock TV and specialized bonus certificate."""
import unittest
import random

from lock_ours import BoxEnergySupport, run_ours
from pocomblock import candidate_models, worst_policy_tv


class LockExperimentTests(unittest.TestCase):
    def test_exact_tv_and_support(self):
        models = candidate_models(4)
        self.assertEqual(worst_policy_tv(models[0], models[0])[0], 0)
        self.assertAlmostEqual(worst_policy_tv(models[0], models[1])[0],
                               worst_policy_tv(models[1], models[0])[0])
        self.assertAlmostEqual(BoxEnergySupport([], 1)((.3, .7)), .8)
        support = BoxEnergySupport([((1., 0.), 100)], 1)
        self.assertAlmostEqual(support((1., 0.)), .1)
        self.assertAlmostEqual(support((0., 1.)), .8)

    def test_ex_post_certificate_audit(self):
        models = candidate_models(4)
        for seed in (1, 2, 3):
            path = run_ours(models, models[seed], seed, 4, 8192, .2, 512)
            self.assertTrue(all(row["certificate_valid"] for row in path))

    def test_support_dominates_feasible_grid(self):
        rng = random.Random(7)
        grid = [i * .04 - .8 for i in range(41)]
        for _ in range(30):
            rows = [((rng.random(), rng.random()), rng.randrange(1, 100))
                    for _ in range(rng.randrange(5))]
            support = BoxEnergySupport(rows, rng.uniform(.01, 100))
            query = (rng.random(), rng.random())
            optimum = support(query)
            for x in grid:
                for y in grid:
                    energy = sum(n * (x * q[0] + y * q[1]) ** 2
                                 for q, n in rows)
                    if energy <= support.energy + 1e-9:
                        self.assertGreaterEqual(optimum + 1e-9,
                                                x * query[0] + y * query[1])

    def test_graded_family_certificate_audit(self):
        models = candidate_models(4, count=16, family="graded")
        path = run_ours(models, models[3], 3, 4, 8192, .2, 512)
        self.assertTrue(all(row["certificate_valid"] for row in path))


if __name__ == "__main__":
    unittest.main()
