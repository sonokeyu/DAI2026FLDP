import unittest

import numpy as np

from compare import compare, disagreement_policy
from rank_sweep import reachable_ranks, run_rank_case, transition
from routing import (
    EPISODES, PREFIXES, Design, RoutingModel, Support, all_policy_tv, diagnostic,
    distance, hellinger, law, mix, nested, plan_bonus, run_algorithm,
)
from toy import all_deterministic_policies


class RoutingTests(unittest.TestCase):
    def test_support_known_optima(self):
        self.assertAlmostEqual(Support([((1., 0.), 32)], 1)((0., 1.))[0], 2)
        # Rank-one strip through the box; the maximizing point lies on an edge.
        value, d = Support([((0.5, 0.5), 16)], 1)((1., 0.))
        self.assertAlmostEqual(value, 2)
        self.assertLessEqual(16 * ((d[0] + d[1]) / 2) ** 2, 1 + 1e-9)
        # Diagonal ellipsoid, whose optimum is inside the box.
        self.assertAlmostEqual(Support([((1., 0.), 9), ((0., 1.), 16)], 1)((.6, .4))[0],
                               np.hypot(.6 / 3, .4 / 4))

    def test_uniform_tv_matches_exhaustive_planning(self):
        first, second = RoutingModel(.2), RoutingModel(.8)
        exhaustive = max(distance(law(first, mix(Design.policy(p))),
                                  law(second, mix(Design.policy(p))))
                         for p in all_deterministic_policies())
        self.assertAlmostEqual(all_policy_tv(first, second), exhaustive)
        chosen = disagreement_policy([first, second])
        self.assertAlmostEqual(distance(law(first, mix(chosen)), law(second, mix(chosen))), exhaustive)

    def test_paper_nested_branch(self):
        # Section 4 chooses R on the negative cue and L on the positive cue.
        roll_in = Design((1.0, 0.0))
        coeff = {x: (2., (0., 2.)) for x in PREFIXES}
        rho = diagnostic(roll_in, coeff)
        first, second = RoutingModel(0), RoutingModel(1)
        p, q = law(first, nested(rho)), law(second, nested(rho))
        self.assertAlmostEqual(p.sum(), 1)
        self.assertAlmostEqual(q.sum(), 1)
        # Index 2 is cut 2 from the unmodified full-policy component.
        relevant = [i for i, e in enumerate(EPISODES)
                    if (e.cue, e.route, e.mode) == ("-", "R", "a")]
        self.assertAlmostEqual(hellinger(p[2, relevant], q[2, relevant]), 1 / 36)

    def test_cuts_detect_error_absent_from_unmodified_diagnostic(self):
        first, second = RoutingModel(.2), RoutingModel(.3)
        _, policy, queries = plan_bonus(first, [0, 0, {}, {}], 1.)
        rho = diagnostic(policy, queries)
        self.assertAlmostEqual(distance(law(first, rho), law(second, rho)), 0)
        self.assertGreater(hellinger(law(first, nested(rho)), law(second, nested(rho))), 0)

    def test_suite_contracts(self):
        models = [RoutingModel(0), RoutingModel(1)]
        route = run_algorithm(models, 1, .5, 1, theory=True)
        self.assertTrue(route[-1]["stopped"])
        self.assertTrue(all(row["calibrated"] for row in route))
        self.assertLessEqual(route[-1]["policy_tv"], .5 / 4)
        self.assertEqual(route[-1]["task_plans"], 4)
        omle = compare(models, 1, "reward_free_omle", 0)
        self.assertEqual(omle[-1]["episodes"], 256)
        self.assertEqual(omle[-1]["updates"], 128)
        self.assertEqual(omle[-1]["policy_tv"], 0)

    def test_fixed_dimension_rank_and_stop(self):
        self.assertEqual(reachable_ranks(transition(4), 6), [1, 2, 3, 4, 4, 4])
        low, high = (run_rank_case(4, rank, 1., 0) for rank in (1, 4))
        self.assertEqual(low["dimension"], high["dimension"])
        self.assertLess(low["mle_calls"], high["mle_calls"])
        self.assertTrue(low["stopped"] and high["stopped"])
        self.assertEqual(high["policy_tv"], 0)


if __name__ == "__main__":
    unittest.main()
