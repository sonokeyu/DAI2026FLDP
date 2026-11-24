import unittest

from toy import (
    Episode, Model, Policy, diagnostic_branch_mass, log_likelihood,
    max_policy_tv, ordinary_mle, optimal_reward, run_stage_two_learning,
    stage_two_support, total_variation, trajectory_distribution,
)


class ThreeStageToyTests(unittest.TestCase):
    def test_exact_paper_example(self):
        hidden = Policy(("R", "R"), ("b",) * 4)
        revealing = Policy(("R", "R"), ("a",) * 4)
        self.assertEqual(total_variation(
            trajectory_distribution(Model.TRUE, hidden),
            trajectory_distribution(Model.WRONG, hidden)), 0)
        self.assertEqual(total_variation(
            trajectory_distribution(Model.TRUE, revealing),
            trajectory_distribution(Model.WRONG, revealing)), 0.5)
        self.assertEqual(max_policy_tv(Model.TRUE, Model.WRONG)[0], 0.5)
        self.assertEqual(optimal_reward(Model.TRUE)[0], 0.75)
        self.assertEqual(diagnostic_branch_mass(), 1 / 36)

    def test_stage_two_support_responds_to_reencoded_coverage(self):
        histories = [("-", "L")] * 32
        self.assertEqual(stage_two_support(Model.WRONG, "-", "R", histories, 1), 2)
        self.assertAlmostEqual(stage_two_support(Model.WRONG, "-", "L", histories, 1),
                               1 / (32 ** 0.5))
        histories += [("-", "R")] * 25
        self.assertAlmostEqual(stage_two_support(Model.WRONG, "-", "R", histories, 1), 0.2)
        # The same physical history is type zero under the true model.
        self.assertAlmostEqual(stage_two_support(Model.TRUE, "-", "R", histories, 1),
                               1 / (57 ** 0.5))

    def test_complete_episode_mle_and_tie(self):
        hidden = [Episode("-", "R", "b", 0)]
        self.assertEqual(ordinary_mle(hidden), Model.WRONG)
        revealing = [Episode("-", "R", "a", 0)]
        self.assertEqual(ordinary_mle(revealing), Model.TRUE)
        self.assertEqual(log_likelihood(Model.WRONG, revealing), float("-inf"))

    def test_passive_can_stop_with_model_error(self):
        passive, stopped = run_stage_two_learning(
            seed=0, batch_size=16, max_rounds=20, test_probability=0)
        self.assertTrue(stopped)
        self.assertEqual(passive[-1].fit, "wrong")
        self.assertEqual(passive[-1].max_policy_tv, 0.5)
        self.assertEqual(passive[-1].task_regret, 0.25)
        active, _ = run_stage_two_learning(
            seed=0, batch_size=16, max_rounds=20, test_probability=0.5)
        self.assertEqual(active[-1].fit, "true")
        self.assertEqual(active[-1].max_policy_tv, 0)
        self.assertEqual(active[-1].task_regret, 0)


if __name__ == "__main__":
    unittest.main()
