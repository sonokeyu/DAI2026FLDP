"""Exact finite-class OMLE and two MLE controls for multistage_pomdp."""
from itertools import combinations
from math import log
import random

from multistage_pomdp import worst_policy_tv


class UniformPolicy:
    def __init__(self, rng):
        self.rng = rng

    def __getitem__(self, history):
        return self.rng.randrange(2)


class CountPolicy(UniformPolicy):
    def __init__(self, rng):
        super().__init__(rng)
        self.counts = {}

    def __getitem__(self, history):
        visits = self.counts.get(history, (0, 0))
        return self.rng.randrange(2) if visits[0] == visits[1] else int(visits[1] < visits[0])

    def record(self, trace):
        for stage in range(len(trace) // 2):
            history = trace[:2 * stage + 1]
            action = trace[2 * stage + 1]
            visits = self.counts.setdefault(history, [0, 0])
            visits[action] += 1


def add_likelihood(models, scores, trace):
    for index, model in enumerate(models):
        probability = model.trajectory_probability(trace)
        scores[index] += log(probability) if probability else float("-inf")


def fitted_index(scores):
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def run_control(models, truth_index, seed, budget, method, checkpoint=64):
    truth = models[truth_index]
    rng = random.Random(seed + (131 if method == "uniform" else 257))
    policy = UniformPolicy(rng) if method == "uniform" else CountPolicy(rng)
    scores = [0.] * len(models)
    rows = []
    for episode in range(1, budget + 1):
        trace = truth.sample(policy, rng)
        add_likelihood(models, scores, trace)
        if method == "count":
            policy.record(trace)
        if episode % checkpoint == 0:
            fit = fitted_index(scores)
            rows.append(dict(method=method, episodes=episode,
                             mle_calls=episode // checkpoint,
                             planning_calls=0, joint_calls=0,
                             fit=fit, tv=worst_policy_tv(truth, models[fit])[0],
                             stopped=False))
    return rows


def run_omle(models, truth_index, seed, budget, epsilon, pair_data):
    truth = models[truth_index]
    rng = random.Random(seed + 389)
    uniform = UniformPolicy(rng)
    scores = [0.] * len(models)
    slack = log(budget * len(models) / .1)
    rows = []
    calls = 0
    for episode in range(0, budget, 2):
        best_score = max(scores)
        active = [i for i, score in enumerate(scores) if score >= best_score - slack]
        disagreement, left, right = max(
            ((pair_data[i, j][0], i, j) for i, j in combinations(active, 2)),
            default=(0., -1, -1))
        fit = fitted_index(scores)
        if disagreement <= epsilon / 4:
            rows.append(dict(method="omle", episodes=episode,
                             mle_calls=0, planning_calls=0, joint_calls=calls,
                             fit=fit, tv=worst_policy_tv(truth, models[fit])[0],
                             stopped=True))
            return rows
        for policy in (uniform, pair_data[left, right][1]):
            add_likelihood(models, scores, truth.sample(policy, rng))
        calls += 1
        if (episode + 2) % 64 == 0:
            fit = fitted_index(scores)
            rows.append(dict(method="omle", episodes=episode + 2,
                             mle_calls=0, planning_calls=0, joint_calls=calls,
                             fit=fit, tv=worst_policy_tv(truth, models[fit])[0],
                             stopped=False))
    return rows
