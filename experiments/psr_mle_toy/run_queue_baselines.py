"""Uniform/count MLE controls and finite-grid OMLE for the queue POMDP.

OMLE uses a sampled finite grid over the 12 transition parameters. The grid
is a practical comparator, not an exact continuous-class OMLE oracle.
"""
from argparse import ArgumentParser
from itertools import combinations
from math import log
from pathlib import Path
import random

import numpy as np

from queue_pomdp import QueueModel, TRUE_WEIGHTS
from run_multistage_experiment import write_csv
from run_queue_ours import fit_mle, worst_policy_tv


class UniformPolicy:
    def __init__(self, rng): self.rng = rng
    def __getitem__(self, history): return self.rng.randrange(2)


class CountPolicy(UniformPolicy):
    def __init__(self, rng):
        super().__init__(rng); self.counts = {}
    def __getitem__(self, history):
        left, right = self.counts.get(history, (0, 0))
        return self.rng.randrange(2) if left == right else int(right < left)
    def record(self, trace):
        for stage in range(4):
            history, action = trace[:2*stage+1], trace[2*stage+1]
            counts = self.counts.setdefault(history, [0, 0]); counts[action] += 1


def run_mle_control(truth, seed, method, budget, checkpoint):
    rng = random.Random(seed + (1001 if method == "uniform" else 2003))
    policy = UniformPolicy(rng) if method == "uniform" else CountPolicy(rng)
    data = {}
    rows = []
    previous = None
    for episode in range(1, budget + 1):
        trace = truth.sample(policy, rng)
        data[trace] = data.get(trace, 0) + 1
        if method == "count": policy.record(trace)
        if episode % checkpoint == 0:
            previous, evaluations, loss = fit_mle(data, previous)
            tv = worst_policy_tv(truth, previous)[0]
            rows.append(dict(method=method, seed=seed, episodes=episode,
                             mle_calls=len(rows)+1, planning_calls=0,
                             joint_calls=0, objective_evals=evaluations,
                             negative_log_likelihood=loss, tv=tv, stopped=False))
    return rows


def grid_models(seed, size):
    rng = np.random.default_rng(seed)
    grid = np.array([0., .25, .5, .75, 1.])
    unique = set()
    while len(unique) < size:
        unique.add(tuple(float(x) for x in rng.choice(grid, 12)))
    return tuple(QueueModel(weights) for weights in unique)


def run_grid_omle(truth, models, seed, budget, epsilon, checkpoint=64):
    rng = random.Random(seed + 3001)
    uniform = UniformPolicy(rng)
    scores = np.zeros(len(models))
    pairs = {}
    rows = []
    joint_calls = 0
    slack = log(budget * len(models) / .1)
    for episode in range(1, budget + 1):
        best = float(scores.max())
        active = [i for i, value in enumerate(scores) if value >= best - slack]
        if episode == 1 or episode % checkpoint == 1:
            for i, j in combinations(active, 2):
                if (i, j) not in pairs:
                    pairs[i, j] = worst_policy_tv(models[i], models[j])
        disagreement, left, right = max(
            ((pairs[i, j][0], i, j) for i, j in combinations(active, 2)),
            default=(0., -1, -1))
        if disagreement <= epsilon / 4:
            fit_index = int(scores.argmax())
            rows.append(dict(method="omle_grid", seed=seed, episodes=episode-1,
                             mle_calls=0, planning_calls=0, joint_calls=joint_calls,
                             objective_evals=0, negative_log_likelihood=-best,
                             tv=worst_policy_tv(truth, models[fit_index])[0],
                             stopped=True, grid_size=len(models)))
            return rows
        if episode % 2:
            policy = uniform
        else:
            policy = pairs[left, right][1]
            joint_calls += 1
        trace = truth.sample(policy, rng)
        for index, model in enumerate(models):
            probability = model.trajectory_probability(trace)
            scores[index] += log(probability) if probability else -1e300
        if episode % checkpoint == 0:
            fit_index = int(scores.argmax())
            rows.append(dict(method="omle_grid", seed=seed, episodes=episode,
                             mle_calls=0, planning_calls=0, joint_calls=joint_calls,
                             objective_evals=0, negative_log_likelihood=-float(scores.max()),
                             tv=worst_policy_tv(truth, models[fit_index])[0],
                             stopped=False, grid_size=len(models)))
    return rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--checkpoint", type=int, default=64)
    parser.add_argument("--omle-grid", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=.4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    truth = QueueModel(TRUE_WEIGHTS)
    rows = []
    for seed in range(args.seeds):
        for method in ("uniform", "count"):
            rows.extend(run_mle_control(truth, seed, method,
                                        args.budget, args.checkpoint))
        models = grid_models(seed, args.omle_grid)
        rows.extend(run_grid_omle(truth, models, seed, args.budget,
                                  args.epsilon, args.checkpoint))
        print(f"completed baseline seed={seed}", flush=True)
    write_csv(args.output / "queue_baselines_runs.csv", rows)
    summary = []
    for method in ("uniform", "count", "omle_grid"):
        group = [row for row in rows if row["method"] == method]
        final = [max((row for row in group if row["seed"] == seed),
                     key=lambda row: row["episodes"]) for seed in range(args.seeds)]
        summary.append(dict(method=method, runs=len(final),
                            success=sum(row["tv"] <= args.epsilon / 4 for row in final),
                            median_tv=float(np.median([row["tv"] for row in final])),
                            median_episodes=float(np.median([row["episodes"] for row in final])),
                            median_mle_calls=float(np.median([row["mle_calls"] for row in final])),
                            median_joint_calls=float(np.median([row["joint_calls"] for row in final])),
                            grid_size=args.omle_grid if method == "omle_grid" else ""))
    write_csv(args.output / "queue_baselines_summary.csv", summary)


if __name__ == "__main__": main()
