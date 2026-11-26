"""OMLE joint-search calls to a confidence-diameter stopping threshold.

The original reward-free OMLE specifies a round budget.  This experiment
also checks its already-computed confidence diameter each round and stops
when it is at most epsilon/4; this is an empirical early-stop variant.
"""
from argparse import ArgumentParser
from itertools import combinations
from math import log
from pathlib import Path
import random

import numpy as np

from queue_mle_class import dense_candidate_class
from queue_omle import core_interventions
from queue_pomdp import TRUE_WEIGHTS
from run_multistage_experiment import write_csv
from run_queue_ours import worst_policy_tv


def run_case(models, truth_index, pairs, epsilon, seed, cap):
    rng = random.Random(seed + 307)
    truth = models[truth_index]
    scores = np.zeros(len(models))
    active = np.ones(len(models), dtype=bool)
    log_likelihoods = {}
    interventions_per_round = truth.horizon - 1
    slack = log(cap * interventions_per_round * len(models) / .1)
    checkpoints = []
    for calls in range(cap + 1):
        disagreement, left, right = max(
            ((tv, i, j) for i, j, tv, _ in pairs if active[i] and active[j]),
            default=(0., -1, -1))
        if calls == 0 or calls & (calls - 1) == 0 or disagreement <= epsilon/4 or calls == cap:
            checkpoints.append(dict(seed=seed, epsilon=epsilon, calls=calls,
                                    episodes=interventions_per_round*calls,
                                    active_models=int(sum(active)),
                                    diameter=disagreement))
        if disagreement <= epsilon/4 or calls == cap:
            fit_index = max((i for i in range(len(models)) if active[i]),
                            key=lambda i: scores[i])
            return dict(seed=seed, epsilon=epsilon, cap=cap, calls=calls,
                        episodes=interventions_per_round*calls,
                        interventions_per_round=interventions_per_round,
                        stopped=disagreement <= epsilon/4,
                        diameter=disagreement, active_models=int(sum(active)),
                        fitted_tv=worst_policy_tv(truth, models[fit_index])[0],
                        truth_in_confidence_set=bool(active[truth_index])), checkpoints
        diagnostic = next(policy for i, j, tv, policy in pairs if i == left and j == right)
        for policy in core_interventions(diagnostic, truth, rng):
            trace = truth.sample(policy, rng)
            if trace not in log_likelihoods:
                log_likelihoods[trace] = np.log(
                    [model.trajectory_probability(trace) for model in models])
            scores += log_likelihoods[trace]
        active &= scores >= float(scores.max()) - slack
        if not active.any():
            raise RuntimeError("OMLE confidence set became empty")
    raise AssertionError("loop must return by its cap")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--epsilons", type=float, nargs="+", default=(.4, .2, .1, .05))
    parser.add_argument("--base-cap", type=int, default=3000)
    parser.add_argument("--max-cap", type=int, default=50000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = dense_candidate_class()
    truth = next(i for i, model in enumerate(raw)
                 if model.weights == tuple(TRUE_WEIGHTS))
    first = max((i for i in range(len(raw)) if i != truth),
                key=lambda i: worst_policy_tv(raw[truth], raw[i])[0])
    order = [first] + [i for i in range(len(raw)) if i != first]
    models = tuple(raw[i] for i in order)
    truth_index = order.index(truth)
    pairs = [(i, j, *worst_policy_tv(models[i], models[j]))
             for i, j in combinations(range(len(models)), 2)]
    finals, checkpoints, summary = [], [], []
    for epsilon in args.epsilons:
        cap = min(args.max_cap, round(args.base_cap * (.4/epsilon)**2))
        group = []
        for seed in range(args.seeds):
            final, path = run_case(models, truth_index, pairs, epsilon, seed, cap)
            group.append(final)
            finals.append(final)
            checkpoints.extend(path)
            print(f"OMLE epsilon={epsilon:g}, seed={seed}, calls={final['calls']}, "
                  f"stopped={final['stopped']}", flush=True)
        summary.append(dict(epsilon=epsilon, seeds=args.seeds, cap=cap,
                            stopped=sum(row["stopped"] for row in group),
                            median_calls=float(np.median([row["calls"] for row in group])),
                            min_calls=min(row["calls"] for row in group),
                            max_calls=max(row["calls"] for row in group),
                            median_episodes=float(np.median([row["episodes"] for row in group])),
                            median_fitted_tv=float(np.median([row["fitted_tv"] for row in group])),
                            target_passes=sum(row["fitted_tv"] <= epsilon/4 for row in group),
                            truth_covered=sum(row["truth_in_confidence_set"] for row in group)))
    write_csv(args.output / "omle_stopping_runs.csv", finals)
    write_csv(args.output / "omle_diameter_checkpoints.csv", checkpoints)
    write_csv(args.output / "omle_stopping_summary.csv", summary)


if __name__ == "__main__":
    main()
