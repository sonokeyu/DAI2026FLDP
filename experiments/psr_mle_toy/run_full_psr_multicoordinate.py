"""Run the complete fitted-PSR loop on an eight-branch routing system.

The four uncertain coordinates are the (cue, route, mode-a) terminal
Bernoulli probabilities. Mode-b outcomes are fixed at 0.5. The misspecified
condition samples true probabilities continuously between the two candidate
values, so the finite MLE class cannot represent every true environment.
"""
import csv
from itertools import product
from pathlib import Path
import random
from statistics import mean, median
from argparse import ArgumentParser

import numpy as np
from routing import CUES, EPISODES, MODES, ROUTES, run_algorithm


COORDINATES = tuple(product(CUES, ROUTES, ("a",)))
COORDINATE_INDEX = {coordinate: index for index, coordinate in enumerate(COORDINATES)}
ENDPOINTS = (.2, .4)
ACCURACIES = (.2, .1)


class BranchModel:
    def __init__(self, probabilities):
        self.probabilities = tuple(probabilities)
        self._weights = np.array([
            .5 * (p if episode.outcome else 1-p)
            for episode in EPISODES
            for p in (self.p(episode.cue, episode.route, episode.mode),)
        ])

    def p(self, cue, route, mode):
        if mode == "b":
            return .5
        return self.probabilities[COORDINATE_INDEX[(cue, route, mode)]]

    def observation_weights(self):
        return self._weights


def candidate_class(size=16):
    patterns = list(product(ENDPOINTS, repeat=len(COORDINATES)))
    rng = random.Random(49157)
    selected = rng.sample(patterns, size)
    return [BranchModel(bits) for bits in selected]


def continuous_truth(seed):
    rng = random.Random(7103 + seed)
    return BranchModel([rng.uniform(*ENDPOINTS) for _ in COORDINATES])


def run_condition(models, seed, epsilon, continuous, scale):
    truth = continuous_truth(seed) if continuous else models[seed % len(models)]
    index = None if continuous else seed % len(models)
    rows = run_algorithm(models, index, epsilon, seed + 991, scale=scale,
                         truth_model=truth if continuous else None)
    final = rows[-1]
    return dict(condition="continuous_truth" if continuous else "realizable",
                seed=seed, epsilon=epsilon, episodes=final["episodes"],
                rounds=final["round"], mle_calls=final["mle_calls"],
                planning_calls=final["exploration_plans"],
                stopped=final["stopped"], bonus=final["bonus"],
                worst_policy_tv=final["policy_tv"],
                target_tv=epsilon/4,
                fitted_index=final["fit"],
                calibrated=final["calibrated"])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--scale", type=float, default=.005,
                        help="batch multiplier for the paper's complete loop")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = candidate_class()
    rows = [run_condition(models, seed, epsilon, continuous, args.scale)
            for epsilon, continuous, seed in product(ACCURACIES, (False, True),
                                                      range(args.seeds))]
    with (args.output / "full_psr_multicoordinate_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for condition, epsilon in product(("realizable", "continuous_truth"), ACCURACIES):
        group = [r for r in rows if r["condition"] == condition and r["epsilon"] == epsilon]
        summary.append(dict(condition=condition, epsilon=epsilon, runs=len(group),
                            stop_rate=mean(r["stopped"] for r in group),
                            median_tv=median(r["worst_policy_tv"] for r in group),
                            p90_tv=sorted(r["worst_policy_tv"] for r in group)[int(.9*(len(group)-1))],
                            within_target=mean(r["worst_policy_tv"] <= epsilon/4 for r in group),
                            median_episodes=median(r["episodes"] for r in group),
                            median_mle_calls=median(r["mle_calls"] for r in group),
                            median_planning_calls=median(r["planning_calls"] for r in group)))
    with (args.output / "full_psr_multicoordinate_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(f"completed {len(rows)} full-PSR runs over {len(COORDINATES)} uncertain branches")


if __name__ == "__main__":
    main()
