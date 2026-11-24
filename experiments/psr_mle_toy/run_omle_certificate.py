"""Run reward-free OMLE with its confidence-set diameter stopping rule.

This is the finite two-model routing specialization of reward-free OMLE.
Each round uses its two intervention episodes, updates the confidence set,
and stops when its exact all-policy TV diameter is at most epsilon/4.
"""
from argparse import ArgumentParser
import csv
from itertools import product
from math import log
from pathlib import Path
import random
from statistics import mean, median


ACCURACIES = (0.4, 0.2, 0.1, 0.05)
DELTA = 0.1


def run_case(epsilon, truth, seed, episode_cap):
    p0, p1 = .2, .2 + epsilon
    slack = log(episode_cap * 2 / DELTA)
    rng = random.Random(seed + 100003 * truth)
    hits = misses = episodes = rounds = pair_scores = 0
    while True:
        loglik = [hits * log(p) + misses * log(1 - p) for p in (p0, p1)]
        best = max(loglik)
        # Use the current likelihood confidence set, as in OMLE; do not
        # permanently discard a model based on a previous round.
        present = [loglik[i] >= best - slack for i in range(2)]
        if not any(present):
            raise RuntimeError("OMLE confidence set became empty")
        diameter = .5 * epsilon if all(present) else 0.0
        choice = max((i for i in range(2) if present[i]), key=lambda i: loglik[i])
        if diameter <= epsilon / 4 or episodes + 2 > episode_cap:
            return dict(epsilon=epsilon, target_tv=epsilon / 4, truth=truth, seed=seed,
                        episodes=episodes, rounds=rounds, joint_calls=rounds,
                        final_diameter=diameter, fitted_model=choice,
                        exact_tv=0.0 if choice == truth else .5 * epsilon,
                        truth_in_confidence_set=present[truth], stopped=diameter <= epsilon / 4,
                        pair_scores=pair_scores, cap=episode_cap)
        # Algorithm 2's two interventions: uniform route then current
        # disagreement route. Mode a is selected by both on the negative cue.
        if all(present):
            pair_scores += 1
        for informative_probability in (.25, .5):
            if rng.random() < informative_probability:
                outcome = rng.random() < (p0 if truth == 0 else p1)
                hits += int(outcome)
                misses += int(not outcome)
        episodes += 2
        rounds += 1


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--cap", type=int, default=16384)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [run_case(eps, truth, seed, args.cap)
            for eps, truth, seed in product(ACCURACIES, (0, 1), range(args.seeds))]
    with (args.output / "omle_certificate_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for eps, truth in product(ACCURACIES, (0, 1)):
        group = [r for r in rows if r["epsilon"] == eps and r["truth"] == truth]
        summary.append(dict(epsilon=eps, target_tv=eps / 4, truth=truth, runs=len(group),
                            stopped=sum(r["stopped"] for r in group),
                            median_episodes=median(r["episodes"] for r in group),
                            median_joint_calls=median(r["joint_calls"] for r in group),
                            median_pair_scores=median(r["pair_scores"] for r in group),
                            success_rate=mean(r["exact_tv"] <= eps / 4 for r in group),
                            confidence_coverage=mean(r["truth_in_confidence_set"] for r in group),
                            max_tv=max(r["exact_tv"] for r in group), cap=args.cap))
    with (args.output / "omle_certificate_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"completed {len(rows)} certified OMLE runs")


if __name__ == "__main__":
    main()
