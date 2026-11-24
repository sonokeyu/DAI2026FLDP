"""Matched finite-model baselines on the partially observed combination lock."""
from argparse import ArgumentParser
import csv
from itertools import combinations, product
from math import log
from pathlib import Path
import random
from statistics import mean, median

from pocomblock import CountPolicy, RandomPolicy, candidate_models, worst_policy_tv


HORIZONS = (4, 6)
CHECKPOINT = 64
DELTA = .1


def update_scores(models, scores, trace):
    for index, model in enumerate(models):
        probability = model.trajectory_probability(trace)
        scores[index] += log(probability) if probability > 0 else float("-inf")


def score_fit(scores):
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def run_control(models, truth, seed, horizon, budget, method):
    rng = random.Random(7117 * horizon + 89 * seed + (0 if method == "uniform" else 1))
    policy = (RandomPolicy(rng, truth.actions) if method == "uniform"
              else CountPolicy(rng, truth.actions))
    scores = [0.0] * len(models)
    records = []
    successes = 0
    for episode in range(1, budget + 1):
        trace = truth.sample(policy, rng)
        successes += trace[-1]
        update_scores(models, scores, trace)
        if method == "count":
            policy.record(trace)
        if episode % CHECKPOINT == 0:
            fitted = score_fit(scores)
            records.append(dict(method=method, horizon=horizon, seed=seed,
                                fit=fitted,
                                episodes=episode, mle_calls=episode // CHECKPOINT,
                                planning_calls=0, joint_calls=0,
                                tv=worst_policy_tv(truth, models[fitted])[0],
                                observed_success_rate=successes / episode,
                                stopped=False))
    return records


def run_omle(models, truth, seed, horizon, budget, epsilon, pair_data):
    rng = random.Random(1789 * horizon + 89 * seed)
    random_policy = RandomPolicy(rng, truth.actions)
    scores = [0.0] * len(models)
    slack = log(budget * len(models) / DELTA)
    records = []
    calls = 0
    for episode in range(0, budget, 2):
        highest = max(scores)
        active = [i for i, score in enumerate(scores) if score >= highest - slack]
        pairs = [(pair_data[i, j][0], i, j) for i, j in combinations(active, 2)]
        diameter, left, right = max(pairs, default=(0.0, -1, -1))
        fitted = score_fit(scores)
        if diameter <= epsilon / 4:
            records.append(dict(method="omle", horizon=horizon, seed=seed,
                                fit=fitted,
                                episodes=episode, mle_calls=0, planning_calls=0,
                                joint_calls=calls, tv=worst_policy_tv(truth, models[fitted])[0],
                                observed_success_rate="", stopped=True))
            return records
        for policy in (random_policy, pair_data[left, right][1]):
            update_scores(models, scores, truth.sample(policy, rng))
        calls += 1
        if (episode + 2) % CHECKPOINT == 0:
            fitted = score_fit(scores)
            records.append(dict(method="omle", horizon=horizon, seed=seed,
                                fit=fitted,
                                episodes=episode + 2, mle_calls=0, planning_calls=0,
                                joint_calls=calls, tv=worst_policy_tv(truth, models[fitted])[0],
                                observed_success_rate="", stopped=False))
    return records


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--epsilon", type=float, default=.2)
    args = parser.parse_args()
    if args.budget % CHECKPOINT:
        parser.error("budget must be divisible by the checkpoint interval")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for horizon in HORIZONS:
        models = candidate_models(horizon)
        pairs = {(i, j): worst_policy_tv(models[i], models[j])
                 for i, j in combinations(range(len(models)), 2)}
        for seed in range(args.seeds):
            truth = models[seed % len(models)]
            for method in ("uniform", "count"):
                rows.extend(run_control(models, truth, seed, horizon, args.budget, method))
            rows.extend(run_omle(models, truth, seed, horizon, args.budget,
                                 args.epsilon, pairs))
        print(f"completed horizon {horizon}", flush=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / "lock_baseline_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    last_by_run = {}
    for row in rows:
        last_by_run[row["horizon"], row["seed"], row["method"]] = row
    final = list(last_by_run.values())
    summary = []
    for horizon, method in product(HORIZONS, ("omle", "uniform", "count")):
        group = [row for row in final if row["horizon"] == horizon and row["method"] == method]
        summary.append(dict(horizon=horizon, method=method, runs=len(group),
                            budget=args.budget, epsilon=args.epsilon,
                            stop_rate=mean(row["stopped"] for row in group),
                            mean_tv=mean(row["tv"] for row in group),
                            median_tv=median(row["tv"] for row in group),
                            target_rate=mean(row["tv"] <= args.epsilon/4 for row in group),
                            median_episodes=median(row["episodes"] for row in group),
                            median_mle_calls=median(row["mle_calls"] for row in group),
                            median_joint_calls=median(row["joint_calls"] for row in group)))
    with (args.output / "lock_baseline_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
