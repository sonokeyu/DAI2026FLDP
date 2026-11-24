"""Exact two-model specialization of All-Policy Model-Estimation E2D.

For the routing model class, every policy either hides the distinguishing
Bernoulli coordinate or reveals it after the negative cue. The E2D saddle
point can therefore be solved in closed form: exploration puts all mass on
the revealing policy, while the output-model distribution is uniform over the
two symmetric candidates. The multiplicative model weights are updated with
eta=1/2 exactly as in the algorithm. MLE is used only as a deterministic tie
break among the two output models, which have equal objective value under the
uniform output distribution in this symmetric instance.

This is an exact finite-instance algorithm specialization, not a general E2D
solver or a reproduction of the paper's full experiments.
"""
from argparse import ArgumentParser
import csv
from itertools import product
from math import log
from pathlib import Path
import random
from statistics import mean


ACCURACIES = (0.4, 0.2, 0.1, 0.05)
ETA = .5
SUCCESS = .9


def run_case(epsilon, truth, seed, cap):
    rng = random.Random(seed + 104729 * truth)
    p = (.2, .2 + epsilon)
    log_weights = [0.0, 0.0]
    hits = misses = 0
    curve = []
    correct = False
    for t in range(cap + 1):
        # Exact saddle point of the all-policy model-estimation E2D objective
        # for two symmetric models and a single informative policy.
        # The output distribution is (1/2,1/2); MLE resolves its tie.
        mle = 1 if log_weights[1] > log_weights[0] else 0
        correct = mle == truth
        if t % 16 == 0 or t == cap:
            curve.append(correct)
        if t == cap:
            break
        # E2D explores with the informative policy, which reaches the
        # distinguishing branch with probability 1/2 due to the cue.
        if rng.random() < .5:
            outcome = rng.random() < p[truth]
            hits += int(outcome)
            misses += int(not outcome)
            for i, model_p in enumerate(p):
                log_weights[i] += ETA * (log(model_p) if outcome else log(1-model_p))
    return dict(epsilon=epsilon, truth=truth, seed=seed, curve=curve,
                informative_samples=hits + misses)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--cap", type=int, default=16384)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = [run_case(eps, truth, seed, args.cap)
            for eps, truth, seed in product(ACCURACIES, (0, 1), range(args.seeds))]
    curve_rows = []
    summary = []
    for eps in ACCURACIES:
        group = [r for r in runs if r["epsilon"] == eps]
        hits_by_truth = []
        for truth in (0, 1):
            sub = [r for r in group if r["truth"] == truth]
            checkpoints = []
            episode_grid = list(range(0, args.cap + 1, 16))
            if episode_grid[-1] != args.cap:
                episode_grid.append(args.cap)
            for index, t in enumerate(episode_grid):
                rate = mean(r["curve"][index] for r in sub)
                checkpoints.append(rate)
                curve_rows.append(dict(epsilon=eps, truth=truth, episodes=t,
                                       correct_rate=rate, game_oracle_calls=t,
                                       optimizer_calls=t))
            # Select the first fixed K whose success is >=90% for both truths
            # and remains there through the analysis cap. This is an external
            # evaluation rule, not an E2D stopping condition.
            hits_by_truth.append(checkpoints)
        eligible = [i for i in range(len(episode_grid))
                    if min(hits_by_truth[0][i], hits_by_truth[1][i]) >= SUCCESS
                    and all(min(hits_by_truth[0][j], hits_by_truth[1][j]) >= SUCCESS
                            for j in range(i, len(episode_grid)))]
        selected = episode_grid[eligible[0]] if eligible else None
        selected_index = eligible[0] if eligible else None
        summary.append(dict(
            epsilon=eps, target_tv=eps/4, runs=len(group),
            selected_fixed_episodes=selected,
            selected_game_optimizer_calls=selected,
            success_rate_at_selected=(min(hits_by_truth[0][selected_index],
                                          hits_by_truth[1][selected_index]) if selected_index is not None else None),
            cap=args.cap, eta=ETA, update_calls_per_episode=1,
            output_model_distribution="uniform; MLE tie break",
        ))
    for name, rows in (("e2d_curves.csv", curve_rows), ("e2d_summary.csv", summary)):
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"completed {len(runs)} exact E2D specialization runs")


if __name__ == "__main__":
    main()
