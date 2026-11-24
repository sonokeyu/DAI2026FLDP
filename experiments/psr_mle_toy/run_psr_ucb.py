"""One-coordinate PSR-UCB specialization for the routing toy.

The original algorithm is general and uses PSR prediction features. In this
instance, all model uncertainty lives in one Bernoulli prediction coordinate:
the outcome after cue '-', route R, and revealing mode a. Its normalized
feature is therefore the indicator of visiting that branch, and the inverse
Gram bonus reduces to alpha / sqrt(lambda + n). The exploration planner picks
the policy that visits that branch whenever the cue allows it.

This is a faithful scalar specialization of the paper's feature bonus and
confidence-constrained MLE, not a general PSR-UCB implementation.
"""
from argparse import ArgumentParser
import csv as csv_module
import random
from itertools import product
from math import log, sqrt
from pathlib import Path
from statistics import mean, median

ACCURACIES = (0.4, 0.2, 0.1, 0.05)
DELTA = 0.1


def run_case(epsilon, truth_index, seed, cap):
    """Run until the scalar PSR-UCB bonus falls below epsilon/4 or cap."""
    rng = random.Random(seed)
    # A finite-horizon confidence schedule, fixed before seeing samples.
    alpha = sqrt(log(2 * cap / DELTA) / 2)
    beta = log(2 * cap / DELTA)
    target_tv = epsilon / 4
    informative = 0
    successes = 0
    episodes = 0
    mle_calls = 0
    planning_calls = 0
    fit_index = 1  # fixed tie convention matches the theory audit
    rows = []

    while True:
        # One-dimensional leverage bonus for the only unresolved PSR feature.
        # The cue occurs with probability 1/2, so the policy-level bonus has
        # the same 1/2 occupancy factor as the exact worst-policy TV.
        bonus = min(0.5 * alpha / sqrt(1 + informative), 0.5)
        # Fit the likelihood confidence set and plan once per outer round.
        loglik = [0.0, 0.0]
        if informative:
            for i, p in enumerate((.2, .2 + epsilon)):
                loglik[i] = (successes * log(p)
                              + (informative - successes) * log(1 - p))
        mle_calls += 1
        planning_calls += 1
        best_loglik = max(loglik)
        confidence = [score >= best_loglik - beta for score in loglik]
        candidates = [i for i, active in enumerate(confidence) if active]
        if candidates:
            fit_index = max(candidates, key=lambda i: loglik[i])
        fitted_tv = 0.0 if fit_index == truth_index else 0.5 * epsilon
        stopped = bonus <= target_tv
        row = dict(
            epsilon=epsilon, target_tv=target_tv, truth=truth_index, seed=seed,
            episodes=episodes, informative_episodes=informative, fit=fit_index,
            policy_tv=fitted_tv, bonus=bonus, stopped=stopped,
            bonus_dominates_observed_error=(fitted_tv <= bonus + 1e-12),
            target_met=(fitted_tv <= target_tv + 1e-12),
            confidence_size=sum(confidence), mle_calls=mle_calls,
            planning_calls=planning_calls, alpha=alpha, beta=beta,
        )
        if episodes % 128 == 0 or stopped or episodes >= cap:
            rows.append(row)
        if stopped or episodes >= cap:
            break

        # PSR-UCB's greedy planner chooses route R and mode a after cue '-'.
        # The cue itself is generated uniformly by the environment.
        cue_is_negative = rng.random() < 0.5
        if cue_is_negative:
            informative += 1
            p_true = .2 if truth_index == 0 else .2 + epsilon
            successes += int(rng.random() < p_true)
        episodes += 1
    return rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--cap", type=int, default=16384)
    args = parser.parse_args()
    if args.seeds < 1 or args.cap < 1:
        parser.error("seeds and episode cap must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [run_case(epsilon, truth, seed, args.cap)
            for epsilon, truth, seed in product(ACCURACIES, (0, 1), range(args.seeds))]
    def write(path, records):
        records = list(records)
        with path.open("w", newline="") as stream:
            writer = csv_module.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    write(args.output / "psr_ucb_rounds.csv", (row for run in rows for row in run))
    summary = []
    for epsilon in ACCURACIES:
        runs = [run[-1] for run in rows if run[-1]["epsilon"] == epsilon]
        for truth in (0, 1):
            group = [r for r in runs if r["truth"] == truth]
            summary.append(dict(
                epsilon=epsilon, target_tv=epsilon / 4, truth=truth,
                runs=len(group), stopped=sum(r["stopped"] for r in group),
                stopped_bonus_checks=sum(r["stopped"] and r["bonus_dominates_observed_error"]
                                          for r in group),
                stopped_target_met=sum(r["stopped"] and r["target_met"] for r in group),
                median_episodes=float(median([r["episodes"] for r in group])),
                median_informative_episodes=float(median([r["informative_episodes"] for r in group])),
                median_mle_calls=float(median([r["mle_calls"] for r in group])),
                median_planning_calls=float(median([r["planning_calls"] for r in group])),
                mean_final_tv=float(mean([r["policy_tv"] for r in group])),
                worst_final_tv=float(max(r["policy_tv"] for r in group)),
                cap=args.cap,
            ))
    write(args.output / "psr_ucb_summary.csv", summary)
    print(f"wrote {len(rows)} runs ({sum(map(len, rows))} rounds) to {args.output}")


if __name__ == "__main__":
    main()
