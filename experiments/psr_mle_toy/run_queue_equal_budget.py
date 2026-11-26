"""Paired epsilon sweep with identical episode budgets and a fixed model class.

OMLE follows its prescribed set of PSR core-intervention policies per round.
Ours uses the theorem's batch scaling B proportional to epsilon**-2.  This
experiment compares interface calls at equal *allocated* sample budgets;
the separate certificate sweep checks stopping.
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
from run_queue_matched import ours
from run_queue_ours import worst_policy_tv


def omle_path(models, truth_index, seed, budget, epsilon, pairs):
    """Run one joint search per core-intervention round."""
    rng = random.Random(seed + 307)
    scores = np.zeros(len(models))
    active = np.ones(len(models), dtype=bool)
    log_likelihoods = {}
    truth = models[truth_index]
    slack = log(budget * len(models) / .1)
    nontrivial_calls = 0
    interventions_per_round = truth.horizon - 1
    for call in range(1, budget // interventions_per_round + 1):
        disagreement, left, right = max(
            ((pairs[i, j][0], i, j) for i, j in combinations(range(len(models)), 2)
             if active[i] and active[j]),
            default=(0., -1, -1))
        nontrivial_calls += int(left >= 0)
        diagnostic = pairs[left, right][1] if left >= 0 else type(
            "Zero", (), {"__getitem__": lambda self, history: 0})()
        for policy in core_interventions(diagnostic, truth, rng):
            trace = truth.sample(policy, rng)
            if trace not in log_likelihoods:
                log_likelihoods[trace] = np.log(
                    [model.trajectory_probability(trace) for model in models])
            scores += log_likelihoods[trace]
        active &= scores >= float(scores.max()) - slack
        if not active.any():
            raise RuntimeError("OMLE confidence set became empty")
    fit_index = max((i for i in range(len(models)) if active[i]),
                    key=lambda i: scores[i])
    return dict(method="omle", seed=seed, epsilon=epsilon,
                episodes=budget, mle_calls=0, planning_calls=0,
                joint_calls=budget // interventions_per_round,
                nontrivial_joint_calls=nontrivial_calls,
                fit_index=fit_index,
                tv=worst_policy_tv(truth, models[fit_index])[0],
                diameter=disagreement)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--epsilons", type=float, nargs="+", default=(.4, .2, .1))
    parser.add_argument("--base-batch", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_models = dense_candidate_class()
    raw_truth_index = next(i for i, model in enumerate(raw_models)
                       if model.weights == tuple(TRUE_WEIGHTS))
    # Likelihood ties at zero samples select index zero.  Put the most
    # separated alternative there for both learners; the model set is fixed.
    first = max((i for i in range(len(raw_models)) if i != raw_truth_index),
                key=lambda i: worst_policy_tv(raw_models[raw_truth_index], raw_models[i])[0])
    order = [first] + [i for i in range(len(raw_models)) if i != first]
    models = tuple(raw_models[i] for i in order)
    truth_index = order.index(raw_truth_index)
    budgets = {12 * round(args.base_batch * (.4 / epsilon) ** 2): epsilon
               for epsilon in args.epsilons}
    if len(budgets) != len(args.epsilons):
        raise ValueError("epsilon values must map to distinct episode budgets")
    pairs = {(i, j): worst_policy_tv(models[i], models[j])
             for i, j in combinations(range(len(models)), 2)}
    all_rows = []
    for seed in range(args.seeds):
        for budget, epsilon in sorted(budgets.items()):
            omle_row = omle_path(models, truth_index, seed, budget, epsilon, pairs)
            all_rows.append(omle_row)
            batch = budget // 12
            ours_rows = ours(models, truth_index, seed, budget, batch, epsilon)
            final = ours_rows[-1]
            if final["episodes"] != budget:
                raise RuntimeError("ours stopped early, so episode budgets are not matched")
            all_rows.append(dict(method="ours", seed=seed, epsilon=epsilon,
                                 episodes=final["episodes"],
                                 mle_calls=final["mle_calls"],
                                 planning_calls=final["planning_calls"],
                                 joint_calls=0, nontrivial_joint_calls=0,
                                 fit_index="", tv=final["tv"],
                                 diameter="", bonus=final["bonus"],
                                 stopped=final["stopped"],
                                 initial_tv=ours_rows[0]["tv"]))
            print(f"seed={seed}, epsilon={epsilon:g}, N={budget}, "
                  f"ours={final['mle_calls']} MLE+planning, "
                  f"OMLE={omle_row['joint_calls']} joint", flush=True)
    write_csv(args.output / "equal_budget_runs.csv", all_rows)
    summary = []
    for budget, epsilon in sorted(budgets.items()):
        ours_group = [row for row in all_rows if row["method"] == "ours" and row["episodes"] == budget]
        omle_group = [row for row in all_rows if row["method"] == "omle" and row["episodes"] == budget]
        summary.append(dict(epsilon=epsilon, inverse_epsilon=1/epsilon,
                            candidates=len(models),
                            episodes=budget, seeds=args.seeds,
                            initial_tv=float(np.median([row["initial_tv"] for row in ours_group])),
                            ours_mle_calls=int(np.median([row["mle_calls"] for row in ours_group])),
                            ours_planning_calls=int(np.median([row["planning_calls"] for row in ours_group])),
                            omle_joint_calls=int(np.median([row["joint_calls"] for row in omle_group])),
                            omle_nontrivial_calls=int(np.median([row["nontrivial_joint_calls"] for row in omle_group])),
                            ours_median_tv=float(np.median([row["tv"] for row in ours_group])),
                            omle_median_tv=float(np.median([row["tv"] for row in omle_group])),
                            ours_accuracy=sum(row["tv"] <= epsilon/4 for row in ours_group),
                            omle_accuracy=sum(row["tv"] <= epsilon/4 for row in omle_group),
                            ours_certificate_stops=sum(row["stopped"] for row in ours_group)))
    write_csv(args.output / "equal_budget_summary.csv", summary)


if __name__ == "__main__":
    main()
