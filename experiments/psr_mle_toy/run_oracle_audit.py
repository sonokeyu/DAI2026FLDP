"""Audit optimization interfaces, accuracy dependence, and bonus stopping."""
from argparse import ArgumentParser
from itertools import product
from math import log
from pathlib import Path
from statistics import median
from unittest.mock import patch

import numpy as np

import compare
import routing
from routing import EPISODES, H, Design, RoutingModel, all_policy_tv, law, mix
from run_suite import csv
from rank_sweep import CANDIDATE_RANKS, run_rank_case
from run_target_accuracy import sustained_checkpoints


ACCURACIES = (0.4, 0.2, 0.1, 0.05)
SUCCESS = 0.9


def interface_audit():
    rows = []
    for number in (2, 4, 8, 16, 32):
        models = [RoutingModel(float(p)) for p in np.linspace(.2, .8, number)]
        with patch.object(compare, "all_policy_tv", wraps=all_policy_tv) as score:
            compare.disagreement_policy(models)
        with patch.object(routing, "fit_mle", wraps=routing.fit_mle) as fit, \
                patch.object(routing, "plan_bonus", wraps=routing.plan_bonus) as plan:
            routing.run_algorithm(models, 0, 1, 0, fixed_budget=(0, 8))
        assert score.call_count == number * (number - 1) // 2
        assert fit.call_count == plan.call_count == 1
        assert fit.call_args.args[0].shape[0] == number
        assert plan.call_args.args[0] in models
        rows.append(dict(candidates=number, omle_joint_calls=1,
                         omle_model_pairs_scored=score.call_count,
                         ours_mle_calls=fit.call_count, ours_planning_calls=plan.call_count,
                         ours_mle_candidates_scored=fit.call_args.args[0].shape[0],
                         ours_models_passed_to_planner=1, ours_model_pairs_scored=0))
    return rows


def intervention_probabilities(models):
    """Controller probabilities of the only informative cue, route, mode."""
    informative = [i for i, ep in enumerate(EPISODES)
                   if (ep.cue, ep.route, ep.mode) == ("-", "R", "a")]
    full = compare.disagreement_policy(models)
    single = compare.disagreement_policy(models[:1])
    designs = [
        Design((.5, .5), (1.,) * 4),
        Design(full.routes, (.5,) * 4),
        Design(single.routes, (.5,) * 4),
    ]
    return tuple(float(law(models[0], mix(d))[0, informative].sum()) for d in designs)


def omle_checkpoints(models, seeds, cap):
    """Batch over repetitions of Algorithm 2, using its two sufficient counts."""
    first_cut, second_cut, second_singleton = intervention_probabilities(models)
    rng = np.random.default_rng(702)
    truth = np.repeat(np.arange(2), seeds)
    p0, p1 = [model.negative_right for model in models]
    p = np.where(truth == 0, p0, p1)
    slack = log(cap * len(models) / 0.1)
    hit_log = log(p1 / p0)
    miss_log = log((1 - p1) / (1 - p0))
    hits = np.zeros(len(truth), dtype=int)
    misses = hits.copy()
    searched_pairs = hits.copy()
    present = np.ones((2, len(truth)), dtype=bool)
    rounds = cap // 2
    fit_correct = np.empty((rounds + 1, 2, seeds), dtype=bool)
    pair_counts = np.empty((rounds + 1, 2, seeds), dtype=int)
    for k in range(rounds + 1):
        difference = hits * hit_log + misses * miss_log
        present[0] &= difference <= slack
        present[1] &= difference >= -slack
        if np.any(~present.any(axis=0)):
            raise RuntimeError("empty OMLE confidence set")
        both = present.all(axis=0)
        choice = (~present[0]) | (present[1] & (difference > 0))
        fit_correct[k] = (choice == truth).reshape(2, seeds)
        pair_counts[k] = searched_pairs.reshape(2, seeds)
        if k == rounds:
            break
        # Only the terminal outcome after (-, R, a) changes these likelihoods.
        # All other logged trajectories contribute the same factor to both.
        draw = rng.random((4, len(truth)))
        relevant0 = draw[0] < first_cut
        relevant1 = draw[1] < np.where(both, second_cut, second_singleton)
        successful0, successful1 = draw[2] < p, draw[3] < p
        hits += (relevant0 & successful0).astype(int) + (relevant1 & successful1)
        misses += (relevant0 & ~successful0).astype(int) + (relevant1 & ~successful1)
        searched_pairs += both
    return fit_correct, pair_counts


def accuracy_audit(seeds, cap):
    summary, stops, curves = [], [], []
    for eps in ACCURACIES:
        print(f"Accuracy epsilon={eps:g}", flush=True)
        # Wrong model TV is eps/2, twice the evaluation target.
        models = [RoutingModel(.2), RoutingModel(.2 + eps)]
        target = eps / 4
        wrong_tv = all_policy_tv(*models)
        assert wrong_tv > target
        fit_correct, pair_counts = omle_checkpoints(models, seeds, cap)
        success_by_truth = fit_correct.mean(axis=2)
        worst_success = success_by_truth.min(axis=1)
        # Choose an empirical fixed K by replay evaluation. OMLE does not
        # observe this statistic or use it to choose its stopping time.
        selected_k = int(sustained_checkpoints(1 - worst_success, 1 - SUCCESS))
        reached = selected_k < len(worst_success)
        for k, (low, high) in enumerate(success_by_truth):
            curves.append(dict(epsilon=eps, target_tv=target, rounds=k, episodes=2*k,
                               lower_truth_success=float(low), upper_truth_success=float(high),
                               worst_truth_success=float(worst_success[k]),
                               median_pairs_scored=float(np.median(pair_counts[k]))))
        summary.append(dict(
            epsilon=eps, target_tv=target, method="omle", reached=reached,
            episodes=2*selected_k if reached else None,
            joint_calls=selected_k if reached else None,
            confidence_updates=selected_k if reached else None,
            median_nontrivial_joint_calls=float(np.median(pair_counts[selected_k])) if reached else None,
            worst_truth_success=float(worst_success[selected_k]) if reached else float(worst_success[-1]),
            episodes_cap=cap, ordinary_mle_calls=None, single_model_plans=None,
            stops=None, false_stops=None, runs=2*seeds, max_stop_tv=None,
        ))
        condition = []
        for truth, seed in product((0, 1), range(seeds)):
            trace = routing.run_algorithm(models, truth, eps, seed, theory=True)
            final = trace[-1]
            row = dict(epsilon=eps, target_tv=target, method="ours", seed=seed,
                       truth=truth, all_rounds_calibrated=all(r["calibrated"] for r in trace),
                       certificate_consistent=all(
                           not r["calibrated"] or r["policy_tv"] <= r["bonus"] + 1e-12
                           for r in trace
                       ),
                       known_dimension_cap=routing.Budget(eps).cap(),
                       **final)
            condition.append(row)
        stops.extend(condition)
        success = min(sum(r["stopped"] and r["policy_tv"] <= target and r["truth"] == truth
                          for r in condition) / seeds for truth in (0, 1))
        stopped = [r for r in condition if r["stopped"]]
        summary.append(dict(
            epsilon=eps, target_tv=target, method="ours", reached=success >= SUCCESS,
            episodes=median(r["episodes"] for r in condition), joint_calls=0, confidence_updates=0,
            median_nontrivial_joint_calls=0, worst_truth_success=success, episodes_cap=None,
            ordinary_mle_calls=median(r["mle_calls"] for r in condition),
            single_model_plans=median(r["exploration_plans"] for r in condition),
            stops=len(stopped), false_stops=sum(r["policy_tv"] > target for r in stopped),
            runs=len(condition), max_stop_tv=max((r["policy_tv"] for r in stopped), default=None),
        ))
    return summary, stops, curves


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--cap", type=int, default=16384)
    args = parser.parse_args()
    if args.seeds < 1 or args.cap < 2 or args.cap % 2:
        parser.error("use positive seeds and an even episode cap")
    args.output.mkdir(parents=True, exist_ok=True)
    csv(args.output / "interface_audit.csv", interface_audit())
    summary, stops, curves = accuracy_audit(args.seeds, args.cap)
    ranks = [run_rank_case(h, rank, eps, seed, theory=True)
             for h, rank, eps, seed in product((4, 6, 8), CANDIDATE_RANKS, (0.4, 0.1),
                                               range(args.seeds))]
    for filename, rows in (("oracle_summary.csv", summary), ("oracle_stops.csv", stops),
                           ("omle_accuracy_curves.csv.gz", curves), ("oracle_rank_stops.csv", ranks)):
        csv(args.output / filename, rows)


if __name__ == "__main__":
    main()
