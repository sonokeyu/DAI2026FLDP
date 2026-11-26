"""Epsilon sweep for the fitted bonus certificate on the 64-state POMDP."""
from argparse import ArgumentParser
from collections import Counter
from itertools import combinations
from math import log, sqrt
from pathlib import Path
import random

import numpy as np

from queue_mle_class import candidate_class, finite_mle
from queue_pomdp import STABILITY, TRUE_WEIGHTS
from queue_psr import StageSupport
from run_multistage_experiment import write_csv
from run_queue_ours import design_law, plan_bonus, worst_policy_tv


def run_case(models, truth_index, epsilon, seed, base_batch, max_rounds):
    truth = models[truth_index]
    batch = round(base_batch * (.4 / epsilon) ** 2)
    rng = np.random.default_rng(seed + 1907)
    data = Counter()
    histories = [Counter() for _ in range(4)]
    initial_count = episodes = 0
    rows = []
    beta = max(1., log(3 * len(models) / .1))
    energy = 8 * STABILITY**2 * 2 * beta

    def draw(law, size, training_stage=None):
        nonlocal episodes
        if not size:
            return
        traces = tuple(law)
        probabilities = np.asarray([law[trace] for trace in traces])
        probabilities /= probabilities.sum()
        counts = rng.multinomial(size, probabilities)
        episodes += size
        for trace, count in zip(traces, counts):
            if count:
                count = int(count)
                data[trace] += count
                if training_stage is not None:
                    histories[training_stage][trace[:2 * training_stage]] += count

    accurate_round = None
    for round_number in range(1, max_rounds + 1):
        fit_index = finite_mle(models, data)
        fit = models[fit_index]
        supports = [StageSupport(fit, stage, histories[stage], len(models))
                    for stage in range(4)]
        root_bonus = min(STABILITY, sqrt(energy / initial_count)) if initial_count else STABILITY
        bonus, policy, coefficients = plan_bonus(fit, supports, root_bonus)
        true_tv = worst_policy_tv(truth, fit)[0]
        if accurate_round is None and true_tv <= epsilon / 4:
            accurate_round = round_number
        stopped = bonus <= 3 * epsilon / 16
        rows.append(dict(seed=seed, epsilon=epsilon, batch=batch,
                         round=round_number, episodes=episodes,
                         mle_calls=round_number, planning_calls=round_number,
                         fit_index=fit_index, bonus=bonus, sup_tv=true_tv,
                         target_tv=epsilon / 4, stop_threshold=3 * epsilon / 16,
                         accurate=(true_tv <= epsilon / 4), stopped=stopped,
                         certificate_valid=(true_tv <= bonus + 1e-8)))
        if stopped:
            break

        initial_count += batch
        draw(design_law(truth, policy, stage=0), batch)
        for stage in range(4):
            core_count = int(rng.binomial(batch, .5))
            draw(design_law(truth, policy, stage=stage), core_count, stage)
            draw(design_law(truth, policy, stage=stage, branch="uniform"),
                 batch - core_count, stage)
        choices = [(None, "full", cut) for cut in range(4)]
        choices += [(stage, branch, cut) for stage in range(4)
                    for branch in ("u", "w") for cut in range(4)]
        probabilities = [1 / 12] * 4 + [1 / 48] * 32
        sizes = rng.multinomial(batch, probabilities)
        for (stage, branch, cut), size in zip(choices, sizes):
            if size:
                draw(design_law(truth, policy, stage=stage, branch=branch,
                                coefficients=coefficients, cut=cut), int(size))

    for row in rows:
        row["first_accurate_round"] = accurate_round or ""
        row["delay_to_stop"] = (rows[-1]["round"] - accurate_round
                                if accurate_round else "")
    return rows


def run_omle(models, truth_index, epsilon, seed, max_calls):
    rng = random.Random(seed + 409)
    uniform = type("Uniform", (), {"__getitem__": lambda self, history: rng.randrange(2)})()
    pair_data = {(i, j): worst_policy_tv(models[i], models[j])
                 for i, j in combinations(range(len(models)), 2)}
    scores = np.zeros(len(models))
    rows = []
    slack = log(max_calls * len(models) / .1)
    truth = models[truth_index]
    for call in range(1, max_calls + 1):
        best = float(scores.max())
        active = [i for i, score in enumerate(scores) if score >= best - slack]
        disagreement, left, right = max(
            ((pair_data[i, j][0], i, j) for i, j in combinations(active, 2)),
            default=(0., -1, -1))
        fit_index = int(scores.argmax())
        tv = worst_policy_tv(truth, models[fit_index])[0]
        rows.append(dict(method="omle", seed=seed, epsilon=epsilon,
                         episodes=2 * (call - 1), calls=call - 1,
                         fitted_tv=tv, diameter=disagreement,
                         stopped=disagreement <= epsilon / 4))
        if disagreement <= epsilon / 4:
            break
        diagnostic_policy = pair_data[left, right][1]
        for policy in (uniform, diagnostic_policy):
            trace = truth.sample(policy, rng)
            for i, model in enumerate(models):
                scores[i] += log(model.trajectory_probability(trace))
    return rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--epsilons", type=float, nargs="+", default=(.4, .2, .1, .05))
    parser.add_argument("--base-batch", type=int, default=100_000)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--omle-max-calls", type=int, default=5000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = candidate_class()
    truth_index = next(i for i, model in enumerate(models)
                       if model.weights == tuple(TRUE_WEIGHTS))
    runs, finals, omle_runs, omle_finals = [], [], [], []
    for epsilon in args.epsilons:
        for seed in range(args.seeds):
            path = run_case(models, truth_index, epsilon, seed,
                            args.base_batch, args.rounds)
            runs.extend(path)
            finals.append(path[-1])
            print(f"epsilon={epsilon}, seed={seed}, calls={len(path)}, "
                  f"TV={path[-1]['sup_tv']:.6g}, bonus={path[-1]['bonus']:.6g}, "
                  f"delay={path[-1]['delay_to_stop']}", flush=True)
        omle_path = run_omle(models, truth_index, epsilon, 20_000,
                             args.omle_max_calls)
        omle_runs.extend(omle_path)
        omle_finals.append(omle_path[-1])
        print(f"OMLE epsilon={epsilon}, calls={omle_path[-1]['calls']}, "
              f"stopped={omle_path[-1]['stopped']}", flush=True)
    write_csv(args.output / "epsilon_runs.csv", runs)
    write_csv(args.output / "epsilon_final.csv", finals)
    write_csv(args.output / "omle_epsilon_runs.csv", omle_runs)
    write_csv(args.output / "omle_epsilon_final.csv", omle_finals)
    summary = []
    for epsilon in args.epsilons:
        group = [row for row in finals if row["epsilon"] == epsilon]
        summary.append(dict(epsilon=epsilon, seeds=len(group),
                            stopped=sum(row["stopped"] for row in group),
                            certificate_passed=sum(row["certificate_valid"] for row in group),
                            stop_accuracy=sum(row["accurate"] for row in group),
                            median_calls=float(np.median([row["mle_calls"] for row in group])),
                            median_episodes=float(np.median([row["episodes"] for row in group])),
                            median_stop_tv=float(np.median([row["sup_tv"] for row in group])),
                            median_stop_bonus=float(np.median([row["bonus"] for row in group])),
                            median_first_accurate_round=float(np.median([
                                row["first_accurate_round"] for row in group])),
                            median_delay=float(np.median([row["delay_to_stop"] for row in group]))))
    write_csv(args.output / "epsilon_summary.csv", summary)


if __name__ == "__main__":
    main()
