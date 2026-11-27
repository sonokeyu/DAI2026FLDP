"""Full bonus-stopping ours versus core-intervention OMLE on 73 PSRs."""
from argparse import ArgumentParser
from itertools import combinations
from math import log
from pathlib import Path
import random

import numpy as np

from queue_mle_class import fine_candidate_class
from queue_omle import core_interventions
from queue_pomdp import TRUE_WEIGHTS
from run_multistage_experiment import write_csv
from run_queue_epsilon_certificate import run_case as run_ours
from run_queue_ours import worst_policy_tv


def run_omle(models, truth_index, pairs, epsilon, seed, cap):
    rng = random.Random(seed + 307)
    truth = models[truth_index]
    pair_i, pair_j, pair_tv, pair_policies = pairs
    scores = np.zeros(len(models))
    active = np.ones(len(models), dtype=bool)
    log_cache = {}
    slack = log(cap * (truth.horizon - 1) * len(models) / .1)
    for calls in range(cap + 1):
        eligible = active[pair_i] & active[pair_j]
        if eligible.any():
            values = np.where(eligible, pair_tv, -1.)
            pair_index = len(values) - 1 - int(np.argmax(values[::-1]))
            diameter = float(values[pair_index])
        else:
            pair_index, diameter = -1, 0.
        stopped = diameter <= epsilon / 4
        if stopped or calls == cap:
            fit_index = max((i for i in range(len(models)) if active[i]), key=lambda i: scores[i])
            return dict(seed=seed, epsilon=epsilon, joint_search_calls=calls + 1,
                        confidence_updates=calls, episodes=(truth.horizon - 1) * calls,
                        stopped=stopped, diameter=diameter,
                        active_models=int(active.sum()),
                        fitted_tv=worst_policy_tv(truth, models[fit_index])[0],
                        truth_covered=bool(active[truth_index]))
        for policy in core_interventions(pair_policies[pair_index], truth, rng):
            trace = truth.sample(policy, rng)
            if trace not in log_cache:
                log_cache[trace] = np.log([m.trajectory_probability(trace) for m in models])
            scores += log_cache[trace]
        active &= scores >= float(scores.max()) - slack


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--epsilons', nargs='+', type=float, default=[.2, .1, .05])
    p.add_argument('--base-batch', type=int, default=10_000_000)
    p.add_argument('--ours-rounds', type=int, default=16)
    p.add_argument('--omle-cap', type=int, default=60_000)
    p.add_argument('--omle-fine-cap', type=int, default=500_000,
                   help='cap used when epsilon <= 0.05')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = fine_candidate_class()
    truth = next(i for i, m in enumerate(raw) if m.weights == tuple(TRUE_WEIGHTS))
    first = max((i for i in range(len(raw)) if i != truth),
                key=lambda i: worst_policy_tv(raw[truth], raw[i])[0])
    order = [first] + [i for i in range(len(raw)) if i != first]
    models = tuple(raw[i] for i in order)
    truth_index = order.index(truth)
    keys = list(combinations(range(len(models)), 2))
    pair_data = [worst_policy_tv(models[i], models[j]) for i, j in keys]
    pairs = (np.asarray([i for i,j in keys]), np.asarray([j for i,j in keys]),
             np.asarray([value for value,policy in pair_data]),
             [policy for value,policy in pair_data])
    ours_rows, ours_finals, omle_finals = [], [], []
    for eps in args.epsilons:
        for seed in range(args.seeds):
            path = run_ours(models, truth_index, eps, seed, args.base_batch, args.ours_rounds)
            ours_rows.extend(path); ours_finals.append(path[-1])
            print(f"ours eps={eps:g} seed={seed} fits={len(path)} stop={path[-1]['stopped']}", flush=True)
            cap = args.omle_fine_cap if eps <= .05 else args.omle_cap
            o = run_omle(models, truth_index, pairs, eps, seed, cap)
            omle_finals.append(o)
            print(f"omle eps={eps:g} seed={seed} joint={o['joint_search_calls']} stop={o['stopped']}", flush=True)
    write_csv(args.output/'ours_rounds.csv', ours_rows)
    write_csv(args.output/'ours_finals.csv', ours_finals)
    write_csv(args.output/'omle_finals.csv', omle_finals)
    summary = []
    for eps in args.epsilons:
        a=[r for r in ours_finals if r['epsilon']==eps]
        b=[r for r in omle_finals if r['epsilon']==eps]
        summary.append(dict(epsilon=eps, seeds=args.seeds,
                            ours_stopped=sum(r['stopped'] for r in a),
                            ours_mle_calls=float(np.median([r['mle_calls'] for r in a])),
                            ours_planning_calls=float(np.median([r['planning_calls'] for r in a])),
                            ours_episodes=float(np.median([r['episodes'] for r in a])),
                            ours_tv=float(np.median([r['sup_tv'] for r in a])),
                            ours_bonus=float(np.median([r['bonus'] for r in a])),
                            ours_certificate_valid=sum(r['certificate_valid'] for r in a),
                            omle_stopped=sum(r['stopped'] for r in b),
                            omle_joint_calls=float(np.median([r['joint_search_calls'] for r in b])),
                            omle_confidence_updates=float(np.median([r['confidence_updates'] for r in b])),
                            omle_episodes=float(np.median([r['episodes'] for r in b])),
                            omle_tv=float(np.median([r['fitted_tv'] for r in b])),
                            omle_truth_covered=sum(r['truth_covered'] for r in b)))
    write_csv(args.output/'summary.csv', summary)


if __name__ == '__main__':
    main()
