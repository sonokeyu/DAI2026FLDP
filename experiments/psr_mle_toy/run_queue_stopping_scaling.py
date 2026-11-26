"""Epsilon sweep in which the fitted-PSR learner runs until bonus stopping.

The class is fixed and realizable across all accuracy settings.  The batch
schedule has the required inverse-square episode scaling up to a mild
logarithmic factor; no fit or planning call cap determines a successful run.
"""
from argparse import ArgumentParser
from math import log2, sqrt
from pathlib import Path

import numpy as np

from queue_mle_class import dense_candidate_class
from queue_pomdp import TRUE_WEIGHTS
from run_multistage_experiment import write_csv
from run_queue_epsilon_certificate import run_case
from run_queue_ours import worst_policy_tv


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--epsilons", type=float, nargs="+", default=(.4, .2, .1, .05))
    parser.add_argument("--base-batch", type=int, default=10_000_000)
    parser.add_argument("--rounds", type=int, default=32)
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
    runs, summary = [], []
    for epsilon in args.epsilons:
        if not 0 < epsilon <= .4:
            parser.error("epsilon must lie in (0, 0.4]")
        log_factor = sqrt(1 + log2(.4 / epsilon))
        effective_base = round(args.base_batch / log_factor)
        paths = [run_case(models, truth_index, epsilon, seed,
                          effective_base, args.rounds)
                 for seed in range(args.seeds)]
        for path in paths:
            for row in path:
                row["batch_log_divisor"] = log_factor
            runs.extend(path)
        finals = [path[-1] for path in paths]
        print(f"epsilon={epsilon:g}, batch={finals[0]['batch']}, "
              f"stopped={sum(row['stopped'] for row in finals)}/{args.seeds}, "
              f"calls={[row['mle_calls'] for row in finals]}", flush=True)
        summary.append(dict(epsilon=epsilon, inverse_epsilon=1/epsilon,
                            candidates=len(models), seeds=args.seeds,
                            batch=finals[0]["batch"],
                            stopped=sum(row["stopped"] for row in finals),
                            all_rounds_certified=sum(all(row["certificate_valid"] for row in path)
                                                     for path in paths),
                            stop_target_passes=sum(row["sup_tv"] <= epsilon/4 for row in finals),
                            median_calls=float(np.median([row["mle_calls"] for row in finals])),
                            call_min=min(row["mle_calls"] for row in finals),
                            call_max=max(row["mle_calls"] for row in finals),
                            median_episodes=float(np.median([row["episodes"] for row in finals])),
                            median_stop_tv=float(np.median([row["sup_tv"] for row in finals])),
                            median_stop_bonus=float(np.median([row["bonus"] for row in finals])),
                            median_first_accurate_round=float(np.median([
                                row["first_accurate_round"] for row in finals]))))
    write_csv(args.output / "stopping_scaling_runs.csv", runs)
    write_csv(args.output / "stopping_scaling_summary.csv", summary)


if __name__ == "__main__":
    main()
