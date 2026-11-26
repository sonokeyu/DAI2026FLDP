"""Continuous-model stress check of the epsilon-dependent stopping schedule.

This numerical MLE experiment is an empirical audit, since its effective
class size is not a proved covering number for the continuous class.
"""
from argparse import ArgumentParser
from math import log2, sqrt
from pathlib import Path

import numpy as np

from queue_pomdp import QueueModel, TRUE_WEIGHTS
from run_multistage_experiment import write_csv
from run_queue_ours import run_case


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--epsilons", type=float, nargs="+", default=(.4, .2, .1, .05))
    parser.add_argument("--base-batch", type=int, default=10_000_000)
    parser.add_argument("--rounds", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    truth = QueueModel(TRUE_WEIGHTS)
    runs, summary = [], []
    for epsilon in args.epsilons:
        if not 0 < epsilon <= .4:
            parser.error("epsilon must lie in (0, 0.4]")
        batch = round(args.base_batch * (.4 / epsilon)**2
                      / sqrt(1 + log2(.4 / epsilon)))
        paths = [run_case(truth, seed, epsilon, batch, args.rounds)
                 for seed in range(args.seeds)]
        runs.extend(row for path in paths for row in path)
        finals = [path[-1] for path in paths]
        print(f"continuous epsilon={epsilon:g}, batch={batch}, "
              f"stopped={sum(row['stopped'] for row in finals)}/{args.seeds}, "
              f"calls={[row['mle_calls'] for row in finals]}", flush=True)
        summary.append(dict(epsilon=epsilon, seeds=args.seeds,
                            batch=batch, stopped=sum(row["stopped"] for row in finals),
                            audit_passes=sum(all(row["audit_valid"] for row in path)
                                             for path in paths),
                            target_passes=sum(row["tv"] <= epsilon/4 for row in finals),
                            median_calls=float(np.median([row["mle_calls"] for row in finals])),
                            median_episodes=float(np.median([row["episodes"] for row in finals])),
                            median_tv=float(np.median([row["tv"] for row in finals])),
                            median_bonus=float(np.median([row["bonus"] for row in finals]))))
    write_csv(args.output / "continuous_stopping_runs.csv", runs)
    write_csv(args.output / "continuous_stopping_summary.csv", summary)


if __name__ == "__main__":
    main()
