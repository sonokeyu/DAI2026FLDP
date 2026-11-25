"""Paired four-method study on multi-stage partially observed PSRs."""
from argparse import ArgumentParser
import csv
from itertools import combinations, product
from pathlib import Path
from statistics import mean, median

import numpy as np

from multistage_pomdp import candidates, worst_policy_tv
from run_multistage_baselines import run_control, run_omle
from run_multistage_ours import run_case
from multistage_psr import prefix_state


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=(3, 4, 5))
    parser.add_argument("--epsilons", type=float, nargs="+", default=(.4, .1))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--baseline-budget", type=int, default=1024)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs, finals = [], []
    for horizon in args.horizons:
        models = candidates(horizon)
        pair_data = {(i, j): worst_policy_tv(models[i], models[j])
                     for i, j in combinations(range(len(models)), 2)}
        block = len(models) // 3
        for rank_index, epsilon, seed in product(range(3), args.epsilons,
                                                  range(args.seeds)):
            truth_index = rank_index * block + block - 1
            states = [prefix_state(models[truth_index], (observation, action))
                      for observation in range(4) for action in range(2)]
            measured_rank = int(np.linalg.matrix_rank(np.array(states)))
            if measured_rank != models[truth_index].rank:
                raise RuntimeError("candidate's measured reachable rank changed")
            case_seed = seed + 1000 * horizon + 100 * rank_index
            paths = {
                "ours": run_case(models, truth_index, case_seed,
                                 epsilon, 1, 8, theory=True),
                "omle": run_omle(models, truth_index, case_seed,
                                 args.baseline_budget, epsilon, pair_data),
                "uniform": run_control(models, truth_index, case_seed,
                                       args.baseline_budget, "uniform"),
                "count": run_control(models, truth_index, case_seed,
                                     args.baseline_budget, "count"),
            }
            for method, path in paths.items():
                for row in path:
                    runs.append({**row, "horizon": horizon,
                                 "rank": models[truth_index].rank,
                                 "epsilon": epsilon, "seed": seed,
                                 "method": method})
                finals.append({**path[-1], "horizon": horizon,
                               "rank": models[truth_index].rank,
                               "epsilon": epsilon, "seed": seed,
                               "method": method})
        print(f"completed H={horizon}", flush=True)
    write_csv(args.output / "multistage_runs.csv", runs)
    write_csv(args.output / "multistage_final.csv", finals)
    summary = []
    for horizon, rank, epsilon, method in product(args.horizons, (1, 2, 4),
                                                   args.epsilons,
                                                   ("ours", "omle", "uniform", "count")):
        group = [row for row in finals if row["horizon"] == horizon
                 and row["rank"] == rank and row["epsilon"] == epsilon
                 and row["method"] == method]
        summary.append(dict(horizon=horizon, rank=rank, epsilon=epsilon,
                            method=method, runs=len(group),
                            target_rate=mean(row["tv"] <= epsilon / 4 for row in group),
                            stop_rate=mean(row["stopped"] for row in group),
                            mean_tv=mean(row["tv"] for row in group),
                            median_episodes=median(row["episodes"] for row in group),
                            median_mle_calls=median(row["mle_calls"] for row in group),
                            median_planning_calls=median(row["planning_calls"] for row in group),
                            median_joint_calls=median(row["joint_calls"]
                                                       if "joint_calls" in row else 0
                                                       for row in group),
                            certificate_audit_rate=(mean(row["certificate_valid"]
                                                         for row in group)
                                                    if method == "ours" else "")))
    write_csv(args.output / "multistage_summary.csv", summary)


if __name__ == "__main__":
    main()
