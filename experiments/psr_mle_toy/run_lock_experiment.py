"""Four-method partially observed combination-lock comparison.

Ours receives its own (larger) episode ceiling because its stopping
certificate can require substantially more data than fitted-model accuracy.
Every method is paired on the same seeds and candidate models.
"""
from argparse import ArgumentParser
import csv
from itertools import combinations, product
from pathlib import Path
from statistics import mean, median

from lock_ours import run_ours
from pocomblock import candidate_models, success_regret, worst_policy_tv
from run_lock_baselines import run_control, run_omle


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--horizons", type=int, nargs="+", default=(4, 6, 8))
    parser.add_argument("--baseline-budget", type=int, default=4096)
    parser.add_argument("--ours-budget", type=int, default=131072)
    parser.add_argument("--ours-batch", type=int, default=4096)
    parser.add_argument("--epsilon", type=float, default=.2)
    parser.add_argument("--family", choices=("pair", "graded"), default="pair")
    args = parser.parse_args()
    if args.baseline_budget % 64 or args.ours_budget % args.ours_batch:
        parser.error("budgets must be divisible by their batch/checkpoint size")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    finals = []
    for horizon in args.horizons:
        models = candidate_models(horizon, count=16 if args.family == "graded" else 8,
                                  family=args.family)
        pairs = {(i, j): worst_policy_tv(models[i], models[j])
                 for i, j in combinations(range(len(models)), 2)}
        for seed in range(args.seeds):
            truth_index = seed % len(models)
            truth = models[truth_index]
            paths = {
                "ours": run_ours(models, truth, seed, horizon, args.ours_budget,
                                  args.epsilon, args.ours_batch),
                "omle": run_omle(models, truth, seed, horizon, args.baseline_budget,
                                 args.epsilon, pairs),
                "uniform": run_control(models, truth, seed, horizon,
                                       args.baseline_budget, "uniform"),
                "count": run_control(models, truth, seed, horizon,
                                     args.baseline_budget, "count"),
            }
            for method, path in paths.items():
                rows.extend(path)
                final = dict(path[-1])
                first_hit = next((row for row in path
                                  if row["tv"] <= args.epsilon / 4), None)
                final.update(first_hit_episodes=first_hit["episodes"] if first_hit else "",
                             first_hit_calls=(first_hit["mle_calls"] +
                                              first_hit["planning_calls"] +
                                              first_hit["joint_calls"]) if first_hit else "",
                             success_regret=success_regret(truth, models[final["fit"]]))
                finals.append(final)
        print(f"completed horizon {horizon}", flush=True)
    write_csv(args.output / "lock_four_method_runs.csv", rows)
    write_csv(args.output / "lock_four_method_final.csv", finals)
    summary = []
    for horizon, method in product(args.horizons,
                                   ("ours", "omle", "uniform", "count")):
        group = [row for row in finals if row["horizon"] == horizon
                 and row["method"] == method]
        hits = [row["first_hit_episodes"] for row in group
                if row["first_hit_episodes"] != ""]
        summary.append(dict(
            horizon=horizon, method=method, runs=len(group),
            episode_ceiling=(args.ours_budget if method == "ours"
                             else args.baseline_budget),
            epsilon=args.epsilon, stop_rate=mean(row["stopped"] for row in group),
            target_rate=mean(row["tv"] <= args.epsilon/4 for row in group),
            mean_tv=mean(row["tv"] for row in group),
            median_episodes=median(row["episodes"] for row in group),
            median_mle_calls=median(row["mle_calls"] for row in group),
            median_planning_calls=median(row["planning_calls"] for row in group),
            median_joint_calls=median(row["joint_calls"] for row in group),
            first_hit_rate=len(hits) / len(group),
            median_first_hit_episodes=median(hits) if hits else "",
            mean_success_regret=mean(row["success_regret"] for row in group),
            certificate_audit_rate=(mean(row["certificate_valid"] for row in group)
                                    if method == "ours" else ""),
        ))
    write_csv(args.output / "lock_four_method_summary.csv", summary)


if __name__ == "__main__":
    main()
