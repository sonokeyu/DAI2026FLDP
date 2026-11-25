"""Secondary multi-round check with batches scaled as inverse epsilon squared."""
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from statistics import median

from multistage_pomdp import candidates
from run_multistage_experiment import write_csv
from run_multistage_ours import run_case


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--base-batch", type=int, default=10_000_000)
    parser.add_argument("--rounds", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs, finals = [], []
    for horizon in (3, 4, 5):
        models = candidates(horizon)
        block = len(models) // 3
        for rank_index, epsilon, seed in product(range(3), (.1, .05),
                                                  range(args.seeds)):
            truth_index = (rank_index + 1) * block - 1
            batch = round(args.base_batch * (.1 / epsilon)**2)
            path = run_case(models, truth_index,
                            20_000 + 100 * horizon + models[truth_index].rank + seed,
                            epsilon, batch, args.rounds, theory=False)
            for row in path:
                runs.append({**row, "batch": batch, "seed": seed})
            finals.append({**path[-1], "batch": batch, "seed": seed})
        print(f"completed multi-round H={horizon}", flush=True)
    write_csv(args.output / "call_scaling_runs.csv", runs)
    write_csv(args.output / "call_scaling_final.csv", finals)
    summary = []
    for horizon, rank, epsilon in product((3, 4, 5), (1, 2, 4), (.1, .05)):
        group = [row for row in finals if row["horizon"] == horizon
                 and row["rank"] == rank and row["epsilon"] == epsilon]
        summary.append(dict(horizon=horizon, rank=rank, epsilon=epsilon,
                            seeds=len(group), batch=group[0]["batch"],
                            stopped=sum(row["stopped"] for row in group),
                            audit_passed=sum(row["certificate_valid"] for row in group),
                            median_mle_calls=median(row["mle_calls"] for row in group),
                            median_planning_calls=median(row["planning_calls"]
                                                         for row in group),
                            median_episodes=median(row["episodes"] for row in group),
                            max_tv=max(row["tv"] for row in group)))
    write_csv(args.output / "call_scaling_summary.csv", summary)


if __name__ == "__main__":
    main()
