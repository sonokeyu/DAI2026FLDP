"""Sweep per-round batch sizes to measure episodes at 2–10 oracle calls."""
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from statistics import median

from multistage_pomdp import candidates
from run_multistage_experiment import write_csv
from run_multistage_ours import run_case


# H=4, r=2, epsilon=.1: one batch setting for each requested call count.
BATCH_BY_TARGET = {
    2: 35_000_000, 3: 20_000_000, 4: 12_000_000,
    5: 9_000_000, 6: 7_000_000, 7: 6_000_000,
    8: 5_000_000, 9: 4_200_000, 10: 4_000_000,
}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = candidates(4)
    truth_index = 2 * (len(models) // 3) - 1
    runs, finals = [], []
    for target, seed in product(BATCH_BY_TARGET, range(args.seeds)):
        batch = BATCH_BY_TARGET[target]
        path = run_case(models, truth_index, 3002 + seed, .1,
                        batch, 16, theory=False)
        for row in path:
            runs.append({**row, "target_calls": target, "batch": batch,
                         "replicate": seed})
        finals.append({**path[-1], "target_calls": target, "batch": batch,
                       "replicate": seed})
        print(f"target={target}, seed={seed}, calls={len(path)}", flush=True)
    write_csv(args.output / "call_budget_runs.csv", runs)
    write_csv(args.output / "call_budget_final.csv", finals)
    summary = []
    for target, batch in BATCH_BY_TARGET.items():
        group = [row for row in finals if row["target_calls"] == target]
        summary.append(dict(target_calls=target, batch=batch,
                            runs=len(group), stopped=sum(row["stopped"] for row in group),
                            calls_min=min(row["mle_calls"] for row in group),
                            calls_median=median(row["mle_calls"] for row in group),
                            calls_max=max(row["mle_calls"] for row in group),
                            episodes_median=median(row["episodes"] for row in group),
                            tv_max=max(row["tv"] for row in group),
                            audit_passed=sum(row["certificate_valid"] for row in group)))
    write_csv(args.output / "call_budget_summary.csv", summary)


if __name__ == "__main__":
    main()
