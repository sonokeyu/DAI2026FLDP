"""Paired batch-size study of the 64-hidden-state POMDP."""
from argparse import ArgumentParser
from pathlib import Path
from statistics import median

from queue_pomdp import QueueModel, TRUE_WEIGHTS
from run_multistage_experiment import write_csv
from run_queue_ours import run_case


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--batches", type=int, nargs="+",
                        default=(10_000_000, 30_000_000,
                                 100_000_000, 1_000_000_000))
    parser.add_argument("--epsilon", type=float, default=.4)
    parser.add_argument("--rounds", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    truth = QueueModel(TRUE_WEIGHTS)
    runs, finals = [], []
    for batch in args.batches:
        for seed in range(args.seeds):
            path = run_case(truth, seed, args.epsilon, batch, args.rounds)
            runs.extend(path)
            finals.append(path[-1])
        print(f"completed batch={batch}", flush=True)
    write_csv(args.output / "queue_sweep_runs.csv", runs)
    write_csv(args.output / "queue_sweep_final.csv", finals)
    summary = []
    for batch in args.batches:
        group = [row for row in finals if row["batch"] == batch]
        summary.append(dict(batch=batch, runs=len(group),
                            stopped=sum(row["stopped"] for row in group),
                            audit_passed=sum(row["audit_valid"] for row in group),
                            calls_min=min(row["mle_calls"] for row in group),
                            calls_median=median(row["mle_calls"] for row in group),
                            calls_max=max(row["mle_calls"] for row in group),
                            episodes_median=median(row["episodes"] for row in group),
                            tv_min=min(row["tv"] for row in group),
                            tv_median=median(row["tv"] for row in group),
                            tv_max=max(row["tv"] for row in group)))
    write_csv(args.output / "queue_sweep_summary.csv", summary)


if __name__ == "__main__":
    main()
