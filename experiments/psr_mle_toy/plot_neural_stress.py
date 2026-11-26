"""Plot neural stress benchmark means and seed-level standard errors."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    args = p.parse_args()
    with (args.input / "neural_runs.csv").open() as f:
        rows = list(csv.DictReader(f))
    values = defaultdict(list)
    for row in rows:
        values[(int(row["episodes"]), row["evaluation"])].append(float(row["nll"]))
    budgets = sorted({key[0] for key in values})
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for split, label, color in [("uniform_test", "Uniform-policy test", "#2563eb"),
                                ("shifted_policy_test", "Action-shifted test", "#dc2626")]:
        means = [np.mean(values[(b, split)]) for b in budgets]
        sem = [np.std(values[(b, split)], ddof=1) / np.sqrt(len(values[(b, split)])) for b in budgets]
        ax.errorbar(budgets, means, yerr=sem, marker="o", capsize=3, lw=2,
                    label=label, color=color)
    ax.set_xscale("log"); ax.set_xlabel("Training episodes")
    ax.set_ylabel("Held-out next-observation NLL (nats)")
    ax.grid(alpha=.25); ax.legend(frameon=False)
    ax.set_title("Neural MLE on a partially observed 256-state task")
    fig.savefig(args.input / "neural_nll.pdf")
    fig.savefig(args.input / "neural_nll.png", dpi=180)


if __name__ == "__main__":
    main()
