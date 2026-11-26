"""Plot the paired equal-episode accuracy and oracle-call sweep."""
from argparse import ArgumentParser
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (args.input / "equal_budget_summary.csv").open(newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: float(row["inverse_epsilon"]))
    x = [float(row["inverse_epsilon"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.5))
    axes[0].plot(x, [int(row["episodes"]) for row in rows], "o-", color="#555555")
    axes[0].set(ylabel="Episodes per method", title="Identical sample budgets", yscale="log")
    axes[1].plot(x, [int(row["ours_mle_calls"]) for row in rows], "o-", color="#246a9b",
                 label="ours: MLE and planning, each")
    axes[1].plot(x, [int(row["omle_joint_calls"]) for row in rows], "s-", color="#438e5a",
                 label="OMLE: scheduled joint search")
    if any(row["omle_nontrivial_calls"] != row["omle_joint_calls"] for row in rows):
        axes[1].plot(x, [int(row["omle_nontrivial_calls"]) for row in rows], "s--",
                     color="#92ae53", label="OMLE: nontrivial search")
    axes[1].set(ylabel="Oracle calls", title="Prescribed update schedules", yscale="log")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(x, [float(row["ours_median_tv"]) for row in rows], "o-", color="#246a9b",
                 label="ours")
    axes[2].plot(x, [float(row["omle_median_tv"]) for row in rows], "s-", color="#438e5a",
                 label="OMLE")
    axes[2].plot(x, [float(row["initial_tv"]) for row in rows], ":", color="#777777",
                 label="shared initial TV")
    axes[2].plot(x, [float(row["epsilon"])/4 for row in rows], "k--", label="ε/4 target")
    axes[2].set(ylabel="Exact worst-policy TV", title="Accuracy at the same N", ylim=(-.01, .16))
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set(xlabel="1/ε", xticks=x)
        ax.grid(alpha=.22)
    fig.tight_layout()
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "equal_budget.pdf")
    fig.savefig(args.output / "equal_budget.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
