"""Plot actual stopping calls and samples for ours and OMLE."""
from argparse import ArgumentParser
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    with path.open(newline="") as stream:
        return {float(row["epsilon"]): row for row in csv.DictReader(stream)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--omle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ours = read(args.ours / "stopping_scaling_summary.csv")
    omle = read(args.omle / "omle_stopping_summary.csv")
    epsilons = sorted(ours.keys() & omle.keys(), reverse=True)
    x = [1/epsilon for epsilon in epsilons]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    axes[0].plot(x, [float(ours[eps]["median_calls"]) for eps in epsilons],
                 "o-", color="#246a9b", label="ours: MLE and planning, each")
    axes[0].errorbar(x, [float(omle[eps]["median_calls"]) for eps in epsilons],
                     yerr=[
                         [float(omle[eps]["median_calls"]) - float(omle[eps]["min_calls"])
                          for eps in epsilons],
                         [float(omle[eps]["max_calls"]) - float(omle[eps]["median_calls"])
                          for eps in epsilons]],
                     fmt="s-", capsize=3, color="#438e5a", label="OMLE: joint searches")
    axes[0].set(ylabel="Calls to stopping", title="Actual oracle calls", yscale="log")
    axes[1].plot(x, [float(ours[eps]["median_episodes"]) for eps in epsilons],
                 "o-", color="#246a9b", label="ours")
    axes[1].plot(x, [float(omle[eps]["median_episodes"]) for eps in epsilons],
                 "s-", color="#438e5a", label="OMLE")
    axes[1].set(ylabel="Physical episodes to stopping", title="Sample cost", yscale="log")
    for ax in axes:
        ax.set(xlabel="1/ε", xticks=x)
        ax.grid(alpha=.22)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "stopping_comparison.pdf")
    fig.savefig(args.output / "stopping_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
