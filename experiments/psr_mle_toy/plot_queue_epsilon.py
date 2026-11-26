"""Plot ours certificate paths and OMLE calls over the epsilon sweep."""
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
    with (args.input / "epsilon_runs.csv").open(newline="") as stream:
        ours = list(csv.DictReader(stream))
    with (args.input / "omle_epsilon_final.csv").open(newline="") as stream:
        omle = list(csv.DictReader(stream))
    epsilons = sorted({float(row["epsilon"]) for row in ours}, reverse=True)
    fig, axes = plt.subplots(1, len(epsilons), figsize=(13, 3.8), sharey=True)
    for axis, epsilon in zip(axes, epsilons):
        group = [row for row in ours if float(row["epsilon"]) == epsilon]
        for seed in sorted({int(row["seed"]) for row in group}):
            path = sorted((row for row in group if int(row["seed"]) == seed),
                          key=lambda row: int(row["round"]))
            x = [int(row["round"]) for row in path]
            axis.plot(x, [max(float(row["sup_tv"]), 1e-7) for row in path],
                      color="#246a9b", alpha=.65, linewidth=1.4)
            axis.plot(x, [float(row["bonus"]) for row in path],
                      color="#d37b28", alpha=.65, linewidth=1.4)
        axis.axhline(epsilon / 4, color="#246a9b", linestyle="--", linewidth=1)
        axis.axhline(3 * epsilon / 16, color="#d37b28", linestyle=":", linewidth=1)
        axis.set(title=f"ε={epsilon}", xlabel="MLE and planning calls",
                 yscale="log", xticks=range(1, max(int(row["round"]) for row in group)+1))
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Exact sup TV and fitted bonus")
    axes[0].plot([], [], color="#246a9b", label="True sup TV")
    axes[0].plot([], [], color="#d37b28", label="Fitted bonus")
    axes[0].plot([], [], color="#555555", linestyle="--", label="ε/4 target")
    axes[0].plot([], [], color="#555555", linestyle=":", label="3ε/16 stop threshold")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Ours on the shared 64-state finite model class")
    fig.tight_layout()
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "epsilon_certificate.pdf")
    plt.close(fig)

    eps = [float(row["epsilon"]) for row in omle]
    calls = [int(row["calls"]) for row in omle]
    stopped = [row["stopped"] == "True" for row in omle]
    fig, axis = plt.subplots(figsize=(5.5, 4))
    axis.plot([1/value for value in eps], calls, marker="o", color="#438e5a")
    for x, y, done in zip([1/value for value in eps], calls, stopped):
        if not done:
            axis.scatter([x], [y], marker="x", s=70, color="#d62728", zorder=3)
    axis.set(xlabel="1/ε", ylabel="OMLE joint searches",
             title="OMLE calls versus target accuracy")
    axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / "omle_epsilon_calls.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
