"""Figures for the 64-hidden-state PSR batch sweep."""
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
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.input / "queue_sweep_summary.csv").open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    with (args.input / "queue_sweep_runs.csv").open(newline="") as stream:
        runs = list(csv.DictReader(stream))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = range(len(summary))
    labels = [f"{float(row['batch']) / 1e6:g}M" for row in summary]
    calls = [float(row["calls_median"]) for row in summary]
    episodes = [float(row["episodes_median"]) / 1e6 for row in summary]
    axes[0].bar(x, calls, color="#246a9b")
    axes[1].bar(x, episodes, color="#d37b28")
    axes[0].set(ylabel="MLE and planning calls each", title="Oracle calls")
    axes[1].set(ylabel="Total episodes (millions)", title="Sample cost", yscale="log")
    for axis in axes:
        axis.set_xticks(list(x), labels)
        axis.set_xlabel("Episodes per training/diagnostic batch")
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle("64 hidden states, reachable PSR rank 16, H=4")
    fig.tight_layout()
    fig.savefig(args.output / "queue_resources.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.5, 4.8))
    colors = ("#246a9b", "#438e5a", "#d37b28", "#a65b8b")
    for row, color in zip(summary, colors):
        batch = row["batch"]
        group = [point for point in runs if point["batch"] == batch]
        axis.scatter([float(point["bonus"]) for point in group],
                     [float(point["tv"]) for point in group],
                     s=18, alpha=.5, color=color,
                     label=f"batch {float(batch)/1e6:g}M")
        stopped = [point for point in group if point["stopped"] == "True"]
        axis.scatter([float(point["bonus"]) for point in stopped],
                     [float(point["tv"]) for point in stopped],
                     s=45, marker="x", color=color)
    axis.axvline(3 * .4 / 16, linestyle="--", color="#555555",
                 linewidth=1, label="Stop threshold")
    axis.plot([1e-5, 1], [1e-5, 1], linestyle=":", color="#888888",
              linewidth=1, label="TV = bonus")
    axis.set(xlabel="Fitted bonus", ylabel="Exact worst-policy TV",
             title="64-state stopping audit", xscale="log", yscale="log",
             xlim=(1e-3, 50), ylim=(1e-6, 1))
    axis.grid(alpha=.2)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "queue_certificate.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
