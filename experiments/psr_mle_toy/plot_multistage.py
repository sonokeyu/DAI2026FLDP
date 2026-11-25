"""Publication-readable certificate and resource figures for the grid."""
from argparse import ArgumentParser
import csv
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.input / "multistage_runs.csv").open(newline="") as stream:
        runs = list(csv.DictReader(stream))
    with (args.input / "multistage_summary.csv").open(newline="") as stream:
        summaries = list(csv.DictReader(stream))
    ours = [row for row in runs if row["method"] == "ours"]
    fig, axis = plt.subplots(figsize=(6.5, 4.8))
    for epsilon, color in ((.1, "#246a9b"), (.05, "#d37b28")):
        group = [row for row in ours if float(row["epsilon"]) == epsilon]
        axis.scatter([float(row["bonus"]) for row in group],
                     [float(row["tv"]) for row in group],
                     s=24, alpha=.55, label=f"ε={epsilon}", color=color)
        stopped = [row for row in group if row["stopped"] == "True"]
        axis.scatter([float(row["bonus"]) for row in stopped],
                     [float(row["tv"]) for row in stopped],
                     s=60, marker="x", linewidth=1.5, color=color)
    maximum = max(float(row["bonus"]) for row in ours)
    axis.plot([1e-4, .07], [1e-4, .07], linestyle="--", color="#555555",
              linewidth=1, label="TV = bonus")
    axis.axvline(3 * .1 / 16, color="#246a9b", alpha=.6, linewidth=1)
    axis.axvline(3 * .05 / 16, color="#d37b28", alpha=.6, linewidth=1)
    axis.set_xscale("log")
    axis.set(xlabel="Fitted bonus", ylabel="Exact worst-policy TV",
             title="Multi-stage PSR stopping certificate",
             xlim=(1e-4, maximum * 1.2), ylim=(-.003, .07))
    axis.legend(frameon=False)
    axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / "certificate.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for rank, color in ((1, "#246a9b"), (2, "#438e5a"), (4, "#a65b8b")):
        for epsilon, style in ((.1, "-"), (.05, "--")):
            group = sorted((row for row in summaries
                            if row["method"] == "ours"
                            and int(row["rank"]) == rank
                            and float(row["epsilon"]) == epsilon),
                           key=lambda row: int(row["horizon"]))
            x = [int(row["horizon"]) for row in group]
            axes[0].plot(x, [float(row["median_mle_calls"]) for row in group],
                         linestyle=style, marker="o", color=color,
                         label=f"r={rank}, ε={epsilon}")
            axes[1].plot(x, [float(row["median_episodes"]) for row in group],
                         linestyle=style, marker="o", color=color)
    axes[0].set(xlabel="Horizon H", ylabel="Median MLE and planning calls",
                title="Oracle calls")
    axes[1].set(xlabel="Horizon H", ylabel="Median episodes",
                title="Sample cost", yscale="log")
    for axis in axes:
        axis.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "resources.pdf")
    plt.close(fig)

    methods = ("ours", "omle", "uniform", "count")
    labels = ("ours", "omle", "uniform + MLE", "count + MLE")
    colors = ("#246a9b", "#d37b28", "#438e5a", "#a65b8b")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    x = range(len(methods))
    for epsilon, offset, shade in ((.1, -.18, 1.), (.05, .18, .65)):
        rates = []
        for method in methods:
            group = [row for row in summaries if row["method"] == method
                     and float(row["epsilon"]) == epsilon]
            rates.append(sum(float(row["target_rate"]) for row in group)
                         / len(group))
        axes[0].bar([position + offset for position in x], rates,
                    width=.34, color=colors, alpha=shade,
                    label=f"ε={epsilon}")
    for method, color, position in zip(methods, colors, x):
        group = [row for row in summaries if row["method"] == method]
        calls = median(float(row["median_mle_calls"])
                       + float(row["median_planning_calls"])
                       + float(row["median_joint_calls"]) for row in group)
        episodes = median(float(row["median_episodes"]) for row in group)
        axes[1].bar(position, calls, color=color)
        axes[2].bar(position, episodes, color=color)
    axes[0].set(ylabel="Fraction meeting TV ≤ ε/4", ylim=(0, 1.05),
                title="All-policy accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set(ylabel="Median optimizer calls", yscale="log",
                title="MLE + planning or joint search")
    axes[2].set(ylabel="Median episodes", yscale="log", title="Sample cost")
    for axis in axes:
        axis.set_xticks(list(x), labels, rotation=22, ha="right")
        axis.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / "comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
