"""Show actual bonus stopping, oracle calls, and episodes over epsilon."""
from argparse import ArgumentParser
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--continuous-input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (args.input / "stopping_scaling_summary.csv").open(newline="") as stream:
        summary = sorted(csv.DictReader(stream), key=lambda row: float(row["inverse_epsilon"]))
    with (args.input / "stopping_scaling_runs.csv").open(newline="") as stream:
        runs = list(csv.DictReader(stream))
    x = [float(row["inverse_epsilon"]) for row in summary]
    calls = [float(row["median_calls"]) for row in summary]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    axes[0].errorbar(x, calls,
                     yerr=[[value - int(row["call_min"]) for value, row in zip(calls, summary)],
                           [int(row["call_max"]) - value for value, row in zip(calls, summary)]],
                     fmt="o-", capsize=3, color="#246a9b", label="finite 49-model class")
    axes[0].set(title="Actual certificate stopping", ylabel="MLE calls = planning calls")
    axes[1].plot(x, [float(row["median_episodes"]) for row in summary], "o-",
                 color="#a76331", label="finite 49-model class")
    axes[1].set(title="Episodes used to stop", ylabel="Median physical episodes",
                yscale="log")
    if args.continuous_input:
        with (args.continuous_input / "continuous_stopping_summary.csv").open(newline="") as stream:
            continuous = sorted(csv.DictReader(stream), key=lambda row: 1/float(row["epsilon"]))
        cx = [1/float(row["epsilon"]) for row in continuous]
        axes[0].plot(cx, [float(row["median_calls"]) for row in continuous], "s--",
                     color="#438e5a", label="continuous MLE audit")
        axes[1].plot(cx, [float(row["median_episodes"]) for row in continuous], "s--",
                     color="#438e5a", label="continuous MLE audit")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    axes[0].set_ylim(bottom=0)
    for epsilon in [float(row["epsilon"]) for row in summary]:
        path = sorted((row for row in runs if int(row["seed"]) == 0
                       and float(row["epsilon"]) == epsilon),
                      key=lambda row: int(row["round"]))
        axes[2].plot([int(row["round"]) for row in path],
                     [float(row["bonus"])/float(row["stop_threshold"]) for row in path],
                     marker="o", markersize=2.5, label=f"ε={epsilon:g}")
    axes[2].axhline(1, color="#555555", linestyle="--", linewidth=1)
    axes[2].set(title="Bonus crosses stop threshold", xlabel="MLE and planning calls",
                ylabel="Bonus / (3ε/16)", yscale="log")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes[:2]:
        ax.set(xlabel="1/ε", xticks=x)
    for ax in axes:
        ax.grid(alpha=.22)
    fig.tight_layout()
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "stopping_scaling.pdf")
    fig.savefig(args.output / "stopping_scaling.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
