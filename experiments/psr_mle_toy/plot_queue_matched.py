"""Matched episode-budget TV curve for the four queue POMDP methods."""
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
    with (args.input / "matched_runs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    methods = ("ours", "omle", "uniform", "count")
    labels = {"ours": "ours", "omle": "omle",
              "uniform": "uniform + MLE", "count": "count + MLE"}
    colors = {"ours": "#246a9b", "omle": "#d37b28",
              "uniform": "#438e5a", "count": "#a65b8b"}
    budgets = (64, 128, 192, 256, 320, 384)
    fig, axis = plt.subplots(figsize=(7, 4.6))
    for method in methods:
        values = []
        for budget in budgets:
            points = [row for row in rows if row["method"] == method
                      and int(row["episodes"]) <= budget]
            latest = {}
            for point in points:
                seed = int(point["seed"])
                if seed not in latest or int(point["episodes"]) > int(latest[seed]["episodes"]):
                    latest[seed] = point
            values.append(sum(float(point["tv"]) for point in latest.values())
                          / len(latest))
        axis.plot(budgets, values, marker="o", linewidth=2,
                  label=labels[method], color=colors[method])
    axis.axhline(.1, linestyle="--", linewidth=1, color="#555555",
                 label="Target ε/4 = 0.1")
    axis.set(xlabel="Same cumulative episode budget",
             ylabel="Exact worst-policy TV, averaged over 8 seeds",
             title="64-state POMDP, shared 25-model class",
             xticks=budgets, ylim=(0, None))
    axis.grid(alpha=.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "matched_tv.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
