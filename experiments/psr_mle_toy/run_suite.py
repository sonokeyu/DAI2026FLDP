"""Run the routing experiments and the observable rank sweep."""
from argparse import ArgumentParser
from collections import defaultdict
from csv import DictWriter
from gzip import compress
from io import StringIO
from itertools import product
from json import dumps
from pathlib import Path
from statistics import mean, median

import numpy as np

from compare import compare
from rank_sweep import run_rank_case
from routing import RoutingModel, run_algorithm


def csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    output = StringIO(newline="")
    writer = DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = output.getvalue()
    if path.suffix == ".gz":
        path.write_bytes(compress(content.encode("utf-8"), mtime=0))
    else:
        path.write_text(content)


def comparison(seeds):
    raw = []
    groups = defaultdict(list)
    # Preserve the methods in the original 256-episode suite.
    methods = ("bonus_test", "random_test", "bonus_no_test", "fitted_psr", "reward_free_omle")
    for gap, truth, method, seed in product((0.1, 0.3, 0.6), (0, 1), methods, range(seeds)):
        models = [RoutingModel(0.2), RoutingModel(0.2 + gap)]
        for row in compare(models, truth, method, seed):
            key = dict(gap=gap, truth=truth, method=method)
            raw.append(dict(**key, seed=seed, **row))
            groups[gap, method, row["episodes"]].append(row)
    summary = []
    rng = np.random.default_rng(702)
    # Each cluster contains both truth settings and all methods use the
    # same resampled seed indices.
    selected = rng.integers(seeds, size=(1000, seeds))
    for (gap, method, episodes), rows in groups.items():
        tv = np.array([row["policy_tv"] for row in rows]).reshape(2, seeds)
        draws = tv.mean(axis=0)[selected].mean(axis=1)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        summary.append(dict(gap=gap, method=method, episodes=episodes,
                            mean_tv=float(tv.mean()), bootstrap_low=float(lo),
                            bootstrap_high=float(hi),
                            mean_targeted=mean(row["targeted"] for row in rows),
                            updates=rows[0]["updates"]))
    return raw, summary


def routing(seeds):
    rounds, final = [], []
    cases = product(
        ("deterministic", "noisy"), (0, 1), (1., 0.5, 0.25),
        ("resized_nested", "resized_direct", "appendix_nested"), range(seeds),
    )
    for env, truth, eps, method, seed in cases:
        models = ([RoutingModel(0), RoutingModel(1)] if env == "deterministic"
                  else [RoutingModel(0.2), RoutingModel(0.3)])
        trace = run_algorithm(models, truth, eps, seed,
                              condition="direct" if method == "resized_direct" else "nested",
                              theory=method == "appendix_nested", scale=8)
        key = dict(environment=env, truth=truth, epsilon=eps, method=method, seed=seed)
        rounds.extend(dict(**key, **row) for row in trace)
        final.append(dict(**key, **trace[-1]))
    return rounds, final


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100, help="seeds per equal-budget condition")
    parser.add_argument("--algorithm-seeds", type=int, default=20)
    args = parser.parse_args()
    if min(args.seeds, args.algorithm_seeds) < 1:
        parser.error("seed counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    comparisons, comparison_summary = comparison(args.seeds)
    rounds, routing_end = routing(args.algorithm_seeds)
    ranks = [run_rank_case(h, r, eps, seed) for h, r, eps, seed in product(
        (4, 6, 8), (1, 2, 4), (1., 0.5, 0.25), range(args.algorithm_seeds))]
    for name, rows in (
        ("comparison_runs", comparisons), ("comparison_summary", comparison_summary),
        ("routing_rounds", rounds), ("routing_final", routing_end), ("rank_runs", ranks),
    ):
        suffix = ".csv.gz" if name in ("comparison_runs", "routing_rounds") else ".csv"
        csv(args.output / (name + suffix), rows)

    summary = {}
    for method in ("resized_nested", "resized_direct", "appendix_nested"):
        runs = [row for row in routing_end if row["method"] == method]
        summary[method] = dict(
            runs=len(runs), stops=sum(row["stopped"] for row in runs),
            false_stops=sum(row["stopped"] and row["policy_tv"] > row["epsilon"] / 4
                            for row in runs),
            uncalibrated_rounds=sum(not row["calibrated"] for row in rounds if row["method"] == method),
            median_calls=median(row["mle_calls"] for row in runs),
            median_episodes=median(row["episodes"] for row in runs),
        )
    summary["rank"] = dict(
        runs=len(ranks), stops=sum(row["stopped"] for row in ranks),
        false_stops=sum(row["stopped"] and row["policy_tv"] > row["epsilon"] / 4
                        for row in ranks),
    )
    document = dumps(summary, indent=2) + "\n"
    (args.output / "suite_summary.json").write_text(document)
    print(document, end="")


if __name__ == "__main__":
    main()
