"""Measure the episode/update tradeoff across independently chosen batches."""
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np

from compare import compare
from routing import L, RoutingModel, run_algorithm
from run_suite import csv
from run_target_accuracy import GAPS, TARGETS, quantile_checkpoint, sustained_checkpoints


BATCHES = {
    "ours": ("fitted_psr", (8, 32, 128, 512, 2048)),
    "omle": ("reward_free_omle", (2, 8, 32, 128, 512, 2048)),
    "uniform": ("random_test", (8, 32, 128, 512, 2048)),
    "count": ("count_test", (8, 32, 128, 512, 2048)),
}


def run_arm(models, truth, method, seed, budget, batch):
    if method == "fitted_psr":
        # Each of four training blocks receives m, and nested diagnostics
        # receive 4m, scaling both parts of the original eight-episode round.
        trace = run_algorithm(
            models, truth, 1.0, seed, fixed_budget=(budget, batch),
            fixed_training=batch // (2 * L),
        )
        return [(row["episodes"], row["policy_tv"], row["targeted"]) for row in trace]
    trace = compare(models, truth, method, seed, episodes=budget, batch=batch,
                    omle_update_every=batch)
    return [(row["episodes"], row["policy_tv"], row["targeted"]) for row in trace]


def frontier(costs):
    return [
        row for row in costs if not any(
            other["episodes"] <= row["episodes"] and other["updates"] <= row["updates"]
            and (other["episodes"] < row["episodes"] or other["updates"] < row["updates"])
            for other in costs
        )
    ]


def sweep(seeds, budget, gaps):
    resample = np.random.default_rng(702).multinomial(seeds, np.full(seeds, 1 / seeds), size=1000)
    raw, curves, results = [], [], []
    for gap, (name, (method, batches)) in product(gaps, BATCHES.items()):
        models = [RoutingModel(.2), RoutingModel(.2 + gap)]
        for batch in batches:
            print(f"delta={gap:g} {name} batch={batch}", flush=True)
            epochs = np.arange(0, budget + 1, batch)
            distances = np.empty((2, seeds, len(epochs)))
            for truth, seed in product((0, 1), range(seeds)):
                trace = run_arm(models, truth, method, seed, budget, batch)
                assert [n for n, _, _ in trace] == epochs.tolist()
                distances[truth, seed] = [tv for _, tv, _ in trace]
                # With two candidate models, off-policy TV is either zero
                # or gap/2. Record one bit per refit, including the empty-data fit.
                raw.append(dict(gap=gap, method=name, batch=batch, truth=truth, seed=seed,
                                fit_error="".join("1" if tv else "0" for _, tv, _ in trace),
                                targeted_at_budget=trace[-1][2]))
            seed_means = distances.mean(axis=0)
            mean_tv = np.mean(seed_means, axis=0)
            bootstrap = resample @ seed_means / seeds
            lo, hi = np.quantile(bootstrap, [0.025, 0.975], axis=0)
            for i, (n, mean, low, high) in enumerate(zip(epochs, mean_tv, lo, hi)):
                curves.append(dict(gap=gap, method=name, batch=batch, episodes=int(n), updates=i,
                                   mean_tv=float(mean), bootstrap_low=float(low),
                                   bootstrap_high=float(high)))
            for target in TARGETS:
                hit = int(sustained_checkpoints(mean_tv, target))
                boot_hits = sustained_checkpoints(bootstrap, target)
                did_hit = hit < len(epochs)
                results.append(dict(
                    gap=gap, target_tv=target, method=name, batch=batch, reached=did_hit,
                    episodes=int(epochs[hit]) if did_hit else None, updates=hit if did_hit else None,
                    budget=budget, updates_at_budget=budget // batch,
                    mean_tv_at_budget=float(mean_tv[-1]),
                    bootstrap_episodes_low=quantile_checkpoint(boot_hits, .025, epochs),
                    bootstrap_episodes_high=quantile_checkpoint(boot_hits, .975, epochs),
                    bootstrap_reached_fraction=float((boot_hits < len(epochs)).mean()),
                ))
    for gap, target in product(gaps, TARGETS):
        comparable = [r for r in results if r["reached"] and r["gap"] == gap
                      and r["target_tv"] == target]
        pooled = frontier(comparable)
        for name in BATCHES:
            own = frontier([r for r in comparable if r["method"] == name])
            for row in (r for r in results if r["gap"] == gap and r["target_tv"] == target
                        and r["method"] == name):
                row["method_frontier"] = row in own
                row["pooled_frontier"] = row in pooled
    return raw, curves, results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--gap", type=float, choices=GAPS, action="append")
    args = parser.parse_args()
    largest_batch = max(batch for _, batches in BATCHES.values() for batch in batches)
    if args.seeds < 1 or args.budget < largest_batch or args.budget % largest_batch:
        parser.error(f"positive seeds and a positive budget divisible by {largest_batch} are required")
    gaps = tuple(sorted(set(args.gap))) if args.gap else GAPS
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in zip(
        ("batch_runs.csv.gz", "batch_curves.csv", "batch_summary.csv"),
        sweep(args.seeds, args.budget, gaps),
    ):
        csv(args.output / filename, rows)


if __name__ == "__main__":
    main()
