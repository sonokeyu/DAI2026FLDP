"""Compare episode and update costs at common worst-policy TV targets."""
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np

from compare import compare
from routing import RoutingModel
from run_suite import csv


GAPS = (0.1, 0.3, 0.6)
TARGETS = (0.020, 0.010, 0.005)
CHECKPOINT = 8
CONFIGS = {
    "omle": ("reward_free_omle", 2),
    "ours": ("fitted_psr", 8),
    "omle_matched": ("reward_free_omle", 8),
}


def sustained_checkpoints(curves, target):
    """Find the first checkpoint that meets target through the end of a run."""
    is_below = curves <= target + 1e-12
    sustained = np.logical_and.accumulate(is_below[..., ::-1], axis=-1)[..., ::-1]
    return np.where(sustained.any(axis=-1), sustained.argmax(axis=-1), curves.shape[-1])


def quantile_checkpoint(values, probability, checkpoints):
    """Keep runs that miss the target as right censored when taking quantiles."""
    index = max(0, int(np.ceil(probability * len(values))) - 1)
    selected = np.partition(values, index)[index]
    return int(checkpoints[selected]) if selected < len(checkpoints) else None


def run(seeds, budget):
    checkpoints = np.arange(0, budget + 1, CHECKPOINT)
    resample = np.random.default_rng(702).multinomial(
        seeds, np.full(seeds, 1 / seeds), size=1000
    )
    raw, curves, results = [], [], []
    for gap, (condition, (method, interval)) in product(GAPS, CONFIGS.items()):
        print(f"Running {condition} at delta={gap}", flush=True)
        models = [RoutingModel(0.2), RoutingModel(0.2 + gap)]
        distances = np.zeros((2, seeds, len(checkpoints)))
        for truth, seed in product((0, 1), range(seeds)):
            rows = compare(models, truth, method, seed, episodes=budget, batch=CHECKPOINT,
                           omle_update_every=interval)
            assert [row["episodes"] for row in rows] == checkpoints.tolist()
            assert [row["updates"] for row in rows] == (checkpoints // interval).tolist()
            distances[truth, seed] = [row["policy_tv"] for row in rows]
            raw.extend(dict(gap=gap, condition=condition, truth=truth, seed=seed, **row)
                       for row in rows)

        # Each bootstrap draw resamples a seed together with both truth settings.
        # Reuse the draws across methods to preserve pairing.
        seed_means = distances.mean(axis=0)
        mean_tv = seed_means.mean(axis=0)
        bootstrap = resample @ seed_means / seeds
        lo, hi = np.quantile(bootstrap, [0.025, 0.975], axis=0)
        for n, tv, low, high in zip(checkpoints, mean_tv, lo, hi):
            curves.append(dict(gap=gap, condition=condition, episodes=int(n),
                               updates=int(n // interval), mean_tv=float(tv),
                               bootstrap_low=float(low), bootstrap_high=float(high)))

        for target in TARGETS:
            idx = int(sustained_checkpoints(mean_tv, target))
            episodes = int(checkpoints[idx]) if idx < len(checkpoints) else None
            hit = sustained_checkpoints(bootstrap, target)
            results.append(dict(
                gap=gap, target_tv=target, condition=condition,
                reached=episodes is not None, episodes=episodes,
                updates=episodes // interval if episodes is not None else None,
                budget=budget, updates_at_budget=budget // interval,
                mean_tv_at_budget=float(mean_tv[-1]),
                bootstrap_episodes_low=quantile_checkpoint(hit, 0.025, checkpoints),
                bootstrap_episodes_high=quantile_checkpoint(hit, 0.975, checkpoints),
                bootstrap_reached_fraction=float((hit < len(checkpoints)).mean()),
            ))
    return raw, curves, results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--budget", type=int, default=1024)
    args = parser.parse_args()
    if args.seeds <= 0 or args.budget <= 0 or args.budget % CHECKPOINT:
        parser.error("use positive seeds and a positive budget divisible by eight")
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in zip(
        ("target_runs.csv.gz", "target_curves.csv", "target_summary.csv"),
        run(args.seeds, args.budget),
    ):
        csv(args.output / filename, rows)


if __name__ == "__main__":
    main()
