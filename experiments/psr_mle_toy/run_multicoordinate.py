"""Multi-coordinate all-policy model-estimation experiment.

An episode chooses one of d observable test branches and receives a Bernoulli
outcome. Each coordinate's parameter is either 0.2 or 0.2+epsilon. For this
finite rank-d family, worst-policy trajectory TV is exactly epsilon/2 if any
coordinate is fitted to the wrong candidate, and zero otherwise.

The `ours_batch` arm uses ordinary MLE and one-model planning: it picks the
coordinate with the largest Hoeffding bonus and samples a precomputed batch
that makes its confidence radius at most epsilon/4. OMLE uses a uniform
intervention plus a targeted intervention each round and maintains the
coordinatewise likelihood confidence product (Bonferroni allocation). This
Traditional uniform and least-count coordinate exploration collect eight
episodes per ordinary-MLE refit and stop at the same coordinatewise Hoeffding
threshold. This is a controlled finite-class audit of call scaling, not the
paper's full sequential PSR benchmark.
"""
from argparse import ArgumentParser
import csv
from itertools import product
from math import ceil, log, sqrt
from pathlib import Path
import random
from statistics import mean, median


DIMENSIONS = (1, 2, 4, 8)
ACCURACIES = (.2, .1)
SEEDS = 100
DELTA = .1
MLE_BATCH = 8


def truth_vector(dimension, seed):
    rng = random.Random(seed + 1907 * dimension)
    return [rng.randrange(2) for _ in range(dimension)]


def fit_bits(hits, misses, eps):
    p0, p1 = .2, .2 + eps
    out = []
    for hit, miss in zip(hits, misses):
        score = hit * log(p1 / p0) + miss * log((1-p1)/(1-p0))
        out.append(int(score > 0))
    return out


def run_ours(dimension, eps, seed, cap):
    truth = truth_vector(dimension, seed)
    rng = random.Random(seed + 32771 * dimension + int(eps*10000))
    n_required = ceil(8 * log(2 * dimension / DELTA) / eps**2)
    hits, misses = [0]*dimension, [0]*dimension
    episodes = mle_calls = planning_calls = fits = 0
    while True:
        bonuses = [sqrt(log(2 * dimension / DELTA) / (2 * max(1, h+m)))
                   for h, m in zip(hits, misses)]
        candidates = [i for i, (h, m) in enumerate(zip(hits, misses)) if h+m < n_required]
        mle_calls += 1
        planning_calls += 1
        fitted = fit_bits(hits, misses, eps)
        tv = .5 * eps if fitted != truth else 0.0
        stopped = not candidates
        if stopped or episodes >= cap:
            return dict(method="ours_batch", dimension=dimension, epsilon=eps,
                        target_tv=eps/4, seed=seed, episodes=episodes,
                        mle_calls=mle_calls, planning_calls=planning_calls,
                        joint_search_calls=0, fitted_errors=sum(a != b for a,b in zip(fitted,truth)),
                        exact_tv=tv, bonus_stop=stopped, bonus=max(bonuses, default=0),
                        n_required_per_coordinate=n_required, episodes_cap=cap)
        arm = max(candidates, key=lambda i: bonuses[i])
        batch = min(n_required - hits[arm] - misses[arm], cap-episodes)
        p = .2 + eps * truth[arm]
        sample_hits = sum(rng.random() < p for _ in range(batch))
        hits[arm] += sample_hits
        misses[arm] += batch - sample_hits
        episodes += batch
        fits += 1


def run_omle(dimension, eps, seed, cap):
    truth = truth_vector(dimension, seed)
    rng = random.Random(seed + 65537 * dimension + int(eps*10000))
    beta = log(2 * dimension * max(2, cap) / DELTA)
    hits, misses = [0]*dimension, [0]*dimension
    episodes = updates = pair_scores = 0
    while True:
        # Coordinatewise confidence product from likelihood-ratio tests.
        ambiguous = []
        p0, p1 = .2, .2 + eps
        for i, (hit, miss) in enumerate(zip(hits, misses)):
            llr = hit * log(p1/p0) + miss * log((1-p1)/(1-p0))
            if abs(llr) <= beta:
                ambiguous.append(i)
        if not ambiguous or episodes + 2 > cap:
            fitted = fit_bits(hits, misses, eps)
            return dict(method="omle", dimension=dimension, epsilon=eps,
                        target_tv=eps/4, seed=seed, episodes=episodes,
                        mle_calls=0, planning_calls=0, joint_search_calls=updates,
                        pair_scores=pair_scores,
                        fitted_errors=sum(a != b for a,b in zip(fitted,truth)),
                        exact_tv=.5*eps if fitted != truth else 0.0,
                        diameter_stop=not ambiguous, episodes_cap=cap,
                        unresolved_coordinates=len(ambiguous))
        # The model-pair search picks a maximally uncertain coordinate. A
        # companion uniform intervention is retained as in reward-free OMLE.
        target = min(ambiguous, key=lambda i: hits[i]+misses[i])
        pair_scores += (2**len(ambiguous)) * (2**len(ambiguous)-1) // 2
        for arm in (rng.randrange(dimension), target):
            p = .2 + eps * truth[arm]
            y = rng.random() < p
            hits[arm] += int(y)
            misses[arm] += int(not y)
        episodes += 2
        updates += 1


def run_traditional(dimension, eps, seed, cap, method):
    """Uniform or least-count coordinate exploration with batched ordinary MLE."""
    truth = truth_vector(dimension, seed)
    rng = random.Random(seed + (98317 if method == "uniform" else 131071)
                        * dimension + int(eps*10000))
    n_required = ceil(8 * log(2 * dimension / DELTA) / eps**2)
    hits, misses = [0]*dimension, [0]*dimension
    counts = [0]*dimension
    episodes = mle_calls = 0
    while episodes < cap:
        for _ in range(min(MLE_BATCH, cap - episodes)):
            if method == "uniform":
                arm = rng.randrange(dimension)
            else:
                least = min(counts)
                ties = [i for i, count in enumerate(counts) if count == least]
                arm = rng.choice(ties)
            p = .2 + eps * truth[arm]
            y = rng.random() < p
            counts[arm] += 1
            hits[arm] += int(y)
            misses[arm] += int(not y)
            episodes += 1
        fitted = fit_bits(hits, misses, eps)
        mle_calls += 1
        stopped = min(counts) >= n_required
        if stopped:
            return dict(method=method, dimension=dimension, epsilon=eps,
                        target_tv=eps/4, seed=seed, episodes=episodes,
                        mle_calls=mle_calls, planning_calls=0,
                        joint_search_calls=0, pair_scores=0,
                        fitted_errors=sum(a != b for a,b in zip(fitted,truth)),
                        exact_tv=.5*eps if fitted != truth else 0.0,
                        bonus_stop=True, n_required_per_coordinate=n_required,
                        episodes_cap=cap)
    fitted = fit_bits(hits, misses, eps)
    return dict(method=method, dimension=dimension, epsilon=eps,
                target_tv=eps/4, seed=seed, episodes=episodes,
                mle_calls=mle_calls, planning_calls=0,
                joint_search_calls=0, pair_scores=0,
                fitted_errors=sum(a != b for a,b in zip(fitted,truth)),
                exact_tv=.5*eps if fitted != truth else 0.0,
                bonus_stop=False, n_required_per_coordinate=n_required,
                episodes_cap=cap)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--cap", type=int, default=131072)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for d, eps, seed in product(DIMENSIONS, ACCURACIES, range(args.seeds)):
        rows.append(run_ours(d, eps, seed, args.cap))
        rows.append(run_omle(d, eps, seed, args.cap))
        rows.append(run_traditional(d, eps, seed, args.cap, "uniform"))
        rows.append(run_traditional(d, eps, seed, args.cap, "count"))
    with (args.output / "multicoordinate_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for method, d, eps in product(("ours_batch", "omle", "uniform", "count"), DIMENSIONS, ACCURACIES):
        group = [r for r in rows if r["method"] == method and r["dimension"] == d and r["epsilon"] == eps]
        summary.append(dict(method=method, dimension=d, epsilon=eps, target_tv=eps/4,
                            runs=len(group), stop_rate=mean(r.get("bonus_stop",r.get("diameter_stop")) for r in group),
                            success_rate=mean(r["exact_tv"] <= eps/4 for r in group),
                            median_episodes=median(r["episodes"] for r in group),
                            median_mle_calls=median(r["mle_calls"] for r in group),
                            median_planning_calls=median(r["planning_calls"] for r in group),
                            median_joint_calls=median(r["joint_search_calls"] for r in group),
                            median_pair_scores=median(r.get("pair_scores",0) for r in group), cap=args.cap))
    with (args.output / "multicoordinate_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(f"completed {len(rows)} runs across dimension and accuracy sweeps")


if __name__ == "__main__":
    main()
