"""Compare four stopping methods on the full-PSR four-branch environment."""
import csv
from itertools import product, combinations
from math import ceil, log
from pathlib import Path
import random
from statistics import mean, median
from argparse import ArgumentParser

import numpy as np

from routing import CUES, MODES, PREFIXES, ROUTES, Design, all_policy_tv, law, mix
from run_full_psr_multicoordinate import ACCURACIES, candidate_class, continuous_truth


DELTA = .1
OMLE_CAP = 131072


def fit_index(models, counts, mask=None):
    logs = np.log(np.stack([m.observation_weights() for m in models]))
    scores = logs @ counts
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    return int(scores.argmax()), scores


def uniform_design():
    return Design((.5, .5), (.5, .5, .5, .5))


def coordinate_design(coordinates):
    routes = []
    modes = []
    for cue in CUES:
        chosen_route = coordinates[CUES.index(cue)]
        routes.append(float(chosen_route == "R"))
        for route in ROUTES:
            modes.append(float(route == chosen_route))
    return Design(tuple(routes), tuple(modes))


def run_mle_baseline(models, truth, seed, eps, method):
    rng = np.random.default_rng(seed + (8101 if method == "uniform" else 9239))
    counts = np.zeros(16, dtype=np.int64)
    branch_counts = np.zeros(4, dtype=np.int64)
    n_required = ceil(8 * log(2 * 4 / DELTA) / eps**2)
    episodes = 0
    uniform = uniform_design()
    while min(branch_counts) < n_required:
        if method == "uniform":
            design = uniform
            batch = min(4096, max(8, (n_required - min(branch_counts)) * 4))
        else:
            route_for_cue = []
            for cue_i in range(2):
                candidates = [cue_i * 2, cue_i * 2 + 1]
                route_for_cue.append(min(candidates, key=lambda i: (branch_counts[i], i)) % 2)
            design = coordinate_design(tuple(ROUTES[i] for i in route_for_cue))
            batch = min(4096, max(8, (n_required - min(branch_counts)) * 2))
        probabilities = law(truth, mix(design)).ravel()
        observed = rng.multinomial(batch, probabilities / probabilities.sum())
        counts += observed
        episodes += batch
        for i, episode in enumerate(__import__("routing").EPISODES):
            if episode.mode == "a":
                branch_counts[CUES.index(episode.cue) * 2 + ROUTES.index(episode.route)] += observed[i]
    fitted, _ = fit_index(models, counts)
    return dict(method=method, episodes=episodes, mle_calls=1,
                planning_calls=0, joint_search_calls=0,
                stopped=True, worst_policy_tv=all_policy_tv(truth, models[fitted]),
                fitted_index=fitted)


def pair_disagreement(models, active):
    pairs = combinations([i for i, yes in enumerate(active) if yes], 2)
    best_pair, diameter = None, -1.0
    for i, j in pairs:
        distance = all_policy_tv(models[i], models[j])
        if distance > diameter:
            best_pair, diameter = (i, j), distance
    if best_pair is None:
        return Design(), 0.0
    first, second = (models[i] for i in best_pair)
    routes, modes = [], [0.0] * len(PREFIXES)
    for cue in CUES:
        route, mode = max(product(ROUTES, MODES),
                          key=lambda rm: abs(first.p(cue, *rm) - second.p(cue, *rm)))
        routes.append(float(route == "R"))
        modes[PREFIXES.index((cue, route))] = float(mode == "a")
    return Design(tuple(routes), tuple(modes)), diameter


def run_omle(models, truth, seed, eps):
    rng = np.random.default_rng(seed + 12113)
    counts = np.zeros(16, dtype=np.int64)
    log_models = np.log(np.stack([m.observation_weights() for m in models]))
    slack = log(OMLE_CAP * len(models) / DELTA)
    episodes = calls = pair_scores = 0
    while episodes + 2 <= OMLE_CAP:
        scores = log_models @ counts
        active = scores >= scores.max() - slack
        policy, diameter = pair_disagreement(models, active)
        calls += 1
        pair_scores += int(active.sum()) * (int(active.sum()) - 1) // 2
        if diameter <= eps / 4:
            fitted = int(np.argmax(np.where(active, scores, -np.inf)))
            return dict(method="omle", episodes=episodes, mle_calls=0,
                        planning_calls=0, joint_search_calls=calls,
                        pair_scores=pair_scores, stopped=True,
                        worst_policy_tv=all_policy_tv(truth, models[fitted]),
                        fitted_index=fitted)
        for design in (uniform_design(), policy):
            probs = law(truth, mix(design)).ravel()
            outcome = int(rng.choice(len(probs), p=probs / probs.sum()))
            counts[outcome] += 1
        episodes += 2
    fitted, _ = fit_index(models, counts)
    return dict(method="omle", episodes=episodes, mle_calls=0,
                planning_calls=0, joint_search_calls=calls,
                pair_scores=pair_scores, stopped=False,
                worst_policy_tv=all_policy_tv(truth, models[fitted]),
                fitted_index=fitted)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = candidate_class()
    runs = []
    for eps, condition, seed in product(ACCURACIES, ("realizable", "continuous_truth"), range(args.seeds)):
        truth = models[seed % len(models)] if condition == "realizable" else continuous_truth(seed)
        for method in ("omle", "uniform", "count"):
            result = (run_omle(models, truth, seed, eps) if method == "omle"
                      else run_mle_baseline(models, truth, seed, eps, method))
            runs.append(dict(condition=condition, epsilon=eps, seed=seed, **result,
                             target_tv=eps/4))
    with (args.output / "full_psr_baseline_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(runs)
    # Add ours from its separately archived complete-loop run.
    ours_path = args.output / "full_psr_multicoordinate_runs.csv"
    ours = list(csv.DictReader(ours_path.open()))
    rows = runs + [dict(condition=r["condition"], epsilon=float(r["epsilon"]),
                        seed=int(r["seed"]), method="ours", episodes=int(r["episodes"]),
                        mle_calls=int(r["mle_calls"]), planning_calls=int(r["planning_calls"]),
                        joint_search_calls=0, stopped=r["stopped"] == "True",
                        worst_policy_tv=float(r["worst_policy_tv"]), target_tv=float(r["target_tv"]))
                   for r in ours]
    summary = []
    for condition, eps, method in product(("realizable", "continuous_truth"), ACCURACIES,
                                           ("ours", "omle", "uniform", "count")):
        group = [r for r in rows if r["condition"] == condition and r["epsilon"] == eps
                 and r["method"] == method]
        summary.append(dict(condition=condition, epsilon=eps, method=method, runs=len(group),
                            stop_rate=mean(r["stopped"] if isinstance(r["stopped"], bool)
                                           else r["stopped"] == "True" for r in group),
                            within_target=mean(r["worst_policy_tv"] <= eps/4 for r in group),
                            median_tv=median(r["worst_policy_tv"] for r in group),
                            p90_tv=sorted(r["worst_policy_tv"] for r in group)[int(.9*(len(group)-1))],
                            median_episodes=median(int(r["episodes"]) for r in group),
                            median_mle_calls=median(int(r["mle_calls"]) for r in group),
                            median_planning_calls=median(int(r["planning_calls"]) for r in group),
                            median_joint_calls=median(int(r["joint_search_calls"]) for r in group)))
    with (args.output / "full_psr_baseline_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(f"completed {len(runs)} baseline runs and summarized all four methods")


if __name__ == "__main__":
    main()
