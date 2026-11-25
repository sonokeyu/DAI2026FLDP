"""Ordinary continuous MLE and fitted bonus planning on dense POMDPs.

The cone's likelihood budget uses an effective class size of 100 as a
numerical probe. It is not a theorem calibration for a continuous class.
"""
from argparse import ArgumentParser
from collections import Counter
from math import log, sqrt
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from continuous_pomdp import ContinuousModel, EMISSION, ENDPOINTS
from multistage_pomdp import worst_policy_tv
from multistage_psr import STABILITY, StageSupport
from run_multistage_experiment import write_csv
from run_multistage_ours import design_law, plan_bonus


EFFECTIVE_CLASS_SIZE = 100


def fit_mle(counts, horizon, previous=None):
    """Numerically maximize the ordinary complete-trajectory likelihood."""
    if not counts:
        return ContinuousModel((.5,) * (horizon - 1)), 0, float("nan")
    traces = np.asarray(tuple(counts), dtype=np.int64)
    weights = np.asarray(tuple(counts.values()), dtype=float)
    weights /= weights.sum()
    observations = traces[:, 0::2]
    actions = traces[:, 1::2]
    emission = np.array([EMISSION[:, observations[:, h]].T
                         for h in range(horizon)])
    endpoints = np.array([
        np.stack((ENDPOINTS[h, 0, actions[:, h]],
                  ENDPOINTS[h, 1, actions[:, h]]))
        for h in range(horizon - 1)])

    def objective(parameters):
        mass = np.full((len(traces), 4), .25)
        for h in range(horizon):
            mass *= emission[h]
            if h < horizon - 1:
                kernel = (endpoints[h, 0]
                          + parameters[h] * (endpoints[h, 1] - endpoints[h, 0]))
                mass = np.einsum("ni,nij->nj", mass, kernel, optimize=True)
        return -float(weights @ np.log(mass.sum(axis=1)))

    starts = [(float(level),) * (horizon - 1) for level in (.1, .5, .9)]
    if previous is not None:
        starts.insert(0, previous.weights)
    results = [minimize(objective, start, method="L-BFGS-B",
                        bounds=[(0., 1.)] * (horizon - 1),
                        options={"maxiter": 300, "ftol": 1e-13, "gtol": 1e-9})
               for start in starts]
    result = min(results, key=lambda candidate: candidate.fun)
    if not np.isfinite(result.fun):
        raise RuntimeError("continuous MLE returned a nonfinite likelihood")
    return ContinuousModel(tuple(float(value) for value in result.x)), sum(
        candidate.nfev for candidate in results), float(result.fun)


def run_case(truth, seed, epsilon, batch, max_rounds):
    rng = np.random.default_rng(seed + 1907)
    data = Counter()
    histories = [Counter() for _ in range(truth.horizon)]
    initial_count = episodes = 0
    fit = None
    rows = []
    beta = max(1., log(3 * EFFECTIVE_CLASS_SIZE / .1))
    energy = 8 * STABILITY**2 * 2 * beta

    def draw(law, size, training_stage=None):
        nonlocal episodes
        if not size:
            return
        traces = tuple(law)
        probabilities = np.asarray([law[trace] for trace in traces])
        probabilities /= probabilities.sum()
        sampled = rng.multinomial(size, probabilities)
        episodes += size
        for trace, count in zip(traces, sampled):
            if count:
                count = int(count)
                data[trace] += count
                if training_stage is not None:
                    histories[training_stage][trace[:2 * training_stage]] += count

    for round_number in range(1, max_rounds + 1):
        fit, objective_evals, loss = fit_mle(data, truth.horizon, fit)
        supports = [StageSupport(fit, h, histories[h], EFFECTIVE_CLASS_SIZE)
                    for h in range(truth.horizon)]
        root_bonus = min(STABILITY, sqrt(energy / initial_count)) if initial_count else STABILITY
        bonus, policy, coefficients = plan_bonus(fit, supports, root_bonus)
        tv = worst_policy_tv(truth, fit)[0]
        row = dict(seed=seed, horizon=truth.horizon, epsilon=epsilon,
                   round=round_number, episodes=episodes, batch=batch,
                   mle_calls=round_number, planning_calls=round_number,
                   objective_evals=objective_evals, negative_log_likelihood=loss,
                   true_weights=str(truth.weights), fitted_weights=str(fit.weights),
                   bonus=bonus, tv=tv, stopped=bonus <= 3 * epsilon / 16,
                   audit_valid=tv <= bonus + 1e-8)
        rows.append(row)
        if row["stopped"] or round_number == max_rounds:
            break

        initial_count += batch
        draw(design_law(truth, policy, stage=0, branch="u"), batch)
        for stage in range(truth.horizon):
            core_count = int(rng.binomial(batch, .5))
            draw(design_law(truth, policy, stage=stage, branch="u"),
                 core_count, stage)
            draw(design_law(truth, policy, stage=stage, branch="uniform"),
                 batch - core_count, stage)
        choices = [(None, "full", cut) for cut in range(truth.horizon)]
        choices += [(stage, branch, cut) for cut in range(truth.horizon)
                    for stage in range(truth.horizon) for branch in ("u", "w")]
        probabilities = ([1 / (3 * truth.horizon)] * truth.horizon
                         + [1 / (3 * truth.horizon**2)]
                         * (2 * truth.horizon**2))
        sizes = rng.multinomial(batch, probabilities)
        for (stage, branch, cut), size in zip(choices, sizes):
            if size:
                draw(design_law(truth, policy, stage=stage, branch=branch,
                                coefficients=coefficients, cut=cut), int(size))
    return rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--batch", type=int, default=10_000_000)
    parser.add_argument("--epsilon", type=float, default=.1)
    parser.add_argument("--rounds", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    truth = ContinuousModel((.23, .67, .42))
    runs = []
    for seed in range(args.seeds):
        path = run_case(truth, seed, args.epsilon, args.batch, args.rounds)
        runs.extend(path)
        print(f"seed={seed}, calls={len(path)}, tv={path[-1]['tv']:.6g}, "
              f"stopped={path[-1]['stopped']}", flush=True)
    write_csv(args.output / "continuous_runs.csv", runs)
    write_csv(args.output / "continuous_final.csv",
              [row for row in runs if row["stopped"] or row["round"] == args.rounds])


if __name__ == "__main__":
    main()
