"""64-state PSR exploration with continuous MLE and exact small-horizon planning.

The effective class-size parameter is a numerical bonus calibration, not a
continuous-class bracketing proof. Exact all-policy TV is used only for audit.
"""
from argparse import ArgumentParser
from collections import Counter
from functools import lru_cache
from math import log, sqrt
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from queue_pomdp import (ACTIONS, EMISSION, FIRST_SYMBOL, INITIAL,
                         INVERSE_ONE, INVERSE_ZERO, OBSERVATIONS,
                         QueueModel, STABILITY, STATE_COUNT, TRUE_WEIGHTS)
from queue_psr import StageSupport
from run_multistage_experiment import write_csv


EFFECTIVE_CLASS_SIZE = 100


@lru_cache(maxsize=None)
def worst_policy_tv(first, second):
    """Exact maximum over all observation-history policies at horizon four."""
    policy = {}

    def visit(stage, first_mass, second_mass, history):
        if stage == first.horizon:
            return abs(float(first_mass.sum() - second_mass.sum())) / 2
        total = 0.
        for observation in OBSERVATIONS:
            branch = history + (observation,)
            best = max(((visit(stage + 1,
                               first.step(stage, first_mass, observation, action),
                               second.step(stage, second_mass, observation, action),
                               branch + (action,)), action) for action in ACTIONS),
                       key=lambda pair: (pair[0], -pair[1]))
            policy[branch] = best[1]
            total += best[0]
        return total

    return visit(0, first.initial_mass, second.initial_mass, ()), policy


def fit_mle(data, previous=None):
    if not data:
        return QueueModel((0.,) * 12), 0, float("nan")
    traces = np.asarray(tuple(data), dtype=np.int64)
    weights = np.asarray(tuple(data.values()), dtype=float)
    weights /= weights.sum()
    observations = traces[:, 0::2]
    actions = traces[:, 1::2]
    emissions = np.array([EMISSION[:, observations[:, h]].T
                          for h in range(4)])

    def objective(parameters):
        mass = np.broadcast_to(INITIAL, (len(traces), STATE_COUNT)).copy()
        stage_weights = parameters.reshape((3, 4))
        for stage in range(4):
            mass *= emissions[stage]
            if stage < 3:
                first = mass[:, INVERSE_ZERO]
                second = mass[:, INVERSE_ONE]
                weight = stage_weights[stage, FIRST_SYMBOL]
                active = actions[:, stage, None]
                mass = (first * (1 - active * weight[INVERSE_ZERO])
                        + second * active * weight[INVERSE_ONE])
        return -float(weights @ np.log(mass.sum(axis=1)))

    starts = [(level,) * 12 for level in (.1, .5, .9)]
    if previous is not None:
        starts.insert(0, previous.weights)
    results = [minimize(objective, start, method="L-BFGS-B",
                        bounds=[(0., 1.)] * 12,
                        options={"maxiter": 300, "ftol": 1e-13, "gtol": 1e-9})
               for start in starts]
    result = min(results, key=lambda candidate: candidate.fun)
    if not np.isfinite(result.fun):
        raise RuntimeError("queue MLE returned a nonfinite likelihood")
    return (QueueModel(tuple(float(value) for value in result.x)),
            sum(candidate.nfev for candidate in results), float(result.fun))


def plan_bonus(model, supports, root_bonus):
    policy, coefficients = {}, {}

    def visit(stage, mass, history):
        if stage == model.horizon:
            return 0.
        bonus, coefficient = supports[stage].solve(history)
        coefficients[history] = coefficient
        value = float(mass.sum()) * bonus
        for observation in OBSERVATIONS:
            branch = history + (observation,)
            best = max(((visit(stage + 1,
                               model.step(stage, mass, observation, action),
                               branch + (action,)), action) for action in ACTIONS),
                       key=lambda pair: (pair[0], -pair[1]))
            policy[branch] = best[1]
            value += best[0]
        return value

    return root_bonus + visit(0, model.initial_mass, ()), policy, coefficients


def design_law(model, policy, *, stage=None, branch="full",
               coefficients=None, cut=None):
    outcomes = {}
    bound = STABILITY / 2

    def actions(history):
        h = (len(history) - 1) // 2
        if cut is not None and h >= cut:
            return ((0, .5), (1, .5)) if h == cut and cut > 0 else ((0, 1.),)
        if stage is None or h < stage:
            return ((policy.get(history, 0), 1.),)
        if h > stage:
            return ((0, 1.),)
        if branch == "uniform":
            return ((0, .5), (1, .5))
        if branch == "w":
            current_dim = 4**model.core_length(h)
            next_dim = 1 if h == model.horizon - 1 else 4**model.core_length(h + 1)
            block = coefficients[history[:-1]][current_dim:].reshape((4, 2, next_dim))
            p_one = min(1., float(np.max(np.abs(block[history[-1], 1]))) / bound)
            return ((0, 1 - p_one), (1, p_one))
        return ((0, 1.),)

    def visit(h, mass, history):
        if h == model.horizon:
            outcomes[history] = outcomes.get(history, 0.) + float(mass.sum())
            return
        for observation in OBSERVATIONS:
            observed = history + (observation,)
            for action, probability in actions(observed):
                if probability:
                    visit(h + 1, probability * model.step(h, mass, observation, action),
                          observed + (action,))

    visit(0, model.initial_mass, ())
    return outcomes


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
        counts = rng.multinomial(size, probabilities)
        episodes += size
        for trace, count in zip(traces, counts):
            if count:
                count = int(count)
                data[trace] += count
                if training_stage is not None:
                    histories[training_stage][trace[:2 * training_stage]] += count

    for round_number in range(1, max_rounds + 1):
        fit, objective_evals, loss = fit_mle(data, fit)
        supports = [StageSupport(fit, stage, histories[stage], EFFECTIVE_CLASS_SIZE)
                    for stage in range(truth.horizon)]
        root_bonus = min(STABILITY, sqrt(energy / initial_count)) if initial_count else STABILITY
        bonus, policy, coefficients = plan_bonus(fit, supports, root_bonus)
        tv = worst_policy_tv(truth, fit)[0]
        rows.append(dict(seed=seed, hidden_states=STATE_COUNT, observations=4,
                         horizon=truth.horizon, epsilon=epsilon, round=round_number,
                         episodes=episodes, batch=batch, mle_calls=round_number,
                         planning_calls=round_number, objective_evals=objective_evals,
                         negative_log_likelihood=loss, true_weights=str(truth.weights),
                         fitted_weights=str(fit.weights), bonus=bonus, tv=tv,
                         stopped=bonus <= 3 * epsilon / 16,
                         audit_valid=tv <= bonus + 1e-8))
        if rows[-1]["stopped"] or round_number == max_rounds:
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
                         + [1 / (3 * truth.horizon**2)] * (2 * truth.horizon**2))
        sizes = rng.multinomial(batch, probabilities)
        for (stage, branch, cut), size in zip(choices, sizes):
            if size:
                draw(design_law(truth, policy, stage=stage, branch=branch,
                                coefficients=coefficients, cut=cut), int(size))
    return rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1_000_000_000)
    parser.add_argument("--epsilon", type=float, default=.4)
    parser.add_argument("--rounds", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    truth = QueueModel(TRUE_WEIGHTS)
    runs = []
    for seed in range(args.seeds):
        path = run_case(truth, seed, args.epsilon, args.batch, args.rounds)
        runs.extend(path)
        print(f"seed={seed}, calls={len(path)}, tv={path[-1]['tv']:.6g}, "
              f"stopped={path[-1]['stopped']}", flush=True)
    write_csv(args.output / "queue_runs.csv", runs)
    write_csv(args.output / "queue_final.csv",
              [row for row in runs if row["stopped"] or row["round"] == args.rounds])


if __name__ == "__main__":
    main()
