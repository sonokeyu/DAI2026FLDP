"""Finite-class Algorithm 1 on the four-state noisy-observation PSR."""
from argparse import ArgumentParser
from collections import Counter
from functools import lru_cache
import csv
from math import log, sqrt
from pathlib import Path

import numpy as np

from multistage_pomdp import candidates, worst_policy_tv
from multistage_psr import STABILITY, StageSupport
from routing import Budget


def plan_bonus(model, supports, root_bonus):
    policy, coefficients = {}, {}

    def visit(stage, mass, history):
        if stage == model.horizon:
            return 0.
        bonus, coefficient = supports[stage].solve(history)
        coefficients[history] = coefficient
        value = float(sum(mass)) * bonus
        for observation in range(4):
            branch = history + (observation,)
            choice = max(((visit(stage + 1,
                                  model.step(stage, mass, observation, action),
                                  branch + (action,)), action)
                          for action in range(2)),
                         key=lambda pair: (pair[0], -pair[1]))
            policy[branch] = choice[1]
            value += choice[0]
        return value

    return root_bonus + visit(0, (.25,) * 4, ()), policy, coefficients


def design_law(model, policy, *, stage=None, branch="full",
               coefficients=None, cut=None):
    """Enumerate the logged design's physical trajectory law exactly."""
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
            block = coefficients[history[:-1]][4:].reshape((4, 2, -1))
            masses = [float(np.max(np.abs(block[history[-1], a])))
                      for a in range(2)]
            p_one = min(1., masses[1] / bound)
            return ((0, 1 - p_one), (1, p_one))
        return ((0, 1.),)

    def visit(h, mass, history):
        if h == model.horizon:
            outcomes[history] = outcomes.get(history, 0.) + sum(mass)
            return
        for observation in range(4):
            observed = history + (observation,)
            for action, probability in actions(observed):
                if probability:
                    next_mass = model.step(h, mass, observation, action)
                    visit(h + 1, tuple(probability * x for x in next_mass),
                          observed + (action,))

    visit(0, (.25,) * 4, ())
    return outcomes


def run_case(models, truth_index, seed, epsilon, batch, max_rounds,
             theory=False):
    truth = models[truth_index]
    count_rng = np.random.default_rng(seed + 1907)
    scores = [0.] * len(models)
    histories = [Counter() for _ in range(truth.horizon)]
    initial_count = episodes = 0
    rows = []
    beta = max(1., log(3 * len(models) / .1))
    energy = 8 * STABILITY**2 * 2 * beta
    budget = Budget(epsilon, candidates=len(models), horizon=truth.horizon,
                    score_bound=STABILITY,
                    risk_scale=2 * STABILITY**2, dimension=4)

    @lru_cache(maxsize=None)
    def log_probabilities(trace):
        return tuple(log(model.trajectory_probability(trace)) for model in models)

    limit = budget.cap() if theory else max_rounds
    for round_number in range(1, limit + 1):
        fitted_index = max(range(len(models)), key=lambda i: (scores[i], -i))
        fit = models[fitted_index]
        supports = [StageSupport(fit, h, histories[h], len(models))
                    for h in range(truth.horizon)]
        root_bonus = min(STABILITY, sqrt(energy / initial_count)) if initial_count else STABILITY
        bonus, policy, coefficients = plan_bonus(fit, supports, root_bonus)
        tv = worst_policy_tv(truth, fit)[0]
        row = dict(seed=seed, truth=truth_index, rank=truth.rank,
                   horizon=truth.horizon, epsilon=epsilon, round=round_number,
                   episodes=episodes, mle_calls=round_number,
                   planning_calls=round_number, fit=fitted_index,
                   bonus=bonus, tv=tv,
                   stopped=bonus <= 3 * epsilon / 16,
                   certificate_valid=tv <= bonus + 1e-8)
        rows.append(row)
        if row["stopped"] or round_number == limit:
            break

        training_batch = budget.theory_training(round_number) if theory else batch
        diagnostic_batch = budget.theory_nested() if theory else batch

        def record(trace, count, training_stage=None):
            nonlocal episodes
            episodes += count
            if training_stage is not None:
                histories[training_stage][trace[:2 * training_stage]] += count
            for i, value in enumerate(log_probabilities(trace)):
                scores[i] += count * value

        def draw(law, size, training_stage=None):
            if not size:
                return
            traces = tuple(law)
            probabilities = np.array([law[trace] for trace in traces])
            probabilities /= probabilities.sum()
            counts = count_rng.multinomial(size, probabilities)
            for trace, count in zip(traces, counts):
                if count:
                    record(trace, int(count), training_stage)

        initial_count += training_batch
        draw(design_law(truth, policy, stage=0, branch="u"), training_batch)
        for stage in range(truth.horizon):
            core_count = int(count_rng.binomial(training_batch, .5))
            draw(design_law(truth, policy, stage=stage, branch="u"),
                 core_count, stage)
            draw(design_law(truth, policy, stage=stage, branch="uniform"),
                 training_batch - core_count, stage)
        choices = [(None, "full", cut) for cut in range(truth.horizon)]
        choices += [(stage, branch, cut) for cut in range(truth.horizon)
                    for stage in range(truth.horizon) for branch in ("u", "w")]
        probabilities = ([1 / (3 * truth.horizon)] * truth.horizon
                         + [1 / (3 * truth.horizon**2)]
                         * (2 * truth.horizon**2))
        sizes = count_rng.multinomial(diagnostic_batch, probabilities)
        for (stage, branch, cut), size in zip(choices, sizes):
            if size:
                draw(design_law(truth, policy, stage=stage, branch=branch,
                                coefficients=coefficients, cut=cut), int(size))
    return rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=(3, 4))
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=.4)
    parser.add_argument("--theory", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for horizon in args.horizons:
        models = candidates(horizon)
        for rank_index in range(3):
            block = len(models) // 3
            truth_index = rank_index * block + block - 1
            for seed in range(args.seeds):
                rows.extend(run_case(models, truth_index,
                                     seed + 1000 * horizon + 100 * rank_index,
                                     args.epsilon, args.batch, args.rounds,
                                     theory=args.theory))
        print(f"completed H={horizon}", flush=True)
    with (args.output / "ours_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
