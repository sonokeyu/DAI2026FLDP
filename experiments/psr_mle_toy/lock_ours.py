"""Terminal-error specialization of the paper's MLE + bonus-planning method.

All candidates share the initial law and operators through H-1. Hence their
only nonzero telescoping error is at H. The terminal witness is a linear
functional of the two good-state prefix masses with coefficient infinity
norm at most .8. The box/ellipsoid support below is an outer relaxation of
the paper's stage program; its value is therefore a conservative certificate.
Uniform terminal interventions calibrate its historical energy by ordinary
complete-episode MLE. No model-pair or confidence-set optimization is used.
"""
from collections import Counter
from math import fsum, log, sqrt
import random

from pocomblock import worst_policy_tv
from run_lock_baselines import score_fit, update_scores


COEFFICIENT_BOUND = .8
DELTA = .1


def _dot(left, right):
    return left[0] * right[0] + left[1] * right[1]


class BoxEnergySupport:
    """Exact two-dimensional box/ellipsoid support, including singular Gram."""

    def __init__(self, rows, energy):
        self.rows = tuple((q, n) for q, n in rows if n)
        self.energy = energy
        self.gram = (
            fsum(n * q[0] ** 2 for q, n in self.rows),
            fsum(n * q[0] * q[1] for q, n in self.rows),
            fsum(n * q[1] ** 2 for q, n in self.rows),
        )
        self.det = fsum(n * m * (q[0] * r[1] - q[1] * r[0]) ** 2
                        for i, (q, n) in enumerate(self.rows)
                        for r, m in self.rows[:i])
        w = COEFFICIENT_BOUND
        a, b, c = self.gram
        candidates = [(0., 0.)]
        for x in (-w, w):
            for y in (-w, w):
                candidates.append((x, y))
        for fixed, free in ((0, c), (1, a)):
            if free:
                center = -b * w / free
                margin = energy - self.det * w * w / free
                if margin >= 0:
                    radius = sqrt(margin / free)
                    for sign in (-1, 1):
                        for other in (center - radius, center + radius):
                            pair = (sign * w, sign * other)
                            candidates.append(pair if fixed == 0 else pair[::-1])
        self.candidates = candidates

    def __call__(self, q):
        if q == (0., 0.):
            return 0.
        a, b, c = self.gram
        choices = list(self.candidates)
        if self.det > 0:
            direction = (c * q[0] - b * q[1], a * q[1] - b * q[0])
            denom = _dot(q, direction)
            if denom > 0:
                scale = sqrt(self.energy / (self.det * denom))
                choices.append((scale * direction[0], scale * direction[1]))
        best = 0.
        for x, y in choices:
            if max(abs(x), abs(y)) > COEFFICIENT_BOUND + 1e-9:
                continue
            x = max(-COEFFICIENT_BOUND, min(COEFFICIENT_BOUND, x))
            y = max(-COEFFICIENT_BOUND, min(COEFFICIENT_BOUND, y))
            energy = a*x*x + 2*b*x*y + c*y*y
            if energy > self.energy * (1 + 1e-8):
                continue
            best = max(best, x*q[0] + y*q[1])
        return best


def plan_bonus(model, support):
    """Exact history-based single-model planning for the terminal bonus."""
    policy = {}

    def visit(stage, mass, history):
        if stage == model.horizon - 1:
            return support(mass[:2])
        if mass[0] + mass[1] == 0:
            return 0.
        total = 0.
        for observation in (0, 1):
            branch = history + (observation,)
            value, action = max(
                ((visit(stage + 1, model.step(stage, mass, observation, action),
                        branch + (action,)), action)
                 for action in range(model.actions)),
                key=lambda pair: (pair[0], -pair[1]),
            )
            policy[branch] = action
            total += value
        return total

    return visit(0, (.5, .5, 0.), ()), policy


class TerminalIntervention:
    def __init__(self, policy, horizon, actions, rng):
        self.policy = policy
        self.horizon = horizon
        self.actions = actions
        self.rng = rng

    def __getitem__(self, history):
        if len(history) == 2 * self.horizon - 1:
            return self.rng.randrange(self.actions)
        return self.policy.get(history, 0)


def run_ours(models, truth, seed, horizon, budget, epsilon, batch=512):
    """Return one row per fit, with an exact ex-post TV audit."""
    rng = random.Random(2399 * horizon + 89 * seed)
    scores = [0.] * len(models)
    histories = Counter()
    reference = models[0]  # common prefix law, known from the model class
    # A fixed upper bound on the number of refits makes the likelihood budget
    # time-uniform over every refit in this finite experiment.
    beta = log(3 * len(models) * (1 + budget // batch) / DELTA)
    energy = 2 * truth.actions**2 * beta
    records = []
    episodes = 0
    for round_number in range(1 + budget // batch):
        fitted_index = score_fit(scores)
        fitted = models[fitted_index]
        rows = []
        for history, count in histories.items():
            mass = reference.prefix_mass(history)
            probability = sum(mass)
            rows.append(((mass[0] / probability, mass[1] / probability), count))
        support = BoxEnergySupport(rows, energy)
        bonus, policy = plan_bonus(fitted, support)
        tv = worst_policy_tv(truth, fitted)[0]
        stopped = bonus <= epsilon / 4
        records.append(dict(method="ours", horizon=horizon, seed=seed,
                            fit=fitted_index,
                            episodes=episodes, round=round_number,
                            mle_calls=round_number + 1,
                            planning_calls=round_number + 1, joint_calls=0,
                            bonus=bonus, beta=beta, energy=energy,
                            tv=tv, stopped=stopped,
                            certificate_valid=(tv <= bonus + 1e-9)))
        if stopped or episodes == budget:
            break
        intervention = TerminalIntervention(policy, horizon, truth.actions, rng)
        for _ in range(min(batch, budget - episodes)):
            trace = truth.sample(intervention, rng)
            update_scores(models, scores, trace)
            histories[trace[:2 * (horizon - 1)]] += 1
            episodes += 1
    return records
