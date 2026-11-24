"""Finite partially observed combination lock with exact policy-TV planning.

There are two good latent states and one absorbing bad state. The current
observation is ambiguous about the bad state on even stages. The final
success bit is observable and is included in the trajectory law.
"""
from dataclasses import dataclass
from collections import defaultdict
from functools import lru_cache
from itertools import product
import random


STATES = (0, 1, 2)
OBSERVATIONS = (0, 1)


@dataclass(frozen=True)
class LockModel:
    correct: tuple[tuple[int, int], ...]
    actions: int = 3
    good_success: float = .9

    @property
    def horizon(self):
        return len(self.correct)

    def emission(self, stage, state, observation):
        p_zero = (.9, .1, .5 if stage % 2 == 0 else .1)[state]
        return p_zero if observation == 0 else 1 - p_zero

    def transition(self, stage, state, action):
        if state == 2 or action != self.correct[stage][state]:
            return (0.0, 0.0, 1.0)
        # Switching good states makes the correct action depend on history.
        return (.7, .3, 0.0) if state == 0 else (.3, .7, 0.0)

    def step(self, stage, mass, observation, action):
        next_mass = [0.0, 0.0, 0.0]
        for state in STATES:
            retained = mass[state] * self.emission(stage, state, observation)
            for following, chance in enumerate(self.transition(stage, state, action)):
                next_mass[following] += retained * chance
        return tuple(next_mass)

    def terminal_law(self, mass):
        good = mass[0] + mass[1]
        bad = mass[2]
        success = self.good_success * good + .1 * bad
        return (sum(mass) - success, success)

    def sample(self, policy, rng):
        state = rng.randrange(2)
        history = ()
        for stage in range(self.horizon):
            observation = int(rng.random() >= self.emission(stage, state, 0))
            action = policy[history + (observation,)]
            history += (observation, action)
            transition = self.transition(stage, state, action)
            draw = rng.random()
            state = next(i for i, mass in enumerate(transition)
                         if draw < sum(transition[:i + 1]))
        success = int(rng.random() < (self.good_success if state != 2 else .1))
        return history + (success,)

    def trajectory_probability(self, trace):
        if len(trace) != 2 * self.horizon + 1:
            raise ValueError("trajectory length does not match horizon")
        mass = (.5, .5, 0.0)
        for stage in range(self.horizon):
            observation, action = trace[2 * stage:2 * stage + 2]
            mass = self.step(stage, mass, observation, action)
        return self.terminal_law(mass)[trace[-1]]

    def prefix_mass(self, history):
        """Unnormalized latent mass after a complete observed-action prefix."""
        if len(history) % 2:
            raise ValueError("prefix must end with an action")
        mass = (.5, .5, 0.0)
        for stage in range(len(history) // 2):
            mass = self.step(stage, mass, *history[2 * stage:2 * stage + 2])
        return mass


class RandomPolicy:
    def __init__(self, rng, actions):
        self.rng = rng
        self.actions = actions

    def __getitem__(self, history):
        return self.rng.randrange(self.actions)


class CountPolicy:
    def __init__(self, rng, actions):
        self.rng = rng
        self.actions = actions
        self.counts = {}

    def __getitem__(self, history):
        visits = self.counts.get(history, [0] * self.actions)
        least = min(visits)
        return self.rng.choice([i for i, count in enumerate(visits) if count == least])

    def record(self, trace):
        for stage in range((len(trace) - 1) // 2):
            history = trace[:2 * stage + 1]
            action = trace[2 * stage + 1]
            self.counts.setdefault(history, [0] * self.actions)[action] += 1


@lru_cache(maxsize=None)
def worst_policy_tv(first, second):
    """Maximize exact full-trajectory TV over shared history-based policies."""
    if first.horizon != second.horizon or first.actions != second.actions:
        raise ValueError("models must share horizon and action set")
    policy = defaultdict(int)

    def visit(stage, mass_first, mass_second, history):
        if mass_first == mass_second and mass_first[0] + mass_first[1] == 0:
            return 0.0
        if stage == first.horizon:
            end_first = first.terminal_law(mass_first)
            end_second = second.terminal_law(mass_second)
            return .5 * sum(abs(a - b) for a, b in zip(end_first, end_second))
        total = 0.0
        for observation in OBSERVATIONS:
            branch = history + (observation,)
            choices = []
            for action in range(first.actions):
                next_first = first.step(stage, mass_first, observation, action)
                next_second = second.step(stage, mass_second, observation, action)
                choices.append((visit(stage + 1, next_first, next_second,
                                      branch + (action,)), action))
            value, action = max(choices, key=lambda pair: (pair[0], -pair[1]))
            policy[branch] = action
            total += value
        return total

    tv = visit(0, (.5, .5, 0.0), (.5, .5, 0.0), ())
    return tv, policy


@lru_cache(maxsize=None)
def success_regret(truth, fitted):
    """Exact regret after planning for the terminal success reward in fitted."""
    def plan(model):
        policy = defaultdict(int)

        def visit(stage, mass, history):
            if mass[0] + mass[1] == 0:
                return .1 * sum(mass)
            if stage == model.horizon:
                return model.terminal_law(mass)[1]
            total = 0.0
            for observation in OBSERVATIONS:
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

    def evaluate(model, policy):
        def visit(stage, mass, history):
            if mass[0] + mass[1] == 0:
                return .1 * sum(mass)
            if stage == model.horizon:
                return model.terminal_law(mass)[1]
            return sum(visit(stage + 1,
                             model.step(stage, mass, observation,
                                        policy[history + (observation,)]),
                             history + (observation,
                                        policy[history + (observation,)]))
                       for observation in OBSERVATIONS)

        return visit(0, (.5, .5, 0.), ())

    optimum, _ = plan(truth)
    _, fitted_policy = plan(fitted)
    return max(0., optimum - evaluate(truth, fitted_policy))


def candidate_models(horizon, count=8, actions=3, seed=49157,
                     family="pair"):
    rng = random.Random(seed + 97 * horizon)
    prefix = tuple((rng.randrange(actions), rng.randrange(actions))
                   for _ in range(horizon - 1))
    terminal = list(product(range(actions), repeat=2))
    if family == "pair":
        if count > len(terminal):
            raise ValueError("candidate count exceeds distinct terminal action pairs")
        return tuple(LockModel(prefix + (last,), actions)
                     for last in rng.sample(terminal, count))
    if family == "graded":
        if count % 2 or count < 4:
            raise ValueError("graded family requires an even count >= 4")
        pairs = rng.sample(terminal, 2)
        levels = tuple(.82 + .01 * i for i in range(count // 2))
        if levels[-1] > .9:
            raise ValueError("graded levels must not exceed .9")
        return tuple(LockModel(prefix + (last,), actions, level)
                     for last in pairs for level in levels)
    raise ValueError("unknown candidate family")
