"""Finite POMDP family with rank-controlled, stage-specific unknown kernels.

This module supplies the common environment and exact evaluation primitive.
It intentionally has no learning algorithm: an algorithm must interact only
through logged trajectories, while the evaluator may inspect both models.
"""
from dataclasses import dataclass
from functools import lru_cache


STATES = range(4)
OBSERVATIONS = range(4)
ACTIONS = range(2)


@dataclass(frozen=True)
class Model:
    rank: int
    shifts: tuple[int, ...]

    @property
    def horizon(self):
        return len(self.shifts) + 1

    @staticmethod
    def emission(state, observation):
        return .75 if state == observation else 1 / 12

    def transition(self, stage, state, action):
        target = (self.shifts[stage] if self.rank == 1 else
                  (state + action + action * self.shifts[stage]) % self.rank)
        return tuple(.05 * (following == target) + .95 / 4
                     for following in STATES)

    def step(self, stage, mass, observation, action):
        observed = tuple(mass[s] * self.emission(s, observation)
                         for s in STATES)
        if stage == self.horizon - 1:
            return observed
        return tuple(sum(observed[s] * self.transition(stage, s, action)[next_state]
                         for s in STATES) for next_state in STATES)

    def trajectory_probability(self, trace):
        if len(trace) != 2 * self.horizon:
            raise ValueError("incorrect trajectory length")
        mass = (.25,) * 4
        for stage in range(self.horizon):
            mass = self.step(stage, mass, *trace[2*stage:2*stage+2])
        return sum(mass)

    def sample(self, policy, rng):
        state = rng.randrange(4)
        history = ()
        for stage in range(self.horizon):
            observation = state if rng.random() < .75 else rng.choice(
                [other for other in STATES if other != state])
            action = policy[history + (observation,)]
            history += (observation, action)
            if stage < self.horizon - 1:
                weights = self.transition(stage, state, action)
                draw = rng.random()
                state = next(s for s in STATES
                             if draw < sum(weights[:s+1]))
        return history


def candidates(horizon):
    if horizon < 2:
        raise ValueError("horizon must be at least two")
    patterns = [(0,) * (horizon - 1)]
    patterns += [tuple(int(i == stage) for i in range(horizon - 1))
                 for stage in range(horizon - 1)]
    if horizon > 2:
        patterns.append((1,) * (horizon - 1))
    return tuple(Model(rank, pattern) for rank in (1, 2, 4)
                 for pattern in patterns)


@lru_cache(maxsize=None)
def worst_policy_tv(first, second):
    """Exact maximum over every observation-history policy, with a witness."""
    if first.horizon != second.horizon:
        raise ValueError("models must share horizon")
    policy = {}

    def visit(stage, first_mass, second_mass, history):
        if stage == first.horizon:
            return abs(sum(first_mass) - sum(second_mass)) / 2
        total = 0.
        for observation in OBSERVATIONS:
            branch = history + (observation,)
            value, action = max(
                ((visit(stage + 1,
                        first.step(stage, first_mass, observation, action),
                        second.step(stage, second_mass, observation, action),
                        branch + (action,)), action) for action in ACTIONS),
                key=lambda item: (item[0], -item[1]),
            )
            policy[branch] = action
            total += value
        return total

    return visit(0, (.25,) * 4, (.25,) * 4, ()), policy
