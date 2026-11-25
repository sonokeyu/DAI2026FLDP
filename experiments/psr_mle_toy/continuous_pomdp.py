"""Realizable continuous POMDP class with dense, stage-specific kernels."""
from dataclasses import dataclass

import numpy as np


_rng = np.random.default_rng(20261125)
# Two independently drawn, positive endpoint kernels at each transition stage.
ENDPOINTS = _rng.dirichlet(np.full(4, 2.), size=(5, 2, 2, 4))
ENDPOINTS = .1 / 4 + .9 * ENDPOINTS
# The default action reveals no stage parameter; intervention with action 1 does.
ENDPOINTS[:, 1, 0] = ENDPOINTS[:, 0, 0]
EMISSION = np.full((4, 4), 1 / 12)
np.fill_diagonal(EMISSION, .75)


@dataclass(frozen=True)
class ContinuousModel:
    weights: tuple[float, ...]

    @property
    def horizon(self):
        return len(self.weights) + 1

    @staticmethod
    def emission(state, observation):
        return float(EMISSION[state, observation])

    def transition(self, stage, state, action):
        weight = self.weights[stage]
        row = ((1 - weight) * ENDPOINTS[stage, 0, action, state]
               + weight * ENDPOINTS[stage, 1, action, state])
        return tuple(float(value) for value in row)

    def step(self, stage, mass, observation, action):
        observed = tuple(mass[s] * self.emission(s, observation)
                         for s in range(4))
        if stage == self.horizon - 1:
            return observed
        return tuple(sum(observed[s] * self.transition(stage, s, action)[next_state]
                         for s in range(4)) for next_state in range(4))

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
                [other for other in range(4) if other != state])
            action = policy[history + (observation,)]
            history += (observation, action)
            if stage < self.horizon - 1:
                weights = self.transition(stage, state, action)
                draw = rng.random()
                state = next(s for s in range(4) if draw < sum(weights[:s+1]))
        return history
