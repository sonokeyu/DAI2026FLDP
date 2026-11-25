"""A 64-hidden-state, four-observation POMDP with a length-three PSR core.

Hidden states are triples of four-valued symbols. Action zero rotates the
triple. At each nonterminal stage action one mixes that rotation with a
second permutation. Its probability has a separate unknown value for each
stage and current hidden symbol, giving twelve continuous parameters.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np


HIDDEN = tuple(product(range(4), repeat=3))
STATE_COUNT = len(HIDDEN)
FIRST_SYMBOL = np.array([hidden[0] for hidden in HIDDEN])
OBSERVATIONS = range(4)
ACTIONS = range(2)
EMISSION = np.full((STATE_COUNT, 4), 1 / 12)
for index, (first, _, _) in enumerate(HIDDEN):
    EMISSION[index, first] = .75
NEXT_ZERO = np.array([HIDDEN.index((second, third, first))
                      for first, second, third in HIDDEN])
NEXT_ONE = np.array([HIDDEN.index((first, third, second))
                     for first, second, third in HIDDEN])
INVERSE_ZERO = np.argsort(NEXT_ZERO)
INVERSE_ONE = np.argsort(NEXT_ONE)
_initial_rng = np.random.default_rng(641604)
INITIAL = _initial_rng.dirichlet(np.full(STATE_COUNT, 2.))


def core_matrix(length):
    """Rows are all noisy observation strings under repeated action zero."""
    if not 1 <= length <= 3:
        raise ValueError("core length must be 1, 2, or 3")
    rows = []
    for observations in product(OBSERVATIONS, repeat=length):
        rows.append([np.prod([.75 if hidden[t] == observation else 1 / 12
                              for t, observation in enumerate(observations)])
                     for hidden in HIDDEN])
    return np.asarray(rows)


CORES = {length: core_matrix(length) for length in (1, 2, 3)}
STABILITY = float(np.linalg.norm(np.linalg.inv(CORES[3]), 1))
TRUE_WEIGHTS = (.20, .35, .55, .75,
                .65, .25, .80, .40,
                .30, .60, .45, .85)


@dataclass(frozen=True)
class QueueModel:
    weights: tuple[float, ...]

    def __post_init__(self):
        if len(self.weights) != 12:
            raise ValueError("queue model needs three stages of four weights")

    @property
    def horizon(self):
        return 4

    @property
    def initial_mass(self):
        return INITIAL.copy()

    def core_length(self, stage):
        return min(3, self.horizon - stage)

    def transition_mass(self, stage, mass, action):
        source = np.asarray(mass)
        first = source[..., INVERSE_ZERO]
        if action == 0:
            return first
        second = source[..., INVERSE_ONE]
        weights = np.asarray(self.weights).reshape((3, 4))[stage, FIRST_SYMBOL]
        return ((1 - weights[INVERSE_ZERO]) * first
                + weights[INVERSE_ONE] * second)

    def step(self, stage, mass, observation, action):
        observed = np.asarray(mass) * EMISSION[:, observation]
        if stage == self.horizon - 1:
            return observed
        return self.transition_mass(stage, observed, action)

    def trajectory_probability(self, trace):
        if len(trace) != 2 * self.horizon:
            raise ValueError("incorrect trajectory length")
        mass = self.initial_mass
        for stage in range(self.horizon):
            mass = self.step(stage, mass, *trace[2*stage:2*stage+2])
        return float(mass.sum())

    def sample(self, policy, rng):
        state = int(rng.choice(STATE_COUNT, p=INITIAL))
        history = ()
        for stage in range(self.horizon):
            first = HIDDEN[state][0]
            observation = first if rng.random() < .75 else int(rng.choice(
                [other for other in OBSERVATIONS if other != first]))
            action = policy[history + (observation,)]
            history += (observation, action)
            if stage < self.horizon - 1:
                weight = self.weights[4 * stage + first]
                state = int(NEXT_ONE[state] if action == 1
                            and rng.random() < weight
                            else NEXT_ZERO[state])
        return history
