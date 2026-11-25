"""PSR coordinates and stage support programs for multistage_pomdp.

The four singleton next-observation tests form a common invertible core.
Their fused norm is L1, so the dual coefficient constraints are infinity
norm bounds. Historical rows are compressed to a 36-column Gram factor.
"""
from functools import lru_cache
from math import log, sqrt

import cvxpy as cp
import numpy as np

from multistage_pomdp import ACTIONS, OBSERVATIONS, STATES


CORE = np.full((4, 4), 1 / 12)
np.fill_diagonal(CORE, .75)
CORE_INVERSE = np.linalg.inv(CORE)
STABILITY = float(np.linalg.norm(CORE_INVERSE, 1))
WIDTH = 4 + 4 * 2 * 4


@lru_cache(maxsize=None)
def operator(model, stage, observation, action):
    if stage == model.horizon - 1:
        return CORE[observation]
    transition = np.array([model.transition(stage, state, action)
                           for state in STATES])
    return CORE @ transition.T @ np.diag(CORE[:, observation]) @ CORE_INVERSE


def prefix_state(model, history):
    if len(history) % 2:
        raise ValueError("history must end with an action")
    state = np.full(4, .25)
    for stage in range(len(history) // 2):
        observation, action = history[2*stage:2*stage+2]
        state = operator(model, stage, observation, action) @ state
    probability = float(state.sum())
    return state / probability if probability else np.full(4, .25)


def feature(model, history):
    stage = len(history) // 2
    q = prefix_state(model, history)
    if stage == model.horizon - 1:
        # Terminal operators are scalar. The formal terminal core has size 1.
        return q, np.array([[operator(model, stage, o, a) @ q
                             for a in ACTIONS] for o in OBSERVATIONS])
    z = np.array([[operator(model, stage, o, a) @ q
                   for a in ACTIONS] for o in OBSERVATIONS])
    return q, z


class StageSupport:
    def __init__(self, model, stage, histories, candidates, delta=.1):
        self.model = model
        self.stage = stage
        self.terminal = stage == model.horizon - 1
        beta = max(1., log(3 * candidates / delta))
        energy = 8 * STABILITY**2 * 2 * beta
        width = 4 + (8 if self.terminal else 32)
        rows = []
        for history, count in histories.items():
            q, z = feature(model, history)
            rows.append(sqrt(count) * np.concatenate((q, z.ravel())))
        self.rows = np.array(rows).reshape((-1, width))
        if len(rows):
            _, singular, right = np.linalg.svd(self.rows, full_matrices=False)
            factor = singular[:, None] * right
        else:
            factor = np.empty((0, width))
        self.coefficient = cp.Variable(width)
        self.query = cp.Parameter(width)
        u = self.coefficient[:4]
        block = self.coefficient[4:].reshape((4, 2, 1 if self.terminal else 4), order="C")
        constraints = [cp.norm_inf(u) <= STABILITY / 2]
        for o in OBSERVATIONS:
            constraints.append(sum(cp.norm_inf(block[o, a]) for a in ACTIONS)
                               <= STABILITY / 2)
        if len(rows):
            constraints.append(cp.norm(factor @ self.coefficient / sqrt(energy), 2) <= 1)
        self.problem = cp.Problem(cp.Maximize(self.query @ self.coefficient), constraints)

    def solve(self, history):
        q, z = feature(self.model, history)
        self.query.value = np.concatenate((q, z.ravel()))
        value = self.problem.solve(solver="CLARABEL", warm_start=True)
        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"stage support failed: {self.problem.status}")
        residual = max(float(np.max(np.atleast_1d(constraint.violation())))
                       for constraint in self.problem.constraints)
        if residual > 1e-6:
            raise RuntimeError(f"stage support constraint residual {residual}")
        return max(0., float(value)), np.asarray(self.coefficient.value).copy()
