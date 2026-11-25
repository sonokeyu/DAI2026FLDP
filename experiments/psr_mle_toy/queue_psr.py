"""Exact predictive features and stage support programs for queue_pomdp."""
from functools import lru_cache
from math import log, sqrt

import cvxpy as cp
import numpy as np

from queue_pomdp import CORES, STABILITY


@lru_cache(maxsize=None)
def prefix_belief(model, history):
    mass = model.initial_mass
    for stage in range(len(history) // 2):
        mass = model.step(stage, mass, *history[2*stage:2*stage+2])
    return mass / mass.sum()


@lru_cache(maxsize=None)
def feature(model, history):
    stage = len(history) // 2
    belief = prefix_belief(model, history)
    q = CORES[model.core_length(stage)] @ belief
    if stage == model.horizon - 1:
        z = np.array([[np.sum(model.step(stage, belief, observation, action))]
                      for observation in range(4) for action in range(2)])
        return q, z.reshape((4, 2, 1))
    next_core = CORES[model.core_length(stage + 1)]
    z = np.array([[next_core @ model.step(stage, belief, observation, action)
                   for action in range(2)] for observation in range(4)])
    return q, z


class StageSupport:
    def __init__(self, model, stage, histories, effective_class_size=100):
        self.model = model
        self.stage = stage
        self.current_dim = 4**model.core_length(stage)
        self.next_dim = (1 if stage == model.horizon - 1
                         else 4**model.core_length(stage + 1))
        width = self.current_dim + 8 * self.next_dim
        beta = max(1., log(3 * effective_class_size / .1))
        energy = 8 * STABILITY**2 * 2 * beta
        rows = []
        for history, count in histories.items():
            q, z = feature(model, history)
            rows.append(sqrt(count) * np.concatenate((q, z.ravel())))
        if rows:
            _, singular, right = np.linalg.svd(np.asarray(rows),
                                               full_matrices=False)
            factor = singular[:, None] * right
        else:
            factor = np.empty((0, width))
        self.coefficient = cp.Variable(width)
        self.query = cp.Parameter(width)
        bound = STABILITY / 2
        constraints = [cp.norm_inf(self.coefficient[:self.current_dim]) <= bound]
        for observation in range(4):
            masses = []
            for action in range(2):
                start = self.current_dim + (2 * observation + action) * self.next_dim
                masses.append(cp.norm_inf(self.coefficient[start:start+self.next_dim]))
            constraints.append(sum(masses) <= bound)
        if len(rows):
            constraints.append(cp.norm(factor @ self.coefficient / sqrt(energy), 2) <= 1)
        self.problem = cp.Problem(cp.Maximize(self.query @ self.coefficient), constraints)

    def solve(self, history):
        q, z = feature(self.model, history)
        self.query.value = np.concatenate((q, z.ravel()))
        value = self.problem.solve(solver="CLARABEL", warm_start=True)
        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"queue support failed: {self.problem.status}")
        residual = max(float(np.max(np.atleast_1d(constraint.violation())))
                       for constraint in self.problem.constraints)
        if residual > 1e-6:
            raise RuntimeError(f"queue support residual {residual}")
        return max(0., float(value)), np.asarray(self.coefficient.value).copy()
