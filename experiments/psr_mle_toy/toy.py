"""Exact three-stage example from Section 4 of the PSR MLE paper.

This module studies the stage-two mechanism. It does not implement the
paper's complete H-stage certificate, SOCP, or prescribed batch sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import inf, log, sqrt
from random import Random


class Model(Enum):
    TRUE = "true"
    WRONG = "wrong"


CUES = ("-", "+")
ROUTES = ("L", "R")
MODES = ("a", "b")


@dataclass(frozen=True)
class Episode:
    cue: str
    route: str
    mode: str
    outcome: int


@dataclass(frozen=True)
class Policy:
    routes: tuple[str, str]  # Indexed by CUES.
    modes: tuple[str, str, str, str]  # Indexed by CUES x ROUTES.

    def route(self, cue: str) -> str:
        return self.routes[CUES.index(cue)]

    def mode(self, cue: str, route: str) -> str:
        return self.modes[CUES.index(cue) * 2 + ROUTES.index(route)]


def hidden_type(model: Model, cue: str, route: str) -> int:
    if route == "L":
        return 0
    return int(cue == "+" or model == Model.WRONG)


def outcome_probability(model: Model, cue: str, route: str, mode: str, y: int) -> float:
    if mode == "b":
        return 0.5
    return float(y == hidden_type(model, cue, route))


def trajectory_distribution(model: Model, policy: Policy) -> dict[Episode, float]:
    result: dict[Episode, float] = {}
    for cue in CUES:
        route = policy.route(cue)
        mode = policy.mode(cue, route)
        for y in (0, 1):
            probability = 0.5 * outcome_probability(model, cue, route, mode, y)
            if probability:
                result[Episode(cue, route, mode, y)] = probability
    return result


def total_variation(p: dict, q: dict) -> float:
    return 0.5 * sum(abs(p.get(x, 0.0) - q.get(x, 0.0)) for x in p.keys() | q.keys())


def all_deterministic_policies():
    for routes in product(ROUTES, repeat=2):
        for modes in product(MODES, repeat=4):
            yield Policy(routes, modes)


def max_policy_tv(first: Model, second: Model) -> tuple[float, Policy]:
    candidates = (
        (
            total_variation(trajectory_distribution(first, p), trajectory_distribution(second, p)),
            p,
        )
        for p in all_deterministic_policies()
    )
    return max(candidates, key=lambda item: item[0])


def expected_reward(model: Model, policy: Policy) -> float:
    return sum(e.outcome * mass for e, mass in trajectory_distribution(model, policy).items())


def optimal_reward(model: Model) -> tuple[float, Policy]:
    return max(((expected_reward(model, p), p) for p in all_deterministic_policies()),
               key=lambda item: item[0])


def log_likelihood(model: Model, episodes: list[Episode]) -> float:
    score = 0.0
    for e in episodes:
        probability = outcome_probability(model, e.cue, e.route, e.mode, e.outcome)
        if probability == 0:
            return -inf
        score += log(0.5 * probability)
    return score


def ordinary_mle(episodes: list[Episode], tie: Model = Model.WRONG) -> Model:
    scores = {m: log_likelihood(m, episodes) for m in Model}
    if scores[Model.TRUE] == scores[Model.WRONG]:
        return tie
    return max(scores, key=scores.get)


def stage_two_support(model: Model, cue: str, route: str,
                      histories: list[tuple[str, str]], energy_budget: float) -> float:
    """Exact Section 4 stage-two SOCP value for its two one-hot state rows.

    Training uses mode b, so each lifted row is (e_i, e_i, m). A coefficient
    on the queried type can attain min(2, sqrt(S/n_i)) while vanishing on
    the other type. No coefficient can exceed either bound.
    """
    if energy_budget < 0:
        raise ValueError("energy_budget must be nonnegative")
    state = hidden_type(model, cue, route)
    count = sum(hidden_type(model, c, r) == state for c, r in histories)
    return min(2.0, sqrt(energy_budget / count)) if count else 2.0


def bonus_policy(model: Model, histories: list[tuple[str, str]],
                 energy_budget: float) -> Policy:
    routes = []
    for cue in CUES:
        left = stage_two_support(model, cue, "L", histories, energy_budget)
        right = stage_two_support(model, cue, "R", histories, energy_budget)
        # A strict advantage is required; ties use L.
        routes.append("R" if right > left + 1e-12 else "L")
    return Policy(tuple(routes), ("b",) * 4)


def expected_stage_two_bonus(model: Model, policy: Policy,
                             histories: list[tuple[str, str]], energy_budget: float) -> float:
    return 0.5 * sum(
        stage_two_support(model, cue, policy.route(cue), histories, energy_budget)
        for cue in CUES
    )


def sample_episode(truth: Model, policy: Policy, rng: Random,
                   test_probability: float = 0.0) -> Episode:
    if not 0.0 <= test_probability <= 1.0:
        raise ValueError("test_probability must be in [0,1]")
    cue = CUES[rng.randrange(2)]
    route = policy.route(cue)
    mode = "a" if rng.random() < test_probability else policy.mode(cue, route)
    y = hidden_type(truth, cue, route) if mode == "a" else rng.randrange(2)
    return Episode(cue, route, mode, y)


@dataclass(frozen=True)
class Round:
    index: int
    fit: str
    fitted_bonus: float
    max_policy_tv: float
    task_regret: float
    episodes: int
    diagnostic_tests: int
    negative_right_tests: int


def run_stage_two_learning(seed: int, batch_size: int = 16, max_rounds: int = 20,
                           energy_budget: float = 1.0, stop_threshold: float = 0.2,
                           test_probability: float = 0.5) -> tuple[list[Round], bool]:
    """Stage-two mechanism loop with reencoded histories and ordinary MLE.

    test_probability=0 models passive task-policy data. 0.5 is the
    current-core branch probability in the paper's training batch, applied
    here to stage two only. Other stages and nested cuts are omitted.
    """
    if batch_size < 1 or max_rounds < 1:
        raise ValueError("batch_size and max_rounds must be positive")
    rng = Random(seed)
    data: list[Episode] = []
    histories: list[tuple[str, str]] = []
    rows: list[Round] = []
    for k in range(1, max_rounds + 1):
        fit = ordinary_mle(data)
        policy = bonus_policy(fit, histories, energy_budget)
        bonus = expected_stage_two_bonus(fit, policy, histories, energy_budget)
        tv, _ = max_policy_tv(fit, Model.TRUE)
        _, fitted_task_policy = optimal_reward(fit)
        true_optimal_value, _ = optimal_reward(Model.TRUE)
        task_regret = true_optimal_value - expected_reward(Model.TRUE, fitted_task_policy)
        tests = sum(e.mode == "a" for e in data)
        targeted = sum(e.cue == "-" and e.route == "R" and e.mode == "a" for e in data)
        rows.append(Round(k, fit.value, bonus, tv, task_regret,
                          len(data), tests, targeted))
        if bonus <= stop_threshold:
            return rows, True
        if k == max_rounds:
            return rows, False
        for _ in range(batch_size):
            episode = sample_episode(Model.TRUE, policy, rng, test_probability)
            data.append(episode)
            histories.append((episode.cue, episode.route))
    return rows, False


def diagnostic_branch_mass() -> float:
    """Paper Section 4: cut 2, full-policy component, - cue, uniform a."""
    return (1 / 3) * (1 / 3) * (1 / 2) * (1 / 2)
