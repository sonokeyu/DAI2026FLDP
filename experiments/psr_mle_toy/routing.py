"""Three-stage Algorithm 1 with exact planning and one core string per stage."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import ceil, fsum, log, log1p, sqrt

import numpy as np

from toy import CUES, MODES, ROUTES, Episode, Policy

H, A, U, D = 3, 2, 1, 2
L, W, Y = H + 1, 2.0, 8.0  # The paper's supplied stability bound is 2.
EPISODES = tuple(Episode(*x) for x in product(CUES, ROUTES, MODES, (0, 1)))
PREFIXES = tuple(product(CUES, ROUTES))
FIXED_ROUTES = (0.0, 0.0)  # Probability of R following each cue.
FIXED_MODES = (0.0,) * 4  # Probability of a following each (cue, route).


@dataclass(frozen=True)
class RoutingModel:
    negative_right: float

    def p(self, cue, route, mode):
        if mode == "b":
            return 0.5
        if route == "L":
            return 0.0
        return 1.0 if cue == "+" else self.negative_right

    def observation_weights(self):
        return np.array([
            0.5 * (p if e.outcome else 1 - p)
            for e in EPISODES for p in (self.p(e.cue, e.route, e.mode),)
        ])


@dataclass(frozen=True)
class Design:
    """Action kernels after the cue and ready observations."""
    routes: tuple[float, ...] = FIXED_ROUTES
    modes: tuple[float, ...] = FIXED_MODES

    def kernel(self):
        return _design_kernel(self.routes, self.modes)

    @classmethod
    def policy(cls, policy):
        return cls(tuple(float(r == "R") for r in policy.routes),
                   tuple(float(m == "a") for m in policy.modes))


@lru_cache(maxsize=None)
def _design_kernel(routes, modes):
    weights = []
    for e in EPISODES:
        right = routes[CUES.index(e.cue)]
        test = modes[PREFIXES.index((e.cue, e.route))]
        weights.append((right if e.route == "R" else 1 - right)
                       * (test if e.mode == "a" else 1 - test))
    return np.array(weights)


def mix(*designs):
    """Return a labeled uniform mixture, retaining duplicate components."""
    return [(1 / len(designs), design) for design in designs]


def law(model, components):
    return np.array([weight * design.kernel() * model.observation_weights()
                     for weight, design in components])


def hellinger(p, q):
    return 0.5 * float(np.square(np.sqrt(p) - np.sqrt(q)).sum())


def distance(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def all_policy_tv(first, second):
    # For a fixed prefix, the action maximizing the outcome TV is optimal.
    return 0.5 * sum(max(abs(first.p(c, r, a) - second.p(c, r, a))
                         for r in ROUTES for a in MODES) for c in CUES)


def task_value(first, second, reward):
    """Value in second of a policy planned for an observable reward in first."""
    def value(model, cue, route, mode):
        p = model.p(cue, route, mode)
        return sum(mass * reward(Episode(cue, route, mode, y))
                   for y, mass in enumerate((1 - p, p)))

    return 0.5 * sum(value(second, cue, *max(product(ROUTES, MODES),
                     key=lambda pair: value(first, cue, *pair))) for cue in CUES)


def vector(p):
    return (1.0 - p, p)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


class Support:
    """Maximize d·q over |d|<=W and sum_i n_i (d·q_i)^2<=S.

    For stages 2 and 3, the projected coefficient ball is exactly this box.
    Candidates are corners, ellipse intersections on all four edges, and
    the unconstrained ellipse optimum when the history matrix has rank two.
    """
    def __init__(self, history, energy):
        self.history = tuple((q, int(n)) for q, n in history if n)
        self.energy = energy
        self.gram = (
            fsum(n * q[0] ** 2 for q, n in self.history),
            fsum(n * q[0] * q[1] for q, n in self.history),
            fsum(n * q[1] ** 2 for q, n in self.history),
        )
        # Cauchy–Binet avoids subtracting large terms for repeated rows.
        self.det = fsum(n * m * (q[0] * r[1] - q[1] * r[0]) ** 2
                        for i, (q, n) in enumerate(self.history)
                        for r, m in self.history[:i])
        a, b, c = self.gram
        self.edges = list(product((-W, W), repeat=2))
        for fixed, free in ((0, c), (1, a)):
            if free:
                center = -b * W / free
                margin = energy - self.det * W ** 2 / free
                if margin >= 0:
                    radius = sqrt(margin / free)
                    for sign in (-1, 1):
                        for other in (center - radius, center + radius):
                            pair = (sign * W, sign * other)
                            self.edges.append(pair if fixed == 0 else pair[::-1])

    def __call__(self, q):
        candidates = [(0.0, 0.0), *self.edges]
        a, b, c = self.gram
        if self.det:
            d = (c * q[0] - b * q[1], a * q[1] - b * q[0])
            denom = dot(q, d)
            if denom > 0:
                scale = sqrt(self.energy / (self.det * denom))
                candidates.append(tuple(v * scale for v in d))
        best, coef = 0.0, (0.0, 0.0)
        for d in candidates:
            if max(map(abs, d)) > W + 1e-10:
                continue
            d = tuple(min(W, max(-W, v)) for v in d)
            energy = fsum(n * dot(d, r) ** 2 for r, n in self.history)
            if energy > self.energy:
                if energy > self.energy + 1e-9 * max(1.0, self.energy):
                    continue
                d = tuple(v * sqrt(self.energy / energy) for v in d)
            value = dot(q, d)
            if value > best + 1e-12:
                best, coef = value, d
        return best, coef


def stage_two(model, history, energy):
    return Support(((vector(model.p(*x, "a")), n)
                    for x, n in history.items()), energy)


def plan_bonus(model, histories, energy):
    """Exact backward induction for b0 + b1 + b2 + b3."""
    root = [min(W, sqrt(energy / n)) if n else W for n in histories[:2]]
    second = stage_two(model, histories[2], energy)
    third = Support(((vector(model.p(*x)), n)
                     for x, n in histories[3].items()), energy)
    second_queries = {x: second(vector(model.p(*x, "a"))) for x in PREFIXES}
    mode_choices = {}
    route_values = {}
    for x in PREFIXES:
        # Fixed tie convention prefers b and then L.
        terminal = {a: third(vector(model.p(*x, a)))[0] for a in ("b", "a")}
        mode_choices[x] = max(terminal, key=lambda a: terminal[a])
        route_values[x] = second_queries[x][0] + terminal[mode_choices[x]]
    routes = tuple(max(ROUTES, key=lambda r: route_values[c, r]) for c in CUES)
    policy = Policy(routes, tuple(mode_choices[x] for x in PREFIXES))
    b = sum(root) + 0.5 * sum(route_values[c, r] for c, r in zip(CUES, routes))
    return b, Design.policy(policy), second_queries


def training_designs(policy):
    # Init and stage 1 use the empty current core. Stage 2 uses a, and
    # stage 3 observes y. Completion picks L then b wherever needed.
    return [
        mix(Design()),
        mix(Design(), Design((0.5, 0.5), (1.0,) * 4)),
        mix(Design(policy.routes, (1.0,) * 4), Design(policy.routes, (0.5,) * 4)),
        mix(policy, policy),
    ]


def diagnostic(policy, queries):
    """Logged rho. Stage-two u=w_a=d/2, w_b=0 realizes the projected d."""
    # Stage-one optimizers split the constant objective evenly between the
    # current-core and transition blocks, directing the latter to L.
    # Stage-three diagnostics alter only the duplicate terminal action.
    stage_two_w = tuple(max(map(abs, queries[x][1])) / W for x in PREFIXES)
    return [(1 / 3, policy), (1 / 9, Design()),
            (1 / 9, Design(FIXED_ROUTES, (1.0,) * 4)),
            (1 / 9, Design(policy.routes, (1.0,) * 4)),
            (1 / 9, Design(policy.routes, stage_two_w)),
            (1 / 9, policy), (1 / 9, policy)]


def nested(rho):
    """Random cut, keeping the component label when it has been drawn."""
    components = [(1 / 3, Design())]  # E0 starts before the rho label.
    for weight, design in rho:
        components += [(weight / 3, Design((0.5, 0.5), (1.0,) * 4)),
                       (weight / 3, Design(design.routes, (0.5,) * 4))]
    return components


PREFIX_INDEX = {
    2: tuple(tuple(i for i, e in enumerate(EPISODES) if (e.cue, e.route) == x)
             for x in PREFIXES),
    3: tuple(tuple(i for i, e in enumerate(EPISODES) if (e.cue, e.route, e.mode) == x)
             for x in product(CUES, ROUTES, MODES)),
}


def conditional_risk(p, q, observed, stage):
    """Sum of complete logged suffix Hellinger risks at the sampled prefixes."""
    if stage not in PREFIX_INDEX:
        return int(observed.sum()) * hellinger(p, q)
    risk = 0.0
    for indices in PREFIX_INDEX[stage]:
        sampled = int(observed[:, indices].sum())
        if not sampled:
            continue
        p_sub, q_sub = p[:, indices], q[:, indices]
        risk += sampled * hellinger(p_sub / p_sub.sum(), q_sub / q_sub.sum())
    return risk


@dataclass(frozen=True)
class Budget:
    epsilon: float
    delta: float = 0.1
    candidates: int = 2
    horizon: int = H
    score_bound: float = W
    risk_scale: float = Y
    dimension: int = D

    @property
    def stages_with_initial(self):
        return self.horizon + 1

    @property
    def beta(self):
        return max(1.0, log(3 * self.candidates / self.delta))

    @property
    def energy(self):
        return 8 * self.risk_scale * self.beta

    @property
    def tau(self):
        return self.epsilon / 8

    @property
    def eta(self):
        return self.tau / (6 * self.horizon * self.score_bound)

    @property
    def historical_energy(self):
        return 2 * self.stages_with_initial * self.energy + 16 * self.risk_scale * self.beta

    def theory_training(self, round_number):
        # c=64 gives mean clipped leverage >=16 log(64 L (k+1)^2 / delta).
        # Chernoff at half the mean gives a summable tail within delta/3.
        c = 64
        stages = self.stages_with_initial
        scale = max(2 * self.historical_energy, (stages * self.score_bound) ** 2)
        return ceil(c * scale / self.tau ** 2 * log(c * stages * (round_number + 1) ** 2 / self.delta))

    def theory_nested(self):
        return ceil(128 * 24 * self.risk_scale * self.stages_with_initial * self.beta / self.eta ** 2)

    def cap(self, rank_bound=None):
        width = self.dimension if rank_bound is None else rank_bound
        stages = self.stages_with_initial
        scale = width ** 2 * stages * self.score_bound ** 2
        k = 1
        while k * log(2) <= width * stages * (
                log1p(k * self.theory_training(k) * scale / self.historical_energy)
                + log1p(128 * k * scale / self.eta ** 2)):
            k *= 2
        return k


REWARDS = {
    "outcome": lambda e: e.outcome,
    "negative_target": lambda e: float(e.cue == "-" and e.outcome == 1 and e.mode == "a"),
    "safe_processing": lambda e: (0.5 if e.mode == "b" else e.outcome) if e.cue == "-" else 0,
    "failure": lambda e: 1 - e.outcome,
}


def fit_mle(log_likelihoods, counts):
    """Unconstrained maximum likelihood with a fixed candidate tie rule."""
    observed = counts > 0
    scores = log_likelihoods[:, observed] @ counts[observed]
    return int(scores.argmax())


def run_algorithm(models, truth_index, epsilon, seed, condition="nested",
                  theory=False, scale=1.0, fixed_budget=None, *, fixed_training=1,
                  truth_model=None):
    """Run Algorithm 1 using exact multinomial draws for logged batch counts."""
    budget = Budget(epsilon, candidates=len(models))
    rng = np.random.default_rng(seed)
    truth = models[truth_index] if truth_model is None else truth_model
    models_obs = np.stack([model.observation_weights() for model in models])
    with np.errstate(divide="ignore"):
        models_log = np.log(models_obs)
    episodes = targeted = 0
    counts = np.zeros(len(EPISODES), dtype=np.int64)
    risks = np.zeros(len(models))
    histories = [0, 0, dict.fromkeys(PREFIXES, 0),
                 dict.fromkeys(product(CUES, ROUTES, MODES), 0)]
    results = []
    limit = budget.cap()
    if fixed_budget is not None:
        total, step = fixed_budget
        if theory or total < 0 or fixed_training < 1 or step <= L * fixed_training or total % step:
            raise ValueError("fixed budget needs a positive multiple of a complete block")
        limit = 1 + total // step
    for k in range(1, limit + 1):
        choice = fit_mle(models_log, counts)
        fit = models[choice]
        b, policy, queries = plan_bonus(fit, histories, budget.energy)
        stopped = b <= 3 * epsilon / 16
        tv = all_policy_tv(fit, truth)
        rho = diagnostic(policy, queries)
        cuts = nested(rho)
        row = dict(round=k, episodes=episodes, fit=choice, bonus=b, policy_tv=tv,
                   stopped=stopped, calibrated=bool(risks[choice] <= budget.beta + 1e-10),
                   conditional_risk=float(risks[choice]), beta=budget.beta,
                   targeted=targeted, mle_calls=k, exploration_plans=k,
                   task_plans=len(REWARDS) if stopped else 0,
                   diagnostic_tv=distance(law(truth, rho), law(fit, rho)),
                   nested_hellinger=hellinger(law(truth, cuts), law(fit, cuts)))
        if stopped:
            row.update({key: task_value(truth, truth, reward) - task_value(fit, truth, reward)
                        for key, reward in REWARDS.items()})
        results.append(row)
        if (stopped and fixed_budget is None) or k == limit:
            break

        m = budget.theory_training(k) if theory else ceil(scale * 2_000 / epsilon ** 2)
        n = budget.theory_nested() if theory else m
        if fixed_budget is not None:
            m, n = fixed_training, step - L * fixed_training
        extra = cuts if condition == "nested" else rho
        for stage, components in enumerate([*training_designs(policy), extra]):
            p = law(truth, components)
            size = m if stage < L else n
            observed = rng.multinomial(size, (p / p.sum()).ravel()).reshape(p.shape)
            leaves = observed.sum(axis=0)
            counts += leaves
            episodes += size
            targeted += sum(int(leaves[i]) for i, e in enumerate(EPISODES)
                            if (e.cue, e.route, e.mode) == ("-", "R", "a"))
            for i, model in enumerate(models):
                if truth_model is not None or i != truth_index:
                    risks[i] += conditional_risk(p, law(model, components), observed, stage)
            if stage < 2:
                histories[stage] += size
            elif stage < L:
                for x, indices in zip(histories[stage], PREFIX_INDEX[stage]):
                    histories[stage][x] += int(leaves[list(indices)].sum())

    return results
