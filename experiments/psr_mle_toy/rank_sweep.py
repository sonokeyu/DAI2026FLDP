"""Algorithm 1 on deterministic observable Markov chains with fixed D."""
from itertools import product
from math import ceil

import numpy as np

from routing import Budget

DIMENSION = 8
CANDIDATE_RANKS = (1, 2, 4)


def transition(rank):
    states = np.arange(DIMENSION)
    return (states[:, None] + np.arange(2)) % rank


def reachable_ranks(model, horizon):
    states = {0}
    ranks = []
    for _ in range(horizon):
        ranks.append(len(states))
        states = {int(model[s, a]) for s, a in product(states, range(2))}
    return ranks


def optimal_bonus(model, bonus):
    horizon = len(bonus)
    policy = np.zeros_like(bonus, dtype=int)
    value = bonus[-1]
    for h in range(horizon - 2, -1, -1):
        future = value[model]
        policy[h] = np.argmax(future, axis=1)
        value = bonus[h] + future.max(axis=1)
    return float(value[0]), policy


def component(horizon, policy, stage=None, action=None):
    """Probabilities of action 1, before and after a designated stage."""
    if stage is None:
        return policy
    roll_in = np.zeros_like(policy, dtype=float)
    roll_in[:stage] = policy[:stage]
    roll_in[stage] = action
    return roll_in


def collect(model, action_probs, count, rng, observed, buffer=None, stage=None):
    """Draw Markov path counts by binomial splits, retaining transition data."""
    n_states = model.shape[0]
    visits = np.zeros(n_states, dtype=np.int64)
    visits[0] = count
    for h, probs in enumerate(action_probs[:-1]):
        advances = rng.binomial(visits, probs)
        actions = np.stack((visits - advances, advances), axis=1)
        sampled = actions > 0
        observed[sampled] = model[sampled]
        if stage == h + 1:
            buffer[h] += actions
        visits.fill(0)
        np.add.at(visits, model.ravel(), actions.ravel())


def run_rank_case(horizon, truth_rank, epsilon, seed, *, theory=False):
    models = [transition(r) for r in CANDIDATE_RANKS]
    # Tie toward a larger model. Every candidate is given on every run.
    models.reverse()
    truth = transition(truth_rank)
    rng = np.random.default_rng(seed)
    observed = np.full_like(truth, -1)
    buffer = np.zeros((horizon - 1, *truth.shape), dtype=np.int64)
    # Observable singleton core tests have Lambda_B=1, U=1, and A=2.
    budget = Budget(epsilon, candidates=len(models), horizon=horizon,
                    score_bound=1.0, risk_scale=2.0, dimension=DIMENSION)
    energy = budget.energy
    cap = budget.cap() if theory else 128
    training_count = episodes = 0
    for k in range(1, cap + 1):
        m = budget.theory_training(k) if theory else ceil(16_000 * (horizon + 1) / epsilon ** 2)
        n = budget.theory_nested() if theory else m
        sampled = observed >= 0
        fit = next(model for model in models if (model[sampled] == observed[sampled]).all())
        frequencies = np.zeros((horizon, DIMENSION), dtype=np.int64)
        frequencies[0, 0] = training_count
        for h, samples in enumerate(buffer, start=1):
            np.add.at(frequencies[h], fit.ravel(), samples.ravel())
        bonus = np.minimum(1.0, np.sqrt(energy / np.maximum(1, frequencies)))
        b, policy = optimal_bonus(fit, bonus)
        b += bonus[0, 0]  # Separate initial-core certificate.
        if b <= 3 * epsilon / 16 or k == cap:
            reachable = {0}
            tv = 0.0
            for _ in range(horizon - 1):
                pairs = list(product(reachable, range(2)))
                if any(fit[s, a] != truth[s, a] for s, a in pairs):
                    tv = 1.0
                    break
                reachable = {int(fit[s, a]) for s, a in pairs}
            result = dict(horizon=horizon, rank=max(reachable_ranks(truth, horizon)),
                          dimension=DIMENSION, epsilon=epsilon, batch=m, seed=seed,
                          stopped=bool(b <= 3 * epsilon / 16), bonus=b, policy_tv=tv,
                          episodes=episodes, mle_calls=k, exploration_plans=k)
            if theory:
                # True rank is measured here, after all learning decisions.
                result.pop("batch")
                result.update(known_dimension_cap=cap, true_rank_cap=budget.cap(result["rank"]))
            return result

        # Each current-core test just observes the current state. Its
        # completion uses action 0. The other branch uses a uniform action.
        fixed = np.zeros_like(policy)
        collect(truth, fixed, m, rng, observed)
        for h in range(horizon):
            n_core = rng.binomial(m, 0.5)
            collect(truth, component(horizon, policy, h, 0.0), n_core,
                    rng, observed, buffer, h)
            collect(truth, component(horizon, policy, h, 0.5), m - n_core,
                    rng, observed, buffer, h)

        # Build rho. Split g=d_s between u and the coefficient for action 0.
        # Its diagnostic chooses action 0 with all the mass (including spare).
        rho = [policy, *(component(horizon, policy, h, 0.0)
                         for h in range(horizon) for _ in range(2))]
        weights = [1 / 3, *[1 / (3 * horizon)] * (2 * horizon)]
        cuts = [fixed, *(component(horizon, design, h, 0.5)
                         for h in range(horizon - 1) for design in rho)]
        sizes = rng.multinomial(n, [1 / horizon, *[w / horizon
                                 for _ in range(horizon - 1) for w in weights]])
        for design, size in zip(cuts, sizes):
            collect(truth, design, size, rng, observed)
        training_count += m
        episodes += (horizon + 1) * m + n
