"""Equal-episode routing comparisons, including finite-class reward-free OMLE."""
from itertools import combinations, product
from math import log

import numpy as np

from routing import (
    CUES, EPISODES, MODES, PREFIXES, ROUTES, Design, all_policy_tv, law, mix,
    run_algorithm, stage_two, vector,
)

METHODS = ("bonus_test", "random_test", "bonus_no_test", "fitted_psr", "reward_free_omle")


def disagreement_policy(models):
    if len(models) < 2:
        return Design()
    first, second = max(combinations(models, 2), key=lambda pair: all_policy_tv(*pair))
    routes = []
    modes = [0.0] * len(PREFIXES)
    for cue in CUES:
        route, mode = max(product(ROUTES, MODES),
                          key=lambda ra: abs(first.p(cue, *ra) - second.p(cue, *ra)))
        routes.append(float(route == "R"))
        modes[PREFIXES.index((cue, route))] = float(mode == "a")
    return Design(tuple(routes), tuple(modes))


def compare(models, truth_index, method, seed, episodes=256, batch=8):
    """Learn on a fixed budget, with results at multiples of batch."""
    if method not in METHODS or batch < 2 or batch % 2 or episodes % batch:
        raise ValueError("use a known method and an even batch dividing episodes")
    if method == "fitted_psr":
        return [dict(episodes=row["episodes"], fit=row["fit"], policy_tv=row["policy_tv"],
                     updates=row["mle_calls"] - 1, candidates=len(models),
                     targeted=row["targeted"])
                for row in run_algorithm(models, truth_index, 1.0, seed,
                                         fixed_budget=(episodes, batch))]
    truth = models[truth_index]
    rng = np.random.default_rng(seed)
    history = dict.fromkeys(PREFIXES, 0)
    counts = np.zeros(len(EPISODES), dtype=np.int64)
    with np.errstate(divide="ignore"):
        log_models = np.log(np.stack([model.observation_weights() for model in models]))
    present = np.ones(len(models), dtype=bool)
    # Explicit toy choice of the universal multiplier in Algorithm 2.
    slack = log(episodes * len(models) / 0.1)
    results = []
    updates = 0
    for n in range(0, episodes + 1, batch):
        for offset in range(0, batch, 2 if method == "reward_free_omle" else batch):
            occupied = counts > 0
            scores = log_models[:, occupied] @ counts[occupied]
            if method == "reward_free_omle":
                present &= scores >= scores.max() - slack
                if not present.any():
                    raise RuntimeError("OMLE confidence set is empty")
            else:
                present[:] = True
            fit = int(np.argmax(np.where(present, scores, -np.inf)))
            if offset == 0:
                results.append(dict(
                    episodes=n, fit=fit, policy_tv=all_policy_tv(truth, models[fit]),
                    updates=updates, candidates=int(present.sum()),
                    targeted=sum(int(counts[i]) for i, e in enumerate(EPISODES)
                                 if (e.cue, e.route, e.mode) == ("-", "R", "a")),
                ))
            if n == episodes:
                return results
            if method == "reward_free_omle":
                policy = disagreement_policy([model for model, active in zip(models, present) if active])
                # Eq. (10): one episode for each of the two stage cuts.
                # The h=1 cut forces the stage-two core, action a.
                for design in (Design((0.5, 0.5), (1.0,) * 4),
                               Design(policy.routes, (0.5,) * 4)):
                    p = law(truth, mix(design)).ravel()
                    counts[rng.choice(len(p), p=p)] += 1
            else:
                if method == "random_test":
                    routes = (0.5, 0.5)
                else:
                    model = models[fit]
                    support = stage_two(model, history, energy=1.0)
                    routes = tuple(float(support(vector(model.p(cue, "R", "a")))[0]
                                         > support(vector(model.p(cue, "L", "a")))[0] + 1e-12)
                                   for cue in CUES)
                test = 0.0 if method == "bonus_no_test" else 0.5
                # Shared uniforms pair the control arms, including when they
                # choose different actions and observe different outcomes.
                for u, v, w, z in rng.random((batch, 4)):
                    cue_i = int(u < 0.5)
                    route_i = int(v < routes[cue_i])
                    mode_i = int(w >= test)
                    cue, route, mode = CUES[cue_i], ROUTES[route_i], MODES[mode_i]
                    y = int(z < truth.p(cue, route, mode))
                    counts[((cue_i * 2 + route_i) * 2 + mode_i) * 2 + y] += 1
                    history[cue, route] += 1
            updates += 1
    return results
