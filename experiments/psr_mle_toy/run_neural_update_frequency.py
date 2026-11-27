"""Neural MLE update-frequency ablation on the 256-state, H=8 POMDP.

The exploration planner maximizes single-model one-step predictive entropy.
It is an empirical neural stress test, not the paper's PSR bonus certificate.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from neural_pomdp_stress import (HORIZON, N_ACTIONS, N_STATES, NeuralPSR,
                                 episode, evaluate, make_data, observe,
                                 transition)


def collect(n, rng, action_table=None):
    observations, actions = np.empty((n, HORIZON), np.int64), np.zeros((n, HORIZON), np.int64)
    for i in range(n):
        state = int(rng.integers(N_STATES))
        for h in range(HORIZON):
            obs = observe(state, rng)
            observations[i, h] = obs
            if h < HORIZON - 1:
                action = (int(rng.integers(N_ACTIONS)) if action_table is None or rng.random() < .5
                          else int(action_table[obs]))
                actions[i, h] = action
                state = transition(state, action, rng)
    return observations, actions


@torch.no_grad()
def plan_entropy(model):
    """One-model planning rule: choose the action of largest predicted entropy at each observation."""
    model.eval()
    obs = torch.eye(N_STATES).repeat_interleave(N_ACTIONS, dim=0)
    actions = torch.eye(N_ACTIONS).repeat(N_STATES, 1)
    logits = model(obs[:, None], actions[:, None])[:, 0]
    logp = logits.log_softmax(-1)
    entropy = -(logp.exp() * logp).sum(-1).reshape(N_STATES, N_ACTIONS)
    return entropy.argmax(-1).cpu().numpy()


def train_steps(model, optimizer, data, steps, rng):
    obs, actions = data
    o = torch.as_tensor(obs, dtype=torch.long)
    a = torch.as_tensor(actions, dtype=torch.long)
    for _ in range(steps):
        ids = torch.as_tensor(rng.integers(len(o), size=256), dtype=torch.long)
        x = nn.functional.one_hot(o[ids, :-1], N_STATES).float()
        aa = nn.functional.one_hot(a[ids, :-1], N_ACTIONS).float()
        y = o[ids, 1:]
        logits = model(x, aa)
        loss = nn.functional.cross_entropy(logits.reshape(-1, N_STATES), y.reshape(-1))
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()


def run_one(seed, calls, mode, budget, gradient_steps, tests):
    torch.manual_seed(3100 + seed)
    rng = np.random.default_rng(5200 + seed)
    model = NeuralPSR(hidden=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    initial = min(300, budget // 10)
    obs, actions = collect(initial, rng)
    remaining = budget - initial
    q, rem = divmod(remaining, calls - 1)
    increments = [q + int(i < rem) for i in range(calls - 1)]
    initial_steps = min(60, gradient_steps // 2)
    step_q, step_rem = divmod(gradient_steps - initial_steps, calls - 1)
    rows = []
    frozen_table = None
    for i in range(calls):
        steps_now = (initial_steps if i == 0 else step_q + int(i - 1 < step_rem))
        train_steps(model, optimizer, (obs, actions), steps_now, rng)
        policy_table = (plan_entropy(model) if mode == 'adaptive' or frozen_table is None
                        else frozen_table)
        if frozen_table is None:
            frozen_table = policy_table.copy()
        if i + 1 == calls:
            for split, test in tests.items():
                nll, accuracy = evaluate(model, test)
                rows.append(dict(seed=seed, mode=mode, requested_mle_calls=calls,
                                 mle_calls=i+1, planning_calls=i+1 if mode == 'adaptive' else 1,
                                 episodes=len(obs), gradient_steps=gradient_steps,
                                 test=split, nll=nll, accuracy=accuracy))
        else:
            chosen = policy_table if mode == 'adaptive' else frozen_table
            new_obs, new_actions = collect(increments[i], rng, chosen)
            obs = np.concatenate((obs, new_obs))
            actions = np.concatenate((actions, new_actions))
    assert len(obs) == budget
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--budget', type=int, default=6000)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--gradient-steps', type=int, default=120)
    p.add_argument('--fit-calls', type=int, nargs='+', default=[3, 5, 10, 20, 40])
    args = p.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(9100 + seed)
        tests = {'uniform': make_data(1200, rng, 'uniform'),
                 'shifted': make_data(1200, rng, 'biased')}
        for mode in ('adaptive', 'frozen'):
            for calls in args.fit_calls:
                result = run_one(seed, calls, mode, args.budget, args.gradient_steps, tests)
                rows.extend(result)
                print(f"seed={seed} mode={mode} calls={calls} NLL={result[0]['nll']:.4f}/{result[1]['nll']:.4f}", flush=True)
    with (args.output / 'neural_update_runs.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = []
    for mode in ('adaptive', 'frozen'):
        for calls in args.fit_calls:
            for split in ('uniform', 'shifted'):
                group = [r for r in rows if r['mode'] == mode and r['requested_mle_calls'] == calls and r['test'] == split]
                summary.append(dict(mode=mode, fit_calls=calls, planning_calls=group[0]['planning_calls'],
                                    episodes=args.budget, seeds=len(group), gradient_steps=args.gradient_steps,
                                    test=split, mean_nll=float(np.mean([r['nll'] for r in group])),
                                    sd_nll=float(np.std([r['nll'] for r in group]))))
    with (args.output / 'neural_update_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)


if __name__ == '__main__':
    main()
