"""Neural MLE stress test on a 256-state partially observed control task.

This is an empirical engineering benchmark, not a finite-class theorem check.
The learner is an autoregressive GRU trained by trajectory conditional MLE.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


HORIZON, N_BITS, N_ACTIONS = 8, 8, 4
N_STATES = 1 << N_BITS
DEVICE = "cpu"


def transition(state, action, rng):
    # Action-dependent nonlinear bit interactions create long-range effects.
    x = state
    rot = ((x << 1) | (x >> 7)) & 255
    mask = (action * 37 + 19) & 255
    y = x ^ mask ^ ((rot & (x >> 2)) & (action * 17 + 31))
    # State/action-dependent stochasticity prevents a trivial deterministic fit.
    for b in range(N_BITS):
        p = .025 + .11 * ((x >> ((b + action) % 8)) & 1)
        if rng.random() < p:
            y ^= 1 << b
    return y & 255


def observe(state, rng):
    obs = state
    for b in range(N_BITS):
        if rng.random() < .13:
            obs ^= 1 << b
    return obs


def episode(actions, rng):
    state = int(rng.integers(N_STATES))
    observations = []
    for h in range(HORIZON):
        observations.append(observe(state, rng))
        if h + 1 < HORIZON:
            state = transition(state, int(actions[h]), rng)
    return observations


def onehot(x, n):
    return np.eye(n, dtype=np.float32)[x]


class NeuralPSR(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.embed = nn.Linear(N_STATES + N_ACTIONS, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, N_STATES)

    def forward(self, obs, acts):
        # Each row predicts O_{t+1}; actions are those taken after O_t.
        x = torch.cat([obs, acts], dim=-1)
        z, _ = self.rnn(torch.tanh(self.embed(x)))
        return self.head(z)


def make_data(n, rng, policy="uniform"):
    obs, acts = [], []
    for _ in range(n):
        if policy == "biased":
            seq = rng.choice(N_ACTIONS, HORIZON - 1, p=[.7, .1, .1, .1])
        else:
            seq = rng.integers(N_ACTIONS, size=HORIZON - 1)
        traj = episode(seq, rng)
        obs.append(traj)
        acts.append(np.r_[seq, 0])
    o = onehot(np.asarray(obs), N_STATES)
    a = onehot(np.asarray(acts), N_ACTIONS)
    return o, a


def fit_mle(model, data, seed, epochs=12):
    torch.manual_seed(seed)
    obs, acts = [torch.tensor(x, device=DEVICE) for x in data]
    x, y = obs[:, :-1], obs[:, 1:].argmax(-1)
    aa = acts[:, :-1]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    n = len(x)
    for _ in range(epochs):
        order = torch.randperm(n)
        model.train()
        for ids in order.split(256):
            logits = model(x[ids], aa[ids])
            loss = nn.functional.cross_entropy(logits.reshape(-1, N_STATES), y[ids].reshape(-1))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step()
    return model


@torch.no_grad()
def evaluate(model, data):
    obs, acts = [torch.tensor(x, device=DEVICE) for x in data]
    model.eval()
    logits = model(obs[:, :-1], acts[:, :-1])
    y = obs[:, 1:].argmax(-1)
    loss = nn.functional.cross_entropy(logits.reshape(-1, N_STATES), y.reshape(-1)).item()
    acc = (logits.argmax(-1) == y).float().mean().item()
    return loss, acc


def policy_score(model, first_obs, sequences, n_rollouts, seed):
    """Evaluate action sequences by autoregressive model rollout; reward targets hidden bits via noisy final observation."""
    rng = np.random.default_rng(seed)
    obs0 = torch.tensor(onehot(np.full(n_rollouts, first_obs), N_STATES), device=DEVICE)
    all_scores = []
    model.eval()
    for seq in sequences:
        current = obs0.clone()
        actions = torch.zeros((n_rollouts, HORIZON - 1, N_ACTIONS), device=DEVICE)
        actions[:] = torch.tensor(onehot(np.asarray(seq), N_ACTIONS), device=DEVICE)
        # autoregress on sampled observations, retaining only observable history.
        hist = [current]
        for t in range(HORIZON - 1):
            past_o = torch.stack(hist, dim=1)
            past_a = actions[:, :t+1]
            logits = model(past_o, past_a)[:, -1]
            dist = torch.distributions.Categorical(logits=logits)
            current = nn.functional.one_hot(dist.sample(), N_STATES).float()
            hist.append(current)
        final = current.argmax(-1).cpu().numpy()
        # Hidden target is a nonlinear parity-like reward, noisy observation gives unbiased score proxy.
        reward = np.mean([((int(x) ^ 0xA5).bit_count() <= 2) for x in final])
        all_scores.append(reward)
    return int(np.argmax(all_scores)), all_scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--episodes", type=int, nargs="+", default=[2000, 10000, 40000])
    p.add_argument("--epochs", type=int, default=12)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(82000 + seed)
        test_uniform = make_data(5000, rng, "uniform")
        test_biased = make_data(3000, rng, "biased")
        for budget in args.episodes:
            train = make_data(budget, rng, "uniform")
            model = fit_mle(NeuralPSR(), train, seed, args.epochs)
            for split, data in [("uniform_test", test_uniform), ("shifted_policy_test", test_biased)]:
                nll, acc = evaluate(model, data)
                rows.append(dict(seed=seed, episodes=budget, mle_fit_calls=1,
                                 planning_calls=0, evaluation=split, nll=nll, accuracy=acc,
                                 states=N_STATES, observations=N_STATES, actions=N_ACTIONS,
                                 horizon=HORIZON, parameters=sum(x.numel() for x in model.parameters())))
            print(f"seed={seed} episodes={budget} uniform_nll={rows[-2]['nll']:.3f} shifted_nll={rows[-1]['nll']:.3f}", flush=True)
    with (args.output / "neural_runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {}
    for budget in args.episodes:
        summary[str(budget)] = {}
        for split in ("uniform_test", "shifted_policy_test"):
            vals = [r["nll"] for r in rows if r["episodes"] == budget and r["evaluation"] == split]
            summary[str(budget)][split] = {"mean_nll": float(np.mean(vals)), "sd_nll": float(np.std(vals))}
    (args.output / "neural_summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "README.md").write_text(
        "# Neural POMDP stress test\n\n"
        f"Latent state space: {N_STATES}; horizon: {HORIZON}; actions: {N_ACTIONS}. "
        "An autoregressive GRU is fit by trajectory conditional MLE. Evaluation reports "
        "next-observation NLL on held-out uniform and action-shifted policies. This is an "
        "empirical prediction stress test; it does not compute the paper's exact all-policy TV bonus certificate.\n")


if __name__ == "__main__":
    main()
