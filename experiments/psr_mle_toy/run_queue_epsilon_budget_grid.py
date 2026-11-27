"""Cross epsilon with episode caps on the same finite, realizable PSR task.

The ours source trajectories use the epsilon-independent geometric batch rule
from run_queue_natural_budget.py with epsilon=0 (no early stop). For any
epsilon, we select the first pre-cap round whose bonus meets 3*epsilon/16.
Since epsilon does not affect that batch rule or its exploration policy, this
is exactly the result of running that empirical rule with early stopping.

OMLE uses the existing full epsilon runs once they stop below a cap. For the
6000-episode cap, longer OMLE paths are rerun with the SAME confidence slack
as their corresponding full runs, and stopped at the cap.
"""
from argparse import ArgumentParser
import csv
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from queue_mle_class import fine_candidate_class
from queue_pomdp import TRUE_WEIGHTS
from run_full_fine_psr import run_omle
from run_multistage_experiment import write_csv
from run_queue_ours import worst_policy_tv


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def as_bool(value):
    return value in (True, 'True', 'true', '1')


def load_ours(raw_paths):
    groups = {}
    for raw_path in raw_paths:
        for row in read_csv(raw_path):
            groups.setdefault((int(row['budget']), int(row['seed'])), []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: int(r['mle_calls']))
        if not all(as_bool(r['certificate_valid']) for r in rows):
            raise ValueError('bonus failed to upper-bound exact TV')
    return groups


def omle_pairs():
    raw = fine_candidate_class()
    truth = next(i for i, m in enumerate(raw) if m.weights == tuple(TRUE_WEIGHTS))
    first = max((i for i in range(len(raw)) if i != truth),
                key=lambda i: worst_policy_tv(raw[truth], raw[i])[0])
    order = [first] + [i for i in range(len(raw)) if i != first]
    models = tuple(raw[i] for i in order)
    keys = list(combinations(range(len(models)), 2))
    pair_data = [worst_policy_tv(models[i], models[j]) for i, j in keys]
    return (models, order.index(truth),
            (np.asarray([i for i, j in keys]),
             np.asarray([j for i, j in keys]),
             np.asarray([tv for tv, policy in pair_data]),
             [policy for tv, policy in pair_data]))


def summarize(rows, budgets, epsilons, seeds):
    summary = []
    for epsilon in epsilons:
        for budget in budgets:
            for method in ('ours', 'omle'):
                group = [r for r in rows if r['epsilon'] == epsilon
                         and r['budget'] == budget and r['method'] == method]
                if len(group) != seeds:
                    raise ValueError(f'missing runs for {epsilon}, {budget}, {method}')
                calls = [r['calls'] for r in group]
                tvs = [r['sup_tv'] for r in group]
                summary.append(dict(method=method, epsilon=epsilon,
                                    inverse_epsilon=1/epsilon, budget=budget,
                                    seeds=seeds,
                                    median_calls=float(np.median(calls)),
                                    min_calls=min(calls), max_calls=max(calls),
                                    median_episodes=float(np.median([r['episodes'] for r in group])),
                                    median_sup_tv=float(np.median(tvs)),
                                    max_sup_tv=max(tvs),
                                    median_bonus_or_diameter=float(np.median(
                                        [r['bonus_or_diameter'] for r in group])),
                                    stopped=sum(r['stopped'] for r in group),
                                    target_passes=sum(r['target_passed'] for r in group),
                                    audit_passes=sum(r['audit_passed'] for r in group)))
    return summary


def draw_grid(summary, budgets, epsilons, output):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
    for col, method in enumerate(('ours', 'omle')):
        matrix = np.asarray([[next(r['median_calls'] for r in summary
                                    if r['method'] == method and r['epsilon'] == eps
                                    and r['budget'] == budget)
                              for budget in budgets] for eps in epsilons])
        stopped = np.asarray([[next(r['stopped'] for r in summary
                                     if r['method'] == method and r['epsilon'] == eps
                                     and r['budget'] == budget)
                               for budget in budgets] for eps in epsilons])
        ax = axes[0, col]
        image = ax.imshow(np.log10(matrix), cmap='viridis', aspect='auto',
                          vmin=0, vmax=6)
        for i in range(len(epsilons)):
            for j in range(len(budgets)):
                color = 'white' if np.log10(matrix[i, j]) < 3 else 'black'
                ax.text(j, i, f'{matrix[i,j]:,.0f}', ha='center', va='center',
                        color=color, fontsize=8)
        ax.set_title(f'{method}: median oracle calls (log color)')
        fig.colorbar(image, ax=ax, fraction=.025)
        ax = axes[1, col]
        image = ax.imshow(stopped, cmap='Blues', aspect='auto', vmin=0, vmax=3)
        for i in range(len(epsilons)):
            for j in range(len(budgets)):
                ax.text(j, i, f'{stopped[i,j]}/3', ha='center', va='center',
                        color='white' if stopped[i,j] >= 2 else 'black')
        ax.set_title(f'{method}: certified stops (out of 3)')
        fig.colorbar(image, ax=ax, fraction=.025)
    def budget_label(n):
        if n < 1_000_000:
            return f'{n/1_000:g}k'
        if n < 1_000_000_000:
            return f'{n/1_000_000:g}m'
        return f'{n/1_000_000_000:g}b'
    labels = [budget_label(n) for n in budgets]
    for ax in axes.flat:
        ax.set_xticks(range(len(budgets)), labels, rotation=35)
        ax.set_yticks(range(len(epsilons)), [f'{1/e:g}' for e in epsilons])
        ax.set_xlabel('episode cap')
        ax.set_ylabel('1 / epsilon')
    fig.savefig(output.with_suffix('.png'), dpi=180)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--ours-raw', type=Path, nargs='+', required=True)
    parser.add_argument('--omle-full', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epsilons', nargs='+', type=float,
                        default=[.4, .2, .1, .05, .025])
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ours = load_ours(args.ours_raw)
    budgets = sorted({budget for budget, seed in ours})
    full = {(float(r['epsilon']), int(r['seed'])): r
            for r in read_csv(args.omle_full)}
    rows = []
    pairs = None
    for epsilon in args.epsilons:
        for seed in range(args.seeds):
            endpoint = full[(epsilon, seed)]
            if not as_bool(endpoint['stopped']):
                raise ValueError(f'OMLE full run did not stop: epsilon={epsilon}, seed={seed}')
            for budget in budgets:
                path = ours[(budget, seed)]
                chosen = next((r for r in path
                               if float(r['bonus']) <= 3*epsilon/16 + 1e-12),
                              path[-1])
                ours_stopped = float(chosen['bonus']) <= 3*epsilon/16 + 1e-12
                ours_tv = float(chosen['sup_tv'])
                rows.append(dict(method='ours', epsilon=epsilon,
                                 inverse_epsilon=1/epsilon, budget=budget, seed=seed,
                                 episodes=int(chosen['episodes']),
                                 calls=int(chosen['mle_calls']),
                                 planning_calls=int(chosen['planning_calls']),
                                 sup_tv=ours_tv,
                                 bonus_or_diameter=float(chosen['bonus']),
                                 stopped=ours_stopped,
                                 target_passed=ours_tv <= epsilon/4 + 1e-12,
                                 audit_passed=as_bool(chosen['certificate_valid']),
                                 source='natural_budget_rounds.csv'))
                if int(endpoint['episodes']) <= budget:
                    omle = endpoint
                    source = 'omle_finals.csv'
                else:
                    if pairs is None:
                        print('Precomputing OMLE model-pair planning table...', flush=True)
                        pairs = omle_pairs()
                    cap = 500000 if epsilon <= .05 else 60000
                    omle = run_omle(*pairs, epsilon, seed, cap,
                                    max_rounds=budget//3)
                    source = 'capped_omle_rerun'
                omle_tv = float(omle['fitted_tv'])
                omle_diameter = float(omle['diameter'])
                omle_stopped = as_bool(omle['stopped'])
                rows.append(dict(method='omle', epsilon=epsilon,
                                 inverse_epsilon=1/epsilon, budget=budget, seed=seed,
                                 episodes=int(omle['episodes']),
                                 calls=int(omle['joint_search_calls']),
                                 planning_calls='', sup_tv=omle_tv,
                                 bonus_or_diameter=omle_diameter,
                                 stopped=omle_stopped,
                                 target_passed=omle_tv <= epsilon/4 + 1e-12,
                                 audit_passed=as_bool(omle['truth_covered'])
                                 and (not omle_stopped or omle_tv <= epsilon/4 + 1e-12),
                                 source=source))
                print(f'eps={epsilon:g} seed={seed} N={budget} '
                      f'ours={rows[-2]["calls"]}/{rows[-2]["stopped"]} '
                      f'omle={rows[-1]["calls"]}/{rows[-1]["stopped"]}',
                      flush=True)
    summary = summarize(rows, budgets, args.epsilons, args.seeds)
    write_csv(args.output/'grid_runs.csv', rows)
    write_csv(args.output/'grid_summary.csv', summary)
    draw_grid(summary, budgets, args.epsilons, args.output/'epsilon_budget_grid')


if __name__ == '__main__':
    main()
