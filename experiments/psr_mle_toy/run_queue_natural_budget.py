"""Equal-budget PSR exploration with data-dependent bonus stopping and no fixed fit count.

The growing batch schedule is a resource-controlled empirical schedule. It is
not the theorem's epsilon-dependent batch formula; that schedule is exercised
by run_full_fine_psr.py.
"""
from argparse import ArgumentParser
from collections import Counter
from math import log, sqrt
from pathlib import Path

import numpy as np

from queue_mle_class import fine_candidate_class, finite_mle
from queue_pomdp import STABILITY, TRUE_WEIGHTS
from queue_psr import StageSupport
from run_multistage_experiment import write_csv
from run_queue_ours import design_law, plan_bonus, worst_policy_tv


def run_case(models, truth_index, seed, budget, epsilon, first_batch, growth):
    rng = np.random.default_rng(seed + 1907)
    truth = models[truth_index]
    data = Counter()
    histories = [Counter() for _ in range(4)]
    episodes = initial_count = 0
    beta = max(1., log(3*len(models)/.1))
    energy = 8*STABILITY**2*2*beta
    rows = []

    def draw(law, size, training_stage=None):
        nonlocal episodes
        if size <= 0:
            return
        traces = tuple(law)
        p = np.asarray([law[t] for t in traces], dtype=float); p /= p.sum()
        for trace, count in zip(traces, rng.multinomial(size, p)):
            if count:
                data[trace] += int(count)
                if training_stage is not None:
                    histories[training_stage][trace[:2*training_stage]] += int(count)
        episodes += size

    call = 0
    while True:
        call += 1
        fit_index = finite_mle(models, data)
        fit = models[fit_index]
        supports = [StageSupport(fit,h,histories[h],len(models)) for h in range(4)]
        root = min(STABILITY,sqrt(energy/initial_count)) if initial_count else STABILITY
        bonus, policy, coefficients = plan_bonus(fit,supports,root)
        tv = worst_policy_tv(truth,fit)[0]
        stopped = bonus <= 3*epsilon/16
        rows.append(dict(seed=seed,budget=budget,epsilon=epsilon,round=call,
                         episodes=episodes,mle_calls=call,planning_calls=call,
                         bonus=bonus,sup_tv=tv,stop_threshold=3*epsilon/16,
                         stopped=stopped,certificate_valid=tv<=bonus+1e-8,
                         fit_index=fit_index))
        if stopped or episodes == budget:
            break
        batch = min(round(first_batch*growth**(call-1)),(budget-episodes)//6)
        if batch <= 0:
            raise RuntimeError('budget must be a multiple of six')
        initial_count += batch
        draw(design_law(truth,policy,stage=0),batch)
        for h in range(4):
            n = int(rng.binomial(batch,.5))
            draw(design_law(truth,policy,stage=h),n,h)
            draw(design_law(truth,policy,stage=h,branch='uniform'),batch-n,h)
        choices=[(None,'full',cut) for cut in range(4)]
        choices += [(h,b,cut) for h in range(4) for b in ('u','w') for cut in range(4)]
        sizes=rng.multinomial(batch,[1/12]*4+[1/48]*32)
        for (h,b,cut),n in zip(choices,sizes):
            if n: draw(design_law(truth,policy,stage=h,branch=b,
                                  coefficients=coefficients,cut=cut),int(n))
    return rows


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--budgets',type=int,nargs='+',default=[6000,12000,24000,48000,96000,192000,384000,768000,999996])
    p.add_argument('--epsilon',type=float,default=.05)
    p.add_argument('--first-batch',type=int,default=100)
    p.add_argument('--growth',type=float,default=1.7)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if any(n%6 for n in args.budgets): raise ValueError('each budget must be divisible by six')
    raw=fine_candidate_class()
    truth=next(i for i,m in enumerate(raw) if m.weights==tuple(TRUE_WEIGHTS))
    first=max((i for i in range(len(raw)) if i!=truth),
              key=lambda i:worst_policy_tv(raw[truth],raw[i])[0])
    order=[first]+[i for i in range(len(raw)) if i!=first]
    models=tuple(raw[i] for i in order); ti=order.index(truth)
    all_rows=[];finals=[]
    for budget in args.budgets:
        for seed in range(args.seeds):
            path=run_case(models,ti,seed,budget,args.epsilon,args.first_batch,args.growth)
            all_rows.extend(path);finals.append(path[-1])
            f=path[-1]
            print(f"N={budget} seed={seed} actual={f['episodes']} fits={f['mle_calls']} "
                  f"TV={f['sup_tv']:.6g} bonus={f['bonus']:.6g} stopped={f['stopped']}",flush=True)
    write_csv(args.output/'natural_budget_rounds.csv',all_rows)
    write_csv(args.output/'natural_budget_finals.csv',finals)
    summary=[]
    for budget in args.budgets:
        group=[r for r in finals if r['budget']==budget]
        summary.append(dict(budget=budget,seeds=args.seeds,
                            median_actual_episodes=float(np.median([r['episodes'] for r in group])),
                            median_mle_calls=float(np.median([r['mle_calls'] for r in group])),
                            median_planning_calls=float(np.median([r['planning_calls'] for r in group])),
                            median_tv=float(np.median([r['sup_tv'] for r in group])),
                            max_tv=max(r['sup_tv'] for r in group),
                            median_bonus=float(np.median([r['bonus'] for r in group])),
                            target_passes=sum(r['sup_tv']<=args.epsilon/4 for r in group),
                            stopped=sum(r['stopped'] for r in group),
                            certificate_valid=sum(r['certificate_valid'] for r in group)))
    write_csv(args.output/'natural_budget_summary.csv',summary)


if __name__=='__main__':main()
