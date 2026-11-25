"""Matched finite-class four-method comparison on the 64-state POMDP."""
from argparse import ArgumentParser
from collections import Counter
from itertools import combinations
from math import log, sqrt
from pathlib import Path
import random

import numpy as np

from queue_pomdp import QueueModel, TRUE_WEIGHTS, STABILITY
from queue_psr import StageSupport
from run_multistage_experiment import write_csv
from run_queue_ours import design_law, plan_bonus, worst_policy_tv


def make_class():
    truth = np.asarray(TRUE_WEIGHTS)
    models = [QueueModel(tuple(truth))]
    for index in range(12):
        for shift in (-.6, .6):
            candidate = truth.copy()
            candidate[index] = np.clip(candidate[index] + shift, .05, .95)
            models.append(QueueModel(tuple(candidate)))
    return tuple(models)


def exact_mle(models, counts):
    if not counts:
        return 0
    scores = np.zeros(len(models))
    for trace, count in counts.items():
        scores += count * np.log([model.trajectory_probability(trace)
                                  for model in models])
    return int(scores.argmax())


def baseline(models, truth_index, seed, method, budget, checkpoint):
    rng = random.Random(seed + (101 if method == "uniform" else 211))
    action_counts = {}
    data = Counter()
    rows = []
    for episode in range(1, budget + 1):
        class Policy:
            def __getitem__(self, history):
                if method == "uniform": return rng.randrange(2)
                counts = action_counts.get(history, (0, 0))
                return rng.randrange(2) if counts[0] == counts[1] else int(counts[1] < counts[0])
        trace = models[truth_index].sample(Policy(), rng)
        data[trace] += 1
        if method == "count":
            for stage in range(4):
                history, action = trace[:2*stage+1], trace[2*stage+1]
                counts = action_counts.setdefault(history, [0, 0]); counts[action] += 1
        if episode % checkpoint == 0:
            fit = exact_mle(models, data)
            rows.append(dict(method=method, seed=seed, episodes=episode,
                             mle_calls=episode//checkpoint, planning_calls=0,
                             joint_calls=0, tv=worst_policy_tv(
                                 models[truth_index], models[fit])[0], stopped=False))
    return rows


def omle(models, truth_index, seed, budget, epsilon, checkpoint):
    rng = random.Random(seed + 307)
    data = Counter()
    uniform = type("Uniform", (), {"__getitem__": lambda self, history: rng.randrange(2)})()
    pair = {(i,j): worst_policy_tv(models[i],models[j])
            for i,j in combinations(range(len(models)),2)}
    scores = np.zeros(len(models)); rows=[]; joint=0
    slack=log(budget*len(models)/.1)
    for episode in range(1,budget+1):
        active=[i for i,score in enumerate(scores) if score >= scores.max()-slack]
        disagreement,left,right=max(((pair[min(i,j),max(i,j)][0],i,j)
                                     for i,j in combinations(active,2)),default=(0.,-1,-1))
        if episode%2:
            policy=uniform
        else:
            policy=pair[min(left,right),max(left,right)][1]; joint+=1
        trace=models[truth_index].sample(policy,rng); data[trace]+=1
        scores += np.array([log(m.trajectory_probability(trace)) for m in models])
        if episode%checkpoint==0:
            fit=int(scores.argmax())
            rows.append(dict(method="omle",seed=seed,episodes=episode,mle_calls=0,
                             planning_calls=0,joint_calls=joint,
                             tv=worst_policy_tv(models[truth_index],models[fit])[0],
                             stopped=disagreement<=epsilon/4))
    return rows


def ours(models,truth_index,seed,episode_budget,batch,epsilon):
    rng=np.random.default_rng(seed+1907); data=Counter()
    histories=[Counter() for _ in range(4)]; initial=episodes=0; rows=[]
    truth=models[truth_index]; beta=log(3*len(models)/.1)
    energy=8*STABILITY**2*2*beta; fit_index=0; round_number=0
    while episodes<=episode_budget:
        fit_index=exact_mle(models,data); fit=models[fit_index]
        supports=[StageSupport(fit,h,histories[h],len(models)) for h in range(4)]
        root=min(STABILITY,sqrt(energy/initial)) if initial else STABILITY
        bonus,policy,coefficients=plan_bonus(fit,supports,root)
        tv=worst_policy_tv(truth,fit)[0]; round_number+=1
        rows.append(dict(method="ours",seed=seed,episodes=episodes,
                         mle_calls=round_number,planning_calls=round_number,
                         joint_calls=0,tv=tv,bonus=bonus,
                         stopped=bonus<=3*epsilon/16))
        if bonus<=3*epsilon/16 or episodes+6*batch>episode_budget: break
        def draw(law,size,stage=None):
            nonlocal episodes
            traces=tuple(law); probs=np.array([law[t] for t in traces]); probs/=probs.sum()
            for trace,count in zip(traces,rng.multinomial(size,probs)):
                if count:
                    data[trace]+=int(count)
                    if stage is not None: histories[stage][trace[:2*stage]]+=int(count)
            episodes+=size
        initial+=batch
        draw(design_law(truth,policy,stage=0),batch)
        for h in range(4):
            n=int(rng.binomial(batch,.5))
            draw(design_law(truth,policy,stage=h),n,h)
            draw(design_law(truth,policy,stage=h,branch="uniform"),batch-n,h)
        choices=[(None,"full",cut) for cut in range(4)]
        choices += [(h,b,cut) for h in range(4) for b in ("u","w") for cut in range(4)]
        probs=[1/12]*4+[1/48]*32
        sizes=rng.multinomial(batch,probs)
        for (h,b,cut),n in zip(choices,sizes):
            if n: draw(design_law(truth,policy,stage=h,branch=b,
                                  coefficients=coefficients,cut=cut),int(n))
    return rows


def main():
    parser=ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seeds",type=int,default=4); parser.add_argument("--budget",type=int,default=384)
    parser.add_argument("--checkpoint",type=int,default=64); parser.add_argument("--batch",type=int,default=32)
    parser.add_argument("--epsilon",type=float,default=.4); args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True); models=make_class(); truth_index=0
    all_rows=[]
    for seed in range(args.seeds):
        for method in ("uniform","count"):
            all_rows.extend(baseline(models,truth_index,seed,method,args.budget,args.checkpoint))
        all_rows.extend(omle(models,truth_index,seed,args.budget,args.epsilon,args.checkpoint))
        all_rows.extend(ours(models,truth_index,seed,args.budget,args.batch,args.epsilon))
        print(f"completed matched seed={seed}",flush=True)
    write_csv(args.output/"matched_runs.csv",all_rows)
    summary=[]
    for method in ("ours","omle","uniform","count"):
        group=[r for r in all_rows if r["method"]==method]
        finals=[max((r for r in group if r["seed"]==s),key=lambda r:r["episodes"]) for s in range(args.seeds)]
        summary.append(dict(method=method,runs=len(finals),
                            tv_at_budget=np.median([r["tv"] for r in finals]),
                            success=sum(r["tv"]<=args.epsilon/4 for r in finals),
                            median_episodes=np.median([r["episodes"] for r in finals]),
                            median_mle_calls=np.median([r["mle_calls"] for r in finals]),
                            median_planning_calls=np.median([r["planning_calls"] for r in finals]),
                            median_joint_calls=np.median([r["joint_calls"] for r in finals])))
    write_csv(args.output/"matched_summary.csv",summary)


if __name__=="__main__": main()
