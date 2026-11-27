"""Exact observable PSR special case: theorem batches, rank and potentials.

Singleton observation cores give q=e_s, U=Lambda_B=W=1, A=2.
The projected lifted coefficient ball is exactly [-1,1]^D: choose
u_s=d_s/2 and w_(s,0,next(s,0))=d_s/2. Its stage support is therefore
min(1,sqrt(S/n_s)). This is an exact reduction, not a substituted bonus.
All candidates share D and are deterministic; likelihood is 0 or -infinity.
Count sampling retains sufficient statistics for complete logged likelihoods.
"""
from argparse import ArgumentParser
from itertools import product
from math import log, log1p, sqrt
from pathlib import Path

import numpy as np

from routing import Budget
from rank_sweep import optimal_bonus
from run_multistage_experiment import write_csv

D = 16
RANKS = (1, 2, 4, 8)


def transition(rank):
    return (np.arange(D)[:, None] + np.arange(2)) % rank


def reachable(model, horizon):
    states = {0}
    result = []
    for h in range(horizon):
        result.append(sorted(states))
        states = {int(model[s, a]) for s in states for a in range(2)}
    return result


def tv(first, second, horizon):
    # A deterministic policy creates a point mass. TV is 1 iff a policy
    # can reach an observed disagreement, otherwise it is exactly zero.
    states = {0}
    for h in range(horizon-1):
        if any(first[s,a] != second[s,a] for s in states for a in range(2)):
            return 1.
        states = {int(first[s,a]) for s in states for a in range(2)}
    return 0.


def design(policy, cut=None, uniform=False):
    p = policy.astype(float).copy()
    if cut is not None:
        p[cut:] = 0.
        if uniform:
            p[cut] = .5
    return p


def occupancy(model, probabilities):
    horizon = len(probabilities)
    rows = np.zeros((horizon, D)); rows[0,0] = 1.
    for h in range(horizon-1):
        flow = np.stack((rows[h]*(1-probabilities[h]),
                         rows[h]*probabilities[h]), axis=1)
        np.add.at(rows[h+1], model.ravel(), flow.ravel())
    return rows


def collect(model, probabilities, count, rng, observed, buffer, stage=None):
    visits = np.zeros(D, dtype=np.int64); visits[0] = count
    for h in range(len(probabilities)-1):
        one = rng.binomial(visits, probabilities[h])
        flows = np.stack((visits-one, one), axis=1)
        observed[flows > 0] = model[flows > 0]
        if stage == h+1:
            buffer[h] += flows
        visits = np.zeros(D, dtype=np.int64)
        np.add.at(visits, model.ravel(), flows.ravel())


def frequencies(model, buffers, initial):
    result = np.zeros((len(buffers)+1,D), dtype=np.int64)
    result[0,0] = initial
    for h, rows in enumerate(buffers):
        np.add.at(result[h+1], model.ravel(), rows.ravel())
    return result


def potential(counts, root, sets, ridge):
    return log1p(root/ridge) + sum(float(np.log1p(counts[h,states]/ridge).sum())
                                  for h,states in enumerate(sets))


def path(model, probabilities):
    state=0; result=[0]
    for h in range(len(probabilities)-1):
        action=int(probabilities[h,state])
        state=int(model[state,action]); result.append(state)
    return tuple(result)


def true_support(model, policy, counts, initial, energy):
    states=path(model,policy)
    weights=np.asarray([initial]+[counts[h,s] for h,s in enumerate(states)],dtype=float)
    if weights.sum()<=energy: return float(len(weights))
    # KKT: d_j=min(1,t/n_j), with d_j=1 for an unobserved row.
    low,high=0.,float(weights.max())
    for _ in range(80):
        t=(low+high)/2
        d=np.minimum(1.,t/np.maximum(weights,1e-300))
        if float(weights@(d*d))>energy: high=t
        else: low=t
    return float(np.minimum(1.,low/np.maximum(weights,1e-300)).sum())


def run_case(horizon, truth_rank, epsilon, seed):
    models = [transition(r) for r in reversed(RANKS)]
    truth = transition(truth_rank)
    sets = reachable(truth,horizon)
    # These are basis vectors, so numerical and symbolic span ranks agree.
    stage_ranks = [int(np.linalg.matrix_rank(np.eye(D)[:,s])) for s in sets]
    rank = max(stage_ranks)
    budget = Budget(epsilon,candidates=len(models),horizon=horizon,
                    score_bound=1.,risk_scale=2.,dimension=D)
    lambda_a = budget.historical_energy/(rank*(horizon+1))
    lambda_b = budget.eta**2/(128*rank*(horizon+1))
    buffers = np.zeros((horizon-1,D,2),dtype=np.int64)
    observed = np.full((D,2),-1,dtype=np.int64)
    expected = np.zeros((horizon,D)); expected_root = 0.
    initial = episodes = 0
    rng = np.random.default_rng(seed)
    rows = []
    # Only D (not measured rank) controls this learner's safety cap.
    cap = budget.cap()
    for k in range(1,cap+1):
        consistent = [m for m in models if np.all(m[observed>=0] == observed[observed>=0])]
        if not consistent:
            raise RuntimeError('truth excluded by deterministic full-log MLE')
        fit = consistent[0]
        counts = frequencies(fit,buffers,initial)
        bonus = np.minimum(1.,np.sqrt(budget.energy/np.maximum(1,counts)))
        b, policy = optimal_bonus(fit,bonus)
        b += min(1.,sqrt(budget.energy/initial)) if initial else 1.
        error = tv(fit,truth,horizon)
        stopped = b <= 3*epsilon/16
        row = dict(horizon=horizon,rank=rank,dimension=D,epsilon=epsilon,seed=seed,
                   round=k,mle_calls=k,planning_calls=k,episodes=episodes,
                   fit_rank=RANKS[::-1][next(i for i,m in enumerate(models) if np.array_equal(m,fit))],
                   bonus=b,sup_tv=error,stopped=stopped,
                   certificate_valid=error<=b+1e-10,
                   stage_ranks=';'.join(map(str,stage_ranks)),
                   rank_verified=rank==truth_rank,
                   known_dimension_cap=cap,true_rank_cap=budget.cap(rank),
                   training_potential=potential(frequencies(truth,buffers,initial),initial,sets,lambda_a),
                   diagnostic_potential=potential(expected,expected_root,sets,lambda_b),
                   training_increment=0.,diagnostic_increment=0.,
                   total_increment=0.,progress_pass=True,cap_inequality_pass=True,
                   training_batch=0,diagnostic_batch=0)
        rows.append(row)
        if stopped:
            break
        if k==cap:
            raise RuntimeError('known-dimension cap reached without certificate')
        m,n = budget.theory_training(k),budget.theory_nested()
        # rho = pi/3 + sum_h (D_h^u+D_h^w)/(3H).
        # Our exact coefficient split permits action-0 completion for both
        # components, including all spare mass. Labels remain distinct.
        rho = [(1/3,policy.astype(float))]
        rho += [(1/(3*horizon),design(policy,h)) for h in range(horizon) for _ in range(2)]
        support=true_support(truth,policy,frequencies(truth,buffers,initial),
                             initial,budget.historical_energy)
        diag_tv=sum(weight*(path(truth,probs)!=path(fit,probs)) for weight,probs in rho)
        first=support>=budget.tau/2-1e-10
        second=diag_tv>=budget.eta-1e-10
        row.update(true_support=support,diagnostic_tv=diag_tv,
                   training_alternative=first,diagnostic_alternative=second,
                   alternatives_pass=first or second)
        for weight, probs in rho:
            expected += weight*occupancy(truth,probs)
        expected_root += 1.
        fixed = np.zeros_like(policy,dtype=float)
        collect(truth,fixed,m,rng,observed,buffers)
        for h in range(horizon):
            coin = int(rng.binomial(m,.5))
            collect(truth,design(policy,h),coin,rng,observed,buffers,h)
            collect(truth,design(policy,h,True),m-coin,rng,observed,buffers,h)
        # E_0 uses initial core; later cuts preserve each rho label.
        cuts = [(1/horizon,fixed)]
        for j in range(1,horizon):
            for weight,probs in rho:
                cut = probs.copy(); cut[j-1:] = 0.; cut[j-1] = .5
                cuts.append((weight/horizon,cut))
        sizes = rng.multinomial(n,[weight for weight,_ in cuts])
        for (_,probs),size in zip(cuts,sizes):
            collect(truth,probs,int(size),rng,observed,buffers)
        initial += m
        episodes += (horizon+1)*m+n
        da = potential(frequencies(truth,buffers,initial),initial,sets,lambda_a)-row['training_potential']
        db = potential(expected,expected_root,sets,lambda_b)-row['diagnostic_potential']
        rhs = rank*(horizon+1)*(log1p(k*m*rank/lambda_a)+log1p(k*rank/lambda_b))
        row.update(training_increment=da,diagnostic_increment=db,total_increment=da+db,
                   progress_pass=da+db>=log(2)-1e-9,
                   cap_inequality_pass=k*log(2)<=rhs+1e-9,
                   training_batch=m,diagnostic_batch=n)
        row['charged_progress_pass'] = (da>=log(2)-1e-9 if first
                                         else second and db>=log(65)-1e-9)
        row['charged_branch']='training' if first else 'diagnostic'
        if not row['progress_pass'] or not row['certificate_valid']:
            raise RuntimeError(f'audit failed: {row}')
    return rows


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--horizons',type=int,nargs='+',default=[8,12,16])
    p.add_argument('--epsilons',type=float,nargs='+',default=[.2,.1,.05])
    p.add_argument('--seeds',type=int,default=3)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    rows=[]; finals=[]
    for h,r,e,s in product(args.horizons,RANKS,args.epsilons,range(args.seeds)):
        path=run_case(h,r,e,s); rows.extend(path); finals.append(path[-1])
        f=path[-1]
        print(f'H={h} r={r} eps={e} seed={s} calls={f["mle_calls"]} '
              f'TV={f["sup_tv"]} stop={f["stopped"]}',flush=True)
    write_csv(args.output/'rounds.csv',rows)
    write_csv(args.output/'finals.csv',finals)
    summary=[]
    for h,r,e in product(args.horizons,RANKS,args.epsilons):
        g=[x for x in finals if (x['horizon'],x['rank'],x['epsilon'])==(h,r,e)]
        if len(g)!=args.seeds: raise RuntimeError('rank verification failed')
        summary.append(dict(horizon=h,rank=r,epsilon=e,seeds=len(g),
                            calls=float(np.median([x['mle_calls'] for x in g])),
                            min_calls=min(x['mle_calls'] for x in g),
                            max_calls=max(x['mle_calls'] for x in g),
                            episodes=float(np.median([x['episodes'] for x in g])),
                            bonus=float(np.median([x['bonus'] for x in g])),
                            stopped=sum(x['stopped'] for x in g),
                            max_tv=max(x['sup_tv'] for x in g),
                            true_rank_cap=g[0]['true_rank_cap'],known_dimension_cap=g[0]['known_dimension_cap']))
    write_csv(args.output/'summary.csv',summary)


if __name__=='__main__': main()
