"""Sequential stage gates: exact observable rank-two PSR horizon pilot."""
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
import numpy as np
from routing import Budget
from run_exact_rank_audit import potential
from run_multistage_experiment import write_csv


def model(horizon, closed=None):
    # State 0 is the live track, state 1 is absorbing dead state.
    t=np.ones((horizon-1,2,2),dtype=np.int64)
    t[:,0,1]=0
    if closed is not None: t[closed,0,1]=1
    return t


def plan(fit, bonus):
    horizon=len(bonus); policy=np.zeros((horizon,2),dtype=np.int64)
    value=bonus[-1].copy()
    for h in range(horizon-2,-1,-1):
        future=value[fit[h]]
        policy[h]=np.argmax(future,axis=1)
        value=bonus[h]+np.max(future,axis=1)
    return float(value[0]),policy


def design(policy, cut=None, uniform=False):
    p=policy.astype(float).copy()
    if cut is not None:
        p[cut:]=0
        if uniform: p[cut]=.5
    return p


def collect(truth, probs, size, rng, observed, buffer, stage=None):
    visits=np.array([size,0],dtype=np.int64)
    for h in range(len(probs)-1):
        one=rng.binomial(visits,probs[h])
        flows=np.stack((visits-one,one),axis=1)
        observed[h][flows>0]=truth[h][flows>0]
        if stage==h+1: buffer[h]+=flows
        visits=np.zeros(2,dtype=np.int64)
        np.add.at(visits,truth[h].ravel(),flows.ravel())


def occupancy(truth, probs):
    rows=np.zeros((len(probs),2));rows[0,0]=1
    for h in range(len(probs)-1):
        flow=np.stack((rows[h]*(1-probs[h]),rows[h]*probs[h]),axis=1)
        np.add.at(rows[h+1],truth[h].ravel(),flow.ravel())
    return rows


def counts_for(fit,buffer,initial):
    out=np.zeros((len(buffer)+1,2),dtype=np.int64);out[0,0]=initial
    for h in range(len(buffer)):
        np.add.at(out[h+1],fit[h].ravel(),buffer[h].ravel())
    return out


def run(horizon,seed,epsilon):
    truth=model(horizon)
    models=[model(horizon,j) for j in range(horizon-1)]+[truth]
    b=Budget(epsilon,candidates=len(models),horizon=horizon,
             score_bound=1.,risk_scale=2.,dimension=2)
    rng=np.random.default_rng(seed); observed=np.full_like(truth,-1)
    buffer=np.zeros_like(truth); initial=episodes=0; rows=[]
    reachable=[{0}]
    for h in range(horizon-1):
        reachable.append({int(truth[h,s,a]) for s in reachable[-1] for a in range(2)})
    stage_ranks=[len(states) for states in reachable]
    assert max(stage_ranks)==2
    sets=[sorted(states) for states in reachable]
    la=b.historical_energy/(2*(horizon+1))
    lb=b.eta**2/(128*2*(horizon+1))
    expected=np.zeros((horizon,2)); expected_root=0.
    for k in range(1,b.cap()+1):
        fit_index=next(i for i,m in enumerate(models)
                       if np.all(m[observed>=0]==observed[observed>=0]))
        fit=models[fit_index]
        counts=counts_for(fit,buffer,initial)
        bonuses=np.minimum(1,np.sqrt(b.energy/np.maximum(1,counts)))
        value,policy=plan(fit,bonuses)
        value+=min(1,np.sqrt(b.energy/initial)) if initial else 1
        stopped=value<=3*epsilon/16
        row=dict(horizon=horizon,rank=2,dimension=2,epsilon=epsilon,seed=seed,
                 round=k,fit_gate=fit_index if fit_index<horizon-1 else -1,
                 bonus=value,episodes=episodes,mle_calls=k,planning_calls=k,
                 stopped=stopped,truth_fitted=fit_index==horizon-1,
                 sup_tv=0 if fit_index==horizon-1 else 1,
                 certificate_valid=(fit_index==horizon-1 or value>=1),
                 stage_ranks=';'.join(map(str,stage_ranks)),
                 training_potential=potential(counts_for(truth,buffer,initial),initial,sets,la),
                 diagnostic_potential=potential(expected,expected_root,sets,lb),
                 training_increment=0.,diagnostic_increment=0.,
                 progress_pass=True,cap_inequality_pass=True,
                 training_batch=0,diagnostic_batch=0,
                 true_rank_cap=b.cap(2),known_dimension_cap=b.cap())
        rows.append(row)
        if stopped: break
        if k==b.cap(): raise RuntimeError('cap reached')
        m,n=b.theory_training(k),b.theory_nested()
        fixed=np.zeros_like(policy,dtype=float)
        collect(truth,fixed,m,rng,observed,buffer)
        for h in range(horizon):
            split=int(rng.binomial(m,.5))
            collect(truth,design(policy,h),split,rng,observed,buffer,h)
            collect(truth,design(policy,h,True),m-split,rng,observed,buffer,h)
        rho=[(1/3,policy.astype(float))]
        rho += [(1/(3*horizon),design(policy,h)) for h in range(horizon) for _ in range(2)]
        for weight,probs in rho: expected+=weight*occupancy(truth,probs)
        expected_root+=1.
        cuts=[(1/horizon,fixed)]
        for j in range(1,horizon):
            for weight,probs in rho:
                cut=probs.copy();cut[j-1:]=0;cut[j-1]=.5
                cuts.append((weight/horizon,cut))
        sizes=rng.multinomial(n,[weight for weight,_ in cuts])
        for (_,probs),size in zip(cuts,sizes):
            collect(truth,probs,int(size),rng,observed,buffer)
        initial+=m;episodes+=(horizon+1)*m+n
        from math import log,log1p
        da=potential(counts_for(truth,buffer,initial),initial,sets,la)-row['training_potential']
        db=potential(expected,expected_root,sets,lb)-row['diagnostic_potential']
        rhs=2*(horizon+1)*(log1p(k*m*2/la)+log1p(k*2/lb))
        row.update(training_increment=da,diagnostic_increment=db,
                   progress_pass=da+db>=log(2)-1e-9,
                   cap_inequality_pass=k*log(2)<=rhs+1e-9,
                   training_batch=m,diagnostic_batch=n)
        assert row['progress_pass'] and row['cap_inequality_pass']
    return rows


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--horizons',type=int,nargs='+',default=[3,4,6,8,12,16])
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--epsilon',type=float,default=.2)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    paths=[run(h,s,a.epsilon) for h,s in product(a.horizons,range(a.seeds))]
    write_csv(a.output/'rounds.csv',[row for path in paths for row in path])
    write_csv(a.output/'finals.csv',[path[-1] for path in paths])
    for path in paths:
        print(path[-1]['horizon'],path[-1]['seed'],len(path),
              [r['fit_gate'] for r in path],path[-1]['stopped'],flush=True)


if __name__=='__main__':main()
