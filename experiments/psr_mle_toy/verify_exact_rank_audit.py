"""Independent cone/planning checks and audit of the completed run data."""
import csv
from itertools import product
from pathlib import Path
import cvxpy as cp
import numpy as np
from run_exact_rank_audit import transition, optimal_bonus
from run_multistage_experiment import write_csv


def main():
    rng=np.random.default_rng(83); errors=[]
    for _ in range(8):
        dim=4; model=transition(4)[:dim]; query=int(rng.integers(dim))
        counts=rng.integers(1,100,size=dim); energy=10.
        u=cp.Variable(dim); w=cp.Variable((dim,2,dim))
        g=cp.hstack([u[s]+sum(w[s,a,model[s,a]] for a in range(2)) for s in range(dim)])
        cons=[cp.norm_inf(u)<=.5,cp.sum_squares(cp.multiply(np.sqrt(counts),g))<=energy]
        cons += [sum(cp.norm_inf(w[s,a,:]) for a in range(2))<=.5 for s in range(dim)]
        value=cp.Problem(cp.Maximize(g[query]),cons).solve(solver='CLARABEL')
        errors.append(abs(value-min(1,np.sqrt(energy/counts[query]))))
    assert max(errors)<1e-6
    for _ in range(10):
        model=transition(4); bonus=rng.random((5,16)); value,_=optimal_bonus(model,bonus)
        exhaustive=[]
        for actions in product(range(2),repeat=4):
            state=0; total=bonus[0,state]
            for h,a in enumerate(actions):
                state=int(model[state,a]); total+=bonus[h+1,state]
            exhaustive.append(total)
        assert abs(value-max(exhaustive))<1e-12
    folder=Path(__file__).parent/'results/exact_rank_audit'
    def read(name):
        with (folder/name).open() as stream: return list(csv.DictReader(stream))
    rounds=read('rounds.csv'); finals=read('finals.csv')
    nonstops=[r for r in rounds if r['stopped']=='False']
    assert len(finals)==108 and len(nonstops)==405
    for r in finals:
        assert r['stopped']=='True' and r['rank_verified']=='True'
        assert float(r['sup_tv'])<=float(r['epsilon'])/4
        assert int(r['mle_calls'])<=int(r['true_rank_cap'])
    assert all(r['certificate_valid']=='True' for r in rounds)
    for r in nonstops:
        for key in ('alternatives_pass','progress_pass','charged_progress_pass','cap_inequality_pass'):
            assert r[key]=='True',r
    result=dict(runs=len(finals),rounds=len(rounds),nonstopping_rounds=len(nonstops),
                certified_stops=len(finals),all_certificate_checks_pass=True,
                all_charged_progress_checks_pass=True,
                training_charged=sum(r['charged_branch']=='training' for r in nonstops),
                diagnostic_only_charged=sum(r['charged_branch']=='diagnostic' for r in nonstops),
                min_training_increment=min(float(r['training_increment']) for r in nonstops),
                min_diagnostic_increment=min(float(r['diagnostic_increment']) for r in nonstops),
                cone_comparisons=8,max_cone_error=max(errors),exhaustive_planning_comparisons=10)
    write_csv(folder/'audit_summary.csv',[result]); print(result)


if __name__=='__main__':main()
