"""Controlled warm-start witness for the diagnostic-only progress branch.

The warm-start buffer is a possible left-route, mode-a log under both models.
The actual bonus planner selects the diagnostic policy; truth is used only to
audit the true support and diagnostic divergence.
"""
from math import sqrt, log1p, log
from pathlib import Path

import numpy as np

from routing import (Budget, CUES, MODES, PREFIXES, ROUTES, RoutingModel,
                     all_policy_tv, diagnostic, distance, hellinger, law,
                     nested, plan_bonus)
from run_multistage_experiment import write_csv


class FlatTruth(RoutingModel):
    def p(self, cue, route, mode):
        return .5 if mode == 'b' else 0.


def run_case(seed, warm_count=1_000_000_000, epsilon=.5):
    truth, candidate = FlatTruth(0.), RoutingModel(1.)
    budget = Budget(epsilon, candidates=2)
    # Each designated prefix count can arise from independent left/a batches.
    # Equal cue counts are chosen for a transparent, reproducible witness.
    histories = [warm_count, warm_count,
                 {x: warm_count//2 if x[1]=='L' else 0 for x in PREFIXES},
                 {(c,r,m):warm_count//2 if r=='L' and m=='a' else 0
                  for c,r in PREFIXES for m in MODES}]
    bonus, policy, queries = plan_bonus(candidate,histories,budget.energy)
    rho = diagnostic(policy,queries)
    cuts = nested(rho)
    diagnostic_tv=distance(law(truth,rho),law(candidate,rho))
    nested_h=hellinger(law(truth,cuts),law(candidate,cuts))
    def mode_b_probability(design):
        return sum(.5*(1-(design.routes[ci] if route=='L' else 1-design.routes[ci]))
                   *(1-design.modes[PREFIXES.index((cue,route))])
                   for ci,cue in enumerate(CUES) for route in ROUTES)
    p_b=sum(weight*mode_b_probability(design) for weight,design in rho)
    rank=2; lam=budget.eta**2/(128*rank*4*budget.score_bound**2)
    # In true coordinates (e0,m), stage-three rho states are e0 and m.
    stage3_logdet=log1p((1-p_b)/lam)+log1p(p_b/lam)
    diagnostic_logdet=log1p(1/lam)*3+stage3_logdet
    # At every visited true state the buffered true feature is identical to
    # a designated row. Cauchy-Schwarz in the joint historical-energy ball
    # bounds its expected support by sqrt(L*E/n).
    true_support_upper=sqrt(4*budget.historical_energy/warm_count)
    assert true_support_upper < budget.tau/2
    assert diagnostic_tv >= budget.eta
    assert bonus > 3*epsilon/16
    assert diagnostic_logdet >= log(65)
    # The candidate and truth have identical left/a warm-start log law.
    assert all(candidate.p(c,'L','a') == truth.p(c,'L','a') for c in CUES)
    true_law, fit_law = law(truth,cuts),law(candidate,cuts)
    rng=np.random.default_rng(seed)
    drawn=rng.multinomial(budget.theory_nested(),(true_law/true_law.sum()).ravel())
    candidate_excluded=any(count and fit_law.ravel()[i]==0
                           for i,count in enumerate(drawn))
    return dict(seed=seed,epsilon=epsilon,warm_count=warm_count,
                fitted_bonus=bonus,stop_threshold=3*epsilon/16,
                true_support_upper=true_support_upper,
                training_threshold=budget.tau/2,
                diagnostic_tv=diagnostic_tv,diagnostic_threshold=budget.eta,
                nested_hellinger=nested_h,
                stage3_mixture_probability=p_b,
                stage3_diagnostic_logdet=stage3_logdet,
                diagnostic_logdet_increment=diagnostic_logdet,
                diagnostic_batch=budget.theory_nested(),
                expected_nested_risk=budget.theory_nested()*nested_h,
                candidate_excluded=candidate_excluded,
                fitted_policy_routes=';'.join(map(str,policy.routes)),
                fitted_policy_modes=';'.join(map(str,policy.modes)),
                all_policy_tv=all_policy_tv(truth,candidate))


def main():
    folder=Path(__file__).parent/'results/diagnostic_witness'
    folder.mkdir(parents=True,exist_ok=True)
    rows=[run_case(seed) for seed in range(3)]
    write_csv(folder/'witness.csv',rows)
    for row in rows: print(row)


if __name__=='__main__': main()
