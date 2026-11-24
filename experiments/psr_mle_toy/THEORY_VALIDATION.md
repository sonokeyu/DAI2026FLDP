# Theory validation for batched PSR learning

The audit tests the optimization interfaces, the number of calls at
increasing accuracy, and the stopping guarantee. It uses the appendix
batch sizes and Theorem 3.1's dimension-based cap for **ours**. All
environments are finite and realizable. Their trajectory distributions
are enumerated or evaluated exactly. Batch counts have the same
distribution as the corresponding independent environment episodes.

## Optimization interfaces

At the initial fit with 2, 4, 8, 16, and 32 candidates, the implemented
**ours** algorithm calls ordinary MLE once and passes its fitted model
to a single planning call. The fitted model is selected by maximizing the
complete-data likelihood over the full candidate class. The planner
scores no model pairs. In this same setting the exact **omle** joint
objective evaluates 1, 6, 28, 120, and 496 model pairs, respectively.
The count instruments the toy optimizer's calls. It does not give a
lower bound for all implementations of the OMLE oracle or a wall-clock
comparison. [Measured interface calls](results/theory/interface_audit.csv)

## Accuracy, calls, and episode cost

For each ε we use a noisy three-stage routing instance with candidates
whose Bernoulli probabilities on the distinguishing branch are 0.2
and 0.2 + ε. The wrong model has exact worst-policy TV ε/2, above the
shared ε/4 target. Each setting has 100 seeds for each true model.
The horizon, core dimensions, candidate count, and stability bound are
fixed. **ours** uses Appendix D with c = 64 and δ = 0.1.

For **omle**, one round uses two episodes under the two intervention
policies. The table reports an empirical fixed horizon. It is the
smallest checkpoint at which at least 90% of runs satisfy the TV target
for each truth orientation, with that rate sustained through the
16,384-episode analysis cap. This uses exact TV after data collection
and never affects the OMLE controller. The experiment retains the
confidence set intersection and selects its highest-likelihood member
for evaluation. The simulation uses the sufficient counts for this
two-model instance. The OMLE pair comparisons remain nontrivial at the
reported median call count. Each round also updates the confidence set.

| ε | omle joint calls | ours MLE calls | ours exploration planning calls | omle episodes | ours median episodes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.40 | 15 | 3 | 3 | 30 | 4.28 × 10¹¹ |
| 0.20 | 69 | 3 | 3 | 138 | 1.71 × 10¹² |
| 0.10 | 392 | 3 | 3 | 784 | 6.84 × 10¹² |
| 0.05 | 973 | 3 | 3 | 1,946 | 2.74 × 10¹³ |

The calls for **ours** include the empty-data fit and its plan, and
exclude planning for rewards furnished after exploration. Its episode
count is the number represented by the sampled multinomial counts.
The appendix prescription reuses each fit for a larger batch as the
accuracy requirement tightens. This yields stable call counts and an
approximately fourfold increase in episodes when ε is halved in this
sweep. The latter behavior follows the chosen batch formulas. This
finite sweep leaves the asymptotic exponents to the proof. The counts
measure different oracle interfaces.
[Figure](results/theory/oracle_audit.pdf) · [Summary](results/theory/oracle_summary.csv) · [OMLE accuracy curves](results/theory/omle_accuracy_curves.csv.gz)

## Stopping and unknown rank

All 800 noisy routing runs of **ours** reached the bonus threshold
and returned exact worst-policy TV zero. Calibration held at each
refit, and the audit checked that maximum bonus upper-bounded exact
worst-policy TV at those refits. The four bounded observable rewards
planned after exploration also had zero regret.

The observable experiment sets the core dimension to eight and changes
the true reachable rank among 1, 2, and 4. It tests horizons 4, 6, and 8
and ε of 0.4 and 0.1, with 100 seeds per cell. For learning decisions
the hard limit is `K(D)`. The verifier computes `K(r)` after the
run from the true simulator. All 1,800 runs stopped by `K(r)` with
exact TV zero. Median calls to each oracle were 2, 3, and 5 at the
three ranks for all six horizon and accuracy combinations. This
verifies the unknown-rank stopping check on the tested family. It
does not estimate the dependence on the horizon from those medians.
[Routing stop checks](results/theory/oracle_stops.csv) · [Unknown-rank stop checks](results/theory/oracle_rank_stops.csv)

The general guarantee and its \(\widetilde O(rH)\) call bound follow from the
proof. The finite audits exercise the hypotheses and conclusions in
two explicit families. The episode count displays the price of the
appendix schedule. The manuscript gives the Λ_B² U H²
sample-bound comparison under its matching assumptions. This audit
does not estimate that factor or claim a sample advantage.

## Reproduce

Run from the experiment directory with NumPy and a TeX installation
with PGFPlots.

```sh
python -B -m unittest -v test_toy test_routing
python -B run_oracle_audit.py --output reproduced/theory
python -B plot_oracle_audit.py reproduced/theory/oracle_summary.csv reproduced/theory/oracle_audit.tex
latexmk -pdf -outdir=reproduced/theory reproduced/theory/oracle_audit.tex
```
