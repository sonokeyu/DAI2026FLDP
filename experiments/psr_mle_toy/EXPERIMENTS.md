# Experiment suite status

This suite evaluates the paper's claimed computational interface, oracle
call scaling, certificate stopping, and downstream reward reuse. Every reported
trajectory TV in the finite routing examples is computed exactly. The true
environment is used for evaluation only; the learners receive sampled
trajectories and their specified candidate model classes.

## Accuracy and matched certificates

The shared routing task uses candidate outcome probabilities `0.2` and
`0.2 + epsilon`, target all-policy trajectory TV `epsilon/4`, and `delta=0.1`.
OMLE uses its reward-free confidence set and two intervention episodes per
round. Its added stopping condition is the exact confidence-set TV diameter
at most `epsilon/4`. The `ours` implementation keeps the manuscript's
Appendix D batches and stops when its bonus reaches `3 epsilon/16`; the theorem
then gives the same `epsilon/4` target.

Over 100 seeds for each of the two truths and four epsilon values, all 800
OMLE runs stopped with a singleton diameter and exact output-model TV zero.
The truth remained in the confidence set in all 800 runs. The theory-driven
`ours` run also had 800/800 certified stops with exact TV zero. This is a
finite-sample audit of the certificates on this family, not a statistical
proof of their general guarantees.

| epsilon | target TV | OMLE median episodes | OMLE joint searches | ours median episodes | ours MLE calls | ours plans |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.40 | 0.1000 | 94 | 47 | 427,648,271,078 | 3 | 3 |
| 0.20 | 0.0500 | 322 | 161 | 1,710,593,084,286 | 3 | 3 |
| 0.10 | 0.0250 | 1,211 | 605.5 | 6,842,372,337,134 | 3 | 3 |
| 0.05 | 0.0125 | 4,305 | 2,152.5 | 27,369,489,348,528 | 3 | 3 |

The main trade-off is clear here: very few fits and plans, with a huge episode
bill under the theorem's explicit batch constants. The experiment shows no
sample advantage for `ours` over OMLE. The two methods' oracle calls do
different work, so call counts alone do not establish a wall-clock speedup.

## Optimization interface scaling

The existing interface audit varies the number of candidate models from 2 to
32. At 32 candidates, the implemented OMLE joint objective evaluates 496
unordered model pairs, whereas the ordinary MLE scores 32 candidates and the
planner receives one fitted model. This measures the objective work in these
finite implementations. It is not a lower bound for other OMLE optimizers.
The factorized multi-coordinate audit below also records pair scores as the
confidence set grows.

## Multiple uncertain coordinates

The added finite rank sweep uses 1, 2, 4, or 8 independent Bernoulli test
coordinates and epsilon 0.2 or 0.1, with 100 random truth vectors per cell.
`ours_batch` chooses the least-covered coordinate, samples a precomputed
Hoeffding batch, and performs one ordinary MLE plus one plan per coordinate.
OMLE uses a uniform intervention and a targeted disagreement intervention,
then updates its likelihood confidence product after each pair.

The expanded 3,200-run sweep also includes uniform random coordinate
exploration and least-count coordinate exploration, both followed by ordinary
MLE refits every eight episodes. All runs stop with exact trajectory TV zero.
For dimension 8 and epsilon 0.1, `ours_batch` uses 9 MLE calls, 9 plans, and
32,488 episodes; OMLE uses a median 3,006.5 joint searches and 6,013 episodes;
uniform uses 33,236 episodes and 4,154.5 MLE calls; least-count uses 32,488
episodes and 4,061 MLE calls. The OMLE objective's median pair-score count is
52,920,412 in that cell. This isolates how pair search can grow with model
uncertainty; `ours_batch` is a purpose-built factorized analogue, not the
complete PSR algorithm or its Appendix D schedule.

## Full PSR on multiple uncertain branches

The complete Algorithm 1 loop is also run in the existing three-stage PSR
simulator with four independently uncertain `(cue, route, mode-a)` outcome
probabilities. The candidate class has 16 models with each branch probability
at 0.2 or 0.4. We use the paper's fitted-model bonus, training batches, initial
core data, nested diagnostics, and stopping rule. The appendix-independent
batch scale is 1, as in the existing finite routing audits. Each condition has
100 seeds. In the realizable condition, the true model is sampled from the
candidate class. In the continuous-truth condition, each true branch
probability is independently sampled from `[0.2, 0.4]`, while the fitted class
retains its 0.2/0.4 grid. This intentionally violates realizability.

All 200 realizable runs stopped with exact worst-policy TV zero. Median
episodes were 33,000,000 at epsilon 0.2 and 132,000,000 at epsilon 0.1, with
133 MLE and 133 planning calls in both settings. The continuous-truth runs
also stopped in 133 rounds, but their median TV was 0.0688 and their 90th
percentile was 0.0882. Only 15% and 1% of those runs, respectively, were within
the epsilon/4 target. The fitting class cannot represent the continuous true
probabilities, so the fitted-model bonus stopping rule does not certify the
true-environment TV in this misspecified condition. This experiment measures
that gap directly. It is not a theorem violation because the theorem assumes
realizability.

Reproduce with:

```sh
python3 experiments/psr_mle_toy/run_full_psr_multicoordinate.py --output experiments/psr_mle_toy/results/full_psr_multicoordinate --seeds 100 --scale 1.0
```

[Summary](results/full_psr_multicoordinate/full_psr_multicoordinate_summary.csv)
· [Per-run results](results/full_psr_multicoordinate/full_psr_multicoordinate_runs.csv)

## Rank, horizon, and reward reuse

The existing unknown-rank audit tests true ranks 1, 2, and 4, core dimension
8, horizons 4, 6, and 8, and epsilon 0.4 and 0.1 with 100 seeds per cell.
All 1,800 runs stopped by their true-rank cap with exact TV zero. Median fit
and planning calls were 2, 3, and 5 by rank. This verifies the rank-dependent
stopping behavior on that constructed family; these cells do not estimate an
asymptotic horizon exponent.

Four bounded rewards are planned on the model returned by the reward-free
learner. In the theory routing audit, each had zero regret. The pre-existing
observable rank sweep also varies rank and horizon; its records are in
[`rank_runs.csv`](results/rank_runs.csv). The exact theory audit is in
[`oracle_rank_stops.csv`](results/theory/oracle_rank_stops.csv).

## Baselines

- `omle`: reward-free OMLE with a confidence-diameter stopping extension.
- `e2d`: exact closed-form saddle point for the symmetric two-model routing
  instance, multiplicative model-weight updates with eta 1/2, and MLE as the
  permitted tie break for its output model. Since the algorithm has no bonus
  certificate here, its fixed horizon is selected after runs to reach 90%
  exact-TV accuracy for both truths through the cap. This is not a general E2D
  solver.
- `psr_ucb`: one-coordinate PSR-UCB specialization. It reproduces the scalar
  feature bonus, not the full horizon-indexed core-test construction.
- `uniform` and `count`: traditional exploration controls already included
  in the earlier equal-budget and batch-sweep experiments.

Recent literature was screened in [`BASELINES.md`](BASELINES.md). PSR-UCB is
the closest recent PSR conference direction found. Recent tabular-MDP
reward-free and reward-monitoring papers have different targets or
assumptions, so they are not treated as matched PSR model-estimation baselines.

## Reproduction commands

From the repository root:

```sh
python3 experiments/psr_mle_toy/run_omle_certificate.py --output experiments/psr_mle_toy/results/matched_certificates --seeds 100 --cap 16384
python3 experiments/psr_mle_toy/run_e2d_toy.py --output experiments/psr_mle_toy/results/matched_certificates/e2d --seeds 100 --cap 16384
python3 experiments/psr_mle_toy/run_multicoordinate.py --output experiments/psr_mle_toy/results/multicoordinate --seeds 100 --cap 131072
python3 experiments/psr_mle_toy/run_oracle_audit.py --output experiments/psr_mle_toy/results/theory --seeds 100 --cap 16384
```

The three newly added runs use Python's standard library. The Appendix D
theory audit additionally requires NumPy. The manuscript remains unchanged.
