# Multi-stage partially observed PSR experiment

This experiment tests the manuscript's ordinary-MLE and single-model
planning loop when every transition stage can differ across candidate
models. It uses four hidden states, four noisy observations, two actions,
and horizons 3, 4, and 5. The observation matrix has diagonal probability
0.75 and off-diagonal probability 1/12. At each stage the action changes
the latent transition distribution when the rank exceeds one. At rank one,
the stage-specific distribution is unknown but independent of action, as
required for a normalized one-dimensional reachable state space. Each candidate has a stage-specific
shift pattern. The true models use a shift at every transition stage.

The common core consists of the four singleton next-observation tests.
Their observation matrix is invertible. The resulting PSR has core dimension
four, one core action string, and signed-input B-stability bound 1.75.
The evaluator checks the reachable core span after the first transition;
its dimension is respectively 1, 2, or 4 in the three rank conditions.
The learner receives the same full candidate class at each rank and does
not receive the true rank.

The `ours` exploration loop uses all stage support programs, ordinary
complete-episode MLE, single-fitted-model bonus planning, frozen training
batches, the diagnostic mixture, and random-cut nested experiments. The
singleton core makes the coefficient geometry an infinity-norm conic
program. Historical feature rows are compressed by SVD. Clarabel solves
the conic programs. Logged action and mixture probabilities are common
across candidate models, so they cancel from the MLE comparison. Batch
counts are drawn from exact multinomial trajectory laws. This has the
same distribution as independent physical episodes while permitting the
appendix's very large sample counts to be represented efficiently.

OMLE uses exact pairwise all-policy TV search within its finite confidence
set. Its diameter stopping rule is an adaptive finite-class extension of
the original prescribed-round analysis. Uniform and observation-history
count exploration use ordinary MLE every 64 episodes. The evaluator
computes exact worst-policy TV by dynamic programming over both models'
unnormalized latent masses. This TV is never supplied to the learners.

Install dependencies into the project environment and run:

```sh
UV_CACHE_DIR=/tmp/psr-uv-cache uv pip install --python .venv/bin/python cvxpy matplotlib
.venv/bin/python -B experiments/psr_mle_toy/run_multistage_experiment.py \
  --output experiments/psr_mle_toy/results/multistage_pomdp \
  --horizons 3 4 5 --epsilons .1 .05 --seeds 4 \
  --baseline-budget 1024
.venv/bin/python -B experiments/psr_mle_toy/run_multistage_call_scaling.py \
  --output experiments/psr_mle_toy/results/multistage_pomdp \
  --seeds 2 --base-batch 10000000 --rounds 16
MPLCONFIGDIR=/tmp/psr-mpl-cache .venv/bin/python -B \
  experiments/psr_mle_toy/plot_multistage.py \
  --input experiments/psr_mle_toy/results/multistage_pomdp \
  --output experiments/psr_mle_toy/results/multistage_pomdp
```

The output contains `multistage_runs.csv` with every checkpoint,
`multistage_final.csv` with the last row per run, and
`multistage_summary.csv` with paired cell summaries. The episode ceilings
are method specific: ours follows the appendix batch formulas and stops
on bonus; the three comparators use at most 1,024 physical episodes.
Therefore episode totals should be compared explicitly, and the model
accuracy columns should be read together with their budgets. Oracle calls
record MLE, single-model exploration planning, and joint pair-policy
search separately. The conic support solve is part of the planner's reward
evaluation and has a separate computational cost from these oracle calls.

The candidate class is deliberately finite and realizable. All transition
rows share a 0.95 uniform background and differ by a 0.05 rank-controlled
component. The multi-stage experiment tests the general proof machinery
more directly than the terminal-only combination lock. Finite grids cannot
establish the asymptotic exponents in the theorem. Solver output is
numerical; the ex-post exact TV audit checks the returned models and the
bonus inequality on each recorded refit.

## Results of the specified grid

There are 72 paired cases per method: three horizons, three true ranks,
two accuracy targets, and four seeds. A case succeeds when the final model's
exact worst-policy TV is at most ε/4. The table aggregates all 72 cases;
the per-cell results are in `multistage_summary.csv`.

| Method | Successes, ε=.1 | Successes, ε=.05 | Median optimization calls | Median episodes |
| --- | ---: | ---: | ---: | ---: |
| ours | 36/36 | 36/36 | 2 MLE + 2 single-model planning | 12,168,175,022,032 |
| omle | 22/36 | 19/36 | 512 joint model-pair/policy searches | 1,024 |
| uniform + MLE | 25/36 | 23/36 | 16 MLE | 1,024 |
| count + MLE | 21/36 | 19/36 | 16 MLE | 1,024 |

All 72 ours runs stopped after the first training batch. The initial fitted
model had positive worst-policy TV in all 72 runs; after the batch, ordinary
MLE selected the true model in all 72. Every stop passed the ex-post audit
`TV <= bonus <= 3ε/16`, hence also `TV <= ε/4`. The OMLE confidence-set
diameter criterion stopped in 2 of 72 runs under its 1,024-episode cap.

These are **different sample budgets**. The appendix batch formula leads to
2.90×10^12 through 5.09×10^13 episodes in this small finite class. Their
trajectory counts are sampled exactly from a multinomial law, which is
statistically equivalent to independently sampling that many episodes;
the program does not execute trillions of physical trajectories. This run
demonstrates the low oracle-call count and checks the stopping certificate
on a genuinely multi-stage PSR. It does not show superior accuracy at equal
sample cost, or establish the asymptotic Õ(rH) call law: all ours cases
stop at two calls, leaving no visible rank dependence. The finite-class
OMLE comparator also uses an adaptive diameter stop beyond the original
prescribed-round algorithm. Joint searches, MLE fits, and conic support
solves have different per-call costs; the table counts interfaces and does
not equate their wall-clock time. These call totals cover reward-free
exploration; requesting a downstream task policy would add one final
single-model reward-planning call.

## Multi-round call scaling check

To expose update behavior hidden by the conservative appendix constants,
a separate check fixes each round's training and nested batch sizes at
10 million for ε=.1 and 40 million for ε=.05. This is a deliberately
specified `1/ε²` batch rule, **not** the theorem's batch formula. There
are two seeds for each H/r/ε cell. Every run stops by round 16 and passes
the ex-post `TV <= bonus` audit. MLE and single-model planning each have
the same number of calls:

| H | r=1 | r=2 | r=4 |
| ---: | ---: | ---: | ---: |
| 3 | 3 | 3 | 4 |
| 4 | 4 | 5 | 8 |
| 5 | 5 | 7 | 10 |

The counts are identical at both ε values; the lower ε uses exactly four
times as many episodes in each H/r cell. In all these tested cells the
number of calls is at most `rH`. This is empirical support for the
rank/horizon and precision behavior, not a proof of the Õ(rH) asymptotic
bound. The final TV is again zero because the finite candidate class is
exactly realizable and the first batch already identifies the truth;
later rounds shrink the data-dependent bonus until it permits stopping.
Detailed refit paths and summaries are in `call_scaling_runs.csv`,
`call_scaling_final.csv`, and `call_scaling_summary.csv`.
