# Baseline scope and literature

This note extends the theory audit without changing the manuscript. The
shared accuracy task uses the noisy routing PSR with two possible truths,
Bernoulli parameters `0.2` and `0.2 + epsilon`, and the same all-policy TV
target `epsilon / 4`.

## Implemented methods

| Label | Method | Comparison role |
| --- | --- | --- |
| `omle` | Reward-free Optimistic Maximum Likelihood Estimation | Direct baseline. Maintains a likelihood confidence set and searches model pairs and policies. |
| `ours` | Ordinary MLE + model planning | Proposed method, using the manuscript's Appendix D batch schedule and bonus stopping rule. |
| `psr_ucb` | Scalar PSR-UCB specialization | Recent PSR baseline. Uses a confidence constrained MLE and a prediction-feature UCB bonus. |
| `uniform` | Uniform action exploration + ordinary MLE | Traditional data-collection baseline, already in `RESULTS.md`. |
| `count` | Least-visited route/mode exploration + ordinary MLE | Traditional count-based baseline, already in `RESULTS.md`. |

`psr_ucb` is a one-coordinate specialization, not a general PSR-UCB codebase.
The toy has one unresolved prediction coordinate: the Bernoulli outcome after
`(-, R, a)`. Its feature is the indicator of visiting that branch, so the
paper's inverse empirical Gram bonus reduces to
`min(alpha / sqrt(lambda + n), 1)`. The exact planner chooses the policy that
reaches this branch when the random cue is negative. The confidence radius is
fixed before sampling using `alpha = sqrt(log(2 K / delta) / 2)` with
`K = 32768`, `delta = 0.1`, and `lambda = 1`. Each iteration refits by MLE,
plans once, and collects one episode. This is a scalar adaptation: the paper's
full algorithm gathers stage-cut trajectories for all `H` stages before each
refit. It stops when the policy-level bonus is at most `epsilon / 4`. This
matches the all-policy error target used in the audit, which is stricter than
the PSR-UCB paper's displayed `epsilon / 2` bonus threshold.

This specialization tests the core feature-bonus mechanism and call schedule
on the same toy, not the full paper's multi-stage core-test data structure or
general PSR confidence construction. The CSV records the observed exact TV
at stop so the bonus can be checked on every run.

## Results at the common target

The table reports medians from 100 seeds under each truth orientation. For
`omle`, the episode and joint-call count is the first fixed horizon where at
least 90% of runs satisfy the target for both truths. For `ours`, the episode
count and oracle calls are from its theorem-driven stopping rule. For
`psr_ucb`, the episode count and oracle calls are from its scalar bonus stop.
These stop rules differ by design. `omle` performs one joint search per
update; this scalar `psr_ucb` adaptation performs one MLE and one plan per
update; `ours` uses separate ordinary-MLE and single-model planning oracles.
The counts instrument these implementations, not wall-clock time or a
universal lower bound.

| epsilon | TV target | method | episodes | MLE or joint calls | planning calls | evaluation |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0.40 | 0.1000 | `omle` | 30 | 15 joint | — | 90% empirical success |
| 0.40 | 0.1000 | `ours` | 427,648,271,078 | 3 MLE | 3 | exact TV 0 |
| 0.40 | 0.1000 | `psr_ucb` | 336 | 337 MLE | 337 | exact TV 0 |
| 0.20 | 0.0500 | `omle` | 138 | 69 joint | — | 91% empirical success |
| 0.20 | 0.0500 | `ours` | 1,710,593,084,286 | 3 MLE | 3 | exact TV 0 |
| 0.20 | 0.0500 | `psr_ucb` | 1,343.5 | 1,344.5 MLE | 1,344.5 | exact TV 0 |
| 0.10 | 0.0250 | `omle` | 784 | 392 joint | — | 90% empirical success |
| 0.10 | 0.0250 | `ours` | 6,842,372,337,134 | 3 MLE | 3 | exact TV 0 |
| 0.10 | 0.0250 | `psr_ucb` | 5,368 | 5,369 MLE | 5,369 | exact TV 0 |
| 0.05 | 0.0125 | `omle` | 1,946 | 973 joint | — | 91% empirical success |
| 0.05 | 0.0125 | `ours` | 27,369,489,348,528 | 3 MLE | 3 | exact TV 0 |
| 0.05 | 0.0125 | `psr_ucb` | 21,441 | 21,442 MLE | 21,442 | exact TV 0 |

The result isolates the intended trade-off in this setup: the appendix batch
schedule gives `ours` very few oracle calls while paying a very large episode
budget. The scalar `psr_ucb` bonus gives a separate recent-PSR comparison and
shows the familiar accuracy-dependent per-episode update cost. In this toy,
`omle` reaches the target with fewer episodes than either. This does not
contradict the theorem: the theorem controls oracle calls and gives a
polynomial episode bound, not a sample advantage over OMLE.

The machine-readable inputs are
[`oracle_summary.csv`](results/theory/oracle_summary.csv) and
[`psr_ucb_summary.csv`](results/psr_ucb/psr_ucb_summary.csv). The detailed
PSR-UCB trace is in [`psr_ucb_rounds.csv`](results/psr_ucb/psr_ucb_rounds.csv).

## Certificate matched OMLE and E2D comparison

The earlier table uses OMLE's retrospective fixed-horizon accuracy selection.
For the algorithmic comparison, a separate OMLE run now stops when its
confidence-set all-policy TV diameter is at most epsilon/4. `ours` retains
its theorem bonus stop. Each method is evaluated on the same exact-TV target.
The two stop rules are both data-driven certificates, while their episode
budgets remain method-specific.

| epsilon | target TV | OMLE episodes / joint searches | ours episodes / MLE / plans | E2D fixed episodes / game solves |
| ---: | ---: | ---: | ---: | ---: |
| 0.40 | 0.1000 | 94 / 47 | 427,648,271,078 / 3 / 3 | 32 / 32 |
| 0.20 | 0.0500 | 322 / 161 | 1,710,593,084,286 / 3 / 3 | 96 / 96 |
| 0.10 | 0.0250 | 1,211 / 605.5 | 6,842,372,337,134 / 3 / 3 | 272 / 272 |
| 0.05 | 0.0125 | 4,305 / 2,152.5 | 27,369,489,348,528 / 3 / 3 | 1,104 / 1,104 |

Each point aggregates 200 OMLE and 200 `ours` runs (100 per truth).
OMLE stopped all runs, had no false stops, and its confidence set contained
the truth at every stop in this finite audit. `ours` also had no false stops
and exact stop TV zero in these same routing runs. E2D's output is evaluated
at a fixed horizon selected externally to achieve at least 90% success for
both truths sustained to the cap; it does not certify stopping.

E2D uses the exact two-model saddle point for this symmetric routing class.
Its game optimizer selects the informative policy and a uniform distribution
over the two output models; ordinary MLE breaks the output tie. This is an
exact specialization of the All-Policy Model-Estimation E2D update, not a
general solver or a full paper reproduction.

## Multi-coordinate rank and precision sweep

[`multicoordinate_summary.csv`](results/multicoordinate/multicoordinate_summary.csv)
contains a finite audit with 1, 2, 4, or 8 independent uncertain test
coordinates. `ours_batch` uses an ordinary MLE and single-model planner, then
collects a fixed Hoeffding-sized batch on the least-covered coordinate.
OMLE collects its uniform and disagreement interventions and updates its
coordinate confidence product after every pair. The two traditional controls
choose either a coordinate uniformly at random or the least-visited
coordinate, and refit ordinary MLE every eight episodes. All methods use the
same coordinatewise Hoeffding stopping threshold (except OMLE, which uses its
likelihood-diameter certificate). Each cell uses 100 random truth vectors.
This environment isolates the dependence on the number of independent model
coordinates; it is not a partially observed benchmark.

At dimension 8 and epsilon 0.1, `ours_batch` used 9 MLE and 9 planning calls
and 32,488 episodes; OMLE used a median of 3,006.5 joint searches and 6,013
episodes. Uniform random exploration used 33,236 episodes and 4,154.5 MLE
calls; least-count exploration used 32,488 episodes and 4,061 MLE calls.
All 3,200 runs stopped with exact TV zero. The OMLE pair-score
instrumentation grows sharply with the number of unresolved coordinates
(median 52.9 million pair evaluations in that cell), while `ours_batch` never
scores model pairs. These measurements concern this factorized optimizer and
should not be read as wall-clock speedups for general solvers.

## Complete PSR loop with multiple uncertain branches

[`full_psr_multicoordinate_summary.csv`](results/full_psr_multicoordinate/full_psr_multicoordinate_summary.csv)
records the complete fitted-PSR algorithm on four uncertain terminal branches
of the three-stage routing system. The finite MLE class has 16 models with
branch probabilities 0.2 or 0.4. In the realizable condition, 100/100 runs
stopped with exact worst-policy TV zero at each of epsilon 0.2 and 0.1. Both
settings used a median of 133 MLE calls and 133 exploration-planning calls.
Median episode costs were 33 million and 132 million, respectively.

The continuous-truth condition draws each branch probability independently
from [0.2, 0.4] while retaining the finite endpoint class for fitting. It is
deliberately misspecified. The four-method comparison is in
[`full_psr_baseline_summary.csv`](results/full_psr_multicoordinate/full_psr_baseline_summary.csv).
All runs stopped. In the realizable cells, all methods had TV zero. At
epsilon 0.1, median episodes were 132 million for ours, 1,051 for OMLE,
28,548 for uniform + MLE, and 14,131 for count + MLE. Under continuous truth,
median TV ranged from 0.0688 to 0.0703 at epsilon 0.2 and from 0.0688 to
0.0694 at epsilon 0.1. Only 15% and 1% of runs, respectively, met the
epsilon/4 target. The two exploration baselines fit once after meeting their
sampling rule, while OMLE searches after each pair of episodes.

Reproduce these runs with:

```sh
python3 experiments/psr_mle_toy/run_omle_certificate.py --output experiments/psr_mle_toy/results/matched_certificates --seeds 100 --cap 16384
python3 experiments/psr_mle_toy/run_e2d_toy.py --output experiments/psr_mle_toy/results/matched_certificates/e2d --seeds 100 --cap 16384
python3 experiments/psr_mle_toy/run_multicoordinate.py --output experiments/psr_mle_toy/results/multicoordinate --seeds 100 --cap 131072
```

## Related papers found

### Closest recent match: PSR-UCB (ICLR 2024)

Huang, Liang, and Yang, [*Provably Efficient UCB-type Algorithms for Learning
Predictive State Representations*](https://arxiv.org/abs/2307.00405), ICLR 2024.
This is the closest recent conference paper: it studies PSRs, uses an
MLE-based confidence set and a planning bonus, and proves all-policy model
accuracy. Its full algorithm collects stage-cut trajectories for all horizon
steps in a round before fitting and planning. The scalar specialization above
reduces this to the one uncertain prediction coordinate. The public search did
not surface an author release of runnable code, so this is our reimplementation
of the one-dimensional mechanism, not a reproduction of the paper's full
benchmark suite.

### Direct reward-free baseline: All-Policy Model-Estimation E2D (ICLR 2023)

Chen, Bai, and Mei, [*Partially Observable RL with B-Stability: Unified
Structural Condition and Sharp Sample-Efficient Algorithms*](https://arxiv.org/abs/2209.14990),
ICLR 2023. Its All-Policy Model-Estimation E2D algorithm directly shares the
model-estimation objective and is the closest theoretical comparator to the
manuscript's reward-free guarantee. Unlike `ours`, it solves a minimax
optimization over exploration-policy and output-model distributions and
updates a model distribution each episode. It remains a useful next baseline;
the public search did not surface a reusable implementation. We have not
included a hand-tuned saddle-point approximation in the measured table.

### Nearby work that is not a matched baseline

- Li, Yan, Chen, and Fan, [*Minimax-optimal reward-agnostic exploration in
  reinforcement learning*](https://proceedings.mlr.press/v247/li24a.html),
  COLT 2024. This is a recent reward-free result for fully observed tabular
  finite-horizon MDPs. It targets downstream values for reward functions,
  rather than all-policy trajectory TV in a general PSR. It is relevant for a
  separate observable-MDP experiment, but not a like-for-like baseline here.
- Kazemipour, Taylor, and Bowling, [*Model-Based Exploration in Monitored
  Markov Decision Processes*](https://proceedings.mlr.press/v267/kazemipour25a.html),
  ICML 2025. It studies exploration when rewards are partly unobserved and
  compares against Directed-E2. The reward-monitoring objective and structural
  assumptions differ from reward-free all-policy model estimation.
- Zhang et al., [*Provable Representation with Efficient Planning for
  Partially Observable Reinforcement Learning*](https://proceedings.mlr.press/v235/zhang24bd.html),
  ICML 2024. Its algorithm targets online reward optimization under an
  L-decodability assumption and learns a latent representation. It is not an
  all-policy model-estimation baseline.

## Reproduce the added comparison

```sh
python3 experiments/psr_mle_toy/run_psr_ucb.py \
  --output experiments/psr_mle_toy/results/psr_ucb --seeds 100 --cap 32768
```
