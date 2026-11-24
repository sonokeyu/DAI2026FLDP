# Partially observed combination-lock experiment

Run the paired four-method experiment with:

```sh
.venv/bin/python -B experiments/psr_mle_toy/run_lock_experiment.py \
  --output experiments/psr_mle_toy/results/pocomblock \
  --seeds 30 --horizons 4 6 8 --baseline-budget 4096 --ours-budget 131072 \
  --ours-batch 4096 --epsilon 0.2

.venv/bin/python -B experiments/psr_mle_toy/run_lock_experiment.py \
  --output experiments/psr_mle_toy/results/pocomblock_graded \
  --seeds 16 --horizons 8 --family graded \
  --baseline-budget 4096 --ours-budget 131072 \
  --ours-batch 4096 --epsilon 0.2
```

The environment is inspired by the partially observed combination lock of
[Guo et al.'s PORL² paper](https://proceedings.mlr.press/v202/guo23d/guo23d.pdf)
(ICML 2023), but is **our own finite
specialization**, not a reproduction of their numerical experiment. It has
two good hidden states, one absorbing bad state, three actions, binary noisy
observations, and a noisy terminal success bit. An incorrect action in either
good state goes to bad. The good states mix after a correct action. On
alternating stages, the bad-state observation overlaps with a good state.
The eight candidate models share their action locks through the penultimate
stage and differ in the final correct-action pair. This isolates the cost of
finding a deep informative intervention; ordinary random action sequences
reach it with probability that decreases exponentially in horizon.

The **graded** variant has 16 candidates: two final correct-action pairs,
each crossed with eight terminal success probabilities from 0.82 to 0.89.
It keeps the same deep partially observed prefix and provides smaller,
non-binary model differences. All candidates remain realizable, and the
coefficient bound of 0.8 remains valid. The graded variant is run at H=8
with one seed for every candidate model.

## What each method actually does

- **ours:** ordinary complete-episode MLE; a single-model, terminal PSR
  bonus support program; exact planning in the fitted model; a frozen batch
  with a uniform last action; and a bonus-based stopping test. Because the
  candidate class has a known common prefix, the initial and earlier-stage
  telescoping errors are exactly zero. The only witness is a two-dimensional
  linear functional of the good-state prefix masses, with coefficient bound
  0.8. The terminal action-observation-success test matrix has rank three
  for every candidate, so these state masses are linear coordinates of its
  predictive state in this known model class. `lock_ours.py` solves the
  resulting box/ellipsoid program exactly.
  The box is a conservative outer relaxation of the manuscript's coefficient
  set. Thus its bonus can be larger than the manuscript's but remains an
  upper certificate. At the final stage, a uniform action experiment has
  conditional `F_H(x)^2 <= 2 A^2 Hellinger^2`; the finite-class likelihood
  calibration gives the historical energy `S = 2 A^2 beta`. The stopping
  test is `bonus <= epsilon/4`. This special class needs no nested
  diagnostic, because all candidate models have the same prefix law and the
  final uniform intervention directly observes the only possible difference.
- **omle:** exact finite-class reward-free OMLE specialization. It retains
  models within a likelihood slack, jointly maximizes trajectory TV over
  a pair of active models and a history policy, then takes one uniform and
  one disagreement-policy episode. It stops when the confidence-set TV
  diameter is at most `epsilon/4`. Its joint search is the expensive oracle
  that ours avoids. The implementation enumerates the tiny finite class and
  solves policy TV by exact dynamic programming. The diameter-based early
  stop is an added finite-class convenience for comparison; the original
  reward-free OMLE analysis prescribes a round count rather than this
  adaptive stopping rule. Pair-policy solutions are cached across runs;
  `joint_calls` counts logical oracle queries, not repeated solve time.
- **uniform:** independent uniform actions and ordinary MLE every 64
  episodes.
- **count:** least-visited action at each observed history and ordinary MLE
  every 64 episodes. This is an observation-history counting heuristic.

## Measurements and interpretation

`lock_four_method_runs.csv` has one row per fit/checkpoint.
`lock_four_method_final.csv` has one row per seed and method.
`lock_four_method_summary.csv` aggregates the paired runs. The primary
accuracy audit is exact maximum full-trajectory TV over every history
policy, computed from the known simulator **only for evaluation**. The
algorithm does not receive this TV. `success_regret` is the exact loss on
the terminal success reward after planning in the fitted model.

The two data ceilings deliberately differ: ours may use up to 131,072
episodes to obtain a numerical certificate, while the baselines use up to
4,096. Thus episode counts must be read directly; this is not a matched
sample comparison. `first_hit_episodes` is an *ex-post* accuracy milestone,
not a stopping rule available to uniform, count, or ours. The confidence
diameter is an actual stopping rule for OMLE; the bonus is an actual
stopping rule for ours. Oracle calls count MLE, exploration-bonus planning,
and joint model-pair/policy searches separately. Each method additionally
plans once for the terminal success reward when measuring `success_regret`;
this common postprocessing call is excluded from the exploration-call
columns. These are different kinds of calls; their counts alone do not
establish wall-clock speed.

## Results

At `epsilon=0.2`, the target is worst-policy TV at most `0.05`. The main
family uses 30 paired seeds. An oracle call below means either one ordinary
MLE fit, one single-model exploration plan, or one joint model-pair/policy
search, as indicated. Episodes are medians; TV values are means. Uniform
and count always run to their 4,096-episode ceiling, so their episode entry
is a budget rather than an adaptive stopping time.

| H | Method | TV ≤ 0.05 | Mean TV | Median episodes | Exploration oracle calls (median) |
|---:|---|---:|---:|---:|---|
| 4 | ours | 30/30 | 0 | 126,976 | 32 MLE + 32 plans |
| 4 | omle | 30/30 | 0 | 141 | 70.5 joint searches |
| 4 | uniform + MLE | 30/30 | 0 | 4,096 | 64 MLE |
| 4 | count + MLE | 30/30 | 0 | 4,096 | 64 MLE |
| 6 | ours | 30/30 | 0 | 114,688 | 29 MLE + 29 plans |
| 6 | omle | 30/30 | 0 | 149 | 74.5 joint searches |
| 6 | uniform + MLE | 19/30 | 0.0962 | 4,096 | 64 MLE |
| 6 | count + MLE | 27/30 | 0.0262 | 4,096 | 64 MLE |
| 8 | ours | 30/30 | 0 | 110,592 | 28 MLE + 28 plans |
| 8 | omle | 30/30 | 0 | 149 | 74.5 joint searches |
| 8 | uniform + MLE | 6/30 | 0.2712 | 4,096 | 64 MLE |
| 8 | count + MLE | 10/30 | 0.2292 | 4,096 | 64 MLE |

The graded H=8 family uses 16 seeds, one per candidate model:

| Method | TV ≤ 0.05 | Mean TV | Nonzero final TV | Median episodes | Exploration oracle calls (median) |
|---|---:|---:|---:|---:|---|
| ours | 16/16 | 0.0011 | 3/16 | 118,784 | 30 MLE + 30 plans |
| omle | 16/16 | 0.0177 | 16/16 | 101 | 50.5 joint searches |
| uniform + MLE | 12/16 | 0.0730 | 12/16 | 4,096 | 64 MLE |
| count + MLE | 12/16 | 0.0776 | 14/16 | 4,096 | 64 MLE |

Ours had no ex-post certificate violations across all 3,150 refit rows in
the two suites, and all 106 of its terminal rows stopped with bonus at most
0.05. At H=8 in the main family, mean terminal-success regret was 0 for
ours/omle, 0.2511 for uniform, and 0.2082 for count. In the graded family
the corresponding regret was 0, 0, 0.0557, and 0.0561. This one reward is
an additional audit; the TV certificate is the all-policy metric relevant
to arbitrary bounded future rewards.

The result supports the narrower computational claim: ours uses ordinary
MLE and single-model planning and makes fewer exploration rounds than this
OMLE comparator makes joint searches. The sample cost is much higher here:
for H=8, 110,592 versus 149 median episodes. These results do not establish
an end-to-end runtime advantage. The finite candidate class also saturates
at exact identification, so this fixed-epsilon experiment does not test the
asymptotic dependence of call counts on `1/epsilon²`.

Plots: [main TV](results/pocomblock/lock_tv.svg),
[main bonus](results/pocomblock/lock_bonus.svg),
[graded TV](results/pocomblock_graded/lock_tv.svg), and
[graded bonus](results/pocomblock_graded/lock_bonus.svg).

This is a realizable finite-model experiment in a POMDP with a known
shared prefix. It verifies the manuscript's key interface and stopping
mechanism in that restricted class. It does **not** test unknown dynamics
at every stage, a general PSR cone solver, misspecification, or the
manuscript's full sample-complexity constants. In particular it must not
be described as a complete implementation of the general Algorithm 1.
