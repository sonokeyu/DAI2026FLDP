# 64-hidden-state POMDP with an exact PSR audit

This is a larger structured partially observed test of ordinary MLE and
single-model bonus planning. It has 64 hidden states, four noisy observation
symbols, two actions, and horizon four. A hidden state is a triple of
four-valued symbols. An observation reports the first symbol correctly
with probability 0.75. Action zero rotates the triple; action one mixes
that rotation with another permutation. Its mixture probability is
unknown separately for each of three transition stages and each value
of the first hidden symbol: **12 continuous parameters** in total.

The core tests are all observation strings of length three under action
zero, shortened near the horizon. The stage core dimensions are 64, 64,
16, and 4. The length-three core is an invertible tensor product of the
four-symbol emission matrix. Its signed-input stability bound is
`(7/4)^3 = 5.359375`. The measured reachable rank is 16, and the learner
does not receive that rank. Tests verify that stage operators factor
through the shorter cores, including the action-dependent stage-two
transition. This gives an exact PSR for the 64-state POMDP, even though
one noisy observation is far from identifying its hidden state.

The learner performs numerical multistart ordinary MLE on all complete
logged trajectories. At each refit it solves the stage support programs
and plans exactly in the single fitted model for their bonus. Frozen
training and nested diagnostic batches use the same design as the
four-state experiment. The evaluator, separate from the learner, computes
the exact maximum TV over all observation-history policies by dynamic
programming on the two models' hidden-state masses. A test recomputes
the returned maximizing policy's TV directly from its trajectory law.

## Batch sweep

The true weights are listed in `queue_pomdp.py`. Four paired seeds were
run at ε=0.4. Every run stopped and had **nonzero final worst-policy TV**.
All recorded refits passed the ex-post `TV <= bonus` audit. The initial
fit's exact worst-policy TV is 0.3352 in every run.

| Episodes per training/diagnostic batch | MLE calls | Planning calls | Median total episodes | Median final TV |
| ---: | ---: | ---: | ---: | ---: |
| 10 million | 9 | 9 | 480 million | 0.0001684 |
| 30 million | 4 | 4 | 540 million | 0.0001650 |
| 100 million | 3 | 3 | 1.2 billion | 0.0000853 |
| 1 billion | 3 | 3 | 12 billion | 0.0000238 |

The first two rows show the computational tradeoff clearly: increasing
each batch threefold reduced the number of MLE and planning calls from
nine each to four each while changing total episodes from 480 million
to 540 million. Increasing the batch further provided little additional
call reduction and substantially increased sample cost. The stopping
threshold was `3ε/16 = 0.075`; the target was all-policy `TV <= ε/4 = 0.1`.
The exact TV audit passed, but the final errors are much smaller than
the target. The experiment illustrates interface and call behavior; it
does not show sample efficiency.

## Reproduce

From the repository root, install the dependencies and run:

```sh
UV_CACHE_DIR=/tmp/psr-uv-cache uv pip install --python .venv/bin/python cvxpy scipy matplotlib
PYTHONPATH=experiments/psr_mle_toy .venv/bin/python -B -m unittest -v experiments.psr_mle_toy.test_queue
.venv/bin/python -B experiments/psr_mle_toy/run_queue_sweep.py \
  --output experiments/psr_mle_toy/results/queue_pomdp \
  --seeds 4 --batches 10000000 30000000 100000000 1000000000 \
  --epsilon .4 --rounds 12
MPLCONFIGDIR=/tmp/psr-mpl-cache .venv/bin/python -B \
  experiments/psr_mle_toy/plot_queue.py \
  --input experiments/psr_mle_toy/results/queue_pomdp \
  --output experiments/psr_mle_toy/results/queue_pomdp
```

The generated `queue_sweep_runs.csv` contains every refit;
`queue_sweep_final.csv` and `queue_sweep_summary.csv` contain final and
aggregated results. `queue_resources.pdf` and `queue_certificate.pdf`
show call/sample cost and bonus/TV respectively. Episode counts use exact
multinomial draws from enumerated physical trajectory laws, with the same
distribution as that many independent physical episodes; the program
does not simulate billions of trajectories one by one.

## Scope

The 64-state system is structured so an exact length-three core and an
exact all-policy TV evaluator remain computationally possible at H=4.
It is substantially larger than the four-state test but is still a
synthetic benchmark, not a long-horizon application. The 12-dimensional
continuous MLE is solved numerically and may be a local optimum. The
bonus uses an effective class size of 100 rather than a proved covering
number for this continuous class. Consequently the observed stopping
inequality is an empirical audit, **not** a theorem-level confidence
certificate. There is no exact continuous-class OMLE joint-search
baseline in this scale test.

## Additional baselines

Uniform random plus continuous MLE, observation-history count exploration
plus continuous MLE, and a finite-grid OMLE approximation were also run
on the same 64-state environment. Each used a budget of 256 physical
episodes and four seeds; MLE was refit every 64 episodes. At this budget,
uniform+MLE reached the target `TV <= ε/4` in 1/4 seeds, count+MLE in
1/4, and finite-grid OMLE in 0/4. Their median final TVs were 0.1195,
0.1198, and 0.1977 respectively. OMLE made 128 joint model-pair/policy
searches per seed and did not stop before the budget ended.

The OMLE model set contains 64 independently sampled grid points from
`{0,.25,.5,.75,1}^12`; it excludes the true parameter. Its confidence
set and max-TV pair search are exact **within this sampled grid**, while
the continuous MLE controls use numerical multistart likelihood
optimization. This is a practical finite-grid baseline, not an exact
continuous-class OMLE guarantee. Its sample budget is much smaller than
ours' stopping batches, so the comparison measures behavior at the stated
budgets rather than equal-sample performance. See
`queue_baselines_runs.csv` and `queue_baselines_summary.csv`.

## Matched four-method comparison

For a fairer comparison, we also ran all four methods on one shared finite
class of 25 models. It contains the true 12-parameter model and 24 models
that differ from it in one parameter by a large amount. Each method sees
the same class. Ours, uniform+MLE, and count+MLE use exact ordinary MLE
over these 25 models. OMLE performs its exact confidence-set and
all-policy pair search within that class. The run uses eight paired seeds,
ε=0.4, and a common cap of 384 physical episodes. Ours keeps its batch
schedule and refits after each 192 collected episodes. The baselines
retain their own updates. The TV curve carries each method's latest fit
forward to each shared episode checkpoint.

At 384 episodes, the median worst-policy TV was 0.0710 for ours, 0.0344
for OMLE, 0.0322 for uniform+MLE, and 0.0303 for count+MLE. The methods
met `TV <= ε/4 = 0.1` in 6/8, 8/8, 8/8, and 8/8 seeds respectively.
Ours made 3 MLE and 3 single-model planning calls. The controls made 6
MLE calls. OMLE made 192 joint model-pair/policy searches. Every method
used the same episode budget, and all TV evaluations were exact over the
full observation-history policy class.

The shared candidate set is deliberately small. It tests whether methods
can distinguish models that differ in a hidden-state-specific transition
parameter. Its 25 models do not cover the full continuous family. Results
therefore establish a matched comparison on this finite 64-state task,
while the continuous batch sweep above remains the larger-class stress
test. Raw paths and aggregates are in `matched/matched_runs.csv` and
`matched/matched_summary.csv`. The plot is `matched/matched_tv.pdf`.

## Accuracy sweep and stopping certificate

The final run varies ε over 0.4, 0.2, 0.1, and 0.05 on the same finite
class. The ours batch is set to `10,000,000 × (0.4/ε)^2`. Across two
seeds, ours stops after **8 MLE calls and 8 single-model planning calls
at every ε**. Total episodes scale with the batch, while the number of
updates stays fixed in this sweep. The exact sup TV is zero at stopping
for these finite-class runs. Every recorded round satisfies
`sup TV <= fitted bonus`, and both seeds stop only when the true TV is
below ε/4. At ε=.4 the initial selected model already meets ε/4, but the
bonus takes seven more rounds to cross the stopping threshold. At ε=.2
the delay is also seven rounds. At ε=.1 and .05 the first accurate fit
appears at round two and stopping occurs six rounds later. This directly
shows that the certificate is sufficient for stopping, while its upper
bound can remain loose for several rounds after the model is accurate.

OMLE was run on the same 25 models with a diameter stopping rule and a
3,000 joint-search cap. It stopped after 692 calls for ε=.4 and 1,304
calls for ε=.2. It had not stopped at the cap for ε=.1 or .05. This
supports increasing OMLE effort as the target tightens and shows constant
ours update counts over the tested range. The OMLE points at smaller ε
are censored at 3,000 calls, so this experiment does not estimate a
precise growth exponent. Two seeds and four ε values are a proof-of-
mechanism check rather than a scaling study.

The run files are `epsilon/epsilon_runs.csv`,
`epsilon/epsilon_summary.csv`, `epsilon/omle_epsilon_runs.csv`, and
`epsilon/omle_epsilon_final.csv`. The figures are
`epsilon/epsilon_certificate.pdf` and `epsilon/omle_epsilon_calls.pdf`.
The initial tie convention selects a fixed nontrue candidate, so the
learner does not start with the hidden true model.

## Equal-episode accuracy sweep

This experiment compares ours and finite-class OMLE at identical total
episode budgets. To keep final errors measurable at larger budgets,
both use the same realizable **73-model** class: 24 one-coordinate
alternatives shifted by 0.6, 24 shifted by about 0.15, 24 shifted by
0.05, and the truth. All 73 models are distinct. Ours' initial MLE tie
selects an alternative with exact all-policy TV **0.1241** from truth;
OMLE initially holds the whole class in its confidence set. There are
eight seeds per setting.

Ours uses `B = 2,000 × (0.4/ε)^2` episodes in each of six experiments
per frozen round. Two such rounds give `N = 12B` physical episodes,
followed by its third MLE and single-model planning call. OMLE gets
the same `N`, executing the three PSR core-intervention policies and
then updating its likelihood confidence set after every joint search.
The confidence set is intersected with the previous one. All reported
OMLE searches involved at least two active candidates.

| ε | Episodes each | Ours MLE / planning calls | OMLE joint searches | Median sup TV: ours / OMLE | `TV < 0.01`: ours / OMLE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 | 24,000 | 3 / 3 | 8,000 | 0.00808 / 0.00715 | 6/8 / 8/8 |
| 0.283 | 48,000 | 3 / 3 | 16,000 | 0.00792 / 0.00771 | 7/8 / 7/8 |
| 0.2 | 96,000 | 3 / 3 | 32,000 | 0.00796 / 0.00662 | 5/8 / 7/8 |
| 0.141 | 192,000 | 3 / 3 | 64,000 | 0 / 0 | 8/8 / 8/8 |

Both methods have median TV below 0.01 at the first three budgets,
with a few individual runs as high as 0.01105. At 192,000 episodes all
eight seeds of both methods recover the truth exactly, a finite-class
effect. Every run meets the paper's looser `TV ≤ ε/4` target. The
episode totals are exactly matched, but the first update is not: ours
collects `N/2` episodes before its first refit, and OMLE collects three
before its next joint search.

The three ours calls are fixed by this two-batch schedule. Its bonus
did not trigger stopping in any of these 32 runs, so this sweep tests
accuracy and interface calls **at an allocated sample budget**, not
the theorem's data-dependent stopping bound. The stopping sweep below
uses a separate 49-model class and much larger batches. The class,
sample totals, and stopping rule must be kept distinct when discussing
the two experiments. Oracle calls also do not measure wall-clock cost
per call.

Reproduce this experiment with:

```sh
.venv/bin/python -B experiments/psr_mle_toy/run_queue_equal_budget.py \
  --output experiments/psr_mle_toy/results/queue_pomdp/equal_budget --seeds 8
MPLCONFIGDIR=/tmp/psr-mpl-cache .venv/bin/python -B \
  experiments/psr_mle_toy/plot_queue_equal_budget.py \
  --input experiments/psr_mle_toy/results/queue_pomdp/equal_budget \
  --output experiments/psr_mle_toy/results/queue_pomdp/equal_budget
```

The full per-seed measurements are in `equal_budget/equal_budget_runs.csv`,
the aggregates in `equal_budget/equal_budget_summary.csv`, and the figure
in `equal_budget/equal_budget.pdf`.

## Actual bonus stopping across accuracy targets

The equal-episode sweep above stops after a fixed number of allocated
batches, so its three calls do not measure the paper's data-dependent
stopping rule. This follow-up runs that rule until its fitted bonus first
falls below `3ε/16`. It retains the same realizable 49-model class and
common initial candidate (initial exact sup TV 0.1241). Each setting
uses four seeds and the same 64-state POMDP.

For this empirical scaling study, the frozen batch size is
`B = 10,000,000 × (0.4/ε)^2 / sqrt(1 + log2(0.4/ε))`, rounded to an
integer. The mild logarithmic divisor makes changes in the number of
stopping rounds visible over four accuracy settings while retaining
inverse-square sample scaling up to logarithms. It is a declared
experimental schedule, not a numerical instantiation of the manuscript's
appendix batch formula.

| ε | Episodes per batch | MLE and planning calls, each | Median episodes to stop | First `TV ≤ ε/4` round | Median stop bonus | Median stop TV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 | 10,000,000 | 9 | 480,000,000 | 2 | 0.07040 | 0 |
| 0.2 | 28,284,272 | 11 | 1,697,056,320 | 2 | 0.03731 | 0 |
| 0.1 | 92,376,048 | 14 | 7,205,331,744 | 2 | 0.01804 | 0 |
| 0.05 | 320,000,000 | 15 | 26,880,000,000 | 2 | 0.00934 | 0 |

All four seeds stopped at every ε. Every recorded round passed the
**ex-post** exact `sup TV ≤ bonus` check, and all stopping fits passed
`sup TV ≤ ε/4`. The number of MLE and planning calls rose from 9 to 15
while `1/ε` rose eightfold and episodes rose 56-fold. The fit first met
the target at round two in every setting, so the bonus delayed stopping
by 7, 9, 12, and 13 further calls. This measures how loose the
certificate is on this task. The final TV is zero because large batches
identify the truth in this finite realizable class; the continuous
stress check below addresses nonzero fitted error.

The call trend depends on the stated batch schedule. A batch exactly
proportional to `1/ε²` can produce a flat call curve over this range,
which is consistent with the paper's near-independent oracle bound.
Neither curve by itself proves an asymptotic rate. The numerical bonus
uses the finite class size in its calibration and exact model planning;
the TV audit checks its behavior on the sampled runs, not the full
high-probability theorem.

Reproduce the finite stopping sweep with:

```sh
.venv/bin/python -B experiments/psr_mle_toy/run_queue_stopping_scaling.py \
  --output experiments/psr_mle_toy/results/queue_pomdp/stopping_scaling \
  --seeds 4 --rounds 32
```

### Continuous-model stress check

The same near-inverse-square batch schedule was also run with numerical
MLE over the 12-parameter continuous POMDP class, for two seeds. This
removes exact finite-class model recovery: every final worst-policy TV is
strictly positive. All eight runs stopped by the bonus, passed the exact
TV target, and passed the ex-post `TV ≤ bonus` audit on every recorded
round.

| ε | MLE and planning calls, each | Median episodes | Median final TV |
| ---: | ---: | ---: | ---: |
| 0.4 | 9 | 480,000,000 | 0.0001739 |
| 0.2 | 12 | 1,866,761,886 | 0.0000784 |
| 0.1 | 15 | 7,759,587,612 | 0.0000549 |
| 0.05 | 17 | 30,720,000,000 | 0.0000200 |

This continuous result is an empirical stress check. The multistart
numerical MLE may be a local optimum, and the effective class-size
parameter in its bonus is not a proved covering number. Therefore its
observed `TV ≤ bonus` inequality is not a theorem-level certificate.

```sh
.venv/bin/python -B experiments/psr_mle_toy/run_queue_continuous_stopping.py \
  --output experiments/psr_mle_toy/results/queue_pomdp/continuous_stopping \
  --seeds 2 --rounds 32
MPLCONFIGDIR=/tmp/psr-mpl-cache .venv/bin/python -B \
  experiments/psr_mle_toy/plot_queue_stopping_scaling.py \
  --input experiments/psr_mle_toy/results/queue_pomdp/stopping_scaling \
  --continuous-input experiments/psr_mle_toy/results/queue_pomdp/continuous_stopping \
  --output experiments/psr_mle_toy/results/queue_pomdp/stopping_scaling
```

The per-round paths, summaries, and figure are under
`stopping_scaling/` and `continuous_stopping/`.

### OMLE joint-search stopping on the same class

OMLE was also rerun on the same fixed 49-model class and truth, with
four seeds per ε. Each round selects the maximum-TV
pair in the current likelihood confidence set, intersects the updated
confidence set with the previous one, and collects three core-action
intervention episodes (the specialization of [Eq. 10 in the original
paper](https://arxiv.org/html/2209.14997) to this core list). We add an
**early-stop check** when that set's
exact all-policy TV diameter is at most `ε/4`; reward-free OMLE in the
paper instead prescribes a number of rounds in advance. Each setting
uses its own confidence slack based on its call cap. None of these 16
runs reached its cap; the truth remained in the confidence set and the
fitted model met the target in every run.

| ε | Ours: MLE / planning calls | OMLE: median joint searches | Ours: median episodes | OMLE: median episodes |
| ---: | ---: | ---: | ---: | ---: |
| 0.4 | 9 / 9 | 1,267 | 480,000,000 | 3,801 |
| 0.2 | 11 / 11 | 9,698 | 1,697,056,320 | 29,094 |
| 0.1 | 14 / 14 | 21,302.5 | 7,205,331,744 | 63,907.5 |
| 0.05 | 15 / 15 | 32,119.5 | 26,880,000,000 | 96,358.5 |

The likelihood slack uses `β = log(K |Π_exp| |Θ| / 0.1)`, with the
theorem's unspecified numerical constant set to one. This is an
experimental calibration; the observed truth coverage is not a
formal guarantee for this exact constant. OMLE's observed joint-search
count increases as the target tightens,
but the increment is smaller at the two hardest settings. The usual
inverse-square accuracy dependence is an **upper-bound scale**, not a
claim that every fixed finite instance follows an exact quadratic
curve. Ours uses far fewer MLE and single-model planning calls here,
while OMLE needs vastly fewer episodes to reach its stopping rule.
These are stopping costs with different sample totals. The equal-episode
sweep above is the matched-budget accuracy comparison; neither table
equates one joint search with one MLE or one planning call in runtime.
Both methods begin with zero collected episodes and the same model class,
but their first updates do not use similar sample amounts:
at ε=0.4 ours collects 60 million episodes before refitting, whereas
OMLE collects three before its next joint search. Consequently the
stopping plot cannot serve as a matched-sample comparison.

```sh
.venv/bin/python -B experiments/psr_mle_toy/run_queue_omle_stopping_scaling.py \
  --output experiments/psr_mle_toy/results/queue_pomdp/omle_stopping \
  --seeds 4 --base-cap 5000 --max-cap 50000
MPLCONFIGDIR=/tmp/psr-mpl-cache .venv/bin/python -B \
  experiments/psr_mle_toy/plot_queue_stopping_comparison.py \
  --ours experiments/psr_mle_toy/results/queue_pomdp/stopping_scaling \
  --omle experiments/psr_mle_toy/results/queue_pomdp/omle_stopping \
  --output experiments/psr_mle_toy/results/queue_pomdp/stopping_scaling
```

The comparison plot is `stopping_scaling/stopping_comparison.pdf`.
Per-seed OMLE paths and censoring status are in `omle_stopping/`.
