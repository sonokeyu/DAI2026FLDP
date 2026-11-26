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

This experiment compares ours and finite-class OMLE at identical episode
budgets as the requested accuracy tightens. Both algorithms use the same
realizable **49-model** class: 24 one-coordinate alternatives shifted by
0.6, 24 shifted by 0.15, and the true model. The common initial
likelihood tie selects the alternative with the largest exact all-policy
TV from truth, **0.1241**. The eight seeds are paired by index.

For ε = 0.4, 0.2, 0.1, ours uses a batch size
`B = 32 × (0.4/ε)^2`. Two exploration batches use `N = 12B` physical
episodes, followed by the third MLE and single-model planning call.
OMLE receives the same `N`, updates its likelihood scores after each
two-episode round, and recomputes the maximum-TV model pair and policy
in its confidence set each round. Its confidence radius uses that
setting's own `N`. All reported OMLE joint searches had at least two
active candidates; none were singleton no-ops.

| ε | Episodes each | Ours MLE / planning calls | OMLE joint searches | Median final sup TV: ours / OMLE | `TV ≤ ε/4`: ours / OMLE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 | 384 | 3 / 3 | 192 | 0.0983 / 0.0314 | 5/8 / 8/8 |
| 0.2 | 1,536 | 3 / 3 | 768 | 0.0200 / 0.0242 | 8/8 / 8/8 |
| 0.1 | 6,144 | 3 / 3 | 3,072 | 0.0238 / 0.0172 | 6/8 / 7/8 |

The sample budgets are exactly matched. The target is not met in all
seeds at ε = 0.4 or 0.1, and OMLE has better median TV at those two
budgets. At ε = 0.2, ours has a slightly smaller median TV. The final
errors remain nonzero at the tightest setting. These measurements support
the call-count tradeoff **at the prescribed budget**. The number of ours
calls is fixed here by the chosen two-batch schedule, while OMLE's number
is fixed by its two-episode update schedule. Thus this sweep does not
independently establish the theorem's stopping-call bound.

The ours bonus did not trigger stopping in any of these 24 runs. The
larger-batch certificate sweep above separately checks that the bonus
eventually stops and upper-bounds exact sup TV, at much higher sample
cost. Neither sweep establishes an asymptotic exponent from three or
four accuracy settings. Oracle calls here also do not measure wall-clock
cost per call; a joint search, MLE fit, and single-model plan are
different computations.

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
