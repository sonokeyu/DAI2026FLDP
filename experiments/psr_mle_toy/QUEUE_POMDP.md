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
