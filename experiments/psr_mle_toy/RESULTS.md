# PSR MLE toy experiments

These experiments use the manuscript's three-stage routing example and its
observable Markov specialization. Experiments run on synthetic trajectories.

## Reproduce

From this directory, with Python 3.9+ and NumPy installed:

```sh
python -B -m unittest -v test_toy test_routing
python -B run_experiments.py --output results
python -B run_suite.py --output reproduced
python -B plot_comparison.py reproduced/comparison_summary.csv reproduced/comparison.tex
latexmk -pdf -outdir=reproduced reproduced/comparison.tex
```

NumPy 2.3.5 was used for the archived runs. The PDF figure uses PGFPlots.
The original dependency-free Section 4 experiment has its own instructions in
[README.md](README.md).

## Equal episode budgets

The routing models differ only at the prefix `(-, R)` under mode `a`.
The competing Bernoulli probabilities are 0.2 and 0.2 + Δ. Mode `b`
always emits a fair bit. Each setting uses 100 seeds with each candidate
serving as truth, for 200 runs per method and Δ. The stage-two controls
share the cue, action, and outcome uniforms by seed. The complete PSR and
OMLE methods sample from their own controller schedules.

The stage-two methods update every eight episodes, with an empirical
support energy of 1. Both test controls choose `a` with probability 1/2.
The full PSR algorithm uses one episode per training and initial-core
block and four nested episodes per update, also totaling eight episodes.
It uses the manuscript's value of S and is evaluated throughout the
fixed budget. The OMLE comparator follows Algorithm 2 with Eq. (10) from
the [authors' arXiv full version](https://arxiv.org/abs/2209.14997).
It enumerates model pairs in the shrinking likelihood confidence set and
computes the exact maximum policy discrepancy. It collects one episode
for each of the two intervention policies before updating. The toy
likelihood slack is log(T |Θ| / 0.1), with multiplier 1. At a checkpoint,
the reported estimate is the highest-likelihood member of the confidence
set with a fixed tie rule. The comparator is implemented in this repository.

Mean worst-policy TV at 256 episodes:

| Method | Δ = 0.1 | Δ = 0.3 | Δ = 0.6 | Exploration updates |
| --- | ---: | ---: | ---: | ---: |
| Stage 2 bonus only | 0.0250 | 0.0750 | 0.1500 | 32 |
| Stage 2 bonus + test | 0.0225 | 0.0533 | 0.0675 | 32 |
| Random route + test | 0.0140 | 0.0060 | 0 | 32 |
| Full fitted PSR | 0.0158 | 0.0128 | 0.0045 | 32 |
| Reward-free OMLE | 0.0095 | 0 | 0 | 128 |

The full PSR loop improves on the stage-two proxy, and random routing
with tests reaches lower final TV than the PSR loop at all three gaps.
At Δ = 0.1, the stage-two bonus arm averages 1.9 informative episodes
(negative cue, right route, mode a) in 256 episodes, versus 31.8 for random
routing with tests, 14.4 for the full PSR loop, and 63.4 for OMLE. Thus
the proxy can reduce its fitted coverage bonus while continuing to miss
the history that separates the models. OMLE attains the lowest final
TV in this comparison and updates four times as often. Counts of updates
describe the schedule. A joint model-pair search and one fitted-model
planning call do different work.

[Figure](results/comparison.pdf) · [Curves and paired bootstrap intervals](results/comparison_summary.csv) · [Per-seed trajectories](results/comparison_runs.csv.gz)

## Fixed TV targets

We reran the two main methods for up to 1,024 episodes, with 100 seeds and
both choices of the true model for each gap. At every eight episodes, we
measure mean worst-policy TV over those 200 runs. For each target
(0.020, 0.010, and 0.005), the reported cost is the first checkpoint at
which the mean is at or below the target and stays there through episode
1,024. This retrospective measure excludes temporary crossings. It uses
truth only to evaluate the recorded runs. A model update counts a
post-batch refit or confidence-set update. The two operations have different
computational costs.

The primary comparison retains the original update intervals of two
episodes for **omle** and eight episodes for **ours**. The sensitivity
check also runs **omle** with frozen exploration for eight episodes per
update. All OMLE conditions use the 1,024-episode horizon in their
likelihood slack. The fixed-budget diagnostic design for **ours** is the
same as in the 256-episode experiment.

Costs at a sustained mean TV of **0.010**:

| Gap | omle episodes / updates | ours episodes / updates | omle at matched interval |
| --- | ---: | ---: | ---: |
| 0.1 | 208 / 104 | 1,008 / 126 | 208 / 26 |
| 0.3 | 88 / 44 | 416 / 52 | 88 / 11 |
| 0.6 | 56 / 28 | 136 / 17 | 56 / 7 |

At the hardest gap, the **ours** result for 0.010 is at the budget boundary:
the mean at episode 1,024 is exactly 0.010, and only 53.6% of the paired
bootstrap draws reach the target within budget. The run-level aggregate
for **ours** misses 0.005 at that gap.

With the original update intervals, **ours** uses fewer updates on the
three targets at Δ = 0.6 and more updates in the other five cells in which
both methods reach the target. It uses more episodes in every comparable
cell. Matching the update interval leaves the **omle** checkpoint mean TV
curves unchanged in this toy experiment, so **omle** uses fewer episodes
and updates in each reachable comparison. These results quantify the costs
of this routing experiment under the selected update schedules.

The figure shows the primary two-method comparison. It marks a miss at
the applicable budget cap. The CSV includes the same-interval control and
marginal percentile bootstrap intervals for the episode count. Resampling
pairs both truth orientations by seed. An empty upper bound indicates the
corresponding percentile did not reach the target by the cap.

```sh
python -B run_target_accuracy.py --output reproduced
python -B plot_target_accuracy.py reproduced/target_summary.csv reproduced/target_accuracy.tex
latexmk -pdf -outdir=reproduced reproduced/target_accuracy.tex
```

[Cost figure](results/target_accuracy.pdf) · [Target costs](results/target_summary.csv) · [TV curves](results/target_curves.csv) · [Per-seed trajectories](results/target_runs.csv.gz)


## Traditional exploration baselines

**uniform** chooses route and processing mode independently and uniformly.
**count** uses the cumulative visit count of each route and mode pair for
the observed cue. At the start of each block it freezes a policy that draws
uniformly among that cue's least visited pairs. Each update fits the same
two-candidate class by ordinary MLE. Both methods use the same 100 seeds,
two truth orientations, eight-episode update interval, and 1,024-episode cap
as **ours**. The plot also includes **omle** at that same interval. Costs
use the sustained mean TV criterion above.

Cost in episodes / updates at a mean TV of **0.010**:

| Gap | omle | ours | uniform | count |
| --- | ---: | ---: | ---: | ---: |
| 0.1 | 208 / 26 | 1,008 / 126 | 440 / 55 | 400 / 50 |
| 0.3 | 88 / 11 | 416 / 52 | 224 / 28 | 216 / 27 |
| 0.6 | 56 / 7 | 136 / 17 | 80 / 10 | 56 / 7 |

Both traditional baselines reach 0.010 with fewer episodes and updates
than **ours** at each gap. At Δ = 0.1 and episode 256, the mean counts
of the distinguishing combination are 14.44 for **ours**, 31.81 for
**uniform**, and 32.09 for **count**. In the hardest case at TV 0.005,
**count** reaches
the target at episode 952 in the observed mean. **uniform** and **ours**
miss it at the cap. The upper bootstrap bound for that **count** result
also runs past the cap, so the exact crossing cost is uncertain.
**Count** has lower cost than **uniform** at 0.010 across the three gaps,
though their ordering varies across the other targets.

Generate the added figure from the same expanded target results.

```sh
python -B run_target_accuracy.py --output reproduced
python -B plot_baselines.py reproduced/target_summary.csv reproduced/traditional_baselines.tex
latexmk -pdf -outdir=reproduced reproduced/traditional_baselines.tex
```

[Same interval comparison figure](results/traditional_baselines.pdf) · [All five configurations](results/target_summary.csv)


## Batch size and the episode/update tradeoff

The matched eight-episode comparison above controls the update interval.
The batch sweep asks how each method can trade episodes for fewer updates
when allowed to choose its own interval. It uses the same three gaps,
200 trials per gap and setting, and mean TV targets of 0.020, 0.010, and
0.005, with a cap of 4,096 episodes. Each setting is evaluated at its
own refits, and a target must hold through the final recorded checkpoint.

**ours**, **uniform**, and **count** use batch sizes of 8, 32, 128, 512,
and 2,048 episodes. **omle** uses the same grid plus the original size
of 2. The larger OMLE batches repeat its two intervention designs under
the same frozen policy. For **ours**, a batch of size B has four training
blocks of B/8 episodes apiece and B/2 nested diagnostic episodes. This
scales both parts of the eight-episode experiment at the same rate.
These are experimental batch schedules. The appendix schedule is studied
separately below. The update counts exclude the empty-data fit. An
OMLE update and an ordinary MLE refit with single-model planning have
different work per call.

Costs at sustained mean TV ≤ 0.010, shown as **episodes / updates**:

| Gap | ours, batch 8 | ours, batch 128 | ours, smallest batch with one update | omle, batch 2 | omle, smallest batch with one update |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 1,128 / 141 | 640 / 5 | 2,048 / 1 | 202 / 101 | 512 / 1 |
| 0.3 | 416 / 52 | 384 / 3 | 512 / 1 | 82 / 41 | 128 / 1 |
| 0.6 | 136 / 17 | 256 / 2 | 512 / 1 | 50 / 25 | 128 / 1 |

The one-update settings are the smallest among the batches tested.
The sweep shows the tradeoff against **omle** at its original interval:
**ours** needs far fewer updates when it commits to a larger batch.
Batching **omle** also lowers its update count. At target 0.010, every
observed **ours** point is dominated by a tested **omle** point with
at most the same episodes and updates, and a strict improvement in at
least one of those costs. These are point estimates for the routing toy
under the tested batch grid. The CSV records paired-bootstrap intervals
and indicates cases in which a target was missed at the cap.

The figure joins the non-dominated points within each method and marks
the original intervals separately. The CSV also contains **uniform** and
**count**. In the per-run archive, each character of `fit_error` denotes
whether the fitted model was wrong at the corresponding update, including
the empty-data fit at position zero. Its exact worst-policy TV is zero
for a correct fit or Δ/2 for a wrong fit. The second retained statistic is
the number of distinguishing tests by the cap.

```sh
python -B run_batch_sweep.py --output reproduced
python -B plot_batch_sweep.py reproduced/batch_summary.csv reproduced/batch_tradeoff.tex
latexmk -pdf -outdir=reproduced reproduced/batch_tradeoff.tex
```

[Batch tradeoff figure](results/batch_tradeoff.pdf) · [Cost summary](results/batch_summary.csv) · [TV curves](results/batch_curves.csv) · [Per-seed fit traces](results/batch_runs.csv.gz)


## Stopping and nested diagnostics

On the three-stage environment, all stage supports are solved by finite
enumeration of corners, edge intersections, and ellipse extrema after
projecting their coefficient sets to one or two dimensions. Backward
induction gives the fitted bonus exactly up to floating-point arithmetic.
The simulator draws counts from each frozen labeled controller's exact
multinomial law. These counts represent IID synthetic episodes, including
for the very large appendix batches. The action and label probabilities
are common across candidates, so the physical counts suffice for the
MLE comparison. The simulator retains those probabilities when computing
logged law distances and likelihood calibration. The duplicate final
action has no effect on observations or rewards and is marginalized.

The stopping rule is the paper's 3ε/16 threshold with an exact expected
bonus in place of Monte Carlo evaluation. We use β and S from the
manuscript and compute the actual conditional Hellinger risk at every
refit. The appendix condition uses Appendix D batch formulas with c = 64.
The resized conditions use 16,000/ε² episodes per block. One uses nested
cuts and the other spends its extra block on the direct diagnostic.
Both the deterministic and noisy routing environments are used in both
truth orientations, with 20 seeds per cell.

All 720 runs stop with exact worst-policy TV 0. Every refit satisfies
the recorded calibration inequality. The four observable rewards planned
after stopping all have zero regret. At ε = 0.5, the resized conditions
use a median of 13 MLE calls and 3.84 million episodes. The appendix
condition uses 3 MLE calls and approximately 274 billion simulated episodes
at that ε. Across the three ε settings, the median appendix MLE count
stays at 3 while its episode count grows roughly fourfold whenever ε halves.

The two resized conditions have the same stopping performance in this
grid. The frozen-controller check isolates a difference. With an empty
buffer and the fixed tie convention, the deterministic example has
direct diagnostic TV 0 and nested squared Hellinger distance 1/12.
The stage-one uniform intervention in the training block can also
reveal that discrepancy, which explains why this environment can show
the added information without showing a final advantage from nesting.
The exact Section 4 example separately checks the specified cut branch's
contribution of 1/36 to squared Hellinger distance.

[Run summary](results/suite_summary.json) · [Final states](results/routing_final.csv) · [Full traces](results/routing_rounds.csv.gz)

## Rank at fixed core dimension

The fully observable chain uses one singleton core test for each of eight
possible states. It starts at state 0 and uses action 0 to stay and action
1 to advance modulo the candidate's ring size. The same model class
contains rings of sizes 1, 2, and 4 on every run. The learner fits a model
from observations and receives no true rank input. The evaluator computes
reachable ranks from the simulator. The number of one-hot vectors reachable
at each stage gives the rank, and its maximum is the ring size at horizons
at least as long as the ring. The projected supports are coordinate bounds,
so they are solved from the reencoded visit counts. This sweep uses resized
batches and an experimental 128-call limit.

All 540 runs stop with worst-policy TV 0. At ε = 0.5, the median MLE and
exploration planning calls are:

| Horizon | Rank 1 | Rank 2 | Rank 4 |
| --- | ---: | ---: | ---: |
| 4 | 2 | 3 | 5 |
| 6 | 2 | 3 | 5 |
| 8 | 3 | 4 | 5 |

These values measure dependence on reachable rank within the selected
deterministic family. The small sweep gives no estimate of an asymptotic
exponent. The general fused-dual cone program and other core-test
geometries are outside the implemented specializations.

[Rank sweep data](results/rank_runs.csv)
