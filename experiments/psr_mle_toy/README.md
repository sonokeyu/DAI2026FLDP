# Three-stage PSR MLE toy experiment

[Multi-stage PSR experiment](MULTISTAGE_POMDP.md) runs the finite-class
ordinary-MLE exploration loop on rank-controlled noisy POMDPs at horizons 3–5, with exact
all-policy TV audits and four-method results.

[Partially observed combination-lock experiment](COMBINATION_LOCK.md) gives
the newer four-method comparison on a deeper POMDP, with exact all-policy TV,
episode costs, oracle calls, and a specialized bonus stopping certificate.

[Theory validation](THEORY_VALIDATION.md) reports the oracle interfaces,
call counts, and stopping certificate under the paper’s assumptions.

[Baseline scope and literature](BASELINES.md) records the direct and
traditional baselines, the PSR-UCB toy specialization, and recent related
papers that were screened for comparability.

[Full experiment report](EXPERIMENTS.md) combines the certificate-matched
OMLE comparison, exact E2D toy specialization, multi-coordinate sweep, and
the existing rank, horizon, and reward-transfer audits.

The expanded suite, full three-stage implementation, reward-free OMLE
comparison, and rank sweep are described in [RESULTS.md](RESULTS.md).

This is a dependency-free experiment for the **Section 4 mechanism** in
*Learning Predictive State Representations with Ordinary Maximum Likelihood
and Model Planning*. Run from this directory with Python 3.9 or newer:

```sh
python3 -m unittest -v test_toy.py
python3 run_experiments.py --seeds 100 --output results
```

`run_experiments.py` prints the exact checks and Monte Carlo summary, and
optionally writes `summary.json` plus one row per condition and seed in
`runs.csv`. Seeds are paired between conditions.

## What is implemented

- The paper's three-stage binary-cue, route, and processing-mode example.
  The true model makes type 1 only after `(+, R)`; the competing model makes
  type 1 after either cue on route `R`. Mode `a` reveals type and mode `b`
  returns an independent fair bit.
- Exact trajectory distributions and maximum trajectory TV over all 64
  deterministic history-dependent policies. For this tiny environment,
  randomized policies cannot increase the maximum TV.
- Ordinary complete-episode MLE over the two-model realizable class. Logged
  action-randomization probabilities are common to both models and therefore
  cancel from the likelihood comparison. Ties initially select the wrong model
  to expose the need for exploration.
- The **exact stage-two support value in this example**, with the paper's
  stage-two coefficient geometry and a buffer of one-hot type rows. For a
  query of type `i` with `n_i` matching historical rows, it is
  `min(2, sqrt(S/n_i))`, taking value 2 when `n_i=0`. Historical physical
  prefixes are reencoded under every new fit.
- A stage-two mechanism loop. It rolls in with the bonus-maximizing route,
  gathers a batch, refits, and checks a stage-two threshold. The active
  condition replaces processing mode `b` with revealing mode `a` with
  probability 1/2; the passive condition never does so.

## Exact identities and default results

The code asserts the paper's stage-two unseen support `2`, hidden-policy TV
`0`, revealing-policy TV `1/2`, true optimal reward `3/4`, and the `1/36`
Hellinger contribution of the specifically identified nested branch. The
last number is a branch contribution, **not** the total nested-experiment
Hellinger distance.

With 100 seeds, batches of 16, `S=1`, and stage-two threshold `0.2`, the
passive condition stops with the wrong model in all runs (final maximum
policy TV `0.5`, task regret `0.25`); the active condition identifies the
true model in all runs (final maximum policy TV and task regret `0`). These
are illustrative toy settings.

## Scope and interpretation

This program is **not** an implementation of the full Algorithm 1 or its
theorem. It omits the initial and other stage programs, general SOCP solver,
the logged mixture controller, nested diagnostic collection, theorem batch
constants, and task-reward planning call. Its stopping rule is a stage-two
proxy and carries no all-policy certificate. In particular, a passive false
stop demonstrates why the test is needed in this proxy; it does not refute
the paper's theorem. The experiment directly checks the analytic example
and the ordinary-MLE-plus-intervention mechanism before expanding to the
complete algorithm.

The paper sources are in `paper/psr-mle-planning/`. The Section 4 example
appears in `overview.tex`; the complete algorithm and batch definitions are
in `main.tex` and `appendix.tex`.
