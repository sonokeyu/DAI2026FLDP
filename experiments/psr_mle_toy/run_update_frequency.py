"""Fixed episode budget ablation for the number of MLE/planning updates.

This controlled ablation uses the finite 64-state POMDP and 73-model class.
It forces the requested update schedule to consume exactly the same budget;
bonus-triggered natural stopping is recorded separately and does not truncate
the schedule. It is therefore an update-frequency experiment, not a claim
about the algorithm's natural stopping count.
"""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np

from queue_mle_class import fine_candidate_class, finite_mle
from queue_pomdp import TRUE_WEIGHTS, STABILITY
from queue_psr import StageSupport
from run_multistage_experiment import write_csv
from run_queue_ours import design_law, plan_bonus, worst_policy_tv


def run_case(models, truth_index, seed, budget, fit_calls, epsilon):
    rng = np.random.default_rng(seed + 5401)
    truth = models[truth_index]
    data = Counter()
    histories = [Counter() for _ in range(4)]
    episodes = initial_count = 0
    rows = []
    collecting_rounds = fit_calls - 1
    # Each complete collection round contains six batches. Distribute integer
    # batch sizes so every requested schedule spends exactly the same N.
    total_batches = budget // 6
    if budget % 6:
        raise ValueError("budget must be divisible by six")
    base, remainder = divmod(total_batches, collecting_rounds)
    batch_sizes = [base + (i < remainder) for i in range(collecting_rounds)]
    beta = max(1., np.log(3 * len(models) / .1))
    energy = 8 * STABILITY**2 * 2 * beta
    natural_stop_round = None

    def draw(law, size, stage=None):
        nonlocal episodes
        traces = tuple(law)
        probs = np.asarray([law[t] for t in traces], dtype=float)
        probs /= probs.sum()
        for trace, count in zip(traces, rng.multinomial(size, probs)):
            if count:
                data[trace] += int(count)
                if stage is not None:
                    histories[stage][trace[:2 * stage]] += int(count)
        episodes += size

    for call in range(1, fit_calls + 1):
        fit_index = finite_mle(models, data)
        fit = models[fit_index]
        supports = [StageSupport(fit, h, histories[h], len(models)) for h in range(4)]
        root = min(STABILITY, np.sqrt(energy / initial_count)) if initial_count else STABILITY
        bonus, policy, coefficients = plan_bonus(fit, supports, root)
        tv = worst_policy_tv(truth, fit)[0]
        stop_threshold = 3 * epsilon / 16
        should_stop = bonus <= stop_threshold
        if should_stop and natural_stop_round is None:
            natural_stop_round = call
        rows.append(dict(seed=seed, epsilon=epsilon, budget=budget,
                         requested_fit_calls=fit_calls, fit_call=call,
                         episodes=episodes, mle_calls=call, planning_calls=call,
                         fit_index=fit_index, bonus=bonus, sup_tv=tv,
                         stop_threshold=stop_threshold,
                         natural_stop_would_trigger=should_stop,
                         accurate=tv <= epsilon / 4,
                         certificate_valid=tv <= bonus + 1e-8))
        if call == fit_calls:
            break
        batch = batch_sizes[call - 1]
        initial_count += batch
        draw(design_law(truth, policy, stage=0), batch)
        for stage in range(4):
            core_count = int(rng.binomial(batch, .5))
            draw(design_law(truth, policy, stage=stage), core_count, stage)
            draw(design_law(truth, policy, stage=stage, branch="uniform"), batch-core_count, stage)
        choices = [(None, "full", cut) for cut in range(4)]
        choices += [(stage, branch, cut) for stage in range(4)
                    for branch in ("u", "w") for cut in range(4)]
        weights = [1/12] * 4 + [1/48] * 32
        sizes = rng.multinomial(batch, weights)
        for (stage, branch, cut), size in zip(choices, sizes):
            if size:
                draw(design_law(truth, policy, stage=stage, branch=branch,
                                coefficients=coefficients, cut=cut), int(size))
    if episodes != budget:
        raise RuntimeError(f"budget mismatch: {episodes} != {budget}")
    for row in rows:
        row["natural_stop_round"] = natural_stop_round or ""
    return rows


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--budget", type=int, default=6000)
    p.add_argument("--epsilon", type=float, default=.05)
    p.add_argument("--fit-calls", type=int, nargs="+", default=(3, 5, 10, 20, 40))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = fine_candidate_class()
    truth = next(i for i, m in enumerate(models) if m.weights == tuple(TRUE_WEIGHTS))
    runs = []
    for calls in args.fit_calls:
        for seed in range(args.seeds):
            path = run_case(models, truth, seed, args.budget, calls, args.epsilon)
            runs.extend(path)
            final = path[-1]
            print(f"fits={calls} seed={seed} N={args.budget} TV={final['sup_tv']:.6g} "
                  f"bonus={final['bonus']:.6g} natural_stop={final['natural_stop_round']}", flush=True)
    write_csv(args.output / "update_frequency_runs.csv", runs)
    summary = []
    for calls in args.fit_calls:
        finals = [r for r in runs if r["requested_fit_calls"] == calls and r["fit_call"] == calls]
        summary.append(dict(fit_calls=calls, seeds=len(finals), episodes=args.budget,
                            epsilon=args.epsilon, median_tv=float(np.median([r["sup_tv"] for r in finals])),
                            max_tv=max(r["sup_tv"] for r in finals),
                            target_passes=sum(r["accurate"] for r in finals),
                            median_bonus=float(np.median([r["bonus"] for r in finals])),
                            certificate_passes=sum(r["certificate_valid"] for r in finals),
                            natural_stop_seeds=sum(bool(r["natural_stop_round"]) for r in finals),
                            median_natural_stop_round=float(np.median([
                                int(r["natural_stop_round"]) for r in finals if r["natural_stop_round"]]))
                                if any(r["natural_stop_round"] for r in finals) else ""))
    write_csv(args.output / "update_frequency_summary.csv", summary)


if __name__ == "__main__":
    main()
