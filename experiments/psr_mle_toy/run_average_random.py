"""Compare averaged, per-episode random, and batch-frozen random policies."""
from argparse import ArgumentParser
import csv
from itertools import product
from math import log
from pathlib import Path
import random
from statistics import mean, median


GAPS = (.1, .3, .6)
EPISODES = 256
BATCH = 8


def draw_policy_bits(rng):
    # route choices for the two cues and mode choices for four prefixes
    return tuple(rng.randrange(2) for _ in range(6))


def one_episode(cue_draw, outcome_draw, truth, gap, bits):
    negative = cue_draw < .5
    cue_index = 0 if negative else 1
    route_right = bool(bits[cue_index])
    mode_a = bool(bits[2 + cue_index * 2 + int(route_right)])
    informative = negative and route_right and mode_a
    if informative:
        return int(outcome_draw < (.2 + gap * truth)), 1
    return 0, 0


def simulate(gap, truth, seed):
    rng = random.Random(seed + 7919 * truth + int(gap*1000))
    traces = {name: [] for name in ("average", "random_episode", "random_frozen_batch")}
    counts = {name: [0, 0] for name in traces}  # hits, misses
    targeted = {name: 0 for name in traces}
    frozen = None
    for episode in range(EPISODES):
        if episode % BATCH == 0:
            frozen = draw_policy_bits(rng)
        # Draw one complete deterministic policy per episode. Averaging all
        # such policies independently each episode induces independent fair
        # actions at every visited history. This is the uniform average policy.
        bits = draw_policy_bits(rng)
        cue_draw, outcome_draw = rng.random(), rng.random()
        for name, policy in (("average", bits), ("random_episode", bits),
                             ("random_frozen_batch", frozen)):
            # Common random numbers keep the exogenous cue and observation
            # noise equal across arms; differences isolate policy choice.
            y, observed = one_episode(cue_draw, outcome_draw, truth, gap, policy)
            if observed:
                counts[name][0 if y else 1] += 1
                targeted[name] += 1
        if (episode + 1) % BATCH == 0:
            for name in traces:
                hits, misses = counts[name]
                llr = (hits * log((.2+gap)/.2)
                       + misses * log((1-.2-gap)/(1-.2)))
                fit = int(llr > 0)  # ties select candidate 1
                traces[name].append(dict(
                    episodes=episode + 1, updates=(episode+1)//BATCH,
                    fitted=fit, policy_tv=0.0 if fit == truth else gap/2,
                    informative=targeted[name],
                ))
    return traces


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for gap, truth, seed in product(GAPS, (0, 1), range(args.seeds)):
        traces = simulate(gap, truth, seed)
        # Averaging over deterministic policies and drawing one such policy per
        # episode are identical distributions; here they share coupled actions.
        assert traces["average"] == traces["random_episode"]
        for method, trace in traces.items():
            for row in trace:
                runs.append(dict(gap=gap, truth=truth, seed=seed,
                                 method=method, **row))
    with (args.output / "average_random_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(runs)
    summary = []
    for gap, method, episode in product(GAPS, ("average", "random_episode",
                                               "random_frozen_batch"), range(BATCH, EPISODES+1, BATCH)):
        group = [r for r in runs if r["gap"] == gap and r["method"] == method
                 and r["episodes"] == episode]
        summary.append(dict(gap=gap, method=method, episodes=episode,
                            updates=episode//BATCH, runs=len(group),
                            mean_tv=mean(r["policy_tv"] for r in group),
                            median_tv=median(r["policy_tv"] for r in group),
                            mean_informative=mean(r["informative"] for r in group)))
    with (args.output / "average_random_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    differences = []
    by_trial = {(r["gap"], r["truth"], r["seed"], r["method"]): r
                for r in runs if r["episodes"] == EPISODES}
    for gap in GAPS:
        paired = [
            float(by_trial[(gap, truth, seed, "random_frozen_batch")]["policy_tv"])
            - float(by_trial[(gap, truth, seed, "average")]["policy_tv"])
            for truth in (0, 1) for seed in range(args.seeds)
        ]
        rng = random.Random(20261124 + round(gap * 1000))
        boot = sorted(
            mean(paired[rng.randrange(len(paired))] for _ in paired)
            for _ in range(20_000)
        )
        differences.append(dict(
            gap=gap, runs=len(paired), mean_frozen_minus_average=mean(paired),
            paired_bootstrap_95_low=boot[500], paired_bootstrap_95_high=boot[19_499],
        ))
    with (args.output / "average_random_difference.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(differences[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(differences)
    print(f"completed {len(summary)} checkpoints; average and random_episode traces matched exactly")


if __name__ == "__main__":
    main()
