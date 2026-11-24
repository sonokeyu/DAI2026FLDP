"""Reproduce exact checks and run paired Monte Carlo toy comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from toy import (
    Model, Policy, diagnostic_branch_mass,
    max_policy_tv, optimal_reward, run_stage_two_learning,
    stage_two_support, total_variation, trajectory_distribution,
)


def exact_checks() -> dict:
    left_history = [("-", "L")] * 32
    hidden_policy = Policy(("R", "R"), ("b",) * 4)
    revealed_policy = Policy(("R", "R"), ("a",) * 4)
    hidden_tv = total_variation(
        trajectory_distribution(Model.TRUE, hidden_policy),
        trajectory_distribution(Model.WRONG, hidden_policy),
    )
    revealed_tv = total_variation(
        trajectory_distribution(Model.TRUE, revealed_policy),
        trajectory_distribution(Model.WRONG, revealed_policy),
    )
    support = stage_two_support(Model.WRONG, "-", "R", left_history, 1.0)
    uniform_tv, _ = max_policy_tv(Model.TRUE, Model.WRONG)
    optimum, _ = optimal_reward(Model.TRUE)
    assert hidden_tv == 0.0
    assert revealed_tv == uniform_tv == 0.5
    assert support == 2.0
    assert optimum == 0.75
    assert diagnostic_branch_mass() == 1 / 36
    return {
        "stage_two_unseen_support": support,
        "hidden_task_policy_tv": hidden_tv,
        "revealing_test_policy_tv": revealed_tv,
        "max_policy_tv": uniform_tv,
        "true_optimal_reward": optimum,
        "specified_nested_branch_hellinger_contribution": diagnostic_branch_mass(),
    }


def experiment(seeds: int, batch_size: int, max_rounds: int,
               energy_budget: float, threshold: float) -> tuple[list[dict], dict]:
    records = []
    summaries = {}
    for name, probability in (("passive", 0.0), ("stage_two_test", 0.5)):
        for seed in range(seeds):
            rounds, stopped = run_stage_two_learning(
                seed=seed, batch_size=batch_size, max_rounds=max_rounds,
                energy_budget=energy_budget, stop_threshold=threshold,
                test_probability=probability,
            )
            last = rounds[-1]
            records.append({"condition": name, "seed": seed, "stopped": stopped,
                            **asdict(last)})
        part = [row for row in records if row["condition"] == name]
        summaries[name] = {
            "seeds": seeds,
            "fit_true_fraction": mean(row["fit"] == "true" for row in part),
            "stopped_fraction": mean(row["stopped"] for row in part),
            "false_stage_two_stop_fraction": mean(
                row["stopped"] and row["max_policy_tv"] > threshold for row in part
            ),
            "mean_final_max_policy_tv": mean(row["max_policy_tv"] for row in part),
            "mean_task_regret": mean(row["task_regret"] for row in part),
            "mean_episodes": mean(row["episodes"] for row in part),
            "mean_mle_calls": mean(row["index"] for row in part),
            "mean_diagnostic_tests": mean(row["diagnostic_tests"] for row in part),
        }
    return records, summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--energy-budget", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    exact = exact_checks()
    records, summary = experiment(args.seeds, args.batch_size, args.max_rounds,
                                  args.energy_budget, args.threshold)
    output = {"exact_checks": exact, "parameters": {
        "seeds": args.seeds, "batch_size": args.batch_size,
        "max_rounds": args.max_rounds, "energy_budget": args.energy_budget,
        "threshold": args.threshold,
    }, "monte_carlo": summary}
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        with (args.output / "runs.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    main()
