"""Combine certificate-stopped OMLE and theorem-stopped ours summaries."""
from argparse import ArgumentParser
import csv
from pathlib import Path
from statistics import median


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def flag(value):
    return value.lower() in ("true", "1", "yes")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--omle", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True,
                        help="oracle_stops.csv from the theorem audit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    omle, ours = read(args.omle), read(args.ours)
    rows = []
    epsilons = sorted({float(row["epsilon"]) for row in omle}, reverse=True)
    for epsilon in epsilons:
        o = [row for row in omle if float(row["epsilon"]) == epsilon]
        a = [row for row in ours if float(row["epsilon"]) == epsilon]
        rows.append(dict(
            epsilon=epsilon, target_tv=epsilon / 4, method="omle",
            runs=len(o), stops=sum(flag(r["stopped"]) for r in o),
            false_stops=sum(float(r["exact_tv"]) > epsilon / 4 for r in o),
            median_episodes=median(int(r["episodes"]) for r in o),
            mle_calls=0, planning_calls=0,
            joint_search_calls=median(int(r["joint_calls"]) for r in o),
            pair_scores=median(int(r["pair_scores"]) for r in o),
            median_stop_tv=median(float(r["exact_tv"]) for r in o),
            confidence_coverage=sum(flag(r["truth_in_confidence_set"]) for r in o) / len(o),
        ))
        rows.append(dict(
            epsilon=epsilon, target_tv=epsilon / 4, method="ours",
            runs=len(a), stops=sum(flag(r["stopped"]) for r in a),
            false_stops=sum(float(r["policy_tv"]) > epsilon / 4 for r in a),
            median_episodes=median(int(float(r["episodes"])) for r in a),
            mle_calls=median(int(float(r["mle_calls"])) for r in a),
            planning_calls=median(int(float(r["exploration_plans"])) for r in a),
            joint_search_calls=0, pair_scores=0,
            median_stop_tv=median(float(r["policy_tv"]) for r in a),
            confidence_coverage="",
        ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} method-accuracy summaries to {args.output}")


if __name__ == "__main__":
    main()
