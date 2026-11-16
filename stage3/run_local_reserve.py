"""Find the first passing fixed reserve within the tested day-22 range."""

import argparse
import csv
import json
import sys
from decimal import Decimal
from pathlib import Path


STAGE3 = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE3.parent / "stage2"))
from run_experiment import check_vidur, load_config, run, write_csv


def prepare_real(source, day, context):
    sys.path.insert(0, str(STAGE3.parent / "stage1"))
    from prepare_traces import DAY_SECONDS, read_days

    _, days = read_days(source, [day])
    submitted = []
    for row in days[day]:
        prefill = int(row["Request tokens"])
        decode = int(row["Response tokens"])
        if min(prefill, decode) < 1:
            raise ValueError("Vidur trace replay would change a nonpositive token count")
        if prefill + decode <= context:
            arrival = Decimal(row["Timestamp"]) - day * DAY_SECONDS
            submitted.append((arrival, prefill, decode))
    return submitted, len(days[day]) - len(submitted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--vidur", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, help="Optional cache from the same 70B TP2 setup")
    args = parser.parse_args()
    plan = json.loads((STAGE3 / "config.json").read_text(encoding="utf-8"))
    day, stream = plan["day_index"], plan["stream"]
    config = load_config(STAGE3 / plan["stage2_config"])
    checkout, work = args.vidur.resolve(), args.work_dir.resolve()
    check_vidur(checkout, config)
    work.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir.resolve() if args.cache_dir else work / "cache"
    submitted, excluded = prepare_real(args.source, day, config["context_tokens"])

    with (STAGE3 / plan["stage2_results"]).open(encoding="utf-8", newline="") as saved:
        baseline = [row for row in csv.DictReader(saved) if int(row["day"]) == day and row["stream"] == stream]
    next_counts = plan["new_replica_counts"]
    if [int(row["replicas"]) for row in baseline] != list(range(plan["base_replicas"], next_counts[0])):
        raise ValueError("Missing or unordered stage 2 baseline configurations")
    for row in baseline:
        if int(row["requests"]) != len(submitted) or int(row["excluded_by_context"]) != excluded:
            raise ValueError("Stage 2 request population does not match the new trace")
        if row["passes"] != "False":
            raise ValueError("A smaller passing reserve already exists in stage 2")

    results = [dict(source="stage2", **row) for row in baseline]
    for replicas in next_counts:
        result = run(config, checkout, work, stream, replicas, submitted, cache_dir=cache)
        late = result["requests"] - result["within_2s"]
        results.append(dict(source="stage3", day=day, **result, late=late,
                            late_fraction=late / result["requests"], excluded_by_context=excluded))
        fields = list(results[0])
        write_csv(work / "local-results.csv", fields, ([row[key] for key in fields] for row in results))
        if result["passes"]:
            break


if __name__ == "__main__":
    main()
