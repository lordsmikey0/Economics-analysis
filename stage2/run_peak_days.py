"""Compare day 21 and 22 on Llama 3 70B, with an extra replica only after a daily failure."""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_experiment import STAGE2, STREAMS, check_vidur, load_config, paired_traces, run, write_csv

sys.path.insert(0, str(STAGE2.parent / "stage1"))
from prepare_traces import DAY_SECONDS, clock_time, read_days, uniform_rows, write_trace


DAYS = (21, 22)
WINDOW_MINUTES = 15
WINDOW_SECONDS = WINDOW_MINUTES * 60


def prepare(source, output, config):
    output.mkdir(parents=True, exist_ok=True)
    fields, days = read_days(source, DAYS)
    inputs = {}
    for day, rows in days.items():
        day_start = day * DAY_SECONDS
        write_trace(output / f"day{day}_real.csv", fields, rows)
        write_trace(output / f"day{day}_uniform.csv", fields, uniform_rows(rows, day_start))
        traces, excluded = paired_traces(dict(config, day_index=day), output)
        print(f"Day {day}: {len(rows)} matching requests, {excluded} excluded by context", flush=True)
        inputs[day] = traces, excluded
    return inputs


def window_results(day, gpus, submitted, late_flags):
    counts = [0] * (DAY_SECONDS // WINDOW_SECONDS)
    late_counts = [0] * len(counts)
    for (arrival, _, _), late in zip(submitted, late_flags, strict=True):
        window = int(arrival // WINDOW_SECONDS)
        counts[window] += 1
        late_counts[window] += late
    for i, (count, late) in enumerate(zip(counts, late_counts)):
        yield dict(day=day, gpus=gpus, window_start=clock_time(i * WINDOW_MINUTES),
                   window_end=clock_time((i + 1) * WINDOW_MINUTES), requests=count,
                   late=late, late_fraction=late / count if count else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Path to BurstGPT_1.csv")
    parser.add_argument("--vidur", required=True, type=Path, help="Path to the pinned Vidur checkout")
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, help="Optionally reuse the same 70B TP2 predictor cache")
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args()
    config = load_config(STAGE2 / "config-70b-tp2.json")
    checkout, work = args.vidur.resolve(), args.work_dir.resolve()
    check_vidur(checkout, config)
    cache = args.cache_dir.resolve() if args.cache_dir else work / "cache"
    inputs = prepare(args.source, work / "inputs", config)

    def experiment(day, stream, replicas):
        traces, excluded = inputs[day]
        run_dir = work / f"day{day}"
        run_dir.mkdir(exist_ok=True)
        late_flags = [] if stream == "real" else None
        result = run(config, checkout, run_dir, stream, replicas, traces[stream],
                     cache_dir=cache, late_flags=late_flags)
        late = result["requests"] - result["within_2s"]
        summary = dict(day=day, **result, late=late, late_fraction=late / result["requests"],
                       excluded_by_context=excluded)
        windows = list(window_results(day, result["gpus"], traces[stream], late_flags)) if stream == "real" else []
        return summary, windows

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        first = list(pool.map(lambda key: experiment(*key, 1), ((d, s) for d in DAYS for s in STREAMS)))
        failed = [(row["day"], row["stream"]) for row, _ in first if not row["passes"]]
        followups = list(pool.map(lambda key: experiment(*key, 2), failed))
    summaries, windows = zip(*(first + followups))
    window_rows = [row for group in windows for row in group]
    write_csv(work / "results.csv", list(summaries[0]), (row.values() for row in summaries))
    write_csv(work / "real-windows.csv", list(window_rows[0]), (row.values() for row in window_rows))


if __name__ == "__main__":
    main()
