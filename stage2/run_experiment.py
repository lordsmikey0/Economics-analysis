"""Replay the paired day-16 traces with the pinned Vidur CPU simulator."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from tempfile import mkdtemp


STAGE2 = Path(__file__).resolve().parent
STREAMS = ("real", "uniform")
TRACE_COLUMNS = ("arrived_at", "num_prefill_tokens", "num_decode_tokens")
TTFT = "prefill_e2e_time"
E2E = "request_e2e_time"


def read_stream(path, day_start):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    arrivals = [Decimal(row["Timestamp"]) - day_start for row in rows]
    if not rows or not all(0 <= time < 86400 for time in arrivals) or arrivals != sorted(arrivals):
        raise ValueError(f"Unsorted or out-of-day arrivals in {path}")
    return rows, arrivals


def paired_traces(config):
    day = config["day_index"]
    stage1 = STAGE2.parent / "stage1"
    (real, real_times), (uniform, uniform_times) = (
        read_stream(stage1 / f"day{day}_{stream}.csv", day * 86400) for stream in STREAMS
    )
    traces = {stream: [] for stream in STREAMS}
    for left, right, real_time, uniform_time in zip(real, uniform, real_times, uniform_times, strict=True):
        if {k: v for k, v in left.items() if k != "Timestamp"} != {
            k: v for k, v in right.items() if k != "Timestamp"
        }:
            raise ValueError("The paired traces differ in a non-timestamp field")
        prefill, decode = (int(left[column]) for column in ("Request tokens", "Response tokens"))
        if min(prefill, decode) < 1:
            raise ValueError("Vidur trace replay would change a nonpositive token count")
        if prefill + decode <= config["context_tokens"]:
            traces["real"].append((real_time, prefill, decode))
            traces["uniform"].append((uniform_time, prefill, decode))
    return traces, len(real) - len(traces["real"])


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def cli_options(options):
    for key, value in options.items():
        if isinstance(value, bool):
            yield f"--{key}" if value else f"--no-{key}"
        else:
            yield f"--{key}"
            yield from map(str, value if isinstance(value, list) else [value])


def percentile(values, fraction):
    return sorted(values)[math.ceil(fraction * len(values)) - 1] if values else None


def summarize(path, submitted, config):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    observed = {}
    post_first_ms = []
    completed = 0
    for row in rows:
        index = int(row["Request Id"])
        if index in observed or not 0 <= index < len(submitted):
            raise ValueError(f"Unexpected or duplicated Vidur request ID {index}")
        _, prefill, decode = submitted[index]
        exported_lengths = (Decimal(row["request_num_prefill_tokens"]), Decimal(row["request_num_decode_tokens"]))
        if exported_lengths != (prefill, decode):
            raise ValueError(f"Vidur changed the token counts for request {index}")
        if bool(row[TTFT]) != bool(row[E2E]):
            raise ValueError(f"Incomplete Vidur metrics for request {index}")
        if not row[TTFT]:
            observed[index] = math.inf
            continue
        first, end = float(row[TTFT]), float(row[E2E])
        if not (math.isfinite(first) and math.isfinite(end) and 0 <= first <= end):
            raise ValueError(f"Invalid latency for request {index}")
        observed[index] = first
        completed += 1
        if decode > 1:
            post_first_ms.append(1000 * (end - first) / (decode - 1))

    ttft = [observed.get(i, math.inf) for i in range(len(submitted))]
    on_time = sum(value <= config["ttft_limit_seconds"] for value in ttft)
    p95 = percentile(ttft, 0.95)
    return dict(
        requests=len(submitted), completed=completed, within_2s=on_time,
        ttft_p95_s=p95 if p95 is not None and math.isfinite(p95) else None,
        within_2s_fraction=on_time / len(submitted),
        passes=on_time / len(submitted) >= config["minimum_pass_fraction"],
        tpot_requests=len(post_first_ms), tpot_p50_ms=percentile(post_first_ms, 0.50),
        tpot_p95_ms=percentile(post_first_ms, 0.95),
    )


def run(config, checkout, work, stream, replicas, submitted, smoke=False):
    if not submitted:
        raise ValueError("No requests to replay")
    name = "smoke" if smoke else f"{stream}_{replicas}gpu"
    run_dir = Path(mkdtemp(prefix=f"{name}_", dir=work))
    trace = run_dir / "trace.csv"
    write_csv(trace, TRACE_COLUMNS, submitted)
    options = dict(config["vidur_args"], cluster_config_num_replicas=replicas,
                   trace_request_generator_config_trace_file=trace,
                   trace_request_generator_config_max_tokens=config["context_tokens"],
                   metrics_config_output_dir=run_dir, metrics_config_cache_dir=work / "cache")
    env = dict(os.environ, WANDB_MODE="disabled", CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED=str(options["seed"]))
    # Run from the checkout so Vidur finds its unmodified profiling CSVs.
    command = [sys.executable, "-B", "-m", "vidur.main", *cli_options(options)]
    log = run_dir / "vidur.log"
    print(f"Starting {name}: {len(submitted)} requests. Log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.run(command, cwd=checkout, env=env, stdout=output, stderr=subprocess.STDOUT)
    if process.returncode:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:])
        raise RuntimeError(f"Vidur failed for {name}; log: {log}\n{tail}")
    metrics = list(run_dir.glob("*/request_metrics.csv"))
    if len(metrics) != 1:
        raise RuntimeError(f"Vidur did not export request metrics; see {log}")
    result = dict(stream=stream, gpus=replicas, **summarize(metrics[0], submitted, config))
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "full"])
    parser.add_argument("--vidur", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="JSON overrides for stage2/config.json")
    parser.add_argument("--smoke-replicas", type=int, choices=[1, 2], default=1)
    parser.add_argument("--jobs", type=int, choices=[1, 2, 3, 4], default=1)
    args = parser.parse_args()
    config = json.loads((STAGE2 / "config.json").read_text(encoding="utf-8"))
    if args.config:
        overrides = json.loads(args.config.read_text(encoding="utf-8"))
        config["vidur_args"].update(overrides.pop("vidur_args", {}))
        config.update(overrides)
    checkout, work = args.vidur.resolve(), args.work_dir.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if commit != config["vidur_commit"]:
        raise ValueError(f"Expected Vidur {config['vidur_commit']}, found {commit}")
    work.mkdir(parents=True, exist_ok=True)
    traces, excluded = paired_traces(config)
    print(f"Excluded from each flow by context limit: {excluded}", flush=True)
    if args.mode == "smoke":
        minute = config["smoke"]["minute_offset"]
        sample = [row for row in traces["real"] if minute * 60 <= row[0] < (minute + 1) * 60]
        sample = sample[:config["smoke"]["requests"]]
        result = run(config, checkout, work, "real", args.smoke_replicas, sample, smoke=True)
        if result["completed"] != config["smoke"]["requests"]:
            raise RuntimeError("The smoke test did not complete the expected sample")
        write_csv(work / "smoke.csv", list(result), [result.values()])
        return

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run, config, checkout, work, stream, replicas, traces[stream])
                   for stream in STREAMS for replicas in (1, 2)]
        results = [future.result() for future in futures]
    rows = [dict(row, excluded_by_context=excluded) for row in results]
    write_csv(work / "results.csv", list(rows[0]), (row.values() for row in rows))


if __name__ == "__main__":
    main()
