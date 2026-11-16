"""Validate and compare temporary cloud reserve policies for day 22."""

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

from run_local_reserve import STAGE3, prepare_real
from run_experiment import E2E, TRACE_COLUMNS, TTFT, check_vidur, cli_options, load_config, summarize, write_csv


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def execute(config, checkout, work, cache, name, submitted, replicas, plan=None):
    directory = Path(mkdtemp(prefix=f"{name}_", dir=work))
    trace = directory / "trace.csv"
    write_csv(trace, TRACE_COLUMNS, submitted)
    options = dict(config["vidur_args"], cluster_config_num_replicas=replicas,
                   trace_request_generator_config_trace_file=trace,
                   trace_request_generator_config_max_tokens=config["context_tokens"],
                   metrics_config_output_dir=directory, metrics_config_cache_dir=cache)
    if plan is None:
        entry = ["-m", "vidur.main"]
    else:
        plan_path = directory / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        entry = [str(STAGE3 / "elastic_vidur.py"), str(plan_path)]
    env = dict(os.environ, WANDB_MODE="disabled", CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED=str(options["seed"]))
    log = directory / "vidur.log"
    print(f"Starting {name}, {len(submitted)} requests; {log}", flush=True)
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.run([sys.executable, "-B", *entry, *cli_options(options)], cwd=checkout,
                                 env=env, stdout=output, stderr=subprocess.STDOUT)
    outputs = list(directory.glob("*/request_metrics.csv"))
    if process.returncode or len(outputs) != 1:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-18:])
        raise RuntimeError(f"Vidur run {name} failed. {log}\n{tail}")
    return outputs[0]


def route_and_bill(path, plan, submitted, pricing):
    data = read_json(path.with_name("elastic.json"))
    routes = {int(request_id): tuple(route) for request_id, route in data["routes"].items()}
    leases = {(lease["replica_id"], lease["started_at"]): lease for lease in data["leases"]}
    assert set(routes) == set(range(len(submitted))) and len(leases) == len(data["leases"])
    charges = {key: 0 for key in leases}
    for index, (replica_id, dispatch, started) in routes.items():
        assert abs(dispatch - float(submitted[index][0])) < 1e-9
        if replica_id == data["local"]:
            assert started is None
        elif plan["mode"] != "always":
            lease = leases[replica_id, started]
            assert lease["ready_at"] <= dispatch < lease["stop_accepting_at"]
            charges[replica_id, started] += 1

    rows = read_csv(path)
    for row in rows:
        replica_id, dispatch, started = routes[int(row["Request Id"])]
        if replica_id != data["local"] and row[TTFT]:
            assert float(row["request_scheduling_delay"]) + 1e-9 >= plan["ingress_seconds"]
            row[TTFT] = float(row[TTFT]) + plan["egress_seconds"]
            if row[E2E]:
                row[E2E] = float(row[E2E]) + plan["egress_seconds"]
                if plan["mode"] != "always":
                    assert dispatch + row[E2E] <= leases[replica_id, started]["released_at"] + 1e-8
    adjusted = path.with_name("client-request-metrics.csv")
    write_csv(adjusted, list(rows[0]), (row.values() for row in rows))

    for key, lease in leases.items():
        assert lease["started_at"] <= lease["stop_accepting_at"] <= lease["released_at"]
        assert lease["accepted"] == charges[key]
        duration = lease["released_at"] - lease["started_at"]
        increment = pricing["billing_increment_seconds_after_minimum"]
        lease["billable_seconds"] = max(pricing["minimum_seconds_per_start"], math.ceil(duration / increment) * increment)
    return adjusted, list(leases.values()), routes


def validate(config, checkout, work, cache, inputs, cloud, pricing):
    sample = [row for row in inputs if row[0] >= cloud["validation"]["sample_start_seconds"]]
    sample = sample[:cloud["validation"]["requests"]]
    assert len(sample) == cloud["validation"]["requests"]
    size = cloud["cloud_replicas"] + 1
    steady = dict(mode="always", ingress_seconds=0, egress_seconds=0)
    native = execute(config, checkout, work, cache, "native_fixture", sample, size)
    elastic = execute(config, checkout, work, cache, "elastic_fixture", sample, size, steady)
    assert read_csv(native) == read_csv(elastic), "Steady-state request metrics differ from native Vidur"
    base_metrics = summarize(native, sample, config)
    print(f"Exact native parity: {len(sample)} requests; P95 {base_metrics['ttft_p95_s']:.6f} s", flush=True)

    timings = map(Decimal, ("86399.70", "86399.91", "86399.99", "86399.999"))
    edge = [(arrival, *row[1:]) for arrival, row in zip(timings, sample[:4], strict=True)]
    boundary_plan = dict(mode="oracle", intervals=[[86394.8, 86399.995]], startup_seconds=5,
                         ingress_seconds=0.02, egress_seconds=0.02)
    path = execute(config, checkout, work, cache, "lifecycle_fixture", edge, size, boundary_plan)
    adjusted, leases, routes = route_and_bill(path, boundary_plan, edge, pricing)
    assert [routes[i][0] for i in range(4)] == [0, 1, 2, 0]
    assert [lease["accepted"] for lease in leases] == [1, 1, 0]
    assert all(lease["released_at"] > 86400 for lease in leases[:2])
    assert leases[2]["released_at"] == boundary_plan["intervals"][0][1]
    assert summarize(adjusted, edge, config)["completed"] == len(edge)
    (work / "validation.json").write_text(json.dumps(dict(native_parity=base_metrics,
        lifecycle=dict(routes=routes, leases=leases)), indent=2), encoding="utf-8")


def compare_costs(cost):
    local = cost["local"]
    return [s["equipment_usd"] / local["amortization_days"] + s["operating_usd_per_day"]
            for s in local["scenarios"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--vidur", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    stage3 = read_json(STAGE3 / "config.json")
    cloud = read_json(STAGE3 / "cloud-config.json")
    cost = read_json(STAGE3 / "cost-assumptions.json")
    pricing, local_costs = cost["cloud_reference"], compare_costs(cost)
    config = load_config(STAGE3 / stage3["stage2_config"])
    checkout, work = args.vidur.resolve(), args.work_dir.resolve()
    check_vidur(checkout, config)
    work.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir.resolve() if args.cache_dir else work / "cache"
    submitted, excluded = prepare_real(args.source, stage3["day_index"], config["context_tokens"])
    assert len(submitted) == 11440 and excluded == 0
    validate(config, checkout, work, cache, submitted, cloud, pricing)
    if args.validate_only:
        return

    def experiment(mode, startup):
        assumptions = cloud["assumptions"]
        policy = cloud[mode] if mode == "oracle" else {"reactive": cloud[mode]}
        plan = dict(mode=mode, startup_seconds=startup, ingress_seconds=assumptions["ingress_seconds"],
                    egress_seconds=assumptions["egress_seconds"], **policy)
        path = execute(config, checkout, work, cache, f"{mode}_{startup}s", submitted,
                       cloud["cloud_replicas"] + stage3["base_replicas"], plan)
        adjusted, leases, _ = route_and_bill(path, plan, submitted, pricing)
        result = summarize(adjusted, submitted, config)
        hours = sum(lease["billable_seconds"] for lease in leases) / 3600
        vm_cost = hours * pricing["usd_per_replica_hour"]
        summary = dict(scenario=mode, startup_seconds=startup, **result,
                       late=result["requests"] - result["within_2s"], vm_hours=hours, vm_cost_usd=vm_cost,
                       cheaper_than_low=vm_cost < local_costs[0], cheaper_than_mid=vm_cost < local_costs[1],
                       cheaper_than_high=vm_cost < local_costs[2])
        print(json.dumps(summary), flush=True)
        return summary, [dict(scenario=mode, startup_seconds=startup, **lease) for lease in leases]

    starts = [cloud["assumptions"][key] for key in ("startup_seconds", "slow_startup_seconds")]
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda delay: experiment("oracle", delay), starts))
        if any(r["passes"] and r["cheaper_than_high"] for r, _ in outcomes):
            outcomes += list(executor.map(lambda delay: experiment("reactive", delay), starts))
    summaries = [r for r, _ in outcomes]
    leases = [lease for _, group in outcomes for lease in group]
    write_csv(work / "cloud-results.csv", list(summaries[0]), (r.values() for r in summaries))
    if leases:
        write_csv(work / "cloud-leases.csv", list(leases[0]), (r.values() for r in leases))


if __name__ == "__main__":
    main()
