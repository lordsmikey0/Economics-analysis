"""Prepare and compare the two arrival traces for BurstGPT day 16."""

import argparse
import csv
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DAY_INDEX = 16
SECONDS_PER_MINUTE = 60
MINUTES_PER_DAY = 24 * 60
DAY_SECONDS = MINUTES_PER_DAY * SECONDS_PER_MINUTE
DAY_START = DAY_INDEX * DAY_SECONDS


def read_days(source, day_indices):
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Timestamp", "Model", "Log Type", "Request tokens", "Response tokens"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing source columns: {', '.join(sorted(missing))}")

        days = {day: [] for day in day_indices}
        for row in reader:
            timestamp = Decimal(row["Timestamp"])
            day = int(timestamp // DAY_SECONDS)
            if (
                day in days
                and row["Model"] == "ChatGPT"
                and row["Log Type"] == "API log"
                and int(row["Response tokens"]) > 0
            ):
                days[day].append(row)

        for day, rows in days.items():
            if not rows:
                raise ValueError(f"No matching requests for day {day}")
        return reader.fieldnames, days


def write_trace(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_trace(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def token_totals(rows):
    return tuple(
        sum(int(row[column]) for row in rows)
        for column in ("Request tokens", "Response tokens")
    )


def arrival_minute(row, day_start=DAY_START):
    elapsed = Decimal(row["Timestamp"]) - day_start
    if not 0 <= elapsed < DAY_SECONDS:
        raise ValueError("Arrival outside the selected day")
    return int(elapsed // SECONDS_PER_MINUTE)


def minute_totals(rows, day_start=DAY_START):
    requests = [0] * MINUTES_PER_DAY
    input_tokens = [0] * MINUTES_PER_DAY
    output_tokens = [0] * MINUTES_PER_DAY
    for row in rows:
        minute = arrival_minute(row, day_start)
        requests[minute] += 1
        input_tokens[minute] += int(row["Request tokens"])
        output_tokens[minute] += int(row["Response tokens"])
    return requests, input_tokens, output_tokens


def clock_time(minute):
    hours, minutes = divmod(minute, 60)
    return f"{hours:02d}:{minutes:02d}"


def write_minute_totals(path, requests, input_tokens, output_tokens):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("minute_start", "requests", "input_tokens", "output_tokens"))
        for minute, values in enumerate(zip(requests, input_tokens, output_tokens)):
            writer.writerow((clock_time(minute), *values))


def plot_counts(path, counts, title, color, upper_limit):
    fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
    ax.stairs(counts, range(MINUTES_PER_DAY + 1), color=color, linewidth=1.1)
    ticks = range(0, MINUTES_PER_DAY + 1, 180)
    ax.set_xticks(ticks, [clock_time(minute) for minute in ticks])
    ax.set_xlim(0, MINUTES_PER_DAY)
    ax.set_ylim(0, upper_limit)
    tick_step = 100 if upper_limit > 200 else 20
    ax.set_yticks(range(0, upper_limit + 1, tick_step))
    ax.set_xlabel("Время суток по трассе")
    ax.set_ylabel("Запросов в минуту")
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(color="#dfe5eb", linewidth=0.65)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_request_counts(directory, day, real_counts, uniform_counts):
    real_peak, uniform_peak = max(real_counts), max(uniform_counts)
    upper_limit = 20 * (max(real_peak, uniform_peak) // 20 + 1)
    plot_counts(
        directory / f"day{day}_real.png", real_counts,
        "Исходные времена прихода", "#245583", upper_limit,
    )
    plot_counts(
        directory / f"day{day}_uniform.png", uniform_counts,
        "Равномерные времена прихода", "#00766c", upper_limit,
    )


def plot_token_usage(path, input_tokens, output_tokens, request_peak_minutes):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5.6), layout="constrained")
    series = (
        (input_tokens, "Входные токены", "#245583"),
        (output_tokens, "Выходные токены", "#00766c"),
    )
    for ax, (values, title, color) in zip(axes, series):
        ax.stairs(values, range(len(values) + 1), color=color, linewidth=1, baseline=None)
        for minute in request_peak_minutes:
            ax.axvline(minute + 0.5, color="#687481", linestyle="--", linewidth=1, zorder=1)

        peak = max(values)
        peak_minutes = [minute + 0.5 for minute, value in enumerate(values) if value == peak]
        ax.scatter(peak_minutes, [peak] * len(peak_minutes), color=color, s=22, zorder=3, clip_on=False)
        if len(peak_minutes) == 1:
            minute = peak_minutes[0]
            label = f"{clock_time(int(minute))} · {peak:,}".replace(",", " ")
            left_of_point = minute >= MINUTES_PER_DAY / 2
            ax.annotate(
                label, (minute, peak), xytext=(-8 if left_of_point else 8, 4),
                textcoords="offset points", ha="right" if left_of_point else "left",
                va="bottom", color=color,
            )
        ax.set_ylim(0, max(1, peak * 1.18))
        ax.set_ylabel("Токенов в минуту")
        ax.set_title(title, loc="left", fontweight="bold", pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(color="#dfe5eb", linewidth=0.65)

    ticks = range(0, MINUTES_PER_DAY + 1, 180)
    axes[-1].set_xticks(ticks, [clock_time(minute) for minute in ticks])
    axes[-1].set_xlim(0, MINUTES_PER_DAY)
    axes[-1].set_xlabel("Время суток по трассе")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the original BurstGPT_1.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    fields, days = read_days(args.source, [DAY_INDEX])
    rows = days[DAY_INDEX]
    count = len(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    real_path = args.output_dir / "day16_real.csv"
    uniform_path = args.output_dir / "day16_uniform.csv"
    write_trace(real_path, fields, rows)
    write_trace(
        uniform_path,
        fields,
        (
            {**row, "Timestamp": f"{DAY_START + Decimal(i * DAY_SECONDS) / count:.9f}"}
            for i, row in enumerate(rows)
        ),
    )

    real, uniform = read_trace(real_path), read_trace(uniform_path)
    if real != rows or len(uniform) != count:
        raise ValueError("The saved traces do not match the selected requests")
    for original, rescheduled in zip(real, uniform):
        if any(original[field] != rescheduled[field] for field in fields if field != "Timestamp"):
            raise ValueError("A non-timestamp field changed or request order was lost")

    totals = token_totals(real)
    if token_totals(uniform) != totals:
        raise ValueError("Input or output token totals differ between traces")

    real_counts, input_tokens, output_tokens = minute_totals(real)
    if (sum(real_counts), sum(input_tokens), sum(output_tokens)) != (count, *totals):
        raise ValueError("Minute totals do not match the real trace")
    minute_totals_path = args.output_dir / "day16_real_per_minute.csv"
    write_minute_totals(minute_totals_path, real_counts, input_tokens, output_tokens)

    uniform_counts = [0] * MINUTES_PER_DAY
    for row in uniform:
        uniform_counts[arrival_minute(row)] += 1
    real_peak, uniform_peak = max(real_counts), max(uniform_counts)
    plot_request_counts(args.output_dir, DAY_INDEX, real_counts, uniform_counts)
    peak_minutes = [i for i, value in enumerate(real_counts) if value == real_peak]
    plot_token_usage(args.output_dir / "day16_real_tokens.png", input_tokens, output_tokens, peak_minutes)
    peak_times = [clock_time(minute) for minute in peak_minutes]
    print(f"Both traces: {count} requests, {totals[0]} input tokens, {totals[1]} output tokens")
    print(f"Real: mean {sum(real_counts) / MINUTES_PER_DAY:.5f}, max {real_peak} requests/min")
    print(f"Uniform: mean {sum(uniform_counts) / MINUTES_PER_DAY:.5f}, max {uniform_peak} requests/min")
    print(f"Real peak minute(s): {', '.join(peak_times)} (trace local time)")
    for label, values in (("Input tokens", input_tokens), ("Output tokens", output_tokens)):
        peak = max(values)
        times = [clock_time(i) for i, value in enumerate(values) if value == peak]
        print(f"{label}: max {peak} tokens/min at {', '.join(times)} (trace local time)")


if __name__ == "__main__":
    main()
