"""Plot the seven-day comparison and regenerate the daily report figures."""

import argparse
import re
from pathlib import Path

from prepare_traces import (
    DAY_INDEX,
    DAY_SECONDS,
    MINUTES_PER_DAY,
    minute_totals,
    plot_request_counts,
    plot_token_usage,
    plt,
    read_days,
)


DAYS = range(16, 23)
STAGE_DIR = Path(__file__).resolve().parent
REPORT_DIR = STAGE_DIR / "peak_reports"


def report_ratios(path):
    """Use the recorded minute totals for all peak minutes in the report."""
    totals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\|\s*\d{2}:\d{2}:\d{2}\s*\|", line):
            fields = line.strip("|").split("|")
            # The table lists requests, input sum and output sum in these columns.
            totals.append(tuple(int("".join(fields[i].split())) for i in (2, 3, 7)))
    if not totals:
        raise ValueError(f"No peak minutes in {path}")

    request_peak, input_peak, output_peak = map(max, zip(*totals))
    busiest = [row for row in totals if row[0] == request_peak]
    ranges = []
    for column, peak in ((1, input_peak), (2, output_peak)):
        ratios = [100 * row[column] / peak for row in busiest]
        ranges.append((min(ratios), max(ratios)))
    return ranges


def percentage_label(low, high):
    label = f"{low:.1f}" if low == high else f"{low:.1f}–{high:.1f}"
    return label.replace(".", ",") + "%"


def plot_summary(path, ratios):
    fig, ax = plt.subplots(figsize=(11, 5.1), layout="constrained")
    width = 0.36
    for series, (label, color, offset) in enumerate((
        ("Входные токены", "#245583", -width / 2),
        ("Выходные токены", "#00766c", width / 2),
    )):
        positions = [day + offset for day in DAYS]
        bounds = [ratios[day][series] for day in DAYS]
        ax.bar(positions, [low for low, _ in bounds], width, label=label, color=color, zorder=3)
        for x, (low, high) in zip(positions, bounds):
            if low < high:
                ax.errorbar(x, low, yerr=[[0], [high - low]], fmt="none", capsize=4,
                            color="#252f3a", linewidth=1.25, zorder=4)
            height = 5 + 11 * series if low < high else 5
            ax.annotate(percentage_label(low, high), (x, high), xytext=(0, height),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9,
                        color="#252f3a")

    ax.axhline(100, color="#5c6873", linestyle="--", linewidth=1, zorder=2)
    ax.set_xticks(list(DAYS))
    ax.set_yticks(range(0, 101, 20), [f"{y}%" for y in range(0, 101, 20)])
    ax.set_ylim(0, 113)
    ax.set_xlabel("Индекс суток")
    ax.set_ylabel("Доля от суточного минутного максимума")
    ax.set_title("Объём токенов в минуту с наибольшим числом запросов",
                 loc="left", fontweight="bold", pad=36)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncols=2, frameon=False,
              borderaxespad=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dfe5eb", linewidth=0.65)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_daily_reports(source, output_dir):
    _, days = read_days(source, DAYS)
    figures_dir = output_dir / "peak_reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for day, rows in days.items():
        directory = output_dir if day == DAY_INDEX else figures_dir
        counts, input_tokens, output_tokens = minute_totals(rows, day * DAY_SECONDS)
        uniform = [0] * MINUTES_PER_DAY
        for index in range(len(rows)):
            # The same 24 h / N spacing as in prepare_traces.py, binned into minutes.
            uniform[index * MINUTES_PER_DAY // len(rows)] += 1
        plot_request_counts(directory, day, counts, uniform)
        request_peak = max(counts)
        busiest = [minute for minute, count in enumerate(counts) if count == request_peak]
        plot_token_usage(directory / f"day{day}_real_tokens.png", input_tokens, output_tokens, busiest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Also regenerate daily figures from BurstGPT_1.csv")
    parser.add_argument("--output-dir", type=Path, default=STAGE_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ratios = {day: report_ratios(REPORT_DIR / f"day{day}.md") for day in DAYS}
    plot_summary(args.output_dir / "peak_token_shares.png", ratios)
    if args.source:
        plot_daily_reports(args.source, args.output_dir)


if __name__ == "__main__":
    main()
