"""Figures for Part 2, drawn from the same functions as mechanism_numbers.py.

Run: python figures.py  (writes figures/*.png)
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import mechanism as m  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LIGHT = "#e1e0d9"
SCENARIO_COLOR = {"fast": BLUE, "medium": ORANGE, "slow": AQUA}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 11, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.linewidth": 2,
})


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_truncation() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    days = np.arange(56)
    for ax, d in zip(axes, m.SCENARIOS):
        final, known = m.simulate_counts(d, growth=math.log(2) / 14)
        ax.bar(days, final, width=0.8, color=LIGHT, label="final count")
        ax.plot(days, known, color=SCENARIO_COLOR[d.name], label="known at the end of day 55")
        share = known[-7:].sum() / final[-7:].sum()
        ax.set_title(f"{d.name} delays (median {d.median:g} days)")
        ax.text(0.03, 0.95, f"last 7 days: {100 * share:.0f}% known", transform=ax.transAxes, va="top",
                ha="left", color=INK, fontsize=9)
        ax.set_xlabel("event day")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=8.5)
    axes[0].set_ylabel("events per day")
    fig.suptitle("A count doubling every 14 days, as known on its last day: the slower the reports, the earlier the false peak",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "fig1_truncated_series.png")


def fig2_wait_or_estimate() -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.4))
    slack = np.linspace(0, 14, 113)
    for d in m.SCENARIOS:
        y = [100 * m.on_time_completeness(d, s) for s in slack]
        ax.plot(slack, y, color=SCENARIO_COLOR[d.name], label=f"{d.name} delays")
    ax.set_xlabel("watermark slack: days after a window closes before its on-time result")
    ax.set_ylabel("share of the window's events in that result (%)")
    ax.set_title("A watermark's slack is a quantile of the delay")
    ax.set_ylim(0, 102)
    ax.legend(loc="lower right")

    waits = np.arange(15)
    stable = m.simulate_reporting()
    shifted = m.simulate_reporting(after=m.BACKLOG, shift_day=240)
    rs = m.compare_policies(stable, range(100, 400), waits)
    rh = m.compare_policies(shifted, range(240, 270), waits)
    bx.plot(waits, 100 * rs.raw, color=MUTED, label="raw count, stable delays")
    bx.plot(waits, 100 * rh.raw, color=MUTED, linestyle="--", label="raw count, delays just lengthened")
    bx.plot(waits, 100 * rs.nowcast, color=ORANGE, marker="o", markersize=4, label="nowcast, stable delays")
    bx.plot(waits, 100 * rh.nowcast, color=ORANGE, linestyle="--", marker="o", markersize=4,
            label="nowcast, delays just lengthened")
    bx.set_yscale("log")
    bx.set_yticks([0.5, 1, 2, 5, 10, 20, 50, 100], ["0.5", "1", "2", "5", "10", "20", "50", "100"])
    bx.set_xlabel("days waited before publishing a day's count")
    bx.set_ylabel("mean absolute error against the final count (%)")
    bx.set_title("Medium delays: waiting buys accuracy slowly, a nowcast at once")
    bx.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig2_wait_or_estimate.png")


def fig3_high_water_mark() -> None:
    shop = m.simulate_shop()
    o = shop.orders
    ev, arr, off = o.event_ts.to_numpy(), o.arrival_ts.to_numpy(), o.offline.to_numpy()
    labels, offline_lost, online_lost = [], [], []
    for interval, label in ((300, "5 min"), (900, "15 min"), (m.HOUR, "1 hour"), (6 * m.HOUR, "6 hours"),
                            (m.DAY, "1 day")):
        start = m.RUN_HOUR * m.HOUR if interval == m.DAY else 0
        lost = ~m.hwm_loaded(ev, arr, interval, start)
        labels.append(label)
        offline_lost.append(100 * (lost & off).sum() / len(o))
        online_lost.append(100 * (lost & ~off).sum() / len(o))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(x, offline_lost, width=0.55, color=ORANGE, label="offline orders lost")
    ax.bar(x, online_lost, width=0.55, bottom=offline_lost, color=BLUE, label="online orders lost",
           edgecolor=SURFACE, linewidth=2)
    for i, (a, b) in enumerate(zip(offline_lost, online_lost)):
        ax.text(i, a + b + 0.4, f"{a + b:.1f}%", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_xlabel("how often the incremental load runs")
    ax.set_ylabel("share of all orders never loaded (%)")
    ax.set_title("A high-water mark on event time loses more rows the more often it runs")
    ax.set_ylim(0, 24)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig3_high_water_mark.png")


def fig4_sent_against_final(run: m.ShopRun) -> None:
    sent = run.sent[run.sent.pattern == "upsert"].set_index("day")
    truth = run.truth.set_index("day")
    days = truth.index.to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axhline(0, color=AXIS, linewidth=1)
    rev = 100 * (sent.revenue - truth.revenue) / truth.revenue
    cnt = 100 * (sent.orders - truth.orders) / truth.orders
    ax.plot(days, rev, color=ORANGE, label="net revenue")
    ax.plot(days, cnt, color=BLUE, label="orders")
    ax.text(days[-1] + 1, rev.iloc[-1], "net revenue", color=INK2, va="center", fontsize=9)
    ax.text(days[-1] + 1, cnt.iloc[-1], "orders", color=INK2, va="center", fontsize=9)
    for d in sorted(m.OUTAGES):
        ax.axvline(d, color=AXIS, linewidth=1, linestyle=":")
    ax.set_xlim(-1, days[-1] + 9)
    ax.set_xlabel("order day (each value was sent at 06:00 the next morning)")
    ax.set_ylabel("sent minus final, % of final")
    ax.text(sorted(m.OUTAGES)[1] + 0.8, -1.25, "dotted lines: days with a real fall in demand", color=INK2,
            fontsize=8.5, va="center")
    ax.set_title("Every morning report was short of orders and, most days, long on revenue")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.45), ncols=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig4_sent_against_final.png")


def fig5_features() -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(len(m.SCENARIOS))
    final_cal, asof_cal = [], []
    for d in m.SCENARIOS:
        c = m.calibration(m.count_features(d, seed=5), m.count_features(d, seed=7))
        final_cal.append(c["trained_on_final"])
        asof_cal.append(c["trained_as_of"])
    ax.bar(x - 0.15, final_cal, width=0.28, color=ORANGE, label="trained on final values")
    ax.bar(x + 0.15, asof_cal, width=0.28, color=BLUE, label="trained on values as known on the day")
    for i, (a, b) in enumerate(zip(final_cal, asof_cal)):
        ax.text(i - 0.15, a + 0.02, f"{a:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.text(i + 0.15, b + 0.02, f"{b:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(1, color=INK2, linewidth=1, linestyle=":")
    ax.set_xticks(x, [f"{d.name} delays" for d in m.SCENARIOS])
    ax.set_ylabel("mean prediction / mean outcome in serving")
    ax.set_ylim(0, 1.3)
    ax.set_title("Training on final values under-predicts by the missing share")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="x", visible=False)

    r = m.leakage_experiment()
    for name, color, label in (("type1", ORANGE, "plan from the current dimension (Type 1)"),
                               ("as_of", BLUE, "plan as of the snapshot (point in time)")):
        y0, y1 = r[name + "_offline_auc"], r[name + "_production_auc"]
        bx.plot([0, 1], [y0, y1], color=color, marker="o", markersize=8, label=label)
        bx.text(-0.04, y0, f"{y0:.3f}", ha="right", va="center", fontsize=9, color=INK)
        bx.text(1.04, y1, f"{y1:.3f}", ha="left", va="center", fontsize=9, color=INK)
    bx.set_xticks([0, 1], ["offline test", "production"])
    bx.set_xlim(-0.35, 1.35)
    bx.set_ylim(0.58, 0.78)
    bx.set_ylabel("AUC of a churn model")
    bx.set_title("The leaky join wins offline and loses in production")
    bx.legend(loc="upper right", fontsize=8.5)
    bx.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig5_features.png")


def fig6_backfill(run: m.ShopRun) -> None:
    truth = run.truth.set_index("day")
    frozen = run.frozen.set_index("event_day")
    after = run.daily.set_index("event_day")
    B = m.BACKFILL_DAY
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.2, 1]})
    d0 = frozen.index.to_numpy()
    ax.plot(d0, 100 * frozen.orders / truth.orders.loc[d0], color=BLUE, label=f"as sent (table on day {B})")
    d1 = after.index.to_numpy()
    ax.plot(d1, 100 * after.orders / truth.orders.loc[d1], color=ORANGE, label="after the backfill (table on day 90)")
    ax.axvline(B - 1.5, color=AXIS, linewidth=1, linestyle=":")
    ax.text(B - 1, 101.1, f"backfill boundary\n(days 0–{B - 2} recomputed)", color=INK2, fontsize=8.5, va="top")
    ax.set_ylim(96, 101.4)
    ax.set_xlabel("order day")
    ax.set_ylabel("daily table / final truth, orders (%)")
    ax.set_title("A backfill turns uniform incompleteness into a step")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=8.5)
    ax.grid(axis="x", visible=False)

    c = m.comparisons(run, "orders")
    days = c.index.to_numpy()
    bx.axhline(0, color=AXIS, linewidth=1)
    bx.plot(days, 100 * (c["live"] / c["truth"] - 1), color=MUTED, label="read live from the base table")
    bx.plot(days, 100 * (c["stored"] / c["truth"] - 1), color=ORANGE, label="read from the daily table")
    mixed = c.index[c["mixed"]]
    bx.axvspan(mixed.min() - 0.5, mixed.max() + 0.5, color=LIGHT, alpha=0.6, linewidth=0)
    bx.text(mixed.max() + 1.5, -5.2, "shaded: the earlier day\nrestated, this day not", color=INK2,
            fontsize=8.5, va="center", ha="left")
    bx.set_ylim(-6, 1.8)
    bx.set_xlabel("report day")
    bx.set_ylabel("day / same weekday a week earlier (% error)")
    bx.set_title("Orders against a week earlier: error of each reading")
    bx.legend(loc="lower left", fontsize=8.5)
    bx.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig6_backfill.png")


if __name__ == "__main__":
    fig1_truncation()
    fig2_wait_or_estimate()
    fig3_high_water_mark()
    run = m.run_shop(m.simulate_shop())
    fig4_sent_against_final(run)
    fig5_features()
    fig6_backfill(run)
