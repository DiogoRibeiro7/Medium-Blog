"""Figures for Part 1, drawn from the same functions as history_numbers.py.

Run: python figures.py  (writes figures/*.png)
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import history as h  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

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


def fig1_triangle() -> None:
    tri = h.taylor_ashe()
    c = h.triangle_matrix(tri)
    n = c.shape[0]
    _, ult = h.chain_ladder_sql(tri)
    f = np.array([np.nansum(c[: n - 1 - k, k + 1]) / np.nansum(c[: n - 1 - k, k]) for k in range(n - 1)])
    full = c.copy()
    for i in range(n):
        for k in range(n - 1):
            if np.isnan(full[i, k + 1]):
                full[i, k + 1] = full[i, k] * f[k]
    inc = np.diff(np.concatenate([np.zeros((n, 1)), full], axis=1), axis=1) / 1e3
    se, _ = h.mack_standard_errors(c)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    vmax = np.nanmax(inc)
    for i in range(n):
        for k in range(n):
            observed = not np.isnan(c[i, k])
            level = min(int(inc[i, k] / vmax * len(BLUE_RAMP)), len(BLUE_RAMP) - 1)
            face = BLUE_RAMP[level] if observed else SURFACE
            ax.add_patch(Rectangle((k + 0.05, n - 1 - i + 0.05), 0.9, 0.9, facecolor=face,
                                   edgecolor=ORANGE if not observed else SURFACE, linewidth=1.2 if not observed else 0,
                                   hatch="////" if not observed else None))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n) + 0.5, [str(k) for k in range(1, n + 1)])
    ax.set_yticks(np.arange(n) + 0.5, [str(i) for i in range(n, 0, -1)])
    ax.grid(False)
    ax.set_xlabel("development year (when the claim amount is known)")
    ax.set_ylabel("accident year (when the loss happened)")
    ax.set_title("Taylor–Ashe triangle: filled cells are known,\nhatched cells are what the chain ladder projects")
    ax.text(0, -1.9, "Darker blue = larger incremental amount in that cell.", ha="left", va="top", color=INK2,
            fontsize=8.5, transform=ax.transData, clip_on=False)

    years = ult["origin"].to_numpy()[1:]
    res = ult["reserve"].to_numpy()[1:] / 1e6
    err = se[1:] / 1e6
    bx.barh(years, res, color=ORANGE, height=0.55)
    bx.errorbar(res, years, xerr=err, fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3)
    bx.set_yticks(years)
    bx.invert_yaxis()
    bx.set_xlabel("reserve still to be reported (millions), ± one Mack standard error")
    bx.set_title("Chain-ladder reserve by accident year")
    bx.grid(axis="y", visible=False)
    total = ult["reserve"].sum() / 1e6
    bx.text(0.98, 0.97, f"total {total:.2f} million, ±13% (Mack, 1993)", transform=bx.transAxes, ha="right",
            va="top", color=INK, fontsize=9)
    fig.tight_layout()
    save(fig, "fig1_runoff_triangle.png")


def fig2_vintages() -> None:
    cases = h.hus_cases()
    final = h.final_counts(cases)
    window = final["2011-05-08":"2011-06-12"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(window.index, window.values, width=0.8, color="#e1e0d9", label="eventually reported (final)")
    for d, color in (("2011-05-24", BLUE), ("2011-05-28", ORANGE), ("2011-06-04", AQUA)):
        D = pd.Timestamp(d)
        k = h.known_by(cases, D)["2011-05-08":]
        ax.plot(k.index, k.values, color=color, label=f"as known on {D:%d %b}")
        ax.axvline(D, color=color, linewidth=1, linestyle=":")
        ax.annotate(f"{D:%d %b}", (D, 64), color=INK2, fontsize=8.5, ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none"))
    ax.set_ylabel("HUS cases by date of hospitalization")
    ax.set_title("STEC O104:H4, Germany 2011: each day's view of the epidemic curve ended in a decline that was not there")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))
    ax.set_ylim(0, 70)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig2_outbreak_vintages.png")


def fig3_nowcast() -> None:
    cases = h.hus_cases()
    final = h.final_counts(cases)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, d in zip(axes, ("2011-05-31", "2011-06-05")):
        D = pd.Timestamp(d)
        nc = h.nowcast_chain_ladder(cases, D)["2011-05-14":]
        fin = final.reindex(nc.index, fill_value=0)
        ax.bar(nc.index, fin.values, width=0.8, color="#e1e0d9", label="final")
        ax.plot(nc.index, nc["known"], color=BLUE, label=f"known on {D:%d %b}")
        mature = nc[nc["completeness"] >= 0.5]
        young = nc[nc["completeness"] < 0.5]
        ax.plot(mature.index, mature["nowcast"], color=ORANGE, label="chain-ladder nowcast (≥ 50% complete)")
        ax.scatter(young.index, young["nowcast"].clip(upper=95), color=ORANGE, marker="x", s=36, linewidths=1.6,
                   label="nowcast on days < 50% complete")
        for t, v in young["nowcast"].items():
            if v > 95:
                ax.annotate(f"{v:.0f}", (t, 95), xytext=(0, 4), textcoords="offset points", ha="center",
                            fontsize=7.5, color=INK2)
        ax.set_title(f"Nowcast made on {D:%d %B}")
        ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22, 29]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set_ylim(0, 105)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("HUS cases by date of hospitalization")
    axes[1].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig3_outbreak_nowcast.png")


def fig4_bitemporal() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x0, x1 = date(2026, 1, 1), date(2026, 4, 15)
    y0, y1 = date(2025, 12, 15), date(2026, 4, 15)
    colors = {10.00: "#9ec5f4", 12.00: ORANGE, 11.50: AQUA}
    for price, vf, vt, rf, rt in h.BITEMPORAL_ROWS:
        vt, rt = min(vt, x1), min(rt, y1)
        ax.add_patch(Rectangle((mdates.date2num(vf), mdates.date2num(rf)), mdates.date2num(vt) - mdates.date2num(vf),
                               mdates.date2num(rt) - mdates.date2num(rf), facecolor=colors[price], edgecolor=SURFACE,
                               linewidth=2, alpha=0.9))
        cx = mdates.date2num(vf) + (mdates.date2num(vt) - mdates.date2num(vf)) / 2
        cy = mdates.date2num(rf) + (mdates.date2num(rt) - mdates.date2num(rf)) * (0.25 if rt == y1 else 0.5)
        ax.text(cx, cy, f"{price:.2f}", ha="center", va="center", color=INK, fontsize=10, fontweight="bold")
    for oid, od, qty in h.ORDERS:
        ax.plot(mdates.date2num(od), mdates.date2num(od), marker="o", color=INK, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"order {oid}", (mdates.date2num(od), mdates.date2num(od)), xytext=(6, -12),
                    textcoords="offset points", fontsize=8, color=INK2)
    today = mdates.date2num(date(2026, 4, 1))
    ax.axhline(today, color=INK2, linewidth=1, linestyle="--")
    ax.text(mdates.date2num(x0) + 1, today + 1.5, "knowledge on 1 April", color=INK, fontsize=8.5)
    ax.set_xlim(mdates.date2num(x0), mdates.date2num(x1))
    ax.set_ylim(mdates.date2num(y0), mdates.date2num(y1))
    ax.xaxis_date()
    ax.yaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlabel("valid time: when the price applies")
    ax.set_ylabel("transaction time: when the database believed it")
    ax.set_title("One price, five bitemporal rows: each invoice reads the row on the diagonal,\n"
                 "a restatement reads the row where the dashed line crosses")
    ax.grid(False)
    fig.tight_layout()
    save(fig, "fig4_bitemporal_price.png")


def fig5_stream() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    panes, dropped = h.stream()
    dropped_set = set(dropped)
    for w in range(5):
        ax.axvline(w * h.WINDOW, color=AXIS, linewidth=1)
    for w in range(4):
        ax.text(w * h.WINDOW + h.WINDOW / 2, 505, f"window {w}", ha="center", color=INK2, fontsize=8.5)
    path = h.watermark_path()
    xs, ys = [], []
    for i, (at, wm) in enumerate(path):
        nxt = path[i + 1][0] if i + 1 < len(path) else 500
        xs += [wm, wm]
        ys += [at, nxt]
    ax.plot(xs, ys, color=BLUE, label="watermark (largest event time seen − 30 s)")
    ax.plot([0, 500], [0, 500], color=GRID, linewidth=1, linestyle="--")
    ax.text(395, 385, "no delay", color=MUTED, fontsize=8, rotation=37)
    for v, et, at in h.EVENTS:
        late = at - et > 60
        color = ORANGE if late else INK2
        marker = "X" if (v, et, at) in dropped_set else "o"
        ax.plot(et, at, marker=marker, color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=1.5,
                linestyle="none")
        ax.annotate(str(v), (et, at), xytext=(7, -3), textcoords="offset points", fontsize=9, color=INK)
    for p in panes.itertuples():
        if p.time == p.time:
            ax.plot(p.window * h.WINDOW + h.WINDOW, p.time, marker="s", color=BLUE, markersize=6, linestyle="none")
    ax.plot([], [], marker="s", color=BLUE, linestyle="none", label="pane emitted (at window end, at arrival time)")
    ax.plot([], [], marker="o", color=ORANGE, linestyle="none", label="late element, accepted (late firing)")
    ax.plot([], [], marker="X", color=ORANGE, linestyle="none", label="late element beyond allowed lateness, dropped")
    ax.set_xlim(0, 490)
    ax.set_ylim(0, 520)
    ax.set_xlabel("event time (seconds after 12:00:00)")
    ax.set_ylabel("processing time: arrival (seconds after 12:00:00)")
    ax.set_title("Twelve elements in event time and processing time: the window fires when the watermark\n"
                 "passes its end, and what arrives later is either a late firing or lost")
    ax.legend(loc="lower right", fontsize=8.5, facecolor=SURFACE, frameon=True, edgecolor="none")
    ax.grid(False)
    fig.tight_layout()
    save(fig, "fig5_event_processing_time.png")


def main() -> None:
    fig1_triangle()
    fig2_vintages()
    fig3_nowcast()
    fig4_bitemporal()
    fig5_stream()


if __name__ == "__main__":
    main()
