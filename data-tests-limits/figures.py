"""Figures for the article, from ``profiles/``, ``sources/`` and ``results/``.

    python figures.py

fig1  E1: every dictionary rule on every delivery of yellow, FHV and high-volume FHV
fig2  E2: the delivered row count against the TLC monthly report, month by month, four feeds
fig3  E2: share of random losses caught, by loss size and detector, at the matched budget
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from evaluate import read_profiles  # noqa: E402
from feed import counts as C  # noqa: E402
from feed.contracts import CONTRACTS  # noqa: E402
from feed.rules import judge  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG, PROF, SRC = HERE / "results", HERE / "figures", HERE / "profiles", HERE / "sources"
SURFACE, INK, INK2, GRID, EMPTY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#efeeea"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#7a5cc4"
CREDIT = "Diogo Ribeiro · github.com/DiogoRibeiro7/Medium-Blog · data: NYC TLC trip records and monthly reports"
NAME = {
    "yellow": "Yellow taxi",
    "green": "Green taxi",
    "fhv": "For-hire vehicles (FHV)",
    "fhvhv": "High-volume for-hire (Uber, Lyft, ...)",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    }
)


def credit(fig):
    fig.text(0.995, 0.005, CREDIT, ha="right", va="bottom", fontsize=7, color=INK2)


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def fig1(P: dict):
    feeds = list(CONTRACTS)
    rows = {f: [r.name for r in judge(P[f][min(P[f])])] for f in feeds}
    allm = sorted({m for f in feeds for m in P[f]})
    fig, axes = plt.subplots(
        len(feeds), 1, figsize=(12, 10.5), gridspec_kw=dict(height_ratios=[len(rows[f]) for f in feeds])
    )
    dict_i = allm.index("2025-03")
    for ax, f in zip(axes, feeds, strict=True):
        M = np.full((len(rows[f]), len(allm)), np.nan)
        for j, m in enumerate(allm):
            if m in P[f]:
                for i, r in enumerate(judge(P[f][m])):
                    M[i, j] = (1 if r.variant == "strict" else 2) if r.fired else 0
        cmap = ListedColormap([EMPTY, INK2, BLUE])
        Mm = np.ma.masked_invalid(M)
        ax.pcolormesh(
            np.arange(len(allm) + 1),
            np.arange(len(rows[f]) + 1),
            Mm,
            cmap=cmap,
            vmin=-0.5,
            vmax=2.5,
            edgecolors=SURFACE,
            linewidth=0.8,
        )
        ax.set_ylim(len(rows[f]), 0)
        rate = np.nanmean(np.where(np.isnan(M), np.nan, M > 0), axis=1)
        ax.set_yticks(np.arange(len(rows[f])) + 0.5)
        ax.set_yticklabels([f"{n}  ({100 * v:.0f}%)" for n, v in zip(rows[f], rate, strict=True)], fontsize=8)
        jan = [i for i, m in enumerate(allm) if m.endswith("-01")]
        ax.set_xticks([i + 0.5 for i in jan])
        ax.set_xticklabels([allm[i][:4] for i in jan])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axvline(dict_i, color=INK, lw=1.0, ls=(0, (2, 2)))
        ax.set_title(NAME[f], loc="left")
    axes[0].text(dict_i + 0.3, -0.4, "dictionary of 18 March 2025", ha="left", va="bottom", fontsize=8, color=INK2)
    fig.text(
        0.01,
        0.99,
        "Rules written from each feed's data dictionary, on every monthly file 2019 to 2026.\n"
        "Dark: a strict rule fired (any row). Blue: a tolerant rule fired (over 1% of rows for some field). "
        "In brackets: share of deliveries on which it fired.",
        fontsize=9,
        color=INK2,
        va="top",
    )
    fee = allm.index("2026-03")
    axes[-1].axvline(fee, color=INK2, lw=0.8, ls=(0, (1, 2)))
    axes[-1].text(
        fee - 0.3, -0.4, "airport fee $3.50 from 15 March 2026", ha="right", va="bottom", fontsize=8, color=INK2
    )
    fig.subplots_adjust(left=0.2, right=0.99, top=0.92, bottom=0.04, hspace=0.34)
    credit(fig)
    save(fig, "fig1_dictionary_rules.png")


def fig2(P: dict):
    rows = C.rows_from_profiles(P)
    report = C.read_report(SRC / "data_reports_monthly.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True)
    for ax, f in zip(axes.ravel(), ("fhvhv", "green", "yellow", "fhv"), strict=True):
        ms = sorted(m for m in rows.get(f, {}) if m in report.get(f, {}) and not (f == "fhv" and m == "2019-01"))
        t = [pd.Timestamp(f"{m}-15") for m in ms]
        v = np.array([100 * (rows[f][m] / report[f][m] - 1) for m in ms])
        ax.axhspan(-1, 1, color=GRID, alpha=0.7, lw=0)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.plot(t, v, color=BLUE, lw=1.8, marker="o", ms=2.5)
        ax.axvline(pd.Timestamp(f"{C.FIRST_EVAL}-01"), color=INK2, lw=0.8, ls=(0, (2, 2)))
        ax.set_title(NAME[f], loc="left")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%" if x else "0"))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        lo, hi = float(np.min(v)), float(np.max(v))
        pad = max(0.5, 0.08 * (hi - lo))
        ax.set_ylim(min(lo, -1.5) - pad, max(hi, 1.5) + pad)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes[:, 0]:
        ax.set_ylabel("file rows vs report, difference")
    axes[0, 0].text(
        pd.Timestamp(f"{C.FIRST_EVAL}-01"),
        0.98,
        " evaluated from May 2021",
        transform=axes[0, 0].get_xaxis_transform(),
        fontsize=8,
        color=INK2,
        va="top",
    )
    fig.suptitle(
        "Rows in each monthly file against the TLC monthly report (trips per day x days); the band is ±1%",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.925,
        "FHV from February 2019: the January 2019 FHV file still holds the high-volume trips.",
        fontsize=8.5,
        color=INK2,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.07, hspace=0.3, wspace=0.2)
    credit(fig)
    save(fig, "fig2_outside_count.png")


def fig3():
    L = pd.read_csv(RES / "losses.csv")
    L = L[L.threshold == "matched"]
    colors = {"volume": INK2, "seasonal": ORANGE, "outside count": BLUE, "outside count, fixed 1%": AQUA}
    labels = {
        "volume": "volume against its history",
        "seasonal": "volume against a seasonal forecast",
        "outside count": "file against the report, against its history",
        "outside count, fixed 1%": "file against the report, fixed 1% limit",
    }
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.6), sharey=True)
    for ax, f in zip(axes, ("fhvhv", "green", "yellow", "fhv"), strict=True):
        for d, c in colors.items():
            x = L[(L.feed == f) & (L.detector == d)].sort_values("loss")
            ax.plot(
                100 * x.loss,
                100 * x.detected.astype(float),
                color=c,
                lw=2,
                marker="o",
                ms=5,
                markeredgecolor=SURFACE,
                ls="--" if d.endswith("1%") else "-",
                label=labels[d],
            )
        ax.set_title(NAME[f].split(" (")[0], loc="left", fontsize=10)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.set_ylim(-3, 103)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("rows lost at random")
    axes[0].set_ylabel("faulted deliveries caught")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.06, 0.9), ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle(
        "Random losses of rows, May 2021 to September 2026: caught at a budget of 5% alarms on clean deliveries",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.76, bottom=0.14, wspace=0.08)
    credit(fig)
    save(fig, "fig3_random_losses.png")


def main():
    P = read_profiles(PROF)
    fig1(P)
    fig2(P)
    fig3()


if __name__ == "__main__":
    main()
