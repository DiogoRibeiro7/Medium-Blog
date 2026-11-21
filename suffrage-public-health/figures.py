"""Figures for the suffrage-public-health article; reads results/ written by suffrage.py.

    python figures.py     # figures/fig1_*.png ... fig4_*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RES = HERE / "results"
FIG = HERE / "figures"

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
INK, INK2, GRID, SURFACE, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#efeeea"
SOURCE = ("Data: Miller (2008), digitized U.S. Mortality Statistics and Financial Statistics of Cities, "
          "published on the author's site. Analysis: Diogo Ribeiro.")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2,
})


def finish(fig, name: str, note: str = "") -> None:
    fig.text(0.01, 0.01, (note + "\n" if note else "") + SOURCE, fontsize=7.5, color=INK2,
             ha="left", va="bottom")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def pct(x):
    return 100 * (np.exp(x) - 1)


def fig1() -> None:
    """Miller's specification against three estimators that use clean comparisons, by age."""
    est = pd.read_csv(RES / "estimates.csv").set_index("outcome")
    ages = ["1-4", "5-9", "10-14", "15-19", "1-19"]
    series = [("A", "Miller's specification (TWFE + state trends, 1900-1936)", ORANGE, "o"),
              ("C", "same, stopped in 1919", VIOLET, "D"),
              ("E", "imputation, 1900-1919", BLUE, "s"),
              ("G", "Callaway-Sant'Anna, not-yet-treated, 1900-1919", AQUA, "^")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True, sharex=True)
    for ax, pre, title in [(axes[0], "ln", "Deaths at each age"),
                           (axes[1], "dd", "Deaths at each age relative to deaths at 25-64")]:
        for j, (k, lab, col, mk) in enumerate(series):
            ypos = np.arange(len(ages)) + (j - 1.5) * 0.17
            vals = [est.loc[f"{pre} {a}", k] for a in ages]
            if k in ("A", "C"):
                ses = [est.loc[f"{pre} {a}", f"se_{k}"] for a in ages]
                ax.errorbar(pct(np.array(vals)), ypos,
                            xerr=[pct(np.array(vals)) - pct(np.array(vals) - 1.96 * np.array(ses)),
                                  pct(np.array(vals) + 1.96 * np.array(ses)) - pct(np.array(vals))],
                            fmt=mk, color=col, ms=6.5, elinewidth=1.4, capsize=0, label=lab)
            else:
                ax.plot(pct(np.array(vals)), ypos, mk, color=col, ms=7, label=lab, linestyle="none")
        ax.axvline(0, color=INK, linewidth=1)
        ax.axvspan(-15, -8, color=SHADE, zorder=0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("effect of a suffrage law on deaths (%)")
        ax.set_yticks(np.arange(len(ages)), [f"ages {a}" for a in ages])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    for ax in axes:
        ax.set_ylim(len(ages) - 0.5, -0.9)
        ax.text(-11.5, -0.62, "Miller's 8-15%", ha="center", va="center", fontsize=8.5, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.01, 0.07),
               ncol=2)
    fig.suptitle("The fall in child deaths is large only in the specification that runs to 1936",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    finish(fig, "fig1_estimators_by_age",
           "Bars: 95% intervals, standard errors clustered by state. Shaded band: the 8-15% range reported by Miller.")


def fig2() -> None:
    """Miller's specification as the sample is extended year by year past 1919."""
    w = pd.read_csv(RES / "window.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), sharey=True)
    for ax, y, title in [(axes[0], "ln 1-19", "Deaths at ages 1-19"),
                         (axes[1], "dd 1-19", "Relative to deaths at 25-64")]:
        for k, col, lab in [("A", ORANGE, "with state trends (Miller)"), ("B", BLUE, "without trends")]:
            b, se = w[f"{k} {y}"], w[f"se {k} {y}"]
            ax.fill_between(w.end, pct(b - 1.96 * se), pct(b + 1.96 * se), color=col, alpha=0.12, linewidth=0)
            ax.plot(w.end, pct(b), color=col, label=lab, marker="o", ms=3.5)
        ax.axhline(0, color=INK, linewidth=1)
        ax.annotate("1900-1919 only:\nclean comparisons exist", xy=(1919, pct(w[f"A {y}"].iloc[0])),
                    xytext=(1922.5, 7), fontsize=8.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
        ax.set_title(title, loc="left")
        ax.set_xlabel("last year in the sample")
        ax.set_xlim(1918.5, 1936.5)
    axes[0].set_ylabel("estimated effect on deaths (%)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
    fig.suptitle("Most of the estimate arrives with 1920 and 1921, years in which no state lacked suffrage",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    finish(fig, "fig2_estimate_by_last_year", "Bands: 95% intervals, standard errors clustered by state.")


def fig3() -> None:
    """Late-minus-early gap in log child deaths around 1920."""
    g = pd.read_csv(RES / "gap_1920_vs_early.csv")
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    for y, col, lab in [("ln 1-19", BLUE, "deaths at ages 1-19"),
                        ("dd 1-19", ORANGE, "deaths at ages 1-19 relative to deaths at 25-64"),
                        ("ln 25-64", INK2, "deaths at 25-64")]:
        d = g[g.outcome == y]
        ax.plot(d.year, d.gap, color=col, label=lab, marker="o", ms=4,
                linewidth=1.4 if y == "ln 25-64" else 2.2, linestyle=(0, (4, 2)) if y == "ln 25-64" else "-")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.axvline(1919.5, color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(1919.65, 0.30, "1920: the remaining\nstates enfranchise women", fontsize=8.5, color=INK, va="top")
    ax.annotate("1916", xy=(1916, g[(g.outcome == "ln 1-19") & (g.year == 1916)].gap.iloc[0]),
                xytext=(1914.7, 0.30), fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.annotate("1918", xy=(1918, g[(g.outcome == "ln 1-19") & (g.year == 1918)].gap.iloc[0]),
                xytext=(1917.0, 0.33), fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.set_xlabel("year")
    ax.set_ylabel("log gap, set to zero in 1919")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("States enfranchised in 1920 minus states enfranchised by 1914", loc="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    finish(fig, "fig3_gap_around_1920",
           "Same states in every year (1910-1925): by 1914 CA CO MT UT WA; in 1920 CT MA MD NC NH NJ PA VT.")


def fig4() -> None:
    """Event studies with clean comparisons, levels and relative to adults."""
    ev = pd.read_csv(RES / "event_study.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, y, title, col in [(axes[0], "ln 1-19", "Deaths at ages 1-19", BLUE),
                              (axes[1], "dd 1-19", "Relative to deaths at 25-64", ORANGE)]:
        d = ev[ev.outcome == y]
        ax.fill_between(d.e, pct(d.lo), pct(d.hi), color=col, alpha=0.15, linewidth=0, step=None)
        ax.plot(d.e, pct(d.att), color=col, marker="o", ms=5)
        ax.axhline(0, color=INK, linewidth=1)
        ax.axvline(-0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        for r in d.itertuples():
            if r.e != -1:
                ax.text(r.e, ax.get_ylim()[0] if False else -38, f"{r.n_treat}", ha="center",
                        fontsize=7.5, color=INK2)
        ax.set_title(title, loc="left")
        ax.set_xlabel("years since the suffrage law (base: the year before)")
        ax.set_xticks(range(-6, 6))
        ax.set_ylim(-40, 40)
    axes[0].set_ylabel("deaths relative to not-yet-treated states (%)")
    axes[0].text(-6.3, -34, "treated states:", fontsize=7.5, color=INK2)
    fig.suptitle("With clean comparisons the estimated drop is a few percent at most, and every band includes zero",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    finish(fig, "fig4_event_study_clean",
           "Callaway-Sant'Anna estimates, 1900-1919. Bands: 95% intervals from 499 state-cluster bootstrap draws.")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
