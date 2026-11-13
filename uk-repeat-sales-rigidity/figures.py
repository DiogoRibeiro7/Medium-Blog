"""Figures for the uk-repeat-sales-rigidity article.

    python figures.py     # after analysis.py; writes figures/fig1_*.png ... fig4_*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as A  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, RES, FIG = HERE / "data", HERE / "results", HERE / "figures"

# First three slots of a validated colour-blind-safe categorical palette (all pairs pass).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SOURCE = ("Contains HM Land Registry data © Crown copyright and database right 2026, licensed "
          "under the Open Government Licence v3.0. CPI: OECD. Chart: Diogo Ribeiro.")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False,
})


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _titles(fig, title, subtitle=None):
    fig.text(0.01, 0.975, title, fontsize=14, fontweight="bold", color=INK, ha="left", va="top")
    if subtitle:
        fig.text(0.01, 0.925, subtitle, fontsize=9.5, color=INK2, ha="left", va="top")


def _finish(fig, name, top=0.9):
    fig.text(0.01, 0.01, SOURCE, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.035, 1, top))
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- fig 1
def fig1_losses_by_year():
    yr = pd.read_csv(RES / "losses_by_year.csv", index_col=0)
    cpi = pd.read_csv(DATA / "uk_cpi_monthly.csv")
    infl = A.annual_inflation(cpi)
    yr = yr[yr.index >= 1996]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.2]})
    ax.plot(yr.index, 100 * yr.p_real, color=ORANGE, marker="o", ms=4)
    ax.plot(yr.index, 100 * yr.p_nominal, color=BLUE, marker="o", ms=4)
    last = yr.index.max()
    ax.annotate("real price lower", (last, 100 * yr.p_real.iloc[-1]), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=INK2, va="center")
    ax.annotate("nominal price lower", (last, 100 * yr.p_nominal.iloc[-1]), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=INK2, va="center")
    ax.set_ylabel("share of resales (%)")
    ax.set_ylim(0, None)
    _grid(ax)
    ax.set_title("Resales below the previous price of the same home", loc="left")
    iy = infl.reindex(yr.index)
    ax2.bar(iy.index, 100 * iy, color=MUTED, width=0.7)
    ax2.set_ylabel("CPI, %")
    ax2.set_title("Consumer-price inflation in the year of resale", loc="left")
    _grid(ax2)
    ax2.set_xlim(1995.5, last + 4)
    ax2.set_xticks(range(1996, int(last) + 1, 4))
    _titles(fig, "How often a home sold for less than its owner paid, 1996–2026",
            "England and Wales, consecutive standard sales of the same address; the last year is "
            "incomplete")
    _finish(fig, "fig1_losses_by_year")


# ----------------------------------------------------------------------------- fig 2
def fig2_distribution_near_zero():
    h = pd.read_csv(DATA / "pairs_hist.csv")
    diffs = pd.read_csv(DATA / "pairs_diffs.csv")
    panels = [("2008–2012", range(2008, 2013)), ("2022–2023", range(2022, 2024))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, (lab, yrs) in zip(axes, panels):
        hn = A.hist_series(h, "nominal", y=list(yrs), hold=A.NEAR_HOLDS)
        ex = float(diffs[diffs.y.isin(yrs) & diffs.hold.isin(A.NEAR_HOLDS)].d0.sum())
        ks = np.arange(-30, 30)
        c = hn.reindex(ks).fillna(0).to_numpy(float)
        total = c.sum()
        rest = c.copy()
        rest[ks == 0] -= ex
        x = (ks + 0.5) * A.BIN * 100
        ax.bar(x[ks < 0], 100 * rest[ks < 0] / total, width=0.45, color=BLUE, zorder=2)
        ax.bar(x[ks >= 0], 100 * rest[ks >= 0] / total, width=0.45, color=BLUE, zorder=2,
               label="resales (0.5% bins)")
        ax.bar(x[ks == 0], 100 * ex / total, bottom=100 * rest[ks == 0] / total, width=0.45,
               color=ORANGE, zorder=2, label="at exactly the old price")
        ax.axvline(0, color=INK2, lw=1)
        th = A.density_jump(hn, 0.0, drop_exact=ex)[0]
        ax.set_title(f"Resales in {lab}: {100 * A.missing_share(th):.0f}% of the small losses "
                     "missing", loc="left", fontsize=10.5)
        ax.set_xlabel("change from the previous price of the same home (%, log points)")
        _grid(ax)
        ax.set_xlim(-15, 15)
    axes[0].set_ylabel("share of resales in the window (%)")
    axes[0].legend(loc="upper left", fontsize=9)
    _titles(fig, "A floor at the price the owner paid",
            "Homes held under ten years. 'Missing': density just below zero against density just above, "
            "exact resales excluded.")
    _finish(fig, "fig2_distribution_near_zero", top=0.88)


# ----------------------------------------------------------------------------- fig 4
def fig3_missing_share_vs_inflation():
    J = pd.read_csv(RES / "density_jump_by_year.csv", index_col=0)
    J = J[J.full_year & J.theta_ex.notna()]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    m = 100 * -np.expm1(-J.theta_ex)
    lo = 100 * -np.expm1(-(J.theta_ex - 1.96 * J.se_ex))
    hi = 100 * -np.expm1(-(J.theta_ex + 1.96 * J.se_ex))
    ax.errorbar(100 * J.inflation, m, yerr=[m - lo, hi - m], fmt="o", color=BLUE, ms=7,
                ecolor=AXIS, elinewidth=1, capsize=0, zorder=3, label="nominal zero (old price)")
    pr = 100 * -np.expm1(-J.theta_real)
    ax.scatter(100 * J.inflation, pr, marker="^", s=40, color=AQUA, zorder=3,
               label="real zero (placebo)")
    for y, r in J.iterrows():
        if r.inflation > 0.035 or y in (2009, 2015, 2020):
            ax.annotate(str(y), (100 * r.inflation, 100 * -np.expm1(-r.theta_ex)), xytext=(6, 4),
                        textcoords="offset points", fontsize=8.5, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("CPI inflation in the year of resale (%)")
    ax.set_ylabel("share of small nominal losses missing (%)")
    _grid(ax, "both")
    ax.legend(loc="lower right", fontsize=9)
    _titles(fig, "Does the floor weaken when inflation is high?",
            "One point per year of resale, homes held under ten years. Whiskers: 95% interval "
            "within the year.")
    _finish(fig, "fig3_missing_share_vs_inflation")


# ----------------------------------------------------------------------------- fig 4
def fig4_floor_vs_market():
    RY = pd.read_csv(RES / "density_jump_region_year.csv")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    hi_inf = RY.y.isin([2022, 2023])
    for mask, col, mk, lab in [(~hi_inf, BLUE, "o", "other years"),
                               (hi_inf, ORANGE, "D", "2022 and 2023 (CPI 7–8%)")]:
        d = RY[mask]
        ax.scatter(100 * d.median_lr_2_5y, 100 * -np.expm1(-d.theta_ex), s=34, marker=mk, color=col,
                   edgecolor=SURFACE, linewidth=1.2, zorder=3, label=lab)
    ax.set_xlabel("median change from the previous price, homes held 2–5 years (%, log points)")
    ax.set_ylabel("share of small nominal losses missing (%)")
    _grid(ax, "both")
    ax.legend(loc="lower right", fontsize=9)
    _titles(fig, "Weak markets weaken the floor; the high-inflation years sit with the rest",
            "One point per region and year: London, South, North and Midlands, Wales; "
            "complete years 1996–2025")
    _finish(fig, "fig4_floor_vs_market")


if __name__ == "__main__":
    fig1_losses_by_year()
    fig2_distribution_near_zero()
    fig3_missing_share_vs_inflation()
    fig4_floor_vs_market()
    print("figures written to", FIG)
