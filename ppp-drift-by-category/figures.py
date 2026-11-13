"""Draw the three figures for the article from ``results/``, numbered in the order the article shows them.

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
KIND = {"goods": SLOTS[0], "services": SLOTS[1], "public": SLOTS[2], "aggregation": SLOTS[3], "mixed": BASE}
KIND_LABEL = {"goods": "Goods", "services": "Services", "public": "Health and education (PPP includes government)",
              "mixed": "Mixed goods and services", "aggregation": "Aggregation terms"}
SOURCE = ("Sources: Eurostat PPPs by analytical category (prc_ppp_ind), household consumption by purpose "
          "(nama_10_co3_p3) and by durability (nama_10_fcs), snapshot 13 Nov 2026; author's calculations.")
NAMES = {"AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CY": "Cyprus", "CZ": "Czechia", "DE": "Germany",
         "DK": "Denmark", "EE": "Estonia", "EL": "Greece", "ES": "Spain", "FI": "Finland", "FR": "France",
         "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg",
         "LV": "Latvia", "MT": "Malta", "NL": "Netherlands", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
         "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia", "CH": "Switzerland", "NO": "Norway", "IS": "Iceland",
         "RS": "Serbia", "AL": "Albania", "BA": "Bosnia and Herz."}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": BASE, "axes.linewidth": 1.0, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "legend.labelcolor": INK2,
})


def tidy(ax, grid_axis: str = "x") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def finish(fig, name: str, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.995)
    fig.text(0.01, 0.995 - 0.30 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    fig.text(0.01, 0.005, SOURCE, ha="left", va="bottom", fontsize=7.2, color=MUTED)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def fig1() -> None:
    t = pd.read_csv(RES / "category_slopes.csv")
    s = open(RES / "analysis_summary.txt", encoding="utf-8").read()
    hh = float(s.split("Household consumption drift: slope ")[1].split(" ")[0])
    gs_lines = [ln for ln in s.splitlines() if ln.startswith(("Consumer goods:", "Consumer services:"))]
    gs = []
    for ln in gs_lines:
        lab = ln.split(":")[0]
        slope = float(ln.split("slope ")[1].split(" ")[0])
        se = float(ln.split("se ")[1].split(",")[0])
        gs.append({"label": f"{lab} (Eurostat split)", "kind": "goods" if "goods" in lab else "services",
                   "slope": slope, "se": se})
    div = t[t.level.isin(["division", "derived"])].copy()
    order = {"goods": 0, "services": 1, "mixed": 2, "public": 3}
    div["o"] = div.kind.map(order)
    div = div.sort_values(["o", "slope"], ascending=[True, False])
    rows = pd.concat([pd.DataFrame(gs), div[["label", "kind", "slope", "se"]]], ignore_index=True)
    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    fig.subplots_adjust(left=0.33, right=0.97, top=0.86, bottom=0.17)
    y = np.arange(len(rows))[::-1].astype(float)
    y[:len(gs)] += 0.6
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.axvline(hh, color=INK, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(hh - 1.5, y.max() + 0.9, f"household consumption {hh:.1f}", ha="right", fontsize=8.5, color=INK2)
    for yi, (_, r) in zip(y, rows.iterrows()):
        col = KIND[r.kind]
        ax.hlines(yi, r.slope - 1.96 * r.se, r.slope + 1.96 * r.se, color=col, linewidth=2.2, alpha=0.55)
        ax.scatter(r.slope, yi, s=60, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   marker="D" if "Eurostat" in r.label else "o")
    ax.set_yticks(y, rows.label, fontsize=8.8)
    ax.set_xlabel("Drift over 2005–2019 per log point of the 2005 price level (points); bars are 95% intervals")
    ax.set_ylim(-0.8, y.max() + 1.5)
    tidy(ax)
    fig.legend(handles=[Patch(color=KIND[k], label=KIND_LABEL[k]) for k in ("goods", "services", "mixed", "public")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.55, 0.035), fontsize=8.5)
    finish(fig, "fig1_drift_slope_by_category",
           "Where the price-level slope of the drift sits, category by category",
           "Slope of each category's drift against the EU27 on the log of the 2005 household price level, "
           "33 European economies; negative means cheaper economies drifted more.")


def fig2() -> None:
    dcm = pd.read_csv(RES / "decomposition.csv")
    nice = {"A_deflator": "Deflator aggregation", "B_ppp": "PPP aggregation", "C_reference": "EU27 reference weights"}
    dcm["label"] = dcm.apply(lambda r: nice.get(r.term, r.label), axis=1)
    order = {"goods": 0, "services": 1, "mixed": 2, "public": 3, "aggregation": 4}
    dcm["o"] = dcm.kind.map(order)
    dcm = dcm.sort_values(["o", "slope"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    fig.subplots_adjust(left=0.30, right=0.95, top=0.86, bottom=0.16)
    cum = 0.0
    y = np.arange(len(dcm) + 1)[::-1]
    for yi, (_, r) in zip(y[:-1], dcm.iterrows()):
        ax.barh(yi, r.slope, left=cum, color=KIND[r.kind], height=0.66, edgecolor=SURFACE)
        ax.text(cum + r.slope + (0.08 if r.slope >= 0 else -0.08), yi, f"{r.slope:+.2f}", va="center",
                ha="left" if r.slope >= 0 else "right", fontsize=8.3, color=INK2)
        if yi > 0:
            ax.plot([cum + r.slope] * 2, [yi - 0.33, yi - 0.67], color=MUTED, linewidth=0.8)
        cum += r.slope
    ax.barh(y[-1], cum, color=INK, height=0.66)
    ax.text(cum - 0.08, y[-1], f"{cum:.2f}", va="center", ha="right", fontsize=8.8, color=INK, fontweight="bold")
    ax.set_yticks(y, list(dcm.label) + ["Household consumption"], fontsize=8.8)
    lows = np.minimum.accumulate(np.r_[0, np.cumsum(dcm.slope.values)])
    highs = np.maximum.accumulate(np.r_[0, np.cumsum(dcm.slope.values)])
    ax.set_xlim(lows.min() - 1.0, highs.max() + 0.7)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("Contribution to the slope of the household-consumption drift on the 2005 price level (points)")
    tidy(ax)
    fig.legend(handles=[Patch(color=KIND[k], label=KIND_LABEL[k]) for k in ("goods", "services", "mixed", "public", "aggregation")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.55, 0.02), fontsize=8.5)
    finish(fig, "fig2_decomposition_of_the_slope",
           "Services divisions carry most of the slope, goods none, and aggregation a third",
           "Exact split of the slope of the household-consumption drift into share-weighted COICOP divisions "
           "and three aggregation terms, 2005–2019.")


def fig3() -> None:
    e = pd.read_csv(RES / "economies.csv", index_col=0)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 5.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.17, wspace=0.25)
    pli = np.exp(e.ln_pli)
    ax.scatter(pli, e.spread, s=34, color=SLOTS[3], edgecolor=INK2, linewidth=0.5, zorder=3)
    b = np.polyfit(e.ln_pli, e.spread, 1)
    xs = np.linspace(e.ln_pli.min(), e.ln_pli.max(), 50)
    ax.plot(np.exp(xs), np.polyval(b, xs), color=INK, linewidth=1.4)
    ax.set_xscale("log")
    ticks = [40, 60, 80, 100, 150]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    for g in ("BG", "AL", "RS", "RO", "ES", "BE", "DE", "IS", "NO", "LU", "CH"):
        if g in e.index:
            ax.annotate(NAMES[g], (pli[g], e.spread[g]), xytext=(4, 3), textcoords="offset points", fontsize=7.8,
                        color=INK2)
    r = np.corrcoef(e.ln_pli, e.spread)[0, 1]
    ax.set_title(f"A. Category price gaps against the price level (r = {r:.2f})")
    ax.set_xlabel("Household-consumption price level, 2005 (EU27 = 100, log scale)")
    ax.set_ylabel("Spread of division price levels around the\nhousehold price level, 2005 (log points)")
    tidy(ax, "both")
    bx.scatter(e.composition, e.B_ppp, s=34, color=SLOTS[3], edgecolor=INK2, linewidth=0.5, zorder=3)
    b = np.polyfit(e.composition, e.B_ppp, 1)
    xs = np.linspace(e.composition.min(), e.composition.max(), 50)
    bx.plot(xs, np.polyval(b, xs), color=INK, linewidth=1.4)
    bx.axhline(0, color=BASE, linewidth=0.8)
    bx.axvline(0, color=BASE, linewidth=0.8)
    for g in ("BG", "AL", "RS", "RO", "LT", "LU", "IE", "IS"):
        if g in e.index:
            bx.annotate(NAMES[g], (e.composition[g], e.B_ppp[g]), xytext=(4, 3), textcoords="offset points",
                        fontsize=7.8, color=INK2)
    r = np.corrcoef(e.composition, e.B_ppp)[0, 1]
    bx.set_title(f"B. PPP aggregation against composition (r = {r:.2f})")
    bx.set_xlim(e.composition.min() - 0.4, e.composition.max() + 0.9)
    bx.set_xlabel("Composition term: shift in spending towards relatively\ncheap divisions, 2005–2019 (points)")
    bx.set_ylabel("PPP aggregation term in the drift (points)")
    tidy(bx, "both")
    finish(fig, "fig3_composition_and_aggregation",
           "The aggregation part of the drift follows spending shifts towards relatively cheap categories",
           "33 European economies. Panel A: expenditure-weighted standard deviation of the twelve division price "
           "levels; panel B: terms of the decomposition in Figure 2.")


def main() -> None:
    for f in (fig1, fig2, fig3):
        f()


if __name__ == "__main__":
    main()
