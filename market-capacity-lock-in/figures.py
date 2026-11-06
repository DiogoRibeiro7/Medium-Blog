"""Figures for the article, from ``results/``.

    python figures.py

Writes ``figures/fig1_*.png`` ... ``figures/fig8_*.png`` in the order they appear.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

# reference palette (light mode): categorical slots in fixed order, text inks, surface
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
VIOLET = "#6250d6"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SOURCE_EUROSTAT = "Source: Eurostat"
CREDIT = "Diogo Ribeiro"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
    }
)


def note(fig, text: str) -> None:
    fig.text(0.01, 0.005, f"{text}   |   {CREDIT}", ha="left", va="bottom", fontsize=8.2, color=MUTED)


def label_end(ax, x, y, text, color=INK2, dx=0.6, dy=0.0, **kw):
    ax.annotate(text, (x, y), xytext=(x + dx, y + dy), textcoords="data", color=color, fontsize=9.3, va="center", **kw)


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------


def fig1_farm() -> None:
    al = pd.read_csv(RES / "evidence_agri_labour.csv", index_col=0)
    ao = pd.read_csv(RES / "evidence_agri_output.csv", index_col=0)
    fa = pd.read_csv(RES / "evidence_farm_age.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw={"width_ratios": [1.1, 1.1, 1.0]})

    ax = axes[0]
    ax.plot(al.index, al.non_salaried, color=C1)
    ax.plot(al.index, al.salaried, color=C2)
    label_end(ax, 2025, al.loc[2025, "non_salaried"], "family and\nself-employed", C1, dx=0.8, dy=55)
    label_end(ax, 2025, al.loc[2025, "salaried"], "hired", C2, dx=0.8, dy=-30)
    ax.set_xlim(1980, 2034)
    ax.set_ylim(0, 800)
    ax.set_title("Farm labour by who does it")
    ax.set_ylabel("thousand annual work units")

    ax = axes[1]
    base = 2005
    idx = pd.DataFrame(
        {
            "output volume": ao.output_volume / ao.loc[base, "output_volume"] * 100,
            "labour input": al.total / al.loc[base, "total"] * 100,
            "real farm-gate price": ao.output_real_price / ao.loc[base, "output_real_price"] * 100,
        }
    ).loc[base:2025]
    for col, c in zip(idx.columns, (C1, C2, C3)):
        ax.plot(idx.index, idx[col], color=c)
    label_end(ax, 2025, idx.loc[2025, "output volume"], "output\nvolume", C1, dx=0.4, dy=4)
    label_end(ax, 2025, idx.loc[2025, "real farm-gate price"], "real farm-\ngate price", C3, dx=0.4, dy=-6)
    label_end(ax, 2025, idx.loc[2025, "labour input"], "labour\ninput", C2, dx=0.4)
    ax.axhline(100, color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(base, 2030.5)
    ax.set_title("More output, less labour, flat prices")
    ax.set_ylabel(f"index, {base} = 100")

    ax = axes[2]
    latest = fa.sort_values("time").groupby("geo").tail(1).set_index("geo")
    order = latest.sort_values("ge65_share").index.tolist()
    names = {"PT": "Portugal", "EU27_2020": "EU-27", "ES": "Spain", "IT": "Italy", "EL": "Greece", "FR": "France", "PL": "Poland", "AT": "Austria"}
    y = np.arange(len(order))
    ax.hlines(y, latest.loc[order, "lt35_share"], latest.loc[order, "ge65_share"], color=GRID, lw=3, zorder=1)
    ax.scatter(latest.loc[order, "lt35_share"], y, s=60, color=C3, zorder=3, edgecolor=SURFACE, linewidth=1.5, label="under 35")
    ax.scatter(latest.loc[order, "ge65_share"], y, s=60, color=C2, zorder=3, edgecolor=SURFACE, linewidth=1.5, label="65 and over")
    ax.set_yticks(y, [f"{names[g]} ({int(latest.loc[g, 'time'])})" for g in order])
    for yy, g in zip(y, order):
        ax.text(latest.loc[g, "ge65_share"] + 1.5, yy, f"{latest.loc[g, 'ge65_share']:.0f}%", va="center", fontsize=8.8, color=INK2)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 62)
    ax.set_xlabel("% of farm managers")
    ax.set_title("Who runs the farms, by age")
    ax.legend(loc="lower right")
    fig.suptitle("Portugal's farms: output holds up while the people who own the work grow old", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=1.03)
    note(fig, f"{SOURCE_EUROSTAT} (aact_ali01, aact_eaa05, ef_m_farmang). Real price deflated by Eurostat. Latest Farm Structure wave per country.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig1_farms.png")


def fig2_housing() -> None:
    h = pd.read_csv(RES / "evidence_housing_obs.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    s = h.completions_long.dropna()
    ax.bar(s.index, s.values / 1000, width=0.8, color=C1, edgecolor=SURFACE, linewidth=0.5)
    ax.set_title("Dwellings completed in new construction (INE long series)")
    ax.set_ylabel("thousand dwellings")
    ax.annotate(f"{s.loc[2002] / 1000:.0f}k in 2002", (2002, s.loc[2002] / 1000), xytext=(1984, 125), color=INK2, fontsize=9.3, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate(f"{s.loc[2015] / 1000:.1f}k in 2015", (2015, s.loc[2015] / 1000), xytext=(2014.2, 40), color=INK2, fontsize=9.3, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate(f"{s.loc[2025] / 1000:.1f}k in 2025,\nbelow 1975 ({s.loc[1975] / 1000:.1f}k)", (2025, s.loc[2025] / 1000), xytext=(2013, 62), color=INK2, fontsize=9.3, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(1973.5, 2026.5)

    ax = axes[1]
    base = 2008
    idx = pd.DataFrame(
        {
            "construction employment": h.construction_employment / h.loc[base, "construction_employment"] * 100,
            "house prices / pay": h.price_to_wage / h.loc[base, "price_to_wage"] * 100,
            "completions": h.completions_long / h.loc[base, "completions_long"] * 100,
        }
    ).loc[base:2025]
    for col, c in zip(idx.columns, (C1, C2, C3)):
        ax.plot(idx.index, idx[col], color=c)
    label_end(ax, 2025, idx.loc[2025, "house prices / pay"], "house prices\nrelative to pay", C2, dx=0.3)
    label_end(ax, 2025, idx.loc[2025, "construction employment"], "construction\nemployment", C1, dx=0.3)
    label_end(ax, 2025, idx.loc[2025, "completions"], "completions", C3, dx=0.3)
    ax.axhline(100, color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(base, 2031)
    ax.set_ylim(0, 170)
    ax.set_title(f"Fast down, slow up ({base} = 100)")
    fig.suptitle("Housing: the price signal is loud, the supply answer is slow", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=1.03)
    note(fig, "Source: INE indicator 0011725 (Fonte: Instituto Nacional de Estatística), Eurostat (prc_hpi_a, nama_10_a10_e, nama_10_a10). Pay = wages per employee.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig2_housing.png")


def fig3_exclusion() -> None:
    d = pd.read_csv(RES / "demand_curves.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(d.price_ratio, 100 * d.housing_priced_out, color=C1)
    ax.plot(d.price_ratio, 100 * d.food_priced_out, color=C2)
    label_end(ax, 4.0, 100 * d.housing_priced_out.iloc[-1], "housing\n(5% at the\nreference price)", C1, dx=0.05)
    label_end(ax, 4.0, 100 * d.food_priced_out.iloc[-1], "food\n(3% at the\nreference price)", C2, dx=0.05, dy=-8)
    r = 1.646
    y = 100 * float(np.interp(r, d.price_ratio, d.housing_priced_out))
    ax.scatter([r], [y], s=55, color=C1, edgecolor=SURFACE, linewidth=1.5, zorder=4)
    ax.annotate(f"house prices / pay,\n2025 vs 2015 (x{r:.2f}):\n{y:.0f}% of entrants", (r, y), xytext=(1.9, 8), fontsize=9.2, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(0.5, 4.6)
    ax.set_ylim(0, 100)
    ax.set_xlabel("price relative to the reference price")
    ax.set_ylabel("% of households who cannot afford their need")
    ax.set_title("The market always clears. This is who it clears out")
    note(fig, "Model: lognormal incomes with Portugal's 2025 Gini (30.9), at most 40% of income on the good")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig3_exclusion.png")


def fig4_fold() -> None:
    b = pd.read_csv(RES / "food_bifurcation.csv")
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    o = b[b.M == 1.0]
    up = o[o.stable & (o.K > 0.3)].sort_values("phi")
    mid = o[~o.stable].sort_values("phi")
    zero = o[o.stable & (o.K == 0.0)].sort_values("phi")
    c = b[(b.M == 0.3) & b.stable & (b.K > 0.3)].groupby("phi").K.max()
    ax.plot(c.index, c.values, color=C2, lw=5, alpha=0.45, solid_capstyle="round", label="capped borrowing (30% of use)", zorder=2)
    ax.plot(up.phi, up.K, color=C1, label="open: viable state", zorder=3)
    ax.plot(mid.phi, mid.K, color=C1, ls=(0, (3, 2)), lw=1.6, label="open: threshold (unstable)", zorder=3)
    ax.plot(zero.phi, zero.K, color=C1, lw=3.0, alpha=0.35, label="open: collapsed state", zorder=3)
    ax.text(0.30, 0.71, "capped: the price rises\nand holds the sector here", fontsize=8.8, color=INK2)
    phi_c = up.phi.max()
    k_c = up.K.iloc[-1]
    ax.annotate("", xy=(phi_c + 0.004, 0.02), xytext=(phi_c + 0.004, k_c - 0.01), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
    ax.text(phi_c + 0.009, 0.25, "fold at about 0.28:\nthe sector drops out", fontsize=9.2, color=INK2)
    ax.annotate("", xy=(0.06, 0.02), xytext=(0.22, 0.02), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, ls="--"))
    ax.text(0.02, 0.06, "lowering the premium again does not bring it back:\nthe collapsed state stays stable at every premium shown", fontsize=8.8, color=INK2)
    ax.set_xlim(0, 0.40)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("premium needed to draw people into the physical work (share of the reference price)")
    ax.set_ylabel("anchored capacity (share of consumption)")
    ax.set_title("With open borrowing the price never moves, and the sector falls off a cliff")
    ax.legend(loc="upper right")
    note(fig, "Model: steady states of anchored capacity; world price = reference price; normal exit 2.5% a year")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig4_fold.png")


def fig5_erosion() -> None:
    o = pd.read_csv(RES / "food_erosion_open_shock_year50.csv")
    c = pd.read_csv(RES / "food_erosion_capped_M030.csv")
    e = pd.read_csv(RES / "food_erosion_open_shock_year5.csv")
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.6), sharex=True)
    rows = [("K", "anchored capacity", 1.0), ("price", "price (reference = 1)", 1.0), ("priced_out", "% who cannot afford\ntheir need", 100.0)]
    for ax, (col, lab, mult) in zip(axes, rows):
        ax.plot(o.t, o[col] * mult, color=C1, label="open borrowing, disruption in year 50")
        ax.plot(e.t, e[col] * mult, color=C3, lw=1.4, ls=(0, (4, 2)), label="open borrowing, same disruption in year 5")
        ax.plot(c.t, c[col] * mult, color=C2, label="borrowing capped at 30%")
        ax.set_ylabel(lab)
        for a in (50, 53):
            ax.axvline(a, color=GRID, lw=1.0, zorder=0)
    for ax in axes:
        ax.axvspan(50, 53, color=GRID, alpha=0.6, zorder=0)
        ax.axvspan(5, 8, color=GRID, alpha=0.35, zorder=0)
    axes[0].text(50.6, 0.9, "disruption:\nborrowing\nhalved", fontsize=8.6, color=INK2, va="top")
    axes[1].text(5.4, 2.6, "same disruption\nin year 5:\nno effect", fontsize=8.6, color=INK2, va="top")
    axes[0].legend(loc="lower left", fontsize=9)
    axes[0].set_title("The open market gives no warning: the price stays flat until the bill arrives")
    axes[2].set_xlabel("years (the effort premium rises from 0.10 to 0.35 over the first 40)")
    note(fig, "Model: food case; disruption caps borrowed supply at half of consumption for three years")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig5_erosion.png")


def fig6_wave() -> None:
    rw = pd.read_csv(RES / "food_retirement_wave.csv")
    cmap = LinearSegmentedColormap.from_list("blue", ["#eef4fc", "#9fc3ee", "#2a78d6", "#103d73"])
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    for ax, M, title in zip(axes, (1.0, 0.3), ("Open borrowing", "Borrowing capped at 30% of use")):
        s = rw[rw.M == M].copy()
        s["ratio"] = (s.K_end / s.K_start).clip(0, 1.0)
        piv = s.pivot_table(index="W_years", columns="phi", values="ratio")
        X, Y = np.meshgrid(piv.columns.values, piv.index.values)
        im = ax.pcolormesh(X, Y, piv.values, cmap=cmap, vmin=0, vmax=1.0, shading="nearest", edgecolors=SURFACE, linewidth=1.0)
        if (piv.values < 0.5).any():
            ax.contour(X, Y, piv.values, levels=[0.5], colors=[C2], linewidths=2)
        ax.set_title(title)
        ax.set_xlabel("effort premium before the wave")
        ax.grid(False)
    axes[0].set_ylabel("years over which the 65+ cohort leaves")
    axes[0].annotate("lost for good: right of\nand below the orange line", xy=(0.272, 8), xytext=(0.13, 6), fontsize=9, color=INK, bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    axes[1].text(0.19, 21, "every case recovers", fontsize=9.5, color=INK, ha="center", bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID))
    fig.subplots_adjust(bottom=0.17)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("capacity 60 years after the wave / before it", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("The same retirements are survivable if they are slow, and fatal if they are fast", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=1.02)
    note(fig, "Model: exit rate during the wave removes 51.9% of capacity (Portugal's farm managers aged 65+, 2023) over the years shown")
    save(fig, "fig6_wave.png")


def fig7_housing_model() -> None:
    fit = pd.read_csv(RES / "housing_fit.csv", index_col=0)
    runs = {
        "fitted": ("as it happened (fitted)", INK),
        "no_bust": ("no 2008-2013 bust", C3),
        "floor_80pct_2008_2016": ("capacity floor at 80%, 2008-2016", C1),
        "rebuild_as_fast_as_loss": ("rebuild as fast as the loss", VIOLET),
        "bust_until_2018": ("bust lasting to 2018", C2),
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    ax = axes[0]
    ax.plot(fit.index, fit.data_capacity, color=C1, marker="o", ms=4.5, lw=0, label="construction employment")
    ax.plot(fit.index, fit.model_capacity, color=C1, label="model capacity")
    ax.set_title("Capacity (2008 = 1)")
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.plot(fit.index, fit.data_price, color=C2, marker="o", ms=4.5, lw=0, label="house prices / pay")
    ax.plot(fit.index, fit.model_price, color=C2, label="model price")
    ax.set_title("Price relative to income (2008 = 1)")
    ax.legend(loc="upper left")
    ax = axes[2]
    for key, (lab, col) in runs.items():
        df = pd.read_csv(RES / f"housing_{key}.csv")
        df = df[(df.year >= 2006) & (df.year <= 2030.01)]
        ax.plot(df.year, 100 * df.priced_out, color=col, lw=2.4 if key == "fitted" else 1.8, label=lab)
    ax.set_title("% of entrant households priced out")
    for a in axes:
        a.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.set_xlim(2006, 2030)
    ax.legend(loc="upper left", fontsize=8.8)
    fig.suptitle("Portugal's housing, 2008-2025, fitted with three parameters, and four other histories", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=1.03)
    note(fig, "Data: Eurostat (nama_10_a10_e, prc_hpi_a, nama_10_a10). Model: capacity speeds from construction employment; household growth from lfst_hhnhtych")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig7_housing_model.png")


def fig8_exposure() -> None:
    e = pd.read_csv(RES / "food_shock_exposure.csv")
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    for cut, col in zip((0.15, 0.30, 0.50), (C3, C1, C2)):
        s = e[e.cut == cut]
        ax.plot(100 * s.K, 100 * s.priced_out, color=col)
        label_end(ax, 2, 100 * s.priced_out.iloc[4] + 7, f"{cut:.0%} of borrowed supply lost", col, dx=0, ha="left")
    ax.axvline(93.4, color=MUTED, lw=0.9, ls=":")
    ax.text(92.5, 62, "anchored share at the\nmodel's starting point", ha="right", fontsize=8.8, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("anchored capacity when the disruption hits (% of consumption)")
    ax.set_ylabel("% who cannot afford their need")
    ax.set_title("What anchored capacity buys: protection when borrowing fails")
    note(fig, "Model: food case; price clears the market during the disruption; Gini 30.9")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig8_exposure.png")


def main() -> None:
    fig1_farm()
    fig2_housing()
    fig3_exclusion()
    fig4_fold()
    fig5_erosion()
    fig6_wave()
    fig7_housing_model()
    fig8_exposure()


if __name__ == "__main__":
    main()
