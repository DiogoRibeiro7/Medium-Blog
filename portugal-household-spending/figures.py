"""Draw the figures from the outputs of ``spending.py``.

Run after ``spending.py``::

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, PURPLE, GOLD, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#d9a21b", "#c8384b"
QUINTILES = ["QU1", "QU2", "QU3", "QU4", "QU5"]
Q_COLOURS = dict(zip(QUINTILES, [ORANGE, GOLD, MUTED, AQUA, BLUE]))
Q_LABELS = dict(zip(QUINTILES, ["Poorest fifth", "Second", "Middle", "Fourth", "Richest fifth"]))
WAVES = [1988, 1994, 1999, 2005, 2010, 2015, 2020]
WAVE_COLOURS = dict(zip(WAVES, plt.cm.viridis(np.linspace(0.92, 0.08, len(WAVES)))))

HBS = "Eurostat household budget survey (hbs_str_t223)"
NA = "Eurostat national accounts (nama_10_co3_p3)"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "lines.linewidth": 2.0, "legend.frameon": False, "legend.labelcolor": INK_2,
})


def finish(fig: plt.Figure, name: str, source: str) -> None:
    fig.text(0.01, 0.01, f"Source: {source}; author's calculations.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)


def by_quintile_over_waves(hb: pd.DataFrame, col: str, title: str, name: str, ylim=None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for q in QUINTILES:
        s = hb.loc[q, col].dropna()
        ax.plot(s.index, 100 * s.values, color=Q_COLOURS[q], marker="o", ms=4, label=Q_LABELS[q])
    ax.set_title(title)
    ax.set_xticks([y for y in WAVES if y in hb.loc["QU1", col].dropna().index])
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=8.5, ncol=2)
    finish(fig, name, f"{HBS}, imputed rent excluded")


# 1. Basic needs in the national accounts, every year.
def na_components(na: pd.DataFrame) -> None:
    other = na["core_ex_imputed"] - na["food_ex_imputed"] - na["rent_actual_ex_imputed"] - na["energy_ex_imputed"]
    parts = [("Food", na["food_ex_imputed"], ORANGE), ("Actual rent", na["rent_actual_ex_imputed"], PURPLE),
             ("Electricity, gas, fuels", na["energy_ex_imputed"], GOLD), ("Water, dwelling repairs", other, MUTED)]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.stackplot(na.index, *[100 * p[1] for p in parts], colors=[p[2] for p in parts],
                 labels=[p[0] for p in parts], alpha=0.95)
    ax.plot(na.index, 100 * na["wider_ex_imputed"], color=INK, lw=1.4, ls="--",
            label="Adding health and running costs of transport")
    ax.set_title("Basic needs, % of household consumption (imputed rent excluded), 1995–2022")
    ax.set_xlim(na.index.min(), na.index.max())
    ax.set_ylim(0, 52)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    finish(fig, "fig01_basic_needs_national_accounts.png", f"{NA}; 2022 provisional")


# 2. The household saving rate.
def saving_rate(s: pd.Series) -> None:
    s = s.loc[1995:]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(s.index, s.values, color=BLUE, marker="o", ms=3)
    ax.annotate(f"{s.min():.1f}% in {s.idxmin()}", (s.idxmin(), s.min()), xytext=(0, -16),
                textcoords="offset points", ha="center", fontsize=8.5, color=INK_2)
    ax.annotate("pandemic", (2020, s[2020]), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8.5, color=INK_2)
    ax.set_title("Gross household saving rate, % of disposable income")
    ax.set_xlim(1995, s.index.max())
    ax.set_ylim(0, 16)
    finish(fig, "fig02_saving_rate.png", "Eurostat nasa_10_ki (households and NPISH); 2024-25 provisional")


# 3. Engel curves: food share across the income distribution, one line per survey.
def engel(hb: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(1, 6)
    for y in WAVES:
        vals = [100 * hb.loc[(q, y), "food"] for q in QUINTILES]
        ax.plot(x, vals, color=WAVE_COLOURS[y], marker="o", ms=4, label=str(y))
    ax.set_xticks(x, [Q_LABELS[q] for q in QUINTILES])
    ax.set_title("Food, % of consumption, by income fifth and survey year")
    ax.legend(loc="upper right", fontsize=8.5, ncol=4)
    finish(fig, "fig03_food_engel_curves.png", f"{HBS}, imputed rent included")


# 7. Composition of basic needs for each income fifth.
def composition_all(hb: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(11, 4.2), sharey=True)
    parts = [("Food", "food_ex_imputed", ORANGE), ("Actual rent", "rent_actual_ex_imputed", PURPLE),
             ("Electricity, gas, fuels", "energy_ex_imputed", GOLD)]
    for ax, q in zip(axes, QUINTILES):
        d = hb.loc[q].dropna(subset=["core_ex_imputed"])
        x = np.arange(len(d))
        bottom = np.zeros(len(d))
        for label, col, colour in parts:
            ax.bar(x, 100 * d[col].values, bottom=bottom, color=colour, width=0.7, label=label)
            bottom += 100 * d[col].values
        rest = 100 * d["core_ex_imputed"].values - bottom
        ax.bar(x, rest, bottom=bottom, color=MUTED, width=0.7, label="Water, dwelling repairs")
        ax.set_xticks(x, [f"'{str(y)[2:]}" for y in d.index], fontsize=8)
        ax.set_title(Q_LABELS[q], fontsize=10)
    axes[0].set_ylim(0, 62)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Basic needs by income fifth, % of consumption (imputed rent excluded), 1988–2015",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK, y=0.985)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.93), ncol=4, fontsize=8.5,
               frameon=False)
    fig.text(0.01, 0.01, f"Source: {HBS}; author's calculations.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    fig.savefig(FIGURES / "fig07_composition_by_quintile.png", dpi=200)
    plt.close(fig)


# 8. Housing including imputed rent, all seven waves (the 2020 break).
def housing_incl_imputed(hb: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for q in QUINTILES:
        s = hb.loc[q, "housing_incl_imputed"]
        ax.plot(s.index, 100 * s.values, color=Q_COLOURS[q], marker="o", ms=4, label=Q_LABELS[q])
    ax.axvspan(2015.5, 2020.5, color=GRID, alpha=0.6, lw=0)
    ax.text(2018, 13, "2020 wave:\nno split of\nimputed rent", ha="center", fontsize=8, color=INK_2)
    ax.set_title("Housing, water and energy including imputed rent, % of consumption")
    ax.set_xticks(WAVES)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    finish(fig, "fig08_housing_incl_imputed_rent.png", HBS)


# 9. Inequality of consumption, total and after basic needs.
def consumption_ratio(resid: pd.DataFrame) -> None:
    tot = resid["consumption_pps_ae"].unstack("quintile")
    after = resid["after_basic_pps_ae"].unstack("quintile").dropna()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(tot.index, tot["QU5"] / tot["QU1"], color=BLUE, marker="o", ms=4, label="All consumption")
    ax.plot(after.index, after["QU5"] / after["QU1"], color=ORANGE, marker="o", ms=4,
            label="Consumption left after basic needs")
    ax.set_title("Richest fifth over poorest fifth, consumption per adult equivalent")
    ax.set_xticks(WAVES)
    ax.set_ylim(0, 8.5)
    ax.legend(loc="upper right", fontsize=8.5)
    finish(fig, "fig09_consumption_ratio.png", "Eurostat hbs_exp_t133 and hbs_str_t223 (PPS per adult equivalent)")


# 10. Median saving rate by income fifth.
def saving_by_quintile(sq: pd.DataFrame) -> None:
    years = list(sq.columns)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(QUINTILES))
    width = 0.26
    colours = [MUTED, BLUE, GOLD]
    for i, y in enumerate(years):
        ax.bar(x + (i - 1) * width, sq.loc[QUINTILES, y].values, width=width, color=colours[i], label=str(y))
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(x, [Q_LABELS[q] for q in QUINTILES])
    ax.set_title("Median saving rate by income fifth, % of disposable income")
    ax.legend(loc="upper left", fontsize=8.5)
    finish(fig, "fig10_saving_by_quintile.png", "Eurostat icw_sr_03 (experimental statistics)")


# 11. Prices of each basic need since 1996.
def component_prices(cp: pd.DataFrame, price: pd.DataFrame) -> None:
    base = cp.loc[1996]
    idx = 100 * cp / base
    series = [("CP044", "Water and dwelling services", AQUA), ("CP043", "Dwelling repairs", MUTED),
              ("CP045", "Electricity, gas, fuels", GOLD), ("CP041", "Actual rent", PURPLE),
              ("CP01", "Food", ORANGE), ("CP00", "All items", INK)]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for code, label, colour in series:
        ax.plot(idx.index, idx[code], color=colour, lw=2.4 if code == "CP00" else 1.8, label=label)
    ax.set_title("Consumer prices, 1996 = 100 (HICP)")
    ax.set_xlim(1996, idx.index.max())
    ax.legend(loc="upper left", fontsize=8.5)
    finish(fig, "fig11_basic_needs_prices.png", "Eurostat HICP (prc_hicp_aind)")


# 12. Inflation by income fifth.
def inflation_by_quintile(inf: pd.DataFrame) -> None:
    periods = [(2005, 2015), (2015, 2025), (2021, 2023)]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(periods))
    width = 0.15
    for i, q in enumerate(QUINTILES):
        vals = [100 * inf[(inf["start"] == s) & (inf["end"] == e) & (inf["quintile"] == q)]["inflation"].iloc[0]
                for s, e in periods]
        ax.bar(x + (i - 2) * width, vals, width=width, color=Q_COLOURS[q], label=Q_LABELS[q])
    ax.set_xticks(x, [f"{s}–{e}" for s, e in periods])
    ax.set_title("Cumulative inflation of each income fifth's basket, %")
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)
    finish(fig, "fig12_inflation_by_quintile.png",
           "Eurostat HICP divisions weighted by household budget survey shares (2005 and 2015 waves, imputed rent removed)")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    for old in FIGURES.glob("*.png"):
        old.unlink()
    na = pd.read_csv(RESULTS / "na_basic_shares.csv", index_col="year")
    s = pd.read_csv(RESULTS / "saving_rate.csv", index_col="year")["saving_rate"]
    hb = pd.read_csv(RESULTS / "hbs_shares_by_quintile.csv", index_col=["quintile", "year"])
    resid = pd.read_csv(RESULTS / "consumption_after_basic_needs.csv", index_col=["quintile", "year"])
    sq = pd.read_csv(RESULTS / "saving_by_quintile.csv", index_col="quant_inc")
    sq.columns = [int(c) for c in sq.columns]
    cp = pd.read_csv(RESULTS / "component_prices_2015_100.csv", index_col="year")
    price = pd.read_csv(RESULTS / "basic_needs_price.csv", index_col="year")
    inf = pd.read_csv(RESULTS / "inflation_by_quintile.csv")

    na_components(na)
    saving_rate(s)
    engel(hb)
    by_quintile_over_waves(hb, "core_ex_imputed",
                           "Food, rent and household energy, % of consumption, by income fifth",
                           "fig04_core_share_by_quintile.png")
    by_quintile_over_waves(hb, "rent_actual_ex_imputed", "Actual rent, % of consumption, by income fifth",
                           "fig05_rent_share_by_quintile.png")
    by_quintile_over_waves(hb, "energy_ex_imputed", "Electricity, gas and fuels, % of consumption, by income fifth",
                           "fig06_energy_share_by_quintile.png")
    composition_all(hb)
    housing_incl_imputed(hb)
    consumption_ratio(resid)
    saving_by_quintile(sq)
    component_prices(cp, price)
    inflation_by_quintile(inf)


if __name__ == "__main__":
    main()
