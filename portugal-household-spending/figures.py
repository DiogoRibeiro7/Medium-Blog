"""Draw the figures from the outputs of ``spending.py`` and ``deeper.py``.

Run after both::

    python spending.py
    python deeper.py
    python figures.py

File names follow the order in which the figures appear in the article.
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
CASH_WAVES = [1988, 1994, 1999, 2005, 2010, 2015]
WAVE_COLOURS = dict(zip(WAVES, plt.cm.viridis(np.linspace(0.92, 0.08, len(WAVES)))))
COUNTRY_COLOURS = {"PT": ORANGE, "ES": BLUE, "EL": AQUA, "IT": PURPLE}
COUNTRY_LABELS = {"PT": "Portugal", "ES": "Spain", "EL": "Greece", "IT": "Italy"}

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


def finish(fig: plt.Figure, name: str, source: str, top: float = 1.0) -> None:
    fig.text(0.01, 0.01, f"Source: {source}; author's calculations.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, top))
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)


def suptitle(fig: plt.Figure, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK, y=0.985)


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


# 1. Basic needs in the national accounts, every year, with the resident-basis bound.
def na_components(na: pd.DataFrame) -> None:
    other = na["core_ex_imputed"] - na["food_ex_imputed"] - na["rent_actual_ex_imputed"] - na["energy_ex_imputed"]
    parts = [("Food", na["food_ex_imputed"], ORANGE), ("Actual rent", na["rent_actual_ex_imputed"], PURPLE),
             ("Electricity, gas, fuels", na["energy_ex_imputed"], GOLD), ("Water, dwelling repairs", other, MUTED)]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.stackplot(na.index, *[100 * p[1] for p in parts], colors=[p[2] for p in parts],
                 labels=[p[0] for p in parts], alpha=0.95)
    ax.plot(na.index, 100 * na["core_ex_imputed_resident"], color=INK, lw=1.6, ls=":",
            label="Core needs over residents' consumption (upper bound)")
    ax.plot(na.index, 100 * na["wider_ex_imputed"], color=INK, lw=1.4, ls="--",
            label="Adding health and transport (running costs and services)")
    ax.set_title("Basic needs, % of household consumption (imputed rent excluded), 1995–2022")
    ax.set_xlim(na.index.min(), na.index.max())
    ax.set_ylim(0, 56)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    finish(fig, "fig01_basic_needs_national_accounts.png",
           "Eurostat nama_10_co3_p3 (domestic concept) and nama_10_gdp; 2022 provisional")


# 2. Price and volume behind the aggregate core share.
def price_volume(pv: pd.DataFrame) -> None:
    periods = [(1996, 2008), (2008, 2013), (2013, 2019), (2019, 2020), (2020, 2022)]
    tot, pr = [], []
    for a, b in periods:
        tot.append(100 * np.log(pv.loc[b, "share"] / pv.loc[a, "share"]))
        pr.append(100 * np.log(pv.loc[b, "rel_price_hicp"] / pv.loc[a, "rel_price_hicp"]))
    vo = [t - p for t, p in zip(tot, pr)]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(periods))
    ax.bar(x - 0.18, pr, width=0.34, color=GOLD, label="Relative price of core needs")
    ax.bar(x + 0.18, vo, width=0.34, color=BLUE, label="Relative volume (residual)")
    ax.scatter(x, tot, color=INK, marker="D", s=36, zorder=3, label="Change in the core share")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(x, [f"{a}–{b}" for a, b in periods])
    ax.set_ylabel("log points")
    ax.set_title("Change in the core basic-needs share, split into price and volume")
    ax.set_ylim(-17.5, 24)
    ax.legend(loc="upper left", fontsize=8.5, ncol=3)
    finish(fig, "fig02_core_share_price_volume.png",
           "Eurostat nama_10_co3_p3 and HICP (Törnqvist price indices); volume is the residual")


# 3. The household saving rate.
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
    finish(fig, "fig03_saving_rate.png", "Eurostat nasa_10_ki (households and NPISH); 2025 provisional")


# 4. Engel curves: food share across the income distribution, one line per survey.
def engel(hb: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(1, 6)
    for y in WAVES:
        vals = [100 * hb.loc[(q, y), "food"] for q in QUINTILES]
        ax.plot(x, vals, color=WAVE_COLOURS[y], marker="o", ms=4, label="2020 (2022/23)" if y == 2020 else str(y))
    ax.set_xticks(x, [Q_LABELS[q] for q in QUINTILES])
    ax.set_title("Food, % of consumption, by income fifth and survey year")
    ax.legend(loc="upper right", fontsize=8.5, ncol=4)
    finish(fig, "fig04_food_engel_curves.png", f"{HBS}, imputed rent included; the 2020 wave was collected in 2022/23")


# 5. Budget elasticities from Working-Leser fits, one dot per wave.
ELASTICITY_ITEMS = [
    ("rent_actual", "Actual rent"), ("housing_cash", "Housing, cash items"), ("energy", "Electricity, gas, fuels"),
    ("food", "Food"), ("health", "Health"), ("communications", "Communications"), ("water_repairs", "Water, repairs"),
    ("clothing", "Clothing"), ("transport_running", "Transport running costs, services"),
    ("restaurants_hotels", "Restaurants, hotels"), ("recreation", "Recreation"), ("vehicles", "Vehicle purchases"),
    ("education", "Education"),
]


def engel_elasticities(wl: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.4))
    y = np.arange(len(ELASTICITY_ITEMS))[::-1]
    for yi, (item, _label) in zip(y, ELASTICITY_ITEMS):
        d = wl[wl["item"] == item].set_index("year")["elasticity"]
        ax.plot([d.min(), d.max()], [yi, yi], color=GRID, lw=3, zorder=1)
        for wave, v in d.items():
            ax.scatter(v, yi, color=WAVE_COLOURS[wave], s=30, zorder=2,
                       label=str(wave) if item == ELASTICITY_ITEMS[0][0] else None)
    ax.axvline(1, color=INK_2, lw=0.9, ls="--")
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.text(1.04, y.max() + 0.6, "luxuries →", fontsize=8.5, color=INK_2, va="center")
    ax.text(0.96, y.max() + 0.6, "← necessities", fontsize=8.5, color=INK_2, va="center", ha="right")
    ax.set_yticks(y, [label for _item, label in ELASTICITY_ITEMS])
    ax.set_ylim(-0.7, y.max() + 1.1)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("budget elasticity at the mean share")
    suptitle(fig, "How spending on each item rises with total spending, by survey wave")
    ax.legend(loc="center right", fontsize=8, ncol=2, title="wave", title_fontsize=8)
    finish(fig, "fig05_engel_elasticities.png",
           "Eurostat hbs_str_t223 and hbs_exp_t133; Working–Leser fits across the five fifths, cash shares", top=0.95)


# 10. The discretionary part of the budget, poorest and richest fifth.
DISC_GROUPS = [
    ("Restaurants, hotels", ["restaurants_hotels"], ORANGE),
    ("Clothing, furnishings", ["clothing", "furnishings"], GOLD),
    ("Vehicle purchases", ["vehicles"], BLUE),
    ("Recreation, education", ["recreation", "education"], AQUA),
    ("Communications", ["communications"], PURPLE),
    ("Alcohol, tobacco, other", ["alcohol_tobacco", "miscellaneous"], MUTED),
]


def discretionary(cs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, q in zip(axes, ["QU1", "QU5"]):
        d = cs.loc[q].loc[CASH_WAVES]
        x = np.arange(len(d))
        bottom = np.zeros(len(d))
        for label, cols, colour in DISC_GROUPS:
            v = 100 * d[cols].sum(axis=1).to_numpy()
            ax.bar(x, v, bottom=bottom, color=colour, width=0.7, label=label)
            bottom += v
        for xi, b in zip(x, bottom):
            ax.text(xi, b + 0.8, f"{b:.0f}", ha="center", fontsize=8, color=INK_2)
        ax.set_xticks(x, [str(y) for y in d.index])
        ax.set_title(Q_LABELS[q], fontsize=10)
    axes[0].set_ylim(0, 72)
    handles, labels = axes[0].get_legend_handles_labels()
    suptitle(fig, "Discretionary items, % of cash consumption, poorest and richest fifth")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.94), ncol=6, fontsize=8, frameon=False)
    finish(fig, "fig10_discretionary_by_quintile.png", f"{HBS}, imputed rent excluded", top=0.9)


# 11. Core basic needs by household type, age of the reference person and urbanisation.
SPLIT_PANELS = [
    ("split_hhcomp.csv", "Household type", [
        ("single adult", "Single adult", ORANGE), ("single adult with children", "Single adult, children", RED),
        ("two adults", "Two adults", GOLD), ("two adults with children", "Two adults, children", BLUE),
        ("three or more adults", "Three+ adults", MUTED), ("three or more adults with children", "Three+ adults, children", AQUA)]),
    ("split_age.csv", "Age of reference person", [
        ("under 30", "Under 30", AQUA), ("30 to 44", "30–44", BLUE), ("45 to 59", "45–59", GOLD),
        ("60 and over", "60 and over", ORANGE)]),
    ("split_deg_urb.csv", "Degree of urbanisation", [
        ("cities", "Cities", BLUE), ("towns and suburbs", "Towns and suburbs", GOLD), ("rural areas", "Rural areas", ORANGE)]),
]


def household_types() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    for ax, (fname, title, groups) in zip(axes, SPLIT_PANELS):
        d = pd.read_csv(RESULTS / fname).set_index(["group", "year"])
        for g, label, colour in groups:
            s = d.loc[g, "core_ex_imputed"].dropna()
            ax.plot(s.index, 100 * s.values, color=colour, marker="o", ms=3.5, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(CASH_WAVES, [f"'{str(y)[2:]}" for y in CASH_WAVES])
        ax.legend(loc="upper right", fontsize=7.5)
    axes[0].set_ylim(24, 50)
    suptitle(fig, "Food, rent, energy, water and repairs, % of cash consumption, by type of household")
    finish(fig, "fig11_household_types.png", "Eurostat hbs_str_t224, hbs_str_t225, hbs_str_t226, imputed rent excluded",
           top=0.95)


# 12. Portugal next to Spain, Greece and Italy.
def peer_countries(pe: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for geo in ("PT", "ES", "EL", "IT"):
        for q, ls in (("QU1", "-"), ("QU5", "--")):
            try:
                s = pe.loc[(geo, q), "core_ex_imputed"]
            except KeyError:
                continue
            axes[0].plot(s.index, 100 * s.values, color=COUNTRY_COLOURS[geo], ls=ls, marker="o", ms=3.5,
                         label=f"{COUNTRY_LABELS[geo]}" if q == "QU1" else None)
        r = pe.loc[(geo, "QU1"), "rent_actual"]
        axes[1].plot(r.index, 100 * r.values, color=COUNTRY_COLOURS[geo], marker="o", ms=3.5,
                     label=COUNTRY_LABELS[geo])
    axes[0].set_title("Core basic needs: poorest fifth (solid), richest fifth (dashed)", fontsize=10)
    axes[1].set_title("Actual rent, poorest fifth", fontsize=10)
    for ax in axes:
        ax.set_xticks(WAVES, [f"'{str(y)[2:]}" for y in WAVES])
    axes[0].set_ylim(15, 62)
    axes[0].legend(loc="center left", fontsize=8.5)
    axes[1].legend(loc="upper left", fontsize=8.5)
    suptitle(fig, "Share of cash consumption, %, Portugal and three southern EU countries")
    finish(fig, "fig12_peer_countries.png",
           "Eurostat hbs_str_t223, imputed rent excluded; 2020 wave where components exist (not Portugal); Italy to 2005",
           top=0.93)


# 13. Housing including imputed rent, all seven waves (the 2020 break).
def housing_incl_imputed(hb: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for q in QUINTILES:
        s = hb.loc[q, "housing_incl_imputed"]
        ax.plot(s.index, 100 * s.values, color=Q_COLOURS[q], marker="o", ms=4, label=Q_LABELS[q])
    ax.axvspan(2015.5, 2020.5, color=GRID, alpha=0.6, lw=0)
    ax.text(2018, 13, "2020 wave,\nfieldwork\n2022/23:\nno split of\nimputed rent", ha="center", fontsize=8, color=INK_2)
    ax.set_title("Housing, water and energy including imputed rent, % of consumption")
    ax.set_xticks(WAVES)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    finish(fig, "fig13_housing_incl_imputed_rent.png", f"{HBS}; the 2020 wave was collected in 2022/23")


# 14. Inequality of consumption, total and after basic needs.
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
    finish(fig, "fig14_consumption_ratio.png",
           "Eurostat hbs_exp_t133 and hbs_str_t223 (PPS per adult equivalent); the 2020 wave was collected in 2022/23")


# 15. Euro amounts per household and month, 2015/16, against the minimum wage.
def euro_amounts(eu: pd.DataFrame, mw_month: float) -> None:
    d = eu.loc[2015].loc[QUINTILES]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    y = np.arange(len(QUINTILES))[::-1]
    ax.barh(y, d["core_hh"], color=ORANGE, height=0.6, label="Core basic needs")
    ax.barh(y, d["left_hh"], left=d["core_hh"], color=BLUE, height=0.6, label="Other cash consumption")
    for yi, (_q, r) in zip(y, d.iterrows()):
        ax.text(r["core_hh"] / 2, yi, f"€{r['core_hh']:.0f}", ha="center", va="center", fontsize=8.5, color="white")
        ax.text(r["cash_hh"] + 40, yi, f"€{r['cash_hh']:,.0f}", va="center", fontsize=8.5, color=INK_2)
    ax.axvline(mw_month, color=INK, lw=1.1, ls="--")
    ax.text(mw_month + 30, y.max() + 0.45, f"minimum wage over 12 months, €{mw_month:.0f}", fontsize=8, color=INK)
    ax.set_yticks(y, [Q_LABELS[q] for q in QUINTILES])
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 2900)
    ax.set_ylim(-0.6, y.max() + 0.9)
    ax.set_title("Cash consumption per household and month, 2015/16, euros (approximate)")
    ax.legend(loc="upper right", fontsize=8.5)
    finish(fig, "fig15_euro_amounts.png",
           "Eurostat hbs_exp_t133 and hbs_str_t223 converted with INE's mean expenditure per household; DGERT")


# 16. Median saving rate by income fifth.
def saving_by_quintile(sq: pd.DataFrame) -> None:
    years = list(sq.columns)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(QUINTILES))
    width = 0.26
    colours = [MUTED, BLUE, GOLD]
    for i, y in enumerate(years):
        ax.bar(x + (i - 1) * width, sq.loc[QUINTILES, y].values, width=width, color=colours[i],
               label="around 2020 (HBS 2022)" if y == 2020 else str(y))
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(x, [Q_LABELS[q] for q in QUINTILES])
    ax.set_title("Median saving rate by income fifth, % of disposable income")
    ax.legend(loc="upper left", fontsize=8.5)
    finish(fig, "fig16_saving_by_quintile.png", "Eurostat icw_sr_03 (experimental statistics)")


# 17. Prices of each basic need since 1996.
def component_prices(cp: pd.DataFrame) -> None:
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
    finish(fig, "fig17_basic_needs_prices.png", "Eurostat HICP (prc_hicp_aind)")


# 18. Inflation by income fifth over fixed windows.
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
    finish(fig, "fig18_inflation_by_quintile.png",
           "Eurostat HICP divisions weighted by household budget survey shares (2005 and 2015 waves, imputed rent removed)")


# 19. The chained index since 1996: the gap between the poorest and richest fifth, and its sensitivity.
def chained_gap(idx: pd.DataFrame, sens: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [2.3, 1]})
    ax = axes[0]
    ax.plot(idx.index, idx["QU1"] - idx["QU5"], color=ORANGE, label="Poorest minus richest fifth")
    ax.plot(idx.index, idx["QU1"] - idx["HICP"], color=INK_2, lw=1.4, ls="--", label="Poorest fifth minus headline HICP")
    ax.axhline(0, color=INK_2, lw=0.8)
    for y in (2005, 2010, 2015):
        ax.axvline(y, color=AXIS, lw=0.8, ls=":")
    ax.text(2010.3, 5.6, "dotted lines: survey\nweights updated", fontsize=7.5, color=MUTED)
    ax.set_xlim(1996, idx.index.max())
    ax.set_ylabel("index points, 1996 = 100")
    ax.set_title("Chained basket indices, difference", fontsize=10)
    ax.legend(loc="upper left", fontsize=8.5)
    ax = axes[1]
    labels = list(sens.index)
    ax.barh(np.arange(len(labels))[::-1], sens.values, color=[ORANGE] + [MUTED] * (len(labels) - 1), height=0.6)
    for yi, v in zip(np.arange(len(labels))[::-1], sens.values):
        ax.text(v + 0.15, yi, f"{v:+.1f}", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(np.arange(len(labels))[::-1], [l.replace("fixed", "weights fixed at") for l in labels], fontsize=8.5)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 9.5)
    ax.set_title("Gap in 2025 under other weights", fontsize=10)
    suptitle(fig, "Inflation of the poorest fifth's basket against the richest fifth's, 1996–2025")
    finish(fig, "fig19_chained_inflation_gap.png",
           "Eurostat HICP divisions and household budget survey weights (1999, 2005, 2010, 2015 waves)", top=0.95)


# 20. HICP weights against resident budget shares, and what that did to 2021-2023.
DIV_LABELS = {"CP01": "Food", "CP02": "Alcohol, tobacco", "CP03": "Clothing", "CP04": "Housing (cash)",
              "CP05": "Furnishings", "CP06": "Health", "CP07": "Transport", "CP08": "Communications",
              "CP09": "Recreation", "CP10": "Education", "CP11": "Restaurants, hotels", "CP12": "Other"}


def hicp_weights(hw: pd.DataFrame) -> None:
    hw = hw.sort_values("contribution_to_gap")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    y = np.arange(len(hw))
    ax = axes[0]
    ax.barh(y + 0.2, 100 * hw["hicp_weight_2023"], height=0.38, color=INK_2, label="HICP weight, 2023")
    ax.barh(y - 0.2, 100 * hw["survey_cash_all_2015"], height=0.38, color=GOLD, label="Budget survey, all households, 2015")
    ax.set_yticks(y, [DIV_LABELS[c] for c in hw.index])
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_title("Weight, % of the basket", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    v = 100 * hw["contribution_to_gap"]
    ax.barh(y, v, color=[BLUE if x > 0 else MUTED for x in v], height=0.6)
    for yi, (c, r) in zip(y, hw.iterrows()):
        ax.text(max(v[c], 0) + 0.01, yi, f"price {100 * r['price_change_2021_2023']:+.0f}%", va="center",
                fontsize=7.5, color=INK_2)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.25, 0.65)
    ax.tick_params(axis="y", left=False)
    ax.set_title("Contribution to the 2021–2023 gap, percentage points", fontsize=10)
    suptitle(fig, "Why the headline index rose more than residents' baskets in 2021–2023")
    finish(fig, "fig20_hicp_weights.png", "Eurostat prc_hicp_inw, prc_hicp_aind and hbs_str_t211", top=0.95)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    for old in FIGURES.glob("fig*.png"):
        old.unlink()
    na = pd.read_csv(RESULTS / "na_basic_shares.csv", index_col="year")
    s = pd.read_csv(RESULTS / "saving_rate.csv", index_col="year")["saving_rate"]
    hb = pd.read_csv(RESULTS / "hbs_shares_by_quintile.csv", index_col=["quintile", "year"])
    resid = pd.read_csv(RESULTS / "consumption_after_basic_needs.csv", index_col=["quintile", "year"])
    sq = pd.read_csv(RESULTS / "saving_by_quintile.csv", index_col="quant_inc")
    sq.columns = [int(c) for c in sq.columns]
    cp = pd.read_csv(RESULTS / "component_prices_2015_100.csv", index_col="year")
    inf = pd.read_csv(RESULTS / "inflation_by_quintile.csv")
    pv = pd.read_csv(RESULTS / "core_share_price_volume_hicp.csv", index_col="year")
    wl = pd.read_csv(RESULTS / "working_leser.csv")
    cs = pd.read_csv(RESULTS / "cash_shares_by_quintile.csv", index_col=["quintile", "year"])
    pe = pd.read_csv(RESULTS / "peers_by_quintile.csv", index_col=["geo", "quintile", "year"])
    eu = pd.read_csv(RESULTS / "euro_levels.csv", index_col=["wave", "quintile"])
    idx = pd.read_csv(RESULTS / "quintile_price_index_chained.csv", index_col=0)
    sens = pd.read_csv(RESULTS / "quintile_inflation_weight_sensitivity.csv", index_col=0)["qu1_minus_qu5_2025"]
    hw = pd.read_csv(RESULTS / "hicp_weights_vs_survey.csv", index_col="coicop")
    mw = pd.read_csv(RESULTS / "minimum_wage_fieldwork.csv", index_col="wave")["monthly_over_12"]

    na_components(na)
    price_volume(pv)
    saving_rate(s)
    engel(hb)
    engel_elasticities(wl)
    by_quintile_over_waves(hb, "core_ex_imputed",
                           "Food, rent, energy, water and repairs, % of consumption, by income fifth",
                           "fig06_core_share_by_quintile.png")
    by_quintile_over_waves(hb, "rent_actual_ex_imputed", "Actual rent, % of consumption, by income fifth",
                           "fig07_rent_share_by_quintile.png")
    by_quintile_over_waves(hb, "energy_ex_imputed", "Electricity, gas and fuels, % of consumption, by income fifth",
                           "fig08_energy_share_by_quintile.png")
    composition_all(hb)
    discretionary(cs)
    household_types()
    peer_countries(pe)
    housing_incl_imputed(hb)
    consumption_ratio(resid)
    euro_amounts(eu, mw[2015])
    saving_by_quintile(sq)
    component_prices(cp)
    inflation_by_quintile(inf)
    chained_gap(idx, sens)
    hicp_weights(hw)


# 9. Composition of basic needs for each income fifth.
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
    suptitle(fig, "Basic needs by income fifth, % of consumption (imputed rent excluded), 1988–2015")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.93), ncol=4, fontsize=8.5,
               frameon=False)
    finish(fig, "fig09_composition_by_quintile.png", HBS, top=0.86)


if __name__ == "__main__":
    main()
