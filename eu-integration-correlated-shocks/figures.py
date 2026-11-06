"""Figures for the EU article, from ``results/``.

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

import evidence as ev_mod

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
VIOLET, RED = "#6250d6", "#e34948"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
CREDIT = "Diogo Ribeiro"
NAMES = {
    "BE": "Belgium", "BG": "Bulgaria", "CZ": "Czechia", "DK": "Denmark", "DE": "Germany", "EE": "Estonia",
    "IE": "Ireland", "EL": "Greece", "ES": "Spain", "FR": "France", "HR": "Croatia", "IT": "Italy",
    "CY": "Cyprus", "LV": "Latvia", "LT": "Lithuania", "LU": "Luxembourg", "HU": "Hungary", "MT": "Malta",
    "NL": "Netherlands", "AT": "Austria", "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SI": "Slovenia",
    "SK": "Slovakia", "FI": "Finland", "SE": "Sweden", "EU27_2020": "EU-27",
}

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


def suptitle(fig, text: str, y: float = 1.03) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=y)


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_shock_map() -> None:
    s = pd.read_csv(RES / "evidence_cereal_shocks.csv", index_col=0)
    eu = pd.read_csv(RES / "evidence_eu_shock.csv", index_col=0).iloc[:, 0]
    c = pd.read_csv(RES / "evidence_correlation.csv", index_col=0)
    ev, evec = np.linalg.eigh(c.values)
    order = pd.Series(evec[:, -2], index=c.index).sort_values().index.tolist()
    order = [g for g in order if g != "CY"] + ["CY"]
    data = (100 * s[order].T).clip(-40, 40)
    cmap = LinearSegmentedColormap.from_list("div", ["#c43b3a", "#e34948", "#f3b5ac", "#f0efec", "#b7d3f6", "#3987e5", "#1c5cab"])
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4.2], hspace=0.08)
    ax0 = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=ax0)
    years = s.index.values
    ax0.bar(years, 100 * eu.loc[years], color=[RED if v < 0 else C1 for v in eu.loc[years]], width=0.75)
    ax0.axhline(0, color=MUTED, lw=0.8)
    ax0.set_ylabel("EU-27, %")
    ax0.set_title("Cereal production: deviation from trend, EU aggregate (top) and each member state (bottom)")
    plt.setp(ax0.get_xticklabels(), visible=False)
    im = ax.imshow(data.values, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40), extent=[years[0] - 0.5, years[-1] + 0.5, len(order) - 0.5, -0.5], interpolation="nearest")
    ax.set_yticks(range(len(order)), [NAMES[g] for g in order], fontsize=8.8)
    ax.grid(False)
    for y in (2003, 2007, 2012, 2018, 2022):
        ax0.annotate(str(y), (y, 100 * eu.loc[y] - 1.2), ha="center", va="top", fontsize=8.5, color=INK2)
    cb = fig.colorbar(im, ax=[ax0, ax], fraction=0.02, pad=0.01)
    cb.set_label("% deviation from trend (clipped at ±40)", color=INK2)
    cb.outline.set_visible(False)
    ax.text(1.0, -0.075, "Rows ordered by the second principal component: south-east at the top, north and Baltic at the bottom; Cyprus last (tiny, noisy).", transform=ax.transAxes, ha="right", fontsize=8.5, color=INK2)
    note(fig, "Source: Eurostat apro_cpsh1 (cereals for grain, harvested production). Log deviation from a Hodrick-Prescott trend, lambda 100.")
    save(fig, "fig1_shock_map.png")


def fig2_pooling() -> None:
    s = pd.read_csv(RES / "evidence_cereal_shocks.csv", index_col=0)
    eu = pd.read_csv(RES / "evidence_eu_shock.csv", index_col=0).iloc[:, 0]
    bs = pd.read_csv(RES / "evidence_bootstrap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    sd = (100 * s.std()).drop("CY").sort_values()
    y = np.arange(len(sd))
    ax.hlines(y, 0, sd.values, color=GRID, lw=3)
    ax.scatter(sd.values, y, color=C1, s=42, zorder=3, edgecolor=SURFACE, linewidth=1.2)
    ax.set_yticks(y, [NAMES[g] for g in sd.index], fontsize=8.6)
    ax.axvline(100 * eu.std(), color=C2, lw=2)
    ax.text(100 * eu.std() + 0.4, len(sd) - 2.5, f"EU-27 aggregate: {100 * eu.std():.1f}%", color=C2, fontsize=9.3)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("standard deviation of the yearly shock, % (Cyprus, 72.7%, not shown)")
    ax.set_title("Each country alone vs the union pooled")
    ax = axes[1]
    w = ev_mod.production_weights(ev_mod.cereal_production())
    periods = []
    for a, b in ((2000, 2008), (2008, 2016), (2016, 2024)):
        sub = s.loc[a:b]
        periods.append((f"{a}-{str(b)[2:]}", ev_mod.pairwise(sub.corr().values).mean(), ev_mod.diversification(sub, w)["n_eff"]))
    full = ev_mod.diversification(s, w)["n_eff"]
    x = np.arange(len(periods))
    ax.bar(x, [p[2] for p in periods], color=C1, width=0.55, edgecolor=SURFACE, linewidth=2)
    for xi, p in zip(x, periods):
        ax.text(xi, p[2] + 0.2, f"{p[2]:.1f}\n(mean corr. {p[1]:.2f})", ha="center", va="bottom", fontsize=9.2, color=INK2)
    lo, hi = bs.n_eff.quantile(0.05), bs.n_eff.quantile(0.95)
    ax.axhspan(lo, hi, color=C2, alpha=0.15, zorder=0)
    ax.axhline(full, color=C2, lw=1.6, ls=(0, (4, 2)))
    ax.text(2.45, full + 0.15, f"2000-24: {full:.1f}\n(90% band\n{lo:.1f} to {hi:.1f})", color=C2, fontsize=9, ha="left")
    ax.set_xticks(x, [p[0] for p in periods])
    ax.set_xlim(-0.5, 3.2)
    ax.set_ylim(0, 10)
    ax.set_ylabel("independent producers the union is equivalent to")
    ax.set_title("How much pooling is worth: effective number of independent producers")
    suptitle(fig, "The single market halves harvest risk, and has been diversifying better over time")
    note(fig, "Source: Eurostat apro_cpsh1; effective number = (sum of w_i sd_i)^2 / Var(sum of w_i shock_i), production weights 2015-24; band from a 3-year block bootstrap.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig2_pooling.png")


def fig3_common() -> None:
    fi = pd.read_csv(RES / "evidence_food_inflation.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for g in fi.columns:
        ax.plot(fi.index, 100 * fi[g], color=MUTED, lw=0.8, alpha=0.55)
    ax.plot(fi.index, 100 * fi.mean(axis=1), color=C2, lw=2.6, label="mean of member states")
    hu = fi["HU"]
    ax.annotate(f"Hungary 2022: {100 * hu.loc[2022]:.1f}%", (2022, 100 * hu.loc[2022]), xytext=(2010, 23), fontsize=9.2, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate(f"lowest in 2022: {100 * fi.loc[2022].min():.1f}%", (2022, 100 * fi.loc[2022].min()), xytext=(2012.5, 12), fontsize=9.2, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_ylabel("food price inflation, % (log change of HICP food)")
    ax.set_title("Food inflation in the 27 member states, 2001-2025")
    ax.legend(loc="upper left")
    ax = axes[1]
    lab = ["cereal production\nshocks", "house price\ngrowth", "food price\ninflation"]
    sh = pd.read_csv(RES / "evidence_cereal_shocks.csv", index_col=0)
    hg = pd.read_csv(RES / "evidence_house_price_growth.csv", index_col=0).loc[2011:2025]
    parts = [ev_mod.common_factor_share(sh), ev_mod.common_factor_share(hg), ev_mod.common_factor_share(fi)]
    vals = [100 * p[0] for p in parts]
    corr = [p[1] for p in parts]
    ax.barh(lab, vals, color=[C1, VIOLET, C2], height=0.55, edgecolor=SURFACE, linewidth=2)
    for i, (v, c) in enumerate(zip(vals, corr)):
        ax.text(v + 1.5, i, f"{v:.0f}% (mean corr. {c:.2f})", va="center", fontsize=9.3, color=INK2)
    ax.set_xlim(0, 112)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of variance in the first common component, %")
    ax.set_title("How common is each shock across member states")
    suptitle(fig, "Harvests diverge, prices move together: 2022 hit every member state")
    note(fig, "Source: Eurostat prc_hicp_aind (CP011), apro_cpsh1, prc_hpi_a. Common component = first principal component of the cross-country correlation matrix.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig3_common.png")


def fig4_insurance() -> None:
    cs = pd.read_csv(RES / "model_correlation_sweep.csv")
    cm = pd.read_csv(RES / "model_common_sweep.csv")
    bs = pd.read_csv(RES / "evidence_bootstrap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axes[0]
    ax.plot(cs.mean_corr, cs.unpooled_national_cvar95_pp, color=C2, label="unpooled: each country keeps its own shock")
    ax.plot(cs.mean_corr, cs.pooled_national_cvar95_pp, color=C1, label="pooled: one EU price")
    lo, hi = bs.mean_corr.quantile(0.05) * 0.9, bs.mean_corr.quantile(0.95) * 0.9
    ax.axvspan(lo, hi, color=GRID, alpha=0.8, zorder=0)
    emp = cs.iloc[(cs.mean_corr - 0.208).abs().argmin()]
    ax.scatter([emp.mean_corr], [emp.pooled_national_cvar95_pp], color=C1, s=55, zorder=4, edgecolor=SURFACE, linewidth=1.5)
    ax.scatter([emp.mean_corr], [emp.unpooled_national_cvar95_pp], color=C2, s=55, zorder=4, edgecolor=SURFACE, linewidth=1.5)
    ax.annotate(f"observed: {emp.pooled_national_cvar95_pp:.1f} vs {emp.unpooled_national_cvar95_pp:.1f}", (emp.mean_corr, emp.pooled_national_cvar95_pp), xytext=(0.32, 4), fontsize=9.2, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    k = int(np.flatnonzero(cs.insurance_tail_pp.to_numpy() <= 0)[0])
    x0, x1 = cs.mean_corr.iloc[k - 1], cs.mean_corr.iloc[k]
    y0, y1 = cs.insurance_tail_pp.iloc[k - 1], cs.insurance_tail_pp.iloc[k]
    cross = x0 + (x1 - x0) * y0 / (y0 - y1)
    ax.axvline(cross, color=MUTED, ls=":", lw=1)
    ax.text(cross + 0.01, 3, f"insurance gone\nat about {cross:.2f}", fontsize=9, color=INK2)
    ax.set_xlabel("mean correlation of national supply shocks")
    ax.set_ylabel("bad-year risk where a household lives\n(pp priced out above baseline, worst 5% of years)")
    ax.set_title("Insurance against weather-type shocks shrinks with correlation")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, 1)
    ax = axes[1]
    ax.plot(cm.common_share, 100 * cm.insurance_tail_share, color=C1, label="bad-year risk")
    ax.plot(cm.common_share, 100 * cm.insurance_mean_share, color=C3, label="average harm")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("share of each country's shock that is common to all (and to the world market)")
    ax.set_ylabel("% of the unpooled harm that pooling removes")
    ax.set_title("A common shock cannot be pooled")
    ax.legend(loc="upper right")
    suptitle(fig, "Model: what the single market insures, as shocks become correlated or common")
    note(fig, "Model: 27 member states, shocks calibrated on 2000-24 cereal data; 40,000 simulated years per point; world buffer 5% of consumption.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig4_insurance.png")


def fig5_by_country() -> None:
    bc = pd.read_csv(RES / "model_by_country.csv", index_col=0)
    bc = bc.sort_values("unpooled_p95_excess_pp")
    fig, ax = plt.subplots(figsize=(11, 8.4))
    y = np.arange(len(bc))
    for yi, (g, r) in zip(y, bc.iterrows()):
        ax.plot([r.pooled_p95_excess_pp, r.unpooled_p95_excess_pp], [yi, yi], color=GRID, lw=3, zorder=1)
    ax.scatter(bc.unpooled_p95_excess_pp, y, color=C2, s=48, zorder=3, edgecolor=SURFACE, linewidth=1.2, label="unpooled")
    ax.scatter(bc.pooled_p95_excess_pp, y, color=C1, s=48, zorder=3, edgecolor=SURFACE, linewidth=1.2, label="pooled")
    ax.set_yticks(y, [f"{NAMES[g]} ({r.self_sufficiency:.2f})" for g, r in bc.iterrows()], fontsize=8.8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.2, 120)
    ax.set_xticks([0, 1, 2, 5, 10, 20, 50, 100], ["0", "1", "2", "5", "10", "20", "50", "100"])
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("households priced out in a 1-in-20 bad year, pp above baseline (log scale)")
    ax.set_title("Who the single market protects: exporters gain most, importers take on some risk")
    ax.legend(loc="upper left")
    ax.text(0.99, 0.02, "In brackets: production share / population share (self-sufficiency proxy)", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.6, color=INK2)
    note(fig, "Model: empirical shock correlation; unpooled keeps normal trade flows and leaves each country's own harvest deviation at home.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig5_by_country.png")


def fig6_ageing() -> None:
    fo = pd.read_csv(RES / "evidence_farm_output_ages.csv", index_col=0)
    fa = pd.read_csv(RES / "evidence_farm_ages.csv", index_col=0)
    both = fa.join(fo[["ge65_output_share"]]).drop(index=["EU27_2020"]).sort_values("ge65_output_share")
    eu = fo.loc["EU27_2020", "ge65_output_share"]
    ag = pd.read_csv(RES / "model_ageing.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    y = np.arange(len(both))
    ax.barh(y, both.ge65_share, color=GRID, height=0.75, label="share of holdings")
    ax.barh(y, both.ge65_output_share, color=C2, height=0.45, label="share of standard output")
    ax.set_yticks(y, [NAMES[g] for g in both.index], fontsize=8.6)
    ax.axvline(eu, color=INK2, lw=1, ls=(0, (3, 2)))
    ax.text(eu + 0.6, 1, f"EU-27 output: {eu:.1f}%", fontsize=9, color=INK2)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("% run by farm managers aged 65 and over (latest Farm Structure wave)")
    ax.set_title("Where the farms are run by people at retirement age")
    ax.legend(loc="lower right")
    ax = axes[1]
    names = ["no loss", "loss where managers are old", "same loss spread evenly"]
    cols = [C1, C2, VIOLET]
    vals_mean = ag.mean_excess_pp.values
    vals_tail = ag.national_cvar95_pp.values
    x = np.arange(3)
    ax.bar(x - 0.18, vals_mean, width=0.36, color=[c for c in cols], alpha=0.55, edgecolor=SURFACE, linewidth=1.5)
    ax.bar(x + 0.18, vals_tail, width=0.36, color=cols, edgecolor=SURFACE, linewidth=1.5)
    for xi, a, b in zip(x, vals_mean, vals_tail):
        ax.text(xi - 0.18, max(a, 0) + 0.4, f"{abs(a) if abs(a) < 0.05 else a:.1f}", ha="center", fontsize=9, color=INK2)
        ax.text(xi + 0.18, b + 0.4, f"{b:.1f}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(x, names, fontsize=9.2)
    ax.set_ylabel("pp of households priced out above baseline")
    ax.set_title(f"Half of that output not replaced: {100 * ag.eu_capacity_loss.iloc[1]:.1f}% of EU supply")
    ax.text(0.02, 0.96, "light bars: average year\ndark bars: worst 5% of years", transform=ax.transAxes, fontsize=9, color=INK2, va="top")
    suptitle(fig, "Ageing is a common shock, and the union cannot pool what every member loses")
    note(fig, "Source: Eurostat ef_m_farmang (holdings and standard output by age of manager). Model: same simulated weather years in all three scenarios.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig6_ageing.png")


def fig7_housing() -> None:
    fall = pd.read_csv(RES / "evidence_construction_falling.csv", index_col=0).iloc[:, 0].loc[2006:2025]
    asym = pd.read_csv(RES / "evidence_construction_asymmetry.csv", index_col=0)
    big = asym[asym["drop"] < -0.30].sort_values("down_rate")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    ax.bar(fall.index, 100 * fall.values, color=[C2 if v >= 0.5 else C1 for v in fall.values], width=0.75, edgecolor=SURFACE, linewidth=1)
    ax.axhline(50, color=MUTED, lw=0.8, ls=":")
    ax.set_ylabel("% of member states")
    ax.set_title("Member states where construction employment fell")
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax = axes[1]
    y = np.arange(len(big))
    ax.hlines(y, 100 * big.down_rate, 100 * big.up_rate, color=GRID, lw=3)
    ax.scatter(100 * big.down_rate, y, color=C2, s=48, zorder=3, edgecolor=SURFACE, linewidth=1.2, label="peak to trough")
    ax.scatter(100 * big.up_rate, y, color=C1, s=48, zorder=3, edgecolor=SURFACE, linewidth=1.2, label="trough to 2025")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, [f"{NAMES[g]} ({int(r.peak)}-{int(r.trough)})" for g, r in big.iterrows()], fontsize=8.8)
    ax.set_xlabel("average change of construction employment, % a year")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_title("Down fast, up slowly: the 11 deepest busts")
    ax.legend(loc="upper left", fontsize=9)
    suptitle(fig, "Housing cannot be traded, and its busts arrived together")
    note(fig, "Source: Eurostat nama_10_a10_e (employment, construction). Peak chosen in 2004-2010; trough is the lowest year after it.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig7_housing.png")


def main() -> None:
    fig1_shock_map()
    fig2_pooling()
    fig3_common()
    fig4_insurance()
    fig5_by_country()
    fig6_ageing()
    fig7_housing()


if __name__ == "__main__":
    main()
