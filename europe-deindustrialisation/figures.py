"""Figures for the article, from ``results/``.

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dependencies import STRATEGIC

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
BLUE_LIGHT = "#86b6ef"
INK, INK2, MUTED, GRID, SURFACE, SHADE, GREY = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb", "#efeeea", "#b5b4ae"
CREDIT = "Diogo Ribeiro"
NAMES = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CY": "Cyprus", "CZ": "Czechia", "DE": "Germany", "DK": "Denmark",
    "EE": "Estonia", "EL": "Greece", "ES": "Spain", "FI": "Finland", "FR": "France", "HR": "Croatia", "HU": "Hungary",
    "IE": "Ireland", "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "MT": "Malta", "NL": "Netherlands",
    "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia",
    "EU27": "EU-27", "EU15": "EU-15", "UK": "United Kingdom", "US": "United States",
}
EU27 = [k for k in NAMES if len(k) == 2 and k not in ("UK", "US")]
OUTLIERS = ["IE", "LU", "MT", "CY"]

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
        "axes.titlesize": 11,
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


def note(fig, text: str, y: float = -0.02) -> None:
    fig.text(0.01, y, f"{text}   |   {CREDIT}", ha="left", va="top", fontsize=8.2, color=MUTED)


def suptitle(fig, text: str, y: float = 1.02) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=y)


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pct(ax, axis: str = "y", decimals: int = 0) -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def fig1_long_run() -> None:
    s = pd.read_csv(RES / "shares_long.csv")
    geos = ["DE", "FR", "IT", "ES", "PT", "UK", "US", "EU15"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), sharex=True, sharey=True)
    for ax, g in zip(axes.flat, geos):
        d = s[(s.geo == g) & (s.year <= 2024)].set_index("year")
        ax.plot(d.index, d.emp_share, color=BLUE, label="share of employment")
        ax.plot(d.index, d.real_share, color=ORANGE, label="share of value added, 2020 prices")
        e = d.emp_share.dropna()
        if len(e):
            y0 = e.idxmax()
            ax.plot([y0], [e.max()], "o", color=BLUE, ms=6, mec=SURFACE, mew=1.5)
            ax.annotate(f"peak {e.max():.0%}\n{y0}", (y0, e.max()), xytext=(6, 2), textcoords="offset points", fontsize=8.2, color=INK2, va="bottom")
            ax.annotate(f"{e.iloc[-1]:.0%}", (e.index[-1], e.iloc[-1]), xytext=(3, -10), textcoords="offset points", fontsize=8.2, color=INK2)
        ax.set_title(NAMES[g], fontsize=10.5)
        ax.set_xlim(1958, 2030)
        ax.set_ylim(0, 0.42)
        pct(ax)
    axes[0, 0].legend(loc="lower left", fontsize=8.6, handlelength=1.4)
    suptitle(fig, "Manufacturing's share of jobs fell almost everywhere; its share of real output fell far less", y=1.03)
    note(fig, "Sources: AMECO (European Commission) via DBnomics. Germany before 1991 is West Germany. Real shares rescaled to equal the nominal share in 2020. EU-15 from 1995")
    fig.tight_layout()
    save(fig, "fig1_long_run_shares.png")


def fig2_countries() -> None:
    d = pd.read_csv(RES / "decomposition.csv")
    d = d[(d.start == 2000) & (d.end == 2024) & d.geo.isin(EU27 + ["EU27"])].sort_values("dlog_real_share")
    fig, ax = plt.subplots(figsize=(8.6, 8.4))
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        bold = r.geo == "EU27"
        if bold:
            ax.axhspan(yi - 0.5, yi + 0.5, color=SHADE, zorder=0)
        ax.plot([r.dlog_emp_share, r.dlog_real_share], [yi, yi], color=GREY, lw=1.4, zorder=1)
    ax.scatter(d.dlog_emp_share, y, s=58, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="employment share")
    ax.scatter(d.dlog_real_share, y, s=50, marker="D", color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=4, label="real value-added share")
    ax.set_yticks(y)
    ax.set_yticklabels([NAMES[g] for g in d.geo])
    for lab, g in zip(ax.get_yticklabels(), d.geo):
        if g == "EU27":
            lab.set_fontweight("bold")
            lab.set_color(INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("change 2000–2024, log points (−0.10 ≈ a 10% fall)")
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.legend(loc="lower right", ncol=1)
    ax.set_title("Jobs share against real output share, 2000–2024, sorted by the real share", loc="left")
    note(fig, "Sources: AMECO via DBnomics; author's calculations. Bulgaria lacks a complete series. Ireland's output is inflated by multinational accounting")
    fig.tight_layout()
    save(fig, "fig2_jobs_vs_output_by_country.png")


def fig3_goods_balance() -> None:
    x = pd.read_csv(RES / "external_cross_section.csv").dropna(subset=["dlog_real_share", "d_goods_balance"])
    core = x[~x.geo.isin(OUTLIERS)]
    out = x[x.geo.isin(OUTLIERS)]
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(0, color=MUTED, lw=1)
    ax.scatter(core.dlog_real_share, core.d_goods_balance, s=48, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="member states")
    ax.scatter(out.dlog_real_share, out.d_goods_balance, s=48, facecolor="none", edgecolor=MUTED, linewidth=1.4, zorder=3, label="IE, LU, MT, CY (excluded from the fit)")
    b, a = np.polyfit(core.dlog_real_share, core.d_goods_balance, 1)
    xs = np.linspace(core.dlog_real_share.min(), core.dlog_real_share.max(), 50)
    ax.plot(xs, a + b * xs, color=ORANGE, lw=2, label=f"least-squares line, slope {b:.0f} points per log point")
    offsets = {"SE": (-16, -4), "EL": (5, -10), "BE": (-6, 6), "IT": (-12, 5), "DE": (-12, 5)}
    for _, r in x.iterrows():
        ax.annotate(r.geo, (r.dlog_real_share, r.d_goods_balance), xytext=offsets.get(r.geo, (4, 3)), textcoords="offset points", fontsize=8.2, color=INK2)
    ax.set_xlabel("change in manufacturing's real value-added share, 2000–2024 (log points)")
    ax.set_ylabel("change in goods balance, 2000–2024 (points of GDP)")
    ax.grid(axis="x", color=GRID)
    ax.legend(loc="upper left", fontsize=8.8)
    ax.set_title("Member states that kept their real manufacturing share tended to improve their goods balance more", loc="left")
    note(fig, "Sources: Eurostat nama_10_gdp (P61, P71, B1GQ), AMECO; author's calculations")
    fig.tight_layout()
    save(fig, "fig3_goods_balance_vs_real_share.png")


def fig4_regions() -> None:
    r = pd.read_csv(RES / "regions_panel.csv")
    panels = [("d_rel_gdp_0823", "GDP per head relative to the national mean,\nchange 2008–2023 (log points)"), ("d_log_pop_0823", "population, change 2008–2023\n(log points)")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    for ax, (col, lab) in zip(axes, panels):
        d = r.dropna(subset=["exposure", col, "country"]).copy()
        d = d[d.groupby("country")[col].transform("size") > 1]
        d["x"] = d.exposure - d.groupby("country").exposure.transform("mean")
        d["y"] = d[col] - d.groupby("country")[col].transform("mean")
        ax.scatter(d.x, d.y, s=12, color=GREY, alpha=0.6, linewidth=0, zorder=1)
        d["bin"] = pd.qcut(d.x, 10, labels=False)
        bm = d.groupby("bin")[["x", "y"]].mean()
        ax.scatter(bm.x, bm.y, s=60, color=BLUE, edgecolor=SURFACE, linewidth=1.6, zorder=3, label="mean of each tenth of regions")
        b = float(np.sum(d.x * d.y) / np.sum(d.x**2))
        xs = np.linspace(d.x.quantile(0.01), d.x.quantile(0.99), 50)
        ax.plot(xs, b * xs, color=ORANGE, lw=2, zorder=2, label=f"within-country slope {b:.4f}")
        ax.axhline(0, color=MUTED, lw=1)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_xlim(d.x.quantile(0.005) - 0.5, d.x.quantile(0.995) + 0.5)
        ax.set_ylim(d.y.quantile(0.01) * 1.3, d.y.quantile(0.99) * 1.3)
        ax.set_ylabel(lab)
        ax.set_xlabel("manufacturing jobs gained or lost 2000–2008, % of 2000 employment\n(relative to the country mean)")
        ax.grid(axis="x", color=GRID)
        ax.legend(loc="lower right", fontsize=8.6)
        ax.set_title(f"{len(d)} NUTS 2 regions, {d.country.nunique()} member states", fontsize=10)
    suptitle(fig, "Regions that lost more manufacturing jobs before 2008 fell behind slowly afterwards")
    note(fig, "Sources: Eurostat nama_10r_3empers, nama_10r_2gdp, demo_r_d2jan; author's calculations. Both axes demeaned within country; no further controls", y=-0.04)
    fig.tight_layout()
    save(fig, "fig9_regional_exposure.png")


def fig5_china() -> None:
    m = pd.read_csv(RES / "dependencies_manufactured.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot(m.year, m.china_share, color=BLUE)
    for yr in (2001, 2008, 2019, int(m.year.max())):
        v = float(m.loc[m.year == yr, "china_share"].iloc[0])
        ax.plot([yr], [v], "o", color=BLUE, ms=6, mec=SURFACE, mew=1.5)
        ax.annotate(f"{v:.0%}", (yr, v), xytext=(-6, 4) if yr != 2001 else (0, 8), textcoords="offset points", fontsize=8.6, color=INK2, ha="center" if yr == 2001 else "right")
    ax.annotate("China joins the WTO", (2001, float(m.loc[m.year == 2001, "china_share"].iloc[0])), xytext=(2003, 0.04), fontsize=8.4, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_ylim(0, 0.36)
    pct(ax)
    ax.set_title("China's share of extra-EU manufactured imports", fontsize=10.5)
    ax = axes[1]
    ax.plot(m.year, m.chapters_china_top, color=BLUE)
    last = m.iloc[-1]
    ax.plot([last.year], [last.chapters_china_top], "o", color=BLUE, ms=6, mec=SURFACE, mew=1.5)
    ax.annotate(f"{int(last.chapters_china_top)} of {int(last.chapters)}", (last.year, last.chapters_china_top), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=8.6, color=INK2)
    first = m.iloc[0]
    ax.annotate(f"{int(first.chapters_china_top)}", (first.year, first.chapters_china_top), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.6, color=INK2)
    ax.set_ylim(0, 50)
    ax.set_title("Manufacturing chapters in which China is the largest supplier", fontsize=10.5)
    suptitle(fig, "In a quarter of a century China went from one supplier among many to the largest one", y=1.04)
    note(fig, "Source: Eurostat Comext DS-045409, EU-27 imports from outside the EU, HS chapters 28–96 except 71 and 93; author's calculations", y=-0.03)
    fig.tight_layout()
    save(fig, "fig7_china_share_manufactured.png")


def fig6_strategic() -> None:
    c = pd.read_csv(RES / "dependencies_concentration.csv", dtype={"product": str})
    c = c[c["product"].isin(STRATEGIC)]
    a = c[c.year == 2019].set_index("product").china_share
    b = c[c.year == 2024].set_index("product").china_share
    top = c[c.year == 2024].set_index("product")
    d = pd.DataFrame({"y2019": a, "y2024": b}).dropna().sort_values("y2024")
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    y = np.arange(len(d))
    for yi, (p, r) in zip(y, d.iterrows()):
        ax.annotate("", xy=(r.y2024, yi), xytext=(r.y2019, yi), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4, shrinkA=4, shrinkB=5, mutation_scale=10))
        lead = top.loc[p, "top_partner"]
        txt = f"{r.y2019:.0%} → {r.y2024:.0%}" + ("" if lead == "CN" else f"  (largest: {lead} {top.loc[p, 'top_share']:.0%})")
        ax.annotate(txt, (max(r.y2019, r.y2024), yi), xytext=(8, -3), textcoords="offset points", fontsize=8.4, color=INK2)
    ax.scatter(d.y2019, y, s=46, color=BLUE_LIGHT, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="2019")
    ax.scatter(d.y2024, y, s=52, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=4, label="2024")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{STRATEGIC[p]} ({p})" for p in d.index], fontsize=9)
    ax.set_xlim(0, 1.08)
    pct(ax, "x")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("China's share of extra-EU imports (by value)")
    ax.legend(loc="lower right")
    ax.set_title("China's share of extra-EU imports in fifteen strategic products, 2019 and 2024", loc="left", fontsize=10.5)
    note(fig, "Source: Eurostat Comext DS-045409, HS 4-digit headings; author's calculations. Labels name the largest supplier where it is not China")
    fig.tight_layout()
    save(fig, "fig8_strategic_products.png")


def fig7_gas() -> None:
    g = pd.read_csv(RES / "dependencies_gas_partners.csv").set_index("year")
    parts = ["RU", "NO", "DZ", "US", "QA"]
    g["other"] = (1 - g[parts].sum(axis=1)).clip(lower=0)
    labels = {"RU": "Russia", "NO": "Norway", "DZ": "Algeria", "US": "United States", "QA": "Qatar", "other": "other (incl. UK)"}
    cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREY]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.stackplot(g.index, [g[p] for p in parts + ["other"]], colors=cols, edgecolor=SURFACE, linewidth=1.2, labels=[labels[p] for p in parts + ["other"]])
    # direct labels: names inside the wide bands, 2024 values in the right margin
    cum = g[parts + ["other"]].cumsum(axis=1)
    for p, yr in (("RU", 2004), ("NO", 2004), ("DZ", 1996), ("other", 2009)):
        mid = cum.loc[yr, p] - g.loc[yr, p] / 2
        ax.text(yr, mid, labels[p], ha="center", va="center", fontsize=9, color=SURFACE if p in ("RU", "NO") else INK, fontweight="bold")
    last = int(g.index.max())
    ypos = {p: cum.loc[last, p] - g.loc[last, p] / 2 for p in parts + ["other"]}
    ypos["QA"] = max(ypos["QA"], ypos["US"] + 0.045)
    ypos["other"] = max(ypos["other"], ypos["QA"] + 0.045)
    for p in parts + ["other"]:
        ax.annotate(f"{labels[p]} {g.loc[last, p]:.0%}", (last, cum.loc[last, p] - g.loc[last, p] / 2), xytext=(last + 0.6, ypos[p]), textcoords="data", va="center", fontsize=8.6, color=INK, annotation_clip=False, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.plot([2021], [g.loc[2021, "RU"]], "o", color=SURFACE, ms=5, zorder=5)
    ax.annotate(f"2021: {g.loc[2021, 'RU']:.0%}", (2021, g.loc[2021, "RU"]), xytext=(-6, -12), textcoords="offset points", ha="right", fontsize=8.6, color=SURFACE, fontweight="bold")
    ax.set_xlim(1990, 2024)
    ax.set_ylim(0, 1)
    pct(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=6, fontsize=8.8)
    ax.set_title("Where the EU's imported natural gas came from, share of extra-EU imports by energy content", loc="left", fontsize=10.5)
    note(fig, "Source: Eurostat nrg_ti_gas (TJ, gross calorific value), EU-27 imports from outside the EU, pipeline and LNG; author's calculations", y=-0.12)
    fig.tight_layout()
    save(fig, "fig6_gas_suppliers.png")


def fig8_production() -> None:
    ip = pd.read_csv(RES / "turn_industrial_production.csv").set_index("time")
    ip = ip[(ip.index >= 2015) & (ip.index <= 2025)]
    focus = {"EU27": INK, "DE": BLUE, "IT": AQUA, "PL": ORANGE}
    rest = ["FR", "ES"]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.axhline(100, color=MUTED, lw=1)
    for g in rest:
        s = ip[g].dropna()
        ax.plot(s.index, s, color=GREY, lw=1.4)
        ax.annotate(f"{NAMES[g]} {s.iloc[-1]:.0f}", (s.index[-1], s.iloc[-1]), xytext=(5, {"FR": 5, "ES": -1}[g]), textcoords="offset points", va="center", fontsize=8.6, color=MUTED)
    for g, col in focus.items():
        s = ip[g].dropna()
        ax.plot(s.index, s, color=col, lw=2.2 if g == "EU27" else 2)
        dy = {"EU27": -6, "DE": -5, "IT": 5, "PL": 0}.get(g, 0)
        ax.annotate(f"{NAMES[g]} {s.iloc[-1]:.0f}", (s.index[-1], s.iloc[-1]), xytext=(5, dy), textcoords="offset points", va="center", fontsize=8.8, color=INK, fontweight="bold" if g == "EU27" else "normal")
    ax.set_xlim(2015, 2026.5)
    ax.set_xticks(range(2015, 2026, 2))
    ax.set_ylabel("index, 2021 = 100")
    ax.set_title("Manufacturing output, 2015–2025: the European total is flat, Germany and Italy are falling", loc="left", fontsize=10.5)
    note(fig, "Source: Eurostat sts_inpr_a, NACE C, calendar adjusted, annual averages")
    fig.tight_layout()
    save(fig, "fig10_industrial_production.png")


def fig9_target() -> None:
    w = pd.read_csv(RES / "turn_windows.csv")
    txt = (RES / "turn_summary.txt").read_text()
    need = float(txt.split("required growth of the nominal share: ")[1].split("%")[0]) / 100
    need_p = float(txt.split("keep falling at the 2000-2024 pace: the real share must grow ")[1].split("%")[0]) / 100
    hist = float(txt.split("EU-27 2000-2024: real share ")[1].split("%")[0]) / 100
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    bins = np.arange(-0.08, 0.07, 0.005)
    ax.hist(w.growth.clip(-0.0799, 0.0699), bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(hist, color=INK, lw=2, label=f"EU-27, 2000–2024: {hist:+.2%} a year")
    ax.axvline(need, color=ORANGE, lw=2, ls="--", label=f"needed for 20% of GDP by 2035 if relative prices stop falling: {need:+.2%}")
    ax.axvline(need_p, color=ORANGE, lw=2, ls=":", label=f"needed if they keep falling at the 2000–2024 pace: {need_p:+.2%}")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.4)
    ax.legend(loc="upper left", fontsize=8.8, frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=1)
    above = int((w.growth >= need).sum())
    ax.set_xlim(-0.08, 0.07)
    pct(ax, "x")
    ax.set_xlabel("average yearly growth of manufacturing's real value-added share over an 11-year window (log)")
    ax.set_ylabel("member-state windows")
    ax.set_title(f"Every 11-year window for every member state since 1995: {above} of {len(w)} grew as fast as the 2035 target requires", loc="left", fontsize=10.5)
    note(fig, "Sources: AMECO via DBnomics, Eurostat; author's calculations. Windows start in 1995 or later and end no later than 2024")
    fig.tight_layout()
    save(fig, "fig9_target_arithmetic.png")


def fig_russia_energy() -> None:
    r = pd.read_csv(RES / "dependencies_russia_energy.csv").set_index("year")
    r = r[r.index >= 1995]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    ax = axes[0]
    ax.plot(r.index, r.gas_russia_share_of_consumption, color=BLUE, label="supplied by Russia")
    ax.plot(r.index, r.gas_eu_production_share_of_consumption, color=ORANGE, label="produced in the EU")
    for col, yr, dy in (("gas_russia_share_of_consumption", 2021, 8), ("gas_russia_share_of_consumption", 2024, 8), ("gas_eu_production_share_of_consumption", 1995, 8), ("gas_eu_production_share_of_consumption", 2024, -14)):
        v = r.loc[yr, col]
        ax.plot([yr], [v], "o", color=BLUE if "russia" in col else ORANGE, ms=6, mec=SURFACE, mew=1.5)
        ax.annotate(f"{v:.0%}", (yr, v), xytext=(10, 2) if yr == 1995 else (0, dy), textcoords="offset points", ha="left" if yr == 1995 else "center", fontsize=8.6, color=INK2)
    ax.axvspan(2022, 2024.5, color=SHADE, zorder=0)
    ax.set_xlim(1995, 2025)
    ax.set_ylim(0, 0.7)
    pct(ax)
    ax.legend(loc="upper right", fontsize=8.8)
    ax.set_title("Natural gas: share of EU consumption", fontsize=10.5)
    ax = axes[1]
    ax.plot(r.index, r.oil_russia_share_of_extra_imports, color=BLUE, label="crude oil")
    ax.plot(r.index, r.coal_russia_share_of_extra_imports, color=AQUA, label="hard coal")
    ax.plot(r.index, r.gas_russia_share_of_extra_imports, color=MUTED, lw=1.4, ls="--", label="natural gas")
    for col, yr, col_c in (("coal_russia_share_of_extra_imports", 2021, AQUA), ("oil_russia_share_of_extra_imports", 2021, BLUE)):
        v = r.loc[yr, col]
        ax.plot([yr], [v], "o", color=col_c, ms=6, mec=SURFACE, mew=1.5)
        ax.annotate(f"{v:.0%}", (yr, v), xytext=(-6, 6), textcoords="offset points", ha="right", fontsize=8.6, color=INK2)
    ax.axvspan(2022, 2024.5, color=SHADE, zorder=0)
    ax.set_xlim(1995, 2025)
    ax.legend(loc="upper left", fontsize=8.8)
    ax.set_title("Russia's share of EU imports from outside the EU", fontsize=10.5)
    suptitle(fig, "Russian energy: a dependence that grew as EU gas production fell, and was cut within two years", y=1.04)
    note(fig, "Sources: Eurostat nrg_ti_gas, nrg_ti_oil, nrg_ti_sff, nrg_cb_gas (EU-27); author's calculations. Shaded: 2022 onwards", y=-0.03)
    fig.tight_layout()
    save(fig, "fig5_russian_energy.png")


def fig_screen() -> None:
    y = pd.read_csv(RES / "screen_by_year.csv")
    a = pd.read_csv(RES / "screen_single_supplier_by_year.csv")
    groups = [("CN", "China", BLUE), ("US", "United States", ORANGE), ("RU", "Russia", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    x = np.arange(len(y))
    for ax, data, getter, ylab, title in (
        (axes[0], y, lambda d, p: d[f"dependent_top_{p}"] if p else d.dependent - sum(d[f"dependent_top_{q}"] for q, _, _ in groups), "products (HS 6-digit)", "Commission screen: number of products"),
        (axes[1], a, lambda d, p: (d[f"value_{p}"] if p else d.value_other) / 1e9, "EUR billion of extra-EU imports", "Single-supplier screen: value of imports"),
    ):
        bottom = np.zeros(len(data))
        for p, lab, colr in groups + [(None, "other suppliers", GREY)]:
            v = getter(data, p).to_numpy(dtype=float)
            ax.bar(x, v, bottom=bottom, color=colr, edgecolor=SURFACE, linewidth=1.5, width=0.72, label=lab)
            bottom += v
        for xi, tot in zip(x, bottom):
            ax.annotate(f"{tot:,.0f}", (xi, tot), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.2, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(data.year, fontsize=9)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylim(0, bottom.max() * 1.12)
    axes[0].legend(loc="upper left", fontsize=8.6, ncol=2)
    suptitle(fig, "Dependence on outside suppliers, by largest supplier, 2002–2025", y=1.04)
    note(
        fig,
        "Source: Eurostat Comext DS-045409, every HS 6-digit product; author's calculations. Both screens require extra-EU imports > 50% of EU imports and > EU exports.\n"
        "Commission screen: HHI of outside suppliers > 0.4. Single-supplier screen: largest outside supplier > 40% of extra-EU imports. Values in current euros",
        y=-0.03,
    )
    fig.tight_layout()
    save(fig, "fig3_dependency_screen.png")


def fig_single_country() -> None:
    top = pd.read_csv(RES / "screen_largest_single_supplier.csv", dtype={"product": str}).head(15)
    lab = pd.read_csv(HERE / "data" / "hs6_labels.csv", dtype={"product": str}).set_index("product").label
    top = top.iloc[::-1]
    cols = {"CN": BLUE, "US": ORANGE, "RU": AQUA}
    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    yy = np.arange(len(top))
    ax.barh(yy, top.imports_extra / 1e9, color=[cols.get(p, GREY) for p in top.top_partner], height=0.66, edgecolor=SURFACE, linewidth=1.5)
    for yi, (_, r) in zip(yy, top.iterrows()):
        mark = "" if r.dependent else " *"
        ax.annotate(f"{NAMES_ISO.get(r.top_partner, r.top_partner)} {r.top_share:.0%}{mark}", (r.imports_extra / 1e9, yi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.3, color=INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels([short_label(lab.get(p, p), code=p) + f" ({p})" for p in top["product"]], fontsize=8.8)
    ax.set_xlim(0, top.imports_extra.max() / 1e9 * 1.3)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("extra-EU imports, EUR billion")
    last = int(pd.read_csv(RES / "screen_by_year.csv").year.max())
    ax.set_title(f"The fifteen largest products under the single-supplier screen, {last}", loc="left", fontsize=10.5)
    note(fig, "Source: Eurostat Comext DS-045409; author's calculations. Labels: largest outside supplier and its share of extra-EU imports. Bars: China blue, United States orange,\nother grey. Names shortened from the HS nomenclature. * not flagged by the Commission screen (HHI of outside suppliers at or below 0.4)")
    fig.tight_layout()
    save(fig, "fig4_single_country_products.png")


NAMES_ISO = {"CN": "China", "US": "United States", "RU": "Russia", "GB": "United Kingdom", "CH": "Switzerland", "NO": "Norway", "TR": "Türkiye", "JP": "Japan", "KR": "South Korea", "IN": "India", "TW": "Taiwan", "VN": "Vietnam", "BR": "Brazil", "UA": "Ukraine", "BY": "Belarus"}


SHORT = {
    "271111": "Liquefied natural gas",
    "850760": "Lithium-ion batteries",
    "841191": "Parts of turbojets and turbopropellers",
    "293399": "Heterocyclic compounds with nitrogen, n.e.s.",
    "854143": "Photovoltaic modules and panels",
    "230400": "Soya-bean oilcake",
    "890190": "Cargo vessels (excl. tankers, reefers, ferries)",
    "030214": "Fresh or chilled Atlantic salmon",
    "271112": "Liquefied propane",
    "120190": "Soya beans",
    "853400": "Printed circuits",
    "852491": "LCD flat panel display modules",
    "100590": "Maize",
    "293339": "Pyridine-ring compounds, n.e.s.",
    "293590": "Sulphonamides",
}


def short_label(s: str, n: int = 50, code: str | None = None) -> str:
    if code in SHORT:
        return SHORT[code]
    s = str(s).replace("{s}", "(s)").split(";")[0].split(" (excl.")[0].split(", whether or not")[0]
    s = s[0].upper() + s[1:] if s else s
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def main() -> None:
    fig1_long_run()
    fig2_countries()
    fig_screen()
    fig_single_country()
    fig_russia_energy()
    fig7_gas()
    fig5_china()
    fig6_strategic()
    fig4_regions()
    fig8_production()


if __name__ == "__main__":
    main()
