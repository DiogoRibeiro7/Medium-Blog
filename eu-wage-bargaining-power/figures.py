"""Figures for the article, from results/ (experiments.py, analysis.py) and data/.

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import minimum_wage_as_paid, national_accounts, oecd_series, read

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, SURFACE, SHADE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb", "#efeeea"
GREY = "#b9b8b3"
CREDIT = "Diogo Ribeiro"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 9.5,
    }
)


def note(fig, text: str, y: float = 0.005) -> None:
    fig.text(0.01, y, text, ha="left", va="bottom", fontsize=8.3, color=INK2, wrap=True)
    fig.text(0.99, y, CREDIT, ha="right", va="bottom", fontsize=8.3, color=MUTED)


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_fingerprints() -> None:
    lev = pd.read_csv(RES / "model_levers.csv")
    panels = [
        ("minimum wage 0.45 -> 0.60", "Minimum wage raised\n(reserve price 0.45 -> 0.60)"),
        ("bargaining weight 0.10 -> 0.25", "Bargaining weight raised\n(beta 0.10 -> 0.25)"),
        ("outside option 0.45 -> 0.55", "Outside option raised\n(b 0.45 -> 0.55 of ability)"),
        ("rival offers halved (non-competes)", "Rival offers halved\n(non-compete clauses)"),
        ("rival offers +50% (mobility)", "Rival offers up by half\n(fewer mobility barriers)"),
        ("equal pay, narrow job bands (20)", "Equal pay within job bands\n(pay transparency, 20 bands)"),
    ]
    qs = ["p10", "p25", "p50", "p75", "p90"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), sharey=True)
    for ax, (key, title) in zip(axes.ravel(), panels):
        g = lev[lev.lever == key].set_index("outcome")
        v = [g.loc[q, "change"] for q in qs]
        sd = [g.loc[q, "change_sd_over_seeds"] for q in qs]
        ax.bar(range(5), v, width=0.62, color=BLUE, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(range(5), v, yerr=[2 * s for s in sd], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
        ax.axhline(0, color=MUTED, linewidth=1)
        for i in (0, 4):
            ax.text(i, v[i] + (1.2 if v[i] >= 0 else -1.2), f"{v[i]:+.1f}%", ha="center",
                    va="bottom" if v[i] >= 0 else "top", fontsize=9, color=INK)
        ax.set_xticks(range(5), ["P10", "P25", "P50", "P75", "P90"])
        ax.set_title(title, fontsize=10.5)
        emp = g.loc["employment_rate", "change"]
        out = g.loc["mean_output", "change"]
        ax.text(0.98, 0.97, f"employment {emp:+.1f} pt\nmatch output {out:+.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
        ax.set_ylim(-25, 36)
    for ax in axes[:, 0]:
        ax.set_ylabel("change in wage quantile, %")
    fig.suptitle("Figure 1. Each lever moves a different part of the simulated wage distribution",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    note(fig, "Agent-based sequential-auction model (model.py), 30,000 workers, 300 firms; baseline: minimum wage 0.45, "
              "beta 0.10, rival offers 5% a month. Mean change against the same seed's baseline over 10 seeds; whiskers "
              "are two standard deviations across seeds.", y=-0.04)
    fig.tight_layout(rect=(0, 0.0, 1, 0.95))
    save(fig, "fig1_lever_fingerprints.png")


def fig2_censoring() -> None:
    g = pd.read_csv(RES / "model_floor_grid.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    titles = {"beta 0.10 -> 0.20": "Bargaining weight 0.10 -> 0.20",
              "lambda1 0.05 -> 0.075": "Rival offers 5% -> 7.5% a month"}
    for ax, (v, t) in zip(axes, titles.items()):
        d = g[g.variant == v].sort_values("kaitz_base")
        lo = np.interp(0.18, d.share_at_floor_base, d.kaitz_base)
        hi = np.interp(0.24, d.share_at_floor_base, d.kaitz_base)
        ax.axvspan(lo, hi, color=SHADE, zorder=0)
        ax.text((lo + hi) / 2, 2.4 if v.startswith("beta") else 7.0, "18-24% of\nworkers on\nthe floor",
                ha="center", va="center", fontsize=8.2, color=INK2)
        for col, lab, c in (("d_p10", "P10", BLUE), ("d_p25", "P25", ORANGE), ("d_p50", "P50", AQUA)):
            ax.plot(d.kaitz_base, d[col], color=c, marker="o", markersize=4.5, label=lab)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_title(t)
        ax.set_xlabel("Kaitz index produced by the minimum wage (minimum / median wage)")
        ax.set_xlim(0.28, 0.92)
        ax.set_ylim(-1, 16)
    axes[0].set_ylabel("change in wage quantile, %")
    axes[0].legend(loc="upper right", ncol=3)
    fig.suptitle("Figure 2. A binding floor censors the other levers at the bottom of the distribution",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    note(fig, "Same model; for each minimum wage from 0.30 to 0.95 (units of the no-floor median), the change in each "
              "quantile when one other parameter is raised, mean over 6 seeds. Shaded: the minimum wages at which 18-24% "
              "of simulated workers are paid the floor.", y=-0.08)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig2_floor_censoring.png")


def fig3_ses() -> None:
    p = pd.read_csv(RES / "ses_floor_panel.csv")
    el = pd.read_csv(RES / "floor_elasticities.csv").set_index("outcome")
    m = p.dropna(subset=["mw_paid"])
    hl = {"PT": ("Portugal", BLUE), "EL": ("Greece", ORANGE), "DE": ("Germany", AQUA)}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, col, lab, key in ((axes[0], "p10_over_median", "10th percentile / median", "p10_over_median"),
                              (axes[1], "p90_over_median", "90th percentile / median", "p90_over_median")):
        ax.scatter(m.kaitz, m[col], s=26, color=GREY, edgecolor=SURFACE, linewidth=1, zorder=2, label="other country-waves")
        for geo, (name, c) in hl.items():
            d = m[m.geo == geo].sort_values("wave")
            ax.plot(d.kaitz, d[col], color=c, marker="o", markersize=6, zorder=3, label=name,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            last = d.iloc[-1]
            ax.annotate(f"{name} {int(d.wave.iloc[0])}-{int(last.wave)}", (last.kaitz, last[col]),
                        xytext=(6, 0), textcoords="offset points", fontsize=8.8, color=INK, va="center")
        e = el.loc[key]
        ax.set_title(f"{lab}\nwithin-country elasticity {e.slope:+.2f} (90% CI {e.ci90_lo:+.2f} to {e.ci90_hi:+.2f})",
                     fontsize=10.5)
        ax.set_xlabel("minimum wage / median monthly earnings (Kaitz index)")
        ax.set_xlim(0.33, 0.72)
    axes[0].set_ylabel("ratio")
    axes[0].legend(loc="upper left", fontsize=8.8)
    fig.suptitle("Figure 3. In EU earnings data the floor pulls up the bottom decile; across countries the top does not follow",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    note(fig, "Eurostat Structure of Earnings Survey 2006, 2010, 2014, 2018, 2022 (full-time employees, October monthly "
              "earnings; enterprises with 10+ employees to 2018) and earn_mw_cur (minimum wage as paid; Greece, Spain and "
              "Portugal converted back from Eurostat's 12-month basis). 21 countries with a statutory minimum wage; "
              "elasticities from log ratios with country and wave fixed effects, bootstrap over countries.", y=-0.1)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig3_floor_in_ses.png")


def fig4_agreements() -> None:
    s = pd.read_csv(RES / "ses_agreement_shares.csv")
    s = s[s.wave == 2022].copy()
    cbc = pd.read_csv(RES / "oecd_coverage.csv").sort_values("year").groupby("geo").tail(1).set_index("geo").value
    s["sort"] = s.none.fillna(-1)
    s = s.sort_values("sort", ascending=False)
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    y = np.arange(len(s))
    left = np.zeros(len(s))
    for col, lab, c in (("multi_employer", "multi-employer (national, sectoral)", BLUE),
                        ("single_employer", "single-employer (company, local unit)", ORANGE),
                        ("other", "other", AQUA), ("none", "no agreement", YELLOW)):
        v = 100 * s[col].fillna(0).values
        ax.barh(y, v, left=left, color=c, edgecolor=SURFACE, linewidth=2, height=0.72, label=lab)
        left += v
    cov = [cbc.get(g, np.nan) for g in s.geo]
    ax.scatter(cov, y, marker="D", s=30, color=INK, zorder=4, label="OECD/AIAS coverage, latest year")
    ax.set_yticks(y, [f"{g}{'*' if np.isnan(n) else ''}" for g, n in zip(s.geo, s.none)])
    ax.set_xlabel("% of employees, enterprises with 10+ employees (sections B-S excluding O)")
    ax.set_xlim(0, 104)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8.8)
    ax.set_title("Figure 4. Who sits at the table: employees by type of collective pay agreement, 2022", pad=48)
    note(fig, "Eurostat earn_ses22_01. * no-agreement share not published; bars show only the categories Eurostat "
              "publishes (coverage in these countries is close to universal through extended sectoral agreements). Diamonds: OECD/AIAS ICTWSS adjusted coverage rate, latest year "
              "available. Portugal's 'other' category (20%) is reported separately by Eurostat.", y=-0.06)
    fig.tight_layout()
    save(fig, "fig4_agreements.png")


def fig5_portugal_floor() -> None:
    mw = minimum_wage_as_paid()
    jan = mw[(mw.geo == "PT") & (mw.semester == "S1")].set_index("year").mw_paid
    na = national_accounts()
    cpe = na[na.geo == "PT"].set_index("year").comp_per_employee_month
    yrs = range(2006, 2026)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    ax.plot(list(yrs), [100 * jan[y] / jan[2006] for y in yrs], color=BLUE, label="minimum wage, as paid")
    ax.plot(list(yrs), [100 * cpe[y] / cpe[2006] for y in yrs], color=ORANGE, label="compensation per employee")
    for ser, c in ((jan, BLUE), (cpe, ORANGE)):
        v = 100 * ser[2025] / ser[2006]
        ax.text(2025.3, v, f"{v:.0f}", color=INK, fontsize=9, va="center")
    ax.set_title("Minimum wage and average pay, index 2006 = 100")
    ax.set_xticks(range(2006, 2026, 4))
    ax.legend(loc="upper left")
    ax.set_xlim(2005.5, 2027)
    ax = axes[1]
    ratio = read("minimum_wage_ratio")
    kz = ratio[(ratio.geo == "PT") & (ratio.nace_r2 == "B-S") & (ratio.indic_se == "MMW_MED_ME_PP")].set_index("time").value
    om = oecd_series("oecd_min_to_average")
    okz = om[(om.geo == "PT") & (om.AGGREGATION_OPERATION == "MEDIAN")].set_index("year").value.sort_index().loc[2000:]
    ses = pd.read_csv(RES / "ses_floor_panel.csv")
    ses = ses[ses.geo == "PT"]
    ax.plot(okz.index, okz.values, color=BLUE, label="OECD: full-time earnings")
    ax.plot(kz.index, kz.values, color=ORANGE, label="Eurostat: 12-month basis")
    ax.plot(ses.wave, 100 * ses.kaitz, color=AQUA, marker="o", linestyle="none", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="SES: October monthly, as paid")
    ax.plot([2019, 2025], [87, 91], color=YELLOW, marker="s", linestyle="none", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="Banco de Portugal: base wage")
    for x, v in ((2019, 87), (2025, 91)):
        ax.text(x, v + 1.6, f"{v}%", ha="center", fontsize=9, color=INK)
    ax.set_ylim(40, 98)
    ax.set_title("Minimum wage / median wage (Kaitz index), four wage concepts, %")
    ax.legend(loc="lower right", fontsize=8.6)
    fig.suptitle("Figure 5. Portugal: the floor grew faster than pay, and how close it sits to the median depends on the wage measured",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    note(fig, "Eurostat earn_mw_cur (January rate; the 14 annual payments converted back to the monthly rate), nama_10_gdp "
              "and nama_10_a10_e (D1 / employees), earn_mw_avgr2 (B-S); OECD MIN2AVE; Structure of Earnings Survey; "
              "Banco de Portugal, Boletim Económico June 2026 (Social Security microdata, private sector).", y=-0.08)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig5_portugal_floor.png")


def fig6_portugal_distribution() -> None:
    ses = pd.read_csv(RES / "ses_floor_panel.csv")
    pt = ses[ses.geo == "PT"].set_index("wave")
    na = national_accounts()
    ws = na.set_index(["geo", "year"]).wage_share_adjusted
    cbc = pd.read_csv(RES / "oecd_coverage.csv")
    cbc = cbc[cbc.geo == "PT"].set_index("year").value
    sh = pd.read_csv(RES / "ses_agreement_shares.csv")
    sh = sh[sh.geo == "PT"].set_index("wave")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    ax = axes[0]
    for col, lab, c in (("mw_paid", "minimum wage", BLUE), ("p10", "P10", ORANGE), ("median", "median", AQUA),
                        ("p90", "P90", YELLOW)):
        v = 100 * pt[col] / pt[col].loc[2006]
        ax.plot(pt.index, v, color=c, marker="o", markersize=5, label=lab)
        dy = {"minimum wage": 3.5, "P10": -3.5}.get(lab, 0.0)
        ax.text(2022.6, v.loc[2022] + dy, f"{lab} {v.loc[2022]:.0f}", fontsize=8.6, color=INK, va="center")
    ax.set_xlim(2005, 2027.5)
    ax.set_title("SES earnings and minimum wage,\nnominal, index 2006 = 100", fontsize=10.5)
    ax = axes[1]
    yrs = list(range(1995, 2026))
    ax.plot(yrs, [100 * ws[("PT", y)] for y in yrs], color=BLUE, label="Portugal")
    ax.plot(yrs, [100 * ws[("EU27_2020", y)] for y in yrs], color=ORANGE, label="EU27")
    ax.set_title("Adjusted wage share, %\n(pay per employee / output per worker)", fontsize=10.5)
    ax.legend(loc="upper right")
    ax = axes[2]
    c = cbc.loc[1995:]
    ax.plot(c.index, c.values, color=BLUE, label="OECD/AIAS coverage (any agreement in force)")
    ax.plot(sh.index, 100 * sh.multi_employer, color=ORANGE, marker="o", markersize=6, label="SES: multi-employer agreement")
    ax.plot(sh.index, 100 * sh.none, color=AQUA, marker="o", markersize=6, label="SES: no agreement")
    ax.set_ylim(0, 105)
    ax.set_title("Collective bargaining, % of employees", fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.2)
    fig.suptitle("Figure 6. Portugal: the bottom decile rode the floor up, the median followed by half, the top barely moved",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    note(fig, "Eurostat Structure of Earnings Survey (full-time, October monthly earnings), earn_mw_cur, nama_10_gdp and "
              "nama_10_a10_e; OECD/AIAS ICTWSS. SES agreement shares refer to enterprises with 10+ employees and are "
              "reported by employers.", y=-0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig6_portugal_distribution.png")


def main() -> None:
    fig1_fingerprints()
    fig2_censoring()
    fig3_ses()
    fig4_agreements()
    fig5_portugal_floor()
    fig6_portugal_distribution()


if __name__ == "__main__":
    main()
