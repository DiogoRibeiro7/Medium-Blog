"""Figures for the article, from results/ (empirics.py, experiments.py, forecast.py).

    python figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inputs import KAITZ_BASE, REFERENCE_KAITZ

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUES = ("#9ec5f2", "#4f93e2", "#1b5aa8")              # light -> dark, for coverage (a magnitude)
INK, INK2, MUTED, GRID, SURFACE, SHADE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb", "#efeeea"
GREY = "#b9b8b3"
CREDIT = "Diogo Ribeiro"
PATH_COLOURS = {"constant_bite": AQUA, "agreement": BLUE, "programme": ORANGE, "union_2027": YELLOW}
PATH_SHORT = {"constant_bite": "indexed to the median", "agreement": "agreement, 1,070 in 2029",
              "programme": "agreement, 1,100 in 2029", "union_2027": "1,100 in 2027, then indexed"}

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


def _path_fan(ax, kp: pd.DataFrame, concept: str, years: tuple[int, ...], label_dy: dict[str, float]) -> None:
    for name in ("constant_bite", "agreement", "programme", "union_2027"):
        d = kp[(kp.path == name) & (kp.concept == concept) & kp.year.isin(years)].sort_values("year")
        col = PATH_COLOURS[name]
        if name in ("agreement", "programme"):
            ax.fill_between(d.year, d.q25, d.q75, color=col, alpha=0.18, lw=0)
            ax.plot(d.year, d.q5, color=col, lw=0.9, ls=":")
            ax.plot(d.year, d.q95, color=col, lw=0.9, ls=":")
        ax.plot(d.year, d.q50, color=col, lw=2.2)
        last = d.iloc[-1]
        ax.annotate(PATH_SHORT[name], (last.year, last.q50), xytext=(6, label_dy.get(name, 0)), textcoords="offset points",
                    va="center", fontsize=8.8, color=INK2)


def fig1_bite_paths() -> None:
    kp = pd.read_csv(RES / "forecast_kaitz_paths.csv")
    emp = json.loads((RES / "empirics.json").read_text())
    p = pd.read_csv(RES / "ses_panel_pt.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6), gridspec_kw={"wspace": 0.55})

    ax = axes[0]
    ax.axhline(1.0, color=INK2, lw=1, ls="--")
    ax.text(2019.1, 1.004, "minimum wage = median base wage", fontsize=8.8, color=INK2, va="bottom")
    ax.plot(list(KAITZ_BASE), list(KAITZ_BASE.values()), "o", color=INK, ms=7, zorder=5)
    for y, v in KAITZ_BASE.items():
        ax.annotate(f"{100 * v:.0f}%", (y, v), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=9, color=INK)
    _path_fan(ax, kp, "base wage (Banco de Portugal)", (2025, 2026, 2027, 2028, 2029),
              {"constant_bite": -2, "agreement": -9, "programme": 4, "union_2027": 8})
    ax.set_xlim(2018.6, 2029.4)
    ax.set_ylim(0.84, 1.10)
    ax.set_xticks(range(2019, 2030))
    ax.set_xticklabels([str(y) if y % 2 else "" for y in range(2019, 2030)])
    ax.set_ylabel("minimum wage / median base wage")
    ax.set_title("Base wages, private sector (Banco de Portugal anchor)")

    ax = axes[1]
    kmax = emp["elasticity"]["kaitz_max"]
    ax.axhline(kmax, color=INK2, lw=1, ls="--")
    ax.text(2006.2, kmax + 0.004, f"highest bite in the EU sample, 2006-2022 ({kmax:.2f})", fontsize=8.8, color=INK2, va="bottom")
    ax.plot(p.wave, p.kaitz, "o-", color=INK, ms=6, lw=1.2, zorder=5)
    ax.annotate("observed (SES)", (2014, float(p.set_index("wave").kaitz[2014])), xytext=(0, -16), textcoords="offset points",
                ha="center", fontsize=8.8, color=INK)
    _path_fan(ax, kp, "SES October earnings", tuple(range(2022, 2030)),
              {"constant_bite": -3, "agreement": -8, "programme": 3, "union_2027": 9})
    ax.set_xlim(2005.5, 2029.4)
    ax.set_ylim(0.45, 0.84)
    ax.set_ylabel("minimum wage / median monthly earnings")
    ax.set_title("Monthly earnings, full-time (Structure of Earnings Survey)")
    fig.suptitle("Portugal: the minimum wage against the median under four paths for 2027-2029",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=1.0)
    note(fig, "Lines: median of 4,000 draws; shaded: 25th-75th percentile; dotted: 5th and 95th (shown for the agreement and the "
              "1,100 target). Draws combine wage-growth projections (Ministry of Finance, Banco de Portugal, CFP) with AR(1) noise, "
              "the estimated spillover of the minimum wage to the median and Portugal's drift.\nSources: Banco de Portugal (June 2026), "
              "Eurostat SES and national accounts; own calculations.", y=-0.07)
    save(fig, "fig1_bite_paths.png")


def fig2_decision_2029() -> None:
    risk = pd.read_csv(RES / "forecast_risk_curve.csv")
    emp_f = pd.read_csv(RES / "forecast_employment.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), gridspec_kw={"wspace": 0.18, "width_ratios": [1, 1.15]})

    ax = axes[0]
    ax.plot(risk.mw_2029, risk.p_ge_095, color=INK2, lw=2)
    ax.plot(risk.mw_2029, risk.p_ge_100, color=INK, lw=2, ls="--")
    rr = risk.set_index("mw_2029")
    ax.annotate("minimum at 95% of the median or more", (1022, float(rr.p_ge_095[1020])), xytext=(0, 12),
                textcoords="offset points", ha="left", fontsize=9, color=INK2,
                bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"))
    ax.annotate("minimum at the median or above", (1022, float(rr.p_ge_100[1020])), xytext=(0, 12),
                textcoords="offset points", ha="left", fontsize=9, color=INK,
                bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"))
    for x, name in ((1070, "agreement"), (1100, "programme")):
        ax.axvline(x, color=PATH_COLOURS[name], lw=1.4)
        r = risk.set_index("mw_2029").loc[x]
        ax.plot([x, x], [r.p_ge_095, r.p_ge_100], "o", color=PATH_COLOURS[name], ms=7, zorder=5)
        ax.text(x + 2, 0.04, f"{x} euro", color=INK2, fontsize=9)
    ax.set_xlim(1020, 1160)
    ax.set_ylim(0, 1)
    ax.set_xlabel("minimum wage in 2029, euro a month (2027 at 970 and 2028 at 1,020)")
    ax.set_ylabel("probability across draws")
    ax.set_title("How likely the minimum reaches the median base wage")

    ax = axes[1]
    paths = ["agreement", "programme", "union_2027"]
    methods = [("OWE -0.04", "published elasticity, broad groups since 2010 (-0.04)", "o", GREY),
               ("OWE -0.14", "published elasticity, median of 88 studies (-0.14)", "s", MUTED),
               ("model", "model: matches below the floor end", "D", INK)]
    for i, name in enumerate(paths):
        for j, (meth, lab, mk, col) in enumerate(methods):
            y = i * 4 + j
            r = emp_f[(emp_f.path == name) & (emp_f.method == meth)].iloc[0]
            ax.plot([r.q5 / 1e3, r.q95 / 1e3], [y, y], color=col, lw=2.2, solid_capstyle="round")
            ax.plot(r.q50 / 1e3, y, mk, color=col, ms=8, mec=SURFACE, mew=1.5, zorder=5,
                    label=lab if i == 0 else None)
        ax.text(-188, i * 4 - 0.75, PATH_SHORT[name], fontsize=9.5, color=INK, fontweight="bold", va="center")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks([])
    ax.set_xlim(-190, 20)
    ax.set_ylim(-1.4, 4 * len(paths) - 1.4)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("employment in 2029 against the indexed path, thousand jobs")
    ax.set_title("Jobs, by the assumption that carries them")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=1, fontsize=8.8)
    fig.suptitle("The 2029 decision: how close the minimum gets to the median, and what it may cost in jobs",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=1.0)
    note(fig, "Left: base-wage Kaitz index in 2029 across 4,000 draws, scanning the 2029 minimum wage. Right: dots are medians, "
              "lines 5th-95th percentiles; the published elasticities are applied to the wage gain of the quarter of employees "
              "nearest the floor (Portugal's own P10 elasticity); 4.56 million employees.", y=-0.24)
    save(fig, "fig2_decision_2029.png")


def fig3_model_along_bite() -> None:
    raw = pd.read_csv(RES / "model_floor_grid_raw.csv")
    g = raw.groupby("m").agg(["mean", "min", "max"])
    kp = pd.read_csv(RES / "forecast_kaitz_paths.csv")
    ses = kp[kp.concept == "SES October earnings"]
    k26 = ses[(ses.path == "constant_bite") & (ses.year == 2026)].iloc[0]
    panels = [("employment_rate", "Employment rate (%)", 100), ("share_at_floor", "Paid the minimum (%)", 100),
              ("job_to_job_per_year", "Job-to-job moves (% of employed a year)", 100),
              ("log_gain_at_move", "Wage gain at a move (log points x 100)", 100)]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8), gridspec_kw={"hspace": 0.45, "wspace": 0.28, "top": 0.86})
    kx = g[("kaitz", "mean")]
    for ax, (col, title, sc) in zip(axes.flat, panels):
        ax.axvspan(k26.q5, k26.q95, color=SHADE, zorder=0)
        ax.fill_between(kx, g[(col, "min")] * sc, g[(col, "max")] * sc, color=GREY, alpha=0.45, lw=0)
        ax.plot(kx, g[(col, "mean")] * sc, color=INK, lw=2)
        ymin, ymax = ax.get_ylim()
        for name in ("agreement", "programme", "union_2027"):
            k29 = ses[(ses.path == name) & (ses.year == 2029)].iloc[0]
            ax.axvline(k29.q50, color=PATH_COLOURS[name], lw=1.8)
        ax.axvline(k26.q50, color=INK2, lw=1.2, ls="--")
        ax.set_xlim(0.45, 0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Kaitz index (minimum wage / median)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=INK2, lw=1.2, ls="--", label="2026 nowcast (median)"),
               Patch(color=SHADE, label="2026 nowcast, 5th-95th percentile")]
    handles += [Line2D([], [], color=PATH_COLOURS[nm], lw=1.8, label=f"2029: {PATH_SHORT[nm]}")
                for nm in ("agreement", "programme", "union_2027")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=3, fontsize=9)
    fig.suptitle("Where the model economy moves as the bite rises (steady states, eight seeds)", x=0.01, ha="left",
                 fontsize=13.5, fontweight="bold", y=1.0)
    note(fig, "Black: mean over seeds; grey band: range over seeds. Light shading: 5th-95th percentile of the 2026 nowcast of "
              "Portugal's SES Kaitz index (dashed line: median). Coloured lines: median 2029 index under each path. The model has no "
              "labour-supply response and no monopsony gain from a higher floor, so its employment loss is an upper bound in its own logic.",
         y=-0.03)
    save(fig, "fig3_model_along_bite.png")


def fig4_transparency() -> None:
    mr = pd.read_csv(RES / "model_reform.csv")
    cal = pd.read_csv(RES / "model_calibration.csv")
    cov = pd.read_csv(RES / "coverage.csv").set_index("geo")
    pi = float(cal.loc[cal.target == -0.02, "visibility"].iloc[0])
    shares = [(round(float(cov.loc["PT", "share_ge150"]), 3), "150+ from 2027"),
              (round(float(cov.loc["PT", "share_ge50"]), 3), "50+ from 2031"),
              (1.0, "every employer")]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), gridspec_kw={"wspace": 0.45})
    ax = axes[0]
    ax.axhspan(-3, -1, xmin=0, xmax=1, color=SHADE, zorder=0)
    ax.text(0.6, -1.15, "calibration range for every employer (-1% to -3%)", fontsize=8.6, color=INK2, va="top")
    for (share, lab), col in zip(shares, BLUES):
        d = mr[(mr.m == 0.65) & np.isclose(mr.cover_share, share) & np.isclose(mr.visibility, pi)].sort_values("year")
        ax.fill_between(d.year, 100 * (d.d_mean_wage_mean - d.d_mean_wage_std), 100 * (d.d_mean_wage_mean + d.d_mean_wage_std),
                        color=col, alpha=0.2, lw=0)
        ax.plot(d.year, 100 * d.d_mean_wage_mean, color=col, lw=2.2, marker="o", ms=4)
        last = d.iloc[-1]
        ax.annotate(f"firms {lab}, {100 * share:.0f}%" if share < 1 else "every employer", (last.year, 100 * last.d_mean_wage_mean), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.8, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlim(0.5, 10.2)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("years after the rule starts")
    ax.set_ylabel("mean wage against no rule (%)")
    ax.set_title("Average wage, by share of employees covered")

    ax = axes[1]
    qs = ["p10", "p25", "p50", "p75", "p90"]
    x = np.arange(len(qs))
    for (share, lab), col in zip(shares, BLUES):
        d = mr[(mr.m == 0.65) & np.isclose(mr.cover_share, share) & np.isclose(mr.visibility, pi) & (mr.year == 5)].iloc[0]
        ax.plot(x, [100 * d[f"d_{q}_mean"] for q in qs], color=col, lw=2.2, marker="o", ms=6,
                label=f"firms {lab}, {100 * share:.0f}% of employees" if share < 1 else "every employer")
    d = mr[(mr.m == 0.45) & np.isclose(mr.cover_share, 1.0) & np.isclose(mr.visibility, pi) & (mr.year == 5)].iloc[0]
    ax.plot(x, [100 * d[f"d_{q}_mean"] for q in qs], color=INK2, lw=1.6, ls="--", marker="s", ms=5,
            label="every employer, low floor (8% at the minimum)")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["P10", "P25", "median", "P75", "P90"])
    ax.set_ylabel("change against no rule, year 5 (%)")
    ax.set_title("Where it lands in the distribution, year 5")
    ax.legend(loc="lower left", fontsize=8.8)
    fig.suptitle("Pay transparency in the model: an equal-pay rule switched on for the largest firms at Portugal's floor",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=1.0)
    note(fig, f"Visibility (the probability a raise must be extended to peers) {pi:.3f}, calibrated so that covering every employer "
              "lowers the mean wage by 2% in years 3-5, the US estimate of Cullen and Pakzad-Hurson (2023). Ten seeds; band: +/- one "
              "standard deviation. Coverage shares: employees of the business economy by enterprise size, Eurostat 2023.", y=-0.1)
    save(fig, "fig4_transparency.png")


def fig5_eu_reference() -> None:
    eu = pd.read_csv(RES / "forecast_eu.csv").sort_values("kaitz_oecd", ascending=True)
    allc = pd.read_csv(RES / "eu_reference.csv").sort_values("kaitz_oecd")
    above = allc[allc.kaitz_oecd >= REFERENCE_KAITZ]
    order = list(eu.geo) + list(above.geo)
    y = {g: i for i, g in enumerate(order)}
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 7.2), sharey=True, gridspec_kw={"wspace": 0.08, "width_ratios": [1, 1.1]})
    ax = axes[0]
    ax.axvline(100 * REFERENCE_KAITZ, color=INK2, lw=1.2, ls="--")
    ax.text(100 * REFERENCE_KAITZ - 0.3, -0.6, "60% reference", fontsize=8.8, color=INK2, va="bottom", ha="right")
    for r in allc.itertuples():
        ax.plot([100 * r.kaitz_oecd, max(100 * r.kaitz_oecd, 100 * REFERENCE_KAITZ)], [y[r.geo]] * 2, color=GREY, lw=2)
        ax.plot(100 * r.kaitz_oecd, y[r.geo], "o", color=BLUE, ms=7, mec=SURFACE, mew=1.2, zorder=5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("minimum wage / median, full-time, 2025 (%)")
    ax.set_title("Bite today (OECD)")
    ax = axes[1]
    for r in eu.itertuples():
        col = GREY if r.beyond_sample else BLUE
        ax.plot([100 * r.dlog_p10_med_q5, 100 * r.dlog_p10_med_q95], [y[r.geo]] * 2, color=col, lw=2.2, solid_capstyle="round")
        ax.plot(100 * r.dlog_p10_med, y[r.geo], "o", ms=7, zorder=5, mfc=SURFACE if r.beyond_sample else col,
                mec=MUTED if r.beyond_sample else SURFACE, mew=1.6 if r.beyond_sample else 1.2)
    for g in above.geo:
        ax.text(0.5, y[g], "already above 60%", fontsize=8.6, color=MUTED, va="center")
    ax.axvline(0, color=INK2, lw=1)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("change in log P10/median if moved to 60% (log points x 100)")
    ax.set_title("What the bottom decile would do")
    fig.suptitle("EU: moving each statutory minimum wage to 60% of the median", x=0.01, ha="left", fontsize=13.5,
                 fontweight="bold", y=0.995)
    note(fig, "Right: bite-dependent elasticity from SES 2006-2022 (country and wave fixed effects), median unchanged; dot = point "
              "estimate, line = 90% bootstrap interval over countries. Hollow grey: the move would take the SES Kaitz index beyond "
              "the highest value in the sample (0.65).", y=-0.05)
    save(fig, "fig5_eu_reference.png")


def main() -> int:
    fig1_bite_paths()
    fig2_decision_2029()
    fig3_model_along_bite()
    fig4_transparency()
    fig5_eu_reference()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
