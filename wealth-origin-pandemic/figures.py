"""Figures for the article. Reads results/, writes figures/fig*.png.

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

# reference palette (light mode); blue / orange validated for CVD separation on this surface
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
})

SOURCE = ("Data: Forbes person-list API (2011-2023 lists), Penn World Table 11.0, OxCGRT, ILOSTAT, World Bank WDI.")


def save(fig, name: str, source: str = SOURCE) -> None:
    fig.text(0.01, 0.005, source, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_exposure_vs_composition() -> None:
    c = pd.read_csv(RES / "country_means.csv").set_index("iso3")
    corr = pd.read_csv(RES / "exposure_correlations.csv")
    rho = corr[corr["variable"] == "inh_share"].set_index("exposure")["spearman"]
    panels = [("stringency", "Lockdown stringency, mean 2020-2021 (index)", False),
              ("deaths_pm", "Confirmed COVID deaths per million, end-2021", False),
              ("tourism_gdp", "Travel credits, % of GDP, mean 2015-2019 (log scale)", True),
              ("contact_emp", "Contact-intensive employment, % of total, 2019", False)]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharey=True)
    for ax, (k, lab, logx) in zip(axes.flat, panels):
        d = c[[k, "inh_share"]].dropna()
        ax.scatter(d[k], d["inh_share"], s=34, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        for iso in d[k].nlargest(2).index:
            ax.annotate(iso, (d.loc[iso, k], d.loc[iso, "inh_share"]), textcoords="offset points",
                        xytext=(-4, 6), fontsize=8, color=INK2, ha="right")
        ax.set_xlabel(lab, fontsize=9.5)
        r_ = 0.0 if abs(rho[k]) < 0.005 else rho[k]
        ax.set_title(f"Spearman correlation {r_:+.2f}" if r_ else "Spearman correlation 0.00", fontsize=10.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    for ax in axes[:, 0]:
        ax.set_ylabel("Inherited share of billionaire wealth\n(country mean, 2011-2023 lists)")
    fig.suptitle("None of the four exposure measures lines up with how much billionaire wealth is inherited",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig1_exposure_vs_composition", SOURCE + " 63 countries with billionaires in at least 7 of 13 lists.")


def fig2_coefficients() -> None:
    f = pd.read_csv(RES / "fits.csv")
    a = pd.read_csv(RES / "average_fits.csv")
    sel = (f["outcome"] == "tfp") & (f["sample"] == "billionaire_countries") & (f["estimator"] == "fe")
    asel = (a["outcome"] == "tfp") & (a["sample"] == "billionaire_countries") & (a["estimator"] == "fe")
    rows = [("No exposure controls", "b0_all_four", "se0_all_four", None),
            ("+ lockdown stringency", "b_stringency", "se_stringency", None),
            ("+ COVID deaths per million", "b_deaths", "se_deaths", None),
            ("+ travel credits, % of GDP", "b_tourism", "se_tourism", None),
            ("+ contact-intensive employment", "b_contact", "se_contact", None),
            ("+ all four", "b_all_four", "se_all_four", None),
            ("Two-list average, no exposure controls", "avg_base", "se_avg_base", "avg"),
            ("Two-list average, + all four", "avg", "se_avg", "avg")]
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    ylab = []
    for i, (lab, bcol, scol, src) in enumerate(rows):
        y = len(rows) - 1 - i
        ylab.append((y, lab))
        for h, col, dy in ((3, BLUE, 0.14), (5, ORANGE, -0.14)):
            r = (a[asel] if src else f[sel])
            r = r[r["horizon"] == h].iloc[0]
            b, s = r[bcol], r[scol]
            ax.hlines(y + dy, b - 1.96 * s, b + 1.96 * s, color=col, lw=2, alpha=0.6)
            ax.scatter([b], [y + dy], s=50, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=f"{h}-year TFP growth" if i == 0 else None)
    ax.axhline(1.5, color=GRID, lw=1.2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks([y for y, _ in ylab], [lab for _, lab in ylab])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Inherited minus self-made coefficient (pp of annual TFP growth per log point), 95% interval")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Pandemic-exposure controls leave the estimate where it was;\n"
                 "averaging two Forbes lists makes it larger")
    fig.tight_layout()
    save(fig, "fig2_coefficients", SOURCE + " 56 countries, country fixed effects, sector shares controlled.")


def fig3_list_years() -> None:
    d = pd.read_csv(RES / "measures_leave_one_year_out.csv")
    d = d[(d["horizon"] == 3) & (d["dropped_year"] > 0)]
    titles = {"wealth": "Wealth on each list", "headcount": "Headcount per 10 million people",
              "two-list average": "Wealth averaged over two lists"}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharey=True)
    for ax, (m, t) in zip(axes, titles.items()):
        sub = d[d["measure"] == m]
        col = np.where(sub["dropped_year"].isin([2019, 2020]), ORANGE, BLUE)
        ax.vlines(sub["dropped_year"], sub["lo"], sub["hi"], color=col, lw=2, alpha=0.55)
        ax.scatter(sub["dropped_year"], sub["b"], s=40, color=col, edgecolor=SURFACE, linewidth=1.3, zorder=3)
        ax.axhline(sub["full_b"].iloc[0], color=INK2, lw=1, ls=(0, (4, 3)))
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"{t}\n(all lists: {sub['full_b'].iloc[0]:+.2f})", fontsize=10.5)
        ax.set_xticks(range(2011, 2021, 3))
        ax.set_xlabel("List year left out")
    axes[0].set_ylabel("Inherited minus self-made\n(pp of three-year TFP growth per log point)")
    fig.suptitle("With one list per year, the estimate leans on 2019-2020 (orange); averaged over two lists, it does not",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_list_years", SOURCE + " Billionaire countries, country fixed effects, sector shares and all four exposures controlled.")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig1_exposure_vs_composition()
    fig2_coefficients()
    fig3_list_years()


if __name__ == "__main__":
    main()
