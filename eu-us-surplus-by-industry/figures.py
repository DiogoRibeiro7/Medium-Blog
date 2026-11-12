"""Draw the four figures for the article from ``results/``.

    python figures.py

Static PNGs for Medium (light surface), one fixed categorical order, legends on dot plots, a diverging
scale with a neutral midpoint for the country matrix.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SOURCE = "Source: OECD STAN Database 2025 edition and national accounts (GDP PPPs), snapshot 12 Nov 2026; author's calculations."
COUNTRY = {"AUT": "Austria", "BEL": "Belgium", "DNK": "Denmark", "FIN": "Finland", "FRA": "France", "DEU": "Germany",
           "NLD": "Netherlands", "SWE": "Sweden"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": BASE, "axes.linewidth": 1.0, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def tidy(ax, grid_axis: str = "x") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def finish(fig, name: str, title: str, subtitle: str, source: str = SOURCE) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.995)
    fig.text(0.01, 0.995 - 0.30 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    fig.text(0.01, 0.005, source, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def fig1() -> None:
    t = pd.read_csv(RES / "contributions_2021.csv", index_col=0)
    g = pd.read_csv(RES / "gross_contributions_2021.csv", index_col=0)["gross_contribution"]
    t = t.drop(index=[s for s in ("T",) if s in t.index])
    t["gross"] = g.reindex(t.index)
    t = t.sort_values("contribution", ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(10, 6.6))
    fig.subplots_adjust(left=0.3, right=0.96, top=0.86, bottom=0.12)
    y = np.arange(len(t))
    for i, r in t.iterrows():
        ax.hlines(i, min(r.contribution, r.gross), max(r.contribution, r.gross), color=BASE, linewidth=2)
    ax.scatter(t.contribution, y, s=58, color=SLOTS[0], edgecolor=SURFACE, linewidth=2, zorder=3,
               label="Net of depreciation")
    ax.scatter(t.gross, y, s=60, facecolor="none", edgecolor=SLOTS[1], linewidth=2, zorder=4,
               label="Gross of depreciation")
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y, t.name, fontsize=9)
    ax.set_xlabel("Contribution to the gap in operating surplus per hour, North-West minus US, USD at GDP PPP")
    ax.legend(loc="lower left", fontsize=8.5)
    tidy(ax, "x")
    finish(fig, "fig1_gap_by_industry_2021", "Where North-West Europe's lower operating surplus per hour sits, 2021",
           "Net gap 8.89 USD per hour; real estate is the largest net contributor and shrinks once depreciation is left in")


def fig2() -> None:
    t = pd.read_csv(RES / "contributions_2021.csv", index_col=0)
    t = t.drop(index=["O", "P", "Q", "T"]).sort_values("profit_share_us").reset_index()
    fig, ax = plt.subplots(figsize=(10, 6.0))
    fig.subplots_adjust(left=0.3, right=0.96, top=0.86, bottom=0.12)
    y = np.arange(len(t))
    for i, r in t.iterrows():
        ax.hlines(i, 100 * min(r.profit_share_nw, r.profit_share_us), 100 * max(r.profit_share_nw, r.profit_share_us),
                  color=BASE, linewidth=2)
    ax.scatter(100 * t.profit_share_nw, y, s=58, color=SLOTS[0], edgecolor=SURFACE, linewidth=2, zorder=3,
               label="North-West Europe (8 countries)")
    ax.scatter(100 * t.profit_share_us, y, s=60, facecolor="none", edgecolor=SLOTS[1], linewidth=2, zorder=4,
               label="United States")
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y, t.name, fontsize=9)
    ax.set_xlabel("Net operating surplus after imputed self-employed labour income, % of value added at factor cost")
    ax.legend(loc="lower right", fontsize=8.5)
    tidy(ax, "x")
    finish(fig, "fig2_profit_share_by_industry_2021", "Net profit share by industry, 2021",
           "No currency conversion needed; the North-West share is lower in 13 of 15 market industries")


def fig3() -> None:
    ct = pd.read_csv(RES / "countries_2021.csv", index_col=0).sort_values("gap")
    cols = [("L", "Real estate"), ("K", "Finance"), ("J_M_N", "Info, professional,\nadmin services"),
            ("C", "Manufacturing"), ("other", "All other"), ("gap", "Total gap")]
    M = ct[[c for c, _ in cols]].values
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f2f1ec", "#eb6834"])
    norm = TwoSlopeNorm(vmin=-7, vcenter=0, vmax=7)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.82, bottom=0.12)
    ax.imshow(np.clip(M, -7, 7), cmap=cmap, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color=INK, fontweight="bold" if j == len(cols) - 1 else "normal")
    ax.set_xticks(range(len(cols)), [lab for _, lab in cols], fontsize=9)
    ax.set_yticks(range(len(ct)), [COUNTRY[c] for c in ct.index], fontsize=9)
    ax.xaxis.tick_top()
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ct), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    finish(fig, "fig3_countries_by_industry_2021",
           "Each North-West country against the US: contributions to the gap, USD per hour, 2021",
           "Blue: the country's operating surplus per hour is lower than the US in that block; orange: higher")


def fig4() -> None:
    s = pd.read_csv(RES / "subindustries_2021.csv")
    s = s[s.industry != s.parent]
    names = {"J58T60": "Publishing and broadcasting", "J61": "Telecommunications", "J62_63": "IT and information services",
             "K64": "Financial services (banks)", "K65": "Insurance and pensions", "K66": "Auxiliary financial services",
             "M69T71": "Legal, accounting, consulting, engineering", "M72": "Scientific R&D",
             "M73T75": "Advertising, other professional", "C21": "Pharmaceuticals", "C26": "Computers and electronics",
             "C29": "Motor vehicles"}
    oo = pd.read_csv(RES / "owner_occupied_housing_2021.csv", index_col=0)
    nw = [c for c in oo.index if c != "USA"]
    hn = oo.loc[nw, "hours"].sum()
    extra = pd.DataFrame([{"parent": "L", "industry": "L68A", "nw": oo.loc[nw, "l68a_gos"].sum() / hn,
                           "us": oo.loc["USA", "l68a_gos"] / oo.loc["USA", "hours"]}])
    extra["gap"] = extra.nw - extra.us
    names["L68A"] = "Owner-occupied housing (gross)"
    s = pd.concat([s, extra], ignore_index=True).iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6.4))
    fig.subplots_adjust(left=0.36, right=0.96, top=0.86, bottom=0.12)
    y = np.arange(len(s))
    for i, r in s.iterrows():
        ax.hlines(i, min(r.nw, r.us), max(r.nw, r.us), color=BASE, linewidth=2)
        ax.text(max(r.nw, r.us) + 0.08, i, f"{r.gap:+.2f}", va="center", fontsize=8, color=INK2)
    ax.scatter(s.nw, y, s=58, color=SLOTS[0], edgecolor=SURFACE, linewidth=2, zorder=3, label="North-West Europe")
    ax.scatter(s.us, y, s=60, facecolor="none", edgecolor=SLOTS[1], linewidth=2, zorder=4, label="United States")
    ax.set_yticks(y, [names[k] for k in s.industry], fontsize=9)
    for sep in (0.5, 3.5, 6.5, 9.5):
        ax.axhline(sep, color=GRID, linewidth=0.8)
    ax.set_xlabel("USD per hour worked in the whole economy, at GDP PPP; label: North-West minus US")
    ax.set_xlim(0, 5.6)
    ax.legend(loc="upper right", fontsize=8.5)
    tidy(ax, "x")
    finish(fig, "fig4_inside_the_industries_2021", "Inside the industries: which parts carry the gap, 2021",
           "Net operating surplus and mixed income without the self-employment imputation; owner-occupied housing gross")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4):
        f()
    print(f"figures written to {FIG}")
