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

# reference palette (light mode); blue / orange / aqua validated for all-pairs CVD separation
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
})


def save(fig, name: str, source: str) -> None:
    fig.text(0.01, 0.005, source, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_forbes_quality() -> None:
    q = pd.read_csv(RES / "forbes_quality.csv")
    q = q[pd.to_numeric(q["status"], errors="coerce") == 200]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax.bar(q.year, q.n_persons, color=np.where(q.year >= 2011, BLUE, MUTED), width=0.75)
    pub = q.dropna(subset=["published_list_size"])
    ax.scatter(pub.year, pub.published_list_size, marker="_", s=260, color=ORANGE, linewidths=3, zorder=3)
    for _, r in pub.iterrows():
        ax.annotate(f"published: {int(r.published_list_size):,}\nAPI: {int(r.n_persons):,}",
                    (r.year, r.published_list_size), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8.5, color=INK2)
    ax.set_title("People returned by the Forbes API for each annual list")
    ax.set_ylabel("Billionaires")
    ax.set_ylim(0, 3800)
    bx.plot(q.year, q.share_selfmade_flag, color=MUTED, lw=2, label="share with the self-made flag filled")
    bx.plot(q.year, q.share_selfmade_true, color=BLUE, lw=2, label="share flagged self-made")
    bx.axvspan(2010.5, 2026.5, color=GRID, alpha=0.5, lw=0, zorder=0)
    bx.text(2011, 0.08, "used in this article: 2011-2023 lists\n(2024-2026 have no growth data yet)", fontsize=8.5, color=INK2)
    bx.set_ylim(0, 1.05)
    bx.set_title("The self-made flag is informative only from 2011")
    bx.legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0.0, 0.93))
    fig.tight_layout()
    save(fig, "fig1_forbes_quality",
         "Data: Forbes person-list API, annual lists 2002-2026, fetched 12 Nov 2026; published counts from contemporary press reports.")


def fig2_landscape() -> None:
    p = pd.read_csv(RES / "panel.csv")
    wb = p[p.n > 0]
    cm = wb.groupby("iso3").agg(years=("year", "size"), inh=("inherited_m", "sum"), sm=("self_made_m", "sum"),
                                tot=("total_pct", "mean"))
    cm = cm[cm.years >= 7]
    cm["share"] = cm.inh / (cm.inh + cm.sm)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.scatter(cm.tot, cm.share, s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xscale("log")
    label = {"USA": (6, -4), "CHN": (6, 4), "IND": (6, -4), "DEU": (6, 2), "FRA": (6, 2), "SWE": (6, 0),
             "RUS": (6, 4), "HKG": (-8, 6), "JPN": (6, 0), "GBR": (6, 0), "KOR": (6, 0), "ITA": (6, -10),
             "BRA": (6, 0), "CHE": (6, 4), "CHL": (6, 0), "ISR": (6, 0), "ESP": (6, 0), "MEX": (6, 4)}
    for iso, off in label.items():
        if iso in cm.index:
            r = cm.loc[iso]
            ax.annotate(iso, (r.tot, r.share), textcoords="offset points", xytext=off, fontsize=8.5,
                        color=INK2, ha="right" if off[0] < 0 else "left")
    ax.set_xlabel("Billionaire wealth, % of GDP (country mean 2011-2023, log scale)")
    ax.set_ylabel("Share of that wealth Forbes classes as inherited")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title(f"Where top fortunes are inherited: {len(cm)} countries with billionaires in at least 7 of 13 years")
    fig.tight_layout()
    save(fig, "fig2_landscape", "Data: Forbes annual lists 2011-2023 (self-made flag), WDI GDP in current dollars.")


def fig3_choice_strips() -> None:
    s = pd.read_csv(RES / "specs.csv")
    s["t"] = s["diff"] / s["se_diff"]
    groups = [
        ("estimator", {"fe": "country + year fixed effects", "pooled": "year effects only"}),
        ("outcome", {"gdppc": "GDP per capita growth", "tfp": "TFP growth"}),
        ("horizon", {1: "1-year horizon", 3: "3-year horizon", 5: "5-year horizon"}),
        ("transform", {"log1p": "log(1 + % of GDP)", "raw": "% of GDP"}),
        ("sample", {"all": "all countries", "billionaire_countries": "billionaire countries",
                    "no_offshore": "no financial centres"}),
        ("controls", {"none": "no control", "income": "initial income control"}),
    ]
    rows = [(col, val, lab) for col, m in groups for val, lab in m.items()]
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    rng = np.random.default_rng(0)
    ax.axvspan(-1.96, 1.96, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.axvline(0, color=INK2, lw=0.8)
    yticks, ylabels = [], []
    y = 0
    prev = None
    for col, val, lab in rows:
        if prev is not None and col != prev:
            y -= 0.6
        sub = s[s[col] == val]
        jitter = rng.uniform(-0.18, 0.18, len(sub))
        c = np.where(sub["t"] < -1.96, BLUE, MUTED)
        ax.scatter(sub["t"], y + jitter, s=12, c=c, linewidths=0, zorder=3)
        ax.plot([sub["t"].median()] * 2, [y - 0.3, y + 0.3], color=INK, lw=2, zorder=4)
        yticks.append(y)
        ylabels.append(lab)
        prev = col
        y -= 1
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("t-statistic of (inherited − self-made) coefficient on subsequent growth")
    n_neg = int((s["t"] < -1.96).sum())
    ax.set_title(f"144 fits: {int((s['diff'] < 0).sum())} negative, {n_neg} beyond −1.96 (blue), none beyond +1.96; "
                 "black bars are medians")
    fig.tight_layout()
    save(fig, "fig3_choice_strips", "Each fit appears once in every group. Grey band: |t| < 1.96. Source: analysis.py.")


def fig4_placebo() -> None:
    p = pd.read_csv(RES / "placebo.csv")
    p = p[p["transform"] == "log1p"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, var, title in ((axes[0], "gdppc", "GDP per capita growth"), (axes[1], "tfp", "TFP growth")):
        for smp, c, dx, lab in (("all", MUTED, -0.18, "all countries"), ("billionaire_countries", BLUE, 0.18,
                                                                         "billionaire countries")):
            q = p[(p.outcome == var) & (p["sample"] == smp)].sort_values("offset")
            x = np.arange(len(q)) + dx
            ax.vlines(x, q.lo, q.hi, color=c, lw=2)
            ax.scatter(x, q["diff"], color=c, s=42, zorder=3, edgecolor=SURFACE, linewidth=1.5, label=lab)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(2.5, color=GRID, lw=1.5)
        ax.set_xticks(range(6))
        ax.set_xticklabels(["t−5→t", "t−3→t", "t−1→t", "t→t+1", "t→t+3", "t→t+5"], fontsize=9)
        ax.set_title(title)
        ax.text(1.0, 0.98, "past growth (placebo)", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8.5, color=INK2)
        ax.text(4.0, 0.98, "future growth", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8.5, color=INK2)
    axes[0].set_ylabel("Inherited − self-made coefficient\n(pp per log point, 95% CI)")
    axes[1].legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig4_placebo",
         "Country and year fixed effects; billionaire wealth as log(1 + % of GDP); income at the start of each window. "
         "Source: analysis.py.")


def fig5_wid_check() -> None:
    w = pd.read_csv(RES / "wid_check.csv")
    lab = {"gdppc": "GDP pc", "tfp": "TFP"}
    est = {"fe": "fixed effects", "pooled": "year effects only"}
    w["label"] = [f"{lab[o]}, {h}-year, {est[e]}" for o, h, e in zip(w.outcome, w.horizon, w.estimator)]
    w = w.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(w))
    c = np.where(w.lo > 0, ORANGE, np.where(w.hi < 0, BLUE, MUTED))
    ax.hlines(y, w.lo, w.hi, color=c, lw=2)
    ax.scatter(w.interaction, y, color=c, s=42, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(w.label, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Interaction: top-1% wealth share × inherited share (pp of growth)")
    ax.text(0.99, 0.02, "Conjecture 1 predicts negative values", transform=ax.transAxes, ha="right",
            fontsize=8.5, color=INK2)
    ax.set_title(f"The literal test: {int(w.n_countries.max())} countries with source-based WID wealth shares")
    fig.tight_layout()
    save(fig, "fig5_wid_check", "Data: WID shweal992j p99p100 with data_quality > 0; Forbes 2011-2023; PWT 11.0. "
         "95% intervals clustered by country. Source: analysis.py.")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig1_forbes_quality()
    fig2_landscape()
    fig3_choice_strips()
    fig4_placebo()
    fig5_wid_check()


if __name__ == "__main__":
    main()
