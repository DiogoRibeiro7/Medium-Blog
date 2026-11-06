"""Figures for the article, from ``data/`` and ``results/``.

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cure import fit, fit_lognormal, km, not_recovered
from episodes import panel
from recovery_models import DEPTH_LABELS, prep, with_bands

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal: shallow -> deep
INK, INK2, MUTED, GRID, SURFACE, SHADE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb", "#efeeea"
CREDIT = "Diogo Ribeiro"
STAGE_COLORS = {"pre-accession": BLUE, "customs union": ORANGE, "single market": AQUA, "euro": YELLOW}
BAND_NAMES = {"<0.30": "under 0.30", "0.30-0.50": "0.30 to 0.50", "0.50-0.80": "0.50 to 0.80", ">0.80": "over 0.80"}

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


def note(fig, text: str, y: float = 0.005) -> None:
    fig.text(0.01, y, f"{text}   |   {CREDIT}", ha="left", va="bottom", fontsize=8.2, color=MUTED, wrap=True)


def suptitle(fig, text: str, y: float = 1.02) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=y)


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pct(ax, axis="y") -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def fig1_four_busts() -> None:
    gva = panel("gva_a64_volume")
    ep = pd.read_csv(RES / "episodes_all.csv")
    ep = ep[(ep.measure == "gva") & (ep.version == "absolute")]
    cases = [
        ("DE", "C29", 2007, "Germany, motor vehicles"),
        ("PT", "C13-C15", 1997, "Portugal, textiles, clothing and leather"),
        ("ES", "F", 2007, "Spain, construction"),
        ("FI", "C26", 2008, "Finland, computers, electronics and optics"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharey=True)
    for ax, (geo, ind, pk, title) in zip(axes.flat, cases):
        x = gva[(geo, ind)].dropna()
        e = ep[(ep.geo == geo) & (ep.industry == ind) & (ep.peak_year == pk)].iloc[0]
        idx = 100 * x / x.loc[pk]
        ax.axhline(100, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.plot(idx.index, idx.values, color=BLUE, lw=2, zorder=3)
        tr = int(e.trough_year)
        ax.plot([pk], [100], "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=4)
        ax.plot([tr], [idx.loc[tr]], "o", ms=8, color=INK, mec=SURFACE, mew=2, zorder=4)
        ax.annotate(f"peak {pk}", (pk, 100), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=INK2)
        ax.annotate(f"trough {tr}\n{100 * (1 - np.exp(-e.depth)):.0f}% below peak", (tr, idx.loc[tr]), xytext=(6, -26), textcoords="offset points", ha="left", fontsize=9, color=INK2)
        if e.full_recovered:
            back = tr + int(e.t_full)
            ax.plot([back], [idx.loc[back]], "o", ms=8, color=AQUA, mec=SURFACE, mew=2, zorder=4)
            ax.annotate(f"back above peak {back}", (back, idx.loc[back]), xytext=(back + 2, 80), textcoords="data", ha="left", fontsize=9, color=INK2, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
        else:
            last = int(idx.index.max())
            ax.annotate(f"{last}: still {100 - idx.loc[last]:.0f}% below", (last, idx.loc[last]), xytext=(-4, 24), textcoords="offset points", ha="right", fontsize=9, color=INK2)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(1994, 2026)
        ax.set_ylim(0, 165)
    for ax in axes[:, 0]:
        ax.set_ylabel("value added, volume (peak year = 100)")
    suptitle(fig, "Four busts in value added: one fast rebound, three that have not come back")
    fig.tight_layout(h_pad=2.2)
    note(fig, "Source: Eurostat nama_10_a64, chain-linked volumes. A bust is a fall of at least 0.20 log points below the running peak.", y=-0.02)
    save(fig, "fig1_four_busts.png")


def km_curve(t: np.ndarray, e: np.ndarray, grid: np.ndarray, min_at_risk: int = 10) -> np.ndarray:
    s = km(t, e, grid)
    at_risk = np.array([(t >= g).sum() for g in grid])
    return np.where(at_risk >= min_at_risk, s, np.nan)


def fig2_still_below_by_depth() -> None:
    a = prep(pd.read_csv(RES / "episodes_all.csv"))
    grid = np.arange(0, 25.01, 0.05)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
    for ax, (m, title) in zip(axes, [("gva", "Value added"), ("emp", "Employment")]):
        d, _ = with_bands(a[(a.measure == m) & (a.version == "relative")])
        for lab, col in zip(DEPTH_LABELS, RAMP):
            s = d[d.band == lab]
            y = km_curve(s.t_full.to_numpy(float), s.full_recovered.to_numpy(), grid)
            ax.step(grid, y, where="post", color=col, lw=2, label=BAND_NAMES[lab])
            ok = np.flatnonzero(~np.isnan(y))
            ax.annotate(f"{BAND_NAMES[lab]} (n={len(s)})", (grid[ok[-1]], y[ok[-1]]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.8, color=INK2)
        ax.set_title(f"{title}, relative to other member states")
        ax.set_xlabel("years since the trough")
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 1.02)
        ax.axvline(10, color=MUTED, lw=1, ls=(0, (3, 3)))
        pct(ax)
    axes[0].set_ylabel("share of busts still below their pre-bust peak")
    axes[1].legend(title="depth of the bust (log points)", title_fontsize=9, loc="lower left")
    suptitle(fig, "The deeper the fall, the larger the share still below the old peak a decade later")
    fig.tight_layout()
    note(fig, "Kaplan-Meier estimates; single-market and euro stages, EU-27 industries, 1995-2025. Curves stop where fewer than 10 busts are still observed.", y=-0.04)
    save(fig, "fig2_still_below_by_depth.png")


def fig3_same_fit_different_futures() -> None:
    a = prep(pd.read_csv(RES / "episodes_all.csv"))
    grid = np.arange(0.5, 40.01, 0.25)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
    for ax, (v, title) in zip(axes, [("absolute", "Own level"), ("relative", "Relative to other member states")]):
        d, bands = with_bands(a[(a.measure == "gva") & (a.version == v)])
        c = fit(d, "t_full", "full_recovered", bands + ["tradable", "euro"], bands + ["tradable"])
        ln = fit_lognormal(d, "t_full", "full_recovered", bands + ["tradable", "euro"])
        s = d[d.band == ">0.80"]
        obs = km_curve(s.t_full.to_numpy(float), s.full_recovered.to_numpy(), grid)
        last_obs = grid[np.flatnonzero(~np.isnan(obs))[-1]]
        ax.axvspan(last_obs, 40, color=SHADE, zorder=0, lw=0)
        ax.text(last_obs + 0.6, 0.97, "beyond the data:\nfewer than 10 busts\nstill observed", va="top", fontsize=8.8, color=INK2)
        cure_c = np.array([not_recovered(c, s, t).mean() for t in grid])
        logn_c = np.array([not_recovered(ln, s, t).mean() for t in grid])
        ax.step(grid, obs, where="post", color=INK, lw=2, label="observed (Kaplan-Meier)", zorder=4)
        ax.plot(grid, cure_c, color=BLUE, lw=2, label="cure model: some never come back", zorder=3)
        ax.plot(grid, logn_c, color=ORANGE, lw=2, ls=(0, (5, 2)), label="lognormal: all come back, eventually", zorder=3)
        for y_, col, dy in ((cure_c[-1], BLUE, 0), (logn_c[-1], ORANGE, 0)):
            ax.annotate(f"{y_:.0%}", (40, y_), xytext=(4, dy), textcoords="offset points", va="center", fontsize=9.5, color=INK, fontweight="bold")
        ax.set_title(f"{title} (n={len(s)})")
        ax.set_xlabel("years since the trough")
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 1.02)
        pct(ax)
    axes[0].set_ylabel("share still below the pre-bust peak")
    axes[0].legend(loc="lower left")
    suptitle(fig, "Value-added busts deeper than 0.80 log points: two models that agree on the observed decade\nand part ways after it", y=1.06)
    fig.tight_layout()
    note(fig, "Both models use the same covariates (depth band, tradability, euro stage); curves average the model predictions over the deepest busts.", y=-0.04)
    save(fig, "fig3_same_fit_different_futures.png")


def fig4_tradable_gap() -> None:
    g = pd.read_csv(RES / "km_tradable_gap.csv")
    g["measure"] = g["sample"].str.split().str[1]
    g["version"] = g["sample"].str.split().str[2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, (m, title) in zip(axes, [("emp", "Employment"), ("gva", "Value added")]):
        rows = [("single market", "absolute"), ("single market", "relative"), ("euro", "absolute"), ("euro", "relative")]
        for i, (st, v) in enumerate(rows):
            r = g[(g.measure == m) & (g.version == v) & (g.stage == st)].iloc[0]
            y = len(rows) - 1 - i
            col = BLUE if v == "absolute" else ORANGE
            ax.plot([r.gap_10_lo, r.gap_10_hi], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot([r.gap_10], [y], "o", ms=9, color=col, mec=SURFACE, mew=2, label=("own level" if v == "absolute" else "relative to the same industry elsewhere") if st == "single market" else None)
            ax.annotate(f"{r.gap_10:+.0%}", (r.gap_10_hi, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK, bbox={"boxstyle": "square,pad=0.15", "fc": SURFACE, "ec": "none"})
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{st}, {'own level' if v == 'absolute' else 'relative'}" for st, v in reversed(rows)])
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(title)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-0.35, 0.6)
        pct(ax, "x")
    axes[1].tick_params(labelleft=False)
    fig.supxlabel("tradable minus non-tradable: difference in the share still below the peak 10 years after the trough", fontsize=10, color=INK2, y=0.06)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=9.5)
    suptitle(fig, "The traded-industry gap is large in raw employment and gone once shared trends are removed")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    note(fig, "Kaplan-Meier at 10 years; lines are 90% intervals from a bootstrap over countries. Tradable = agriculture, mining, manufacturing.", y=-0.1)
    save(fig, "fig4_tradable_gap.png")


def fig5_busts_by_stage() -> None:
    L = prep(pd.read_csv(RES / "long_episodes.csv"))
    L = L[L.stage != "outside"]
    rows = [("emp", "absolute", "Employment, own level"), ("emp", "relative", "Employment, relative"), ("gva", "absolute", "Value added, own level"), ("gva", "relative", "Value added, relative")]
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, (m, v, lab) in enumerate(rows):
        d = L[(L.measure == m) & (L.version == v)]
        y0 = len(rows) - 1 - i
        for st, col in STAGE_COLORS.items():
            s = d[d.stage == st]
            ax.scatter(s.trough_year + rng.uniform(-0.3, 0.3, len(s)), y0 + rng.uniform(-0.28, 0.28, len(s)), s=42, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=st if i == 0 else None)
        before = int((d.stage.isin(["pre-accession", "customs union"])).sum())
        ax.annotate(f"{before} of {len(d)} before the single market", (1958.6, y0 + 0.36), fontsize=8.6, color=INK2, va="bottom", zorder=5, bbox={"boxstyle": "square,pad=0.15", "fc": SURFACE, "ec": "none"})
    for yr, txt in ((1968, "customs union\ncomplete"), (1993, "single\nmarket"), (1999, "euro")):
        ax.axvline(yr, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.text(yr + 0.4, -0.75, txt, fontsize=8.4, color=INK2, va="bottom")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[2] for r in reversed(rows)])
    ax.set_ylim(-0.8, len(rows) - 0.35)
    ax.set_xlim(1958, 2026)
    ax.set_xlabel("year of the trough")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID)
    ax.legend(title="stage of the country at the trough", title_fontsize=9, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    suptitle(fig, "Sixty-five years of sector data, and most large busts still come after 1993", y=1.08)
    fig.tight_layout()
    note(fig, "AMECO, five sectors, EU member states and the UK until 2020; busts of at least 0.15 log points with an observed peak, transition-era episodes excluded.", y=-0.06)
    save(fig, "fig5_busts_by_stage.png")


def main() -> None:
    fig1_four_busts()
    fig2_still_below_by_depth()
    fig3_same_fit_different_futures()
    fig4_tradable_gap()
    fig5_busts_by_stage()


if __name__ == "__main__":
    main()
