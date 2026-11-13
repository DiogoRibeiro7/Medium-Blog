"""Figures for the housing-losses-inflation article.

    python figures.py     # after analysis.py; writes figures/fig1_*.png ... fig5_*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as A  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

# Palette: first three categorical slots of a validated colour-blind-safe set (all pairs pass).
NOMINAL = "#2a78d6"     # blue   - nominal price / visible part of a loss
HIDDEN = "#eb6834"      # orange - real price / the part inflation produced
TOTAL = "#1baf7a"       # aqua   - real total return with rents
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SOURCE_JST = "Data: Jordà-Schularick-Taylor Macrohistory Database, R6. Chart: Diogo Ribeiro."
SOURCE_OECD = "Data: OECD analytical house price indicators. Chart: Diogo Ribeiro."

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False,
})


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _source(fig, text):
    fig.text(0.01, 0.01, text, fontsize=8.5, color=MUTED, ha="left", va="bottom")


def _titles(fig, title, subtitle=None):
    fig.text(0.01, 0.975, title, fontsize=14, fontweight="bold", color=INK, ha="left", va="top")
    if subtitle:
        fig.text(0.01, 0.925, subtitle, fontsize=9.5, color=INK2, ha="left", va="top")


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- fig 1
def fig1_three_lenses():
    p = A.load_jst()
    E = pd.read_csv(RES / "episodes_jst_returns_series.csv")
    countries = [("FRA", "France"), ("NOR", "Norway"), ("USA", "United States"),
                 ("NLD", "Netherlands")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharey=True)
    for ax, (iso, name) in zip(axes.ravel(), countries):
        g = p[p.iso == iso].set_index("year")
        ok = g[["dlcg", "dltr", "dlc"]].notna().all(axis=1)
        seg = A.segments(g.dlcg.where(ok))
        s = max(seg, key=len).index
        d = g.loc[s, ["dlcg", "dltr", "dlc"]].copy()
        d.iloc[0] = 0.0
        N = np.exp(d.dlcg.cumsum())
        R = np.exp((d.dlcg - d.dlc).cumsum())
        T = np.exp((d.dltr - d.dlc).cumsum())
        for _, e in E[(E.iso == iso) & (E.hidden_share > 0.5)].iterrows():
            ax.axvspan(e.peak, e.trough, color=HIDDEN, alpha=0.10, lw=0)
        ax.plot(N.index, N, color=NOMINAL)
        ax.plot(R.index, R, color=HIDDEN)
        ax.plot(T.index, T, color=TOTAL)
        ax.set_yscale("log")
        ax.set_title(f"{name}, {s[0]}–{s[-1]}", loc="left")
        _grid(ax)
        for ser, col, lab in [(N, NOMINAL, "nominal price"), (R, HIDDEN, "real price"),
                              (T, TOTAL, "real total return")]:
            ax.annotate(lab, (ser.index[-1], ser.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8.5, color=INK2, va="center")
        ax.set_xlim(s[0], s[-1] + 22)
        ax.set_xticks([x for x in range(1880, 2021, 20) if x >= s[0]])
    for ax in axes[:, 0]:
        ax.set_ylabel(f"index, first year = 1 (log scale)")
    _titles(fig, "The same houses through three lenses",
            "Shaded: real price falls of 10% or more in which most of the fall came through "
            "inflation, not through a lower nominal price")
    handles = [plt.Line2D([], [], color=c, lw=2) for c in (NOMINAL, HIDDEN, TOTAL)]
    fig.legend(handles, ["nominal price", "real price (CPI-deflated)",
                         "real total return (price + net rent)"], loc="upper left",
               ncol=3, fontsize=9, bbox_to_anchor=(0.005, 0.895))
    _source(fig, SOURCE_JST)
    fig.tight_layout(rect=(0, 0.03, 1, 0.85))
    _save(fig, "fig1_three_lenses")


# ----------------------------------------------------------------------------- fig 2
def fig2_holding_losses():
    H = pd.read_csv(RES / "holding_periods.csv")
    d = H[(H["sample"] == "JST returns set") & (H.h == 10) & (H.group == "inflation")]
    d = d.set_index("value").loc[A.INFL_LABELS]
    x = np.arange(len(d))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.6))
    series = [("p_nominal_loss", None, None, NOMINAL, "nominal price lower"),
              ("p_real_loss", "p_real_loss_lo", "p_real_loss_hi", HIDDEN, "real price lower"),
              ("p_total_return_loss", "p_tr_loss_lo", "p_tr_loss_hi", TOTAL,
               "real total return negative")]
    for i, (col, lo, hi, c, lab) in enumerate(series):
        xs = x + (i - 1) * (w + 0.02)
        ax.bar(xs, 100 * d[col], width=w, color=c, label=lab, zorder=2)
        if lo:
            ax.errorbar(xs, 100 * d[col], yerr=[100 * (d[col] - d[lo]), 100 * (d[hi] - d[col])],
                        fmt="none", ecolor=INK2, elinewidth=1, capsize=0, zorder=3)
        tops = d[hi] if hi else d[col]
        for xx, v, top in zip(xs, d[col], tops):
            ax.text(xx, 100 * top + 1.2, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=8.5,
                    color=INK2)
    ax.set_xticks(x, [f"{k}\n({n:,} purchases)" for k, n in zip(d.index, d.windows)])
    ax.set_xlabel("average consumer-price inflation over the ten years")
    ax.set_ylabel("share of ten-year holdings (%)")
    ax.set_ylim(0, 100)
    _grid(ax)
    ax.legend(loc="upper left", fontsize=9.5)
    _titles(fig, "After ten years, how often was the owner behind?",
            "16 countries, every purchase year 1870–2010. Whiskers: 95% interval with countries "
            "resampled; numbers are the shares.")
    _source(fig, SOURCE_JST)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    _save(fig, "fig2_holding_losses")


# ----------------------------------------------------------------------------- fig 3
def fig3_hidden_vs_inflation():
    EJ = pd.read_csv(RES / "episodes_jst.csv")
    EQ = pd.read_csv(RES / "episodes_oecd.csv")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.scatter(100 * EJ.infl_annual, 100 * EJ.hidden_share, s=46, marker="o", color=NOMINAL,
               edgecolor=SURFACE, linewidth=1.5, label="annual data, 18 countries, 1870–2020", zorder=3)
    ax.scatter(100 * EQ.infl_annual, 100 * EQ.hidden_share, s=52, marker="^", color=HIDDEN,
               edgecolor=SURFACE, linewidth=1.5, label="quarterly data, 35 OECD countries, 1970–2026",
               zorder=3)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(-8, 400)
    ax.set_xticks([-5, 0, 2, 5, 10, 20, 50, 100, 300], ["−5", "0", "2", "5", "10", "20", "50",
                                                         "100", "300"])
    ax.set_ylim(-5, 108)
    ax.set_xlabel("consumer-price inflation during the fall, % a year (linear below 10, log above)")
    ax.set_ylabel("share of the real fall that was hidden (%)")
    _grid(ax, "both")
    notes = [
        (EJ, "JPN", 1991, "Japan 1991–2009", (-7.2, 17)),
        (EJ, "IRL", 2006, "Ireland 2006–12", (-7.2, 31)),
        (EJ, "USA", 2006, "US 2006–12", (-7.2, 50)),
        (EJ, "GBR", 1973, "UK 1973–77", (24, 87)),
        (EJ, "DEU", 1913, "Germany 1913–22", (110, 87)),
    ]
    for E, iso, peak, text, pos in notes:
        r = E[(E.iso == iso) & (E.peak == peak)]
        if len(r):
            r = r.iloc[0]
            ax.annotate(text, (100 * r.infl_annual, 100 * r.hidden_share), xytext=pos,
                        textcoords="data", fontsize=8.5, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=4))
    ax.legend(loc="lower right", fontsize=9)
    _titles(fig, "The more inflation during a fall, the less of it shows in the price",
            "Each mark is a real house-price fall of 10% or more, peak to trough. Hidden = the "
            "part not matched by a lower nominal price.")
    _source(fig, "Data: Jordà-Schularick-Taylor Macrohistory Database R6; OECD analytical house "
            "price indicators. Chart: Diogo Ribeiro.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    _save(fig, "fig3_hidden_vs_inflation")


# ----------------------------------------------------------------------------- fig 4
def fig4_two_busts():
    EQ = pd.read_csv(RES / "episodes_oecd.csv")
    t05, t09, t20 = 2005 * 4, 2009 * 4 + 3, 2020 * 4
    panels = [("Real peak in 2005–2009", EQ[(EQ.peak >= t05) & (EQ.peak <= t09)]),
              ("Real peak in 2020 or later", EQ[EQ.peak >= t20])]
    n = max(len(b) for _, b in panels)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * n + 2.4), sharex=True)
    for ax, (title, B) in zip(axes, panels):
        B = B.sort_values("real_fall")
        y = np.arange(len(B))
        vis = 100 * B.real_fall * B.visible_share
        hid = 100 * B.real_fall * B.hidden_share
        ax.barh(y, vis, color=NOMINAL, height=0.68, label="nominal price fell")
        ax.barh(y, hid, left=vis + 0.25, color=HIDDEN, height=0.68,
                label="inflation outran the price")
        ax.set_yticks(y, [f"{i}  {pl[:4]}" for i, pl in zip(B.iso, B.peak_label)], fontsize=8.5)
        ax.tick_params(axis="y", length=0)
        med_inf = 100 * B.infl_annual.median()
        ax.set_title(f"{title}\n{len(B)} falls · inflation {med_inf:.1f}% a year · "
                     f"{100 * B.hidden_share.median():.0f}% hidden",
                     loc="left", fontsize=10.5)
        _grid(ax, "x")
        ax.set_xlabel("real fall from peak to trough (%), split by where it came from")
    axes[0].legend(loc="lower right", fontsize=9)
    _titles(fig, "Two busts, two kinds of loss",
            "OECD members, real house-price falls of 10% or more, peak to trough. Panel "
            "titles give medians; bars split each fall in proportion to its log contributions.")
    _source(fig, SOURCE_OECD + " Quarterly, seasonally adjusted, deflated by the "
            "private consumption deflator.")
    fig.tight_layout(rect=(0, 0.035, 1, 0.91))
    _save(fig, "fig4_two_busts")


# ----------------------------------------------------------------------------- fig 5
def fig5_portugal():
    d = pd.read_csv(RES / "portugal_annual.csv").set_index("year")
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for a, b in [(1973, 1979), (2007, 2013)]:
        ax.axvspan(a, b, color=HIDDEN, alpha=0.10, lw=0)
    ax.axvline(1988.5, color=AXIS, lw=1, ls=(0, (3, 3)))
    ax.text(1989.2, 250, "national price index\nfrom 1989", fontsize=8.5, color=MUTED, va="center")
    for col, c, lab in [("N", NOMINAL, "nominal price"), ("R", HIDDEN, "real price"),
                        ("T_J", TOTAL, "real total return")]:
        ax.plot(d.index, d[col], color=c)
        ax.annotate(lab, (d.index[-1], d[col].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=9, color=INK2, va="center")
    ax.set_yscale("log")
    ax.set_xlim(1948, 2036)
    ax.set_xticks(range(1950, 2021, 10))
    ax.set_ylabel("index, 1948 = 1 (log scale)")
    _grid(ax)
    ax.set_ylim(0.5, 600)
    ax.text(1973.3, 0.6, "1973–79: real −54%,\nnominal +90%", fontsize=8.5, color=INK2)
    ax.text(2004.0, 0.6, "2007–13: real −26%,\nnominal −18%", fontsize=8.5, color=INK2)
    _titles(fig, "Portugal: one fall hidden by inflation, one in plain sight",
            "Real total return uses JST's net rental return to 2020, rolled forward with the OECD "
            "price-to-rent ratio.")
    _source(fig, "Data: Jordà-Schularick-Taylor Macrohistory Database R6 (to 1988 prices, to 2020 "
            "CPI and rents); OECD (after). Chart: Diogo Ribeiro.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    _save(fig, "fig5_portugal")


if __name__ == "__main__":
    fig1_three_lenses()
    fig2_holding_losses()
    fig3_hidden_vs_inflation()
    fig4_two_busts()
    fig5_portugal()
    print("figures written to", FIG)
