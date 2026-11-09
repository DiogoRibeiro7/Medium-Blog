"""Figures for the article, from ``data/`` and ``results/``.

    python figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from common import FIGURES, RESULTS, bp_series, pt_curve

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
BLUE_LIGHT = "#86b6ef"
INK, INK2, MUTED, GRID, SURFACE, SHADE, GREY = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb", "#efeeea", "#b5b4ae"
CREDIT = "Diogo Ribeiro"

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
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def years(ax, step: int = 2) -> None:
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))


def mx(periods) -> np.ndarray:
    p = pd.PeriodIndex(periods, freq="M")
    return (p.year + (p.month - 1) / 12 + 1 / 24).to_numpy()


def fig1_composition() -> None:
    w = bp_series({12561432: "ot", 12561431: "otrv", 12561433: "bt", 12561444: "imf", 12561445: "loans_eur",
                   12561438: "ca", 12561434: "ct", 12561447: "total"})
    w.index = pd.PeriodIndex(w.index, freq="M")
    w = w.groupby(level=0).last().loc["2010-01":].fillna(0.0) / 1000
    x = mx(w.index)
    layers = [("ot", "Treasury bonds (OT and OTRV)", BLUE, w.ot + w.otrv), ("bt", "Treasury bills", BLUE_LIGHT, w.bt),
              ("imf", "IMF loan", RED, w.imf), ("loans_eur", "loans in euro (EFSF, EFSM; later SURE and others)", ORANGE, w.loans_eur),
              ("ca", "savings certificates", AQUA, w.ca), ("ct", "Treasury certificates", YELLOW, w.ct)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a1.stackplot(x, *[v for *_, v in layers], colors=[c for _, _, c, _ in layers], labels=[lab for _, lab, _, _ in layers], alpha=0.95, lw=0)
    a1.set_ylabel("EUR billion")
    a1.set_title("Direct State debt by instrument, end of month")
    a1.legend(loc="upper left", fontsize=8.4)
    a1.set_xlim(2010, x.max() + 0.1)
    years(a1, 2)
    share = pd.DataFrame({"official loans (IMF and euro loans)": (w.imf + w.loans_eur) / w.total,
                          "savings and Treasury certificates": (w.ca + w.ct) / w.total}) * 100
    for (k, s), c in zip(share.items(), (ORANGE, AQUA)):
        a2.plot(x, s, color=c, lw=2.3, label=k)
    a2.set_ylabel("% of direct State debt")
    a2.set_title("Shares of official and retail debt")
    a2.set_ylim(0, 40)
    a2.legend(loc="upper right", fontsize=8.6)
    a2.set_xlim(2010, x.max() + 0.1)
    years(a2, 2)
    suptitle(fig, "Official loans peaked at more than a third of the State's debt in 2014; retail certificates now hold about a sixth")
    note(fig, "BPstat (Banco de Portugal from IGCP data), direct State debt at nominal value. The IMF loan is the State's only non-euro loan. Euro loans include all loans in euro.")
    fig.tight_layout()
    save(fig, "fig1_composition")


def fig2_imf_cost() -> None:
    p = pd.read_csv(RESULTS / "imf_paths.csv")
    p = p[(p.month >= "2011-05") & (p.month <= "2024-12")]
    curve = pt_curve()
    x = mx(p.month)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1, 1.15]))
    a1.fill_between(x, p.credit_a, color=SHADE, lw=0, label="original schedule")
    a1.plot(x, p.credit_b, color=RED, lw=2.4, label="actual")
    a1.axhline(3.0 * 1.0297, xmin=0, xmax=(2016.13 - 2011) / (2025 - 2011), color=INK2, lw=1, ls="--")
    a1.axhline(1.875 * 2.0601, xmin=(2016.13 - 2011) / (2025 - 2011), xmax=1, color=INK2, lw=1, ls="--")
    a1.text(2019.2, 4.3, "surcharge threshold", fontsize=8.5, color=INK2)
    a1.set_xlim(2011, 2025)
    a1.set_ylabel("IMF credit outstanding, SDR billion")
    a1.set_title("Credit outstanding: original schedule and actual")
    a1.legend(loc="upper right", fontsize=8.6)
    years(a1, 2)
    a2.plot(x, p.rate_b.where(p.credit_b > 0), color=RED, lw=2.4, label="IMF: average rate of charge (actual path)")
    a2.plot(x, p.rate_a.where(p.credit_a > 0.05), color=RED, lw=1.4, ls=":", label="IMF: same, had the schedule been kept")
    a2.plot(x, p.sdr_rate, color=MUTED, lw=1.4, label="SDR interest rate (rebuilt)")
    cx = curve.loc["2011-05":"2024-12"]
    a2.plot(mx(cx.index.astype(str)), cx[3.0], color=BLUE, lw=1.8, label="Portugal 3-year yield")
    a2.plot(mx(cx.index.astype(str)), cx[10.0], color=BLUE_LIGHT, lw=1.6, label="Portugal 10-year yield")
    a2.axvspan(2015.17, 2018.95, color=SHADE, lw=0)
    a2.text(2017.05, 13.3, "early repayments", ha="center", fontsize=8.5, color=MUTED)
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_ylim(-1, 14)
    a2.set_xlim(2011, 2025)
    a2.set_ylabel("%")
    a2.set_title("Cost of IMF credit against Portugal's market yields")
    a2.legend(loc="upper right", fontsize=8.2)
    years(a2, 2)
    suptitle(fig, "From 2015 the IMF's surcharged credit cost more than Portuguese bonds of similar maturity; it was repaid by 2018, not 2024")
    note(fig, "IMF charges: SDR rate rebuilt from its basket + 100 bp + 200 bp on credit above the threshold (time-based surcharge left out). Yields: BPstat.")
    fig.tight_layout()
    save(fig, "fig2_imf_cost")


def fig3_imf_savings() -> None:
    r = pd.read_csv(RESULTS / "imf_replacement.csv")
    s = (RESULTS / "imf_summary.txt").read_text()
    import re
    ten_pre = float(re.search(r"interest saving to the original due dates ([\d,]+)", s).group(1).replace(",", ""))
    ten_post = float(re.search(r"ten-year bonds cost [\d,]+: gain ([-\d,]+)", s).group(1).replace(",", ""))
    avoided, cpn, fx, ante = r.avoided.sum(), r.coupons.sum(), r.fx.sum(), r.ex_ante.sum()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
    steps = [("IMF charges\navoided", avoided, RED), ("replacement\ncoupons", -cpn, BLUE), ("exchange rate\non principal", fx, YELLOW)]
    base = 0.0
    for i, (lab, v, c) in enumerate(steps):
        a1.bar(i, v, bottom=base if v >= 0 else base + v, color=c, width=0.6)
        a1.text(i, base + v - 40 if v >= 0 else base + v - 30, f"{v:+,.0f}", ha="center", va="top", fontsize=9.5,
                color="white" if v > 0 else INK2, fontweight="bold" if v > 0 else "normal")
        base += v
    a1.bar(3, base, color=INK, width=0.6)
    a1.text(3, base + 35, f"{base:,.0f}", ha="center", fontsize=9.5, color=INK, fontweight="bold")
    a1.bar(4.2, ante, color="none", edgecolor=INK2, lw=1.6, width=0.6)
    a1.text(4.2, ante + 35, f"{ante:,.0f}", ha="center", fontsize=9.5, color=INK2)
    a1.hlines(1900, 2.6, 4.6, color=GREEN, lw=1.4, ls="--")
    a1.text(4.62, 1900, "IGCP\nestimate,\n2019:\n~1,900", fontsize=8.3, color=GREEN, va="center")
    a1.set_xticks([0, 1, 2, 3, 4.2], [s[0] for s in steps] + ["ex post", "ex ante\n(rates frozen)"])
    a1.set_xlim(-0.5, 5.3)
    a1.set_ylabel("EUR million, undiscounted")
    a1.set_ylim(0, 2500)
    a1.set_title("Matched replacement: each instalment refinanced to its due date")
    labels = ["matched\nreplacement", "ten-year bonds:\nto the due dates", "ten-year bonds:\nafter the due dates", "ten-year bonds:\ntotal"]
    vals = [avoided - cpn + fx, ten_pre + fx, ten_post, ten_pre + fx + ten_post]
    cols = [INK, BLUE, ORANGE, RED]
    a2.bar(range(4), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        a2.text(i, v + (60 if v >= 0 else -60), f"{v:+,.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9.5, color=INK2)
    a2.axhline(0, color=MUTED, lw=0.9)
    a2.set_xticks(range(4), labels, fontsize=9)
    a2.set_ylim(-3300, 2400)
    a2.set_title("The same repayments refinanced with ten-year bonds, ex post")
    suptitle(fig, "Repaying the IMF early saved about EUR 1.6 billion ex post; refinancing it all at ten years would have given that back")
    note(fig, "EUR million, undiscounted. Ten-year variant: after each original due date, the counterfactual refinances at the realised Portuguese yield to the same end date.")
    fig.tight_layout()
    save(fig, "fig3_imf_savings")


def fig4_european() -> None:
    e = pd.read_csv(RESULTS / "european.csv")
    e = e[(e.due != "unknown") | (e.years == 5.0)]
    labels = [f"{r.lender} {r.month}\n{('due ' + r.due) if r.due != 'unknown' else 'tranche not published, 5y'}\nEUR {r.amount:.1f} bn" for r in e.itertuples()]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    xs = np.arange(len(e))
    ax.bar(xs - 0.18, e.yield_matched, 0.34, color=BLUE, label="break-even: Portuguese yield for the same maturity")
    ax.bar(xs + 0.18, e.rate_low, 0.34, color=ORANGE, label="published average rate of the lender")
    for i, r in enumerate(e.itertuples()):
        ax.text(i, max(r.yield_matched, r.rate_low) + 0.18, f"{r.saving_low:+,.0f} m", ha="center", fontsize=9.5, color=GREEN if r.saving_low > 0 else RED, fontweight="bold")
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xticks(xs, labels, fontsize=8.8)
    ax.set_ylabel("%")
    ax.set_ylim(-0.6, 3.9)
    ax.legend(loc="upper left", fontsize=8.8)
    suptitle(fig, "Repaying the EFSF in 2019 paid; at the EFSM's average rate, the 2023 and 2025 repayments at par did not")
    note(fig, "Saving at par = (lender's rate - break-even) x amount x years. Tranche rates are not published; average rates as reported from IGCP data (documented.csv).")
    fig.tight_layout()
    save(fig, "fig4_european")


def fig5_exchanges() -> None:
    d = pd.read_csv(RESULTS / "exchanges.csv")
    g = d.groupby(["date", "kind"], sort=False).agg(premium=("premium", "sum"), saved=("interest_saved", "sum"), nominal=("nominal", "sum"),
                                                     years=("years_deferred", lambda s: np.average(s, weights=d.loc[s.index, "nominal"]))).reset_index()
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    xs = np.arange(len(g))
    ax.bar(xs - 0.18, g.saved, 0.34, color=GREEN, label="interest no longer paid, to the old maturity")
    ax.bar(xs + 0.18, g.premium, 0.34, color=RED, label="premium paid above face value")
    for i, r in g.iterrows():
        top = max(r.saved, r.premium)
        txt = f"EUR {r.nominal:,.0f} m" + (f"\n+{r.years:.1f} years" if r.kind == "exchange" else "\ncash buyback")
        ax.text(i, top + 4, txt, ha="center", va="bottom", fontsize=8.4, color=INK2)
    ax.set_xticks(xs, [f"{r.date[:7]}\n{r.kind}" for r in g.itertuples()], fontsize=9)
    ax.set_ylabel("EUR million")
    ax.set_ylim(0, 210)
    ax.legend(loc="upper right", fontsize=8.8)
    suptitle(fig, "In every documented exchange the premium paid exceeded the interest no longer paid; cash buybacks roughly broke even")
    note(fig, "Bought bonds priced at the Portuguese yield for their residual maturity (BPstat). Labels: nominal bought and years by which redemption was pushed back.")
    fig.tight_layout()
    save(fig, "fig5_exchanges")


def fig6_retail() -> None:
    st = pd.read_csv(RESULTS / "retail_stock.csv")
    rt = pd.read_csv(RESULTS / "retail_rates.csv")
    m = pd.read_csv(RESULTS / "retail_monthly.csv")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1, 1.15, 1]))
    st = st[st.month >= "2015-01"]
    x = mx(st.month)
    a1.plot(x, st.ca / 1000, color=AQUA, lw=2.4, label="savings certificates")
    a1.plot(x, st.ct / 1000, color=YELLOW, lw=2.0, label="Treasury certificates")
    a1.axvline(2023.42, color=MUTED, lw=0.9, ls=":")
    a1.text(2023.3, 44, "series E\nclosed", ha="right", va="top", fontsize=8.4, color=MUTED)
    a1.set_ylabel("EUR billion")
    a1.set_title("Retail stocks")
    a1.legend(loc="upper left", fontsize=8.6)
    years(a1, 2)
    x = mx(rt.month)
    a2.plot(x, rt.series_e_new, color=AQUA, lw=2.4, label="series E rate (Euribor 3m + 1, cap 3.5)")
    a2.plot(x, rt.series_f_new, color=GREEN, lw=2.0, label="series F rate (Euribor 3m, cap 2.5)")
    a2.plot(x, rt.euribor12m, color=MUTED, lw=1.6, label="12-month Euribor")
    a2.plot(x, rt.pt5y, color=BLUE, lw=1.6, label="Portugal 5-year")
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_ylabel("%")
    a2.set_title("Base rates on new subscriptions and market rates")
    a2.set_ylim(-0.7, 6.0)
    a2.legend(loc="upper left", fontsize=7.9, ncol=2)
    years(a2, 1)
    m["y"] = m.month.str[:4]
    g = m.groupby(["y", "series"]).apply(lambda f: f.cost.sum() - f.cost_short.sum(), include_groups=False).unstack().fillna(0)
    ys = np.arange(len(g))
    a3.bar(ys - 0.18, g.get("E", 0), 0.34, color=AQUA, label="series E vintages")
    a3.bar(ys + 0.18, g.get("F", 0), 0.34, color=GREEN, label="series F vintages")
    a3.axhline(0, color=MUTED, lw=0.9)
    a3.set_xticks(ys, g.index)
    a3.set_ylabel("EUR million")
    a3.set_title("Cost against 12-month Euribor, by year")
    a3.legend(loc="upper left", fontsize=8.6)
    suptitle(fig, "Series E was cheap while its cap bound and dear once rates fell; series F has cost about the market rate")
    note(fig, "BPstat stocks; ECB Euribor. 2022-2026 vintages from monthly net inflows (series E to May 2023, F after). 2026: January-September.")
    fig.tight_layout()
    save(fig, "fig6_retail")


def run() -> None:
    fig1_composition()
    fig2_imf_cost()
    fig3_imf_savings()
    fig4_european()
    fig5_exchanges()
    fig6_retail()


if __name__ == "__main__":
    run()
