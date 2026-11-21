"""The five figures, drawn from results/ (run growth_numbers.py first).

python figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import growth_core as gc  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATA = HERE / "data"
FIG = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
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
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "legend.labelcolor": INK2,
    }
)


def footer(fig, source):
    fig.text(0.01, 0.01, f"Source: {source}. Figure: {CREDIT}.", fontsize=8, color=MUTED, ha="left", va="bottom")


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def heading(fig, title, subtitle):
    fig.text(0.01, 0.965, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.905, subtitle, fontsize=9.5, color=INK2, va="top")


def label_end(ax, x, y, text, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", fontsize=9.5, color=INK2)


# -------------------------------------------------------------------------------------------------
def fig1_value_in_time(r_hist, cape_med, cape_now):
    years = np.arange(0, 151)
    cases = [
        ("Flat cash flow, r = 8%", 0.08, 0.0, BLUE),
        ("4% growth, r = 8%", 0.08, 0.04, ORANGE),
        (f"Index at median CAPE {cape_med:.1f}", r_hist, r_hist - 1 / cape_med, AQUA),
        (f"Index at CAPE {cape_now:.0f} (autumn 2026)", r_hist, r_hist - 1 / cape_now, YELLOW),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for label, r, g, col in cases:
        received = 1 - np.array([gc.share_beyond(n, r, g) for n in years])
        ax.plot(years, 100 * received, color=col, label=label)
    ax.axvline(50, color=MUTED, lw=1, ls="--")
    ax.text(51, 4, "50 years", color=INK2, fontsize=9)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Years from today")
    ax.set_ylabel("Share of today's value already received (%)")
    ax.legend(loc="lower right", fontsize=9.5)
    heading(
        fig,
        "How much of a valuation depends on the distant future",
        f"Growing perpetuities. The index cases use the 1871–2023 real return of {100 * r_hist:.1f}% and the lowest growth consistent with the CAPE.",
    )
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.09, right=0.97)
    footer(fig, "own calculation; CAPE from Shiller data")
    save(fig, "fig1_value_in_time")


def fig2_firm_in_market():
    d = pd.read_csv(RESULTS / "logistic_path.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
    a1.plot(d.year, 100 * d.share, color=BLUE)
    a1.set_ylim(0, 102)
    a1.set_xlim(0, 80)
    a1.set_xlabel("Year")
    a1.set_ylabel("Market share (%)")
    a1.set_title("Share of its market", fontsize=11)
    a2.plot(d.year, 100 * d.firm_growth, color=BLUE, label="Firm revenue growth")
    a2.plot(d.year, 100 * d.market_growth, color=ORANGE, ls="--", label="Market growth")
    a2.set_ylim(0, 17)
    a2.set_xlim(0, 80)
    a2.set_xlabel("Year")
    a2.set_ylabel("Growth per year (%)")
    a2.set_title("Growth of the firm and of its market", fontsize=11)
    a2.legend(loc="upper right", fontsize=9.5)
    heading(
        fig,
        "A firm can outgrow its market only until it is the market",
        "A firm starting at 5% of a market that grows 3% a year, gaining share along a logistic curve.",
    )
    fig.subplots_adjust(top=0.76, bottom=0.16, left=0.07, right=0.98, wspace=0.25)
    footer(fig, "own simulation")
    save(fig, "fig2_firm_in_market")


def fig3_eps_vs_economy():
    d = pd.read_csv(RESULTS / "eps_gdp_index.csv")
    eps_smooth = d.real_eps.rolling(10).mean()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(d.year, d.real_gdp, color=BLUE, label="Real GDP")
    ax.plot(d.year, d.real_gdp_pc, color=ORANGE, label="Real GDP per person")
    ax.plot(d.year, d.real_eps, color=AQUA, lw=1, alpha=0.45)
    ax.plot(d.year, eps_smooth, color=AQUA, label="Real S&P earnings per share (10-year mean; thin line annual)")
    ax.set_yscale("log")
    ax.set_xlim(1871, 2022)
    ax.set_ylabel("Index, 1871–1880 average = 1 (log scale)")
    ax.legend(loc="upper left", fontsize=9.5)
    last = d.iloc[-10:].mean()
    for _col, val in (("real_gdp", last.real_gdp), ("real_gdp_pc", last.real_gdp_pc), ("real_eps", last.real_eps)):
        label_end(ax, 2022, val, f"×{val:.0f}")
    ax.set_xlim(1871, 2034)
    heading(
        fig,
        "Earnings per share have tracked income per person, not the economy",
        "United States, 1871–2022, inflation removed.",
    )
    fig.subplots_adjust(top=0.84, bottom=0.1, left=0.09, right=0.97)
    footer(fig, "Shiller S&P Composite (datasets/s-and-p-500); Maddison Project Database 2023 via Our World in Data")
    save(fig, "fig3_eps_vs_economy")


def fig4_implied_growth(r_hist, cape_now, eps_long, gdp_long):
    d = pd.read_csv(RESULTS / "implied_growth_series.csv")
    floor = -4.0
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.plot(d.year, 100 * d.g_min, color=BLUE, label="Minimum perpetual real growth implied by the price")
    ok = d.dropna(subset=["realised_eps_growth_next20"])
    ax.plot(ok.year, 100 * ok.realised_eps_growth_next20, color=ORANGE, label="Real EPS growth over the next 20 years")
    box = {"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5}
    ax.axhline(100 * eps_long, color=MUTED, lw=1.2, ls="--")
    ax.axhline(100 * gdp_long, color=MUTED, lw=1.2, ls=":")
    ax.text(
        2031,
        100 * eps_long - 0.15,
        f"real EPS growth,\n1871–2022: {100 * eps_long:.1f}%",
        color=INK2,
        fontsize=8.5,
        va="center",
    )
    ax.text(
        2031,
        100 * gdp_long + 0.15,
        f"US real GDP growth,\n1871–2022: {100 * gdp_long:.1f}%",
        color=INK2,
        fontsize=8.5,
        va="center",
    )
    g_now = r_hist - 1 / cape_now
    ax.scatter([2026.7], [100 * g_now], s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=5)
    ax.annotate(
        f"autumn 2026, CAPE {cape_now:.0f}: {100 * g_now:.1f}%",
        (2026.7, 100 * g_now),
        xytext=(0, 14),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK2,
        bbox=box,
    )
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlim(1881, 2030)
    ax.set_ylim(floor, 6.2)
    ax.set_ylabel("Percent per year")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.06), ncol=2, fontsize=9.5)
    heading(
        fig,
        "The growth the S&P price requires, against the growth delivered",
        f"Required real return {100 * r_hist:.1f}% (the 1871–2023 average); implied growth = that return minus 1/CAPE.\n"
        "The blue line runs below −4% around 1917–1922, 1932, 1942 and 1977–1985, when the CAPE was very low.",
    )
    fig.subplots_adjust(top=0.8, bottom=0.16, left=0.08, right=0.83)
    footer(fig, "Shiller S&P Composite (datasets/s-and-p-500); CAPE 2026 as reported by CFA Institute")
    save(fig, "fig4_implied_growth")


def fig5_energy():
    w = pd.read_csv(DATA / "world.csv").set_index("year")
    gdp = w.gdp.dropna()
    e = w.energy_twh
    c = pd.read_csv(RESULTS / "country_decoupling.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.9), gridspec_kw={"width_ratios": [1, 1.1]})
    a1.plot(gdp.index, gdp / gdp.loc[1970], color=BLUE, marker="o", ms=4, label="World GDP (Maddison benchmark years)")
    a1.plot(e.loc[1970:].index, e.loc[1970:] / e.loc[1970], color=ORANGE, label="World primary energy")
    prod = gdp / e.loc[gdp.index]
    a1.plot(prod.index, prod / prod.loc[1970], color=AQUA, marker="o", ms=4, label="GDP per unit of energy")
    a1.set_xlim(1968, 2026)
    a1.set_ylim(0.8, 6.4)
    a1.set_ylabel("Index, 1970 = 1")
    a1.set_title("The world", fontsize=11)
    a1.legend(loc="upper left", fontsize=9)

    x = np.log(c.gdp_ratio)
    y = np.log(c.e_ratio)
    is_abs = c["class"] == "absolute"
    a2.scatter(x[~is_abs], y[~is_abs], s=22, color=MUTED, alpha=0.6, edgecolor=SURFACE, linewidth=0.8, label="Others")
    a2.scatter(x[is_abs], y[is_abs], s=30, color=BLUE, edgecolor=SURFACE, linewidth=1.2, label="Grew with less energy")
    lim = (np.log(0.35), np.log(14))
    a2.plot(lim, lim, color=MUTED, lw=1, ls="--")
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.axvline(0, color=MUTED, lw=0.8)
    a2.set_xlim(*lim)
    a2.set_ylim(*lim)
    a2.text(np.log(0.42), np.log(0.62), "equal growth", color=INK2, fontsize=8.5, rotation=45)
    ticks = [0.5, 1, 2, 4, 8]
    a2.set_xticks(np.log(ticks))
    a2.set_xticklabels([f"×{t:g}" for t in ticks])
    a2.set_yticks(np.log(ticks))
    a2.set_yticklabels([f"×{t:g}" for t in ticks])
    a2.set_xlabel("Change in real GDP, 2000 to 2022")
    a2.set_ylabel("Change in primary energy, 2000 to 2022")
    a2.set_title("Countries, 2000 to 2022", fontsize=11)
    a2.legend(loc="upper left", fontsize=9)
    offsets = {"United States": (70, -34), "Japan": (-46, -26), "United Kingdom": (40, -42)}
    for name, off in offsets.items():
        row = c[c.country == name].iloc[0]
        a2.annotate(
            name,
            (np.log(row.gdp_ratio), np.log(row.e_ratio)),
            xytext=off,
            textcoords="offset points",
            fontsize=8,
            color=INK2,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
        )
    heading(
        fig,
        "Growth has needed more energy, except in a minority of rich countries",
        "Left: world totals. Right: each dot is a country; dots below the horizontal line used less primary energy in 2022 than in 2000.",
    )
    fig.subplots_adjust(top=0.8, bottom=0.15, left=0.06, right=0.98, wspace=0.22)
    footer(
        fig,
        "Energy Institute Statistical Review and EIA; Maddison Project Database 2023; both via Our World in Data. "
        "Countries with at least 1 million people",
    )
    save(fig, "fig5_energy")


def main():
    s = pd.read_csv(DATA / "sp500_monthly.csv")
    tr = gc.real_total_return_index(s.price, s.dividend, s.cpi)
    r_hist = tr.iloc[-1] ** (12 / (len(s) - 1)) - 1
    cape_med = s.cape.dropna().median()
    eps = pd.read_csv(RESULTS / "eps_vs_gdp.csv").iloc[0]
    fig1_value_in_time(r_hist, cape_med, 40.0)
    fig2_firm_in_market()
    fig3_eps_vs_economy()
    fig4_implied_growth(r_hist, 40.0, eps.real_eps, eps.real_gdp)
    fig5_energy()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
