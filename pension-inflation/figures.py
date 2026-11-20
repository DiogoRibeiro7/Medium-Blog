"""Draw the article's figures from the tables pension_numbers.py writes to results/.

python figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import pension_core as pc  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LIGHT_BLUE, MID_BLUE, LIGHT_GREY = "#cde2fb", "#86b6ef", "#e9e8e2"
RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    }
)
CREDIT = "Data: OECD (consumer prices, long-term rates), Eurostat (life tables). Code: Medium-Blog/pension-inflation."
CREDIT += " Figure: Diogo Ribeiro."


def frame(fig, title, subtitle, top_in=0.85, bottom_in=0.7, credit=CREDIT):
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - top_in / h, bottom=bottom_in / h)
    fig.text(0.01, 1 - 0.08 / h, title, fontsize=11, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.01, 1 - 0.36 / h, subtitle, fontsize=8.5, color=INK2, ha="left", va="top", linespacing=1.4)
    fig.text(0.01, 0.06 / h, credit, fontsize=7.2, color=MUTED, ha="left", va="bottom")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def percent_axis(ax, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def fig1_fan():
    f = pd.read_csv(RES / "fan.csv")
    e = pd.read_csv(RES / "euro_era_1999.csv")
    age = 65 + f.k
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.fill_between(age, f.p10, f.p90, color=LIGHT_BLUE, lw=0, label="Middle 80% of the 1,010 histories")
    ax.fill_between(age, f.p25, f.p75, color=MID_BLUE, lw=0, label="Middle 50%")
    ax.plot(age, f.p50, color=BLUE, lw=2.2, label="Median history")
    ax.plot(age, f.steady2, color=ORANGE, lw=2, label="Steady 2% inflation, the euro-area target")
    ax.plot(65 + e.k, e["median"], color=INK, lw=1.6, label="Started in 1999 (median of 21 countries, to 2025)")
    for a in (75, 85):
        k = a - 65
        ax.annotate(
            f"{100 * f.p50[k]:.0f}%",
            (a, f.p50[k]),
            xytext=(6, -14),
            textcoords="offset points",
            fontsize=8.5,
            color=BLUE,
            fontweight="bold",
        )
        ax.annotate(
            f"{100 * f.steady2[k]:.0f}%",
            (a, f.steady2[k]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.5,
            color=ORANGE,
            fontweight="bold",
        )
    grid(ax)
    ax.set_xlim(65, 90)
    ax.set_ylim(0, 1.05)
    percent_axis(ax)
    ax.set_xlabel("Age of the pensioner (pension starts at 65)")
    ax.set_ylabel("Purchasing power of the first payment")
    ax.legend(loc="lower left", fontsize=8.5)
    frame(
        fig,
        "A pension fixed in money, replayed on 23 countries' prices from 1955",
        "Real value of a payment that never changes, for pensions started each year 1955-2000 in the OECD's 1973"
        " members (without Turkey).\nEach history uses its own country's consumer prices.",
    )
    fig.savefig(FIG / "fig1_fan.png")
    plt.close(fig)


def fig2_heatmap():
    h = pd.read_csv(RES / "real_at_85.csv")
    h = h[h.start <= 2005]
    order = h.groupby("country").real85.median().sort_values(ascending=False).index
    years = np.arange(1955, 2006)
    m = np.full((len(order), len(years)), np.nan)
    for i, c in enumerate(order):
        sub = h[h.country == c].set_index("start").real85
        for j, y in enumerate(years):
            if y in sub.index:
                m[i, j] = 1 - sub[y]
    cmap = LinearSegmentedColormap.from_list("loss", RAMP)
    cmap.set_bad(LIGHT_GREY)
    fig, ax = plt.subplots(figsize=(10, 7.2))
    im = ax.imshow(m, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(order)), [pc.NAMES[c] for c in order], fontsize=8.5)
    ticks = [j for j, y in enumerate(years) if y % 5 == 0]
    ax.set_xticks(ticks, [str(years[j]) for j in ticks])
    ax.set_xlabel("Year the pension started (at 65)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    cb.set_label("Purchasing power lost by 85", color=INK2)
    cb.outline.set_visible(False)
    frame(
        fig,
        "Who lost the most: it depends on the country, and on the decade a pension met",
        "Share of a fixed pension's purchasing power gone by age 85, by country and start year. Countries ordered by"
        " their median.\nGrey: no prices for that start year, or not yet 20 years of prices after it.",
        top_in=0.9,
    )
    fig.savefig(FIG / "fig2_heatmap.png")
    plt.close(fig)


def fig3_rules():
    r = pd.read_csv(RES / "rules_summary.csv").set_index("rule")
    order = ["fixed", "half", "step2", "cap25", "cap3", "step3", "cap5", "minus05"]
    r = r.loc[order]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    y = np.arange(len(order))[::-1]
    for yi, (_, row) in zip(y, r.iterrows(), strict=True):
        ax.plot([row.worst_tenth_median, row.calm_median], [yi, yi], color=GRID, lw=3, solid_capstyle="round")
    ax.scatter(r.worst_tenth_median, y, s=70, color=ORANGE, zorder=3, label="Worst tenth of histories", ec="white")
    ax.scatter(r.calm_median, y, s=70, color=BLUE, zorder=3, label="Calmer half of histories", ec="white")
    for yi, (_, row) in zip(y, r.iterrows(), strict=True):
        label = {"va": "center", "fontsize": 8, "color": INK2}
        if abs(row.calm_median - row.worst_tenth_median) < 0.03:
            ax.text(row.worst_tenth_median - 0.02, yi, f"{100 * row.calm_median:.0f}% in both", ha="right", **label)
            continue
        ax.text(row.worst_tenth_median - 0.02, yi, f"{100 * row.worst_tenth_median:.0f}%", ha="right", **label)
        ax.text(row.calm_median + 0.02, yi, f"{100 * row.calm_median:.0f}%", ha="left", **label)
    ax.axvline(1, color=INK2, lw=0.8)
    ax.text(1.005, y[0] + 0.55, "full price indexation", fontsize=8, color=INK2, va="bottom")
    ax.set_yticks(y, [pc.RULE_LABELS[k] for k in order])
    ax.set_xlim(0.15, 1.18)
    ax.set_ylim(-0.6, len(order) - 0.2)
    percent_axis(ax, "x")
    grid(ax, "x")
    ax.set_xlabel("Lifetime real income kept, counting each year by the chance of being alive (median)")
    ax.legend(loc="lower left", fontsize=8.5)
    frame(
        fig,
        "Partial indexation protects in the calm histories and fails in the rough ones",
        "Each rule replayed on the same 918 pension starts, 1955-1996, ages 65 to 94. The worst tenth are the"
        " histories where\na fixed pension lost the most; the calmer half are those where it lost less than the"
        " median.",
    )
    fig.savefig(FIG / "fig3_rules.png")
    plt.close(fig)


def fig4_level_vs_indexed():
    x = pd.read_csv(RES / "level_vs_indexed.csv")
    med = x.groupby("start").value_r2.median()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axhspan(1, 2.0, color="#f4f8fd", lw=0)
    ax.scatter(x.start, x.value_r2, s=12, color=MID_BLUE, alpha=0.8, lw=0, label="One country and purchase year")
    ax.plot(med.index, med.values, color=BLUE, lw=2.2, label="Median across countries")
    ax.axhline(1, color=INK2, lw=0.9)
    ax.text(1955.5, 1.93, "The level annuity turned out better", fontsize=8.5, color=INK2, va="top")
    ax.text(1955.5, 0.6, "The indexed annuity turned out better", fontsize=8.5, color=INK2, va="bottom")
    grid(ax)
    ax.set_xlim(1955, 1997)
    ax.set_ylim(0.5, 2.0)
    percent_axis(ax)
    ax.set_xlabel("Year the annuity was bought, at 65")
    ax.set_ylabel("Real value of the level annuity\n(indexed annuity = 100%)")
    ax.legend(loc="lower right", fontsize=8.5)
    frame(
        fig,
        "Buying protection in the 1960s paid; in the 1980s it cost",
        "A level annuity priced at each country's 10-year bond yield of that year, against an indexed annuity at a 2%"
        " real rate,\nsame premium, valued over ages 65 to 94 at the prices that followed. 473 purchases in 22"
        " countries.",
    )
    fig.savefig(FIG / "fig4_level_vs_indexed.png")
    plt.close(fig)


def fig5_portugal():
    p = pd.read_csv(RES / "portugal_pension.csv")
    p = p[p.year <= 2025]
    fig, ax = plt.subplots(figsize=(10, 5.0))
    ax.axvspan(2010.5, 2015.5, color=LIGHT_GREY, lw=0)
    ax.axvspan(2015.5, 2017.5, color="#f3f2ee", lw=0)
    ax.text(2013, 1.065, "update suspended\nfor all pensions", ha="center", va="top", fontsize=8, color=INK2)
    ax.text(2016.5, 1.065, "pensions above\n1.5 or 2 IAS\nnot updated", ha="center", va="top", fontsize=8, color=INK2)
    ax.plot(p.year, p.real, color=BLUE, lw=2.2, marker="o", ms=4, label="A pension of 1,000 euros in 2008")
    ax.plot(
        p.year,
        p.lowest_real,
        color=ORANGE,
        lw=2,
        marker="o",
        ms=4,
        label="The lowest bracket's percentage updates (fixed-euro rises not included)",
    )
    ax.axhline(1, color=INK2, lw=0.9)
    low = p.loc[p.real.idxmin()]
    ax.annotate(
        f"{100 * low.real:.1f}% in {int(low.year)}",
        (low.year, low.real),
        xytext=(-70, -22),
        textcoords="offset points",
        fontsize=8.5,
        color=BLUE,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 0.8},
    )
    grid(ax)
    ax.set_xlim(2007.6, 2025.4)
    ax.set_ylim(0.85, 1.07)
    ax.set_yticks(np.arange(0.86, 1.065, 0.02))
    percent_axis(ax)
    ax.set_xticks(range(2008, 2026, 2))
    ax.set_ylabel("Purchasing power of the 2008 amount")
    ax.legend(loc="lower left", fontsize=8.5)
    frame(
        fig,
        "Portugal: the monthly amount never fell, yet the pension lost 11% of its value by 2022",
        "Monthly amount of a general-regime pension rebuilt from the annual update ordinances, 2008-2025, deflated by"
        " consumer prices.\nOne-off payments (the half pension of October 2022) and temporary cuts are not included.",
        credit="Data: Diario da Republica (pension update ordinances), OECD (consumer prices)."
        " Code: Medium-Blog/pension-inflation. Figure: Diogo Ribeiro.",
    )
    fig.savefig(FIG / "fig5_portugal.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig1_fan()
    fig2_heatmap()
    fig3_rules()
    fig4_level_vs_indexed()
    fig5_portugal()


if __name__ == "__main__":
    main()
