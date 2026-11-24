"""The five figures, drawn from results/ (run vat_numbers.py first).

python figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import vat_core as vc  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIG = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FIFTH_LABELS = ["Poorest\nfifth", "Second", "Middle", "Fourth", "Richest\nfifth"]
SOURCE = "Eurostat (snapshot in DiogoRibeiro7/data, eurostat-food-vat-incidence). Figure: Diogo Ribeiro."

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


def heading(fig, title, subtitle):
    fig.text(0.01, 0.985, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.945, subtitle, fontsize=9.5, color=INK2, va="top")


def footer(fig, note=""):
    text = f"{note}  Source: {SOURCE}" if note else f"Source: {SOURCE}"
    fig.text(0.01, 0.008, text, fontsize=8, color=MUTED, ha="left", va="bottom")


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def fig1_engel():
    """Dumbbell: food share of cash spending, poorest and richest fifth, 26 member states (2015 wave)."""
    f = pd.read_csv(RESULTS / "food_by_country_2015.csv", index_col="geo").sort_values("share_ratio_q1_q5")
    fig, ax = plt.subplots(figsize=(9, 9.2))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.86, bottom=0.08)
    y = np.arange(len(f))
    for i, (_g, r) in enumerate(f.iterrows()):
        ax.plot([100 * r.share_q5, 100 * r.share_q1], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.scatter(100 * f.share_q5, y, s=58, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5, label="Richest fifth")
    ax.scatter(100 * f.share_q1, y, s=58, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=1.5, label="Poorest fifth")
    ax.set_yticks(y)
    ax.set_yticklabels([vc.NAMES[g] for g in f.index])
    for lab, g in zip(ax.get_yticklabels(), f.index, strict=True):
        if g == "PT":
            lab.set_fontweight("bold")
            lab.set_color(INK)
    for i, (_g, r) in enumerate(f.iterrows()):
        ax.text(max(100 * r.share_q1, 100 * r.share_q5) + 0.8, i, f"×{r.share_ratio_q1_q5:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("Food and non-alcoholic drink, % of household spending (imputed rents excluded)")
    ax.set_xlim(0, 56)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncols=2)
    heading(
        fig,
        "The poorest fifth spends a larger share of its budget on food in 23 of 26 countries",
        "Food share of spending by income fifth, budget-survey wave 2015 (collected 2012–2017). Ratio = poorest ÷ richest.\n"
        "Below 1 (Netherlands, Denmark, Belgium) a food VAT cut is roughly proportional; above 1 it is progressive.",
    )
    footer(fig, "Italy publishes no survey by income fifth.")
    save(fig, "fig1_food_shares")


def fig2_euros_and_shares():
    """Scatter: euros ratio (richest/poorest) against share ratio (poorest/richest), food and restaurants."""
    f = pd.read_csv(RESULTS / "food_by_country_2015.csv", index_col="geo")
    r = pd.read_csv(RESULTS / "restaurants_by_country_2015.csv", index_col="geo")
    fig, ax = plt.subplots(figsize=(9, 7.2))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.84, bottom=0.11)
    ax.axhline(1, color=MUTED, lw=1, zorder=1)
    ax.axvline(1, color=MUTED, lw=1, zorder=1)
    dot = {"s": 60, "edgecolor": SURFACE, "linewidth": 1.5, "zorder": 3}
    ax.scatter(f.euros_ratio_q5_q1, f.share_ratio_q1_q5, color=BLUE, label="Food and drink at home", **dot)
    ax.scatter(r.euros_ratio_q5_q1, r.share_ratio_q1_q5, color=ORANGE, label="Restaurants and cafés", **dot)
    pt_x = [f.loc["PT", "euros_ratio_q5_q1"], r.loc["PT", "euros_ratio_q5_q1"]]
    pt_y = [f.loc["PT", "share_ratio_q1_q5"], r.loc["PT", "share_ratio_q1_q5"]]
    ax.scatter(pt_x, pt_y, s=150, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=4)
    arrow = {"arrowstyle": "-", "color": INK2, "lw": 0.8}
    ax.annotate("Portugal", (pt_x[0], pt_y[0]), xytext=(2.4, 1.85), fontsize=9, color=INK, arrowprops=arrow)
    food_labels = {"NL": (6, -10), "CY": (6, 4), "SI": (-4, 8), "FR": (6, 2), "SE": (-18, -12)}
    rest_labels = {"PT": (6, 4), "DE": (-10, -14), "ES": (6, 4), "PL": (6, 4), "EE": (-6, 8), "BG": (-6, -14), "RO": (6, 4)}
    label = {"textcoords": "offset points", "fontsize": 8.5, "color": INK2}
    for g, off in food_labels.items():
        ax.annotate(g, (f.loc[g, "euros_ratio_q5_q1"], f.loc[g, "share_ratio_q1_q5"]), xytext=off, **label)
    for g, off in rest_labels.items():
        ax.annotate(g, (r.loc[g, "euros_ratio_q5_q1"], r.loc[g, "share_ratio_q1_q5"]), xytext=off, **label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [1, 1.5, 2, 3, 5, 8, 12]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    yt = [0.2, 0.3, 0.5, 0.7, 1, 1.5, 2]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{t:g}" for t in yt])
    ax.minorticks_off()
    ax.set_xlim(0.9, 17)
    ax.set_ylim(0.15, 2.4)
    ax.set_xlabel("Euros saved by the richest fifth ÷ euros saved by the poorest (per adult equivalent)")
    ax.set_ylabel("Saving as % of budget: poorest ÷ richest")
    ax.text(16, 2.25, "Richest saves more euros,\npoorest saves a larger share", ha="right", va="top", fontsize=9, color=INK2)
    ax.text(2.6, 0.17, "Richest saves more euros\nand a larger share", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62))
    heading(
        fig,
        "Food: more euros for the richest, a larger share for the poorest",
        "Each dot is a member state: a 1% fall in prices, fully passed on, budget-survey wave 2015. Log scales.\n"
        "Food sits in the upper right in 23 of 26 countries; restaurants sit in the lower right in 25 of 26 (Poland is the exception).",
    )
    footer(fig)
    save(fig, "fig2_euros_and_shares")


def fig3_where_the_money_goes():
    """Grouped bars: each fifth's share of the total saving and of total spending, EU27 and Portugal."""
    eu = pd.read_csv(RESULTS / "eu27_profile_2015.csv", index_col=0)
    pt = pd.read_csv(RESULTS / "portugal_2020.csv", index_col=0)
    panels = [
        ("EU27, wave 2015", 100 * eu["share_of_total"], 100 * eu["spend_share_of_total"]),
        ("Portugal, survey of 2022/23", pt["saving_share_of_total_pct"], pt["spend_share_of_total_pct"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.17, wspace=0.08)
    x = np.arange(5)
    w = 0.38
    for ax, (title, saving, spend) in zip(axes, panels, strict=True):
        ax.bar(x - w / 2 - 0.01, saving, w, color=BLUE, label="Share of the euros saved")
        ax.bar(x + w / 2 + 0.01, spend, w, color=AQUA, label="Share of household spending")
        for xi, v in zip(x - w / 2 - 0.01, saving, strict=True):
            ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=8, color=INK2)
        for xi, v in zip(x + w / 2 + 0.01, spend, strict=True):
            ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(FIFTH_LABELS, fontsize=9)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 40)
    axes[0].set_ylabel("% of the total")
    axes[0].legend(loc="upper left")
    heading(
        fig,
        "Richer fifths receive more of the money, but less than their share of spending",
        "Each income fifth's share of the euros saved by a cut in food VAT, against its share of all household spending.\n"
        "The saving rises with income less steeply than spending does, so it narrows the gap between fifths.",
    )
    footer(fig, "EU27: imputed rents excluded. Portugal: published total.")
    save(fig, "fig3_where_the_money_goes")


def fig4_portugal_yardsticks():
    """Three small panels for Portugal: euros a month, % of spending, % of disposable income."""
    pt = pd.read_csv(RESULTS / "portugal_2020.csv", index_col=0)
    inc = pd.read_csv(RESULTS / "food_pct_income_2020.csv", index_col="geo").loc["PT", vc.FIFTHS]
    cut = vc.price_fall(0.06, 0.0)
    euros = pt["saving_6to0_eur_month_now"]
    pct_spend = pt["saving_6to0_pct_budget"]
    pct_income = cut * inc
    panels = [
        ("Euros per household a month", euros, "{:.0f} €", BLUE),
        ("% of household spending", pct_spend, "{:.2f}%", BLUE),
        ("% of disposable income", pct_income, "{:.2f}%", BLUE),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 5.0))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.76, bottom=0.17, wspace=0.28)
    x = np.arange(5)
    for ax, (title, values, fmt, color) in zip(axes, panels, strict=True):
        ax.bar(x, values, 0.62, color=color)
        top = float(np.max(values))
        for xi, v in zip(x, values, strict=True):
            ax.text(xi, v + 0.02 * top, fmt.format(v), ha="center", fontsize=8.5, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(["Poorest", "2nd", "Middle", "4th", "Richest"], fontsize=9)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK)
        ax.set_ylim(0, top * 1.18)
        ax.grid(axis="x", visible=False)
    heading(
        fig,
        "Portugal: the same cut, measured three ways",
        "Saving from taking VAT on all food and non-alcoholic drink from 6% to 0% (prices 5.66% lower), by income fifth.\n"
        "Euros rise with income; shares of spending and of income fall with it. The income yardstick is the steepest.",
    )
    footer(fig, "Euros at 2025 food prices. Income: Eurostat ICW (experimental).")
    save(fig, "fig4_portugal_three_yardsticks")


def fig5_who_is_rich():
    """Dot plot: where a single person with 1,500 euros a month (12 payments) sits in each income distribution."""
    w = pd.read_csv(RESULTS / "who_is_rich.csv", index_col=0).drop(index="EU27_2020", errors="ignore")
    col = "single, 1,500 x 12"
    col14 = "single, 1,500 x 14"
    pos = w[col].fillna(92.0)
    pos14 = w[col14].fillna(92.0)
    order = pos.sort_values().index
    fig, ax = plt.subplots(figsize=(9, 9.2))
    fig.subplots_adjust(left=0.17, right=0.95, top=0.86, bottom=0.08)
    y = np.arange(len(order))
    ax.axvspan(80, 100, color="#eef3fb", zorder=0)
    ax.text(90, len(order) - 0.3, "Richest fifth", ha="center", va="bottom", fontsize=9, color=INK2)
    for i, g in enumerate(order):
        ax.plot([pos[g], pos14[g]], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.scatter(pos[order], y, s=58, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="1,500 € × 12")
    ax.scatter(pos14[order], y, s=58, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="1,500 € × 14")
    ax.set_yticks(y)
    ax.set_yticklabels([vc.NAMES[g] for g in order])
    for lab, g in zip(ax.get_yticklabels(), order, strict=True):
        if g == "PT":
            lab.set_fontweight("bold")
            lab.set_color(INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(order) + 0.4)
    ax.set_xlabel("Percent of people with a lower equivalised disposable income")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    heading(
        fig,
        "Where a single person taking home 1,500 euros a month sits in each country",
        "Position in the distribution of equivalised disposable income, EU-SILC 2025 (incomes of 2024), interpolated\n"
        "between decile cut-offs. Positions above the ninth decile are drawn at 92.",
    )
    footer(fig)
    save(fig, "fig5_who_is_rich")


def main():
    fig1_engel()
    fig2_euros_and_shares()
    fig3_where_the_money_goes()
    fig4_portugal_yardsticks()
    fig5_who_is_rich()


if __name__ == "__main__":
    main()
