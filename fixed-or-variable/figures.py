"""Draw the article's figures from the tables mortgage_numbers.py writes to results/.

python figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LIGHT_BLUE, LIGHT_ORANGE, LIGHT_GREY = "#c6dcf6", "#f8c9b2", "#e9e8e2"
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
CREDIT = (
    "Data: Eurostat (Euribor, ECB euro yield curves, HICP, national accounts). Code: Medium-Blog/fixed-or-variable."
)
CREDIT += " Figure: Diogo Ribeiro."


def frame(fig, title, subtitle, top_in=0.8, bottom_in=0.7):
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - top_in / h, bottom=bottom_in / h)
    fig.text(0.01, 1 - 0.08 / h, title, fontsize=11, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.01, 1 - 0.36 / h, subtitle, fontsize=8.5, color=INK2, ha="left", va="top", linespacing=1.4)
    fig.text(0.01, 0.06 / h, CREDIT, fontsize=7.2, color=MUTED, ha="left", va="bottom")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def dates(index):
    return pd.to_datetime(pd.Index(index) + "-01")


def fig1_rates():
    d = pd.read_csv(RES / "rates.csv").set_index("month")
    x = dates(d.index)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    neg = d.euribor12 < 0
    ax.fill_between(x, -1.9, 6.5, where=neg, color=LIGHT_GREY, lw=0, step="mid")
    ax.text(pd.Timestamp("2019-02-01"), 6.1, "Euribor\nbelow zero", ha="center", va="top", fontsize=8, color=INK2)
    ax.plot(
        x, d.par10, color=ORANGE, lw=2, label="10-year yield (AAA government bonds): the base of a 10-year fixed rate"
    )
    ax.plot(x, d.euribor12, color=BLUE, lw=2, label="12-month Euribor: the base of a variable rate")
    ax.axhline(0, color=INK2, lw=0.8)
    grid(ax)
    ax.set_ylim(-1.9, 6.5)
    ax.set_ylabel("Per cent a year")
    ax.legend(loc="lower left", fontsize=8.5, ncols=2)
    frame(
        fig,
        "Two reference rates: what a fixed loan locks in, and what a variable loan pays",
        "Euro area, monthly averages, January 1999 to October 2026. A bank adds its margin to either rate.",
    )
    fig.savefig(FIG / "fig1_rates.png")
    plt.close(fig)


def fig2_cost():
    d = pd.read_csv(RES / "cohorts.csv").set_index("start")
    x = dates(d.index)
    y = d.extra_fixed / 1000
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = np.where(d.complete, np.where(y > 0, ORANGE, BLUE), np.where(y > 0, LIGHT_ORANGE, LIGHT_BLUE))
    ax.bar(x, y, width=28, color=colors, lw=0)
    ax.axhline(0, color=INK2, lw=0.8)
    last = x[d.complete.to_numpy()][-1]
    ax.axvline(last, color=MUTED, lw=0.8, ls="--")
    ax.text(
        last, 33, "  later loans: ten years not\n  yet over, compared to Oct 2026", fontsize=8, color=INK2, va="top"
    )
    ax.text(pd.Timestamp("1999-06-01"), 33, "Above zero: fixing cost more", fontsize=8.5, color=INK2, va="top")
    ax.text(pd.Timestamp("1999-06-01"), -13, "Below zero: fixing saved money", fontsize=8.5, color=INK2, va="bottom")
    grid(ax)
    ax.set_ylim(-14, 34)
    ax.set_ylabel("Thousand euros per 100,000 borrowed")
    ax.set_xlabel("Month the loan started")
    frame(
        fig,
        "Fixing for ten years cost more than going variable for almost every loan started before 2015",
        "Extra interest paid over the first ten years by a 10-year fixed loan, against a variable loan started the same"
        " month. 30-year loans, margin 1 point on both.",
    )
    fig.savefig(FIG / "fig2_cost.png")
    plt.close(fig)


def fig3_forecast():
    d = pd.read_csv(RES / "cohorts.csv").set_index("start")
    d = d[d.complete]
    x = dates(d.index)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.fill_between(x, d.euribor_avg, d.par_start, color=LIGHT_ORANGE, lw=0)
    ax.plot(x, d.par_start, color=ORANGE, lw=2, label="10-year yield when the loan started (what fixing locked in)")
    ax.plot(x, d.euribor_avg, color=BLUE, lw=2, label="12-month Euribor, averaged over the next ten years (what came)")
    ax.axhline(0, color=INK2, lw=0.8)
    grid(ax)
    ax.set_ylim(-0.6, 6.5)
    ax.set_ylabel("Per cent a year")
    ax.set_xlabel("Month the loan started")
    ax.legend(loc="upper right", fontsize=8.5)
    frame(
        fig,
        "Rates kept falling below what the yield curve expected",
        "Euro area,"
        " loans started January 1999 to November 2016. The shaded gap is most of what variable borrowers saved.",
    )
    fig.savefig(FIG / "fig3_forecast.png")
    plt.close(fig)


def fig4_paths():
    d = pd.read_csv(RES / "payment_paths.csv").set_index("month")
    starts = ["2005-01", "2015-01", "2021-10"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.4), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, s in zip(axes, starts, strict=True):
        v = d[f"{s}|variable"].dropna()
        f = d[f"{s}|fixed"].dropna()
        x = dates(v.index)
        ax.step(x, v, where="post", color=BLUE, lw=2, label="Variable")
        ax.plot(dates(f.index), f, color=ORANGE, lw=2, label="Fixed for ten years")
        peak = v.idxmax()
        ax.annotate(
            f"{v.max():.0f}",
            (pd.Timestamp(peak + "-01"), v.max()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK2,
        )
        ax.annotate(
            f"{v.iloc[0]:.0f}", (x[0], v.iloc[0]), xytext=(2, -12), textcoords="offset points", fontsize=8, color=INK2
        )
        ax.set_title(f"Loan started {pd.Timestamp(s + '-01'):%b %Y}", fontsize=9.5, color=INK, loc="left")
        grid(ax)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(3))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    axes[0].set_ylabel("Euros a month per 100,000 borrowed")
    axes[0].set_ylim(250, 620)
    axes[1].legend(loc="upper left", fontsize=8.5)
    frame(
        fig,
        "The variable payment moves once a year; in 2023 it jumped by up to two thirds",
        "Monthly payment of a 30-year loan of 100,000 euros,"
        " first ten years (or to October 2026). Margin 1 point on both.",
    )
    fig.savefig(FIG / "fig4_paths.png")
    plt.close(fig)


def fig5_premium_payout():
    d = pd.read_csv(RES / "cohorts.csv").set_index("start")
    d = d[d.index <= "2022-12"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for complete, color, label in [
        (True, BLUE, "Ten years complete (started 1999 to 2016)"),
        (False, ORANGE, "Compared to October 2026 (started 2016 to 2022)"),
    ]:
        g = d[d.complete == complete]
        ax.scatter(100 * g.rise, g.extra_fixed / 1000, s=18, color=color, alpha=0.8, lw=0, label=label)
    ax.axhline(0, color=INK2, lw=0.8)
    grid(ax, "both")
    ax.set_xlabel("Largest rise in the variable payment over the period (%)")
    ax.set_ylabel("Extra interest paid by fixing (thousand euros)")
    ax.text(68, 31, "Fixing cost more", ha="right", va="top", fontsize=8.5, color=INK2)
    ax.text(1, -12.5, "Fixing saved money", ha="left", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xlim(-2, 70)
    ax.set_ylim(-13, 32)
    ax.legend(loc="upper center", fontsize=8.5)
    frame(
        fig,
        "Fixing paid off mainly for the loans whose variable payment rose the most",
        "Each dot is a loan started in one month, 1999 to 2022: the premium of fixing against the shock it would have"
        " avoided. Per 100,000 borrowed.",
    )
    fig.savefig(FIG / "fig5_premium_payout.png")
    plt.close(fig)


def ecdf(ax, values, color, label, ls="-"):
    v = np.sort(np.asarray(values, float))
    y = np.arange(1, len(v) + 1) / len(v)
    ax.step(v, y, where="post", color=color, lw=2, ls=ls, label=label)


def fig6_new_loan():
    d = pd.read_csv(RES / "new_loan.csv")
    k = pd.read_csv(RES / "key_numbers.csv").set_index("name")["value"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"wspace": 0.25})
    for kind, color, label in [
        ("raw", BLUE, "Past errors as they were"),
        ("demeaned", ORANGE, "Past errors without their average"),
    ]:
        g = d[d.kind == kind]
        ecdf(axes[0], 100 * g.rise, color, label)
        ecdf(axes[1], g.extra_fixed / 1000, color, label)
    axes[0].set_xlabel("Largest rise in the variable payment, first ten years (%)")
    axes[1].set_xlabel("Extra interest paid by fixing (thousand euros)")
    for ax in axes:
        ax.set_ylabel("Share of scenarios at or below")
        grid(ax, "both")
    axes[1].axvline(0, color=INK2, lw=0.8)
    axes[0].legend(loc="lower right", fontsize=8.5)
    frame(
        fig,
        "A loan started in October 2026, replayed against the yield curve's past mistakes",
        f"Variable starts at {k['variable_rate_now']:.2f}%, fixed at {k['fixed_rate_now']:.2f}%. Each scenario adds to"
        " today's curve the errors it made for one start month, 1999 to 2017. Per 100,000 borrowed.",
    )
    fig.savefig(FIG / "fig6_new_loan.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig1_rates()
    fig2_cost()
    fig3_forecast()
    fig4_paths()
    fig5_premium_payout()
    fig6_new_loan()


if __name__ == "__main__":
    main()
