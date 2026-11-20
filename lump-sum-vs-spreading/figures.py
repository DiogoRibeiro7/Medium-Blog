"""Draw the article's figures from the tables lump_numbers.py writes to results/.

python figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lump_core as lc  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LIGHT_BLUE, MID_BLUE = "#cde2fb", "#86b6ef"
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
CREDIT = "Data: OECD share price indices and interest rates, monthly. Code: Medium-Blog/lump-sum-vs-spreading."
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


def percent(ax, axis="y", digits=0):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.{digits}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def fig1_countries():
    c = pd.read_csv(RES / "by_country.csv").sort_values("lump_wins")
    y = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(10, 7.4))
    for yi, (_, r) in zip(y, c.iterrows(), strict=True):
        ax.plot([r.lump_wins, r.lump_wins_div3], [yi, yi], color=GRID, lw=3, solid_capstyle="round")
    ax.scatter(c.lump_wins, y, s=60, color=BLUE, zorder=3, ec="white", label="Share prices only (as published)")
    ax.scatter(
        c.lump_wins_div3,
        y,
        s=60,
        color=ORANGE,
        zorder=3,
        ec="white",
        label="With an assumed dividend yield of 3% a year",
    )
    ax.axvline(0.5, color=INK2, lw=0.9)
    ax.text(0.502, len(c) - 0.4, "half of start months", fontsize=8, color=INK2, va="bottom")
    ax.set_yticks(y, [lc.NAMES[k] for k in c.country], fontsize=9)
    ax.set_xlim(0.4, 0.8)
    ax.set_ylim(-0.7, len(c) + 0.2)
    percent(ax, "x")
    grid(ax, "x")
    ax.set_xlabel("Start months in which the lump sum was ahead one year after the last instalment")
    ax.legend(loc="lower right", fontsize=8.5)
    frame(
        fig,
        "Investing at once beat 12 monthly instalments in more than half of start months, in 22 of 23 markets",
        "Every start month from the 1950s (or the first month of data) to October 2025. Cash waiting to be invested"
        " earns the country's\n3-month interbank rate (overnight rate where the 3-month rate is missing).",
        top_in=0.9,
    )
    fig.savefig(FIG / "fig1_countries.png")
    plt.close(fig)


def fig2_advantage():
    a = pd.read_csv(RES / "advantage.csv").advantage
    bins = np.arange(-0.5, 0.52, 0.02)
    clipped = a.clip(-0.499, 0.499)
    fig, ax = plt.subplots(figsize=(10, 5.0))
    win = clipped[clipped > 0]
    lose = clipped[clipped <= 0]
    ax.hist(
        lose,
        bins=bins,
        color=ORANGE,
        ec="white",
        lw=0.6,
        label=f"Spreading did better ({100 * len(lose) / len(a):.0f}%)",
    )
    ax.hist(
        win, bins=bins, color=BLUE, ec="white", lw=0.6, label=f"Lump sum did better ({100 * len(win) / len(a):.0f}%)"
    )
    ax.axvline(a.median(), color=INK, lw=1.2)
    ax.text(
        a.median() + 0.012,
        ax.get_ylim()[1] * 0.80,
        f"median lead\n{100 * a.median():.1f}%",
        fontsize=8.5,
        color=INK,
        va="top",
    )
    grid(ax)
    percent(ax, "x")
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel("Lump sum's lead over 12 monthly instalments, one year after the last instalment (clipped at ±50%)")
    ax.set_ylabel("Start months")
    ax.legend(loc="upper left", fontsize=8.5)
    frame(
        fig,
        "Most years the gap is small and in favour of the lump sum; the large gaps go both ways",
        "13,241 start months in 23 markets, share prices without dividends. A positive value means the lump sum ended"
        " ahead by that share.",
    )
    fig.savefig(FIG / "fig2_advantage.png")
    plt.close(fig)


def fig3_outcomes():
    q = pd.read_csv(RES / "outcome_cdf.csv")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(q.lump, q.q, color=BLUE, lw=2.2, label="All at once")
    ax.plot(q.spread, q.q, color=ORANGE, lw=2.2, label="12 monthly instalments")
    ax.plot(q.half, q.q, color=MUTED, lw=1.6, label="Half at once, half in 12 instalments")
    ax.axvline(1, color=INK2, lw=0.9)
    a = pd.read_csv(RES / "advantage.csv")
    below_l = (a.lump < 0.8).mean()
    below_s = (a.spread < 0.8).mean()
    ax.scatter([0.8, 0.8], [below_l, below_s], s=30, color=[BLUE, ORANGE], zorder=4)
    ax.text(
        0.52,
        0.62,
        f"Worth less than 0.8 after a year:\n{100 * below_l:.1f}% of starts all at once,\n"
        f"{100 * below_s:.1f}% with 12 instalments",
        fontsize=8.5,
        color=INK2,
        va="top",
    )
    grid(ax)
    grid(ax, "x")
    ax.set_xlim(0.5, 1.7)
    ax.set_ylim(0, 1)
    percent(ax)
    ax.set_xlabel("Value one year after the last instalment, windfall = 1")
    ax.set_ylabel("Share of start months with a value below this")
    ax.legend(loc="lower right", fontsize=8.5)
    frame(
        fig,
        "What spreading buys: fewer bad years, paid for with fewer good ones",
        "Distribution of the windfall's value one year after the last instalment, 13,241 start months in 23 markets,"
        " share prices\nwithout dividends. A curve further left at the bottom means more bad outcomes.",
    )
    fig.savefig(FIG / "fig3_outcomes.png")
    plt.close(fig)


def fig4_lengths():
    t = pd.read_csv(RES / "lengths.csv")
    fig, ax = plt.subplots(figsize=(10, 5.0))
    gain = t.p5_spread - t.p5_lump
    ax.plot(t.median_advantage, gain, color=MID_BLUE, lw=1.4, zorder=2)
    ax.scatter(t.median_advantage, gain, s=70, color=BLUE, zorder=3, ec="white")
    for _, r in t.iterrows():
        ax.annotate(
            f"{int(r.months)} months\nlump sum ahead in {100 * r.lump_wins:.0f}%",
            (r.median_advantage, r.p5_spread - r.p5_lump),
            xytext=(-60, 12) if r.months == 36 else (8, -18),
            textcoords="offset points",
            fontsize=8,
            color=INK2,
        )
    grid(ax)
    grid(ax, "x")
    ax.set_yticks(np.arange(0, 0.225, 0.05))
    percent(ax, "x", 1)
    percent(ax)
    ax.set_xlim(0, 0.032)
    ax.set_ylim(0, 0.24)
    ax.set_xlabel("Median cost of spreading: the lump sum's median lead")
    ax.set_ylabel("Protection: the 5th-percentile outcome,\nspread minus lump sum (windfall = 100%)")
    frame(
        fig,
        "Longer spreading costs more, and its protection stops growing after about two years",
        "Each point is one length of spreading, from 3 to 36 monthly instalments, over all start months in 23 markets"
        " (share prices without dividends).",
    )
    fig.savefig(FIG / "fig4_lengths.png")
    plt.close(fig)


def fig5_conditions():
    t = pd.read_csv(RES / "conditions.csv")
    cash = t[t.kind == "cash"].reset_index(drop=True)
    cond = t[t.kind == "condition"].reset_index(drop=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5.0), gridspec_kw={"width_ratios": [5, 3]}, sharey=True)
    w = 0.38
    for ax, part, title in ((a1, cash, "Cash rate in the start month"), (a2, cond, "Stock index in the start month")):
        x = np.arange(len(part))
        ax.bar(x - w / 2, part.lump_wins, w, color=BLUE, label="Share prices only")
        ax.bar(x + w / 2, part.lump_wins_div3, w, color=ORANGE, label="With 3% dividends")
        for xi, v in zip(x, part.lump_wins, strict=True):
            ax.text(xi - w / 2, v + 0.01, f"{100 * v:.0f}%", ha="center", fontsize=7.5, color=INK2)
        for xi, v in zip(x, part.lump_wins_div3, strict=True):
            ax.text(xi + w / 2, v + 0.01, f"{100 * v:.0f}%", ha="center", fontsize=7.5, color=INK2)
        labels = [
            g.replace("20% or more below the high", "20%+ below\nits high")
            .replace("at an all-time high", "at an\nall-time high")
            .replace("anywhere else", "anywhere\nelse")
            for g in part.group
        ]
        ax.set_xticks(x, labels, fontsize=8.5)
        ax.axhline(0.5, color=INK2, lw=0.9)
        ax.set_title(title, fontsize=9.5, color=INK2, loc="left")
        grid(ax)
    a1.set_ylim(0, 0.8)
    percent(a1)
    a1.set_ylabel("Start months in which the lump sum was ahead")
    a1.legend(loc="upper right", fontsize=8.5)
    frame(
        fig,
        "High cash rates narrowed the lump sum's edge; an all-time high was no reason to wait",
        "Share of start months in which a lump sum beat 12 monthly instalments, by the cash rate and by where the stock"
        " index stood\nin the start month. 13,241 start months in 23 markets.",
        top_in=1.15,
    )
    fig.savefig(FIG / "fig5_conditions.png")
    plt.close(fig)


def fig6_us():
    u = pd.read_csv(RES / "us_total.csv")
    when = pd.to_datetime(u.start)
    fig, ax = plt.subplots(figsize=(10, 5.0))
    ax.fill_between(when, 0, u.advantage, where=u.advantage > 0, color=MID_BLUE, lw=0, label="Lump sum ahead")
    ax.fill_between(when, 0, u.advantage, where=u.advantage <= 0, color=ORANGE, lw=0, label="Spreading ahead")
    ax.axhline(0, color=INK2, lw=0.9)
    for date, text in (("1973-01", "1973-74"), ("1987-08", "1987"), ("2000-08", "2000-02"), ("2007-10", "2008")):
        row = u[u.start == date]
        if len(row):
            ax.annotate(
                text,
                (pd.to_datetime(date), float(row.advantage.iloc[0])),
                xytext=(0, -14),
                textcoords="offset points",
                fontsize=8,
                color=INK2,
                ha="center",
            )
    grid(ax)
    percent(ax)
    ax.set_ylim(-0.45, 0.45)
    ax.set_xlabel("Start month")
    ax.set_ylabel("Lump sum's lead over 12 instalments")
    ax.legend(loc="upper left", fontsize=8.5)
    frame(
        fig,
        "The United States with dividends: the lump sum led in two start months out of three",
        "S&P composite with dividends reinvested (Shiller), cash at the US 3-month or overnight rate, start months July"
        " 1954 to June 2022.",
        credit="Data: R. Shiller (S&P composite and dividends, via datasets/s-and-p-500), OECD (interest rates)."
        " Code: Medium-Blog/lump-sum-vs-spreading. Figure: Diogo Ribeiro.",
    )
    fig.savefig(FIG / "fig6_us.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig1_countries()
    fig2_advantage()
    fig3_outcomes()
    fig4_lengths()
    fig5_conditions()
    fig6_us()


if __name__ == "__main__":
    main()
