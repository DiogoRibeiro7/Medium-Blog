"""Draw the article's figures from the tables battle_numbers.py writes to results/.

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
BLUE, ORANGE = "#2a78d6", "#eb6834"
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
    "Data: CDB90 (US Army Concepts Analysis Agency, 1990), open version by J. B. Arnold (CC0). Figure: Diogo Ribeiro."
)
RATIOS = [0.25, 0.5, 1, 2, 4, 8, 16]


def frame(fig, title, subtitle=None, note=None, top_in=0.75, bottom_in=0.75):
    """Title and subtitle above the plot, a note and the data credit below."""
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - top_in / h, bottom=bottom_in / h)
    fig.text(0.01, 1 - 0.08 / h, title, fontsize=11, fontweight="bold", color=INK, ha="left", va="top")
    if subtitle:
        fig.text(0.01, 1 - 0.34 / h, subtitle, fontsize=8.5, color=INK2, ha="left", va="top")
    y = 0.06 / h
    fig.text(0.01, y, CREDIT, fontsize=7.2, color=MUTED, ha="left", va="bottom")
    if note:
        fig.text(0.01, y + 0.2 / h, note, fontsize=8, color=INK2, ha="left", va="bottom")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def ratio_axis(ax, lo=0.25, hi=16):
    ax.set_xscale("log")
    ax.set_xticks(RATIOS, ["1:4", "1:2", "1:1", "2:1", "4:1", "8:1", "16:1"])
    ax.set_xticks([], minor=True)
    ax.set_xlim(lo, hi)


def fig1_who_wins():
    w = pd.read_csv(RES / "win_by_ratio.csv")
    c = pd.read_csv(RES / "win_curve.csv")
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ratio_axis(ax, 0.25, 17)
    x = np.exp(c.lr)
    ax.plot(x, 100 * (x > 1), color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(
        1.12,
        93,
        "Lanchester's square law, equal quality:\nthe larger side always wins",
        fontsize=8,
        color=INK2,
        va="top",
    )
    ax.plot(x, 100 * c.p, color=BLUE, linewidth=2)
    xm = np.exp(w.lr_median)
    ax.vlines(xm, 100 * w.w_lo, 100 * w.w_hi, color=BLUE, linewidth=1.6, alpha=0.55)
    ax.scatter(xm, 100 * w.win, s=40, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
    ax.text(
        5.5,
        100 * c.p[(x > 5.5).idxmax()] - 9,
        "fitted curve and\nbattles grouped by ratio",
        fontsize=8.5,
        color=INK,
        ha="left",
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Battles won by the attacker (%)")
    ax.set_xlabel("Attacker's strength : defender's strength (log scale)")
    grid(ax)
    frame(
        fig,
        "More men help, but far less than Lanchester's square law implies",
        "614 decisive land battles, 1600 to 1982; dots: groups by force ratio with 95% intervals",
        note="The attacker wins 60% of battles at equal numbers and 68% with twice the men.",
        bottom_in=1.1,
    )
    fig.savefig(FIG / "fig1_who_wins.png")
    plt.close(fig)


def fig2_casualties():
    c = pd.read_csv(RES / "casualties.csv")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ratio_axis(ax, 0.2, 20)
    x = np.exp(c.lr)
    ax.scatter(x, np.exp(c.llr), s=10, color=BLUE, alpha=0.35, edgecolor="none")
    grid_x = np.exp(np.linspace(np.log(0.2), np.log(20), 50))
    b, a = np.polyfit(c.lr, c.llr, 1)
    ax.plot(grid_x, np.exp(a + b * np.log(grid_x)), color=BLUE, linewidth=2.2)
    for slope, lab in ((-1, "square law"), (0, "linear law")):
        y = np.exp(a + b * c.lr.mean() + slope * (np.log(grid_x) - c.lr.mean()))
        ax.plot(grid_x, y, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.text(15.5, y[-1] * (0.55 if slope == 0 else 1.3), lab, fontsize=8, color=INK2, ha="right")
    ax.text(15.5, np.exp(a + b * np.log(15.5)) * 2.0, "fitted", fontsize=8.5, color=INK, ha="right")
    ax.set_yscale("log")
    ax.set_yticks([0.01, 0.1, 1, 10, 100], ["1:100", "1:10", "1:1", "10:1", "100:1"])
    ax.set_yticks([], minor=True)
    ax.set_ylim(0.005, 200)
    ax.set_ylabel("Attacker's losses : defender's losses")
    ax.set_xlabel("Attacker's strength : defender's strength (log scale)")
    grid(ax)
    frame(
        fig,
        "Bigger attacking armies do not lose fewer men relative to the enemy",
        "654 battles with strengths and losses on both sides; dashed lines: what each law predicts, through the data",
        note="The square law says losses fall in proportion to the force ratio; they rise slightly instead.",
        bottom_in=1.1,
    )
    fig.savefig(FIG / "fig2_losses.png")
    plt.close(fig)


def fig3_what_explains():
    p = pd.read_csv(RES / "predictors.csv").set_index("model")
    rows = [
        ("Numbers", "numbers", BLUE),
        ("Numbers + terrain", "numbers + terrain", BLUE),
        ("Numbers + defender's posture", "numbers + posture", BLUE),
        ("Numbers + commanders' earlier record", "numbers + prior record", BLUE),
        ("Numbers + surprise", "numbers + surprise", BLUE),
        ("All of the above", "numbers + surprise + posture + terrain + record", BLUE),
        ("Numbers + coded leadership", "numbers + coded leadership", ORANGE),
        ("Numbers + all ten coded advantages", "numbers + all ten coded advantages", ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    n = len(rows)
    for i, (_, key, col) in enumerate(rows):
        v = 100 * p.loc[key, "mz_r2"]
        y = n - 1 - i
        ax.barh(y, v, height=0.62, color=col, edgecolor="white", linewidth=2)
        ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(n - 1, -1, -1), [r[0] for r in rows])
    ax.set_xlim(0, 90)
    ax.set_xlabel("Share of the variation in who won that the model explains (%)")
    grid(ax, "x")
    ax.text(48, 3.2, "known or counted\nbefore the battle", color=BLUE, fontsize=8.5, va="center")
    ax.text(62, 1.0, "coded by historians\nwho knew the result", color=ORANGE, fontsize=8.5, va="center")
    frame(
        fig,
        "Surprise and earlier records explain more than numbers; the codes explain most",
        "Logistic models of who won, 578 decisive battles with every variable known",
        note="Share of variation: McKelvey-Zavoina R-squared (variance of the fitted latent outcome over its total).",
        bottom_in=1.1,
    )
    fig.subplots_adjust(left=0.36)
    fig.savefig(FIG / "fig3_what_explains.png")
    plt.close(fig)


def fig4_hindsight():
    t = pd.read_csv(RES / "commanders.csv")
    t = t[(t.lead_win != 0) | (t.lead_loss != 0)].sort_values("battles", ascending=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for i, r in enumerate(t.itertuples()):
        ax.plot([r.lead_loss, r.lead_win], [i, i], color=GRID, linewidth=2.4, solid_capstyle="round", zorder=1)
        ax.scatter(r.lead_loss, i, s=42, color=ORANGE, edgecolor="white", linewidth=1.2, zorder=3)
        ax.scatter(r.lead_win, i, s=42, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
    ax.set_yticks(range(len(t)), [f"{n} ({int(b)})" for n, b in zip(t.name, t.battles, strict=True)])
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Coded leadership advantage of the commander's side (−3 to +3)")
    ax.set_xlim(-2, 3)
    grid(ax, "x")
    ax.scatter([], [], s=42, color=BLUE, label="mean in his victories")
    ax.scatter([], [], s=42, color=ORANGE, label="mean in his defeats")
    ax.legend(loc="upper right", fontsize=8.5)
    frame(
        fig,
        "The same commander's leadership is coded higher in the battles he won",
        "Commanders with at least two victories and two defeats in CDB90 and a non-zero code; battles in brackets",
        note="Ten more commanders with mixed records have a code of zero in every battle and are not drawn.",
        bottom_in=1.1,
    )
    fig.subplots_adjust(left=0.36)
    fig.savefig(FIG / "fig4_coded_leadership.png")
    plt.close(fig)


def fig5_eras():
    e = pd.read_csv(RES / "eras.csv")
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    x = np.arange(len(e))
    ax.axhline(50, color=MUTED, linewidth=1)
    ax.text(-0.45, 51, "a coin toss", fontsize=8, color=INK2, ha="left", va="bottom")
    ax.vlines(x, 100 * e.larger_lo, 100 * e.larger_hi, color=BLUE, linewidth=2.2)
    ax.scatter(x, 100 * e.larger_won, s=60, color=BLUE, edgecolor="white", linewidth=1.4, zorder=3)
    for xi, r in zip(x, e.itertuples(), strict=True):
        ax.text(xi + 0.08, 100 * r.larger_won, f"{100 * r.larger_won:.0f}%", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [f"{r.era}\n({r.n} battles)" for r in e.itertuples()])
    ax.set_ylim(25, 85)
    ax.set_xlim(-0.5, len(e) - 0.4)
    ax.set_ylabel("Won by the larger side (%)")
    grid(ax)
    frame(
        fig,
        "The larger army usually won only in the era of the world wars",
        "Decisive battles with unequal numbers, by era; bars: 95% intervals resampling campaigns",
        bottom_in=0.85,
    )
    fig.savefig(FIG / "fig5_eras.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig1_who_wins()
    fig2_casualties()
    fig3_what_explains()
    fig4_hindsight()
    fig5_eras()


if __name__ == "__main__":
    main()
