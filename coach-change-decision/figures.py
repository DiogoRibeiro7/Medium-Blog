"""Draw the article's figures from the tables coach_numbers.py writes to results/.

python figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

# chart ink and the first two categorical slots of the reference palette
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
    "Data: transfermarkt-datasets (CC0) and football-data.co.uk via datasets/football-datasets. Figure: Diogo Ribeiro."
)
AVERAGE = 1.375  # league-average points per match, printed by coach_numbers.py


def changes():
    """Number of changes in the comparison, from the table coach_numbers.py writes."""
    return len(pd.read_csv(RES / "changes.csv"))


def frame(fig, title, subtitle=None, note=None, top_in=0.75, bottom_in=0.75):
    """Title and subtitle above the plot, a note and the data credit below."""
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - top_in / h, bottom=bottom_in / h)
    fig.text(
        0.01,
        1 - 0.08 / h,
        title,
        fontsize=11,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    if subtitle:
        fig.text(0.01, 1 - 0.34 / h, subtitle, fontsize=8.5, color=INK2, ha="left", va="top")
    y = 0.06 / h
    fig.text(0.01, y, CREDIT, fontsize=7.2, color=MUTED, ha="left", va="bottom")
    if note:
        fig.text(0.01, y + 0.2 / h, note, fontsize=8, color=INK2, ha="left", va="bottom")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def path_axes(ax):
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.65, 2.02, "new manager from here", fontsize=8, color=INK2, va="top")
    ax.axhline(AVERAGE, color=GRID, linewidth=1.5)
    ax.text(-9.4, AVERAGE + 0.03, "league average", fontsize=8, color=MUTED, va="bottom")
    ax.set_xticks(range(-9, 11))
    ax.set_xticklabels([str(i) if i % 2 else "" for i in range(-9, 11)])
    ax.set_xlim(-9.6, 10.6)
    ax.set_ylim(0.15, 2.05)
    ax.set_xlabel("League matches before (negative) and after the decision; 0 is the outgoing manager's last")
    ax.set_ylabel("Points per match")
    grid(ax)


def fig1_bounce():
    p = pd.read_csv(RES / "event_paths.csv", index_col="match")
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    path_axes(ax)
    ax.plot(
        p.index,
        p.changed,
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=4.5,
        markeredgecolor="white",
        markeredgewidth=1,
    )
    ax.text(
        10.4,
        p.changed.iloc[-1] - 0.32,
        "clubs that\nchanged manager",
        fontsize=8.5,
        color=INK,
        ha="right",
    )
    frame(
        fig,
        "After a change of manager, results recover at once",
        f"Mean points per match around {changes()} in-season changes, big five leagues, 2012/13 to 2025/26",
        bottom_in=0.95,
    )
    fig.savefig(FIG / "fig1_the_bounce.png")
    plt.close(fig)


def fig2_shared():
    p = pd.read_csv(RES / "event_paths.csv", index_col="match")
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    path_axes(ax)
    ax.plot(
        p.index,
        p.kept,
        color=ORANGE,
        linewidth=2,
        marker="o",
        markersize=4.5,
        markeredgecolor="white",
        markeredgewidth=1,
        label="Same spot, kept the manager that week",
    )
    ax.plot(
        p.index,
        p.changed,
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=4.5,
        markeredgecolor="white",
        markeredgewidth=1,
        label="Changed the manager",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=8.5, handlelength=1.6)
    frame(
        fig,
        "Clubs in the same spot that kept their manager recovered almost as much",
        "Each change against the five nearest clubs at the same stage, with the same last-five points and last result",
        note="Matched on the latest result and on points over the last five and ten; what matters is matches 1 to 10.",
        bottom_in=1.15,
    )
    fig.savefig(FIG / "fig2_the_shared_bounce.png")
    plt.close(fig)


def fig3_who():
    r = pd.read_csv(RES / "change_rate_by_form.csv")
    order = ["0-1", "2-3", "4-5", "6-7", "8-9", "10-11", "12-15"]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for flag, color, label in [
        (1, BLUE, "In the bottom three"),
        (0, ORANGE, "Anywhere else"),
    ]:
        d = r[r.in_drop == flag].set_index("last5").reindex(order)
        d = d[d.n >= 30]
        x = [order.index(i) for i in d.index]
        ax.plot(
            x,
            100 * d.rate,
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=1,
            label=label,
        )
    ax.legend(loc="upper right", fontsize=8.5, title="Club's place in the table", title_fontsize=8.5)
    ax.set_xticks(range(len(order)), order)
    ax.set_xlabel("Points from the last five league matches")
    ax.set_ylabel("Chance of a new manager before\nthe next match (%)")
    ax.set_ylim(0, None)
    grid(ax)
    frame(
        fig,
        "Bad runs trigger changes, and the bottom three roughly doubles the chance",
        "Share of decision points followed by a change, 28,991 decision points with an established manager",
        note="Groups with fewer than 30 decision points are not drawn.",
        bottom_in=1.1,
    )
    fig.savefig(FIG / "fig3_who_gets_changed.png")
    plt.close(fig)


def interval_rows(ax, rows, colors, labels):
    """Horizontal estimate-and-interval rows, top to bottom."""
    n = len(rows)
    for i, (r, c) in enumerate(zip(rows.itertuples(), colors, strict=True)):
        yv = n - 1 - i
        if pd.notna(r.lo):
            ax.plot([r.lo, r.hi], [yv, yv], color=c, linewidth=2.2, solid_capstyle="round")
        ax.scatter(r.estimate, yv, s=46, color=c, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(
            "0.0" if abs(r.estimate) < 0.05 else f"{r.estimate:+.1f}",
            (r.hi if pd.notna(r.hi) else r.estimate, yv),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK,
        )
    ax.set_yticks(range(n - 1, -1, -1), labels)
    ax.axvline(0, color=MUTED, linewidth=1)
    grid(ax, "x")


def fig4_yardsticks():
    y = pd.read_csv(RES / "yardsticks.csv").set_index("yardstick").loc[["own", "stuck", "same_spot"]]
    labels = [
        "The club's own last five matches",
        "Clubs that kept their manager\nand stuck with him for ten matches",
        "Clubs in the same spot that kept\ntheir manager that week",
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    interval_rows(ax, y, [MUTED, MUTED, BLUE], labels)
    ax.set_xlabel("Points over the next ten matches: changed clubs minus the yardstick")
    ax.set_xlim(-1.6, 6.6)
    ax.set_ylim(-0.6, 2.6)
    frame(
        fig,
        f"The same {changes()} changes look like a triumph, a mistake or a small gain",
        "Depending on what they are compared with; dots are estimates, bars 95% intervals",
        note="Intervals resample club-seasons.",
        bottom_in=1.1,
    )
    fig.subplots_adjust(left=0.36)
    fig.savefig(FIG / "fig4_three_yardsticks.png")
    plt.close(fig)


def fig5_subgroups():
    s = pd.read_csv(RES / "subgroups.csv")
    labels = {
        "all changes": "All changes",
        "inside the bottom three": "Made inside the bottom three",
        "outside the bottom three": "Made outside it",
        "first half of the season": "First half of the season",
        "second half of the season": "Second half",
        "squad worth more than the league median": "Squad worth more than the league median",
        "squad worth less than the league median": "Squad worth less",
        "results below what shots on target predict": "Results below what shots on target predict",
        "results in line with shots on target": "Results in line",
    }
    colors = [BLUE] + [MUTED] * (len(s) - 1)
    for i, g in enumerate(s.group):
        if g == "squad worth more than the league median":
            colors[i] = ORANGE
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    interval_rows(ax, s, colors, [f"{labels[g]} ({n})" for g, n in zip(s.group, s.changes, strict=True)])
    ax.set_xlabel("Points over the next ten matches: changed clubs minus clubs in the same spot")
    ax.set_xlim(-1.3, 2.6)
    frame(
        fig,
        "The gain sits mostly with clubs that have expensive squads",
        "Same-spot estimate by the situation at the time of the change; number of changes in brackets",
        note="Eight groups, each at a 95% level: one of them could clear zero by chance alone.",
        bottom_in=1.1,
    )
    fig.subplots_adjust(left=0.47)
    fig.savefig(FIG / "fig5_where_it_differs.png")
    plt.close(fig)


def fig6_checks():
    c = pd.read_csv(RES / "checks.csv").set_index("check")
    s = pd.read_csv(RES / "sensitivity.csv").set_index("variant")
    rows = [
        ("Main estimate (five comparison clubs)", *c.loc["main"]),
        ("Linear model on all decision points", *c.loc["regression"]),
    ]
    for name in [
        "1 comparison club",
        "3 comparison clubs",
        "10 comparison clubs",
        "caliper 0.1 sd",
        "caliper 0.5 sd",
        "stage within one match",
        "distance matching only",
        "without quick moves to another club",
    ]:
        nice = {
            "caliper 0.1 sd": "Tighter propensity caliper",
            "caliper 0.5 sd": "Looser propensity caliper",
            "stage within one match": "Same stage within one match",
            "without quick moves to another club": "Without managers who left for another club",
            "distance matching only": "Distance matching only, no exact match on form",
        }.get(name, name.capitalize())
        rows.append((nice, s.loc[name, "estimate"], np.nan, np.nan))
    rows += [
        ("Placebo: a kept club treated as if it had changed", *c.loc["placebo"]),
        ("Matches 14 to 10 before, never matched on", *c.loc["pre_period"]),
    ]
    d = pd.DataFrame(rows, columns=["label", "estimate", "lo", "hi"])
    colors = [BLUE, BLUE] + [MUTED] * 8 + [ORANGE, ORANGE]
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    interval_rows(ax, d, colors, list(d.label))
    ax.axhspan(1.5, 9.5, color="#f0efec", zorder=0, linewidth=0)
    ax.set_xlabel("Points (over ten matches; the last row over five)")
    ax.set_xlim(-0.9, 1.4)
    frame(
        fig,
        "Small and positive under every matching choice; the checks find nothing",
        "Blue: the estimate two ways; grey: matching choices; orange: checks that should be near zero",
        note="Grey rows show the estimate only; their intervals are about as wide as the main one.",
        bottom_in=1.1,
    )
    fig.subplots_adjust(left=0.5)
    fig.savefig(FIG / "fig6_checks.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig1_bounce()
    fig2_shared()
    fig3_who()
    fig4_yardsticks()
    fig5_subgroups()
    fig6_checks()


if __name__ == "__main__":
    main()
