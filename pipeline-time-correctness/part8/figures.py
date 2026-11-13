"""Figures for Part 8.

Run: python figures.py  (after hub_numbers.py, which writes results/; a few seconds)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import hub as H  # noqa: E402

OUT = H.HERE / "figures"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
PURPLE = "#7b5cc4"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 11, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.linewidth": 2,
})
SERIES = (("publication", "published (the hub's truth)", ORANGE), ("reporting", "reporting date", BLUE),
          ("reference", "reference date", PURPLE), ("onset", "onset date (cases with a known onset)", AQUA))


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_four_datings() -> None:
    d = H.daily()
    g = d[d.location == "GM"].set_index("date")
    hol = H.holidays()["DE"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (a, b, title) in zip(axes, (("2021-03-21", "2021-04-17", "Easter 2021"),
                                        ("2022-04-03", "2022-04-30", "Easter 2022"))):
        x = g.loc[a:b]
        for day in x.index:
            if day.date() in hol:
                ax.axvspan(day - pd.Timedelta(hours=12), day + pd.Timedelta(hours=12), color=GRID, alpha=0.9, lw=0)
        for col, label, colour in SERIES:
            ax.plot(x.index, 100 * x[col] / x[col].mean(), color=colour, label=label,
                    lw=2.4 if col == "onset" else 1.8)
        ax.set_title(title)
        ax.set_xticks([x.index[k] for k in range(0, len(x), 7)],
                      [x.index[k].strftime("%a %d %b") for k in range(0, len(x), 7)])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("cases per day, % of the four-week mean")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    save(fig, "fig1_four_datings.png")


def fig2_who_shares_the_gap() -> None:
    hol = H.holidays()
    d = H.daily()
    m = H.jhu_mornings(d, hol, 1)
    groups = pd.Series(np.where(m.holiday, m.which.astype(str), m.index.to_period("W").astype(str)), index=m.index)
    rows = []
    for t, label in (("final", "Johns Hopkins'\nfinal version"), ("reporting", "RKI, reporting\ndate"),
                     ("reference", "RKI, reference\ndate")):
        e = m[f"e_{t}"]
        diff = e[m.holiday].mean() - e[~m.holiday].mean()
        lo, hi = H.cluster_boot(pd.concat([e, m.holiday], axis=1), groups,
                                lambda z: z.iloc[:, 0][z.iloc[:, 1]].mean() - z.iloc[:, 0][~z.iloc[:, 1]].mean())
        rows.append((label, diff, lo, hi))
    J = pd.read_csv(H.HERE / "results" / "national_weeks.csv", parse_dates=["week"])
    hz = []
    for h, g2 in J.groupby("horizon"):
        lo, hi = H.cluster_boot(g2.d_gap.reset_index(drop=True), g2.week.reset_index(drop=True), lambda z: z.mean())
        hz.append((f"{int(h)} week{'s' if h > 1 else ''}\nahead", -g2.d_gap.mean(), -hi, -lo))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"wspace": 0.3})
    for i, (label, v, lo, hi) in enumerate(rows):
        ax.plot([100 * lo, 100 * hi], [i, i], color=AXIS, lw=3, solid_capstyle="round", zorder=1)
        ax.scatter([100 * v], [i], s=60, color=BLUE if i else MUTED, zorder=3, edgecolor=SURFACE, linewidth=1.2)
        ax.text(100 * v, i + 0.22, f"{100 * v:+.1f}", ha="center", fontsize=9, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("holiday mornings minus other mornings, error in percentage points")
    ax.grid(axis="y", visible=False)
    ax.set_title("Johns Hopkins' morning 7-day count for Germany,\nscored against", fontsize=10.5)
    for i, (label, v, lo, hi) in enumerate(hz):
        bx.plot([i, i], [100 * lo, 100 * hi], color=AXIS, lw=3, solid_capstyle="round", zorder=1)
        bx.scatter([i], [100 * v], s=60, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=1.2)
        bx.text(i + 0.12, 100 * v, f"{100 * v:+.1f}", va="center", fontsize=9, color=INK2)
    bx.axhline(0, color=INK2, lw=0.8)
    bx.set_xticks(range(len(hz)), [h[0] for h in hz], fontsize=9)
    bx.set_xlim(-0.5, len(hz) - 0.3)
    bx.set_ylabel("percentage points")
    bx.grid(axis="x", visible=False)
    bx.set_title("Hub forecasts in national holiday weeks: how much worse\nthey look against publication than "
                 "against reference date", fontsize=10.5)
    save(fig, "fig2_who_shares_the_gap.png")


if __name__ == "__main__":
    fig1_four_datings()
    fig2_who_shares_the_gap()
