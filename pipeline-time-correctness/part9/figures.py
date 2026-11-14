"""Figures for Part 9.

Run: python figures.py  (after calendar_numbers.py, which writes results/; about half a minute)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import learned as L  # noqa: E402

OUT = L.HERE / "figures"
RES = L.HERE / "results"
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


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_fingerprints(df: pd.DataFrame) -> None:
    """Each producer's file, day by day: unchanged, changed, or not counted, with the week on the x axis and
    the day of the week on the y axis."""
    doc = pd.read_csv(L.DATA / "documented_changes.csv", parse_dates=["first_day"]).set_index("producer")
    show = (("Spain", "Spain"), ("Florida", "Florida"), ("Ohio", "Ohio"), ("Belgium", "Belgium"))
    monday0 = pd.Timestamp("2020-03-23")
    weeks = (pd.Timestamp("2023-03-09") - monday0).days // 7 + 1
    cmap = ListedColormap([SURFACE, GRID, BLUE])
    fig, axes = plt.subplots(len(show), 1, figsize=(12.5, 7.4), sharex=True, gridspec_kw={"hspace": 0.55})
    for ax, (name, title) in zip(axes, show):
        g = df[df.producer == name]
        img = np.zeros((7, weeks))
        k = ((g.date - monday0).dt.days // 7).to_numpy()
        ok = (k >= 0) & (k < weeks)
        val = np.where(g.eligible, np.where(g.flat == 1, 2, 1), 0)
        img[g.wd.to_numpy()[ok], k[ok]] = val[ok]
        ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest",
                  extent=(-0.5, weeks - 0.5, 6.5, -0.5))
        ax.set_yticks(range(7), ["Mon", "", "Wed", "", "Fri", "", "Sun"], fontsize=8)
        ax.grid(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=10.5)
        if name in doc.index:
            x = (doc.loc[name, "first_day"] - monday0).days / 7
            ax.plot([x, x], [-0.5, 6.5], color=ORANGE, lw=1.6)
            ax.annotate("announced change", (x, -0.5), xytext=(4, 3), textcoords="offset points",
                        color=INK2, fontsize=8, ha="left", va="bottom", annotation_clip=False)
    ticks = [d for d in pd.date_range("2020-04-01", "2023-03-01", freq="6MS")]
    axes[-1].set_xticks([(d - monday0).days / 7 for d in ticks], [d.strftime("%b %Y") for d in ticks])
    from matplotlib.patches import Patch
    handles = [Patch(color=BLUE, label="file unchanged from the day before"),
               Patch(color=GRID, label="file changed"),
               Patch(facecolor=SURFACE, edgecolor=AXIS, label="not counted (too few cases, or after the last change)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    save(fig, "fig1_fingerprints.png")


def fig2_memory() -> None:
    est = pd.read_csv(RES / "estimators.csv")
    fams = (("window", "sliding window, by day of the week", BLUE, "o"),
            ("half-life", "exponential forgetting, by day of the week", AQUA, "s"),
            ("change points by day of the week", "change points, each day of the week on its own", PURPLE, "^"),
            ("change points shared by the producer's days", "change points shared by the producer's seven days",
             ORANGE, "D"))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for fam, label, colour, marker in fams:
        g = est[est.family == fam]
        ax.plot(g.days, g.logloss, color=colour, marker=marker, ms=5, label=label)
    ax.set_xscale("log")
    ax.set_xticks([7, 14, 28, 56, 91, 182, 365, 730, 1456], ["7", "14", "28", "56", "91", "182", "365", "730", "1456"])
    ax.minorticks_off()
    ax.set_xlabel("memory, in days (window length; half-life; expected time between change points)")
    ax.set_ylabel("mean log loss, test period (lower is better)")
    ax.legend(loc="upper center", fontsize=9, ncol=1)
    ax.set_ylim(0.2, 0.5)
    save(fig, "fig2_memory.png")


def fig3_profile() -> None:
    obs = np.loadtxt(RES / "profile_observed.csv", delimiter=",")
    k = np.arange(-12, 13)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    sims = ((13, "simulated 13-week ramp", PURPLE), (8, "simulated 8-week ramp", AQUA),
            (4, "simulated 4-week ramp", ORANGE), (0, "simulated step", MUTED))
    for rw, label, colour in sims:
        sp = np.loadtxt(RES / f"profile_ramp{rw}.csv", delimiter=",")
        m = np.nanmean(sp, 0)
        ax.plot(k, 100 * m, color=colour, lw=1.4, ls="--" if rw == 0 else "-", label=label)
    m = np.nanmean(obs, 0)
    ax.plot(k, 100 * m, color=BLUE, lw=2.6, marker="o", ms=4, label=f"observed ({len(obs)} changes)")
    ax.axvspan(-1.5, 0.5, color=GRID, alpha=0.6, lw=0)
    ax.annotate("weeks -1 and 0 are\nfixed by the method", (-0.5, 50), ha="center", va="center", fontsize=8,
                color=INK2)
    ax.set_xticks(range(-12, 13, 2))
    ax.set_xlabel("weeks from the change (week 0 starts on the first day the new schedule shows)")
    ax.set_ylabel("files unchanged on the days that changed, %")
    ax.set_ylim(-3, 103)
    ax.legend(loc="center left", fontsize=9, bbox_to_anchor=(0.0, 0.62))
    save(fig, "fig3_profile.png")


def main() -> None:
    df = L.frame()
    fig1_fingerprints(df)
    fig2_memory()
    fig3_profile()


if __name__ == "__main__":
    main()
