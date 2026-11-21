"""The article's figures, from the sources and the results.

python figures.py --sources sources --results results --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from staff import comb, data  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY = "#a9a8a3"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 12,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.titlesize": 13,
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
    }
)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_profiles(needs: dict, out: Path) -> None:
    """Mean daily requirement with the comb teeth for 8- and 12-hour tours and the steepest climb."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    for ax, d, unit in zip(axes, ("ems_city", "nyc311"), ("ambulances", "agents"), strict=True):
        m = needs[d].reshape(-1, 24).mean(axis=0)
        hours = np.arange(24)
        ax.plot(hours, m, color=INK, lw=2, zorder=3)
        top = m.max() * 1.18
        for L, col, y in ((8, BLUE, 1.10), (12, ORANGE, 1.04)):
            phi = comb.phase_by_comb(m, L)
            teeth = np.arange(phi, 24, L)
            for t in teeth:
                ax.axvline(t, color=col, lw=1.6, alpha=0.9, zorder=2)
            ax.text(teeth[0] + 0.25, m.max() * y, f"{L}-hour teeth", color=col, fontsize=10, va="bottom")
        climb = int(np.argmax(m - np.roll(m, 1)))
        ax.annotate(
            "steepest climb",
            (climb, m[climb]),
            xytext=(climb - 7.5, m[climb] * 0.92),
            fontsize=10,
            color=INK2,
            arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8},
        )
        ax.set_ylim(0, top)
        ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
        ax.set_xlabel("Hour of day")
        ax.set_ylabel(f"Mean requirement ({unit})")
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(data.LABELS[d])
    save(fig, out, "fig1_profiles_and_teeth.png")


def fig_binding(results: Path, out: Path) -> None:
    """Share of days on which each clock hour binds, by tour length: EMS citywide and 311."""
    share = pd.read_csv(results / "binding_share.csv")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f2f6fc", "#9ec5f4", "#2a78d6", "#103a75"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9))
    for ax, d in zip(axes, ("ems_city", "nyc311"), strict=True):
        grid = np.array(
            [share[(share.dataset == d) & (share.L == L)].sort_values("hour")["all"].to_numpy() for L in data.LENGTHS]
        )
        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_yticks(range(len(data.LENGTHS)), [f"{L}-hour" for L in data.LENGTHS])
        ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
        ax.set_xlabel("Hour of day")
        ax.set_title(data.LABELS[d])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Share of days the hour binds")
    cb.outline.set_visible(False)
    save(fig, out, "fig2_binding_share.png")


def fig_paid(results: Path, out: Path) -> None:
    """Paid hours of the cheapest roster relative to 8-hour tours."""
    cells = pd.read_csv(results / "cells.csv")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = list(data.LENGTHS)
    for d in data.DATASETS:
        cd = cells[cells.dataset == d].groupby("L")["ip"].sum()
        rel = [cd[L] / cd[8] for L in x]
        if d in ("ems_city", "nyc311"):
            col = BLUE if d == "ems_city" else ORANGE
            ax.plot(x, rel, "o-", color=col, lw=2, ms=7, zorder=3, label=data.LABELS[d])
            ax.annotate(
                f"{rel[-1]:.2f}",
                (x[-1], rel[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
                color=INK,
            )
        else:
            ax.plot(x, rel, "-", color=GREY, lw=1, zorder=2)
    ax.plot([], [], "-", color=GREY, lw=1, label="EMS, each borough")
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_xticks(x, [f"{L} h" for L in x])
    ax.set_xlabel("Tour length")
    ax.set_ylabel("Paid hours, relative to 8-hour tours")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Paid hours of the cheapest roster with perfect knowledge of the workload")
    save(fig, out, "fig3_paid_hours.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", type=Path, default=Path("sources"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    needs = data.requirements(a.sources)
    fig_profiles(needs, a.out)
    fig_binding(a.results, a.out)
    fig_paid(a.results, a.out)


if __name__ == "__main__":
    main()
