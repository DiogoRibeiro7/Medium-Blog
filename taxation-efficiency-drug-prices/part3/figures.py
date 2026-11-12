"""Figures for Part 3. Reads figure_data.json written by reference_numbers.py.

    python figures.py

Writes two PNG files to figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COUNTRY = {"ES": ("Spain", BLUE, "o"), "IT": ("Italy", ORANGE, "s"), "BE": ("Belgium", AQUA, "^")}


def setup() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = "Inter" if "Inter" in names else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family,
        "font.size": 10,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
    })


def note(fig, text: str) -> None:
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def times(v, _pos=None) -> str:
    return f"{v:g}×"


def euros(v, _pos=None) -> str:
    return f"€{v:,.0f}" if v >= 1 else f"€{v:g}"


def fig_reference_prices(data: dict) -> None:
    rows = data["molecules"]
    rows = sorted(rows, key=lambda r: r["cap"])
    n = len(rows)
    fig, ax = plt.subplots(figsize=(8.4, 0.36 * n + 1.9))
    y = np.arange(n)
    ax.axvline(1.0, color=INK2, lw=1.0, zorder=1)
    for i, r in enumerate(rows):
        vals = [r[k] for k in COUNTRY if r[k] is not None] + [1.0]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=2.0, zorder=1, solid_capstyle="round")
    for k, (name, color, marker) in COUNTRY.items():
        xs = [r[k] for r in rows]
        ax.scatter(xs, y, s=46, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.2, zorder=3, label=name)
    ax.scatter([r["cap"] for r in rows], y, s=150, marker="|", color=INK, linewidth=2.4, zorder=4,
               label="Portuguese ceiling (average of the four)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['brand'].capitalize()}  {r['pack']}" for r in rows], fontsize=8.5)
    ax.set_xscale("log")
    ticks = [0.25, 0.5, 0.75, 1, 1.5, 2, 3]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(times))
    lo = min(min(r[k] for k in COUNTRY if r[k] is not None) for r in rows)
    hi = max(max(r[k] for k in COUNTRY if r[k] is not None) for r in rows)
    ax.set_xlim(min(0.22, lo * 0.9), max(3.2, hi * 1.1))
    ax.set_ylim(-0.7, n - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Ex-factory price of the same presentation, relative to France (France = 1×)")
    ax.set_title("Reference prices for the brands that all four countries list")
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.55, 0.025), ncol=4, fontsize=8.5,
               handletextpad=0.3, columnspacing=1.2)
    note(fig, "Most sold presentation of each molecule in France. Sources: AIFA, Ministerio de Sanidad, SAM (Belgium), "
              "BDPM and Open Medic (France).")
    fig.savefig(OUT / "fig1_reference_prices.png", dpi=200)
    plt.close(fig)


def fig_check(data: dict) -> None:
    pts = data["check"]
    cap = np.array([p["cap"] for p in pts])
    pva = np.array([p["pva"] for p in pts])
    pvp = np.array([p["pvp"] for p in pts])
    exposed = pvp > 30
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    lo, hi = 0.8, max(cap.max(), pva.max()) * 1.3
    xs = np.array([lo, hi])
    ax.fill_between(xs, xs * 0.8, xs * 1.25, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.plot(xs, xs, color=INK2, lw=1.0, zorder=1)
    ax.scatter(cap[~exposed], pva[~exposed], s=24, color=ORANGE, alpha=0.85, edgecolor=SURFACE, linewidth=0.8,
               zorder=2, label="retail price 30 euros or less (left alone by the 2026 revision)")
    ax.scatter(cap[exposed], pva[exposed], s=30, color=BLUE, alpha=0.9, edgecolor=SURFACE, linewidth=0.8,
               zorder=3, label="retail price above 30 euros")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator([t for t in ticks if t <= hi]))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(FuncFormatter(euros))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Ceiling rebuilt from the four reference countries (ex-factory, euros per pack)")
    ax.set_ylabel("Portugal's maximum price (ex-factory, euros per pack)")
    ax.set_title("Original brands in Infarmed's generic-group list against the rebuilt rule")
    ax.text(1.25, 1.25 * 1.12, "Portugal's price = rebuilt rule", color=INK2, fontsize=8.5, ha="left", va="bottom",
            rotation=45, rotation_mode="anchor")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    note(fig, "Grey band: from 20 percent below to 25 percent above the rule. Third-quarter 2026 list; reference prices of mid to late 2026.")
    fig.savefig(OUT / "fig2_check_against_portugal.png", dpi=200)
    plt.close(fig)


def main() -> None:
    setup()
    OUT.mkdir(exist_ok=True)
    data = json.loads((HERE / "figure_data.json").read_text())
    fig_reference_prices(data)
    fig_check(data)


if __name__ == "__main__":
    main()
