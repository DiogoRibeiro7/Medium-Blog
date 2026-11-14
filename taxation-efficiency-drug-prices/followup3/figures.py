"""Figures for follow-up 3. Reads figure_data.json written by us_side_numbers.py.

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
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"

NAMES = {"humira": "Humira", "januvia": "Januvia", "enbrel": "Enbrel", "trulicity": "Trulicity",
         "imbruvica": "Imbruvica", "victoza": "Victoza", "restasis": "Restasis", "keytruda": "Keytruda",
         "stelara": "Stelara", "remicade": "Remicade", "pneumovax23": "Pneumovax 23", "jardiance": "Jardiance",
         "farxiga": "Farxiga", "humalog": "Humalog", "novolog": "NovoLog", "symbicort": "Symbicort",
         "otezla": "Otezla", "prezista": "Prezista family", "tysabri": "Tysabri", "botox": "Botox",
         "xarelto": "Xarelto", "aranesp": "Aranesp"}


def setup() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = "Inter" if "Inter" in names else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 10,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "lines.linewidth": 2.0,
    })


def title(fig, main: str, sub: str) -> None:
    fig.text(0.01, 0.975, main, ha="left", va="top", fontsize=11.5, fontweight="semibold", color=INK)
    fig.text(0.01, 0.925, sub, ha="left", va="top", fontsize=9.5, color=INK2)


def note(fig, text: str) -> None:
    fig.text(0.01, 0.012, text, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def fig_line(data: dict) -> None:
    line = {int(k): v for k, v in data["line"].items()}
    xs = sorted(k for k in line if 3 <= k <= 20)
    net = [line[x][0] for x in xs]
    us = [line[x][1] for x in xs]
    ab = [line[x][2] for x in xs]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.08, right=0.82, top=0.8, bottom=0.18)
    ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
    for x0, lab in [(5, "the line used in follow-up 2"), (10, "ten years")]:
        ax.axvline(x0, color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.text(x0 + 0.15, 8.6, lab, color=MUTED, fontsize=8, va="top")
    ax.plot(xs, us, color=BLUE, marker="o", ms=4.5, zorder=3)
    ax.plot(xs, ab, color=ORANGE, marker="o", ms=4.5, zorder=3)
    ax.plot(xs, net, color=INK, marker="o", ms=4.5, zorder=4)
    for ys, lab, dy in [(us, "from US revenue", -14), (ab, "from revenue abroad", 9),
                        (net, "change in own US share\n(the sum of the two)", -1)]:
        ax.annotate(lab, (xs[-1], ys[-1]), xytext=(8, dy), textcoords="offset points", va="center", color=INK2,
                    fontsize=9)
    ax.set_xticks(xs)
    ax.set_xlim(2.6, 20.4)
    ax.set_ylim(-14, 9)
    ax.set_xlabel("Medicines counted as old: at least this many years since their first US approval")
    ax.set_ylabel("Points of the pooled US share, 2016-2025")
    ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=4.5, label="from US revenue"),
                       Line2D([], [], color=ORANGE, marker="o", ms=4.5, label="from revenue abroad"),
                       Line2D([], [], color=INK, marker="o", ms=4.5, label="change in own US share")],
              loc="upper right", fontsize=8.5, ncol=3, bbox_to_anchor=(1.0, 0.93))
    title(fig, "Whether the fall comes from the United States depends on where 'old' begins",
          "Each medicine's change in its own US share, weighted as in follow-up 2, split exactly into the part due "
          "to its US revenue\nand the part due to its revenue abroad, for the medicines counted as old")
    note(fig, "Nine companies that tag revenue by medicine and country in every year 2016-2025, 1,225 pairs of "
              "consecutive years.\nSource: SEC XBRL filings; FDA approval dates. Code: Medium-Blog, "
              "taxation-efficiency-drug-prices/followup3.")
    fig.savefig(OUT / "fig1_where_old_begins.png", dpi=200)
    plt.close(fig)


def fig_medicare(data: dict) -> None:
    pts = data["medicare"]
    fig, ax = plt.subplots(figsize=(10, 6.4))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.85, bottom=0.17)
    vol = np.array([p["vol"] for p in pts]) / 100
    net = np.array([p["net"] for p in pts]) / 100
    size = np.array([p["U0"] for p in pts]) / 1e9
    lo, hi = -3.1, 2.15
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax.text(-2.05, -1.25, "on the dashed line, US revenue\nmoved with Medicare volume", color=MUTED, fontsize=8, ha="center")
    ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
    ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
    col = np.where(net < 0, ORANGE, BLUE)
    ax.scatter(vol, net, s=18 + 22 * size, c=col, alpha=0.85, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    offsets = {"humira": (12, -4), "januvia": (10, -2), "enbrel": (10, 0), "trulicity": (8, 9),
               "imbruvica": (-10, 4), "victoza": (8, -6), "restasis": (10, 0), "keytruda": (10, 0),
               "stelara": (10, 4), "remicade": (10, -6), "pneumovax23": (10, 0), "jardiance": (8, -9),
               "farxiga": (-8, 10), "novolog": (8, 9), "symbicort": (-8, 9)}
    for p, x, y in zip(pts, vol, net):
        if p["key"] in offsets:
            dx, dy = offsets[p["key"]]
            ax.annotate(NAMES.get(p["key"], p["key"]), (x, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=8, color=INK2, ha="left" if dx > 0 else "right", va="center")
    ratios = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8]
    ticks = [float(np.log(r)) for r in ratios]
    lab = [f"{r - 1:+.0%}".replace("+0%", "0") for r in ratios]
    ax.set_xticks(ticks, lab)
    ax.set_yticks(ticks, lab)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Change in Medicare volume, 2020-2024 (dosage units in Part D, billing units in Part B)")
    ax.set_ylabel("Change in the company's US revenue, 2020-2024")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color=ORANGE, label="US revenue fell"),
                       Line2D([], [], ls="", marker="o", color=BLUE, label="US revenue grew")],
              loc="upper left", fontsize=8.5)
    title(fig, "When old medicines lost US revenue, Medicare mostly kept buying them",
          "Medicines at least ten years past first US approval, with US revenue of 0.5 billion dollars or more in "
          "2020; circle area by that revenue; log scales")
    note(fig, "Below the dashed line, US revenue fell more than Medicare volume: net price, discounts or other payers. "
              "NovoLog and Symbicort sit above it\nbecause unbranded or authorised generic versions leave the brand "
              "rows in Medicare while the company still reports them under the brand.\nSource: SEC XBRL filings; CMS Medicare Part D and Part B Spending by Drug, 2020-2024. Code: "
              "Medium-Blog, taxation-efficiency-drug-prices/followup3.")
    fig.savefig(OUT / "fig2_medicare_volume_and_us_revenue.png", dpi=200)
    plt.close(fig)


def main() -> None:
    setup()
    OUT.mkdir(exist_ok=True)
    data = json.loads((HERE / "figure_data.json").read_text(encoding="utf-8"))
    fig_line(data)
    fig_medicare(data)


if __name__ == "__main__":
    main()
