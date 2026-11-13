"""Figures for Part 4. Reads figure_data.json written by access_numbers.py.

    python figures.py

Writes three PNG files to figures/.
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
from matplotlib.patches import Patch  # noqa: E402
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
DOT = "#9a998f"
FILING = "#d6d5cc"
DECISION = "#6f6e68"
LEGAL_DECISION, LEGAL_NEGOTIATION = 75, 42


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
        "axes.titlesize": 10.5,
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


def thousands(v, _pos=None) -> str:
    return f"{v:,.0f}"


def fig_three_explanations(data: dict) -> None:
    rows = data["countries"]
    corr = {c["var"]: c for c in data["correlations"]}
    panels = (
        ("pop_m", "lpop", "Market size", "Population, millions (log scale)", True),
        ("dist_km", "dist_km", "Distance", "Capital to Brussels, km", False),
        ("pli", "pli", "Price level of medicines", "EURIPID 2017 index (20 EU countries = 100)", False),
    )
    named = {"PT", "DE", "RO", "CH", "EL", "LT", "AT", "FI"}
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.6), sharey=True)
    for ax, (key, cvar, title, xlabel, log) in zip(axes, panels):
        pts = [r for r in rows if r[key] is not None]
        xs = np.array([r[key] for r in pts], float)
        ys = np.array([r["days"] for r in pts], float)
        xl = np.log10(xs) if log else xs
        b = np.polyfit(xl, ys, 1)
        grid = np.linspace(xl.min(), xl.max(), 50)
        ax.plot(10 ** grid if log else grid, np.polyval(b, grid), color=AXIS, lw=1.4, zorder=1)
        other = np.array([r["geo"] != "PT" for r in pts])
        ax.scatter(xs[other], ys[other], s=34, color=DOT, edgecolor=SURFACE, linewidth=1.0, zorder=2)
        ax.scatter(xs[~other], ys[~other], s=70, color=ORANGE, edgecolor=SURFACE, linewidth=1.4, zorder=3)
        offsets = {"LT": ((-6, 4), "right"), "CH": ((-6, -4), "right"), "AT": ((5, 4), "left")}
        for r, x, y in zip(pts, xs, ys):
            if r["geo"] in named:
                pt = r["geo"] == "PT"
                off, ha = offsets.get(r["geo"], ((5, 4), "left"))
                ax.annotate(r["country"], (x, y), xytext=off, textcoords="offset points", ha=ha,
                            fontsize=8.5 if pt else 7.8, color=INK if pt else INK2,
                            fontweight="semibold" if pt else "normal")
        c = corr[cvar]
        ax.set_title(f"{title}\nr = {c['r']:+.2f}  ({c['n']} countries)", fontsize=10, loc="left")
        ax.set_xlabel(xlabel, fontsize=8.5)
        if log:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 50]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(thousands))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean days from EU authorisation to availability")
    axes[0].yaxis.set_major_formatter(FuncFormatter(thousands))
    axes[0].set_ylim(0, 1200)
    fig.suptitle("Time to availability of new medicines in 29 European countries, against three explanations",
                 x=0.01, ha="left", fontsize=11.5, fontweight="semibold", color=INK)
    note(fig, "W.A.I.T. 2025 (medicines authorised 2021-2024, availability on 5 January 2026). EU27 without Malta, plus "
              "Iceland, Norway and Switzerland; the price panel has the 24 countries in the EURIPID study. Grey line: least squares.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUT / "fig1_size_distance_price.png", dpi=200)
    plt.close(fig)


def segment(ax, y, left, width, color, label, hatch=None, text_color="white"):
    ax.barh(y, width, left=left, height=0.56, color=color, edgecolor=SURFACE, linewidth=2.0, hatch=hatch, zorder=2)
    if width >= 70:
        box = dict(boxstyle="round,pad=0.25", facecolor=color, edgecolor="none") if hatch else None
        ax.text(left + width / 2, y, f"{width:.0f}", ha="center", va="center", fontsize=9, color=text_color,
                fontweight="semibold", zorder=3, bbox=box)


def fig_pipeline(data: dict) -> None:
    eu = data["pipeline"]["europe"]
    pt = data["pipeline"]["portugal_report"]
    tm = data["tracker_means"]
    rows = [
        ("Europe, average\nCRA Access Hurdles Portal 2026", [(eu["filing"], FILING, INK2, None),
                                                             (eu["decision"], DECISION, "white", None)]),
        ("Portugal, all concluded processes\nInfarmed report, medicines of 2020-2025",
         [(pt["filing"], FILING, INK2, None), (pt["infarmed"], BLUE, "white", None),
          (pt["company_negotiation"], AQUA, "white", "//")]),
        (f"Portugal, {tm['n']} approved processes\nInfarmed process tracker, 2026",
         [(tm["lag"], FILING, INK2, None), (tm["infarmed"], BLUE, "white", None), (tm["company"], AQUA, "white", None),
          (tm["negotiation"], ORANGE, "white", None)]),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    for i, (label, segs) in enumerate(rows):
        y = len(rows) - 1 - i
        left = 0.0
        for w, color, tc, hatch in segs:
            segment(ax, y, left, w, color, label, hatch=hatch, text_color=tc)
            left += w
        ax.text(left + 14, y, f"{left:,.0f} days", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.6)
    ax.set_xlim(0, 1250)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_xlabel("Mean days from EU authorisation")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(color=FILING, label="Before the company files"),
               Patch(color=DECISION, label="After filing (not split)"),
               Patch(color=BLUE, label="Infarmed and Ministry"),
               Patch(facecolor=AQUA, edgecolor=SURFACE, hatch="//", label="Company and negotiation"),
               Patch(color=AQUA, label="Company"),
               Patch(color=ORANGE, label="Negotiation")]
    ax.legend(handles=handles, ncol=3, fontsize=8.2, loc="lower left", bbox_to_anchor=(0.0, 1.02), handlelength=1.4)
    ax.set_title("Where the time goes: before and after the company files", pad=44)
    note(fig, "Means. Europe: reimbursed medicines authorised Jan 2022 to Jun 2025. Portugal (report): approvals, refusals and "
              "withdrawals; the published parts add to 923 against\na published total of 922, by rounding. Portugal (tracker): "
              "centrally authorised new active substances approved.")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig2_where_the_time_goes.png", dpi=200)
    plt.close(fig)


def fig_processes(data: dict) -> None:
    rows = data["tracker"]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(8.6, 0.2 * n + 2.0))
    for i, r in enumerate(rows):
        left = 0
        for key, color in (("infarmed", BLUE), ("negotiation", ORANGE), ("company", AQUA)):
            ax.barh(i, r[key], left=left, height=0.72, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=2)
            left += r[key]
    legal = LEGAL_DECISION + LEGAL_NEGOTIATION
    ax.axvline(legal, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["name"] for r in rows], fontsize=7.2)
    ax.set_ylim(-0.7, n - 0.3)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_xlabel("Days from submission to conclusion, by who the tracker attributes them to")
    handles = [Patch(color=BLUE, label="Infarmed"), Patch(color=ORANGE, label="Negotiation"), Patch(color=AQUA, label="Company"),
               Line2D([], [], color=INK, lw=1.1, ls=(0, (4, 3)),
                      label="Legal clock: 75 days of assessment + 30 working days of negotiation")]
    ax.legend(handles=handles, ncol=4, fontsize=8.2, loc="lower left", bbox_to_anchor=(0.0, 1.0), handlelength=1.6,
              columnspacing=1.2)
    ax.set_title(f"The {n} approved processes for new medicines in Infarmed's tracker", pad=26)
    note(fig, "Centrally authorised new active substances approved (Concluído - Deferido), tracker of 12 November 2026. "
              "Sorted by total days.")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "fig3_approved_processes.png", dpi=200)
    plt.close(fig)


def main() -> None:
    setup()
    OUT.mkdir(exist_ok=True)
    data = json.loads((HERE / "figure_data.json").read_text())
    fig_three_explanations(data)
    fig_pipeline(data)
    fig_processes(data)


if __name__ == "__main__":
    main()
