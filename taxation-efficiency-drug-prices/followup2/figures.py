"""Figures for follow-up 2. Reads figure_data.json written by portfolio_numbers.py.

    python figures.py

Writes three PNG files to figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1f9e76"
LIGHT = "#bdbbb3"


def setup() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = "Inter" if "Inter" in names else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 10,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
        "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "lines.linewidth": 2.0,
    })


def note(fig, text: str) -> None:
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def fig_decomposition(data: dict) -> None:
    d = data["decomposition"]
    terms = [("entry", "Entry into the reported set"), ("between", "Shift of revenue towards or away from it"),
             ("within", "Change in each medicine's own US share"), ("exit", "Exit from the reported set")]
    groups = [("launched in the period", "Medicines first approved in the US in 2017-2025", ORANGE),
              ("launched before", "Medicines first approved by 2016", BLUE)]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(left=0.33, right=0.97, top=0.86, bottom=0.17)
    y = 0
    ticks, labels = [], []
    for g, glabel, color in groups:
        net = sum(d["cohort"][g][k] for k, _ in terms)
        ax.barh(y, 100 * net, height=0.62, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(100 * net + (0.15 if net >= 0 else -0.15), y, f"{100 * net:+.2f}", va="center",
                ha="left" if net >= 0 else "right", fontsize=9, color=INK, fontweight="semibold")
        ticks.append(y)
        labels.append(glabel)
        y -= 0.8
        for k, lab in terms:
            v = 100 * d["cohort"][g][k]
            ax.barh(y, v, height=0.46, color=color, alpha=0.35, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.text(v + (0.15 if v >= 0 else -0.15), y, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                    fontsize=8.5, color=INK2)
            ticks.append(y)
            labels.append("    " + lab)
            y -= 0.62
        y -= 0.5
    total = 100 * d["change"]
    ax.barh(y, total, height=0.62, color=INK2, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.text(total + 0.15, y, f"{total:+.2f}", va="center", ha="left", fontsize=9, color=INK, fontweight="semibold")
    ticks.append(y)
    labels.append(f"Change in the US share, {100 * d['s0']:.1f}% to {100 * d['s1']:.1f}%")
    ax.set_yticks(ticks, labels, fontsize=9)
    for t, lab in zip(ax.get_yticklabels(), labels):
        if not lab.startswith("    "):
            t.set_fontweight("semibold")
            t.set_color(INK)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=AXIS, linewidth=1, zorder=2)
    ax.set_xlim(-8.2, 9.5)
    ax.set_xlabel("Percentage points of the pooled US share of revenue, 2016 to 2025")
    fig.text(0.01, 0.955, f"The US share of nine companies' medicines rose {100 * d['change']:.1f} points in 2016-2025, and "
             "medicines launched in the period account for all of it;\nolder medicines lost US share of their own",
             ha="left", va="top", fontsize=10.5, fontweight="semibold", color=INK)
    note(fig, "Exact Griliches-Regev split of each year's change, medicine by medicine, summed over 2017-2025. Nine "
              "companies that report revenue by medicine and country every year: AbbVie, Amgen,\nAstraZeneca, Biogen, "
              "Eli Lilly, Gilead, Johnson & Johnson, Merck & Co, Novartis. First US approval from Drugs@FDA and the "
              "Purple Book.")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fig1_where_the_rise_comes_from.png", dpi=200)
    plt.close(fig)


def fig_lifecycle(data: dict) -> None:
    lc = data["lifecycle"]
    ages = lc["age"]
    fig, ax = plt.subplots(figsize=(10, 5.0))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.2)
    ax.fill_between(ages, [100 * v for v in lc["q1"]], [100 * v for v in lc["q3"]], color=ORANGE, alpha=0.18,
                    linewidth=0, zorder=2)
    ax.plot(ages, [100 * v for v in lc["median"]], color=ORANGE, marker="o", markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=2, zorder=4)
    ax.plot(ages, [100 * v for v in lc["weighted"]], color=INK2, linestyle=(0, (4, 3)), linewidth=1.6, zorder=3)
    for a, n in zip(ages, lc["n"]):
        ax.text(a, 28.5, f"n={n}", ha="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(ages, [f"{a}" for a in ages])
    ax.set_xlim(-0.4, max(ages) + 0.4)
    ax.set_ylim(25, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel("Years since first US approval")
    ax.set_ylabel("US share of the medicine's revenue")
    ax.grid(axis="x", visible=False)
    ax.set_title("A new medicine earns most of its first two years' revenue in the United States, and the share falls\n"
                 "as it reaches other markets")
    handles = [Line2D([], [], color=ORANGE, marker="o", label="Median medicine"),
               Patch(color=ORANGE, alpha=0.18, label="Middle half of medicines"),
               Line2D([], [], color=INK2, linestyle=(0, (4, 3)), label="All revenue at that age (weighted)")]
    ax.legend(handles=handles, loc="upper right", ncol=3, fontsize=8.5, bbox_to_anchor=(1.0, 1.0))
    note(fig, f"Medicines first approved in the US from {lc['first']}, with revenue of at least $50 million in the year "
              f"(reports cover 2016-2025, so those approved before 2016 enter at age 1 or 2); {lc['companies']} companies.")
    fig.savefig(OUT / "fig2_us_share_over_the_life_of_a_medicine.png", dpi=200)
    plt.close(fig)


def fig_companies(data: dict) -> None:
    rows = data["companies"]
    cos = sorted({r["company"] for r in rows})
    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.subplots_adjust(left=0.08, right=0.83, top=0.87, bottom=0.17)
    hi = {"Eli Lilly": ORANGE, "Novo Nordisk": BLUE, "AbbVie": GREEN}
    ends = []
    for co in cos:
        pts = sorted((r for r in rows if r["company"] == co), key=lambda r: r["py"])
        x0, y0 = 100 * pts[0]["n"], 100 * pts[0]["s"]
        x1, y1 = 100 * pts[-1]["n"], 100 * pts[-1]["s"]
        c = hi.get(co, MUTED)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8 if co in hi else 1.3, shrinkA=4, shrinkB=4,
                                    mutation_scale=12))
        ax.plot(x0, y0, marker="o", markersize=5.5, markerfacecolor=SURFACE, markeredgecolor=c, markeredgewidth=1.4,
                linestyle="none", zorder=4)
        ax.plot(x1, y1, marker="o", markersize=7, color=c, markeredgecolor=SURFACE, markeredgewidth=1.5,
                linestyle="none", zorder=5)
        ends.append([x1, y1, co, c, pts[0]["py"]])
    ends.sort(key=lambda e: -e[1])
    placed = []
    for x1, y1, co, c, fy in ends:
        ly = y1
        for px, py_ in placed:
            if abs(px - x1) < 12 and abs(py_ - ly) < 2.2:
                ly = py_ - 2.4
        placed.append((x1, ly))
        lab = co if fy == 2016 else f"{co} (from {fy})"
        ax.text(x1 + 1.3, ly, lab, fontsize=8.5, va="center", color=INK if co in hi else INK2,
                fontweight="semibold" if co in hi else "normal")
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlim(-2, 82)
    ax.set_xlabel("Share of the company's dated-medicine revenue from medicines less than five years old")
    ax.set_ylabel("US share of that revenue")
    ax.set_title("Within a company, more revenue from new medicines goes with a higher US share; across companies,\n"
                 "it does not")
    handles = [Line2D([], [], marker="o", markerfacecolor=SURFACE, markeredgecolor=INK2, linestyle="none",
                      label="first year"),
               Line2D([], [], marker="o", color=INK2, linestyle="none", label="2025")]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5)
    p2 = data["p2"]
    note(fig, f"Arrows join each company's first year (2016 unless shown) to 2025. Within companies, net of year "
              f"effects, a 10-point rise in the new-medicine share goes with a {10 * p2['b_fe']:.1f}-point rise in the "
              f"US share;\nacross companies in 2025 the rank correlation is {p2['rho']:+.2f}.")
    fig.savefig(OUT / "fig3_new_medicines_and_us_share_by_company.png", dpi=200)
    plt.close(fig)


def main() -> None:
    setup()
    data = json.loads((HERE / "figure_data.json").read_text(encoding="utf-8"))
    fig_decomposition(data)
    fig_lifecycle(data)
    fig_companies(data)


if __name__ == "__main__":
    main()
