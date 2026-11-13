"""Figures for follow-up 1. Reads figure_data.json written by rebates_numbers.py.

    python figures.py

Writes two PNG files to figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
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


def fig_trend(data: dict) -> None:
    s = data["series"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.07, right=0.80, top=0.90, bottom=0.27)

    fr = {int(k): v for k, v in s["FR"].items()}
    ax.plot(list(fr), [100 * v for v in fr.values()], color=BLUE, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    ax.plot([2023], [100 * s["FR_igf"]["2023"]], marker="o", markersize=7, markerfacecolor=SURFACE,
            markeredgecolor=BLUE, markeredgewidth=1.6, linestyle="none", zorder=3)

    be = {int(k): v for k, v in s["BE"].items()}
    ax.plot(list(be), [100 * v for v in be.values()], color=GREEN, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)

    pt = s["PT"]
    inf = {int(k): v for k, v in pt["infarmed"].items()}
    yrs = sorted(inf)
    ax.plot(yrs, [100 * inf[y][1] for y in yrs], color=ORANGE, marker="o", markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=2, zorder=4)
    for y in yrs:
        ax.plot([y, y], [100 * inf[y][0], 100 * inf[y][2]], color=ORANGE, linewidth=5, alpha=0.30,
                solid_capstyle="round", zorder=2)
    api = {int(k): v for k, v in pt["apifarma"].items()}
    api[2024] = pt["apifarma_cip_2024"]
    ax.plot(list(api), [100 * v for v in api.values()], marker="D", markersize=6, markerfacecolor=SURFACE,
            markeredgecolor=ORANGE, markeredgewidth=1.6, linestyle="none", zorder=3)
    agr = {int(k): v for k, v in pt["agreement"].items() if int(k) >= 2014}
    ax.plot(list(agr), [100 * v for v in agr.values()], marker="s", markersize=5.5, color=MUTED, linestyle="none",
            zorder=3)

    # direct labels at the right end of each line
    ax.text(2024.35, 100 * be[2024] + 0.8, "Belgium", color=INK, fontsize=9.5, fontweight="semibold", va="center")
    ax.text(2024.35, 100 * fr[2024] - 0.8, "France", color=INK, fontsize=9.5, fontweight="semibold", va="center")
    ax.text(2025.35, 100 * inf[2025][1], "Portugal", color=INK, fontsize=9.5, fontweight="semibold", va="center")

    ax.set_xlim(2013.5, 2026.4)
    ax.set_xticks(range(2014, 2026))
    ax.set_ylim(0, 27)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_ylabel("Returned by industry, share of gross public spending")
    ax.grid(axis="x", visible=False)
    ax.set_title("The share of public spending on medicines that industry pays back has risen everywhere,\n"
                 "and Portugal's is about half of France's and Belgium's")

    handles = [
        Line2D([], [], color=BLUE, marker="o", label="France: product and early-access rebates (DREES)"),
        Line2D([], [], color=BLUE, marker="o", markerfacecolor=SURFACE, linestyle="none",
               label="France: rebates, IGAS-IGF count (2023)"),
        Line2D([], [], color=GREEN, marker="o", label="Belgium: compensations under art. 81/111 conventions (INAMI)"),
        Line2D([], [], color=ORANGE, marker="o", label="Portugal: contracts and agreement, Infarmed (bar: range)"),
        Line2D([], [], color=ORANGE, marker="D", markerfacecolor=SURFACE, linestyle="none",
               label="Portugal: figures from industry studies (APIFARMA)"),
        Line2D([], [], color=MUTED, marker="s", linestyle="none",
               label="Portugal: the industry agreement alone (a floor)"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.09), ncol=2, fontsize=8.2,
              handlelength=2.2, columnspacing=1.6)
    note(fig, "Denominators: France, reimbursable medicines outside hospital tariffs; Belgium, INAMI gross spending on "
              "medicines; Portugal, SNS hospital and pharmacy charges.\nBelgium 2024 provisional. The 2018 and 2022 "
              "APIFARMA shares are plotted as published; the newsletter does not state their base.\nLevies, CEIF and "
              "France's safeguard clause are left out (with them, 2024: France 26%, Belgium 29%, Portugal 11-16%).")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fig1_share_returned_by_industry.png", dpi=200)
    plt.close(fig)


def fig_prices(data: dict) -> None:
    n = data["net"]
    rows = [("France", "FR", BLUE), ("Belgium", "BE", GREEN), ("Portugal", "PT", ORANGE)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.84, bottom=0.28)
    ys = {"FR": 2, "BE": 1, "PT": 0}
    for label, k, c in rows:
        y = ys[k]
        lst = n["list"][k]
        if k == "PT":
            lo, mid, hi = n["net"]["PT"]
            ax.plot([mid, lst], [y, y], color=c, linewidth=2, alpha=0.6, zorder=2)
            ax.plot([lo, hi], [y, y], color=c, linewidth=9, alpha=0.30, solid_capstyle="round", zorder=2)
            net = mid
            blo, bhi = n["net_broad"]["PT"]
            ax.plot([blo, bhi], [y - 0.2, y - 0.2], color=MUTED, linewidth=2, zorder=2)
        else:
            net = n["net"][k]
            ax.plot([net, lst], [y, y], color=c, linewidth=2, alpha=0.6, zorder=2)
            ax.plot([n["net_broad"][k]], [y - 0.2], marker="|", markersize=11, color=MUTED, markeredgewidth=2)
        ax.plot([lst], [y], marker="o", markersize=10, markerfacecolor=SURFACE, markeredgecolor=c,
                markeredgewidth=2, zorder=3)
        ax.plot([net], [y], marker="o", markersize=10, color=c, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        ax.text(lst + 0.012, y + 0.02, f"list {lst:.3f}", va="center", ha="left", fontsize=8.5, color=INK2)
        left = n["net"]["PT"][0] if k == "PT" else net
        ax.text(left - 0.012, y + 0.02, f"net {net:.3f}", va="center", ha="right", fontsize=8.5, color=INK,
                fontweight="semibold")
    ax.axvline(n["net"]["FR"], color=BLUE, linewidth=0.8, linestyle=(0, (3, 3)), alpha=0.7, zorder=1)
    ax.set_yticks([2, 1, 0], ["France", "Belgium", "Portugal"])
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, 2.5)
    ax.set_xlim(0.66, 1.24)
    ax.set_xlabel("Price relative to France's list price (presentations priced in all four reference countries, Part 3)")
    rf, rb = n["ratio_fr"], n["ratio_be"]
    ax.set_title(f"Portugal's list price sits just above France's; after returns its net price is about "
                 f"{100 * (rf[0] - 1):.0f}% above France's\nand close to Belgium's (2024)")
    handles = [
        Line2D([], [], marker="o", markersize=8, markerfacecolor=SURFACE, markeredgecolor=INK2, linestyle="none",
               label="List price"),
        Line2D([], [], marker="o", markersize=8, color=INK2, linestyle="none",
               label="Net of rebates and contract returns"),
        Line2D([], [], color=MUTED, marker="|", markersize=10, markeredgewidth=2, linestyle="none",
               label="Net also of levies, CEIF and safeguard clause"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=3, fontsize=8.5)
    note(fig, f"Net prices apply each country's 2024 share returned (France 22.0%, Belgium 22.4%, Portugal 10.2-15.2%, "
              f"the shaded bar)\nto the Part 3 list ratios. Portugal's net over France's {rf[0]:.2f} "
              f"[{rf[1]:.2f}, {rf[2]:.2f}], over Belgium's {rb[0]:.2f} [{rb[1]:.2f}, {rb[2]:.2f}], 90% intervals.")
    fig.savefig(OUT / "fig2_list_and_net_prices.png", dpi=200)
    plt.close(fig)


def main() -> None:
    setup()
    data = json.loads((HERE / "figure_data.json").read_text(encoding="utf-8"))
    fig_trend(data)
    fig_prices(data)


if __name__ == "__main__":
    main()
