"""Figures for Part 2. Reads figure_data.json written by prices_numbers.py.

    python figures.py

Writes four PNG files to figures/.
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
BASKET_COLOR = {"single-source": BLUE, "multi-source": AQUA, "mixed": ORANGE}


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


def dollars(v, _pos=None) -> str:
    if v >= 1:
        return f"${v:,.0f}"
    return f"${v:.2f}".rstrip("0").rstrip(".") if v >= 0.01 else f"${v:.3f}"


def times(v, _pos=None) -> str:
    return f"{v:g}×"


def fig_scatter(data: dict) -> None:
    mol = data["molecules"]
    p_us = np.array(mol["p_us"])
    p_fr = np.array(mol["p_fr_usd"])
    v_us = np.array(mol["v_us"])
    basket = np.array(mol["basket"])
    names = np.array(mol["molecule"])

    fig, ax = plt.subplots(figsize=(8.6, 8.4))
    lo, hi = 0.005, 3000
    xs = np.array([lo, hi])
    for k, lab, x0 in ((1, "same price", 0.012), (3, "3× more in the US", 0.0062), (10, "10× more in the US", 0.0062)):
        ax.plot(xs, k * xs, color=AXIS if k > 1 else MUTED, lw=1, ls=(0, (4, 3)) if k > 1 else "-", zorder=1)
        ax.text(x0, k * x0 * 1.25, lab, rotation=45, rotation_mode="anchor", fontsize=8, color=INK2,
                ha="left", va="bottom")
    size = 6 + 900 * np.sqrt(v_us / v_us.max())
    order = np.argsort(-v_us)
    for b in ("multi-source", "mixed", "single-source"):
        sel = order[basket[order] == b]
        ax.scatter(p_fr[sel], p_us[sel], s=size[sel], color=BASKET_COLOR[b], alpha=0.55,
                   edgecolor=SURFACE, linewidth=0.8, label=b, zorder=3)

    def at(n):
        i = np.where(names == n)[0]
        return (p_fr[i[0]], p_us[i[0]]) if len(i) else None

    # (molecule or molecules, text position, horizontal alignment)
    labels = [
        (["apixaban"], (2.6, 4.0), "left"),
        (["empagliflozin", "dapagliflozin", "rivaroxaban"], (3.2, 45), "left"),
        (["sitagliptin"], (0.09, 45), "right"),
        (["palbociclib"], (20, 1500), "right"),
        (["tafamidis"], (500, 250), "left"),
        (["dasatinib"], (8, 600), "right"),
        (["abirateron"], (40, 8), "left"),
        (["atorvastatin"], (0.5, 0.04), "left"),
        (["gabapentin"], (0.03, 0.4), "right"),
    ]
    for mols, (tx, ty), ha in labels:
        pts = [at(n) for n in mols if at(n) is not None]
        if not pts:
            continue
        text = ", ".join(mols) if len(mols) < 3 else ",\n".join([", ".join(mols[:2]), mols[2]])
        ax.text(tx, ty, text, fontsize=8, color=INK, ha=ha, va="center")
        for (px, py) in pts[:1]:
            ax.annotate("", xy=(px, py), xytext=(tx, ty), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7,
                                                                          shrinkA=2, shrinkB=4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100, 1000]))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(FuncFormatter(dollars))
    ax.set_xlabel("France, reimbursement base plus dispensing fee, per tablet or capsule (USD)")
    ax.set_ylabel("US Medicare Part D, gross spending per tablet or capsule (USD)")
    ax.set_title("Price per tablet or capsule in 2024,\none dot per molecule sold in both countries")
    leg = ax.legend(loc="upper left", fontsize=9, markerscale=0.35, title="basket, by generic share of units",
                    title_fontsize=8.5)
    leg.get_title().set_color(INK2)
    ax.text(0.99, 0.02, "dot area proportional to US spending", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=MUTED)
    note(fig, "Sources: CMS Part D Spending by Drug 2024; Open Medic 2024 and BDPM; FDA NDC directory; ECB 2024 rate.\n"
              "Code: taxation-efficiency-drug-prices/part2.")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(OUT / "fig1_unit_prices.png", dpi=200)
    plt.close(fig)


def fig_indices(data: dict) -> None:
    rows = [
        ("Brand with brand (naive)", data["naive"]["originator"], None, MUTED),
        ("Generic with generic (naive)", data["naive"]["generic"], None, MUTED),
        ("Single-source molecules", data["baskets"]["single-source"], "single-source", BLUE),
        ("Mixed: generics in France only", data["baskets"]["mixed, generic in France only"], "fr", ORANGE),
        ("Mixed: generics in the US only", data["baskets"]["mixed, generic in US only"], "us", ORANGE),
        ("Multi-source molecules", data["baskets"]["multi-source"], "multi-source", AQUA),
        ("All matched molecules", data["baskets"]["all"], "all", INK2),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, d, key, col) in zip(ys, rows):
        if key is not None:
            lo, hi = d["ci"]["F"]
            ax.plot([lo, hi], [y, y], color=col, lw=6, alpha=0.18, solid_capstyle="round", zorder=1)
        ax.plot([d["L"], d["P"]], [y, y], color=col, lw=1.4, zorder=2)
        ax.scatter([d["L"]], [y], s=60, marker="o", color=SURFACE, edgecolor=col, linewidth=2, zorder=3)
        ax.scatter([d["P"]], [y], s=60, marker="s", color=col, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.scatter([d["F"]], [y], s=34, marker="|", color=INK, linewidth=1.6, zorder=4)
        right = max(d["L"], d["P"], d["ci"]["F"][1] if key is not None else 0)
        ax.text(right * 1.1, y, f"Fisher {d['F']:.2f}  ({d['n']} molecules)", va="center",
                fontsize=8.5, color=INK2)
    ax.axvline(1, color=MUTED, lw=1)
    ax.text(1.03, ys[0] + 0.55, "same price", fontsize=8, color=INK2, va="bottom")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 90)
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 50]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(times))
    ax.set_ylim(-0.7, len(rows) - 0.2)
    ax.set_xlabel("US price over French price (log scale)")
    ax.set_title("The same two price lists, read through different baskets and weights")
    handles = [
        plt.Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK2, markeredgewidth=2,
                   markersize=7, label="US volumes as weights (Laspeyres)"),
        plt.Line2D([], [], marker="s", ls="", color=INK2, markersize=7, label="French volumes as weights (Paasche)"),
        plt.Line2D([], [], color=INK2, lw=6, alpha=0.18, label="Fisher, 95% range when molecules are resampled"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.3, bbox_to_anchor=(0.5, 0.045))
    note(fig, "Gross prices: Part D spending before rebates; French reimbursement base plus the dispensing fee. "
              "Code: taxation-efficiency-drug-prices/part2.")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig2_indices_by_basket.png", dpi=200)
    plt.close(fig)


def fig_mixing(data: dict) -> None:
    mx = data["mixing"]
    single = np.array(mx["single"])
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(single, mx["us"], color=BLUE, label="US volumes within each basket")
    ax.plot(single, mx["fr"], color=ORANGE, label="French volumes within each basket")
    xu, xf = 1 - mx["share_us"], 1 - mx["share_fr"]
    ax.scatter([xu], [mx["L"]], s=60, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=4)
    ax.scatter([xf], [mx["P"]], s=60, color=ORANGE, edgecolor=SURFACE, linewidth=1.2, zorder=4)
    ax.annotate(f"US mix, US volumes: {mx['L']:.2f}×\n({xu:.1%} of units single-source)", xy=(xu, mx["L"]),
                xytext=(xu * 1.6, mx["L"] - 1.3), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.annotate(f"French mix, French volumes: {mx['P']:.2f}×\n({xf:.1%} of units single-source)", xy=(xf, mx["P"]),
                xytext=(xf / 9, mx["P"] + 0.9), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.set_xscale("log")
    ax.set_xlim(0.001, 1)
    ax.set_ylim(0, 6.6)
    ax.xaxis.set_major_locator(FixedLocator([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.1%}" if v < 0.01 else f"{v:.0%}"))
    ax.yaxis.set_major_formatter(FuncFormatter(times))
    ax.set_xlabel("Share of units in single-source molecules (log scale)")
    ax.set_ylabel("Index over all these molecules, US over France")
    ax.set_title("An all-drug index is set by the generic share as much as by prices")
    ax.legend(loc="lower right", fontsize=8.5)
    note(fig, "Single-source and multi-source molecules; prices fixed at 2024 levels, only the split of units changes. "
              "Code: taxation-efficiency-drug-prices/part2.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig3_mixing_curve.png", dpi=200)
    plt.close(fig)


def fig_mfp(data: dict) -> None:
    rows = sorted(data["mfp"], key=lambda r: r["r_before"])
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    ys = np.arange(len(rows))
    for y, r in zip(ys, rows):
        ax.annotate("", xy=(r["r_after"], y), xytext=(r["r_before"], y),
                    arrowprops=dict(arrowstyle="-|>", color=AXIS, lw=1.4, shrinkA=5, shrinkB=5))
        ax.scatter([r["r_before"]], [y], s=55, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=1)
        ax.scatter([r["r_after"]], [y], s=55, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1)
        ax.text(r["r_before"] * 1.08, y, f"{r['r_before']:.1f}×", va="center", fontsize=8, color=INK2)
        ax.text(r["r_after"] / 1.08, y, f"{r['r_after']:.1f}×", va="center", ha="right", fontsize=8, color=INK2)
    labels = [f"{r['drug']} ({int(r['ipay'])})" + (" *" if r["nadac_unit"] != r["nadac_unit"] else "") for r in rows]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.grid(axis="y", visible=False)
    ax.axvline(1, color=MUTED, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 40)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(times))
    ax.set_xlabel("US price per tablet or capsule over the French price (log scale)")
    ax.set_title("Medicare's negotiated prices against France, oral drugs selected for 2026 and 2027")
    handles = [plt.Line2D([], [], marker="o", ls="", color=ORANGE, markersize=7,
                          label="2024: pharmacy acquisition cost (NADAC) against France 2024"),
               plt.Line2D([], [], marker="o", ls="", color=BLUE, markersize=7,
                          label="negotiated price against the current French price")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.3, bbox_to_anchor=(0.5, 0.045))
    note(fig, "* specialty drug without a NADAC: 2024 uses Part D gross spending per unit. French prices exclude the "
              "dispensing fee. Code: taxation-efficiency-drug-prices/part2.")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(OUT / "fig4_negotiated_prices.png", dpi=200)
    plt.close(fig)


def main() -> None:
    setup()
    OUT.mkdir(exist_ok=True)
    data = json.loads((HERE / "figure_data.json").read_text())
    fig_scatter(data)
    fig_indices(data)
    fig_mixing(data)
    fig_mfp(data)


if __name__ == "__main__":
    main()
