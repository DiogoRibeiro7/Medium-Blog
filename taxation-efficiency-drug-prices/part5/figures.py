"""Figures for Part 5. Reads figure_data.json written by export_numbers.py.

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
from matplotlib.ticker import PercentFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#b9b8ae"

MOTIVES = (("Exportação", "Export", ORANGE, "o"), ("Criticidade", "Critical medicine", BLUE, "s"),
           ("Escassez", "Shortage", AQUA, "^"))
REASONS = {
    "Fabrico": "Manufacturing",
    "Aumento da procura": "Rising demand",
    "Comercial": "Commercial decision",
    "Logístico": "Logistics",
    "Qualidade": "Quality defect",
    "Fabrico da substância ativa": "Active-substance supply",
    "Regulamentar": "Regulatory",
    "Preço": "Price",
    "Inesperado ou desastres naturais": "Unexpected events",
}


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


def hbars(ax, labels, values, highlight, fmt, xmax):
    y = np.arange(len(labels))[::-1]
    colors = [ORANGE if lab == highlight else NEUTRAL for lab in labels]
    ax.barh(y, values, height=0.68, color=colors, edgecolor=SURFACE, linewidth=2)
    for yi, v, lab in zip(y, values, labels):
        ax.text(v + xmax * 0.012, yi, fmt(v), va="center", fontsize=8.5, color=INK if lab == highlight else INK2,
                fontweight="semibold" if lab == highlight else "normal")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, xmax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def fig_register(data: dict) -> None:
    rows = data["register"]
    dest = sorted(rows, key=lambda r: -r["dest_share"])[:10]
    left = sorted(rows, key=lambda r: r["left_out"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 6.2), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.42})
    hbars(a, [r["country"] for r in dest], [100 * r["dest_share"] for r in dest], "Portugal",
          lambda v: f"{v:.0f}%" if v >= 1 else f"{v:.1f}%", 64)
    a.set_title("Destination named in active notices")
    a.set_xlabel("Share of the 23,724 active notices")
    a.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    hbars(b, [r["country"] for r in left], [100 * r["left_out"] for r in left], "Portugal",
          lambda v: f"{v:.1f}%", 52)
    b.set_title("Left out as a possible source country")
    b.set_xlabel("Share of notices naming 15+ sources that omit it")
    b.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    b.tick_params(axis="y", labelsize=8)
    fig.suptitle("Parallel distribution in the EMA register: medicines go to Germany and the Nordics, and may come from almost anywhere",
                 x=0.01, ha="left", fontsize=11.5, fontweight="semibold", color=INK)
    note(fig, "Source: EMA, parallel distribution register (IRIS), active notices, November 2026. "
              "Left: the ten most named destinations. Right: EU and EEA countries in the register.")
    fig.subplots_adjust(left=0.14, right=0.97, top=0.89, bottom=0.12)
    fig.savefig(OUT / "fig1_register_destinations_sources.png", dpi=200)
    plt.close(fig)


def fig_lists(data: dict) -> None:
    sizes = data["sizes"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    series = (("lnp", "Prior-notification list", BLUE, "o"), ("suspended", "Exports suspended", AQUA, "s"),
              ("prohibited", "Monthly prohibitions (2026)", ORANGE, "D"))
    for typ, label, color, marker in series:
        pts = sorted((np.datetime64(r["date"]), r["n_named"]) for r in sizes if r["type"] == typ)
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        ax.plot(x, y, color=color, marker=marker, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=label)
    ax.set_ylim(0, 290)
    ax.set_xlim(np.datetime64("2021-03-01"), np.datetime64("2026-12-31"))
    ax.set_ylabel("Presentations named on the list")
    ax.set_title("Portugal's export-control lists, 2021 to 2026: presentations named one by one on each list")
    ax.legend(loc="upper left", fontsize=8.5)
    note(fig, "Source: Infarmed, lists published under Deliberações 391/2021 and 234/2025, read from the live page and Wayback "
              "snapshots. Since 2026 the suspended list also bans quetiapine,\nestradiol and isosorbide mononitrate as whole classes "
              "(not counted here). Three lists could not be downloaded (LNP April and August 2024, suspended January 2026).")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.17)
    fig.savefig(OUT / "fig2_list_sizes.png", dpi=200)
    plt.close(fig)


def fig_prohibitions(data: dict) -> None:
    rows = [r for r in data["prohibited"] if r["export"] is not None and r["intent"] is not None]
    rng = np.random.default_rng(20261112)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot([0, 100], [0, 100], color=AXIS, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(100, 97, "exports equal to intentions", ha="right", va="top", fontsize=8, color=MUTED, rotation=0)
    for key, label, color, marker in MOTIVES:
        sub = [r for r in rows if r["motive"] == key]
        x = np.array([min(r["intent"], 3.0) for r in sub])
        y = np.array([min(r["export"], 1.0) for r in sub])
        x = x + np.where(x == 0, rng.uniform(-0.04, 0.04, len(x)), 0)
        y = y + np.where(y == 0, rng.uniform(-0.012, 0.012, len(y)), 0)
        ax.scatter(x * 100, y * 100, s=34, color=color, marker=marker, edgecolor=SURFACE, linewidth=0.8, alpha=0.9,
                   label=f"{label} (n={len(sub)})", zorder=3)
    ax.set_xlim(-10, 310)
    ax.set_ylim(-4, 104)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel("Export intentions notified, as a share of the packs the company sold in Portugal (capped at 300%)")
    ax.set_ylabel("Exports carried out, same base (capped at 100%)")
    ax.set_title("Infarmed's 2026 export prohibitions: bans given for export reasons fall on presentations with large\n"
                 "export intentions and exports; bans for shortage fall on presentations that are hardly exported")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8.5, borderaxespad=0.3, handletextpad=0.3)
    ax.set_title(ax.get_title(loc="left"), pad=24)
    note(fig, "Source: Infarmed, lists of export prohibitions, January to October 2026 (five lists, 294 rows with both indicators). "
              "Each point is one presentation on one list.\nThe base is the quantity declared for Infarmed's 0.4% sales levy; "
              "points at zero are spread slightly to show their number.")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.16)
    fig.savefig(OUT / "fig3_prohibition_indicators.png", dpi=200)
    plt.close(fig)


def fig_reasons(data: dict) -> None:
    raw = data["reasons"]
    total21 = sum(raw["2021"].values())
    shares = {"2021": {k: 100 * v / total21 for k, v in raw["2021"].items()}, "2024": raw["2024"], "2025": raw["2025"]}
    order = sorted(REASONS, key=lambda k: -shares["2025"][k])
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    y = np.arange(len(order))[::-1]
    h = 0.26
    for i, (year, color) in enumerate((("2021", NEUTRAL), ("2024", BLUE), ("2025", ORANGE))):
        vals = [shares[year][k] for k in order]
        ax.barh(y + (1 - i) * h, vals, height=h, color=color, edgecolor=SURFACE, linewidth=1.5, label=year)
    for yi, k in zip(y, order):
        ax.text(shares["2025"][k] + 0.8, yi - h, f"{shares['2025'][k]:.1f}%", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [REASONS[k] for k in order])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 65)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel("Share of shortages (labels: 2025)")
    ax.set_title("Reasons companies give Infarmed for shortages: manufacturing first, price last")
    ax.legend(loc="lower right", fontsize=8.5)
    note(fig, "Source: Infarmed, availability reports 2021, 2024 and 2025. 2021: shortages notified (1,779); "
              "2024 and 2025: share of presentations in shortage, read from Infarmed's charts.")
    fig.subplots_adjust(left=0.22, right=0.97, top=0.93, bottom=0.14)
    fig.savefig(OUT / "fig4_shortage_reasons.png", dpi=200)
    plt.close(fig)


def main() -> None:
    setup()
    OUT.mkdir(exist_ok=True)
    data = json.loads((HERE / "figure_data.json").read_text(encoding="utf-8"))
    fig_register(data)
    fig_lists(data)
    fig_prohibitions(data)
    fig_reasons(data)


if __name__ == "__main__":
    main()
