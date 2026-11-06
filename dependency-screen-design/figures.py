"""Figures for the article, from ``results/``.

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
BLUE_LIGHT = "#86b6ef"
INK, INK2, MUTED, GRID, SURFACE, SHADE, GREY = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb", "#efeeea", "#b5b4ae"
CREDIT = "Diogo Ribeiro"
NAMES = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CY": "Cyprus", "CZ": "Czechia", "DE": "Germany", "DK": "Denmark",
    "EE": "Estonia", "EL": "Greece", "ES": "Spain", "FI": "Finland", "FR": "France", "HR": "Croatia", "HU": "Hungary",
    "IE": "Ireland", "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "MT": "Malta", "NL": "Netherlands",
    "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia",
    "EU27": "EU-27", "EU15": "EU-15", "UK": "United Kingdom", "US": "United States",
}
EU27 = [k for k in NAMES if len(k) == 2 and k not in ("UK", "US")]
OUTLIERS = ["IE", "LU", "MT", "CY"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
    }
)


def note(fig, text: str, y: float = -0.02) -> None:
    fig.text(0.01, y, f"{text}   |   {CREDIT}", ha="left", va="top", fontsize=8.2, color=MUTED)


def suptitle(fig, text: str, y: float = 1.02) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=y)


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pct(ax, axis: str = "y", decimals: int = 0) -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)



SUPPLIER_COLOURS = {"CN": BLUE, "US": ORANGE, "RU": AQUA}
SUP_NAMES = {"BD": "Bangladesh", "MY": "Malaysia", "PK": "Pakistan", "CN": "China", "US": "United States", "RU": "Russia", "GB": "United Kingdom", "CH": "Switzerland", "NO": "Norway", "TR": "Türkiye", "JP": "Japan", "KR": "South Korea", "IN": "India", "TW": "Taiwan", "VN": "Vietnam", "BR": "Brazil", "UA": "Ukraine", "ZA": "South Africa", "CL": "Chile", "AU": "Australia", "CA": "Canada", "MA": "Morocco", "ID": "Indonesia"}


def fig1_hhi_vs_top() -> None:
    from panel import load

    d = load()
    g = d[d.year == 2025]
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    ax.scatter(g.top_share, g.hhi, s=5, color=GREY, alpha=0.45, linewidth=0, zorder=1, label="HS 6-digit products, 2025")
    s = np.linspace(0, 1, 200)
    ax.plot(s, s, color=INK2, lw=1.2, ls="--", zorder=2)
    ax.plot(s, s**2, color=INK2, lw=1.2, ls="--", zorder=2)
    ax.text(0.17, 0.205, "HHI ≤ largest share", fontsize=8.4, color=INK2, rotation=37, ha="center", va="bottom")
    ax.text(0.30, 0.055, "HHI ≥ largest share squared", fontsize=8.4, color=INK2, rotation=20, ha="center", va="bottom")
    ax.axhline(0.4, color=ORANGE, lw=1.6)
    ax.axvline(0.4, color=BLUE, lw=1.6)
    ax.fill_between([0.4, 0.4 ** 0.5], 0, 0.4, color=BLUE, alpha=0.08, zorder=0)
    ax.text(0.405, 0.02, "largest supplier above 40%,\nHHI at or below 0.4", fontsize=8.4, color=BLUE, va="bottom")
    ax.text(0.02, 0.415, "Commission threshold: HHI > 0.4", fontsize=8.4, color=INK, va="bottom")
    cases = d[((d["product"] == "271121") & (d.year == 2021)) | ((d["product"] == "271111") & (d.year == 2025)) | ((d["product"] == "270112") & (d.year == 2021)) | ((d["product"] == "850760") & (d.year == 2025)) | ((d["product"] == "854143") & (d.year == 2025))]
    names = {"271121": "pipeline gas 2021 (Russia)", "271111": "LNG 2025 (United States)", "270112": "coal 2021 (Russia)", "850760": "lithium-ion batteries 2025 (China)", "854143": "solar modules 2025 (China)"}
    off = {"271121": (8, -14), "271111": (8, -14), "270112": (-8, -14), "850760": (-8, 8), "854143": (-8, -12)}
    for _, r in cases.iterrows():
        ax.plot([r.top_share], [r.hhi], "o", color=INK, ms=6, mec=SURFACE, mew=1.5, zorder=4)
        dx, dy = off[r["product"]]
        ax.annotate(names[r["product"]], (r.top_share, r.hhi), xytext=(dx, dy), textcoords="offset points", ha="left" if dx > 0 else "right", fontsize=8.4, color=INK)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    pct(ax, "x")
    ax.set_xlabel("largest outside supplier's share of extra-EU imports")
    ax.set_ylabel("Herfindahl-Hirschman index of outside suppliers")
    ax.grid(axis="x", color=GRID)
    ax.set_title("The concentration index against the largest supplier's share", loc="left")
    note(fig, "Source: Eurostat Comext DS-045409, EU-27 imports by HS 6-digit product; author's calculations")
    fig.tight_layout()
    save(fig, "fig1_hhi_vs_largest_share.png")


def fig2_gas() -> None:
    e = pd.read_csv(RES / "energy_gas.csv").set_index("year")
    e = e[e.index >= 1990]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.plot(e.index, e.hhi, color=ORANGE, label="HHI of outside suppliers (Commission criterion: above 0.4)")
    ax.plot(e.index, e.top_share_of_supply, color=BLUE, label="largest outside supplier's share of EU supply (production + extra-EU imports)")
    ax.axhline(0.4, color=ORANGE, lw=1, ls="--")
    ax.axhline(0.25, color=BLUE, lw=1, ls="--")
    ax.text(2025.0, 0.4, "0.4", va="center", fontsize=8.4, color=ORANGE)
    ax.text(2025.0, 0.25, "25%", va="center", fontsize=8.4, color=BLUE)
    last_on = int(e[e.hhi > 0.4].index.max())
    ax.annotate(f"HHI last above 0.4 in {last_on}", (last_on, e.loc[last_on, "hhi"]), xytext=(20, 18), textcoords="offset points", fontsize=8.4, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate(f"Russia {e.loc[2019, 'top_share_of_supply']:.0%} of EU supply", (2019, e.loc[2019, "top_share_of_supply"]), xytext=(-12, 14), textcoords="offset points", ha="right", fontsize=8.4, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate(f"Norway {e.loc[2024, 'top_share_of_supply']:.0%}", (2024, e.loc[2024, "top_share_of_supply"]), xytext=(-4, -30), textcoords="offset points", ha="right", fontsize=8.4, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.axvspan(2022, 2024.3, color=SHADE, zorder=0)
    ax.set_xlim(1990, 2026.2)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("index or share")
    ax.legend(loc="upper right", fontsize=8.8)
    ax.set_title("Natural gas: what the concentration index saw, and what exposure saw, EU-27 1990–2024", loc="left", fontsize=10.5)
    note(fig, "Sources: Eurostat nrg_ti_gas (imports by partner, TJ GCV) and nrg_cb_gas (EU production); author's calculations. Shaded: 2022 onwards")
    fig.tight_layout()
    save(fig, "fig2_gas_known_case.png")


def scatter_pair(x, y, xlabel, ylabel, title, name, xthr, ythr, xlog=False, ylog=False, quad=None) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ax.scatter(x, y, s=8, color=BLUE, alpha=0.35, linewidth=0)
    ax.axvline(xthr, color=ORANGE, lw=1.4)
    ax.axhline(ythr, color=ORANGE, lw=1.4)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if quad:
        for (qx, qy, text, ha, va) in quad:
            ax.text(qx, qy, text, transform=ax.transAxes, fontsize=8.6, color=INK, ha=ha, va=va, bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", color=GRID)
    ax.set_title(title, loc="left", fontsize=10.5)
    return fig, ax


def fig3_importance() -> None:
    d = pd.read_csv(RES / "supply_products.csv", dtype={"prodcom": str})
    a, b = d.importance > 0.5, d.outside_share > 0.5
    quad = [
        (0.98, 0.98, f"both above 50%: {int((a & b).sum())}", "right", "top"),
        (0.98, 0.02, f"Commission proxy only: {int((a & ~b).sum())}", "right", "bottom"),
        (0.02, 0.98, f"outside share of supply only: {int((~a & b).sum())}", "left", "top"),
        (0.02, 0.02, f"neither: {int((~a & ~b).sum())}", "left", "bottom"),
    ]
    fig, ax = scatter_pair(d.importance, d.outside_share, "Commission proxy: extra-EU imports / all EU imports (incl. intra-EU)", "extra-EU imports / EU supply (production + extra-EU imports)", f"Importance: the trade proxy against the outside share of EU supply, {len(d):,} products, 2025", "fig3", 0.5, 0.5, quad=quad)
    pct(ax, "x")
    pct(ax, "y")
    note(fig, "Sources: Eurostat Prodcom DS-059367 (2025), Comext DS-045409 (2025, CN 8-digit, summed to Prodcom codes); author's calculations")
    fig.tight_layout()
    save(fig, "fig3_importance_vs_outside_share.png")


def fig4_substitutability() -> None:
    d = pd.read_csv(RES / "supply_products.csv", dtype={"prodcom": str})
    x = d.substitutability.clip(1e-3, 1e3)
    y = d.eu_share
    a, b = d.substitutability > 1, y < 0.5
    quad = [
        (0.98, 0.02, f"ratio > 1, EU production below imports: {int((a & b).sum())}", "right", "bottom"),
        (0.98, 0.98, f"ratio > 1, EU production above imports: {int((a & ~b).sum())}", "right", "top"),
        (0.02, 0.02, f"ratio <= 1, EU production below imports: {int((~a & b).sum())}", "left", "bottom"),
        (0.02, 0.98, f"neither: {int((~a & ~b).sum())}", "left", "top"),
    ]
    fig, ax = scatter_pair(x, y, "Commission proxy: extra-EU imports / total EU exports (log scale)", "EU production / EU supply (production + extra-EU imports)", f"Substitutability: the trade proxy against EU production's share of supply, {len(d):,} products, 2025", "fig4", 1.0, 0.5, xlog=True, quad=quad)
    pct(ax, "y")
    note(fig, "Sources: Eurostat Prodcom DS-059367 (2025), Comext DS-045409 (2025); author's calculations. Total EU exports include exports between member states")
    fig.tight_layout()
    save(fig, "fig4_substitutability_vs_eu_share.png")


SHORT = {
    "26201100": "Laptop computers",
    "26302210": "Smartphones",
    "27202430": "Lithium-ion batteries",
    "21105200": "Hormones and their derivatives",
    "30311600": "Parts of civil jet and turboprop engines",
    "30313410": "Civil aircraft over 15 tonnes",
    "30311200": "Civil jet and turboprop engines",
    "26302320": "Network equipment (switching, routing)",
    "20145280": "Pyridine- and quinoline-ring compounds",
    "26112230": "Photovoltaic modules",
    "24412040": "Unwrought gold",
    "21103159": "Pyrimidine-ring compounds",
    "24412050": "Semi-manufactured gold",
    "14102300": "Knitted T-shirts and vests",
    "26521200": "Watches (base metal)",
}


def short(s: str, n: int = 52) -> str:
    s = str(s).replace("{s}", "(s)").split(";")[0].split(" (excluding")[0].split(" (excl.")[0]
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def fig5_ranking() -> None:
    big = pd.read_csv(RES / "supply_largest.csv", dtype={"prodcom": str}).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.6, 7.0))
    yy = np.arange(len(big))
    ax.barh(yy, big.sup_imports / 1e9, color=[SUPPLIER_COLOURS.get(p, GREY) for p in big.top_partner], height=0.66, edgecolor=SURFACE, linewidth=1.5)
    for yi, (_, r) in zip(yy, big.iterrows()):
        ax.annotate(f"{SUP_NAMES.get(r.top_partner, r.top_partner)}: {r.exposure:.0%} of EU supply" + ("" if r.commission else "  *"), (r.sup_imports / 1e9, yi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.4, color=INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels([SHORT.get(p, short(l)) for p, l in zip(big.prodcom, big.label)], fontsize=8.6)
    ax.set_xlim(0, big.sup_imports.max() / 1e9 * 1.45)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("imports from the largest outside supplier, EUR billion")
    ax.set_title("The largest exposures under the supply-based screen, 2025", loc="left", fontsize=10.5)
    note(fig, "Sources: Eurostat Prodcom DS-059367, Comext DS-045409; author's calculations. Products whose largest outside supplier provides more than 25% of EU supply\n(production + extra-EU imports), ranked by imports from that supplier. Bars: China blue, United States orange, other grey. * not flagged by the Commission screen")
    fig.tight_layout()
    save(fig, "fig5_supply_screen_ranking.png")


def fig6_agreement() -> None:
    d = pd.read_csv(RES / "supply_products.csv", dtype={"prodcom": str})
    d["sup"] = d.sup_imports / 1e9
    cats = [("both", d.commission & d.proposed, BLUE), ("Commission screen only", d.commission & ~d.proposed, ORANGE), ("supply-based screen only", ~d.commission & d.proposed, AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    for ax, (val, lab) in zip(axes, ((None, "products"), ("sup", "EUR billion of imports from the largest outside supplier"))):
        x = np.arange(len(cats))
        v = [int(m.sum()) if val is None else float(d.loc[m, val].sum()) for _, m, _ in cats]
        ax.bar(x, v, color=[c for *_, c in cats], width=0.62, edgecolor=SURFACE, linewidth=1.5)
        for xi, vi in zip(x, v):
            ax.annotate(f"{vi:,.0f}", (xi, vi), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.6, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([c for c, *_ in cats], fontsize=8.8)
        ax.set_ylabel(lab)
        ax.set_ylim(0, max(v) * 1.15)
    axes[0].set_title("Number of products", fontsize=10.5)
    axes[1].set_title("Value at stake", fontsize=10.5)
    suptitle(fig, f"Where the two screens agree and disagree, {len(d):,} products, 2025", y=1.04)
    note(fig, "Sources: Eurostat Prodcom DS-059367, Comext DS-045409; author's calculations. Supply-based screen: largest outside supplier > 25% of EU supply (production + extra-EU imports)", y=-0.03)
    fig.tight_layout()
    save(fig, "fig6_screen_agreement.png")


def fig7_stability() -> None:
    p = pd.read_csv(RES / "panel_persistence.csv")
    t = pd.read_csv(RES / "supply_thresholds.csv")
    th = pd.read_csv(RES / "panel_thresholds.csv").dropna(subset=["threshold"])
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    ax = axes[0]
    for col, lab, colr in (("commission", "Commission screen", ORANGE), ("single", "single-supplier screen (trade data)", BLUE)):
        q = p[(p.screen == col) & (p["from"] == 2019)].sort_values("to")
        ax.plot(q["to"], q.share, color=colr, marker="o", ms=5, mec=SURFACE, mew=1.2, label=lab)
    ax.set_ylim(0, 1)
    pct(ax)
    ax.set_xticks([2021, 2022, 2023, 2024, 2025])
    ax.legend(loc="lower left", fontsize=8.6)
    ax.set_title("Share of products flagged in 2019 still flagged later", fontsize=10.5)
    ax = axes[1]
    ax.plot(th.threshold, th.hhi, color=ORANGE, marker="o", ms=4, mec=SURFACE, mew=1, label="Commission: HHI threshold")
    ax.plot(th.threshold, th.top_share, color=BLUE, marker="o", ms=4, mec=SURFACE, mew=1, label="trade data: largest-share threshold")
    ax.plot(t.threshold, t.products, color=AQUA, marker="o", ms=4, mec=SURFACE, mew=1, label="production + trade data: exposure threshold")
    ax.axvline(0.4, color=MUTED, lw=1, ls="--")
    ax.axvline(0.25, color=MUTED, lw=1, ls=":")
    pct(ax, "x")
    ax.set_ylabel("products flagged, 2025")
    ax.legend(loc="upper right", fontsize=8.4)
    ax.set_title("Products flagged as each threshold moves", fontsize=10.5)
    suptitle(fig, "Stability and threshold sensitivity", y=1.04)
    note(fig, "Source: Eurostat Comext DS-045409 (HS 6-digit, 2002–2025) and Prodcom DS-059367 (2025); author's calculations. The trade-data curves keep the other two Commission criteria;\nthe production + trade curve counts Prodcom products, a different product list", y=-0.03)
    fig.tight_layout()
    save(fig, "fig7_stability_and_thresholds.png")


def main() -> None:
    fig1_hhi_vs_top()
    fig2_gas()
    fig3_importance()
    fig4_substitutability()
    fig5_ranking()
    fig6_agreement()
    fig7_stability()


if __name__ == "__main__":
    main()
