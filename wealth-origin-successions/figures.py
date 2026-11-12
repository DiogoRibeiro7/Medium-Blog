"""Figures for the article. Reads results/ and data/, writes figures/fig*.png.

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
DATA = HERE / "data"
FIG = HERE / "figures"

# reference palette (light mode); blue / orange validated for CVD separation on this surface
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
})

SOURCE = "Data: Forbes person-list API (2011-2026 lists, matched in memory), Penn World Table 11.0, World Bank WDI."
KS = [-3, -2, -1, 0, 1, 2]


def save(fig, name: str, source: str = SOURCE) -> None:
    fig.text(0.01, 0.005, source, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_funnel() -> None:
    d = pd.read_csv(DATA / "succession_diagnostics.csv").fillna(0)
    sh = pd.read_csv(RES / "succession_shocks.csv")
    st = sh[sh["shock"] == "strict"]
    steps = [
        ("Self-made billionaires who left the list\nbetween consecutive lists, 2011-2026", d["exits"].sum()),
        ("... aged 70 or more on their last list", d["exits_old"].sum()),
        ("Founders matched to new heirs\n(same country, last name or source)", d["matched_strict"].sum()),
        ("... of whom aged 70 or more", d["matched_strict_old"].sum()),
        ("Country-years with a matched succession,\nbillionaire countries", len(st)),
        ("... where heirs received more than 0.1% of GDP", int((st["pct_gdp"] > 0.1).sum())),
        ("... more than 1% of GDP", int((st["pct_gdp"] > 1.0).sum())),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    y = np.arange(len(steps))[::-1]
    vals = np.array([v for _, v in steps], dtype=float)
    ax.barh(y, vals, color=[MUTED, MUTED, BLUE, BLUE, ORANGE, ORANGE, ORANGE], height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v * 1.08, yi, f"{int(v):,}", va="center", fontsize=10, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(3, 4500)
    ax.set_yticks(y, [s for s, _ in steps], fontsize=9.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Count (log scale)")
    ax.set_title("From 1,650 departures to a few dozen usable successions")
    fig.tight_layout()
    save(fig, "fig1_funnel", SOURCE + " Grey: departures; blue: matched successions; orange: events in the estimation sample.")


def fig2_shock_sizes() -> None:
    sh = pd.read_csv(RES / "succession_shocks.csv")
    st = sh[sh["shock"] == "strict"].set_index(["iso3", "year"])
    old = sh[sh["shock"] == "old founders"].set_index(["iso3", "year"])["pct_gdp"]
    st["all_old"] = [abs(old.get(k, 0.0) - v) < 1e-9 for k, v in st["pct_gdp"].items()]
    st = st.reset_index()
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for flag, col, lab in ((True, BLUE, "every founder aged 70 or more"), (False, ORANGE, "includes a founder under 70")):
        s = st[st["all_old"] == flag]
        ax.scatter(s["year"] + rng.uniform(-0.18, 0.18, len(s)), s["pct_gdp"], s=46, color=col, edgecolor=SURFACE,
                   linewidth=1.2, zorder=3, label=lab)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.axhline(0.1, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(range(2012, 2027, 2))
    ax.set_xlabel("First Forbes list without the founder")
    ax.set_ylabel("Founder's wealth passing to heirs,\n% of the previous year's GDP")
    ax.set_ylim(0.002, 9)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.set_title("Most successions are small relative to the economy; a handful exceed 1% of GDP")
    fig.tight_layout()
    save(fig, "fig2_shock_sizes", SOURCE + " One point per country and list year, billionaire countries.")


def fig3_event_study() -> None:
    es = pd.read_csv(RES / "event_studies.csv")
    sel = (es["outcome"] == "tfp") & (es["sample"] == "billionaire countries")
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    for shock, col, dx, lab in (("strict", BLUE, -0.09, "successions (heirs matched on the next list)"),
                                ("old exits, any", ORANGE, 0.09, "every departure of a founder aged 70+, heirs found or not")):
        r = es[sel & (es["shock"] == shock)].iloc[0]
        b = np.array([r[f"b_k{k}"] for k in KS])
        s = np.array([r[f"se_k{k}"] for k in KS])
        x = np.array(KS) + dx
        ax.vlines(x, b - 1.96 * s, b + 1.96 * s, color=col, lw=2, alpha=0.55)
        ax.plot(x, b, color=col, lw=1.4, alpha=0.8)
        ax.scatter(x, b, s=46, color=col, edgecolor=SURFACE, linewidth=1.3, zorder=3, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvspan(-0.5, 2.5, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.text(1.0, ax.get_ylim()[1] * 0.92, "after the list that first\nomits the founder", ha="center", va="top", fontsize=8.5, color=INK2)
    ax.set_xticks(KS, ["-3", "-2", "-1\n(year of\ndeparture)", "0", "+1", "+2"])
    ax.set_xlabel("Calendar year of TFP growth relative to the first list without the founder")
    ax.set_ylabel("pp of annual TFP growth per log point\n(of 1 + wealth passing as % of GDP)")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("TFP growth around successions: no pre-trend, a fall afterwards, but only where heirs are visible")
    fig.tight_layout()
    save(fig, "fig3_event_study", SOURCE + " Billionaire countries, country and year effects, 95% intervals clustered by country.")


def fig4_permutation() -> None:
    dr = pd.read_csv(RES / "permutation_draws.csv")
    pm = pd.read_csv(RES / "permutation.csv")
    panels = [("strict", "Successions"), ("old founders", "Successions of founders aged 70+"),
              ("old exits, any", "Every departure of a founder aged 70+")]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0), sharey=True)
    for ax, (shock, title) in zip(axes, panels):
        col = f"{shock} | billionaire countries"
        r = pm[(pm["shock"] == shock) & (pm["sample"] == "billionaire countries")].iloc[0]
        ax.hist(dr[col], bins=40, color=MUTED, alpha=0.75, edgecolor=SURFACE)
        ax.axvline(r["observed_post"], color=BLUE if shock != "old exits, any" else ORANGE, lw=2.4)
        ax.set_title(f"{title}\nobserved {r['observed_post']:+.2f}; {r['p_le']:.1%} of draws at or below", fontsize=10)
        ax.set_xlabel("Cumulative effect, years 0 to +2")
    axes[0].set_ylabel("Draws (of 1,000)")
    fig.suptitle("Reshuffling each country's succession years: the observed effects sit in the lower tail, not beyond it",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig4_permutation", SOURCE + " TFP, billionaire countries; each draw permutes the timing of every country's shocks.")


def fig5_robustness() -> None:
    rb = pd.read_csv(RES / "robustness.csv")
    rb = rb[~rb["variant"].str.startswith("leave one")]
    variants = list(dict.fromkeys(rb["variant"]))
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    for shock, col, dy, lab in (("strict", BLUE, 0.15, "all successions"), ("old founders", ORANGE, -0.15, "founders aged 70+")):
        s = rb[rb["shock"] == shock].set_index("variant").loc[variants]
        y = np.arange(len(variants))[::-1] + dy
        ax.hlines(y, s["post"] - 1.96 * s["se_post"], s["post"] + 1.96 * s["se_post"], color=col, lw=2, alpha=0.55)
        ax.scatter(s["post"], y, s=46, color=col, edgecolor=SURFACE, linewidth=1.3, zorder=3, label=lab)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(np.arange(len(variants))[::-1], [v[0].upper() + v[1:] for v in variants], fontsize=9.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Cumulative effect on TFP growth, years 0 to +2 (pp per log point), 95% interval")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("The estimate survives most checks but not leaving out 2020-2021\nor controlling for pandemic exposure")
    fig.tight_layout()
    save(fig, "fig5_robustness", SOURCE + " Billionaire countries, country and year effects.")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig1_funnel()
    fig2_shock_sizes()
    fig3_event_study()
    fig4_permutation()
    fig5_robustness()


if __name__ == "__main__":
    main()
