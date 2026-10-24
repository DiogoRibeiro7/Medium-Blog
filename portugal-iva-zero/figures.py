"""Draw the figures from the outputs of ``ivazero.py``.

Run after it::

    python ivazero.py
    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, PURPLE, GOLD, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#d9a21b", "#c8384b"
Q_LABELS = {"QU1": "Poorest fifth", "QU2": "Second", "QU3": "Middle", "QU4": "Fourth", "QU5": "Richest fifth"}

EVENTS = [("2023-03", "announced"), ("2023-05", "first full month"), ("2023-10", "end announced"), ("2024-01", "VAT back")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "lines.linewidth": 2.0, "legend.frameon": False, "legend.labelcolor": INK_2,
})


def finish(fig: plt.Figure, name: str, source: str, top: float = 1.0) -> None:
    fig.text(0.01, 0.01, f"Source: {source}; author's calculations.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, top))
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)


def suptitle(fig: plt.Figure, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK, y=0.985)


def to_dates(index) -> pd.DatetimeIndex:
    return pd.to_datetime([f"{p}-15" for p in index])


def mark_events(ax, labels: bool = True, which=None) -> None:
    for m, label in EVENTS:
        if which and m not in which:
            continue
        x = pd.Timestamp(f"{m}-01")
        ax.axvline(x, color=AXIS, lw=0.9, ls=":")
        if labels:
            ax.text(x, 1.0, f" {label}", transform=ax.get_xaxis_transform(), fontsize=7.5, color=MUTED,
                    va="top", rotation=90)


# 1. The treatment, read from INE's constant-tax index.
def tax_wedge(tw: pd.DataFrame, t: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = to_dates(tw.columns)
    for code, row in tw.iterrows():
        g = t.loc[code, "group"]
        colour = {"treated": ORANGE, "partial": GOLD, "oils_23": PURPLE}[g]
        ax.plot(x, 100 * row.values, color=colour, lw=1.4 if g != "partial" else 1.0, alpha=0.85)
    for g, colour, label in (("treated", ORANGE, "Fully covered (13 subclasses)"),
                             ("partial", GOLD, "Partly covered (12)"), ("oils_23", PURPLE, "Other edible oils (23%)")):
        ax.plot([], [], color=colour, label=label)
    ax.axhline(-100 * np.log(1.06), color=INK_2, lw=0.8, ls="--")
    ax.text(x[1], -100 * np.log(1.06) - 0.6, "log(1/1.06) = −5.83", fontsize=8, color=INK_2, va="top")
    mark_events(ax)
    ax.set_ylabel("log points")
    ax.set_ylim(-22, 2)
    ax.set_title("Tax wedge of Portuguese food subclasses: log HICP minus log HICP at constant taxes")
    ax.legend(loc="lower left", fontsize=8.5)
    finish(fig, "fig01_tax_wedge.png", "Eurostat prc_hicp_midx and prc_hicp_cind (INE); rebased to February 2023")


# 2. The raw price indices.
def baskets(bk: pd.DataFrame) -> None:
    b = 100 * bk.loc["2021-01":]
    x = to_dates(b.index)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(x, b["pt_treated"], color=ORANGE, label="Portugal, covered subclasses")
    ax.plot(x, b["pt_control"], color=BLUE, label="Portugal, untreated food subclasses")
    ax.plot(x, b["comparison_treated"], color=ORANGE, ls="--", lw=1.4, label="Comparison countries, same subclasses")
    ax.plot(x, b["comparison_control"], color=BLUE, ls="--", lw=1.4, label="Comparison countries, untreated")
    ax.axhline(0, color=INK_2, lw=0.8)
    mark_events(ax)
    ax.set_ylabel("log points relative to February 2023")
    ax.set_title("Food prices, fixed-weight baskets with Portuguese 2023 weights")
    ax.legend(loc="upper left", fontsize=8.5)
    finish(fig, "fig02_price_baskets.png", "Eurostat prc_hicp_midx and prc_hicp_inw; comparison: AT BE DE DK EL FI IT LT LU LV SE")


# 3. Event study.
def event_study(es: pd.DataFrame) -> None:
    e = es.loc["2021-06":]
    x = to_dates(e.index)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for col, colour, label in (("within", MUTED, "Within Portugal"), ("across", ORANGE, "Across countries")):
        ax.fill_between(x, 100 * (e[col] - 1.96 * e[f"{col}_se"]), 100 * (e[col] + 1.96 * e[f"{col}_se"]),
                        color=colour, alpha=0.18, lw=0)
        ax.plot(x, 100 * e[col], color=colour, lw=1.8, label=label)
    ax.step(x, 100 * e["tax_wedge"], where="mid", color=INK, lw=1.2, ls="--", label="Tax wedge (full pass-through)")
    ax.axhline(0, color=INK_2, lw=0.8)
    mark_events(ax)
    ax.set_ylabel("log points relative to February 2023")
    ax.set_title("Covered minus untreated food subclasses, with 95% intervals")
    ax.legend(loc="lower left", fontsize=8.5)
    finish(fig, "fig03_event_study.png", "Eurostat prc_hicp_midx and prc_hicp_cind; intervals clustered by subclass")


# 4. Dose-response.
def dose_response(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colours = {"treated": ORANGE, "partial": GOLD, "control": BLUE}
    for g, label in (("control", "Untreated"), ("partial", "Partly covered"), ("treated", "Fully covered")):
        s = d[d["group"] == g]
        ax.scatter(100 * s["wedge_entry"], 100 * s["effect_march_to_may"], color=colours[g], s=28, label=label,
                   zorder=3, alpha=0.9)
    x = 100 * d["wedge_entry"].to_numpy()
    y = 100 * d["effect_march_to_may"].to_numpy()
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(-6.2, 0.4, 50)
    ax.plot(xs, a + b * xs, color=INK, lw=1.2, label=f"Fitted line, slope {b:.2f}")
    ax.plot(xs, xs, color=INK_2, lw=0.9, ls="--", label="Full pass-through")
    tr = d[d["group"] == "treated"]
    for code in (tr["effect_march_to_may"].idxmin(), tr["effect_march_to_may"].idxmax()):
        r = d.loc[code]
        ax.annotate(r["label"], (100 * r["wedge_entry"], 100 * r["effect_march_to_may"]), xytext=(6, 0),
                    textcoords="offset points", fontsize=7.5, color=INK_2, va="center")
    ax.grid(axis="x", color=GRID)
    ax.set_xlabel("change in the tax wedge, March to May 2023, log points")
    ax.set_ylabel("price change relative to comparison countries")
    ax.set_title("Price change against the tax cut, every Portuguese food subclass")
    ax.legend(loc="upper left", fontsize=8.5)
    finish(fig, "fig04_dose_response.png", "Eurostat prc_hicp_midx and prc_hicp_cind")


# 5. Subclass effects.
def subclass_effects(d: pd.DataFrame) -> None:
    s = d[d["group"].isin(["treated", "partial"])].sort_values("effect_may_dec_2023")
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    y = np.arange(len(s))[::-1]
    ax.barh(y, 100 * s["effect_may_dec_2023"], color=[ORANGE if g == "treated" else GOLD for g in s["group"]],
            height=0.65)
    ax.scatter(100 * s["wedge_entry"], y, color=INK, marker="|", s=120, zorder=3, label="Tax wedge change")
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_yticks(y, s["label"], fontsize=8.5)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("log points, mean of May–December 2023, relative to February 2023 and to comparison countries")
    ax.set_title("Price effect by subclass: fully covered (orange) and partly covered (gold)")
    ax.legend(loc="upper right", fontsize=8.5)
    finish(fig, "fig05_subclass_effects.png", "Eurostat prc_hicp_midx and prc_hicp_cind")


# 6. Placebos.
def placebos(jumps: pd.DataFrame, es: pd.DataFrame, pcountries: pd.Series, actual_across: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    entry = es.loc["2023-05", "across"] - es.loc["2023-03", "across"]
    exit_ = es.loc["2024-01", "across"] - es.loc["2023-12", "across"]
    for ax, col, actual, title, label in (
            (axes[0], "across_2m", entry, "Two-month changes, 2019–2022", "March–May 2023"),
            (axes[1], "across_1m", exit_, "One-month changes, 2019–2022", "Dec 2023–Jan 2024")):
        v = 100 * jumps[col].dropna()
        ax.hist(v, bins=np.arange(-6, 6.5, 0.5), color=MUTED, alpha=0.85)
        ax.axvline(100 * actual, color=ORANGE, lw=2)
        ax.text(100 * actual + (0.15 if actual < 0 else -0.15), ax.get_ylim()[1] * 1.02, label, color=ORANGE,
                fontsize=8, va="bottom", ha="left" if actual < 0 else "right")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("change in the gap, log points")
    ax = axes[2]
    s = pd.concat([pcountries, pd.Series({"PT": actual_across})]).sort_values()
    ax.barh(np.arange(len(s)), 100 * s.values, color=[ORANGE if g == "PT" else MUTED for g in s.index], height=0.65)
    ax.set_yticks(np.arange(len(s)), s.index)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_title("Placebo countries, May–Dec 2023", fontsize=10)
    ax.set_xlabel("mean gap, log points")
    suptitle(fig, "The across-country estimator where nothing happened, against IVA Zero")
    finish(fig, "fig06_placebos.png", "Eurostat prc_hicp_midx and prc_hicp_cind", top=0.95)


# 7. Synthetic control.
def synthetic(sg: pd.DataFrame, weights: pd.Series) -> None:
    s = sg.loc["2021-01":]
    x = to_dates(s.index)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(x, 100 * s["portugal"], color=ORANGE, label="Portugal")
    ax.plot(x, 100 * s["synthetic"], color=INK_2, ls="--", lw=1.6,
            label="Synthetic Portugal (" + ", ".join(f"{g} {v:.2f}" for g, v in weights.sort_values(ascending=False).items() if v >= 0.05) + ")")
    mark_events(ax)
    ax.set_ylabel("log points")
    ax.set_title("Relative price of the covered basket (covered minus untreated food)")
    ax.legend(loc="center left", fontsize=8.5)
    ax = axes[1]
    ax.bar(x, 100 * s["gap"], width=20, color=[ORANGE if "2023-05" <= p <= "2023-12" else MUTED for p in s.index])
    ax.axhline(0, color=INK_2, lw=0.8)
    mark_events(ax, labels=False)
    ax.set_ylabel("gap")
    finish(fig, "fig07_synthetic_control.png", "Eurostat prc_hicp_midx and prc_hicp_inw; donor weights fitted on 2018-01 to 2023-02")


# 8. The reversal.
def reversal(ex: pd.DataFrame) -> None:
    e = ex.loc["2023-06":"2024-09"]
    x = to_dates(e.index)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(x, 100 * e["within"], color=MUTED, marker="o", ms=3.5, label="Within Portugal")
    ax.plot(x, 100 * e["across"], color=ORANGE, marker="o", ms=3.5, label="Across countries")
    ax.step(x, 100 * e["tax_wedge"], where="mid", color=INK, lw=1.2, ls="--", label="Tax wedge (full pass-through)")
    ax.axhline(0, color=INK_2, lw=0.8)
    mark_events(ax, which=["2023-10", "2024-01"])
    ax.set_ylabel("log points relative to October 2023")
    ax.set_title("Covered minus untreated food subclasses around the reversal")
    ax.legend(loc="lower right", fontsize=8.5)
    finish(fig, "fig08_reversal.png", "Eurostat prc_hicp_midx and prc_hicp_cind")


# 9. Headline inflation with and without the measure.
def headline(hc: pd.DataFrame) -> None:
    h = hc.loc["2022-06":"2025-06"]
    x = to_dates(h.index)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(x, h["actual"], color=INK, label="Published HICP")
    ax.plot(x, h["without_measure"], color=ORANGE, ls="--", label="Without the measure (estimated)")
    ax.fill_between(x, h["actual"], h["without_measure"], color=ORANGE, alpha=0.15, lw=0)
    mark_events(ax, which=["2023-05", "2024-01"])
    ax.set_ylabel("% change on a year earlier")
    ax.set_title("Portuguese HICP inflation, all items")
    ax.legend(loc="upper right", fontsize=8.5)
    finish(fig, "fig09_headline_inflation.png", "Eurostat prc_hicp_midx and prc_hicp_cind")


# 10. Incidence by income fifth.
def incidence(inc: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(inc))
    ax = axes[0]
    ax.bar(x, inc["saving_month"], color=ORANGE, width=0.6)
    for xi, v in zip(x, inc["saving_month"]):
        ax.text(xi, v + 0.2, f"€{v:.2f}", ha="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(x, [Q_LABELS[q] for q in inc.index], fontsize=8.5)
    ax.set_title("Saving per household and month, euros", fontsize=10)
    ax = axes[1]
    ax.bar(x, 100 * inc["saving_pct_consumption"], color=BLUE, width=0.6)
    for xi, v in zip(x, inc["saving_pct_consumption"]):
        ax.text(xi, 100 * v + 0.01, f"{100 * v:.2f}%", ha="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(x, [Q_LABELS[q] for q in inc.index], fontsize=8.5)
    ax.set_title("Saving as % of household consumption", fontsize=10)
    suptitle(fig, "Estimated saving from IVA Zero by income fifth, April 2023 – January 2024")
    finish(fig, "fig10_incidence.png",
           "Eurostat hbs_str_t223 and hbs_exp_t133 (2020 wave, fieldwork 2022/23), INE mean expenditure, HICP weights",
           top=0.95)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    for old in FIGURES.glob("fig*.png"):
        old.unlink()
    t = pd.read_csv(RESULTS / "treatment_by_subclass.csv", index_col="coicop")
    tw = pd.read_csv(RESULTS / "tax_wedge_pt.csv", index_col="coicop").loc[:, :"2024-12"]
    bk = pd.read_csv(RESULTS / "baskets.csv", index_col="period")
    es = pd.read_csv(RESULTS / "event_study.csv", index_col=0)
    d = pd.read_csv(RESULTS / "dose_response.csv", index_col="coicop")
    jumps = pd.read_csv(RESULTS / "placebo_jumps.csv", index_col=0)
    pcountries = pd.read_csv(RESULTS / "placebo_countries.csv", index_col=0)["mean_gap_may_dec_2023"]
    sg = pd.read_csv(RESULTS / "synthetic_gap.csv", index_col=0)
    sw = pd.read_csv(RESULTS / "synthetic_weights.csv", index_col=0)["weight"]
    ex = pd.read_csv(RESULTS / "event_study_reversal.csv", index_col=0)
    hc = pd.read_csv(RESULTS / "headline_counterfactual.csv", index_col="period")
    inc = pd.read_csv(RESULTS / "incidence_by_quintile.csv", index_col="quintile")
    policy = [f"2023-{m:02d}" for m in range(5, 13)]

    tax_wedge(tw, t)
    baskets(bk)
    event_study(es)
    dose_response(d)
    subclass_effects(d)
    placebos(jumps, es, pcountries, es.loc[policy, "across"].mean())
    synthetic(sg, sw)
    reversal(ex)
    headline(hc)
    incidence(inc)


if __name__ == "__main__":
    main()
