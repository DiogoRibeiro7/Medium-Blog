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
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
BLUE_LIGHT, BLUE_DARK = "#86b6ef", "#184f95"
INK, INK2, MUTED, GRID, SURFACE, SHADE, GREY = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb", "#efeeea", "#b5b4ae"
CREDIT = "Diogo Ribeiro"
SOURCE = "Source: Eurostat Comext DS-045409, EU-27 imports by HS 6-digit product and partner, 2002-2025"
MODEL_COLOR = {"trajectory": BLUE, "boosting": ORANGE, "snapshot": AQUA, "near_miss": YELLOW, "expected_value": MAGENTA}
MODEL_NAME = {"trajectory": "Trajectory model (logistic)", "boosting": "Gradient-boosted trees", "snapshot": "Snapshot model (levels only)", "near_miss": "Near-miss rule", "expected_value": "Boosted trees x value at stake"}
ORDER = ("trajectory", "boosting", "snapshot", "near_miss")
ORDER_ALL = ORDER + ("expected_value",)
COUNTRY = {"CN": "China", "US": "United States", "RU": "Russia", "GB": "United Kingdom", "CH": "Switzerland", "NO": "Norway", "TR": "Türkiye", "JP": "Japan", "KR": "South Korea", "IN": "India", "TW": "Taiwan", "VN": "Viet Nam", "BR": "Brazil", "UA": "Ukraine", "BY": "Belarus", "QA": "Qatar", "DZ": "Algeria", "MA": "Morocco", "ZA": "South Africa", "CL": "Chile", "AU": "Australia", "CA": "Canada", "ID": "Indonesia", "TH": "Thailand", "MY": "Malaysia", "SG": "Singapore", "MX": "Mexico"}

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
    fig.text(0.01, y, f"{text}   |   {CREDIT}", ha="left", va="top", fontsize=8.2, color=MUTED, wrap=True)


def suptitle(fig, text: str, y: float = 1.02) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=y)


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def read(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(RES / name, keep_default_na=False, na_values=[""], **kw)


# 1. how many products the screen flags, and how many enter it for good each year
def fig1_flows() -> None:
    f = read("panel_flows.csv")
    f = f[f["year"] >= 2003]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"wspace": 0.28})
    a.plot(f["year"], f["flagged"], color=BLUE)
    a.set_title("Products flagged by the single-supplier screen")
    a.set_ylim(0, f["flagged"].max() * 1.15)
    last = f.iloc[-1]
    a.annotate(f"{int(last['flagged']):,}", (last["year"], last["flagged"]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    first = f.iloc[0]
    a.annotate(f"{int(first['flagged']):,}", (first["year"], first["flagged"]), xytext=(-4, 8), textcoords="offset points", ha="left", fontsize=9, color=INK2)
    w = 0.38
    b.bar(f["year"] - w / 2, f["entries"], width=w, color=BLUE_LIGHT, label="Crossed into the screen")
    dur = f[(f["year"] >= 2005) & (f["year"] <= 2024)]
    b.bar(dur["year"] + w / 2, dur["durable_entries"], width=w, color=BLUE, label="Durable entries (2005-2024)")
    b.set_title("Entries per year")
    b.legend(loc="upper left")
    b.set_ylim(0, f["entries"].max() * 1.25)
    for ax in (a, b):
        ax.set_xlim(2002.3, 2025.7)
    suptitle(fig, "Many products cross the line every year; far fewer stay across it")
    note(fig, "A durable entry is a product flagged in two consecutive years after three years unflagged, so it can be dated only from 2005 to 2024. The screen: largest outside supplier > 40% of extra-EU imports,\nextra-EU imports > 50% of all EU imports, extra-EU imports > EU exports. " + SOURCE, y=-0.03)
    save(fig, "fig1_flows")


# 2. the run-up to a durable entry
def fig2_run_up() -> None:
    q = read("panel_event_paths.csv")
    q = q[q["rel"] <= 2]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), gridspec_kw={"wspace": 0.3})
    a, b, c = axes
    a.fill_between(q["rel"], q["share_q25"], q["share_q75"], color=BLUE_LIGHT, alpha=0.45, lw=0, label="Middle half of products")
    a.plot(q["rel"], q["share_med"], color=BLUE, label="Median")
    a.axhline(0.4, color=INK2, lw=1, ls="--")
    a.text(q["rel"].min(), 0.415, "40% threshold", fontsize=8.5, color=INK2)
    a.set_title("Share of the entering supplier")
    a.set_ylim(0, 1)
    a.legend(loc="lower right", fontsize=8.5)
    b.plot(q["rel"], q["extra_med"], color=BLUE)
    b.axhline(0.5, color=INK2, lw=1, ls="--")
    b.set_title("Extra-EU share of EU imports (median)")
    b.set_ylim(0, 1)
    c.plot(q["rel"], q["ratio_med"], color=BLUE)
    c.axhline(1.0, color=INK2, lw=1, ls="--")
    c.set_ylim(0, 1.6)
    c.set_title("Extra-EU imports / EU exports (median)")
    for ax in axes:
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_xlabel("Years relative to entry")
        ax.set_xticks(range(int(q["rel"].min()), 3))
    suptitle(fig, "Before entry the supplier's share climbs for years; the import-export test flips at the end")
    note(fig, "Durable entries 2005-2024 (all products). Lines are medians across products at each year relative to entry; the dashed lines are the screen thresholds.\nThe jump in year 0 is partly mechanical: entry is defined by crossing the thresholds. " + SOURCE, y=-0.04)
    save(fig, "fig2_run_up")


# 3. which criterion arrives last, and how long the supplier had already been dominant
def fig3_last_criterion() -> None:
    e = read("panel_events.csv", dtype={"product": str})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"wspace": 0.35, "width_ratios": [1, 1.2]})
    names = {"c_top": "Largest supplier > 40%", "c_extra": "Extra-EU share > 50%", "c_ratio": "Imports > EU exports", "joint": "Two or more together"}
    v = e["last"].map(names).value_counts().reindex(list(names.values())).fillna(0)
    a.barh(range(len(v)), v.values / len(e), color=BLUE, height=0.55)
    a.set_yticks(range(len(v)), v.index)
    a.invert_yaxis()
    a.grid(axis="x", color=GRID)
    a.grid(axis="y", visible=False)
    a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    for i, x in enumerate(v.values / len(e)):
        a.text(x + 0.01, i, f"{x:.0%}", va="center", fontsize=9, color=INK2)
    a.set_xlim(0, max(v.values / len(e)) * 1.25)
    a.set_title("Criterion met last, in the entry year")
    yrs = (e["entry"] - e["since_c_top"]).clip(upper=10)
    counts = yrs.value_counts().reindex(range(0, 11)).fillna(0) / len(e)
    b.bar(counts.index, counts.values, color=BLUE, width=0.7)
    b.set_xticks(range(0, 11), [str(i) for i in range(10)] + ["10+"])
    b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    b.set_xlabel("Years the largest-supplier criterion had already been met")
    b.set_title("How long the 40% line had already been crossed")
    suptitle(fig, "The import-export test is the one most often met last")
    note(fig, "Durable entries 2005-2024. Years counted back from the entry year over the unbroken run in which the criterion was met; runs that begin in a code's first\nyear in the data are cut at that year. " + SOURCE, y=-0.04)
    save(fig, "fig3_last_criterion")


# 4. the rolling out-of-time design
def fig4_design() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.set_xlim(2001.0, 2026.5)
    ax.set_ylim(-0.8, 3.7)
    ax.axis("off")
    ax.grid(False)

    def box(x0, x1, y, **kw):
        ax.add_patch(FancyBboxPatch((x0 - 0.42, y - 0.17), (x1 - x0) + 0.84, 0.34, boxstyle="round,pad=0.0,rounding_size=0.08", **kw))

    for T, y in [(2013, 2.8), (2017, 1.6), (2021, 0.4)]:
        box(2005, T - 4, y, fc=BLUE_LIGHT, ec="none")
        box(T - 3, T - 1, y, fc=SHADE, ec="none")
        box(T, T, y, fc=BLUE, ec="none")
        box(T + 1, T + 3, y, fc="none", ec=ORANGE, lw=1.5)
        ax.text(2004.3, y, f"Scored in {T}", ha="right", va="center", fontsize=9.5, color=INK)
        ax.text((2005 + T - 4) / 2, y + 0.3, f"training origins 2005-{T - 4}", ha="center", fontsize=8.5, color=INK2)
        ax.text(T - 2, y - 0.38, "outcome pending", ha="center", fontsize=8.0, color=MUTED)
        ax.text(T + 2, y + 0.3, f"entry {T + 1}-{T + 3}?", ha="center", fontsize=8.5, color=ORANGE)
    for yr in range(2004, 2027, 2):
        ax.text(yr, -0.7, str(yr), ha="center", fontsize=8.5, color=INK2)
    suptitle(fig, "Each year's watchlist is scored by a model that could have existed that year", y=1.0)
    note(fig, "Light blue: origins whose outcome was known by the scoring year T (t <= T - 4). Grey: origins whose three-year window had not closed. Dark blue: the scoring year,\nwith features from T-3 to T. Orange: the window in which an entry is predicted. Three of the twelve scoring years with known outcomes are drawn.   " , y=0.02)
    save(fig, "fig4_design")


# 5. discrimination by scoring year
def fig5_by_origin() -> None:
    m = read("warning_metrics.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"wspace": 0.25})
    for k in ORDER:
        g = m[m["model"] == k]
        a.plot(g["origin"], g["auc"], color=MODEL_COLOR[k], marker="o", ms=4, label=MODEL_NAME[k])
        b.plot(g["origin"], g["recall_200"], color=MODEL_COLOR[k], marker="o", ms=4)
    a.set_title("Area under the ROC curve")
    a.set_ylim(0.5, 1.0)
    b.set_title("Share of next three years' entries on a 200-product list")
    b.set_ylim(0, 1)
    b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    a.legend(loc="lower left", fontsize=8.5)
    for ax in (a, b):
        ax.set_xlabel("Scoring year")
    suptitle(fig, "Out of time, the fitted models rank future entries better than the near-miss rule")
    note(fig, "Rolling origins; each year's models are fitted only on origins whose outcomes were known by then.\n" + SOURCE, y=-0.03)
    save(fig, "fig5_by_origin")


def recall_curve(p: pd.DataFrame, model: str, ks: np.ndarray, weight: str | None = None) -> np.ndarray:
    out = []
    for k in ks:
        r = []
        for _, g in p.groupby("year"):
            w = g[weight].fillna(0) if weight else pd.Series(1.0, index=g.index)
            pos = (g["y"] == 1)
            if w[pos].sum() == 0:
                continue
            r.append(w[pos & (g[f"rank_{model}"] <= k)].sum() / w[pos].sum())
        out.append(np.mean(r))
    return np.array(out)


# 6. recall against watchlist size, by count and by value
def fig6_recall() -> None:
    p = read("warning_predictions.csv.gz", dtype={"product": str})
    p = p[p["y"].notna()]
    ks = np.array([25, 50, 100, 150, 200, 300, 400, 600, 800, 1000])
    n = p.groupby("year").size().mean()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"wspace": 0.22})
    for k in ORDER_ALL:
        a.plot(ks, recall_curve(p, k, ks), color=MODEL_COLOR[k], label=MODEL_NAME[k])
        b.plot(ks, recall_curve(p, k, ks, "entry_top_value"), color=MODEL_COLOR[k])
    a.plot(ks, ks / n, color=GREY, lw=1.2, ls=":", label="Random list")
    b.plot(ks, ks / n, color=GREY, lw=1.2, ls=":")
    a.set_title("Share of entries caught")
    b.set_title("Share of entering import value caught")
    for ax in (a, b):
        ax.set_xlabel("Size of the watchlist (products)")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        ax.axvline(200, color=MUTED, lw=1, ls="--")
    a.legend(loc="upper left", fontsize=8.5)
    suptitle(fig, "Ranked by probability, a short list catches entries; ranked by value at stake, it catches their value")
    note(fig, f"Averages over scoring years 2010-2021 of the share of entries within three years that were on the list; about {n:,.0f} unflagged products are ranked each year.\nValue is the entering supplier's imports in the entry year; value at stake is the largest supplier's imports in the scoring year. " + SOURCE, y=-0.03)
    save(fig, "fig6_recall")


# 7. lead time
def fig7_lead() -> None:
    ld = read("warning_leads.csv", dtype={"product": str})
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    shades = {3: BLUE_DARK, 2: BLUE, 1: BLUE_LIGHT, 0: SHADE}
    labels = {3: "3 years ahead", 2: "2 years ahead", 1: "1 year ahead", 0: "Not on the list"}
    for i, k in enumerate(ORDER_ALL):
        c = ld[f"lead_{k}"].value_counts(normalize=True).reindex([3, 2, 1, 0]).fillna(0)
        left = 0.0
        for lead, share in c.items():
            ax.barh(i, share, left=left, color=shades[lead], height=0.6, edgecolor=SURFACE, lw=2, label=labels[lead] if i == 0 else None)
            if share > 0.05:
                ax.text(left + share / 2, i, f"{share:.0%}", ha="center", va="center", fontsize=9, color=SURFACE if lead >= 2 else INK)
            left += share
        w = ld["top_value"].fillna(0)
        ax.text(1.015, i, f"{(w * (ld[f'lead_{k}'] > 0)).sum() / w.sum():.0%}", va="center", fontsize=9.5, color=INK, transform=ax.get_yaxis_transform())
    ax.text(1.015, -0.62, "by value", va="center", fontsize=8.5, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(ORDER_ALL)), [MODEL_NAME[k] for k in ORDER_ALL])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=9)
    suptitle(fig, "How early each score first put a future entry on its 200-product list", y=1.03)
    note(fig, f"{len(ld):,} durable entries {int(ld['entry'].min())}-{int(ld['entry'].max())}; earliest of the three scoring years before entry at which the product ranked in the top 200.\nRight: share of the entering supplier's imports (entry year) that was on the list at least once. " + SOURCE, y=-0.2)
    save(fig, "fig7_lead")


# 8. where the signal is
def fig8_features() -> None:
    abl = read("warning_ablation.csv")
    perm = read("warning_permutation.csv")
    perm = perm[perm["group"] != "none"].sort_values("auc_drop")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"wspace": 0.95, "width_ratios": [1, 1.1]})
    a.scatter(abl["auc"], range(len(abl)), color=BLUE, s=60, zorder=3)
    a.set_yticks(range(len(abl)), abl["features"])
    a.invert_yaxis()
    a.grid(axis="x", color=GRID)
    a.grid(axis="y", visible=False)
    for i, (ap, auc) in enumerate(zip(abl["ap"], abl["auc"])):
        a.text(auc + 0.0015, i, f"{auc:.3f}  (AP {ap:.3f})", va="center", fontsize=8.5, color=INK2)
    a.set_xlim(0.86, 0.91)
    a.set_title("Logistic model, adding feature groups (AUC; axis from 0.86)")
    b.barh(range(len(perm)), perm["auc_drop"], color=[BLUE if v > 0 else GREY for v in perm["auc_drop"]], height=0.55)
    b.set_yticks(range(len(perm)), perm["group"])
    b.axvline(0, color=INK2, lw=1)
    b.grid(axis="x", color=GRID)
    b.grid(axis="y", visible=False)
    for i, v in enumerate(perm["auc_drop"]):
        b.text(max(v, 0) + 0.003, i, f"{v:+.3f}", va="center", fontsize=8.5, color=INK2)
    b.set_xlim(min(perm["auc_drop"].min(), 0) - 0.005, perm["auc_drop"].max() * 1.25)
    b.set_title("Boosted trees: AUC lost when a group is shuffled")
    suptitle(fig, "The signal sits in where a product stands, above all its imports against EU exports")
    note(fig, "Left: logistic model, pooled out-of-time scores 2010-2021, feature groups added in order. Right: model fitted on origins up to 2016 and scored on 2020-2021;\neach group's columns are shuffled together, 20 times. " + SOURCE, y=-0.04)
    save(fig, "fig8_features")


CASES: list[tuple[str, str]] = [
    ("850760", "Lithium-ion batteries (850760)"),
    ("847130", "Laptops (847130)"),
    ("271111", "Liquefied natural gas (271111)"),
    ("250410", "Natural graphite powder (250410)"),
    ("810194", "Unwrought tungsten (810194)"),
    ("292429", "Cyclic amides, incl. paracetamol (292429)"),
]


# 9. cases
def fig9_cases() -> None:
    if not CASES:
        return
    d = read("panel.csv.gz", dtype={"product": str, "top_partner": str})
    d["flagged"] = d["flagged"].astype(str).str.lower().eq("true")
    p = read("warning_predictions.csv.gz", dtype={"product": str})
    n = len(CASES)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12.5, 3.3 * rows), sharex=True, gridspec_kw={"hspace": 0.55, "wspace": 0.22})
    for ax, (code, name) in zip(axes.flat, CASES):
        g = d[d["product"] == code].set_index("year")
        for y, fl in g["flagged"].items():
            if fl:
                ax.axvspan(y - 0.5, y + 0.5, color=SHADE, lw=0)
        ax.plot(g.index, g["top_share"], color=INK2, lw=1.5)
        ax.axhline(0.4, color=MUTED, lw=1, ls="--")
        pr = p[p["product"] == code]
        on = pr[pr["rank_boosting"] <= 200]
        ax.scatter(on["year"], g["top_share"].reindex(on["year"]).to_numpy(), color=BLUE, s=36, zorder=3, edgecolor=SURFACE, lw=1)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlim(2002, 2025.5)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    suptitle(fig, "Six products: when the model listed them and when the screen flagged them", y=1.0)
    note(fig, "Line: largest outside supplier's share of extra-EU imports. Grey bands: years flagged by the screen. Blue dots: years the boosted-tree model ranked the product\nin its top 200 (scoring years 2010-2025; a product is scored only when unflagged for three years). " + SOURCE, y=0.0)
    save(fig, "fig9_cases")


def main() -> int:
    for f in (fig1_flows, fig2_run_up, fig3_last_criterion, fig4_design, fig5_by_origin, fig6_recall, fig7_lead, fig8_features, fig9_cases):
        f()
        print(f.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
