"""The five figures of the article, drawn from data/panel.csv and results/.

python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
REST = "#c3c2b7"
SOURCE = "Source: INE, Statistics Portugal; author's calculations. Code: github.com/DiogoRibeiro7/Medium-Blog"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    }
)

GROUP_ORDER = ["Lisboa", "Porto", "Lisbon metro ring", "Porto metro ring", "Medium cities", "Rest of mainland", "Islands"]


def finish(fig: plt.Figure, name: str, title: str) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, SOURCE, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def hgrid(ax: plt.Axes, axis: str = "x") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig1_where_growth_went() -> None:
    t = pd.read_csv(RES / "population_by_group.csv", index_col=0).reindex(GROUP_ORDER)
    dec = pd.read_csv(RES / "gibrat_deciles.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(t))[::-1]
    h = 0.36
    a.barh(y + h / 2, t.share_of_pop_2021, height=h - 0.04, color=BLUE, label="Share of residents, Dec 2021")
    a.barh(y - h / 2, t.share_of_growth_2125, height=h - 0.04, color=ORANGE, label="Share of growth, Dec 2021 to Dec 2025")
    for yy, v1, v2 in zip(y, t.share_of_pop_2021, t.share_of_growth_2125, strict=False):
        a.text(v1 + 0.6, yy + h / 2, f"{v1:.1f}%", va="center", fontsize=8, color=INK2)
        a.text(v2 + 0.6, yy - h / 2, f"{v2:.1f}%", va="center", fontsize=8, color=INK2)
    a.set_yticks(y, t.index)
    a.set_xlabel("% of Portugal")
    a.set_xlim(0, 40)
    hgrid(a)
    a.legend(loc="lower right", frameon=False, fontsize=8.5)
    a.set_title("Who holds the residents, who took the growth")

    for period, color, lab in [("2011-2021", BLUE, "2011 to 2021 census"), ("2021-2025", ORANGE, "Dec 2021 to Dec 2025")]:
        d = dec[dec.period == period]
        b.plot(
            d.decile,
            d.mean_growth_pct_year,
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=lab,
        )
        last = d.iloc[-1]
        b.text(10.25, last.mean_growth_pct_year, lab, va="center", fontsize=8.5, color=INK2)
    b.axhline(0, color=AXIS, linewidth=1)
    b.set_xticks(range(1, 11))
    b.set_xlim(0.6, 12.8)
    b.set_xlabel("Size decile of the municipality at the start (1 = smallest)")
    b.set_ylabel("Mean population growth, % a year")
    hgrid(b, "y")
    b.legend(loc="upper left", frameon=False, fontsize=8.5)
    b.set_title("Larger municipalities grow faster (mainland)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    finish(fig, "fig1_where_growth_went", "Growth went to the Lisbon area, and to the largest municipalities")


def fig2_infrastructure() -> None:
    t = pd.read_csv(RES / "infrastructure_by_group.csv", index_col=0).reindex(GROUP_ORDER)
    panels = [
        ("pre_2025", "Pre-school enrolments per 100 resident children aged 3-5,\n2024/25 (dashed line: one per child)", 100, "%.0f"),
        ("commute_collective_min", "Mean commute by public transport, minutes, 2021", None, "%.0f"),
        ("beds", "Public (SNS and PPP) hospital beds per 1,000 residents, 2024\n(each metro area taken whole: beds sit in the core)", None, "%.1f"),
        ("price_to_income_2024", "Years of median household income for 90 m², 2024", None, "%.1f"),
    ]
    p = pd.read_csv(HERE / "data" / "panel.csv", dtype={"dico": str})
    areas = {
        "Lisbon metro area": ["Lisboa", "Lisbon metro ring"],
        "Porto metro area": ["Porto", "Porto metro ring"],
        "Medium cities": ["Medium cities"],
        "Rest of mainland": ["Rest of mainland"],
        "Islands": ["Islands"],
    }
    beds = pd.Series({k: 1000 * p[p.group.isin(v)].sns_beds_2024.sum() / p[p.group.isin(v)].pop_est_2024.sum() for k, v in areas.items()})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    rings = {"Lisbon metro ring", "Porto metro ring"}
    for ax, (col, title, ref, fmt) in zip(axes.flat, panels, strict=False):
        series = beds if col == "beds" else t[col]
        y = np.arange(len(series))[::-1]
        colors = [BLUE if g in rings else REST for g in series.index]
        ax.barh(y, series, height=0.62, color=colors)
        for yy, v in zip(y, series, strict=False):
            ax.text(v + 0.01 * series.max(), yy, fmt % v, va="center", fontsize=8.5, color=INK2)
        if ref is not None:
            ax.axvline(ref, color=INK2, linewidth=1, linestyle=(0, (3, 2)))
        ax.set_yticks(y, series.index)
        ax.set_xlim(0, series.max() * 1.18)
        hgrid(ax)
        ax.set_title(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    finish(fig, "fig2_infrastructure_by_group", "The metro rings (blue) grew without the schools and transport their residents need")


def fig3_saturation_vs_growth() -> None:
    m = pd.read_csv(RES / "saturation.csv", dtype={"dico": str}).set_index("dico")
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    rest = m[~m.group.isin(["Lisboa", "Lisbon metro ring", "Porto", "Porto metro ring", "Medium cities"])]
    ax.scatter(rest.saturation, rest.growth_2125, s=22, color=REST, edgecolor=SURFACE, linewidth=0.8, label="Rest of mainland", zorder=2)
    for groups, color, lab in [
        (["Lisboa", "Lisbon metro ring"], BLUE, "Lisbon metro area"),
        (["Porto", "Porto metro ring"], ORANGE, "Porto metro area"),
        (["Medium cities"], AQUA, "Medium cities"),
    ]:
        d = m[m.group.isin(groups)]
        ax.scatter(d.saturation, d.growth_2125, s=46, color=color, edgecolor=SURFACE, linewidth=1.5, label=lab, zorder=3)
    bins = pd.qcut(m.saturation, 10)
    binned = m.groupby(bins, observed=True)[["saturation", "growth_2125"]].mean()
    ax.plot(binned.saturation, binned.growth_2125, color=INK2, linewidth=2, zorder=4, label="Mean of each saturation decile")
    offsets = {"Lisboa": (6, -11), "Viseu": (-30, 6)}
    for name in ["Moita", "Seixal", "Montijo", "Lisboa", "Porto", "Coimbra", "Braga", "Portimão", "Alenquer", "Viseu"]:
        r = m[m.name == name].iloc[0]
        ax.annotate(name, (r.saturation, r.growth_2125), xytext=offsets.get(name, (5, 4)), textcoords="offset points", fontsize=8, color=INK)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("Saturation index, 2021 (equal weights; higher = more pressure on housing, schools, transport, hospitals)")
    ax.set_ylabel("Population growth, Dec 2021 to Dec 2025, % a year")
    hgrid(ax, "y")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    finish(fig, "fig3_saturation_vs_growth", "After 2021, people kept moving to the places that were already under the most pressure")


def fig4_rank_uncertainty() -> None:
    m = pd.read_csv(RES / "saturation.csv", dtype={"dico": str}).set_index("dico")
    top = m.nsmallest(30, "rank").iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8.2))
    y = np.arange(len(top))
    color_of = {"Lisbon metro ring": BLUE, "Porto metro ring": ORANGE, "Medium cities": AQUA}
    for yy, r in zip(y, top.itertuples(), strict=False):
        c = color_of.get(r.group, MUTED)
        ax.plot([r.rank_p05, r.rank_p95], [yy, yy], color=c, linewidth=2, solid_capstyle="round")
        ax.scatter([r.rank], [yy], s=40, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_yticks(
        y,
        [
            f"{n} ({g.replace(' metro ring', ' ring').replace('Rest of mainland', 'outside metros').replace('Medium cities', 'medium city')})"
            for n, g in zip(top.name, top.group, strict=False)
        ],
        fontsize=8.5,
    )
    ax.axvline(30, color=INK2, linewidth=1, linestyle=(0, (3, 2)))
    ax.text(31, len(top) - 0.6, "top 30", fontsize=8, color=INK2)
    ax.set_xlabel(
        "Rank among 278 mainland municipalities (1 = most saturated). Dot: equal weights; line: 5th to 95th percentile over 5,000 random weightings"
    )
    ax.set_xlim(0, 125)
    hgrid(ax)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    finish(fig, "fig4_rank_uncertainty", "Which places are most saturated depends on the weights, but the Setúbal peninsula stays near the top")


def fig5_what_to_build() -> None:
    c = pd.read_csv(RES / "candidate_nodes.csv", dtype={"dico": str}).set_index("dico")
    med = c[c.role == "medium city"].sort_values("saturation")
    ring = c[c.role == "metro ring"].sort_values("saturation", ascending=False).head(12)
    dims = ["housing", "schools", "transport", "health"]
    blank = pd.DataFrame(np.nan, index=["__gap__"], columns=dims)
    mat = pd.concat([med[dims], blank, ring[dims]])
    labels = list(med.name) + [""] + list(ring.name)
    cmap = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b8322f"])
    fig, ax = plt.subplots(figsize=(8.2, 11.5))
    im = ax.imshow(mat.values, cmap=cmap, vmin=-2.5, vmax=2.5, aspect="auto")
    for i in range(mat.shape[0]):
        if labels[i] == "":
            continue
        row = mat.iloc[i]
        j = int(np.nanargmax(row.values))
        for jj in range(len(dims)):
            v = row.iloc[jj]
            weight = "bold" if jj == j else "normal"
            ax.text(jj, i, f"{v:+.1f}", ha="center", va="center", fontsize=8, fontweight=weight, color="#ffffff" if abs(v) > 1.6 else INK)
    ax.set_xticks(range(len(dims)), ["Housing", "Schools", "Transport", "Hospitals"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(labels)), labels, fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    gap = len(med)
    ax.set_title("Medium cities, least saturated first", pad=26, fontsize=9.5)
    ax.text(-0.6, gap + 0.15, "Twelve most saturated metro-ring municipalities", fontsize=9, fontweight="bold", color=INK, ha="left", va="center")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.025, pad=0.02)
    cb.set_label("Pressure score (z units): blue = slack, red = pressure. Bold = the binding constraint of each row", color=INK2, fontsize=8.5)
    cb.outline.set_visible(False)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    finish(fig, "fig5_what_to_build_where", "What to build where: the binding constraint of each candidate")


def main() -> None:
    fig1_where_growth_went()
    fig2_infrastructure()
    fig3_saturation_vs_growth()
    fig4_rank_uncertainty()
    fig5_what_to_build()
    print("figures written to", FIG.relative_to(HERE))


if __name__ == "__main__":
    main()
