"""The three figures of the article, drawn from results/.

python run_analysis.py > output.txt
python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
REST = "#c3c2b7"
SOURCE_PT = "Source: INE, Statistics Portugal; author's calculations. Code: github.com/DiogoRibeiro7/Medium-Blog"
SOURCE_EU = "Source: Eurostat city statistics (Urban Audit); author's calculations. Code: github.com/DiogoRibeiro7/Medium-Blog"

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


def finish(fig: plt.Figure, name: str, title: str, source: str) -> None:
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, source, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig1_core_shares() -> None:
    s = pd.read_csv(RES / "core_shares.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    fig.subplots_adjust(top=0.9, bottom=0.16)
    y = np.arange(len(s))[::-1]
    h = 0.38
    for off, col, colour in [(h / 2, "Lisboa", BLUE), (-h / 2, "Porto", ORANGE)]:
        ax.barh(y + off, s[col], height=h - 0.05, color=colour, label=f"{col} municipality, share of its metro area")
        for yy, v in zip(y, s[col], strict=True):
            ax.text(v + 0.8, yy + off, f"{v:.1f}%", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, s.index)
    ax.set_xlim(0, 80)
    ax.set_xlabel("% of the metro area (18 municipalities for Lisbon, 17 for Porto)")
    grid(ax, "x")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    finish(fig, "fig1_core_shares", "Lisbon and Porto: each centre's share of its metro area, by what is counted", SOURCE_PT)


# Label offsets (points) for the two highlighted FUAs, chosen so that the labels do not cross.
LABELS = {"PT001": ("Lisbon", (9, -13), "left"), "PT002": ("Porto", (-14, 10), "right")}


def scatter_base(ax: plt.Axes, e: pd.DataFrame, x: str, y: str) -> None:
    rest = e[e.country != "PT"]
    pt = e[e.country == "PT"]
    label = f"Other FUAs ({len(rest)}, {rest.country.nunique()} countries)"
    ax.scatter(rest[x], rest[y], s=22, color=REST, edgecolor=SURFACE, linewidth=0.6, label=label, zorder=2)
    ax.scatter(pt[x], pt[y], s=42, color=ORANGE, edgecolor=SURFACE, linewidth=1.0, label=f"Portuguese FUAs ({len(pt)})", zorder=3)
    for stem, (name, offset, ha) in LABELS.items():
        r = e.loc[stem]
        ax.scatter([r[x]], [r[y]], s=70, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=4)
        ax.annotate(name, (r[x], r[y]), xytext=offset, textcoords="offset points", fontsize=9.5, fontweight="bold", color=INK, ha=ha, zorder=5)


def fig2_homes_residents() -> None:
    e = pd.read_csv(RES / "europe_fuas.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(9, 6.2))
    lim = (-2.6, 2.9)
    cols = ["homes_centre_minus_ring", "res_centre_minus_ring"]
    if (e[cols].min().min() < lim[0]) or (e[cols].max().max() > lim[1]):
        raise ValueError("a point falls outside the axes of figure 2")
    ax.plot(lim, lim, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    note = "dashed line: residents per dwelling\ngrow as fast in the centre as in the ring"
    ax.text(-2.5, 1.55, note, fontsize=8.5, color=INK2, ha="left", va="center")
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
    scatter_base(ax, e, "homes_centre_minus_ring", "res_centre_minus_ring")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Dwellings: centre growth minus ring growth (% a year)")
    ax.set_ylabel("Residents: centre growth minus ring growth (% a year)")
    ax.text(-2.5, -0.25, "centre loses ground\nin homes and residents", fontsize=8.5, color=MUTED, va="center")
    ax.text(2.85, 0.95, "centre gains ground\nin homes and residents", fontsize=8.5, color=MUTED, ha="right", va="center")
    grid(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    finish(fig, "fig2_homes_and_residents", f"Across {len(e)} European urban areas, a centre's residents follow its dwellings", SOURCE_EU)


def fig3_country_benchmark() -> None:
    b = pd.read_csv(RES / "country_benchmark.csv", index_col=0)
    b = b[b.n >= 5].sort_values("national_employment_minus_centre_dwellings")
    names = {
        "ES": "Spain",
        "PT": "Portugal",
        "PL": "Poland",
        "FR": "France",
        "SK": "Slovakia",
        "DE": "Germany",
        "CH": "Switzerland",
        "FI": "Finland",
        "BG": "Bulgaria",
    }
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(top=0.9, bottom=0.15)
    y = np.arange(len(b))
    for yy, (_, r) in zip(y, b.iterrows(), strict=True):
        ax.plot([r.centre_dwellings, r.national_employment], [yy, yy], color=AXIS, linewidth=2.0, zorder=1)
    ax.scatter(b.national_employment, y, s=64, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3, label="Employment, national accounts")
    ax.scatter(b.centre_dwellings, y, s=64, color=AQUA, edgecolor=SURFACE, linewidth=1.2, zorder=3, label="Dwellings in the centres (Urban Audit)")
    ax.scatter(b.fua_jobs, y, s=58, marker="D", facecolor=SURFACE, edgecolor=ORANGE, linewidth=1.6, zorder=4, label="Jobs in the FUAs (Urban Audit)")
    ax.set_yticks(y, [f"{names.get(c, c)} ({int(n)})" for c, n in zip(b.index, b.n, strict=True)])
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=0)
    ax.set_xlabel("Median growth across the country's FUAs, % a year (about 2015 to 2021)")
    grid(ax, "x")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    pt = b.loc["PT"]
    ax.annotate(
        f"Portugal's city jobs series runs\n{pt.fua_jobs - pt.national_employment:.1f} points a year ahead of\nnational employment",
        (pt.fua_jobs, y[list(b.index).index("PT")]),
        xytext=(-12, -30),
        textcoords="offset points",
        fontsize=8.5,
        color=INK2,
        ha="right",
        va="top",
        arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 0.8},
    )
    source = SOURCE_EU.replace("city statistics (Urban Audit)", "city statistics (Urban Audit) and national accounts (nama_10_pe)")
    finish(fig, "fig3_country_benchmark", "Against national employment, centre dwellings lagged most in Spain and Portugal", source)


def main() -> None:
    fig1_core_shares()
    fig2_homes_residents()
    fig3_country_benchmark()
    print("wrote", ", ".join(sorted(p.name for p in FIG.glob("*.png"))))


if __name__ == "__main__":
    main()
