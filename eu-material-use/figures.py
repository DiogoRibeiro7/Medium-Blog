"""The five figures, drawn from results/ (run materials_numbers.py first).

python figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIG = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SOURCE = "Eurostat (snapshot in DiogoRibeiro7/data, eurostat-material-use). Figure: Diogo Ribeiro."

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "legend.labelcolor": INK2,
    }
)
NAMES = {"TOTAL": "All materials", "MF1": "Biomass", "MF2": "Metal ores", "MF3": "Non-metallic minerals", "MF4": "Fossil energy materials"}


def heading(fig, title, subtitle):
    fig.text(0.01, 0.975, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.93, subtitle, fontsize=9.5, color=INK2, va="top")


def footer(fig):
    fig.text(0.01, 0.008, f"Source: {SOURCE}", fontsize=8, color=MUTED, ha="left", va="bottom")


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def fig1_index():
    d = pd.read_csv(RESULTS / "eu_index.csv")
    fig, ax = plt.subplots(figsize=(9, 5.6))
    fig.subplots_adjust(left=0.08, right=0.76, top=0.84, bottom=0.10)
    series = (
        ("gdp", INK, "GDP (volume)"),
        ("dmc_nonfossil", BLUE, "Material use without\nfossil fuels"),
        ("dmc", AQUA, "Material use (DMC)"),
        ("rmc", YELLOW, "Material footprint (RMC)"),
        ("dmc_fossil", ORANGE, "Fossil energy materials"),
    )
    offsets = {"dmc": -2, "rmc": -9, "dmc_nonfossil": 5}
    for col, color, label in series:
        ax.plot(d.year, d[col], color=color, linewidth=2.6 if col == "gdp" else 2.0)
        ax.annotate(f"{label} {d[col].iloc[-1]:.0f}", (d.year.iloc[-1], d[col].iloc[-1]), xytext=(8, offsets.get(col, 0)),
                    textcoords="offset points", va="center", fontsize=9.5, color=INK2, annotation_clip=False)  # fmt: skip
    ax.axhline(100, color=MUTED, linewidth=1)
    ax.set_xlim(2000, 2024)
    ax.set_ylabel("Index, 2000 = 100")
    heading(fig, "The EU's material use fell because it burned less fossil fuel",
            "EU27, 2000 to 2024. Domestic material consumption (DMC) by group, the footprint in raw material equivalents, and GDP")  # fmt: skip
    footer(fig)
    save(fig, "fig1_index")


def fig2_drift():
    p = pd.read_csv(RESULTS / "panel_changes.csv").dropna(subset=["d", "y"])
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    fig.subplots_adjust(left=0.11, right=0.96, top=0.86, bottom=0.11)
    pt = p[p.geo == "PT"]
    rest = p[p.geo != "PT"]
    ax.scatter(100 * rest.y, 100 * rest.d, s=14, color=BLUE, alpha=0.4, edgecolor="none", label="A member state in one year")
    ax.scatter(100 * pt.y, 100 * pt.d, s=30, color=ORANGE, edgecolor=SURFACE, linewidth=1, label="Portugal", zorder=3)
    q = p.copy()
    q["yd"] = q.y - q.groupby("geo").y.transform("mean")
    q["dd"] = q.d - q.groupby("geo").d.transform("mean")
    b = (q.yd * q.dd).sum() / (q.yd**2).sum()
    a = (q.groupby("geo").d.mean() - b * q.groupby("geo").y.mean()).mean()
    x = np.linspace(-10, 10, 2)
    ax.plot(x, 100 * a + b * x, color=INK, linewidth=1.8, label=f"Typical country: {100 * a:+.1f}% + {b:.2f} x GDP growth")
    be = -100 * a / b
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.plot([be], [0], marker="o", color=INK, markersize=7, zorder=4)
    ax.annotate(f"Flat at {be:.1f}% growth", (be, 0), xytext=(10, -22), textcoords="offset points", fontsize=9.5, color=INK,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": SURFACE, "edgecolor": MUTED, "linewidth": 0.6})  # fmt: skip
    ax.set_xlim(-10, 10)
    ax.set_ylim(-30, 30)
    ax.set_xlabel("GDP growth (%)")
    ax.set_ylabel("Change in material use, DMC (%)")
    ax.legend(loc="upper left", fontsize=9)
    heading(fig, "Material use follows growth year to year, around a slow downward drift",
            "27 member states, 2001 to 2024: yearly log changes, in percent. Line fitted with a separate intercept per country")  # fmt: skip
    footer(fig)
    save(fig, "fig2_drift")


def fig3_footprint():
    c = pd.read_csv(RESULTS / "countries_footprint.csv")
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    fig.subplots_adjust(left=0.11, right=0.96, top=0.86, bottom=0.11)
    ax.scatter(100 * c.dmc_change, 100 * c.rmc_change, s=46, color=BLUE, edgecolor=SURFACE, linewidth=1, zorder=3)
    for _, r in c.iterrows():
        ax.annotate(r.geo, (100 * r.dmc_change, 100 * r.rmc_change), xytext=(4, 3), textcoords="offset points", fontsize=8.5, color=INK2)
    x = np.linspace(-95, 70, 2)
    ax.plot(x, x, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(44, 26, "Footprint and DMC\nchanged alike", fontsize=9, color=INK2)
    ax.text(-90, 40, "Above the line: the footprint fell\nless than DMC (or rose more)", fontsize=9, color=INK2)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlim(-95, 70)
    ax.set_ylim(-95, 70)
    ax.set_xlabel("Change in domestic material consumption, log points")
    ax.set_ylabel("Change in material footprint (RMC), log points")
    heading(fig, "Counting imports at their raw-material weight changes little",
            "Member states, 2008 to 2024. The footprint counts the raw materials behind imports and exports")  # fmt: skip
    footer(fig)
    save(fig, "fig3_footprint")


def fig4_fates():
    d = pd.read_csv(RESULTS / "processed_fates.csv")
    order = ["TOTAL", "MF3", "MF1", "MF4", "MF2"]
    d = d.set_index("code").loc[order[::-1]]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.subplots_adjust(left=0.24, right=0.96, top=0.80, bottom=0.17)
    y = np.arange(len(d))
    left = np.zeros(len(d))
    parts = (("accumulated", BLUE, "Added to stocks (buildings, infrastructure, goods)"),
             ("other_material_use", AQUA, "Other material use (leaves use as waste or is dispersed)"),
             ("not_material_use", ORANGE, "Not material use (burned for energy, eaten, or exported)"))  # fmt: skip
    for col, color, label in parts:
        ax.barh(y, 100 * d[col], left=left, color=color, height=0.62, edgecolor=SURFACE, linewidth=2, label=label)
        for yi, l0, v in zip(y, left, 100 * d[col], strict=True):
            if v >= 8:
                ax.text(l0 + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        left = left + 100 * d[col].to_numpy()
    ax.set_yticks(y, [f"{NAMES[c]}\n({g:.2f} Gt)" for c, g in zip(d.index, d.processed_gt, strict=True)], fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of processed material, 2023 (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.38, -0.16), ncol=1, fontsize=8.8)
    fig.subplots_adjust(bottom=0.33)
    heading(fig, "Most processed material is burned or built into stocks",
            "EU27, processed material (inputs plus recycling and backfilling) by what happens to it, Eurostat's Sankey data")  # fmt: skip
    footer(fig)
    save(fig, "fig4_fates")


def fig5_targets():
    path = pd.read_csv(RESULTS / "eu_path.csv")
    sc = pd.read_csv(RESULTS / "ceilings.csv")
    tg = json.loads((RESULTS / "targets.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 6.2))
    fig.subplots_adjust(left=0.10, right=0.80, top=0.86, bottom=0.11)
    m = np.linspace(3.0, 7.6, 2)
    double_label = f"{100 * tg['double_2020']:.1f}% (double 2020)"
    for rate, label, style in ((tg["cid"], "24% (Clean Industrial Deal)", "-"), (tg["double_2020"], double_label, "--"),
                               (path.recycled.iloc[-1] / path.material_use.iloc[-1], "12.2% (2024)", ":")):  # fmt: skip
        ax.plot(m, rate * m, color=INK2, linewidth=1.2, linestyle=style)
        ax.annotate(
            label, (m[-1], rate * m[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2, annotation_clip=False
        )
    ax.plot(path.material_use, path.recycled, color=BLUE, linewidth=2.2, marker="o", markersize=3.5, label="EU27, 2010 to 2024")
    ax.annotate("2010", (path.material_use.iloc[0], path.recycled.iloc[0]), xytext=(4, -12), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.annotate("2024", (path.material_use.iloc[-1], path.recycled.iloc[-1]), xytext=(-30, 8), textcoords="offset points", fontsize=8.5, color=INK2)
    mu = path.material_use.iloc[-1]
    labels = [
        "Now",
        "All non-mineral waste",
        "+ construction and demolition",
        "+ combustion and treatment",
        "+ soils and dredging",
        "All treated waste",
    ]
    for (_, r), lab in zip(sc.iterrows(), labels, strict=True):
        if lab == "Now":
            continue
        ax.plot([mu], [r.recycled_gt], marker="s", color=ORANGE, markersize=6, zorder=3)
        ax.annotate(f"{lab} ({100 * r.rate:.1f}%)", (mu, r.recycled_gt), xytext=(8, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=8.5, color=INK2, annotation_clip=False)  # fmt: skip
    ax.set_xlim(3.0, 7.6)
    ax.set_ylim(0, 2.2)
    ax.set_xlabel("Material use: DMC plus recycled material (billion tonnes)")
    ax.set_ylabel("Recycled material (billion tonnes)")
    ax.legend(loc="upper left", fontsize=9)
    heading(fig, "Recycling alone reaches the 2030 targets only with soils and mining waste",
            "EU27. Lines: combinations that give a circular material use rate. "
            "Squares: 2024 use with more of 2024's treated waste recycled")  # fmt: skip
    footer(fig)
    save(fig, "fig5_targets")


if __name__ == "__main__":
    fig1_index()
    fig2_drift()
    fig3_footprint()
    fig4_fates()
    fig5_targets()
