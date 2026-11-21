"""Draw the five PNG figures used in the article.

Run ``python run.py`` first. Requires matplotlib (see requirements.txt).
Everything is drawn from local files; no network access is needed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from vacancy_tax import OTHER_REASONS_NO_OR_MINOR_REPAIRS, load_identification, load_vancouver

ROOT = Path(__file__).resolve().parent

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GREY = "#8f8e89"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "legend.frameon": False,
})


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def read_summary(root: Path) -> dict:
    with (root / "results" / "summary.json").open(encoding="utf-8") as source:
        return json.load(source)


def euros(value: float, _pos: int | None = None) -> str:
    return f"€{value:,.0f}"


def tidy(ax: plt.Axes, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def note(fig: plt.Figure, text: str) -> None:
    fig.text(0.012, 0.015, text, fontsize=8.5, color=INK_2, ha="left", va="bottom", wrap=True)


def save(fig: plt.Figure, name: str, root: Path) -> None:
    fig.savefig(root / "figures" / name, dpi=200)
    plt.close(fig)


def plot_census_to_flagged(root: Path = ROOT) -> None:
    """From the Census vacant stock to the properties flagged for the surcharge."""
    flagged = load_identification(root / "data" / "municipal_identification.csv")
    labels = [
        "Vacant dwellings, Census 2021",
        "…vacant for reasons other than\nbeing for sale or rent",
        "…of which in buildings needing\nno or only minor repairs",
        "Vacant properties flagged for the\nIMI surcharge, tax year 2023",
    ]
    census = [723_215, 375_118, OTHER_REASONS_NO_OR_MINOR_REPAIRS]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    y = list(range(len(labels)))[::-1]
    ax.barh(y[:3], census, color=BLUE, height=0.56, edgecolor=SURFACE, linewidth=2)
    ax.barh(y[3], flagged.vacant_general_rule, color=ORANGE, height=0.56, edgecolor=SURFACE, linewidth=2,
            label=f"General rule, 3× ({flagged.vacant_general_rule:,})")
    ax.barh(y[3], flagged.vacant_pressure_zones, left=flagged.vacant_general_rule, color=AQUA, height=0.56,
            edgecolor=SURFACE, linewidth=2, label=f"Pressure zones, 10× ({flagged.vacant_pressure_zones:,})")
    for yy, value in zip(y[:3], census):
        ax.text(value + 8_000, yy, f"{value:,}", va="center", color=INK, fontsize=11)
    ax.text(flagged.vacant_total + 8_000, y[3],
            f"{flagged.vacant_total:,}  ({flagged.municipalities_applying} of 308 municipalities)",
            va="center", color=INK, fontsize=11)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 900_000)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v / 1000:,.0f}k"))
    ax.set_title("The Census counts dwellings; the tax reaches a sliver of them", pad=12)
    ax.legend(loc="lower right", fontsize=10)
    tidy(ax, "x")
    note(fig, "Sources: INE Census 2021; IHRU/OHARU (2024), Table 11; Ministry of Finance via Lusa (1 May 2024). "
              "Census units are dwellings; tax units are properties (prédios or frações), and years differ.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig01_census_to_flagged.png", root)


def plot_incentives(root: Path = ROOT) -> None:
    """What each instrument is worth per year to the owner of one illustrative flat."""
    rows = read_table(root / "results" / "tax_scenarios.csv")
    surcharge = {r["multiplier"]: float(r["surcharge_eur"]) for r in rows}
    summary_cut = float(read_summary(root)["rate_cut_gain_upper_bound_eur"])
    labels = ["Cut in tax on rents,\n25% to 10% (at most)", "Surcharge,\ngeneral rule (3×)",
              "Surcharge, pressure\nzone, first year (10×)", "Surcharge, pressure\nzone cap (20×)"]
    values = [summary_cut, surcharge["3"], surcharge["10"], surcharge["20"]]
    colors = [BLUE, ORANGE, ORANGE, ORANGE]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    bars = ax.bar(range(4), values, color=colors, width=0.56, edgecolor=SURFACE, linewidth=2)
    ax.bar_label(bars, labels=[euros(v) for v in values], padding=4, color=INK, fontsize=11)
    ax.set_xticks(range(4), labels)
    ax.yaxis.set_major_formatter(FuncFormatter(euros))
    ax.set_ylim(0, max(values) * 1.15)
    ax.set_ylabel("Euros a year")
    ax.set_title("Two levers on the same decision, priced for one flat", pad=12)
    tidy(ax)
    note(fig, "Assumptions: taxable value (VPT) €150,000; IMI rate 0.35%; rent €900 a month, let 90% of the year. "
              "The cut is an upper bound (all rent taxed at the autonomous rate). Not a record of any assessment.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig02_incentives_eur.png", root)


def plot_expected_surcharge(root: Path = ROOT) -> None:
    """Expected yearly surcharge as a function of the probability of being flagged."""
    rows = read_table(root / "results" / "expected_surcharge.csv")
    summary = read_summary(root)
    pi = [100 * float(r["detection_probability"]) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    series = (("3", BLUE, "3×, general rule"), ("10", ORANGE, "10×, pressure zone"), ("20", AQUA, "20×, cap"))
    for m, color, label in series:
        values = [float(r[f"expected_surcharge_{m}x_eur"]) for r in rows]
        ax.plot(pi, values, color=color, linewidth=2, label=label)
        ax.text(101, values[-1], label, color=INK, fontsize=10, va="center")
    cut = float(summary["rate_cut_gain_upper_bound_eur"])
    ax.axhline(cut, color=GREY, linewidth=1.4, linestyle=(0, (5, 3)))
    ax.text(36, cut - 520, f"Most the 2026 cut adds to letting: {euros(cut)}", color=INK_2, fontsize=10)
    scale = 100 * float(summary["scale_detection_ratio"])
    ax.axvline(scale, color=GREY, linewidth=1.2)
    ax.text(scale + 1.2, 8_600, f"{scale:.1f}%: flagged properties ÷\ngood-condition census group",
            color=INK_2, fontsize=9.5, va="top")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10_500)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(euros))
    ax.set_xlabel("Probability that the owner is classified vacant and assessed in a given year")
    ax.set_ylabel("Expected surcharge, euros a year")
    ax.set_title("A multiplier is worth what detection makes of it", pad=12)
    ax.legend(loc="upper left", fontsize=10)
    tidy(ax)
    note(fig, "Same illustrative flat as Figure 2 (ordinary IMI €525). The vertical line divides counts with different "
              "units and years and is shown for scale, not as an estimate of any owner's probability.")
    fig.tight_layout(rect=(0, 0.05, 0.9, 1))
    save(fig, "fig03_expected_surcharge.png", root)


def plot_cost_per_lease(root: Path = ROOT) -> None:
    """Annual fiscal cost of the 2026 cut per additional lease it brings about."""
    summary = read_summary(root)
    costs = summary["rate_cut_annual_cost_eur"]
    benchmark = float(summary["break_even_additional_leases"]["benchmark_annual_contract_rent_eur"])
    xs = [1_000 * (1.05 ** k) for k in range(0, 109)]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for key, color, label in (("irs", BLUE, "IRS cut only (€279.1m a year)"),
                              ("irs_and_irc", ORANGE, "IRS and IRC measures (€308.9m)")):
        total = float(costs[key])
        ax.plot(xs, [total / x for x in xs], color=color, linewidth=2, label=label)
    ax.axhline(benchmark, color=GREY, linewidth=1.4, linestyle=(0, (5, 3)))
    ax.text(1_100, benchmark * 1.12, f"Annual rent of the illustrative flat ({euros(benchmark)})",
            color=INK_2, fontsize=10)
    for key, color in (("irs_only", BLUE), ("irs_and_irc", ORANGE)):
        n = float(summary["break_even_additional_leases"][key])
        ax.plot([n], [benchmark], marker="o", markersize=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2)
    n_irs = float(summary["break_even_additional_leases"]["irs_only"])
    ax.annotate(f"break-even near {n_irs / 1000:.0f}k additional leases", xy=(n_irs, benchmark),
                xytext=(n_irs * 1.6, benchmark * 3.2), fontsize=10, color=INK_2,
                arrowprops={"arrowstyle": "-", "color": GREY, "linewidth": 1})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1_000, 150_000)
    ax.set_ylim(1_500, 350_000)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v / 1000:,.0f}k"))
    ax.yaxis.set_major_formatter(FuncFormatter(euros))
    ax.set_xlabel("Leases that exist because of the cut (not formalised, not already let)")
    ax.set_ylabel("Fiscal cost per additional lease, euros a year")
    ax.set_title("A general cut pays for every lease to buy the marginal one", pad=12)
    ax.legend(loc="upper right", fontsize=10)
    tidy(ax, "both")
    note(fig, "Annual costs: UTAO assessment of the 2026 housing tax package. Both axes logarithmic. "
              "The number of additional leases is unknown; the curve shows what each value would imply.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig04_cost_per_additional_lease.png", root)


def plot_vancouver(root: Path = ROOT) -> None:
    """Vancouver's declared vacancy rate by reference year, with the tax rate."""
    years = load_vancouver(root / "data" / "vancouver_eht.csv")
    x = [y.reference_year for y in years]
    rate = [float(y.declared_vacancy_rate_pct) for y in years]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(x, rate, color=BLUE, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    for xx, rr in zip(x, rate):
        ax.text(xx, rr + 0.045, f"{rr:.2f}%", ha="center", color=INK, fontsize=10)
    for year, label in ((2017, "tax 1%"), (2020, "tax 1.25%"), (2021, "tax 3%")):
        ax.axvline(year - 0.5, color=GRID, linewidth=1.2)
        ax.text(year - 0.42, 1.36, label, color=INK_2, fontsize=10, va="top")
    ax.set_xlim(2016.5, 2024.5)
    ax.set_ylim(0, 1.4)
    ax.set_xticks(x)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.1f}%"))
    ax.set_ylabel("Vacant properties, % of those required to declare")
    ax.set_title("Vancouver: declared vacancy under an empty-homes tax, 2017–2024", pad=12)
    tidy(ax)
    note(fig, "Source: City of Vancouver, Empty Homes Tax annual reports (2024 and 2025). Declared, determined and "
              "deemed vacant. Monitoring data from the tax itself, not an estimate of its effect.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig05_vancouver_declared_vacancy.png", root)


def main() -> None:
    ROOT.joinpath("figures").mkdir(parents=True, exist_ok=True)
    plot_census_to_flagged()
    plot_incentives()
    plot_expected_surcharge()
    plot_cost_per_lease()
    plot_vancouver()


if __name__ == "__main__":
    main()
