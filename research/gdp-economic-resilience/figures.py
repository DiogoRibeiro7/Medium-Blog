"""Generate publication figures and a traceable evidence package.

Only deterministic analytical CSV outputs from the pinned data pipelines are
read. Figures are not substitutes for the underlying evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import pipeline as p
import eurostat_sectors as euro

HERE = Path(__file__).resolve().parent
Row = dict[str, str]
Record = dict[str, Any]
BLUE, ORANGE, TEAL, INK, MUTED = "#2465A3", "#B95720", "#187D77", "#202B36", "#687380"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.axisbelow": True, "grid.color": "#DDE2E7",
    "legend.frameon": False, "svg.hashsalt": "gdp-economic-resilience-v1",
})
COUNTRY_NAMES = {
    "EU27": "EU-27", "USA": "United States", "CHN": "China", "IND": "India",
    "GBR": "United Kingdom", "JPN": "Japan", "CHE": "Switzerland",
}
SHOCK_LABELS = {
    "financial_crisis": "Financial crisis (2008–10)",
    "pandemic": "Pandemic (2020–22)",
    "energy_price_shock": "Energy-price shock (2022–24)",
}
MODELS = ("trend_5y", "trend_10y", "flat")
MODEL_LABELS = ("5-year trend", "10-year trend", "Flat baseline")
FEATURE_LABELS = {
    "agriculture_pct_gdp": "Agriculture", "industry_pct_gdp": "Industry",
    "services_pct_gdp": "Services", "manufacturing_pct_gdp": "Manufacturing",
    "three_sector_hhi": "Three-sector HHI",
}
SECTOR_LABELS = {
    "A": "Agriculture", "B-E": "Industry (excl. construction)",
    "F": "Construction", "G-I": "Trade, transport, hospitality",
    "J": "Information and communication", "K": "Finance and insurance",
    "L": "Real estate", "M_N": "Professional and admin.",
    "O-Q": "Administration, education, health", "R-U": "Arts and other services",
    "C": "Manufacturing", "M": "Professional and scientific",
    "N": "Administrative and support", "O": "Public administration",
    "P": "Education", "Q": "Health and social work",
}
SCENARIOS = {
    "price_compression_only": "Price cut; fixed demand",
    "price_compression_elastic_demand": "Price cut; unit elasticity",
    "displacement_without_replacement": "Displacement only",
    "combined_no_replacement": "Combined; no replacement",
    "combined_partial_domestic_replacement": "Combined; partial replacement",
    "full_domestic_volume_replacement": "Full domestic replacement",
}
SOURCES = {
    "world_bank": p.COMMIT, "eurostat_a10": euro.SOURCES["a10"][1],
    "eurostat_a64": euro.SOURCES["a64"][1],
}


def digest(path: Path) -> str:
    """Hash exact file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_table(path: Path) -> list[Row]:
    """Reject malformed or empty analytical CSV files."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Invalid CSV header: {path}")
        rows = list(reader)
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"Missing or malformed CSV records: {path}")
    return rows


def number(row: Row, column: str) -> float | None:
    """Interpret blanks as unavailable, never zero."""
    value = row[column]
    if value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {column}: {value}")
    return result


def required(row: Row, column: str) -> float:
    """Fail explicitly when a required plotted value is missing."""
    value = number(row, column)
    if value is None:
        raise ValueError(f"Missing required figure value: {column}")
    return value


def select(rows: list[Row], **filters: str) -> list[Row]:
    """Match exact CSV field values."""
    return [row for row in rows if all(row[k] == v for k, v in filters.items())]


def one(rows: list[Row], **filters: str) -> Row:
    """Require exactly one record for a figure key."""
    result = select(rows, **filters)
    if len(result) != 1:
        raise ValueError(f"Expected exactly one matching row: {filters}; got {len(result)}")
    return result[0]


def heading(fig: Figure, title: str, subtitle: str, source: str) -> None:
    """Add a consistent editorial title, scope and source line."""
    fig.text(0.025, 0.975, title, va="top", color=INK, fontsize=18, weight="bold")
    fig.text(0.025, 0.925, subtitle, va="top", color=MUTED, fontsize=11)
    fig.text(0.025, 0.02, source, va="bottom", color=MUTED, fontsize=9)


def figure_gdp(rows: list[Row]) -> tuple[Figure, list[Row], str]:
    """Show two distinct measures using the same nonoverlapping entities."""
    selected = select(rows, group="fixed_eu27_comparators", year="2025")
    if {row["entity"] for row in selected} != set(COUNTRY_NAMES) or len(selected) != 7:
        raise ValueError("Unexpected GDP comparator set")
    selected.sort(key=lambda row: required(row, "nominal_usd"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), sharey=True)
    fig.subplots_adjust(left=0.19, right=0.95, top=0.82, bottom=0.15, wspace=0.3)
    labels = [COUNTRY_NAMES[row["entity"]] for row in selected]
    for ax, key, label, color in zip(
        axes, ("nominal_usd", "ppp_international_dollars"),
        ("Nominal GDP (trillion US dollars)", "PPP GDP (trillion international dollars)"),
        (BLUE, TEAL), strict=True,
    ):
        values = [required(row, key) / 1e12 for row in selected]
        bars = ax.barh(labels, values, color=color, height=0.65)
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=10)
        ax.set_xlim(0, max(values) * 1.18)
        ax.set_xlabel(label)
        ax.grid(axis="x")
    caption = ("Fixed EU-27 and six non-EU comparators, 2025. Same order in both panels; "
               "PPP is not foreign-exchange purchasing power.")
    heading(fig, "Economic size depends on the unit", "2025 • fixed EU-27 and six non-EU comparators",
            "Source: World Bank WDI • EU-27 calculated from complete member observations.")
    return fig, selected, caption


def figure_composition(rows: list[Row]) -> tuple[Figure, list[Row], str]:
    """Compare the published nonoverlapping ten-sector EU partition."""
    selected = [one(rows, level="a10", geo="EU27_2020", sector=sector, year=str(year))
                for sector in euro.A10 for year in (2000, 2024)]
    labels = [SECTOR_LABELS[sector] for sector in euro.A10]
    fig, ax = plt.subplots(figsize=(12, 7.4))
    fig.subplots_adjust(left=0.34, right=0.94, top=0.81, bottom=0.15)
    positions = list(range(len(labels)))
    for year, offset, color in ((2000, -0.18, BLUE), (2024, 0.18, TEAL)):
        values = [required(one(selected, sector=code, year=str(year)), "gva_share_pct")
                  for code in euro.A10]
        ax.barh([y + offset for y in positions], values, height=0.33, color=color, label=str(year))
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Share of current-price gross value added (%)")
    ax.grid(axis="x")
    ax.legend(loc="lower right")
    caption = ("Published EU-27 A10 groups in 2000 and 2024. Manufacturing belongs to industry; "
               "shares are nominal GVA shares, not real output contributions. Source flags are retained in evidence.")
    heading(fig, "Services are not one economic activity", "EU-27 • ten nonoverlapping activity groups",
            "Source: Eurostat nama_10_a10 • pinned November 2026 snapshot.")
    return fig, selected, caption


def figure_prices(rows: list[Row]) -> tuple[Figure, list[Row], str]:
    """Separate relative volume and deflator changes in selected A64 sections."""
    sectors = ("C", "J", "K", "L", "M", "N", "O", "P", "Q")
    selected = [one(rows, level="a64", geo="EU27_2020", sector=sector, start="2000", end="2024")
                for sector in sectors]
    keys = ("relative_volume_change_log_points", "relative_deflator_change_log_points",
            "nominal_share_change_log_points")
    plotted = [row for row in selected if all(number(row, key) is not None for key in keys)]
    if not plotted:
        raise ValueError("No complete EU share decomposition")
    fig, ax = plt.subplots(figsize=(12, 7.4))
    fig.subplots_adjust(left=0.34, right=0.94, top=0.81, bottom=0.16)
    y = list(range(len(plotted)))
    for key, offset, color, label in zip(keys[:2], (-0.19, 0.19), (BLUE, ORANGE),
                                         ("Relative volume", "Relative deflator"), strict=True):
        ax.barh([i + offset for i in y], [required(row, key) for row in plotted],
                height=0.34, color=color, label=label)
    ax.scatter([required(row, keys[2]) for row in plotted], y, marker="D",
               color=INK, s=30, label="Total nominal share change", zorder=3)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(y, [SECTOR_LABELS[row["sector"]] for row in plotted])
    ax.invert_yaxis()
    ax.set_xlabel("Change in log share × 100 (log points)")
    ax.grid(axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=9)
    caption = ("Relative real volume plus relative implicit deflator equals nominal share change. "
               f"Selected EU-27 sections, 2000–2024; {len(plotted)}/{len(selected)} complete. "
               "These are log points, not percentage points.")
    heading(fig, "A rising nominal share can be a price effect",
            f"EU-27 • selected activities, 2000–2024 • {len(plotted)}/{len(selected)} complete",
            "Source: Eurostat nama_10_a64 • accounting decomposition, not causal attribution.")
    return fig, selected, caption


def figure_wealth(rows: list[Row]) -> tuple[Figure, list[Row], str]:
    """Compare annualized within-country real growth, excluding flagged Bulgaria."""
    selected = select(rows, start="1996", end="2020")
    if len(selected) != len(p.COUNTRIES) or {row["country"] for row in selected} != set(p.COUNTRIES):
        raise ValueError("Unexpected wealth comparison sample")
    points = []
    for row in selected:
        if row["country"] == "BGR" or row["comparable"] != "True":
            continue
        g = required(row, "real_gdp_pc_growth_pct")
        w = required(row, "real_wealth_pc_growth_pct")
        if g <= -100 or w <= -100:
            continue
        points.append((row["country"], 100 * math.log1p(g / 100) / 24,
                       100 * math.log1p(w / 100) / 24))
    if not points:
        raise ValueError("No valid wealth comparisons")
    fig, ax = plt.subplots(figsize=(9.8, 7.7))
    fig.subplots_adjust(left=0.13, right=0.94, top=0.82, bottom=0.16)
    xmin = min(0, min(x for _, x, _ in points), min(y for _, _, y in points))
    xmax = max(0, max(x for _, x, _ in points), max(y for _, _, y in points))
    pad = max(0.2, (xmax - xmin) * 0.08)
    limits = (xmin - pad, xmax + pad)
    ax.plot(limits, limits, linestyle="--", color=MUTED, label="Equal growth")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    for country, x, y in points:
        divergent = x > 0 and y < 0
        ax.scatter(x, y, color=ORANGE if divergent else BLUE, s=46, alpha=0.85, zorder=3)
        if divergent or country in {"USA", "DEU", "PRT", "IRL", "CHN", "IND"}:
            ax.annotate(country, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=9)
    ax.set(xlim=limits, ylim=limits, aspect="equal",
           xlabel="Real GDP per capita (annualized log growth, %)",
           ylabel="Real comprehensive wealth per capita (annualized log growth, %)")
    ax.grid()
    ax.legend(loc="upper left")
    excluded = sorted(set(p.COUNTRIES) - {country for country, _, _ in points})
    caption = ("1996–2020, within-country annualized log growth. Orange: GDP rises while wealth falls. "
               f"n={len(points)}; excluded {', '.join(excluded)}. Bulgaria is excluded because of its "
               "flagged 1996 baseline; all original rows remain in the evidence.")
    heading(fig, "GDP growth and national wealth need not move together",
            f"1996–2020 • {len(points)} countries • excluded {', '.join(excluded)}",
            "Source: World Bank WDI and CWON 2024 • chained wealth and GDP compared by growth, not levels.")
    return fig, selected, caption


def matched_shocks(rows: list[Row], episode: str) -> list[Row]:
    """Use identical complete countries for every benchmark in one episode."""
    panel: dict[str, dict[str, Row]] = {model: {} for model in MODELS}
    for row in select(rows, shock=episode, metric="real_gdp_per_capita"):
        model, country = row["benchmark"], row["country"]
        if country in panel[model]:
            raise ValueError("Duplicate country/model")
        if row["complete_common_window"] == "True" and number(row, "cumulative_shortfall_pct_benchmark") is not None:
            panel[model][country] = row
    countries = sorted(set.intersection(*(set(records) for records in panel.values())))
    return [panel[model][country] for model in MODELS for country in countries]


def figure_shocks(rows: list[Row]) -> tuple[Figure, list[Row], str]:
    """Show cross-country medians and IQRs under alternative loss benchmarks."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.8), sharey=True)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.80, bottom=0.20, wspace=0.32)
    selected = []
    for ax, episode in zip(axes, SHOCK_LABELS, strict=True):
        sample = matched_shocks(rows, episode)
        selected.extend(sample)
        n = len(sample) // len(MODELS)
        if n < 2:
            raise ValueError(f"Insufficient matched shock data: {episode}")
        for i, model in enumerate(MODELS):
            values = [required(row, "cumulative_shortfall_pct_benchmark") for row in sample
                      if row["benchmark"] == model]
            q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
            median = statistics.median(values)
            ax.errorbar(median, i, xerr=[[median - q1], [q3 - median]], fmt="o",
                        color=BLUE, capsize=4, markersize=6)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_yticks(range(3), MODEL_LABELS)
        ax.set_title(SHOCK_LABELS[episode] + f"\n(n={n})", fontsize=11)
        ax.grid(axis="x")
    axes[0].invert_yaxis()
    # The panels share identical x-axis limits for honest visual comparison.
    low = min(ax.get_xlim()[0] for ax in axes)
    high = max(ax.get_xlim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(low, high)
    fig.text(0.56, 0.10, "Three-year cumulative shortfall (% of benchmark)",
             ha="center", color=INK, fontsize=11)
    caption = ("Real GDP per capita, matched countries within each episode. Points: median; bars: "
               "interquartile range across countries, not confidence intervals. Negative shortfall "
               "means outperformance. Trend benchmarks do not identify causal effects.")
    heading(fig, "Measured crisis losses depend on the benchmark",
            "Country medians and middle 50% • equal three-year windows",
            "Source: World Bank WDI • five-/ten-year anchored trends and flat baseline.")
    return fig, selected, caption


def figure_substitution(rows: list[Row]) -> tuple[Figure, list[Row], str]:
    """Compare hypothetical nominal and constant-base-price changes."""
    selected = [one(rows, dataset="eurostat_a64", entity="EU27_2020", year="2024",
                    portfolio="selected_business_services", scenario=scenario)
                for scenario in SCENARIOS]
    exposure = required(selected[0], "target_share_pct")
    if any(required(row, "target_share_pct") != exposure for row in selected):
        raise ValueError("Scenario weights differ")
    fig, ax = plt.subplots(figsize=(12, 6.7))
    fig.subplots_adjust(left=0.34, right=0.94, top=0.81, bottom=0.19)
    pos = list(range(len(selected)))
    for key, offset, color, label in (
        ("nominal_change_pct_baseline_aggregate", -0.18, ORANGE, "Nominal GVA"),
        ("base_price_change_pct_baseline_aggregate", 0.18, BLUE, "Base-price proxy"),
    ):
        ax.barh([i + offset for i in pos], [required(row, key) for row in selected],
                height=0.32, color=color, label=label)
    ax.set_yticks(pos, list(SCENARIOS.values()))
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Change as % of baseline aggregate GVA")
    ax.grid(axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    caption = ("Hypothetical direct effects for EU-27 selected business services (J/K/M/N), "
               f"2024 baseline share {exposure:.1f}% of GVA. Assumptions are not measured AI exposure "
               "or forecasts; the base-price proxy is not a chain-linked GDP estimate.")
    heading(fig, "Lost market value is not the same as lost output",
            f"Hypothetical EU-27 stress scenarios • business services • {exposure:.1f}% of baseline GVA",
            "Source: Eurostat nama_10_a64 • scenario assumptions in SUBSTITUTION.md.")
    return fig, selected, caption


def figure_correlations(rows: list[Row]) -> tuple[Figure, list[Row], str]:
    """Display rank associations and their actual sample sizes for each method."""
    features = list(FEATURE_LABELS)
    selected = [one(rows, shock=episode, structure_feature=feature, benchmark=model,
                    sample="all_countries") for feature in features
                for episode in SHOCK_LABELS for model in MODELS]
    matrix = []
    for feature in features:
        values = []
        for row in select(selected, structure_feature=feature):
            r = number(row, "spearman_r")
            values.append(math.nan if r is None else r)
        matrix.append(values)
    fig, ax = plt.subplots(figsize=(13, 5.9))
    fig.subplots_adjust(left=0.20, right=0.91, top=0.77, bottom=0.20)
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    for i, feature in enumerate(features):
        for j, row in enumerate(select(selected, structure_feature=feature)):
            value = number(row, "spearman_r")
            label = "NA" if value is None else f"{value:+.2f}"
            color = "white" if value is not None and abs(value) > 0.65 else INK
            ax.text(j, i, f"{label}\nn={row['n']}", ha="center", va="center",
                    fontsize=9, color=color)
    ax.set_yticks(range(len(features)), [FEATURE_LABELS[f] for f in features])
    ax.set_xticks(range(9), ["5y", "10y", "Flat"] * 3)
    for index, label in enumerate(("Financial crisis", "Pandemic", "Energy-price shock")):
        ax.text(index * 3 + 1, -0.75, label, ha="center", va="bottom", fontsize=10)
    for x in (2.5, 5.5):
        ax.axvline(x, color="white", linewidth=3)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Spearman correlation")
    caption = ("Rank correlations between pre-shock economic structure and GDP-per-capita "
               "shortfall. Samples are matched across benchmarks within each feature and episode. "
               "Cells report sample size; NA is undefined, not zero. Descriptive only.")
    heading(fig, "Does economic structure predict resilience?",
            "Pre-shock features versus subsequent shortfall • rank correlations by benchmark",
            "Source: World Bank WDI • matched-sample robustness analysis; no causal interpretation.")
    return fig, selected, caption


def figure_classification(rows: list[Row]) -> tuple[Figure, list[Row], str]:
    """Show concentration rank agreement, not incomparable HHI levels."""
    years = (2000, 2010, 2019, 2021, 2024)
    selected = [one(rows, dataset=dataset, year=str(year)) for dataset in ("wdi", "eurostat")
                for year in years]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.81, bottom=0.20)
    for dataset, offset, color, label in (
        ("wdi", -0.09, BLUE, "WDI: 3 vs 4 sectors"),
        ("eurostat", 0.09, TEAL, "Eurostat: A10 vs A64"),
    ):
        sample = select(selected, dataset=dataset)
        values = [number(row, "spearman_r") for row in sample]
        ax.plot([i + offset for i in range(len(years))],
                [math.nan if v is None else v for v in values], "o-", color=color, label=label)
        for i, (row, value) in enumerate(zip(sample, values, strict=True)):
            if value is not None:
                ax.annotate(f"n={row['n']}", (i + offset, value), xytext=(0, 10 if offset < 0 else -15),
                            textcoords="offset points", ha="center", fontsize=9, color=color)
    ax.set_xticks(range(len(years)), [str(y) for y in years])
    ax.set_ylim(-1.05, 1.12)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_ylabel("Spearman correlation of concentration ranks")
    ax.grid(axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)
    caption = ("Matched-country HHI rank agreement across classifications, not comparison of "
               "absolute HHI values. WDI and Eurostat panels have different country samples; "
               "sample sizes are annotated. Missing values produce breaks.")
    heading(fig, "Diversification rankings depend on classification",
            "Matched-country rank agreement • five selected years",
            "Source: World Bank WDI and Eurostat • excludes the EU aggregate from country correlations.")
    return fig, selected, caption


Builder = Callable[[list[Row]], tuple[Figure, list[Row], str]]
SPECS: tuple[tuple[str, str, Builder], ...] = (
    ("fig01_nominal_ppp", "gdp_rankings_2025.csv", figure_gdp),
    ("fig02_sector_composition", "eurostat/eurostat_sections.csv", figure_composition),
    ("fig03_volume_deflator", "eurostat/eurostat_endpoint_changes.csv", figure_prices),
    ("fig04_gdp_wealth", "gdp_wealth_endpoint_growth.csv", figure_wealth),
    ("fig05_shock_benchmarks", "shocks/wdi_shock_metrics.csv", figure_shocks),
    ("fig06_substitution", "substitution/named_scenarios.csv", figure_substitution),
    ("fig07_structure_robustness", "robustness/structure_correlation_sensitivity.csv", figure_correlations),
    ("fig08_classification_robustness", "robustness/classification_sensitivity.csv", figure_classification),
)


def write_evidence(path: Path, rows: list[Row]) -> None:
    """Preserve full selected records in source column order."""
    p.write_csv(path, rows)


def run(results: Path, output: Path) -> None:
    """Draw eight figures and record source/table/asset SHA-256 fingerprints."""
    output.mkdir(parents=True, exist_ok=True)
    manifest: Record = {"schema_version": 1, "registry_commits": SOURCES, "figures": []}
    for name, relative, build in SPECS:
        source = results / relative
        rows = read_table(source)
        fig, selected, caption = build(rows)
        try:
            png, svg = output / f"{name}.png", output / f"{name}.svg"
            evidence = output / f"{name}.csv"
            fig.savefig(png, dpi=200, metadata={"Software": "GDP resilience figures"})
            fig.savefig(svg, metadata={"Date": None, "Creator": "GDP resilience figures"})
            write_evidence(evidence, selected)
            manifest["figures"].append({
                "id": name, "input_table": relative, "input_sha256": digest(source),
                "selected_rows": len(selected), "caption": caption,
                "assets": {file.name: digest(file) for file in (png, svg, evidence)},
            })
        finally:
            plt.close(fig)
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(SPECS)} figures with evidence and source hashes")


def main() -> None:
    """CLI for publication assets generated from existing analytical outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "publication")
    args = parser.parse_args()
    run(args.results, args.output)


if __name__ == "__main__":
    main()
