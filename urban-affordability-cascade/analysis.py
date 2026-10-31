"""Reproduce the observed Porto AMP diagnostics and hypothetical cascade models.

Only the 17-municipality INE-derived panel is observational. Relocation and
commuting exercises deliberately use declared, unestimated parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
INPUT_PANEL = ROOT / "data" / "amp_municipal_panel_2019_2025.csv"
SOURCE_COMMIT = "0407faed1f08209b3d7682e19fc57e7120e62879"
SOURCE_BLOB_SHA = "9bade430163315e7c5da323251daa0715e1ffede"
SUBSET_SHA256 = "88502e15b55a4d46536eed42ddc1924d30f555498d2c4a7e7a1b41fbeed71c66"
MUNICIPALITIES: frozenset[str] = frozenset(
    {
        "Arouca", "Espinho", "Gondomar", "Maia", "Matosinhos",
        "Oliveira de Azeméis", "Paredes", "Porto", "Póvoa de Varzim",
        "Santa Maria da Feira", "Santo Tirso", "São João da Madeira",
        "Trofa", "Vale de Cambra", "Valongo", "Vila do Conde",
        "Vila Nova de Gaia",
    }
)
PALETTE = {"ink": "#17324d", "blue": "#2166ac", "orange": "#c36c2b", "teal": "#227a70"}


@dataclass(frozen=True, slots=True)
class Observation:
    """One observed municipality-year row; None denotes missing publication."""

    municipality: str
    year: int
    price_m2: float | None
    rent_m2: float | None
    income: float | None
    population: float | None
    completions: float | None


@dataclass(frozen=True, slots=True)
class CascadeStage:
    """Purely hypothetical transition counts at one abstract distance tier."""

    tier: str
    incoming: float
    absorbed: float
    onward: float


def _optional_nonnegative(value: str, label: str, *, strictly_positive: bool) -> float | None:
    """Parse an official numeric cell without imputing unpublished entries."""
    if value.strip() == "":
        return None
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric {label}: {value!r}") from exc
    if not math.isfinite(numeric) or (numeric <= 0 if strictly_positive else numeric < 0):
        raise ValueError(f"Invalid nonnegative/positive {label}: {value!r}")
    return numeric


def load_panel(path: Path = INPUT_PANEL) -> dict[tuple[str, int], Observation]:
    """Read and audit the fixed 17-by-7 Porto metropolitan extract."""
    if path == INPUT_PANEL:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != SUBSET_SHA256:
            raise ValueError("The fixed data extract does not match its recorded SHA-256")
    required = {
        "municipality", "year", "price_m2", "rent_m2", "income",
        "population", "completions",
    }
    observations: dict[tuple[str, int], Observation] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Panel is missing required columns: {sorted(required)}")
        for row in reader:
            municipality = row["municipality"]
            if municipality not in MUNICIPALITIES:
                raise ValueError(f"Unexpected municipality: {municipality!r}")
            try:
                year = int(row["year"])
            except ValueError as exc:
                raise ValueError(f"Non-integer year: {row['year']!r}") from exc
            if year not in range(2019, 2026):
                raise ValueError(f"Unexpected publication year: {year}")
            key = (municipality, year)
            if key in observations:
                raise ValueError(f"Duplicate municipality-year: {key!r}")
            item = Observation(
                municipality=municipality,
                year=year,
                price_m2=_optional_nonnegative(row["price_m2"], "price_m2", strictly_positive=True),
                rent_m2=_optional_nonnegative(row["rent_m2"], "rent_m2", strictly_positive=True),
                income=_optional_nonnegative(row["income"], "income", strictly_positive=True),
                population=_optional_nonnegative(row["population"], "population", strictly_positive=True),
                completions=_optional_nonnegative(row["completions"], "completions", strictly_positive=False),
            )
            observations[key] = item
    expected = {(name, year) for name in MUNICIPALITIES for year in range(2019, 2026)}
    if set(observations) != expected:
        missing = sorted(expected - set(observations))
        raise ValueError(f"The municipal panel is not a complete 17 x 7 grid: {missing[:5]}")
    # Missing rent observations in 2019 and missing incomes in 2025 are source
    # limitations. Do not let an unexpected missing value enter a growth result.
    for name in MUNICIPALITIES:
        for year in range(2019, 2026):
            obs = observations[name, year]
            if obs.price_m2 is None or obs.population is None:
                raise ValueError(f"Missing required price/population observation: {name} {year}")
            if year >= 2020 and obs.rent_m2 is None:
                raise ValueError(f"Missing new-lease rental value: {name} {year}")
            if year <= 2024 and obs.income is None:
                raise ValueError(f"Missing fiscal income value: {name} {year}")
    return observations


def require_float(value: float | None, label: str) -> float:
    """Reject missing source values rather than quietly converting to zero."""
    if value is None:
        raise ValueError(f"Missing value for {label}")
    return value


def percent_change(earlier: float, later: float) -> float:
    """Percentage growth with a positive published baseline."""
    if earlier <= 0 or not all(math.isfinite(x) for x in (earlier, later)):
        raise ValueError("A positive finite baseline and finite endpoint are required")
    return 100.0 * (later / earlier - 1.0)


def observed_summary(panel: dict[tuple[str, int], Observation]) -> list[dict[str, str | float]]:
    """Calculate transparent endpoint comparisons, without causal attribution."""
    result: list[dict[str, str | float]] = []
    for name in sorted(MUNICIPALITIES):
        a, b, c = panel[name, 2019], panel[name, 2020], panel[name, 2024]
        end = panel[name, 2025]
        p0, p1 = require_float(a.price_m2, name), require_float(end.price_m2, name)
        r0, r1 = require_float(b.rent_m2, name), require_float(end.rent_m2, name)
        pop0 = require_float(a.population, name)
        pop1 = require_float(end.population, name)
        income0, income1 = require_float(a.income, name), require_float(c.income, name)
        price_2024 = require_float(c.price_m2, name)
        result.append({
            "municipality": name,
            "sale_price_2019_eur_m2": p0,
            "sale_price_2025_eur_m2": p1,
            "sale_price_growth_2019_2025_pct": percent_change(p0, p1),
            "new_lease_rent_2020_eur_m2": r0,
            "new_lease_rent_2025_eur_m2": r1,
            "new_lease_rent_growth_2020_2025_pct": percent_change(r0, r1),
            "population_growth_2019_2025_pct": percent_change(pop0, pop1),
            "illustrative_100m2_value_to_fiscal_income_2019": 100 * p0 / income0,
            "illustrative_100m2_value_to_fiscal_income_2024": 100 * price_2024 / income1,
        })
    return result


def displacement_chain(displaced: float, absorption: Sequence[float]) -> list[CascadeStage]:
    """Model conservation of hypothetical households across sequential tiers.

    Each tier absorbs a specified fraction of arrivals; remaining households
    continue outward. All fractions are assumptions, not fitted probabilities.
    The final `onward` amount leaves the represented model boundary.
    """
    if not math.isfinite(displaced) or displaced < 0:
        raise ValueError("Displaced count must be finite and nonnegative")
    if not absorption:
        raise ValueError("At least one distance tier is required")
    remaining = float(displaced)
    output: list[CascadeStage] = []
    for i, fraction in enumerate(absorption, start=1):
        if not math.isfinite(fraction) or fraction < 0 or fraction > 1:
            raise ValueError("Absorption fractions must lie in [0, 1]")
        absorbed = remaining * fraction
        subsequent = remaining - absorbed
        output.append(CascadeStage(f"Tier {i}", remaining, absorbed, subsequent))
        remaining = subsequent
    return output


def commute_counterfactual(distance_km: float) -> dict[str, float]:
    """Return illustrative monthly housing, travel and time costs in EUR.

    Parameters are chosen for an explanation, not estimated from Porto data:
    base housing EUR 1,100; asymptotic saving EUR 600; decay scale 25 km;
    travel EUR 35 + EUR 4/km; time valuation EUR 5.5/km each month.
    """
    if not math.isfinite(distance_km) or distance_km < 0:
        raise ValueError("Commuting distance must be finite and nonnegative")
    rent = 1100 - 600 * (1 - math.exp(-distance_km / 25))
    transport = 35 + 4 * distance_km
    time = 5.5 * distance_km
    return {
        "distance_km": distance_km,
        "illustrative_rent_eur_month": rent,
        "illustrative_transport_eur_month": transport,
        "illustrative_time_value_eur_month": time,
        "illustrative_total_eur_month": rent + transport + time,
    }


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Save a nonempty homogeneous table using UTF-8 and stable column order."""
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format(ax: Any, *, xlabel: str = "", ylabel: str = "") -> None:
    """Shared publication plot formatting."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#dadfe4", alpha=0.6, linewidth=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def produce_figures(
    panel: dict[tuple[str, int], Observation], summary: list[dict[str, str | float]],
    output: Path,
) -> None:
    """Produce exactly eight observed-or-labeled-hypothetical original figures."""
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 170})

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(output / name, bbox_inches="tight", facecolor="white")
        plt.close()

    prices = sorted(summary, key=lambda r: float(r["sale_price_growth_2019_2025_pct"]))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([str(x["municipality"]) for x in prices],
            [float(x["sale_price_growth_2019_2025_pct"]) for x in prices], color=PALETTE["blue"])
    _format(ax, xlabel="2019–2025 change in median sale price per m² (%)")
    ax.set_title("Observed: all 17 AMP municipalities — median sold-dwelling prices")
    save("fig1_sale_price_growth_2019_2025.png")

    rents = sorted(summary, key=lambda r: float(r["new_lease_rent_growth_2020_2025_pct"]))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([str(x["municipality"]) for x in rents],
            [float(x["new_lease_rent_growth_2020_2025_pct"]) for x in rents], color=PALETTE["orange"])
    _format(ax, xlabel="2020–2025 change in median new-contract rent per m² (%)")
    ax.set_title("Observed: new leases, not the rent paid by all tenants")
    save("fig2_new_lease_rent_growth_2020_2025.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    years = list(range(2019, 2026))
    for name in sorted(MUNICIPALITIES - {"Porto"}):
        base = require_float(panel[name, 2019].price_m2, name)
        series = [100 * require_float(panel[name, y].price_m2, name) / base for y in years]
        ax.plot(years, series, color="#bec7cf", alpha=0.64, linewidth=1)
    porto_base = require_float(panel["Porto", 2019].price_m2, "Porto")
    porto_values = [100 * require_float(panel["Porto", y].price_m2, "Porto") / porto_base for y in years]
    ax.plot(years, porto_values, color=PALETTE["blue"], linewidth=2.5, marker="o", label="Porto municipality")
    ax.plot([], [], color="#bec7cf", label="Other 16 municipalities (individual indexes)")
    ax.legend(loc="upper left", frameon=False)
    _format(ax, ylabel="Municipality-specific median sale-price index (2019=100)")
    ax.set_title("Observed: co-movement, not migration or spillover identification")
    save("fig3_sale_price_index_2019_2025.png")

    ratios = sorted(summary, key=lambda r: float(r["illustrative_100m2_value_to_fiscal_income_2024"]))
    fig, ax = plt.subplots(figsize=(10, 7))
    y = list(range(len(ratios)))
    ax.scatter([float(x["illustrative_100m2_value_to_fiscal_income_2019"]) for x in ratios],
               y, marker="o", color=PALETTE["teal"], label="2019")
    ax.scatter([float(x["illustrative_100m2_value_to_fiscal_income_2024"]) for x in ratios],
               y, marker="D", color=PALETTE["orange"], label="2024")
    ax.set_yticks(y, [str(x["municipality"]) for x in ratios])
    ax.legend(frameon=False)
    _format(ax, xlabel="100 m² × median sale €/m² / annual median gross fiscal-household income")
    ax.set_title("Derived proxy only: NOT a mortgage affordability or individual income measure")
    save("fig4_illustrative_value_income_ratio.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for name in ("Porto", "Matosinhos", "Vila Nova de Gaia", "Maia", "Valongo", "Póvoa de Varzim"):
        xx = list(range(2020, 2026))
        yy = [require_float(panel[name, year].rent_m2, f"{name} {year}") for year in xx]
        ax.plot(xx, yy, marker=".", linewidth=2, label=name)
    ax.legend(frameon=False, ncol=2)
    _format(ax, ylabel="Median newly contracted rent (EUR/m²/month)")
    ax.set_title("Observed: selected municipalities; sample of new contracts changes")
    save("fig5_new_lease_rent_trajectories.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    x = [float(r["population_growth_2019_2025_pct"]) for r in summary]
    y = [float(r["sale_price_growth_2019_2025_pct"]) for r in summary]
    ax.scatter(x, y, color=PALETTE["blue"], s=47)
    for r in summary:
        if r["municipality"] in {"Porto", "Gondomar", "Vila Nova de Gaia", "Arouca", "Matosinhos"}:
            ax.annotate(str(r["municipality"]),
                        (float(r["population_growth_2019_2025_pct"]),
                         float(r["sale_price_growth_2019_2025_pct"])),
                        xytext=(4, 5), textcoords="offset points", fontsize=8)
    _format(ax, xlabel="2019–2025 municipality resident-population change (%)",
            ylabel="2019–2025 median sale-price change (%)")
    ax.set_title("Observed: association is not evidence of displacement")
    save("fig6_population_vs_price_growth.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    scenarios = {
        "Constrained absorption (hypothetical)": [0.2, 0.2, 0.25],
        "Responsive absorption (hypothetical)": [0.55, 0.6, 0.75],
    }
    for label, fractions in scenarios.items():
        stages = displacement_chain(1000.0, fractions)
        colour = PALETTE["orange"] if "Constrained" in label else PALETTE["teal"]
        ax.plot([1, 2, 3, 4], [1000, stages[0].onward, stages[1].onward, stages[2].onward],
                marker="o", linewidth=2, color=colour, label=label)
    ax.set_xticks([1, 2, 3, 4], ["Origin exits", "After tier 1", "After tier 2", "Outside tier 3"])
    ax.legend(frameon=False)
    _format(ax, ylabel="Unabsorbed hypothetical households (initially 1,000)")
    ax.set_title("ILLUSTRATIVE ONLY: assumed capacity changes cascade depth")
    save("fig7_hypothetical_cascade_absorption.png")

    commuting = [commute_counterfactual(float(k)) for k in range(0, 61, 2)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([r["distance_km"] for r in commuting],
            [r["illustrative_rent_eur_month"] for r in commuting],
            label="Assumed rent", color=PALETTE["blue"], linewidth=2)
    ax.plot([r["distance_km"] for r in commuting],
            [r["illustrative_total_eur_month"] for r in commuting],
            label="Rent + travel + time", color=PALETTE["orange"], linewidth=2)
    ax.legend(frameon=False)
    _format(ax, xlabel="Hypothetical one-way commute distance (km)", ylabel="EUR/month")
    ax.set_title("ILLUSTRATIVE ONLY: rent savings can disappear in generalized cost")
    save("fig8_hypothetical_generalized_cost.png")


def run(output: Path) -> dict[str, Any]:
    """Run all checks, figures and provenance-labelled output tables."""
    panel = load_panel()
    summary = observed_summary(panel)
    output.mkdir(parents=True, exist_ok=True)
    _save_csv(output / "results" / "amp_growth_summary.csv", summary)
    cascade_rows: list[dict[str, Any]] = []
    for scenario, fractions in (
        ("hypothetical_constrained_absorption", [0.2, 0.2, 0.25]),
        ("hypothetical_responsive_absorption", [0.55, 0.6, 0.75]),
    ):
        for s in displacement_chain(1000.0, fractions):
            cascade_rows.append({"evidence_class": "hypothetical", "scenario": scenario,
                                 "tier": s.tier, "incoming": s.incoming,
                                 "absorbed": s.absorbed, "onward": s.onward})
    _save_csv(output / "results" / "hypothetical_cascade.csv", cascade_rows)
    _save_csv(output / "results" / "hypothetical_commuting_cost.csv",
              [dict(evidence_class="hypothetical", **commute_counterfactual(float(k)))
               for k in range(0, 61, 2)])
    produce_figures(panel, summary, output / "figures")
    metadata: dict[str, Any] = {
        "observational_source": "DiogoRibeiro7/portugal-housing-generations/results/integrated_panel.csv",
        "observational_source_commit": SOURCE_COMMIT,
        "observational_source_blob_sha": SOURCE_BLOB_SHA,
        "local_extract_sha256": SUBSET_SHA256,
        "source_underlying_publications": "INE price, new-lease rent, tax income, resident population and building statistics; see source project provenance",
        "observed_municipalities": len(MUNICIPALITIES),
        "observed_years": [2019, 2025],
        "missingness": "2019 new-lease rent and 2025 fiscal-household income not assumed/imputed",
        "research_status": "descriptive observational comparisons plus separately labelled illustrative models",
        "relocation_flow_data_present": False,
        "causal_effect_estimated": False,
        "figure_count": 8,
    }
    (output / "results" / "provenance.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT,
                        help="Directory for reproducible figures and tables (default: article folder)")
    args = parser.parse_args(argv)
    metadata = run(args.output)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
