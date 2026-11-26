"""Transparent technological substitution stress scenarios, not AI forecasts.

Use current-price value added as exposure weights. Price and volume factors
are explicit assumptions, not observed estimates; no supply-chain propagation
or welfare valuation is inferred from these direct accounting calculations.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import eurostat_sectors as euro
import pipeline as p

Value = float | None
Record = dict[str, Any]
EURO_YEARS = (2019, 2024)
WDI_YEARS = (2019, 2021)
PORTFOLIOS = {
    "information_communication": ("J",),
    "finance_insurance": ("K",),
    "professional_scientific_technical": ("M",),
    "administrative_support": ("N",),
    "selected_business_services": ("J", "K", "M", "N"),
    "trade_transport_hospitality": ("G", "H", "I"),
    "manufacturing_comparator": ("C",),
    "administration_education_health_comparator": ("O", "P", "Q"),
}
WDI_PORTFOLIOS = {"all_services_stress": p.SRV, "manufacturing_comparator": p.MAN}


@dataclass(frozen=True)
class Exposure:
    """Share of baseline GDP or GVA in a named, nonoverlapping portfolio."""

    dataset: str
    entity: str
    year: int
    portfolio: str
    sectors: str
    denominator: str
    share: Value
    source_flags: str = ""


@dataclass(frozen=True)
class Scenario:
    """Independent hypothetical displacement, repricing and domestic replacement."""

    name: str
    displaced: float
    price_cut: float
    elasticity: float
    domestic_replacement: float
    replacement_intensity: float

    def __post_init__(self) -> None:
        for name in ("displaced", "domestic_replacement", "replacement_intensity"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not math.isfinite(self.price_cut) or not 0 <= self.price_cut < 1:
            raise ValueError("price_cut must lie in [0, 1)")
        if not math.isfinite(self.elasticity) or not 0 <= self.elasticity <= 3:
            raise ValueError("elasticity must lie in [0, 3]")


NAMED = (
    Scenario("unchanged", 0, 0, 0, 0, 0),
    Scenario("price_compression_only", 0, 0.25, 0, 0, 0),
    Scenario("price_compression_elastic_demand", 0, 0.25, 1, 0, 0),
    Scenario("displacement_without_replacement", 0.25, 0, 0, 0, 0),
    Scenario("combined_no_replacement", 0.25, 0.25, 0, 0, 0),
    Scenario("combined_partial_domestic_replacement", 0.25, 0.25, 0.5, 0.5, 0.5),
    Scenario("full_domestic_volume_replacement", 0.25, 0, 0, 1, 1),
)


def grid() -> list[Scenario]:
    """Enumerate an unweighted sensitivity grid; not a probability distribution."""
    return [
        Scenario(f"grid_{i:03}", d, cut, elasticity, recapture, 0.5)
        for i, (d, cut, elasticity, recapture) in enumerate(itertools.product(
            (0, 0.2, 0.4), (0, 0.2, 0.4), (0, 0.5, 1), (0, 0.5, 1)))
    ]


def valid_share(share: Value) -> Value:
    """A share outside the accounting domain is unavailable, never clipped."""
    return share if share is not None and math.isfinite(share) and 0 <= share <= 1 else None


def effects(share: Value, scenario: Scenario) -> Record:
    """Compute direct changes as percentage points of the baseline aggregate.

    Normalize the original portfolio's baseline price and volume to one.
    The volume proxy is valued at baseline prices, not a chain-linked GDP series.
    Replacement is incremental domestic value added, outside the target, with
    a base-price deflator of one. It is not a transfer of gross revenue.
    """
    price = 1 - scenario.price_cut
    volume = (1 - scenario.displaced) * price ** (-scenario.elasticity)
    retained_nominal = price * volume
    replacement = scenario.displaced * scenario.domestic_replacement * scenario.replacement_intensity
    nominal_factor = retained_nominal + replacement
    volume_factor = volume + replacement
    # Exact decomposition relative to the base-price scenario.
    price_term = (price - 1) * volume
    base_price_term = volume_factor - 1
    share = valid_share(share)
    result = {
        "remaining_activity_deflator_factor": price,
        "remaining_activity_volume_factor": volume,
        "replacement_gva_factor": replacement,
        "portfolio_plus_replacement_nominal_factor": nominal_factor,
        "portfolio_plus_replacement_base_price_factor": volume_factor,
        "nominal_change_pct_baseline_aggregate": None,
        "base_price_change_pct_baseline_aggregate": None,
        "price_effect_pct_baseline_aggregate": None,
        "replacement_pct_baseline_aggregate": None,
        "accounting_residual_pp": None,
    }
    if share is not None:
        nominal_change = 100 * share * (nominal_factor - 1)
        real_proxy_change = 100 * share * base_price_term
        price_effect = 100 * share * price_term
        result.update(
            nominal_change_pct_baseline_aggregate=nominal_change,
            base_price_change_pct_baseline_aggregate=real_proxy_change,
            price_effect_pct_baseline_aggregate=price_effect,
            replacement_pct_baseline_aggregate=100 * share * replacement,
            accounting_residual_pp=nominal_change - real_proxy_change - price_effect,
        )
    return result


def euro_exposures(gva: dict[euro.CellKey, euro.Cell]) -> list[Exposure]:
    """Use published total GVA and disjoint current-price NACE sections."""
    results = []
    for geo in euro.GEO:
        for year in EURO_YEARS:
            total = euro.get(gva, geo, year, "TOTAL", "CP_MEUR")
            for portfolio, sectors in PORTFOLIOS.items():
                cells = [euro.get(gva, geo, year, sector, "CP_MEUR") for sector in sectors]
                # A negative component must not be hidden by a positive group sum.
                valid = all(c.value is not None and c.value >= 0 for c in cells)
                value = math.fsum(c.value for c in cells if c.value is not None) if valid else None
                share = valid_share(p.ratio(value, total.value))
                flags = ";".join(sorted({c.flag for c in [total, *cells] if c.flag}))
                results.append(Exposure("eurostat_a64", geo, year, portfolio, ";".join(sectors),
                                        "current_price_gva", share, flags))
    return results


def wdi_exposures(wdi: p.Snapshot) -> list[Exposure]:
    """Broad international comparisons; fixed EU27 from GDP-weighted country shares.

    This intentionally severe all-services scenario is not an AI exposure
    estimate. GDP rather than GVA is the WDI denominator.
    """
    results = []
    groups = {country: (country,) for country in p.COUNTRIES}
    groups["EU27"] = p.EU27
    for entity, countries in sorted(groups.items()):
        for year in WDI_YEARS:
            for portfolio, code in WDI_PORTFOLIOS.items():
                nominal = [wdi.value(p.GDP, country, year) for country in countries]
                shares = [wdi.value(code, country, year) for country in countries]
                complete = all(g is not None and g > 0 and s is not None and 0 <= s <= 100
                               for g, s in zip(nominal, shares, strict=True))
                if complete:
                    total = math.fsum(g for g in nominal if g is not None)
                    exposure = math.fsum(g * s / 100 for g, s in zip(nominal, shares, strict=True)
                                         if g is not None and s is not None)
                    share = valid_share(p.ratio(exposure, total))
                else:
                    share = None
                results.append(Exposure("wdi", entity, year, portfolio, code,
                                        "current_price_gdp", share))
    return results


def exposure_record(exposure: Exposure) -> Record:
    """Expose the denominator and input coverage for every result."""
    return {
        "dataset": exposure.dataset, "entity": exposure.entity, "year": exposure.year,
        "portfolio": exposure.portfolio, "sectors": exposure.sectors,
        "denominator": exposure.denominator,
        "target_share_pct": None if exposure.share is None else 100 * exposure.share,
        "source_flags": exposure.source_flags, "available": exposure.share is not None,
    }


def scenario_rows(exposures: list[Exposure], scenarios: list[Scenario] | tuple[Scenario, ...]) -> list[Record]:
    """Keep observed weights and hypothetical assumptions separately identifiable."""
    return [
        {**exposure_record(exposure), "scenario": scenario.name,
         "displaced_fraction": scenario.displaced, "price_cut_fraction": scenario.price_cut,
         "demand_elasticity": scenario.elasticity,
         "domestic_replacement_fraction": scenario.domestic_replacement,
         "replacement_value_intensity": scenario.replacement_intensity,
         **effects(exposure.share, scenario)}
        for exposure in exposures for scenario in scenarios
    ]


def reverse_stress(exposure: Exposure, loss_pct: float) -> Record:
    """Find the isolated price/volume cut needed for a target nominal loss.

    Assumes no demand response, no replacement and unchanged other activity.
    A 100% price reduction is a limiting case, not a valid price grid point.
    """
    if not math.isfinite(loss_pct) or loss_pct <= 0:
        raise ValueError("Target loss must be positive and finite")
    required = loss_pct / (100 * exposure.share) if exposure.share is not None and exposure.share > 0 else None
    return {
        **exposure_record(exposure), "target_nominal_loss_pct_aggregate": loss_pct,
        "required_isolated_cut_fraction": required,
        "reachable_by_volume_cut": None if exposure.share is None else required is not None and required <= 1,
        "reachable_by_strictly_positive_price": None if exposure.share is None else required is not None and required < 1,
    }


def run(data_dir: Path, output: Path, fetch: bool = False) -> None:
    """Reproduce all stress results from previously registered source bytes."""
    gva = euro.decode(euro.load_input("a64", 0, data_dir, fetch), "B1G", ("CP_MEUR",))
    wdi = p.decode_rows(p.verified_files(p.WDI, data_dir, fetch), p.WDI)
    exposures = euro_exposures(gva) + wdi_exposures(wdi)
    scenarios = grid()
    tables = {
        "exposures.csv": [exposure_record(x) for x in exposures],
        "named_scenarios.csv": scenario_rows(exposures, NAMED),
        "sensitivity_grid.csv": scenario_rows(exposures, scenarios),
        "reverse_stress.csv": [reverse_stress(x, loss) for x in exposures for loss in (1, 5, 10)],
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        p.write_csv(output / name, rows)
    summary = {
        "registry_commits": {"wdi": p.COMMIT, "eurostat_a64": euro.SOURCES["a64"][1]},
        "eurostat_years": list(EURO_YEARS), "wdi_years": list(WDI_YEARS),
        "named_scenarios": [scenario.name for scenario in NAMED],
        "grid_size": len(scenarios),
        "available_exposures": sum(x.share is not None for x in exposures),
        "missing_exposures": sum(x.share is None for x in exposures),
        "tables": {name: len(rows) for name, rows in tables.items()},
        "scope": "Hypothetical direct static accounting; not an empirical AI exposure estimate, forecast or welfare measure.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    """Run using existing pinned data contracts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=p.HERE / "data")
    parser.add_argument("--output", type=Path, default=p.HERE / "results" / "substitution")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    run(args.data_dir, args.output, args.fetch)


if __name__ == "__main__":
    main()
