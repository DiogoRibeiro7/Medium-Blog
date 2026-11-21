"""Accounting for Portugal's IMI surcharge on vacant homes.

Observed inputs are hand transcriptions of published figures (see
``data/SOURCES.md``); each loader checks them against the values below and
stops if anything differs. Everything else is arithmetic on stated,
hypothetical parameters. Nothing here estimates a causal effect of the tax.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Literal

MarketStatus = Literal["for_sale_or_rent", "other_reasons"]
RepairStatus = Literal["no_or_minor_repairs", "medium_or_major_repairs"]

# INE/LNEC (2023), Census 2021.
TOTAL_CLASSIC_DWELLINGS = 5_970_677
HABITUAL_RESIDENCES = 4_142_581
SECONDARY_RESIDENCES = 1_104_881
VACANT_DWELLINGS = 723_215
OTHER_REASONS_NO_OR_MINOR_REPAIRS = 248_534

# OECD (2026) count of vacant dwellings identified by municipalities in 2023,
# used only as a cross-check of the Lusa figures.
OECD_IDENTIFIED_2023 = 12_803

# The 2026 cut in the autonomous IRS rate on residential rents.
RATE_BEFORE_CUT = Decimal("0.25")
RATE_AFTER_CUT = Decimal("0.10")


@dataclass(frozen=True)
class CensusCell:
    """Mutually exclusive dwellings in one IHRU Table 11 census cell."""

    market_status: MarketStatus
    repair_status: RepairStatus
    dwellings: int

    def __post_init__(self) -> None:
        if self.market_status not in ("for_sale_or_rent", "other_reasons"):
            raise ValueError("Invalid market_status")
        if self.repair_status not in ("no_or_minor_repairs", "medium_or_major_repairs"):
            raise ValueError("Invalid repair_status")
        if type(self.dwellings) is not int or self.dwellings < 0:
            raise ValueError("dwellings must be a nonnegative integer")


@dataclass(frozen=True)
class Scenario:
    """Hypothetical property and rental inputs for one dwelling; not estimates."""

    taxable_property_value_eur: Decimal = Decimal("150000")
    imi_rate: Decimal = Decimal("0.0035")
    monthly_rent_eur: Decimal = Decimal("900")
    occupied_fraction: Decimal = Decimal("0.90")
    gross_rent_retained_fraction: Decimal = Decimal("0.70")
    additional_annual_rental_cost_eur: Decimal = Decimal("1200")

    def __post_init__(self) -> None:
        if self.taxable_property_value_eur <= 0 or self.monthly_rent_eur < 0:
            raise ValueError("Property value must be positive; rent nonnegative")
        if not Decimal("0") < self.imi_rate <= Decimal("0.01"):
            raise ValueError("IMI rate must be a proportion between 0 and 1%")
        for field in (self.occupied_fraction, self.gross_rent_retained_fraction):
            if not Decimal("0") <= field <= Decimal("1"):
                raise ValueError("Fractions must be in [0, 1]")
        if self.additional_annual_rental_cost_eur < 0:
            raise ValueError("Additional annual cost must be nonnegative")


@dataclass(frozen=True)
class MunicipalIdentification:
    """Vacant and ruined properties flagged for the aggravated IMI rate."""

    tax_year: int
    municipalities_applying: int
    vacant_general_rule: int
    vacant_pressure_zones: int
    ruins: int

    @property
    def vacant_total(self) -> int:
        return self.vacant_general_rule + self.vacant_pressure_zones


@dataclass(frozen=True)
class VancouverYear:
    """One reference year of Vancouver's Empty Homes Tax monitoring data."""

    reference_year: int
    declared_vacancy_rate_pct: Decimal
    eht_rate_pct: Decimal
    vacant_properties: int | None
    required_to_declare: int | None


def _read_rows(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != columns:
            raise ValueError(f"Unexpected columns in {path.name}")
        return list(reader)


def load_census(path: Path) -> list[CensusCell]:
    """Parse and strictly validate the cited 2021 national census table."""
    rows = _read_rows(path, ["market_status", "repair_status", "dwellings"])
    cells = [
        CensusCell(
            market_status=row["market_status"],  # runtime-checked by dataclass
            repair_status=row["repair_status"],  # runtime-checked by dataclass
            dwellings=int(row["dwellings"]),
        )
        for row in rows
    ]
    expected: dict[tuple[str, str], int] = {
        ("for_sale_or_rent", "no_or_minor_repairs"): 236_927,
        ("for_sale_or_rent", "medium_or_major_repairs"): 111_170,
        ("other_reasons", "no_or_minor_repairs"): 248_534,
        ("other_reasons", "medium_or_major_repairs"): 126_584,
    }
    observed = {(x.market_status, x.repair_status): x.dwellings for x in cells}
    if len(cells) != 4 or observed != expected:
        raise ValueError("Census cells differ from the attributed IHRU 2024 Table 11")
    if sum(x.dwellings for x in cells) != VACANT_DWELLINGS:
        raise ValueError("Census vacancy totals do not reconcile")
    if HABITUAL_RESIDENCES + SECONDARY_RESIDENCES + VACANT_DWELLINGS != TOTAL_CLASSIC_DWELLINGS:
        raise ValueError("National dwelling stock does not reconcile")
    return cells


def load_identification(path: Path) -> MunicipalIdentification:
    """Load the tax-year 2023 municipal counts reported by Lusa (1 May 2024)."""
    rows = _read_rows(
        path,
        ["tax_year", "municipalities_applying", "vacant_general_rule", "vacant_pressure_zones", "ruins"],
    )
    if len(rows) != 1:
        raise ValueError("Expected exactly one tax year")
    record = MunicipalIdentification(**{key: int(value) for key, value in rows[0].items()})
    if (record.tax_year, record.municipalities_applying, record.vacant_general_rule,
            record.vacant_pressure_zones, record.ruins) != (2023, 64, 5_729, 7_047, 4_305):
        raise ValueError("Identification counts differ from the cited Lusa report")
    if abs(record.vacant_total - OECD_IDENTIFIED_2023) / OECD_IDENTIFIED_2023 > 0.01:
        raise ValueError("Lusa and OECD counts disagree by more than 1%")
    return record


def load_vancouver(path: Path) -> list[VancouverYear]:
    """Load Vancouver's declared vacancy rates and tax rates by reference year."""
    rows = _read_rows(
        path,
        ["reference_year", "declared_vacancy_rate_pct", "eht_rate_pct", "vacant_properties", "required_to_declare"],
    )
    years = [
        VancouverYear(
            reference_year=int(row["reference_year"]),
            declared_vacancy_rate_pct=Decimal(row["declared_vacancy_rate_pct"]),
            eht_rate_pct=Decimal(row["eht_rate_pct"]),
            vacant_properties=int(row["vacant_properties"]) if row["vacant_properties"] else None,
            required_to_declare=int(row["required_to_declare"]) if row["required_to_declare"] else None,
        )
        for row in rows
    ]
    if [y.reference_year for y in years] != list(range(2017, 2025)):
        raise ValueError("Vancouver series must cover 2017-2024 in order")
    for year in years:
        if year.vacant_properties is not None and year.required_to_declare is not None:
            implied = Decimal(100 * year.vacant_properties) / Decimal(year.required_to_declare)
            if abs(implied - year.declared_vacancy_rate_pct) > Decimal("0.01"):
                raise ValueError(f"Vancouver {year.reference_year}: count and rate disagree")
    return years


def load_rate_cut_cost(path: Path) -> dict[str, Decimal]:
    """Annual revenue loss of the 2026 cut, in euros, from UTAO's assessment."""
    rows = _read_rows(path, ["measure", "annual_revenue_loss_eur_million"])
    costs = {row["measure"]: Decimal(row["annual_revenue_loss_eur_million"]) * Decimal(1_000_000) for row in rows}
    expected = {
        "irs_autonomous_rate_25_to_10": Decimal("279100000"),
        "irc_exclusion_of_half_of_rental_income": Decimal("29800000"),
    }
    if costs != expected:
        raise ValueError("Rate-cut costs differ from the cited UTAO figures")
    return costs


def imi_liability_eur(scenario: Scenario, multiplier: Decimal) -> Decimal:
    """Annual IMI on the taxable value (VPT), not the market value."""
    if multiplier < 1:
        raise ValueError("A valid illustrative multiplier must be >= 1")
    return scenario.taxable_property_value_eur * scenario.imi_rate * multiplier


def surcharge_eur(scenario: Scenario, multiplier: Decimal) -> Decimal:
    """Extra IMI owed because the dwelling is classified vacant: (m - 1) x B."""
    return imi_liability_eur(scenario, multiplier) - imi_liability_eur(scenario, Decimal(1))


def net_rental_flow_before_imi_eur(scenario: Scenario) -> Decimal:
    """Expected annual net cash flow from letting, before IMI."""
    return (
        Decimal(12)
        * scenario.monthly_rent_eur
        * scenario.occupied_fraction
        * scenario.gross_rent_retained_fraction
        - scenario.additional_annual_rental_cost_eur
    )


def rent_received_eur(scenario: Scenario) -> Decimal:
    """Gross rent received in a year, after allowing for months without a tenant."""
    return Decimal(12) * scenario.monthly_rent_eur * scenario.occupied_fraction


def rate_cut_gain_upper_bound_eur(scenario: Scenario) -> Decimal:
    """Most the 25% -> 10% cut can add to a year's letting income.

    The bound treats all rent received as taxable at the autonomous rate. Rent
    net of deductible expenses is smaller, so the true gain is smaller.
    """
    return (RATE_BEFORE_CUT - RATE_AFTER_CUT) * rent_received_eur(scenario)


def max_option_value_consistent_with_letting_eur(
    scenario: Scenario,
    multiplier: Decimal,
    detection_probability: Decimal = Decimal(1),
    survival_probability: Decimal = Decimal(1),
    rate_cut: bool = False,
) -> Decimal:
    """Largest private value of keeping the dwelling empty, U, at which letting still wins.

    The owner lets when U <= F + pi * rho * (m - 1) * B, where F is the net
    rental flow, pi the probability of being classified vacant, rho the
    probability that the surcharge survives legal challenge, m the multiplier
    and B ordinary IMI. Ordinary IMI and any capital gain are paid or earned
    under both choices and cancel. A decision identity, not an estimate.
    """
    flow = net_rental_flow_before_imi_eur(scenario)
    if rate_cut:
        flow += rate_cut_gain_upper_bound_eur(scenario)
    return flow + expected_surcharge_eur(scenario, multiplier, detection_probability, survival_probability)


def expected_surcharge_eur(
    scenario: Scenario,
    multiplier: Decimal,
    detection_probability: Decimal,
    survival_probability: Decimal = Decimal(1),
) -> Decimal:
    """Expected yearly surcharge: pi x rho x (m - 1) x B."""
    for value in (detection_probability, survival_probability):
        if not Decimal(0) <= value <= Decimal(1):
            raise ValueError("probabilities must be between zero and one")
    return detection_probability * survival_probability * surcharge_eur(scenario, multiplier)


def detection_probability_matching_rate_cut(scenario: Scenario, multiplier: Decimal) -> Decimal:
    """Detection probability at which the expected surcharge equals the cut's upper bound."""
    return rate_cut_gain_upper_bound_eur(scenario) / surcharge_eur(scenario, multiplier)


def cost_per_additional_lease_eur(annual_cost_eur: Decimal, additional_leases: int) -> Decimal:
    """Annual fiscal cost of a general rate cut divided by the leases it adds."""
    if additional_leases <= 0:
        raise ValueError("additional_leases must be positive")
    return annual_cost_eur / Decimal(additional_leases)


def break_even_additional_leases(annual_cost_eur: Decimal, benchmark_eur: Decimal) -> Decimal:
    """Additional leases needed for the cost per additional lease to fall to a benchmark."""
    if benchmark_eur <= 0:
        raise ValueError("benchmark must be positive")
    return annual_cost_eur / benchmark_eur


def audit_sample_size(
    population: int,
    margin: float,
    proportion: float = 0.5,
    z: float = 1.959963984540054,
) -> int:
    """Simple random sample needed to estimate a proportion to +/- margin (95% by default).

    Uses the normal approximation with the finite-population correction. With
    proportion = 0.5 the answer holds whatever the true share turns out to be.
    """
    if population <= 0 or not 0 < margin < 1 or not 0 < proportion < 1:
        raise ValueError("invalid inputs")
    n0 = z * z * proportion * (1 - proportion) / (margin * margin)
    return math.ceil(n0 / (1 + (n0 - 1) / population))


def hypothetical_released_dwellings(pool: int, eligible_share: Decimal, response_share: Decimal) -> Decimal:
    """Pool x f x p: scenario arithmetic, not a forecast of additional housing."""
    if type(pool) is not int or pool < 0:
        raise ValueError("pool must be a nonnegative integer")
    if any(not Decimal(0) <= x <= Decimal(1) for x in (eligible_share, response_share)):
        raise ValueError("shares must be between zero and one")
    return Decimal(pool) * eligible_share * response_share


def census_summary(cells: Iterable[CensusCell]) -> dict[str, int]:
    """Auditable counts that keep the Census categories separate."""
    rows = list(cells)
    return {
        "total_classic_dwellings": TOTAL_CLASSIC_DWELLINGS,
        "habitual_residences": HABITUAL_RESIDENCES,
        "secondary_residences": SECONDARY_RESIDENCES,
        "total_vacant": sum(c.dwellings for c in rows),
        "vacant_for_sale_or_rent": sum(c.dwellings for c in rows if c.market_status == "for_sale_or_rent"),
        "vacant_for_other_reasons": sum(c.dwellings for c in rows if c.market_status == "other_reasons"),
        "vacant_no_or_minor_repairs": sum(c.dwellings for c in rows if c.repair_status == "no_or_minor_repairs"),
        "vacant_other_reasons_no_or_minor_repairs": sum(
            c.dwellings for c in rows
            if c.market_status == "other_reasons" and c.repair_status == "no_or_minor_repairs"
        ),
    }
