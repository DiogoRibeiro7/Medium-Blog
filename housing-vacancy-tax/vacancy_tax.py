"""Reproducible accounting and illustrative vacancy-tax scenarios for Portugal.

This module does not estimate a causal vacancy-tax treatment effect. Census
counts come from the IHRU/OHARU (2024) transcription of Census 2021 Table 11;
all owner-response parameters are explicitly hypothetical.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Literal

MarketStatus = Literal["for_sale_or_rent", "other_reasons"]
RepairStatus = Literal["no_or_minor_repairs", "medium_or_major_repairs"]

TOTAL_CLASSIC_DWELLINGS = 5_970_677
HABITUAL_RESIDENCES = 4_142_581
SECONDARY_RESIDENCES = 1_104_881
VACANT_DWELLINGS = 723_215
OTHER_REASONS_NO_OR_MINOR_REPAIRS = 248_534


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
    """Hypothetical property tax and rental-return inputs; not market estimates."""

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


def load_census(path: Path) -> list[CensusCell]:
    """Parse and strictly validate the fixed, cited 2021 national census table."""
    cells: list[CensusCell] = []
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["market_status", "repair_status", "dwellings"]:
            raise ValueError("Unexpected CSV columns")
        for row in reader:
            cells.append(
                CensusCell(
                    market_status=row["market_status"],  # runtime-checked by dataclass
                    repair_status=row["repair_status"],  # runtime-checked by dataclass
                    dwellings=int(row["dwellings"]),
                )
            )
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


def imi_liability_eur(scenario: Scenario, multiplier: Decimal) -> Decimal:
    """Annual IMI on *taxable assessed value* (VPT), not market value.

    Multiplier is 1 (ordinary), 3 (vacant), 10 (pressure-zone year 1),
    or 20 (standard statutory pressure-zone maximum); legal exemptions and
    possible municipal variations are deliberately outside this toy model.
    """
    if multiplier < 1:
        raise ValueError("A valid illustrative multiplier must be >= 1")
    return scenario.taxable_property_value_eur * scenario.imi_rate * multiplier


def net_rental_flow_before_imi_eur(scenario: Scenario) -> Decimal:
    """Expected annual net cashflow from letting, before IMI and vacancy option value."""
    return (
        Decimal(12)
        * scenario.monthly_rent_eur
        * scenario.occupied_fraction
        * scenario.gross_rent_retained_fraction
        - scenario.additional_annual_rental_cost_eur
    )


def max_option_value_consistent_with_letting_eur(scenario: Scenario, multiplier: Decimal) -> Decimal:
    """Owner's reservation-option threshold for choosing rental over vacancy.

    With both choices bearing ordinary IMI, an additional vacant-property tax
    (multiplier - 1)*ordinary_IMI makes letting attractive when:

       value_of_keeping_empty <= net_rental_flow + extra_vacancy_tax.

    The same hypothetical capital appreciation under both choices cancels.
    This is a decision identity, not a behavioural/causal estimate.
    """
    base = imi_liability_eur(scenario, Decimal(1))
    extra_tax = imi_liability_eur(scenario, multiplier) - base
    return net_rental_flow_before_imi_eur(scenario) + extra_tax


def hypothetical_released_dwellings(pool: int, eligible_share: Decimal, response_share: Decimal) -> Decimal:
    """Upper-level scenario arithmetic, NOT a forecast of additional housing.

    The shares describe two unobserved filters: the subset legally/physically
    eligible and the subset of this eligible group that changes its use.
    """
    if type(pool) is not int or pool < 0:
        raise ValueError("pool must be a nonnegative integer")
    if any(not Decimal(0) <= x <= Decimal(1) for x in (eligible_share, response_share)):
        raise ValueError("shares must be between zero and one")
    return Decimal(pool) * eligible_share * response_share


def census_summary(cells: Iterable[CensusCell]) -> dict[str, int]:
    """Produce auditable counts and keep the Census categories separate."""
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
