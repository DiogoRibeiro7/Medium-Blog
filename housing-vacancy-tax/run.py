"""Write every table and number used in the article.

From the companion directory:
    python run.py
    python figures.py
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from vacancy_tax import (
    OECD_IDENTIFIED_2023,
    OTHER_REASONS_NO_OR_MINOR_REPAIRS,
    VACANT_DWELLINGS,
    Scenario,
    audit_sample_size,
    break_even_additional_leases,
    census_summary,
    cost_per_additional_lease_eur,
    detection_probability_matching_rate_cut,
    expected_surcharge_eur,
    hypothetical_released_dwellings,
    imi_liability_eur,
    load_census,
    load_identification,
    load_rate_cut_cost,
    load_vancouver,
    max_option_value_consistent_with_letting_eur,
    net_rental_flow_before_imi_eur,
    rate_cut_gain_upper_bound_eur,
    rent_received_eur,
    surcharge_eur,
)

DEFAULT_ROOT = Path(__file__).resolve().parent
MULTIPLIERS = (Decimal(1), Decimal(3), Decimal(10), Decimal(20))
ADDITIONAL_LEASES = (1_000, 2_500, 5_000, 10_000, 20_000, 25_000, 50_000, 100_000, 200_000)

# Viseu, ECO/Lusa 27 March 2025: 103 vacant buildings in the pressure zone,
# surcharge of about EUR 58,000 in total. Used only for a back-of-envelope
# range of the implied taxable value at the legal range of IMI rates.
VISEU_BUILDINGS = 103
VISEU_SURCHARGE_EUR = Decimal(58_000)


def money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def create_results(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Write results from the transcribed sources and the stated assumptions."""
    data = root / "data"
    cells = load_census(data / "census_2021_ihru_table11.csv")
    flagged = load_identification(data / "municipal_identification.csv")
    vancouver = load_vancouver(data / "vancouver_eht.csv")
    costs = load_rate_cut_cost(data / "rate_cut_utao.csv")
    scenario = Scenario()
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)

    cut_gain = rate_cut_gain_upper_bound_eur(scenario)
    tax_rows = [
        {
            "multiplier": str(m),
            "annual_imi_eur": money(imi_liability_eur(scenario, m)),
            "surcharge_eur": money(surcharge_eur(scenario, m)),
            "option_value_threshold_eur": money(max_option_value_consistent_with_letting_eur(scenario, m)),
            "option_value_threshold_with_rate_cut_eur": money(
                max_option_value_consistent_with_letting_eur(scenario, m, rate_cut=True)
            ),
        }
        for m in MULTIPLIERS
    ]
    write_csv(out / "tax_scenarios.csv", tax_rows)

    expected_rows = []
    for step in range(0, 101):
        pi = Decimal(step) / Decimal(100)
        row = {"detection_probability": str(pi)}
        for m in MULTIPLIERS[1:]:
            row[f"expected_surcharge_{m}x_eur"] = money(expected_surcharge_eur(scenario, m, pi))
        expected_rows.append(row)
    write_csv(out / "expected_surcharge.csv", expected_rows)

    irs = costs["irs_autonomous_rate_25_to_10"]
    irs_irc = irs + costs["irc_exclusion_of_half_of_rental_income"]
    lease_rows = [
        {
            "additional_leases": str(n),
            "cost_per_additional_lease_irs_eur": money(cost_per_additional_lease_eur(irs, n)),
            "cost_per_additional_lease_irs_irc_eur": money(cost_per_additional_lease_eur(irs_irc, n)),
        }
        for n in ADDITIONAL_LEASES
    ]
    write_csv(out / "cost_per_additional_lease.csv", lease_rows)

    annual_contract_rent = Decimal(12) * scenario.monthly_rent_eur
    break_even = {
        "benchmark_annual_contract_rent_eur": money(annual_contract_rent),
        "irs_only": str(break_even_additional_leases(irs, annual_contract_rent).quantize(Decimal(1))),
        "irs_and_irc": str(break_even_additional_leases(irs_irc, annual_contract_rent).quantize(Decimal(1))),
    }

    census = census_summary(cells)
    detection = {
        "flagged_vacant_2023": flagged.vacant_total,
        "flagged_general_rule": flagged.vacant_general_rule,
        "flagged_pressure_zones": flagged.vacant_pressure_zones,
        "flagged_ruins": flagged.ruins,
        "municipalities_applying": flagged.municipalities_applying,
        "oecd_identified_2023": OECD_IDENTIFIED_2023,
        "share_of_census_vacant": round(flagged.vacant_total / VACANT_DWELLINGS, 4),
        "share_of_other_reasons_good_condition": round(flagged.vacant_total / OTHER_REASONS_NO_OR_MINOR_REPAIRS, 4),
        "share_of_vacant_other_reasons": round(flagged.vacant_total / census["vacant_for_other_reasons"], 4),
    }
    scale_pi = Decimal(flagged.vacant_total) / Decimal(OTHER_REASONS_NO_OR_MINOR_REPAIRS)

    audit_rows = [
        {"population": str(pop), "margin": str(e), "sample_size": str(audit_sample_size(pop, e))}
        for pop in (OTHER_REASONS_NO_OR_MINOR_REPAIRS, VACANT_DWELLINGS)
        for e in (0.01, 0.02, 0.03, 0.05)
    ]
    write_csv(out / "audit_sample_sizes.csv", audit_rows)

    sensitivity_rows = [
        {
            "eligible_share": str(f),
            "response_share": str(p),
            "hypothetically_released_dwellings": str(
                hypothetical_released_dwellings(OTHER_REASONS_NO_OR_MINOR_REPAIRS, f, p).quantize(Decimal(1))
            ),
        }
        for f in (Decimal("0.05"), Decimal("0.10"), Decimal("0.25"), Decimal("0.50"))
        for p in (Decimal("0.05"), Decimal("0.10"), Decimal("0.20"), Decimal("0.40"))
    ]
    write_csv(out / "sensitivity.csv", sensitivity_rows)

    viseu_per_building = VISEU_SURCHARGE_EUR / VISEU_BUILDINGS
    viseu_ordinary = viseu_per_building / Decimal(9)
    viseu = {
        "surcharge_per_building_eur": money(viseu_per_building),
        "implied_ordinary_imi_eur_if_first_year_10x": money(viseu_ordinary),
        "implied_vpt_eur_at_0.45pct": money(viseu_ordinary / Decimal("0.0045")),
        "implied_vpt_eur_at_0.30pct": money(viseu_ordinary / Decimal("0.0030")),
    }

    first, last = vancouver[0], vancouver[-1]
    vancouver_summary = {
        "declared_vacancy_rate_2017_pct": str(first.declared_vacancy_rate_pct),
        "declared_vacancy_rate_2024_pct": str(last.declared_vacancy_rate_pct),
        "relative_fall_in_rate": round(float(1 - last.declared_vacancy_rate_pct / first.declared_vacancy_rate_pct), 3),
        "vacant_properties_2024": last.vacant_properties,
        "required_to_declare_2024": last.required_to_declare,
    }

    pi_match = {
        f"{m}x": str(detection_probability_matching_rate_cut(scenario, m).quantize(Decimal("0.001")))
        for m in MULTIPLIERS[1:]
    }
    at_scale = {
        f"{m}x": money(expected_surcharge_eur(scenario, m, scale_pi)) for m in MULTIPLIERS[1:]
    }

    summary: dict[str, Any] = {
        "census_2021": census,
        "detection_tax_year_2023": detection,
        "hypothetical_parameters": {name: str(value) for name, value in asdict(scenario).items()},
        "net_rental_flow_before_imi_eur": money(net_rental_flow_before_imi_eur(scenario)),
        "rent_received_eur": money(rent_received_eur(scenario)),
        "rate_cut_gain_upper_bound_eur": money(cut_gain),
        "tax_scenarios": tax_rows,
        "detection_probability_matching_rate_cut": pi_match,
        "scale_detection_ratio": str(scale_pi.quantize(Decimal("0.0001"))),
        "expected_surcharge_at_scale_ratio_eur": at_scale,
        "rate_cut_annual_cost_eur": {"irs": money(irs), "irs_and_irc": money(irs_irc)},
        "break_even_additional_leases": break_even,
        "cost_per_additional_lease": lease_rows,
        "audit_sample_sizes": audit_rows,
        "viseu_back_of_envelope": viseu,
        "vancouver": vancouver_summary,
        "interpretation": (
            "Census and tax-administration counts are observed; tax amounts are arithmetic on a "
            "hypothetical dwelling; detection ratios compare different units and years and are "
            "given for scale only; nothing here is a causal estimate."
        ),
    }
    with (out / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)
        output.write("\n")
    (out / "summary.txt").write_text(render_text(summary), encoding="utf-8")
    return summary


def render_text(s: dict[str, Any]) -> str:
    """A plain listing of every number the article quotes."""
    c, d = s["census_2021"], s["detection_tax_year_2023"]
    lines = [
        "CENSUS 2021 (INE; IHRU Table 11)",
        f"  classic dwellings {c['total_classic_dwellings']:,}; habitual {c['habitual_residences']:,}; "
        f"secondary {c['secondary_residences']:,}; vacant {c['total_vacant']:,}",
        f"  vacant for sale or rent {c['vacant_for_sale_or_rent']:,}; for other reasons {c['vacant_for_other_reasons']:,}",
        f"  other reasons, no or minor repairs {c['vacant_other_reasons_no_or_minor_repairs']:,}",
        "",
        "FLAGGED FOR THE SURCHARGE, TAX YEAR 2023 (Lusa, 1 May 2024)",
        f"  municipalities {d['municipalities_applying']}; vacant {d['flagged_vacant_2023']:,} "
        f"(general rule {d['flagged_general_rule']:,}, pressure zones {d['flagged_pressure_zones']:,}); "
        f"ruins {d['flagged_ruins']:,}; OECD count {d['oecd_identified_2023']:,}",
        f"  as a share of census vacant {d['share_of_census_vacant']:.2%}; of vacant for other reasons "
        f"{d['share_of_vacant_other_reasons']:.2%}; of the good-condition other-reasons group "
        f"{d['share_of_other_reasons_good_condition']:.2%}",
        "",
        "ILLUSTRATIVE DWELLING (assumptions, not estimates)",
        f"  parameters {s['hypothetical_parameters']}",
        f"  rent received {s['rent_received_eur']}; net rental flow before IMI {s['net_rental_flow_before_imi_eur']}",
        f"  most the 25% -> 10% cut adds: {s['rate_cut_gain_upper_bound_eur']} a year",
    ]
    for row in s["tax_scenarios"]:
        lines.append(
            f"  {row['multiplier']}x: IMI {row['annual_imi_eur']}, surcharge {row['surcharge_eur']}, "
            f"largest U consistent with letting {row['option_value_threshold_eur']} "
            f"({row['option_value_threshold_with_rate_cut_eur']} with the rate cut)"
        )
    lines += [
        f"  detection probability at which the expected surcharge equals the cut's bound: {s['detection_probability_matching_rate_cut']}",
        f"  expected surcharge at the scale ratio {s['scale_detection_ratio']}: {s['expected_surcharge_at_scale_ratio_eur']}",
        "",
        "THE 2026 RATE CUT (UTAO)",
        f"  annual cost {s['rate_cut_annual_cost_eur']}",
        f"  additional leases at which cost per lease equals the annual contract rent "
        f"{s['break_even_additional_leases']['benchmark_annual_contract_rent_eur']}: "
        f"IRS {s['break_even_additional_leases']['irs_only']}, IRS+IRC {s['break_even_additional_leases']['irs_and_irc']}",
    ]
    for row in s["cost_per_additional_lease"]:
        lines.append(
            f"  {int(row['additional_leases']):>7,} additional leases: {row['cost_per_additional_lease_irs_eur']} "
            f"(IRS) / {row['cost_per_additional_lease_irs_irc_eur']} (IRS+IRC) per lease per year"
        )
    lines += ["", "AUDIT SAMPLE SIZES (95%, worst case p = 0.5)"]
    for row in s["audit_sample_sizes"]:
        lines.append(f"  population {int(row['population']):,}, margin {row['margin']}: n = {int(row['sample_size']):,}")
    v, b = s["vancouver"], s["viseu_back_of_envelope"]
    lines += [
        "",
        "VISEU, BACK OF ENVELOPE (103 buildings, about EUR 58,000)",
        f"  {b}",
        "",
        "VANCOUVER (City EHT annual reports)",
        f"  {v}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Folder containing data/")
    args = parser.parse_args()
    summary = create_results(args.root)
    print(render_text(summary))


if __name__ == "__main__":
    main()
