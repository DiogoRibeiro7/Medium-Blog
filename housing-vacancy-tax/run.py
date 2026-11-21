"""Generate deterministic tables for the Portugal vacancy-tax article.

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
from decimal import Decimal
from pathlib import Path
from typing import Any

from vacancy_tax import (
    OTHER_REASONS_NO_OR_MINOR_REPAIRS,
    Scenario,
    census_summary,
    hypothetical_released_dwellings,
    imi_liability_eur,
    load_census,
    max_option_value_consistent_with_letting_eur,
    net_rental_flow_before_imi_eur,
)

DEFAULT_ROOT = Path(__file__).resolve().parent


def create_results(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Write results from the attributed census table and explicit assumptions."""
    data_path = root / "data" / "census_2021_ihru_table11.csv"
    cells = load_census(data_path)
    scenario = Scenario()
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    multipliers: tuple[Decimal, ...] = (Decimal(1), Decimal(3), Decimal(10), Decimal(20))
    tax_rows: list[dict[str, str]] = []
    for multiplier in multipliers:
        liability = imi_liability_eur(scenario, multiplier)
        tax_rows.append({
            "multiplier": str(multiplier),
            "annual_imi_eur": str(liability),
            "incremental_vs_ordinary_eur": str(liability - imi_liability_eur(scenario, Decimal(1))),
            "option_value_threshold_eur": str(max_option_value_consistent_with_letting_eur(scenario, multiplier)),
        })
    with (results_dir / "tax_scenarios.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(tax_rows[0]))
        writer.writeheader()
        writer.writerows(tax_rows)

    eligible_shares = tuple(Decimal(x) for x in ("0.05", "0.10", "0.25", "0.50"))
    response_shares = tuple(Decimal(x) for x in ("0.05", "0.10", "0.20", "0.40"))
    sensitivity_rows: list[dict[str, str]] = []
    for eligible in eligible_shares:
        for response in response_shares:
            homes = hypothetical_released_dwellings(OTHER_REASONS_NO_OR_MINOR_REPAIRS, eligible, response)
            sensitivity_rows.append({
                "eligible_share": str(eligible),
                "response_share": str(response),
                "hypothetically_released_dwellings": str(homes),
            })
    with (results_dir / "sensitivity.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)

    summary: dict[str, Any] = {
        "source": "IHRU/OHARU (2024), Table 11, Portugal Census 2021",
        "observation_year": 2021,
        "source_report_url": "https://www.portaldahabitacao.pt/documents/20126/58203/1_Estudo%2BPNH-_VFinal.pdf/9ed08595-a9d2-0708-d845-ec3a93d56c3e",
        "census": census_summary(cells),
        "hypothetical_parameters": {name: str(value) for name, value in asdict(scenario).items()},
        "net_rental_flow_before_imi_eur": str(net_rental_flow_before_imi_eur(scenario)),
        "tax_scenarios": tax_rows,
        "sensitivity_scenarios": sensitivity_rows,
        "interpretation": "Census is 2021 observed stock; tax liabilities are arithmetic; owner-choice thresholds and mobilisations are illustrative, not causal forecasts.",
    }
    with (results_dir / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)
        output.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Working folder containing data/")
    args = parser.parse_args()
    summary = create_results(args.root)
    print(json.dumps({"census": summary["census"], "tax_scenarios": summary["tax_scenarios"]}, indent=2))


if __name__ == "__main__":
    main()
