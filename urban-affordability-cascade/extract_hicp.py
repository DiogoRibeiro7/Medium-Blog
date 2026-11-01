"""Validate a frozen official Eurostat Portugal annual HICP JSON-stat extract.

This source is a *national general-consumer-price deflator*, not a housing
price index, housing affordability measure, or a municipal inflation series.
The official response bytes are pinned and parsing never contacts the API.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from analysis import ROOT


SOURCE = ROOT / "data" / "portugal_eurostat_hicp_2019_2025.json"
SOURCE_SHA256 = "11742a14bd70d43c22e7170b07b65081f582d2a85eeac22005448f20327bd117"
SOURCE_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "prc_hicp_ainr?geo=PT&freq=A&coicop18=TOTAL&unit=INX_A_AVG"
    "&sinceTimePeriod=2019&untilTimePeriod=2025"
)
DOI = "https://doi.org/10.2908/PRC_HICP_AINR"
OUTPUT = "portugal_hicp_annual_2019_2025.csv"
PROVENANCE = "portugal_hicp_annual_provenance.json"
EXPECTED_YEARS = tuple(range(2019, 2026))
EXPECTED_DIMENSIONS = ("freq", "unit", "coicop18", "geo", "time")
EXPECTED_MEMBERS = ("A", "INX_A_AVG", "TOTAL", "PT")


def parse_hicp_json(document: dict[str, Any]) -> dict[int, float]:
    """Validate exact JSON-stat selectors, no flags/missing cells, and 7 years.

    This is also separately testable against synthetic corruptions. It is not
    a replacement for checking the original downloaded source SHA-256.
    """
    if not isinstance(document, dict) or document.get("class") != "dataset":
        raise ValueError("Expected a Eurostat JSON-stat dataset")
    if document.get("source") != "ESTAT" or document.get("id") != list(EXPECTED_DIMENSIONS):
        raise ValueError("Wrong publisher or JSON-stat dimension identity")
    if document.get("size") != [1, 1, 1, 1, len(EXPECTED_YEARS)]:
        raise ValueError("Unexpected HICP dimension sizes")
    dimensions = document.get("dimension")
    if not isinstance(dimensions, dict) or set(dimensions) != set(EXPECTED_DIMENSIONS):
        raise ValueError("Wrong HICP dimension set")
    for name, code in zip(EXPECTED_DIMENSIONS[:-1], EXPECTED_MEMBERS):
        index = dimensions[name].get("category", {}).get("index")
        if index != {code: 0}:
            raise ValueError(f"Wrong selected {name} code")
    time_index = dimensions["time"].get("category", {}).get("index")
    expected_index = {str(year): position for position, year in enumerate(EXPECTED_YEARS)}
    if time_index != expected_index:
        raise ValueError("Missing, duplicated or misordered annual HICP times")
    values = document.get("value")
    if not isinstance(values, dict) or set(values) != {str(i) for i in range(len(EXPECTED_YEARS))}:
        raise ValueError("Missing or extra HICP annual observations")
    statuses = document.get("status", {})
    if statuses not in ({}, None):
        raise ValueError("Flagged HICP observation needs explicit manual review")
    observed: dict[int, float] = {}
    for idx, year in enumerate(EXPECTED_YEARS):
        value = values[str(idx)]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Non-numeric HICP index")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("HICP indices must be positive and finite")
        observed[year] = float(value)
    if observed[2025] != 100.0:
        raise ValueError("The selected index must have 2025=100")
    if (observed[2019], observed[2020], observed[2022]) != (83.07, 82.97, 90.54):
        raise ValueError("Published Portugal HICP reference observations changed")
    return observed


def load_hicp(path: Path = SOURCE) -> dict[int, float]:
    """Load the byte-identical frozen Eurostat response, rejecting revisions."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("HICP source SHA-256 mismatch: source revision requires review")
    return parse_hicp_json(json.loads(raw))


def run(output: Path = ROOT) -> dict[str, Any]:
    """Write the validated seven-row extract and attributable provenance."""
    series = load_hicp()
    folder = output / "data"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / OUTPUT
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("year", "portugal_hicp_annual_average_2025_100"))
        for year in EXPECTED_YEARS:
            writer.writerow((year, f"{series[year]:.2f}"))
    source_document = json.loads(SOURCE.read_text(encoding="utf-8"))
    report = {
        "evidence_class": "observed_official_national_consumer_price_index",
        "publisher": "Eurostat",
        "dataset": "prc_hicp_ainr",
        "doi": DOI,
        "source_query": SOURCE_URL,
        "accessed_on_utc": "2026-11-01",
        "source_dataset_updated": source_document["updated"],
        "source_response_sha256": SOURCE_SHA256,
        "derived_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dimensions": {"geo": "PT", "freq": "A", "coicop18": "TOTAL", "unit": "INX_A_AVG"},
        "reference_period": "2025=100; annual mean of monthly indices",
        "years": list(EXPECTED_YEARS),
        "source_credit_and_reuse": (
            "Eurostat statistical data may be reused with source acknowledgement; "
            "https://ec.europa.eu/eurostat/help/copyright-notice"
        ),
        "limits": [
            "national all-items consumer prices, not local municipal price levels",
            "not a house-sale price index, new-contract rent index, or housing cost burden",
            "2025=100 index base does not change ratio-based growth calculations",
            "a new API response with different checksum must be reviewed and repinned",
        ],
    }
    (folder / PROVENANCE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = run(args.output)
    print(f"Validated Eurostat {report['dataset']} national HICP for {len(report['years'])} years")


if __name__ == "__main__":
    main()
