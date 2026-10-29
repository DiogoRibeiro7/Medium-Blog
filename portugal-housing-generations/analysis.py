"""Reproduce selected descriptive claims about Portuguese housing affordability.

The only time-series input is the official Eurostat JSON-stat response for
`prc_hpi_a`, Portugal, total purchases, annual average index (2015=100).
No causal effects or electoral preferences are estimated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EUROSTAT_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "prc_hpi_a?geo=PT&purchase=TOTAL&unit=I15_A_AVG"
)


def parse_hpi(document: dict[str, Any]) -> dict[int, float]:
    """Decode annual Portugal HPI from JSON-stat using dimension indexes.

    Raises:
        ValueError: If dimensions, filters, or numeric observations are invalid.
    """
    dimensions = document.get("id")
    sizes = document.get("size")
    if not isinstance(dimensions, list) or not isinstance(sizes, list):
        raise ValueError("Missing JSON-stat dimensions")
    if len(dimensions) != len(sizes) or not all(isinstance(n, int) and n > 0 for n in sizes):
        raise ValueError("Invalid JSON-stat dimension sizes")

    expected = {"geo": "PT", "purchase": "TOTAL", "unit": "I15_A_AVG", "freq": "A"}
    dimension_data = document.get("dimension", {})
    for dimension, code in expected.items():
        if dimension not in dimensions:
            raise ValueError(f"Missing dimension: {dimension}")
        indexes = dimension_data[dimension]["category"]["index"]
        if not isinstance(indexes, dict) or list(indexes) != [code]:
            raise ValueError(f"Unexpected selection for {dimension}: {indexes}")

    time_index = dimension_data["time"]["category"]["index"]
    if not isinstance(time_index, dict):
        raise ValueError("Invalid time dimension")
    values = document.get("value", {})
    if not isinstance(values, dict):
        raise ValueError("Invalid observations")
    result: dict[int, float] = {}
    for year, position in time_index.items():
        if not isinstance(position, int) or not isinstance(year, str) or not year.isdigit():
            raise ValueError("Invalid year index")
        coordinates = {name: 0 for name in dimensions}
        coordinates["time"] = position
        flat_index = 0
        for name, size in zip(dimensions, sizes, strict=True):
            flat_index = flat_index * size + coordinates[name]
        observation = values.get(str(flat_index))
        if observation is None:
            continue  # Eurostat can omit missing early observations.
        if not isinstance(observation, (float, int)) or isinstance(observation, bool):
            raise ValueError(f"Non-numeric observation in {year}")
        result[int(year)] = float(observation)
    if 2015 not in result or abs(result[2015] - 100.0) > 0.01:
        raise ValueError("Expected 2015=100 HPI series")
    return dict(sorted(result.items()))


def summarize(series: dict[int, float]) -> dict[str, float | int]:
    """Compute transparent arithmetic comparisons, not causal estimates."""
    if 2015 not in series or 2025 not in series or 2024 not in series:
        raise ValueError("Years 2015, 2024 and 2025 are required")
    return {
        "hpi_2015": series[2015],
        "hpi_2024": series[2024],
        "hpi_2025": series[2025],
        "growth_2015_2025_pct": round(100 * (series[2025] / series[2015] - 1), 2),
        "growth_2024_2025_pct": round(100 * (series[2025] / series[2024] - 1), 2),
        "owner_1977_1986_pct": 61,
        "owner_1987_1996_pct": 42,
        "owner_gap_percentage_points": 42 - 61,
        "vacant_dwellings_2021_pct": 12,
        "secondary_dwellings_2021_pct": 19,
    }


def main() -> None:
    """Read a pinned source snapshot and export auditable derived data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "data" / "eurostat_prc_hpi_a_PT_2026-10-29.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results.json")
    args = parser.parse_args()
    raw: Any = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Expected JSON-stat object")
    series = parse_hpi(raw)
    report = {"source_url": EUROSTAT_URL, "source_updated": raw.get("updated"), "yearly_hpi": series, "descriptives": summarize(series)}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["descriptives"], indent=2))


if __name__ == "__main__":
    main()
