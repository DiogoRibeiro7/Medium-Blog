"""Reproduce an explicitly exploratory 2026 citizenship-based comparison.

The billionaire counts are a secondary transcription of Forbes 2026
citizenship-attributed country totals, NOT the original figure's residence
counts. Poverty and GDP are drawn from source_pipeline.py's retrieved,
SHA-256-audited OWID and World Bank inputs. No causal effect is inferred.

Usage:
    python secondary_citizenship.py \
      --snapshots data/source_snapshots \
      --output results/citizenship_exploratory
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import statsmodels.api as sm

from analysis import analyse
from source_pipeline import (
    assemble_panel,
    read_forbes_aggregate,
    read_owid_poverty,
    read_wdi_indicator,
)
from source_preflight import verify_manifest


def run_study(
    snapshots: Path,
    forbes_csv: Path,
    output: Path,
) -> dict[str, Any]:
    """Join genuine external snapshots to secondary counts and fit HC3 OLS.

    Args:
        snapshots: Previously captured and checksum-verified official data.
        forbes_csv: Reviewed secondary citizenship aggregate transcription.
        output: Folder for a machine-readable audit and joined dataset.

    Returns:
        Coverage, original-source hashes, fitted specifications and
        leave-one-country-out sensitivity summaries.

    Raises:
        ValueError: On unverified, inconsistent or inadequate data.
    """
    hashes = verify_manifest(snapshots)
    poverty = read_owid_poverty(snapshots / "owid_30usd_2021_ppp.csv")
    gdp = read_wdi_indicator(
        snapshots / "wdi_gdp_ppp_constant.json",
        indicator="NY.GDP.PCAP.PP.KD",
        field="gdp_ppp_per_capita",
    )
    population = read_wdi_indicator(
        snapshots / "wdi_population.json",
        indicator="SP.POP.TOTL",
        field="population",
    )
    forbes, attribution = read_forbes_aggregate(forbes_csv)
    if attribution != "citizenship":
        raise ValueError("Exploratory comparison accepts citizenship only")
    if int(forbes["billionaires"].sum()) != 3359 or len(forbes) != 76:
        raise ValueError("Secondary transcription unexpectedly changed")
    panel, coverage = assemble_panel(
        poverty=poverty, gdp=gdp, population=population,
        forbes=forbes, method=attribution,
    )
    if len(panel) < 40:
        raise ValueError("Insufficient independently matched observations")
    # The common OLS input contract describes the poverty observation year,
    # not the underlying household survey interview year.
    models = analyse(panel)
    sensitivity: dict[str, dict[str, float | int]] = {}
    for iso3 in ("USA", "CHN", "IND", "RUS"):
        subset = panel[panel["iso3"] != iso3]
        if len(subset) == len(panel):
            continue
        sample = analyse(subset)["models"]["both"]
        sensitivity[iso3] = {
            "n": sample["n"],
            "billionaire_coef": sample["coefficients"][
                "log10_billionaires_per_10m"
            ],
            "hc3_p_value": sample["hc3_p_values"][
                "log10_billionaires_per_10m"
            ],
            "r_squared": sample["r_squared"],
        }
    # Full transcription coverage matters: unreported countries must not
    # become zero-count observations. Keep a record of all excluded names.
    observed = set(panel["iso3"])
    excluded = forbes.loc[
        ~forbes["iso3"].isin(observed), ["iso3", "country", "billionaires"]
    ].sort_values("iso3")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to replace existing evidence output: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output / "matched_citizenship_panel.csv", index=False)
    excluded.to_csv(output / "unmatched_forbes_countries.csv", index=False)
    report: dict[str, Any] = {
        "interpretation": "exploratory correlation only, not causal",
        "not_replication": (
            "2026 Forbes CITIZENSHIP secondary transcription; the original "
            "chart uses a different, unpublished country selection"
        ),
        "source_type": "secondary transcription for Forbes, official OWID/WDI",
        "source_manifest": {
            k: {"sha256": v["sha256"], "url": v["source_url"]}
            for k, v in hashes.items()
        },
        "coverage": coverage,
        "total_secondary_billionaires": int(forbes["billionaires"].sum()),
        "n_forbes_secondary_countries": int(len(forbes)),
        "n_matched_countries": int(len(panel)),
        "n_unmatched_secondary_countries": int(len(excluded)),
        "models": models,
        "leave_one_country_out": sensitivity,
        "limitations": [
            "Transcription is secondary and incomplete; not independently "
            "verified against Forbes individual records.",
            "Country of citizenship is not country of residence.",
            "Poverty observations come from differing survey years and types.",
            "GDP and population may come from later years than poverty.",
            "This is not a wealth-concentration or firm-scaling variable.",
            "OLS and HC3 statistical inference cannot identify causality.",
        ],
    }
    (output / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    """Parse fully explicit source/output paths and print coverage only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshots", type=Path, required=True,
        help="Folder containing the OWID/WDI source snapshot and manifest",
    )
    parser.add_argument(
        "--forbes", type=Path,
        default=Path("data/forbes_2026_citizenship_secondary.csv"),
        help="Secondary Forbes citizenship transcription, not residence counts",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Fresh folder to hold exploratory estimates and exclusion ledger",
    )
    args = parser.parse_args()
    report = run_study(args.snapshots, args.forbes, args.output)
    print(
        f"Exploratory citizenship comparison: "
        f"{report['n_matched_countries']} matched countries; "
        f"{report['n_unmatched_secondary_countries']} excluded. "
        f"See {args.output / 'analysis.json'}."
    )


if __name__ == "__main__":
    main()
