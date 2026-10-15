"""Pre-specified economic-regime sensitivity of descriptive Okun coefficients."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eu_okun.audit import audit_snapshot
from eu_okun.baseline import _read_panel
from eu_okun.comparison import ComparisonRow, fit_comparisons
from eu_okun.contract import EU27, EurostatDataError, quarter_sequence

# A quarter is eligible only if every dependent variable and all lagged inputs
# come from the stated regime; the first four quarters of each interval are
# reserved for GDP-lag warm-up (unemployment differences need one quarter).
WINDOWS: dict[str, tuple[str, str]] = {
    "pre_pandemic": ("2009-Q1", "2019-Q4"),
    "post_2021": ("2022-Q1", "2026-Q2"),
}
PANDEMIC_START = "2020-Q1"
PANDEMIC_END = "2021-Q4"
FOCUS = ("PT", "ES", "NL", "LU")


@dataclass(frozen=True)
class RegimeRow:
    """One coefficient-sum estimate for a named prespecified sample."""

    regime: str
    geo: str
    gdp_lags: int
    controls_participation: bool
    n: int
    sample_start: str
    sample_end: str
    gdp_sum: float
    gdp_sum_hac_se: float
    ci95_low: float
    ci95_high: float
    r_squared: float


def regime_comparisons(
    panel: dict[tuple[str, str], dict[str, float | None]],
    end_quarter: str,
) -> tuple[list[RegimeRow], dict[str, dict[str, Any]]]:
    """Fit each regime on a common EU-27 sample or record non-estimability.

    A post-2021 regression starts no earlier than 2022-Q2; GDP lag
    history cannot cross the regime boundary. The pandemic-excluded
    model removes observations in 2020–2021 as well as any outcome
    that needs those quarters as predictors or for differences.
    """
    start = "2009-Q1"
    full = quarter_sequence(start, end_quarter)
    configurations: dict[str, tuple[list[str], tuple[int, ...], int]] = {
        "full": (full, (0, 1, 2), 30),
        "pre_pandemic": (
            [q for q in full if WINDOWS["pre_pandemic"][0] <= q <= WINDOWS["pre_pandemic"][1]],
            (0, 1, 2), 25,
        ),
        "pandemic_excluded": (
            [q for q in full if q < PANDEMIC_START or q > PANDEMIC_END],
            (0, 1, 2), 25,
        ),
        "post_2021": (
            [q for q in full if q >= WINDOWS["post_2021"][0]],
            (0, 1), 12,
        ),
    }

    output: list[RegimeRow] = []
    metadata: dict[str, dict[str, Any]] = {}
    for name, (eligible, lags, minimum) in configurations.items():
        # Limit the predictor history to within a contiguous regime. For
        # pandemic exclusion, allow pre-2020 and post-2021 history while
        # removing the excluded quarters from the input panel itself.
        if name == "pre_pandemic":
            allowed = set(q for q in quarter_sequence("2008-Q1", "2019-Q4"))
        elif name == "post_2021":
            allowed = set(eligible)
        elif name == "pandemic_excluded":
            allowed = {
                q for q in quarter_sequence("2008-Q1", end_quarter)
                if not PANDEMIC_START <= q <= PANDEMIC_END
            }
        else:
            allowed = set(quarter_sequence("2008-Q1", end_quarter))
        restricted = {
            key: values for key, values in panel.items() if key[1] in allowed
        }
        if not eligible:
            metadata[name] = {"status": "not_estimable", "reason": "Empty regime"}
            continue
        try:
            rows, dates = fit_comparisons(
                restricted, eligible, countries=EU27, lag_orders=lags,
                hac_bandwidth=4, min_quarters=minimum,
            )
        except (EurostatDataError, ValueError) as exc:
            metadata[name] = {"status": "not_estimable", "reason": str(exc)}
            continue
        metadata[name] = {
            "status": "estimated", "n_quarters": len(dates),
            "first": dates[0], "last": dates[-1], "lag_orders": list(lags),
        }
        for row in rows:
            output.append(_to_regime_row(name, row))
    return output, metadata


def _to_regime_row(regime: str, result: ComparisonRow) -> RegimeRow:
    """Keep essential sensitivity information and numerical uncertainty."""
    return RegimeRow(
        regime, result.geo, result.gdp_lags, result.controls_participation,
        result.n, result.sample_start, result.sample_end,
        result.gdp_sum, result.gdp_sum_hac_se, result.ci95_low,
        result.ci95_high, result.r_squared,
    )


def write_regimes(directory: Path, output_dir: Path) -> int:
    """Validate Eurostat provenance and write reproducible regime estimates."""
    audit = audit_snapshot(directory)
    end = audit["analysis_period"]["end"]
    if not isinstance(end, str):
        raise EurostatDataError("Invalid final quarter.")
    panel = _read_panel(directory / "processed" / "eu27_quarterly.csv")
    rows, status = regime_comparisons(panel, end)
    if not rows:
        raise EurostatDataError("No estimable regimes; inspect data and definitions.")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "scope": "Descriptive OLS/HAC robustness; not a structural shock analysis",
        "snapshot_manifest_sha256": hashlib.sha256(
            (directory / "snapshot_manifest.json").read_bytes()
        ).hexdigest(),
        "windows": status,
        "models": [asdict(row) for row in rows],
    }
    (output_dir / "regimes.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "regimes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    report = [
        "# Okun-law sensitivity to macroeconomic regimes", "",
        "These are conditional statistical associations, not causal effects.", "",
        "| Regime | Status | EU-wide common quarters |",
        "| --- | --- | ---: |",
    ]
    for name, info in status.items():
        report.append(
            f"| {name} | {info['status']} | {info.get('n_quarters', 'n/a')} |"
        )
    report += [
        "", "## Four pre-specified focus countries: K=1 without participation",
        "", "| Regime | Country | Coefficient sum | 95% HAC interval | n |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name in status:
        for geo in FOCUS:
            matches = [
                r for r in rows if r.regime == name and r.geo == geo
                and r.gdp_lags == 1 and not r.controls_participation
            ]
            if matches:
                r = matches[0]
                report.append(
                    f"| {name} | {geo} | {r.gdp_sum:+.3f} | "
                    f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}] | {r.n} |"
                )
            else:
                report.append(f"| {name} | {geo} | n/a | not estimable | n/a |")
    report.extend([
        "",
        "The estimates for different regimes need not use identical periods;",
        "comparisons therefore describe sample sensitivity, not an identified",
        "structural break or a formal test of coefficient differences.",
        "The post-2021 sample is short and uncertainty should be interpreted",
        "accordingly. Country-level confidence intervals are not joint",
        "cross-country tests. No causal GDP shock is identified.",
        "",
    ])
    (output_dir / "REGIMES.md").write_text("\n".join(report), encoding="utf-8")
    return len(rows)
