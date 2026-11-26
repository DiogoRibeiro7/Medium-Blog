"""Descriptive shock losses and recovery from pinned WDI and Eurostat data.

All projected paths are mechanical benchmarks, not identified counterfactuals.
Missing observations never become zero or silently shorten a comparison window.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import eurostat_sectors as euro
import pipeline as p

Value = float | None
Series = dict[int, Value]
Record = dict[str, Any]


@dataclass(frozen=True)
class Shock:
    """An annual onset, trend fitting endpoint and common comparison horizon."""

    name: str
    onset: int
    fit_end: int

    @property
    def anchor(self) -> int:
        """Year immediately preceding the specified shock."""
        return self.onset - 1

    @property
    def end(self) -> int:
        """Equal three-year comparison window for all three shocks."""
        return self.onset + 2


SHOCKS = (
    Shock("financial_crisis", 2008, 2007),
    Shock("pandemic", 2020, 2019),
    Shock("energy_price_shock", 2022, 2019),
)
MODELS = ("trend_5y", "trend_10y", "flat")
LAST_YEAR = 2025
METRICS = ("real_gdp", "real_gdp_per_capita")


def positive(value: Value) -> bool:
    """Exclude nonpositive or nonfinite observations from log calculations."""
    return value is not None and math.isfinite(value) and value > 0


def trend_slope(series: Series, end: int, years: int) -> Value:
    """OLS slope of log levels over exactly the requested annual observations."""
    x = list(range(end - years + 1, end + 1))
    observed = [series.get(year) for year in x]
    if years < 2 or not all(positive(v) for v in observed):
        return None
    logs = [math.log(v) for v in observed if v is not None]
    mean_x = math.fsum(x) / years
    mean_y = math.fsum(logs) / years
    return math.fsum((a - mean_x) * (b - mean_y)
                     for a, b in zip(x, logs, strict=True)) / math.fsum(
                         (a - mean_x) ** 2 for a in x)


def baseline(series: Series, shock: Shock, model: str) -> tuple[Value, Value]:
    """Anchor at last pre-onset level; estimate slope from uncontaminated years.

    For 2022 use trends ending 2019 but anchor at 2021. This deliberately
    conditions on the pandemic-era level rather than attributing it to energy.
    """
    anchor = series.get(shock.anchor)
    if not positive(anchor):
        return None, None
    if model == "flat":
        return anchor, 0.0
    if model not in MODELS:
        raise ValueError(f"Unknown benchmark: {model}")
    slope = trend_slope(series, shock.fit_end, 5 if model == "trend_5y" else 10)
    return anchor, slope


def expected_value(anchor: Value, slope: Value, elapsed: int) -> Value:
    """Project the anchored exponential trend without using post-onset outcomes."""
    if anchor is None or slope is None:
        return None
    expected = anchor * math.exp(slope * elapsed)
    return expected if positive(expected) else None


def recovery(series: Series, shock: Shock, anchor: Value, last: int) -> Record:
    """Identify the first return to the anchor after an observed drop.

    A missing year before a return makes the event timing unidentified.
    Only drops within the fixed comparison window initiate a recovery event.
    Later crises must not create a new event for an earlier episode.
    """
    result: Record = {"level_recovery_status": "missing_anchor",
                      "first_drop_year": None, "level_recovery_year": None,
                      "years_from_onset_to_recovery": None}
    if not positive(anchor):
        return result
    drop = None
    for year in range(shock.onset, last + 1):
        if drop is None and year > shock.end:
            result["level_recovery_status"] = "no_drop"
            return result
        value = series.get(year)
        if not positive(value):
            result.update(level_recovery_status="missing_followup", first_drop_year=drop)
            return result
        if value < anchor and drop is None:
            drop = year
        if drop is not None and value >= anchor:
            result.update(level_recovery_status="recovered", first_drop_year=drop,
                          level_recovery_year=year,
                          years_from_onset_to_recovery=year - shock.onset)
            return result
    result.update(level_recovery_status="right_censored" if drop is not None else "no_drop",
                  first_drop_year=drop)
    return result


def measure(series: Series, shock: Shock, model: str,
            last: int = LAST_YEAR) -> tuple[list[Record], Record]:
    """Return annual paths plus comparable-window metrics and recovery status."""
    if last < shock.end:
        raise ValueError("Follow-up must include the fixed three-year window")
    anchor, slope = baseline(series, shock, model)
    paths = []
    for year in range(shock.onset, last + 1):
        actual = series.get(year)
        expected = expected_value(anchor, slope, year - shock.anchor)
        valid = positive(actual) and positive(expected)
        paths.append({
            "year": year, "years_since_onset": year - shock.onset,
            "actual": actual, "benchmark": expected,
            "actual_index_anchor_100": 100 * actual / anchor if positive(actual) and positive(anchor) else None,
            "benchmark_index_anchor_100": 100 * expected / anchor if positive(expected) and positive(anchor) else None,
            "gap_pct_benchmark": 100 * (actual / expected - 1) if valid else None,
            "in_common_window": year <= shock.end,
        })
    window = [row for row in paths if row["in_common_window"]]
    usable = [row for row in window if row["gap_pct_benchmark"] is not None]
    complete = len(usable) == len(window)
    result: Record = {
        "onset": shock.onset, "anchor_year": shock.anchor, "fit_end": shock.fit_end,
        "benchmark": model, "comparison_end": shock.end, "followup_end": last,
        "anchor_level": anchor, "trend_annual_growth_pct": None if slope is None else 100 * math.expm1(slope),
        "observed_common_years": sum(positive(row["actual"]) for row in window),
        "comparable_common_years": len(usable), "complete_common_window": complete,
        "cumulative_shortfall_pct_benchmark": None, "onset_gap_pct_benchmark": None,
        "endpoint_gap_pct_benchmark": None, "worst_gap_pct_benchmark": None,
        "max_drop_from_anchor_pct": None,
    }
    if complete:
        actual_sum = math.fsum(row["actual"] for row in window)
        expected_sum = math.fsum(row["benchmark"] for row in window)
        # Signed, including positive surprises: negative shortfall = outperformance.
        result.update(cumulative_shortfall_pct_benchmark=100 * (expected_sum - actual_sum) / expected_sum,
                      onset_gap_pct_benchmark=window[0]["gap_pct_benchmark"],
                      endpoint_gap_pct_benchmark=window[-1]["gap_pct_benchmark"],
                      worst_gap_pct_benchmark=min(row["gap_pct_benchmark"] for row in window))
    if positive(anchor) and all(positive(row["actual"]) for row in window):
        result["max_drop_from_anchor_pct"] = max(
            0.0, 100 * (1 - min(row["actual"] for row in window) / anchor))
    result.update(recovery(series, shock, anchor, last))
    return paths, result


def wdi_paths(snapshot: p.Snapshot) -> tuple[list[Record], list[Record]]:
    """Produce country-level GDP and GDP per capita paths, without region overlaps."""
    paths, metrics = [], []
    for country in p.COUNTRIES:
        for metric in METRICS:
            series = {year: (snapshot.value(p.REAL, country, year) if metric == "real_gdp"
                             else p.ratio(snapshot.value(p.REAL, country, year),
                                          snapshot.value(p.POP, country, year)))
                      for year in range(1995, LAST_YEAR + 1)}
            for shock in SHOCKS:
                for model in MODELS:
                    rows, result = measure(series, shock, model)
                    identity = {"country": country, "metric": metric, "shock": shock.name}
                    paths.extend({**identity, "benchmark_model": model, **row} for row in rows)
                    metrics.append({**identity, **result})
    return paths, metrics


def eurostat_paths(gva: dict[euro.CellKey, euro.Cell]) -> tuple[list[Record], list[Record]]:
    """Use published A10 real GVA series, with manufacturing shown separately."""
    paths, metrics = [], []
    for geo in euro.GEO:
        for sector in ("TOTAL", *euro.A10, "C"):
            series = {year: euro.get(gva, geo, year, sector, "CLV20_MEUR").value
                      for year in range(1995, LAST_YEAR + 1)}
            for shock in SHOCKS:
                for model in MODELS:
                    rows, result = measure(series, shock, model)
                    identity = {"geo": geo, "sector": sector, "shock": shock.name}
                    paths.extend({**identity, "benchmark_model": model, **row,
                                  "source_flag": euro.get(gva, geo, row["year"], sector, "CLV20_MEUR").flag}
                                 for row in rows)
                    metrics.append({**identity, **result})
    return paths, metrics


def association_rows(snapshot: p.Snapshot, metrics: list[Record]) -> list[Record]:
    """Pair pre-onset structure with subsequent losses; report missing covariates."""
    results = []
    starts = {shock.name: shock.anchor for shock in SHOCKS}
    for row in metrics:
        if row["metric"] != "real_gdp_per_capita" or row["benchmark"] != "trend_5y":
            continue
        country, year = row["country"], starts[row["shock"]]
        agr, industry, services, manufacturing = (
            snapshot.value(code, country, year) for code in (p.AGR, p.IND, p.SRV, p.MAN))
        _, hhi, _ = p.concentration([agr, industry, services])
        results.append({
            "country": country, "shock": row["shock"], "structure_year": year,
            "agriculture_pct_gdp": agr, "industry_pct_gdp": industry,
            "services_pct_gdp": services, "manufacturing_pct_gdp": manufacturing,
            "three_sector_hhi": hhi,
            "cumulative_shortfall_pct_benchmark": row["cumulative_shortfall_pct_benchmark"],
            "complete_common_window": row["complete_common_window"],
        })
    return results


def pearson(x: list[float], y: list[float]) -> Value:
    """Compute descriptive sample correlation; undefined for constant inputs."""
    if len(x) != len(y) or len(x) < 3:
        return None
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    xx = math.fsum((a - mx) ** 2 for a in x)
    yy = math.fsum((b - my) ** 2 for b in y)
    if xx <= 0 or yy <= 0:
        return None
    return math.fsum((a - mx) * (b - my)
                     for a, b in zip(x, y, strict=True)) / math.sqrt(xx * yy)


def associations(rows: list[Record]) -> list[Record]:
    """Pairwise complete cases, per shock, never pool repeated countries."""
    result = []
    columns = ("agriculture_pct_gdp", "industry_pct_gdp", "services_pct_gdp",
               "manufacturing_pct_gdp", "three_sector_hhi")
    for shock in SHOCKS:
        sample = [row for row in rows if row["shock"] == shock.name]
        for column in columns:
            valid = [row for row in sample if row[column] is not None
                     and row["cumulative_shortfall_pct_benchmark"] is not None]
            result.append({
                "shock": shock.name, "structure_feature": column,
                "n": len(valid), "excluded": len(sample) - len(valid),
                "pearson_r": pearson([row[column] for row in valid],
                                     [row["cumulative_shortfall_pct_benchmark"] for row in valid]),
                "countries": ";".join(row["country"] for row in valid),
            })
    return result


def run(data_dir: Path, output: Path, fetch: bool = False) -> None:
    """Generate all tables from existing verified registry files."""
    wdi = p.decode_rows(p.verified_files(p.WDI, data_dir, fetch), p.WDI)
    gva = euro.decode(euro.load_input("a10", 0, data_dir, fetch), "B1G", ("CP_MEUR", "CLV20_MEUR"))
    paths, metrics = wdi_paths(wdi)
    ep, em = eurostat_paths(gva)
    pairs = association_rows(wdi, metrics)
    correlations = associations(pairs)
    tables = {"wdi_shock_paths.csv": paths, "wdi_shock_metrics.csv": metrics,
              "eurostat_shock_paths.csv": ep, "eurostat_shock_metrics.csv": em,
              "pre_shock_structure.csv": pairs, "structure_loss_correlations.csv": correlations}
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        p.write_csv(output / name, rows)
    summary = {
        "registry_commits": {"wdi": p.COMMIT, "eurostat_a10": euro.SOURCES["a10"][1]},
        "shocks": [{"name": shock.name, "onset": shock.onset, "fit_end": shock.fit_end,
                    "comparison_end": shock.end} for shock in SHOCKS],
        "benchmarks": list(MODELS), "followup_end": LAST_YEAR,
        "complete_wdi_windows": {
            shock.name: sum(row["shock"] == shock.name and row["metric"] == "real_gdp_per_capita"
                            and row["benchmark"] == "trend_5y" and row["complete_common_window"]
                            for row in metrics) for shock in SHOCKS},
        "tables": {name: len(rows) for name, rows in tables.items()},
        "scope": "Descriptive trend-relative deviations; no causal attribution or identified counterfactual.",
    }
    (output / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, indent=2))


def main() -> None:
    """Run with optional pinned-input retrieval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=p.HERE / "data")
    parser.add_argument("--output", type=Path, default=p.HERE / "results" / "shocks")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    run(args.data_dir, args.output, args.fetch)


if __name__ == "__main__":
    main()
