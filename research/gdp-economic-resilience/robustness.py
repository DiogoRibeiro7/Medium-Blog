"""Sensitivity analysis for GDP, wealth, historical losses and sector concentration.

All statistics are descriptive and use explicitly matched country samples.
No p-values, independent-country sampling assumption or causal estimates.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

import eurostat_sectors as euro
import historical_shocks as shocks
import pipeline as p

Value = float | None
Record = dict[str, Any]
Pair = tuple[str, float, float]
YEARS = (2000, 2010, 2019, 2021, 2024)
FEATURES = ("agriculture_pct_gdp", "industry_pct_gdp", "services_pct_gdp",
            "manufacturing_pct_gdp", "three_sector_hhi")


def finite(value: Value) -> bool:
    """Identify available numeric observations without zero imputation."""
    return value is not None and math.isfinite(value)


def sign(value: float) -> int:
    """Distinguish positive, negative and exact zero changes."""
    return (value > 0) - (value < 0)


def midranks(values: list[float]) -> list[float]:
    """Average the ranks of ties, using ascending one-based ranks."""
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for index in ordered[start:end]:
            ranks[index] = average
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> Value:
    """Spearman correlation with tie handling and undefined constant samples."""
    if len(x) != len(y) or len(x) < 3:
        return None
    return shocks.pearson(midranks(x), midranks(y))


def paired_statistics(pairs: list[Pair]) -> Record:
    """Summarize a matched sample; leave-one-out is influence, not uncertainty."""
    pairs = sorted(pairs)
    if len({row[0] for row in pairs}) != len(pairs):
        raise ValueError("Duplicate country in paired statistics")
    x, y = [row[1] for row in pairs], [row[2] for row in pairs]
    if not all(math.isfinite(v) for v in [*x, *y]):
        raise ValueError("Paired statistics require finite values")
    linear = shocks.pearson(x, y)
    rank = spearman(x, y)
    loo = []
    for index, (country, _, _) in enumerate(pairs):
        omitted = shocks.pearson(x[:index] + x[index + 1:], y[:index] + y[index + 1:])
        if omitted is not None:
            loo.append((omitted, country))
    low, high = min(loo, default=(None, None)), max(loo, default=(None, None))
    return {
        "n": len(pairs), "countries": ";".join(row[0] for row in pairs),
        "pearson_r": linear, "spearman_r": rank,
        "loo_pearson_min": low[0], "loo_min_omits": low[1],
        "loo_pearson_max": high[0], "loo_max_omits": high[1],
        "loo_defined": len(loo),
        "loo_sign_reversal": None if linear is None or not loo else
        any(sign(value) * sign(linear) < 0 for value, _ in loo),
    }


def annual_log_growth(change: Value, years: int) -> Value:
    """Convert endpoint percent growth into annualized log percentage points."""
    if years <= 0:
        raise ValueError("Window length must be positive")
    if not finite(change) or change <= -100:
        return None
    return 100 * math.log1p(change / 100) / years


def wealth_sensitivity(wdi: p.Snapshot, cwon: p.Snapshot) -> tuple[list[Record], list[Record]]:
    """Compare windows with both available and fixed-common country samples."""
    rows = p.wealth_rows(wdi, cwon)
    windows = sorted({(row["start"], row["end"]) for row in rows})
    valid: dict[tuple[int, int], dict[str, tuple[float, float]]] = {}
    for start, end in windows:
        panel = {}
        for row in rows:
            if (row["start"], row["end"]) != (start, end):
                continue
            g = annual_log_growth(row["real_gdp_pc_growth_pct"], end - start)
            w = annual_log_growth(row["real_wealth_pc_growth_pct"], end - start)
            if g is not None and w is not None:
                panel[row["country"]] = (g, w)
        valid[start, end] = panel
    common = set.intersection(*(set(valid[window]) for window in windows))
    result = []
    for start, end in windows:
        panel = valid[start, end]
        samples = {
            "available": set(panel),
            "exclude_bulgaria": set(panel) - {"BGR"},
            "common_all_windows": common,
        }
        for sample, countries in samples.items():
            pairs = [(country, *panel[country]) for country in sorted(countries)]
            gaps = [g - w for _, g, w in pairs]
            divergence = [country for country, g, w in pairs if g > 0 and w < 0]
            result.append({
                "start": start, "end": end, "sample": sample,
                "excluded_from_requested": len(p.COUNTRIES) - len(pairs),
                "mean_annual_log_gap_pp": statistics.mean(gaps) if gaps else None,
                "median_annual_log_gap_pp": statistics.median(gaps) if gaps else None,
                "gdp_up_wealth_down_count": len(divergence),
                "gdp_up_wealth_down_countries": ";".join(divergence),
                **paired_statistics(pairs),
            })
    stability = []
    for country in p.COUNTRIES:
        available = [window for window in windows if country in valid[window]]
        divergent = [window for window in available
                     if valid[window][country][0] > 0 and valid[window][country][1] < 0]
        stability.append({
            "country": country, "available_windows": len(available),
            "requested_windows": len(windows), "divergent_windows": len(divergent),
            "divergent_window_ids": ";".join(f"{a}-{b}" for a, b in divergent),
            "divergent_in_every_requested_window": len(divergent) == len(windows),
            "divergent_in_every_available_window": bool(available) and len(divergent) == len(available),
            "bulgaria_1996_baseline_flag": country == "BGR",
        })
    return result, stability


def benchmark_panel(metrics: list[Record], episode: str, metric: str) -> dict[str, dict[str, float]]:
    """Index finite shortfalls by benchmark and country; reject duplicate records."""
    result: dict[str, dict[str, float]] = {model: {} for model in shocks.MODELS}
    seen = set()
    for row in metrics:
        if row["shock"] != episode or row["metric"] != metric:
            continue
        key = (row["benchmark"], row["country"])
        if key in seen:
            raise ValueError("Duplicate country/benchmark shortfall")
        seen.add(key)
        loss = row["cumulative_shortfall_pct_benchmark"]
        if finite(loss) and row["complete_common_window"]:
            result[key[0]][key[1]] = loss
    return result


def benchmark_sensitivity(metrics: list[Record]) -> list[Record]:
    """Compare alternative loss estimates on countries observed for all models."""
    result = []
    for episode in shocks.SHOCKS:
        for metric in shocks.METRICS:
            panel = benchmark_panel(metrics, episode.name, metric)
            common = sorted(set.intersection(*(set(v) for v in panel.values())))
            for first, second in itertools.combinations(shocks.MODELS, 2):
                pairs = [(country, panel[first][country], panel[second][country])
                         for country in common]
                differences = [abs(a - b) for _, a, b in pairs]
                flips = [country for country, a, b in pairs if sign(a) * sign(b) < 0]
                result.append({
                    "shock": episode.name, "metric": metric, "first_benchmark": first,
                    "second_benchmark": second, "sample": "common_all_benchmarks",
                    "median_absolute_difference_pp": statistics.median(differences) if differences else None,
                    "strict_sign_reversal_count": len(flips),
                    "strict_sign_reversal_countries": ";".join(flips),
                    **paired_statistics(pairs),
                })
    return result


def structure_sensitivity(wdi: p.Snapshot, metrics: list[Record]) -> list[Record]:
    """Hold each feature's sample fixed across benchmarks; compare EU-only too."""
    original = shocks.association_rows(wdi, metrics)
    covariates = {(row["shock"], row["country"]): row for row in original}
    results = []
    for episode in shocks.SHOCKS:
        panel = benchmark_panel(metrics, episode.name, "real_gdp_per_capita")
        common = set.intersection(*(set(values) for values in panel.values()))
        for feature in FEATURES:
            available = {country for country in common
                         if (episode.name, country) in covariates
                         and finite(covariates[episode.name, country][feature])}
            for sample, universe in (("all_countries", set(p.COUNTRIES)), ("eu27_members", set(p.EU27))):
                included = sorted(available & universe)
                for model in shocks.MODELS:
                    pairs = [(country, covariates[episode.name, country][feature], panel[model][country])
                             for country in included]
                    results.append({
                        "shock": episode.name, "structure_feature": feature, "benchmark": model,
                        "sample": sample, "excluded_from_universe": len(universe) - len(pairs),
                        **paired_statistics(pairs),
                    })
    return results


def granularity_sensitivity(wdi: p.Snapshot, a10: dict[euro.CellKey, euro.Cell],
                            a64: dict[euro.CellKey, euro.Cell]) -> list[Record]:
    """Check rankings across classifications; never equate HHI levels."""
    results = []
    wdi_rows = p.sector_rows(wdi)
    broad = euro.concentration_rows("a10", a10)
    detailed = euro.concentration_rows("a64", a64)
    by_key = {(row["geo"], row["year"]): row["hhi"] for row in detailed}
    for year in YEARS:
        wp = [(row["country"], row["hhi_three_broad_sectors"], row["hhi_four_sectors"])
              for row in wdi_rows if row["year"] == year
              and finite(row["hhi_three_broad_sectors"]) and finite(row["hhi_four_sectors"])]
        ep = [(row["geo"], row["hhi"], by_key.get((row["geo"], year)))
              for row in broad if row["year"] == year and row["geo"] != "EU27_2020"
              and finite(row["hhi"]) and finite(by_key.get((row["geo"], year)))]
        for dataset, first, second, pairs, requested in (
            ("wdi", "three_broad", "four_manufacturing_split", wp, len(p.COUNTRIES)),
            ("eurostat", "a10", "a64_a_to_t", ep, len(euro.GEO) - 1),
        ):
            results.append({
                "dataset": dataset, "year": year, "first_classification": first,
                "second_classification": second, "excluded": requested - len(pairs),
                **paired_statistics(pairs),
            })
    return results


def run(data_dir: Path, output: Path, fetch: bool = False) -> None:
    """Reproduce robustness checks using the existing immutable inputs."""
    wdi = p.decode_rows(p.verified_files(p.WDI, data_dir, fetch), p.WDI)
    cwon = p.decode_rows(p.verified_files(p.CWON, data_dir, fetch), p.CWON)
    a10 = euro.decode(euro.load_input("a10", 0, data_dir, fetch), "B1G", ("CP_MEUR",))
    a64 = euro.decode(euro.load_input("a64", 0, data_dir, fetch), "B1G", ("CP_MEUR",))
    _, metrics = shocks.wdi_paths(wdi)
    wealth, stability = wealth_sensitivity(wdi, cwon)
    tables = {
        "wealth_window_sensitivity.csv": wealth,
        "wealth_country_stability.csv": stability,
        "shock_benchmark_sensitivity.csv": benchmark_sensitivity(metrics),
        "structure_correlation_sensitivity.csv": structure_sensitivity(wdi, metrics),
        "classification_sensitivity.csv": granularity_sensitivity(wdi, a10, a64),
    }
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in tables.items():
        p.write_csv(output / filename, rows)
    summary = {
        "registry_commits": {"world_bank": p.COMMIT, "eurostat_a10": euro.SOURCES["a10"][1],
                             "eurostat_a64": euro.SOURCES["a64"][1]},
        "tables": {name: len(rows) for name, rows in tables.items()},
        "wealth_divergent_in_every_requested_window": [
            row["country"] for row in stability if row["divergent_in_every_requested_window"]],
        "common_wealth_sample_sizes": sorted({row["n"] for row in wealth if row["sample"] == "common_all_windows"}),
        "scope": "Matched-sample descriptive sensitivity; leave-one-out ranges are not confidence intervals.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    """Run with an optional pinned-source fetch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=p.HERE / "data")
    parser.add_argument("--output", type=Path, default=p.HERE / "results" / "robustness")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    run(args.data_dir, args.output, args.fetch)


if __name__ == "__main__":
    main()
