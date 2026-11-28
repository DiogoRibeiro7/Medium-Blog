"""Offline accounting analysis of Eurostat government spending and production."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

COUNTRIES = set("AT BE BG HR CY CZ DK EE FI FR DE EL HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE".split())
YEARS = range(1995, 2026)
FISCAL = {"TE", "B1G", "P3", "D62PAY", "D632PAY", "D1PAY", "P2", "P51G", "D41PAY", "TR", "B9"}
SERIES = sorted(FISCAL | {"growth", "nominal_gdp"})
SPEC = {
    "fiscal.json": ("gov_10a_main", "PC_GDP", FISCAL),
    "growth.json": ("nama_10_gdp", "CLV_PCH_PRE", {"B1GQ"}),
    "nominal_gdp.json": ("nama_10_gdp", "CP_MEUR", {"B1GQ"}),
}
Row = dict[str, Any]


def digest(raw: bytes) -> str:
    """Return a SHA-256 digest for exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def read_snapshot(directory: Path, manifest_sha: str) -> dict[str, Any]:
    """Fail closed if the independently supplied manifest or source bytes differ."""
    raw_manifest = (directory / "manifest.json").read_bytes()
    if digest(raw_manifest) != manifest_sha.lower():
        raise ValueError("Manifest SHA-256 mismatch")
    manifest = json.loads(raw_manifest)
    entries = manifest["files"]
    if len(entries) != len(SPEC) or {e["path"] for e in entries} != set(SPEC):
        raise ValueError("Unexpected or duplicate source files")
    payloads = {}
    for entry in entries:
        name = entry["path"]
        raw = (directory / name).read_bytes()
        if len(raw) != entry["bytes"] or digest(raw) != entry["sha256"]:
            raise ValueError(f"Integrity failure: {name}")
        dataset, unit, items = SPEC[name]
        if entry["dataset"] != dataset:
            raise ValueError(f"Wrong source dataset: {name}")
        payload = json.loads(raw)
        expected = {"freq": {"A"}, "unit": {unit}, "na_item": items, "geo": COUNTRIES}
        if name == "fiscal.json":
            expected["sector"] = {"S13"}
        for dim, codes in expected.items():
            if set(payload["dimension"][dim]["category"]["index"]) != codes:
                raise ValueError(f"Wrong dimension {dim} in {name}")
        payloads[name] = payload
    return payloads


def cells(payload: dict[str, Any]) -> list[tuple[dict[str, str], float | None, str]]:
    """Expand JSON-stat in its declared dimension order, retaining missing cells and flags."""
    if payload.get("class") != "dataset":
        raise ValueError("Expected a JSON-stat dataset")
    ids, sizes = payload["id"], payload["size"]
    if len(ids) != len(sizes) or len(set(ids)) != len(ids):
        raise ValueError("Invalid dimensions")
    categories: list[list[str]] = []
    for dim, size in zip(ids, sizes):
        idx = payload["dimension"][dim]["category"]["index"]
        ordered = sorted(idx, key=idx.get) if isinstance(idx, dict) else list(idx)
        if len(ordered) != size or (isinstance(idx, dict) and set(idx.values()) != set(range(size))):
            raise ValueError(f"Invalid positions for {dim}")
        categories.append(ordered)
    values, status = payload["value"], payload.get("status", {})
    count = math.prod(sizes)
    for field in (values, status):
        if isinstance(field, list) and len(field) != count:
            raise ValueError("Dense JSON-stat array length mismatch")
        if isinstance(field, dict) and any(not k.isdigit() or int(k) >= count for k in field):
            raise ValueError("Invalid sparse JSON-stat index")

    def get(array: Any, position: int) -> Any:
        if isinstance(array, dict):
            return array.get(str(position))
        if isinstance(array, list):
            return array[position]
        if isinstance(array, str):
            return array
        return None

    result = []
    for position, coordinates in enumerate(itertools.product(*categories)):
        value = get(values, position)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("Non-numeric or non-finite observation")
        flag = get(status, position)
        result.append((dict(zip(ids, coordinates)), None if value is None else float(value), str(flag or "")))
    return result


def build_panel(payloads: dict[str, Any]) -> list[Row]:
    """Join explicitly on country and year; leave absent observations missing."""
    rows: dict[tuple[str, int], Row] = {
        (geo, year): {"geo": geo, "year": year, **{s: None for s in SERIES},
                      **{s + "_flag": "" for s in SERIES}}
        for geo in sorted(COUNTRIES) for year in YEARS
    }
    seen: set[tuple[str, int, str]] = set()
    for name, payload in payloads.items():
        for coords, value, flag in cells(payload):
            geo, year = coords["geo"], int(coords["time"])
            key = coords["na_item"] if name == "fiscal.json" else name.removesuffix(".json")
            identity = (geo, year, key)
            if identity in seen or (geo, year) not in rows or key not in SERIES:
                raise ValueError(f"Unexpected or duplicate observation {identity}")
            seen.add(identity)
            rows[(geo, year)][key] = value
            rows[(geo, year)][key + "_flag"] = flag
    for row in rows.values():
        te, b1g = row["TE"], row["B1G"]
        row["spending_minus_s13_gva_pp"] = te - b1g if te is not None and b1g is not None else None
        row["arithmetic_complement_of_spending_pct"] = 100 - te if te is not None else None
        row["non_s13_gva_plus_net_product_taxes_pct"] = 100 - b1g if b1g is not None else None
        row["benefits_and_purchased_in_kind_pct_gdp"] = (
            row["D62PAY"] + row["D632PAY"] if row["D62PAY"] is not None and row["D632PAY"] is not None else None
        )
    return list(rows.values())


def common_year(rows: list[Row]) -> int:
    """Return the latest complete cross-section for the two central measures."""
    for year in reversed(YEARS):
        eligible = {r["geo"] for r in rows if r["year"] == year and r["TE"] is not None and r["B1G"] is not None}
        if eligible == COUNTRIES:
            return year
    raise ValueError("No complete EU27 TE/B1G cross-section")


def decompose(e0: float, y0: float, e1: float, y1: float) -> tuple[float, float]:
    """Symmetric additive decomposition of change in 100 * expenditure / GDP."""
    if y0 <= 0 or y1 <= 0:
        raise ValueError("GDP must be positive")
    numerator = 100 * (e1 - e0) * (1 / y0 + 1 / y1) / 2
    denominator = 100 * (e0 + e1) * (1 / y1 - 1 / y0) / 2
    return numerator, denominator


def denominator_rows(rows: list[Row]) -> list[Row]:
    """Reconstruct spending from source ratios; compare adjacent complete years only."""
    indexed = {(r["geo"], r["year"]): r for r in rows}
    output = []
    for row in rows:
        prior = indexed.get((row["geo"], row["year"] - 1))
        if prior is None or any(r[k] is None for r in (prior, row) for k in ("TE", "nominal_gdp")):
            continue
        y0, y1 = prior["nominal_gdp"], row["nominal_gdp"]
        n, d = decompose(prior["TE"] * y0 / 100, y0, row["TE"] * y1 / 100, y1)
        change = row["TE"] - prior["TE"]
        if not math.isclose(n + d, change, abs_tol=1e-8):
            raise ArithmeticError("Decomposition failed closure")
        output.append({"geo": row["geo"], "year": row["year"], "ratio_change_pp": change,
                       "nominal_spending_contribution_pp": n, "nominal_gdp_contribution_pp": d})
    return output


def correlation(x: list[float], y: list[float]) -> float | None:
    """Pearson correlation, undefined for constant or insufficient samples."""
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return statistics.correlation(x, y)


def associations(rows: list[Row]) -> list[Row]:
    """Descriptive correlations on matched complete observations; never causal."""
    output = []
    indexed = {(r["geo"], r["year"]): r for r in rows}
    for horizon in (0, 1):
        for exclude_flags in (False, True):
            grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for row in rows:
                future = indexed.get((row["geo"], row["year"] + horizon))
                if future is None or row["TE"] is None or future["growth"] is None:
                    continue
                if exclude_flags and (row["TE_flag"] or future["growth_flag"]):
                    continue
                grouped[row["geo"]].append((row["TE"], future["growth"]))
            grouped = {k: v for k, v in grouped.items() if len(v) >= 2}
            pooled = [p for group in grouped.values() for p in group]
            centered = []
            for group in grouped.values():
                mx, my = statistics.mean(p[0] for p in group), statistics.mean(p[1] for p in group)
                centered.extend((x - mx, y - my) for x, y in group)
            for method, points in (("pooled", pooled), ("within_country", centered)):
                output.append({"growth_horizon_years": horizon, "exclude_flagged": exclude_flags,
                               "method": method, "n": len(points), "countries": len(grouped),
                               "pearson_r": correlation([p[0] for p in points], [p[1] for p in points])})
    return output


def write_csv(path: Path, rows: list[Row], fieldnames: list[str] | None = None) -> None:
    """Write stable UTF-8 CSV without imputing missing data."""
    if not rows and fieldnames is None:
        raise ValueError(f"No output rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(output: Path, comparison: list[Row], france: list[Row], decomposition: list[Row], stats: list[Row]) -> None:
    """Generate transparent, separately labelled publication figure candidates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest = output / "figures"
    dest.mkdir()
    fig, ax = plt.subplots(figsize=(11, 6))
    ordered = sorted(comparison, key=lambda r: r["TE"])
    x = list(range(len(ordered)))
    ax.bar([i - .2 for i in x], [r["TE"] for r in ordered], width=.4, label="Total government expenditure")
    ax.bar([i + .2 for i in x], [r["B1G"] for r in ordered], width=.4, label="General government gross value added")
    ax.set_xticks(x, [r["geo"] for r in ordered], rotation=45)
    ax.set_ylabel("% of GDP")
    ax.set_title(f"EU27: expenditure and production are distinct ({ordered[0]['year']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label in (("TE", "Total expenditure"), ("B1G", "S13 gross value added"),
                       ("P3", "Government final consumption"), ("D62PAY", "Cash benefits"),
                       ("D632PAY", "Purchased transfers in kind")):
        ax.plot([r["year"] for r in france], [r[key] for r in france], label=label)
    ax.set_ylabel("% of GDP")
    ax.set_title("France: fiscal expenditure is not government value added")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "france.png", dpi=180)
    plt.close(fig)

    fr = [r for r in decomposition if r["geo"] == "FR"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = [r["year"] for r in fr]
    ax.bar([y - .2 for y in x], [r["nominal_spending_contribution_pp"] for r in fr], .4, label="Nominal spending")
    ax.bar([y + .2 for y in x], [r["nominal_gdp_contribution_pp"] for r in fr], .4, label="Nominal GDP")
    ax.plot(x, [r["ratio_change_pp"] for r in fr], color="black", marker=".", label="Total ratio change")
    ax.axhline(0, color="grey", linewidth=.7)
    ax.set_ylabel("Percentage points")
    ax.set_title("France: symmetric accounting decomposition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "denominator.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, method in enumerate(("pooled", "within_country")):
        subset = [r for r in stats if r["method"] == method and not r["exclude_flagged"] and r["pearson_r"] is not None]
        ax.bar([r["growth_horizon_years"] + (i - .5) * .35 for r in subset],
               [r["pearson_r"] for r in subset], .35, label=method)
    ax.axhline(0, color="grey", linewidth=.7)
    ax.set_xticks([0, 1], ["Same year", "Following year"])
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Spending ratio and real GDP growth: descriptive, not causal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "associations.png", dpi=180)
    plt.close(fig)


def run(directory: Path, output: Path, manifest_sha: str) -> None:
    """Verify inputs, build outputs, and record the precise run identity."""
    panel = build_panel(read_snapshot(directory, manifest_sha))
    year = common_year(panel)
    comparison = [r for r in panel if r["year"] == year]
    france = [r for r in panel if r["geo"] == "FR"]
    decomposition, stats = denominator_rows(panel), associations(panel)
    coverage = [{"series": key, "non_null": sum(r[key] is not None for r in panel),
                 "missing": sum(r[key] is None for r in panel),
                 "flagged": sum(bool(r[key + "_flag"]) for r in panel)} for key in SERIES]
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("panel", panel), ("coverage", coverage), ("comparison", comparison),
                       ("france", france), ("denominator", decomposition), ("growth_associations", stats)):
        write_csv(output / (name + ".csv"), data)
    figures(output, comparison, france, decomposition, stats)
    (output / "run_metadata.json").write_text(json.dumps({
        "manifest_sha256": manifest_sha.lower(), "comparison_year": year, "panel_rows": len(panel),
        "interpretation": "Descriptive accounting; not causal. S13 is not all public ownership.",
        "flags": "Preserved in panel; association sensitivity excludes all flagged pairs."
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Command-line entry point; no implicit network downloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    run(args.input, args.output, args.manifest_sha256)


if __name__ == "__main__":
    main()
