"""The EU as a whole: a labour-force-weighted aggregate (``AGGREGATE_PROTOCOL.md``).

The two-lag Okun sum of the EU from the official ``EU27_2020`` series, and the
labour-force-weighted mean of the country sums, against their unweighted mean. All
three share the synchronised block bootstrap over quarters of the case study, so their
differences have a joint distribution. Associations only.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.aggregate_data import EU, audit_eu_aggregate, load_eu_aggregate
from eu_okun.asymmetry import Series, load_series
from eu_okun.audit import audit_snapshot
from eu_okun.baseline import _read_panel
from eu_okun.contract import EU27
from eu_okun.heterogeneity import (
    ALPHA,
    FIRST,
    LAST,
    PRE_LAST,
    block_counts,
    holm,
    lag_rows,
    quarters_for,
    two_sided_p,
)

REPLICATES = 1999
SEED = 20261029
MAX_CONDITION = 1e8
MAX_DROPPED = 0.05


@dataclass(frozen=True)
class Settings:
    """Choices that the sensitivity analyses vary (protocol, section 7)."""

    key: str = "primary"
    lags: int = 2
    block: int = 4
    last: str = LAST


PRIMARY = Settings()
SENSITIVITY: tuple[Settings, ...] = (
    Settings("lags_1", lags=1), Settings("lags_4", lags=4), Settings("block_8", block=8),
    Settings("pre_2020", last=PRE_LAST))


# ----------------------------------------------------------------------------------
# Weights, the EU series and the data check
# ----------------------------------------------------------------------------------


def labour_force(aggregate: Mapping[tuple[str, str], Mapping[str, float | None]], geo: str,
                 quarter: str) -> float:
    cell = aggregate.get((geo, quarter), {})
    unemployed, rate = cell.get("unemployed_ths"), cell.get("rate")
    if unemployed is None or rate is None or rate <= 0:
        return math.nan
    return unemployed / (rate / 100.0)


def weights(aggregate: Mapping[tuple[str, str], Mapping[str, float | None]],
            quarters: Sequence[str]) -> dict[str, float]:
    """Each country's share of the EU labour force, averaged over FIRST..LAST."""
    span = quarters[quarters.index(FIRST): quarters.index(LAST) + 1]
    shares = []
    for q in span:
        lf = np.asarray([labour_force(aggregate, g, q) for g in EU27])
        if np.all(np.isfinite(lf)):
            shares.append(lf / lf.sum())
    mean = np.mean(np.vstack(shares), axis=0)
    return {g: float(w) for g, w in zip(EU27, mean, strict=True)}


def eu_series(aggregate: Mapping[tuple[str, str], Mapping[str, float | None]],
              quarters: Sequence[str]) -> Series:
    def get(name: str) -> np.ndarray:
        values = [aggregate.get((EU, q), {}).get(name) for q in quarters]
        return np.asarray([np.nan if v is None else float(v) for v in values])
    return Series(get("gdp"), np.diff(get("rate"), prepend=np.nan))


def data_check(aggregate: Mapping[tuple[str, str], Mapping[str, float | None]],
               quarters: Sequence[str]) -> dict[str, float]:
    """The labour-force-weighted rate of the countries against the official EU rate."""
    built, official = [], []
    for q in quarters:
        lf = np.asarray([labour_force(aggregate, g, q) for g in EU27])
        rates = np.asarray([aggregate.get((g, q), {}).get("rate") or math.nan for g in EU27])
        eu = aggregate.get((EU, q), {}).get("rate")
        if np.all(np.isfinite(lf)) and np.all(np.isfinite(rates)) and eu is not None:
            built.append(float(lf @ rates / lf.sum()))
            official.append(float(eu))
    b, o = np.asarray(built), np.asarray(official)
    return {"quarters": len(b), "max_abs_difference": float(np.max(np.abs(b - o))),
            "change_correlation": float(np.corrcoef(np.diff(b), np.diff(o))[0, 1])}


# ----------------------------------------------------------------------------------
# Estimates for every draw
# ----------------------------------------------------------------------------------


def _sum_draws(s: Series, quarters: Sequence[str], settings: Settings, counts: np.ndarray,
               first_index: int) -> np.ndarray:
    """The two-lag sum at the point (index 0) and in every draw."""
    rows = lag_rows(s.du, s.g, quarters, settings.lags, FIRST, settings.last)
    v = np.column_stack([rows.x, rows.y])
    p = v.shape[1]
    w = np.vstack([np.ones(len(rows.q)), counts[:, rows.q - first_index]])
    gram = np.asarray(w @ np.einsum("ri,rj->rij", v, v).reshape(len(rows.q), p * p))
    gram = gram.reshape(-1, p, p)
    a, b = gram[:, :-1, :-1], gram[:, :-1, -1]
    out = np.full(b.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = np.linalg.cond(a) < MAX_CONDITION
    out[ok] = np.linalg.solve(a[ok], b[ok][..., None])[..., 0]
    return out[:, 1:].sum(axis=1)


@dataclass
class Fit:
    settings: Settings
    country: dict[str, np.ndarray] = field(default_factory=dict)
    eu: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def unweighted(self) -> np.ndarray:
        return np.mean(np.vstack([self.country[g] for g in EU27]), axis=0)

    def weighted(self, w: Mapping[str, float]) -> np.ndarray:
        return np.sum(np.vstack([w[g] * self.country[g] for g in EU27]), axis=0)


def fit(series: Mapping[str, Series], eu: Series, quarters: Sequence[str],
        settings: Settings = PRIMARY, replicates: int = REPLICATES,
        seed: int = SEED) -> Fit:
    first_index = quarters.index(FIRST)
    n_quarters = quarters.index(settings.last) - first_index + 1
    counts = block_counts(n_quarters, settings.block, replicates, np.random.default_rng(seed))
    out = Fit(settings)
    for geo in EU27:
        out.country[geo] = _sum_draws(series[geo], quarters, settings, counts, first_index)
    out.eu = _sum_draws(eu, quarters, settings, counts, first_index)
    return out


# ----------------------------------------------------------------------------------
# Tests (protocol, section 5)
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Test:
    setting: str
    key: str
    statistic: str
    estimate: float
    low: float
    high: float
    dropped: float
    p_value: float
    holm: float
    verdict: str


def _interval(values: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    draws = values[1:]
    finite = draws[np.isfinite(draws)]
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high), 1.0 - len(finite) / len(draws), finite


def evaluate(f: Fit, w: Mapping[str, float]) -> list[Test]:
    raw = []
    for key, statistic, values in (
            ("T1", "labour-force-weighted minus unweighted mean", f.weighted(w) - f.unweighted()),
            ("T2", "EU aggregate minus labour-force-weighted mean", f.eu - f.weighted(w))):
        low, high, dropped, finite = _interval(values)
        raw.append((key, statistic, float(values[0]), low, high, dropped, two_sided_p(finite)))
    adjusted = holm([r[6] for r in raw])
    out = []
    for (key, statistic, point, low, high, dropped, p), h in zip(raw, adjusted, strict=True):
        verdict = ("unreliable" if dropped > MAX_DROPPED else
                   "differs" if h < ALPHA else "not determined")
        out.append(Test(f.settings.key, key, statistic, point, low, high, dropped, p, h,
                        verdict))
    return out


def check_reproduction(f: Fit, elasticities_csv: Path) -> int:
    """The country sums equal the case study's unemployment elasticities."""
    with elasticities_csv.open(encoding="utf-8") as stream:
        committed = {r["geo"]: float(r["elasticity"]) for r in csv.DictReader(stream)
                     if r["measure"] == "du"}
    for geo, value in committed.items():
        if abs(float(f.country[geo][0]) - value) > 1e-9:
            raise ValueError(f"{geo}: {f.country[geo][0]!r} does not reproduce the case "
                             f"study's {value!r}; the run stops.")
    return len(committed)


# ----------------------------------------------------------------------------------
# Study, report and files
# ----------------------------------------------------------------------------------


@dataclass
class Study:
    weights: dict[str, float]
    primary: Fit
    tests: list[Test]
    sensitivity: list[Test]
    description: dict[str, Any]
    reproduced: int
    provenance: dict[str, Any]


def describe(f: Fit, w: Mapping[str, float], check: Mapping[str, float],
             cce: Mapping[str, float]) -> dict[str, Any]:
    def band(values: np.ndarray) -> dict[str, Any]:
        low, high, dropped, _ = _interval(values)
        return {"point": float(values[0]), "interval": [low, high], "dropped": dropped}

    top = sorted(w, key=lambda g: -w[g])[:5]
    return {
        "theta_mg": band(f.unweighted()), "theta_lf": band(f.weighted(w)),
        "theta_eu": band(f.eu),
        "top_weights": {g: w[g] for g in top},
        "top_share": float(sum(w[g] for g in top)),
        "cce_unweighted": float(np.mean([cce[g] for g in EU27])),
        "cce_weighted": float(sum(w[g] * cce[g] for g in EU27)),
        "data_check": dict(check),
    }


def read_cce(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as stream:
        return {r["geo"]: float(r["cce"]) for r in csv.DictReader(stream)}


def build_study(data_dir: Path, snapshot: Path, elasticities_csv: Path, cce_csv: Path,
                replicates: int = REPLICATES) -> Study:
    audit = audit_snapshot(data_dir)
    manifest = audit_eu_aggregate(snapshot)
    quarters = quarters_for()
    series = load_series(_read_panel(data_dir / "processed" / "eu27_quarterly.csv"), quarters)
    aggregate = load_eu_aggregate(snapshot)
    w = weights(aggregate, quarters)
    eu = eu_series(aggregate, quarters)
    primary = fit(series, eu, quarters, PRIMARY, replicates)
    reproduced = check_reproduction(primary, elasticities_csv)
    sensitivity = [t for s in SENSITIVITY
                   for t in evaluate(fit(series, eu, quarters, s, replicates), w)]
    provenance = {"eurostat_manifest_sha256": audit["snapshot_manifest_sha256"],
                  "aggregate_downloaded_at_utc": manifest["downloaded_at_utc"],
                  "aggregate_files": {f["key"]: {"updated": f["updated"], "sha256": f["sha256"]}
                                      for f in manifest["files"]}}
    return Study(w, primary, evaluate(primary, w), sensitivity,
                 describe(primary, w, data_check(aggregate, quarters), read_cce(cce_csv)),
                 reproduced, provenance)


def _band(point: float, low: float, high: float) -> str:
    return f"{point:+.2f} [{low:+.2f}, {high:+.2f}]"


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def render(st: Study, command: str) -> str:
    d = st.description
    check = d["data_check"]
    mg, lf, eu = d["theta_mg"], d["theta_lf"], d["theta_eu"]
    gdp_vintage = st.provenance["aggregate_files"]["gdp"]["updated"][:10]
    lines = [
        "# The EU as a whole: a labour-force-weighted aggregate", "",
        "Generated by `eu-okun aggregate` under [AGGREGATE_PROTOCOL.md](../AGGREGATE_PROTOCOL.md), "
        "registered before its data were frozen or any estimate was computed. Two-lag sums "
        "of the unemployment change on GDP growth, on the headline sample (2009-Q3 to "
        "2026-Q2, windows touching 2020-2021 excluded). 95% synchronised block-bootstrap "
        f"intervals (block 4, B = {REPLICATES}, seed {SEED}). Associations only.", "",
        f"**Reproduction.** The {st.reproduced} country sums equal the case study's "
        "unemployment elasticities.", "",
        f"**Data check.** The labour-force-weighted rate of the 27 countries matches the "
        f"official `EU27_2020` rate within {check['max_abs_difference']:.2f} pp over "
        f"{check['quarters']} quarters; the correlation of their quarterly changes is "
        f"{check['change_correlation']:.3f}. The EU growth rate is from the {gdp_vintage} "
        "release, later than the frozen country series (2026-10-15).", "",
        "## Estimates", "",
        "| Estimate | Two-lag sum [95%] |",
        "| --- | --- |",
        f"| Unweighted mean of the 27 country sums | {_band(mg['point'], *mg['interval'])} |",
        f"| Labour-force-weighted mean | {_band(lf['point'], *lf['interval'])} |",
        f"| The EU aggregate (`EU27_2020` series) | {_band(eu['point'], *eu['interval'])} |",
        "", "## Registered tests (two-sided, Holm over the two)", "",
        "| Test | Difference | Estimate [95%] | Draws dropped | p | Holm p | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for t in st.tests:
        lines.append(f"| {t.key} | {t.statistic} | {_band(t.estimate, t.low, t.high)} | "
                     f"{t.dropped:.1%} | {_p(t.p_value)} | {_p(t.holm)} | {t.verdict} |")
    lines += [
        "", "## Sensitivity (fixed in advance)", "",
        "| Setting | Test | Estimate [95%] | Holm p | Verdict |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for t in st.sensitivity:
        lines.append(f"| {t.setting} | {t.key} | {_band(t.estimate, t.low, t.high)} | "
                     f"{_p(t.holm)} | {t.verdict} |")
    top = ", ".join(f"{g} {100 * w:.1f}%" for g, w in d["top_weights"].items())
    lines += [
        "", "## Weights and the CCE sums (descriptive)", "",
        f"- The five largest labour forces hold {100 * d['top_share']:.0f}% of the weight: "
        f"{top}.",
        f"- CCE sums (CCE_PROTOCOL.md): unweighted mean {d['cce_unweighted']:+.2f}, "
        f"labour-force-weighted {d['cce_weighted']:+.2f}.", "",
        "| Country | Weight | Country sum |",
        "| --- | ---: | ---: |",
    ]
    for g in EU27:
        lines.append(f"| {g} | {100 * st.weights[g]:.2f}% | "
                     f"{float(st.primary.country[g][0]):+.2f} |")
    lines += [
        "", "## Reading", "",
        "- The EU growth rate weights countries by GDP, and the aggregate also carries the "
        "correlation of shocks across countries. The aggregate sum therefore need not equal "
        "any average of the country sums. None of these numbers is causal.",
        "", "## Reproduction", "", "```bash", command, "```", "",
    ]
    return "\n".join(lines)


def _json(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json(v) for v in value]
    return value


def write_report(data_dir: Path, snapshot: Path, elasticities_csv: Path, cce_csv: Path,
                 output_dir: Path, report: Path | None = None) -> Study:
    st = build_study(data_dir, snapshot, elasticities_csv, cce_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        with (output_dir / name).open("w", encoding="utf-8", newline="\n") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write("weights.csv", [{"geo": g, "weight": st.weights[g],
                           "country_sum": float(st.primary.country[g][0])} for g in EU27])
    write("tests.csv", [asdict(t) for t in st.tests])
    write("sensitivity.csv", [asdict(t) for t in st.sensitivity])
    summary = {"reproduced": st.reproduced, "seed": SEED, "replicates": REPLICATES,
               **st.description, "verdicts": {t.key: t.verdict for t in st.tests},
               "provenance": st.provenance}
    (output_dir / "summary.json").write_text(json.dumps(_json(summary), indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    if report is not None:
        command = (f"poetry run eu-okun aggregate --data-dir {data_dir.as_posix()} --snapshot "
                   f"{snapshot.as_posix()} --case-study {elasticities_csv.as_posix()} --cce "
                   f"{cce_csv.as_posix()} --output-dir {output_dir.as_posix()} --report "
                   f"{report.as_posix()}")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render(st, command), encoding="utf-8", newline="\n")
    return st
