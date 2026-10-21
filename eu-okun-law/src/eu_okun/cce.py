"""A common-correlated-effects mean-group panel (``CCE_PROTOCOL.md``).

Pesaran's (2006) CCE-MG: each country's two-lag regression of the unemployment change on
GDP growth is augmented with cross-country averages of GDP growth (with the same lags) and
of the unemployment change, which approximate unobserved common factors. The country sums
are averaged. Plain mean-group OLS is computed from the same rows for comparison, and both
use the synchronised block bootstrap over quarters of the case study. Associations only.
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

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
    spearman,
    two_sided_p,
)
from eu_okun.hours import full_rows

REPLICATES = 1999
SEED = 20261028
MAX_CONDITION = 1e8
MAX_DROPPED = 0.05


@dataclass(frozen=True)
class Settings:
    """Choices that the sensitivity analyses vary (protocol, section 7)."""

    key: str = "primary"
    lags: int = 2
    block: int = 4
    leave_one_out: bool = False
    exclude: tuple[str, ...] = ()
    last: str = LAST
    pandemic: bool = False            # True: every row, pandemic windows included (sec. 6)


PRIMARY = Settings()
SENSITIVITY: tuple[Settings, ...] = (
    Settings("lags_1", lags=1), Settings("lags_4", lags=4),
    Settings("leave_one_out", leave_one_out=True), Settings("block_8", block=8),
    Settings("without_IE_LU", exclude=("IE", "LU")), Settings("pre_2020", last=PRE_LAST))
FULL_SAMPLE = Settings("full_sample", pandemic=True)


# ----------------------------------------------------------------------------------
# Designs
# ----------------------------------------------------------------------------------


def averages(series: Mapping[str, Series], countries: Sequence[str],
             leave_out: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Cross-country means of g and du per quarter, over the countries with data."""
    members = [g for g in countries if g != leave_out]
    g = np.vstack([series[c].g for c in members])
    du = np.vstack([series[c].du for c in members])
    with warnings.catch_warnings():
        # A quarter in which no country has data (the first du) has no average.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(g, axis=0), np.nanmean(du, axis=0)


@dataclass(frozen=True)
class Design:
    """One country's rows: V = [1, g lags, gbar lags, dubar, du] and each row's quarter."""

    v: np.ndarray
    q: np.ndarray


def design(s: Series, gbar: np.ndarray, dubar: np.ndarray, quarters: Sequence[str],
           settings: Settings) -> Design:
    if settings.pandemic:
        rows = full_rows(s.du, s.g, quarters, settings.lags)
    else:
        rows = lag_rows(s.du, s.g, quarters, settings.lags, FIRST, settings.last)
    bars = np.column_stack([gbar[rows.q - k] for k in range(settings.lags + 1)]
                           + [dubar[rows.q]])
    keep = np.all(np.isfinite(bars), axis=1)
    v = np.column_stack([rows.x, bars, rows.y])[keep]
    return Design(v, rows.q[keep])


# ----------------------------------------------------------------------------------
# Estimates for every draw
# ----------------------------------------------------------------------------------


def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(b.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = np.linalg.cond(a) < MAX_CONDITION
    if ok.any():
        out[ok] = np.linalg.solve(a[ok], b[ok][..., None])[..., 0]
    return out


@dataclass
class Fit:
    """Point estimates (index 0) and bootstrap draws of the country sums of one setting."""

    settings: Settings
    countries: list[str]
    ols: dict[str, np.ndarray] = field(default_factory=dict)
    cce: dict[str, np.ndarray] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)

    def mean(self, which: str) -> np.ndarray:
        source = self.ols if which == "ols" else self.cce
        return np.mean(np.vstack([source[g] for g in self.countries]), axis=0)


def fit(series: Mapping[str, Series], quarters: Sequence[str], settings: Settings = PRIMARY,
        replicates: int = REPLICATES, seed: int = SEED) -> Fit:
    first_index = quarters.index(FIRST)
    n_quarters = quarters.index(settings.last) - first_index + 1
    counts = block_counts(n_quarters, settings.block, replicates, np.random.default_rng(seed))
    countries = [g for g in EU27 if g not in settings.exclude]
    out = Fit(settings, countries)
    k = settings.lags + 1
    shared = averages(series, countries)
    for geo in countries:
        gbar, dubar = averages(series, countries, geo) if settings.leave_one_out else shared
        d = design(series[geo], gbar, dubar, quarters, settings)
        out.rows[geo] = len(d.q)
        p = d.v.shape[1]
        weights = np.vstack([np.ones(len(d.q)), counts[:, d.q - first_index]])
        outer = np.einsum("ri,rj->rij", d.v, d.v).reshape(len(d.q), p * p)
        gram = np.asarray(weights @ outer).reshape(-1, p, p)
        ols = _solve(gram[:, :1 + k, :1 + k], gram[:, :1 + k, -1])
        cce = _solve(gram[:, :-1, :-1], gram[:, :-1, -1])
        out.ols[geo] = ols[:, 1:1 + k].sum(axis=1)
        out.cce[geo] = cce[:, 1:1 + k].sum(axis=1)
    return out


# ----------------------------------------------------------------------------------
# Tests (protocol, section 5)
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Test:
    setting: str
    key: str
    statistic: str
    countries: int
    estimate: float
    low: float
    high: float
    dropped: float
    p_value: float
    holm: float
    verdict: str


def _draws(values: np.ndarray) -> tuple[float, np.ndarray, float]:
    draws = values[1:]
    finite = draws[np.isfinite(draws)]
    return float(values[0]), finite, 1.0 - len(finite) / len(draws)


def mean_group_t(f: Fit) -> tuple[float, float, float]:
    """Pesaran's (2006) mean-group statistic: estimate, standard error, two-sided p."""
    theta = np.asarray([f.cce[g][0] for g in f.countries])
    n = len(theta)
    se = float(np.sqrt(np.sum((theta - theta.mean()) ** 2) / (n * (n - 1))))
    t = float(theta.mean()) / se
    return float(theta.mean()), se, float(2 * stats.t.sf(abs(t), n - 1))


def evaluate(f: Fit) -> list[Test]:
    point1, se1, p1 = mean_group_t(f)
    _, finite1, dropped1 = _draws(f.mean("cce"))
    low1, high1 = (float(np.percentile(finite1, 2.5)), float(np.percentile(finite1, 97.5)))
    point2, finite2, dropped2 = _draws(f.mean("cce") - f.mean("ols"))
    low2, high2 = (float(np.percentile(finite2, 2.5)), float(np.percentile(finite2, 97.5)))
    raw = [("P1", "CCE mean-group sum", point1, low1, high1, dropped1, p1, -1),
           ("P2", "CCE minus plain mean group", point2, low2, high2, dropped2,
            two_sided_p(finite2), 1)]
    adjusted = holm([r[6] for r in raw])
    out = []
    for (key, statistic, point, low, high, dropped, p, sign), h in zip(raw, adjusted,
                                                                       strict=True):
        if dropped > MAX_DROPPED:
            verdict = "unreliable"
        else:
            verdict = ("supported" if h < ALPHA and np.sign(point) == sign else
                       "contradicted" if h < ALPHA else "not determined")
        out.append(Test(f.settings.key, key, statistic, len(f.countries), point, low, high,
                        dropped, p, h, verdict))
    return out


def country_rows(f: Fit) -> list[dict[str, Any]]:
    out = []
    for geo in f.countries:
        row: dict[str, Any] = {"geo": geo, "rows": f.rows[geo]}
        for name, values in (("ols", f.ols[geo]), ("cce", f.cce[geo])):
            point, finite, dropped = _draws(values)
            row.update({name: point, f"{name}_low": float(np.percentile(finite, 2.5)),
                        f"{name}_high": float(np.percentile(finite, 97.5)),
                        f"{name}_dropped": dropped})
        out.append(row)
    return out


def check_reproduction(f: Fit, elasticities_csv: Path) -> int:
    """The OLS sums equal the case study's unemployment elasticities (section 3)."""
    with elasticities_csv.open(encoding="utf-8") as stream:
        committed = {r["geo"]: float(r["elasticity"]) for r in csv.DictReader(stream)
                     if r["measure"] == "du"}
    for geo, value in committed.items():
        if abs(float(f.ols[geo][0]) - value) > 1e-9:
            raise ValueError(f"{geo}: {f.ols[geo][0]!r} does not reproduce the case study's "
                             f"{value!r} (protocol, section 3); the run stops.")
    return len(committed)


# ----------------------------------------------------------------------------------
# Study, report and files
# ----------------------------------------------------------------------------------


@dataclass
class Study:
    primary: Fit
    tests: list[Test]
    sensitivity: list[Test]
    description: dict[str, Any]
    reproduced: int
    provenance: dict[str, Any]


def describe(f: Fit, full: Fit, bayes: Mapping[str, float]) -> dict[str, Any]:
    cce = np.asarray([f.cce[g][0] for g in f.countries])
    ols = np.asarray([f.ols[g][0] for g in f.countries])
    rows = country_rows(f)
    with_bayes = [g for g in f.countries if g in bayes]
    return {
        "countries": f.countries,
        "mean_ols": float(ols.mean()), "mean_cce": float(cce.mean()),
        "mean_group_se": mean_group_t(f)[1],
        "cce_below_zero": [r["geo"] for r in rows if r["cce_high"] < 0],
        "ols_below_zero": [r["geo"] for r in rows if r["ols_high"] < 0],
        "closer_to_zero": [g for g in f.countries if abs(f.cce[g][0]) < abs(f.ols[g][0])],
        "spearman_ols": spearman(cce, ols),
        "spearman_bayes": spearman(np.asarray([f.cce[g][0] for g in with_bayes]),
                                   np.asarray([bayes[g] for g in with_bayes])),
        "full_sample": {"mean_ols": float(full.mean("ols")[0]),
                        "mean_cce": float(full.mean("cce")[0])},
    }


def read_bayes(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as stream:
        return {r["geo"]: float(r["theta_median"]) for r in csv.DictReader(stream)
                if r["spec"] == "primary"}


def study_from_series(series: Mapping[str, Series], quarters: Sequence[str],
                      elasticities_csv: Path, bayes: Mapping[str, float],
                      replicates: int = REPLICATES,
                      provenance: dict[str, Any] | None = None) -> Study:
    primary = fit(series, quarters, PRIMARY, replicates)
    reproduced = check_reproduction(primary, elasticities_csv)
    sensitivity = [t for s in SENSITIVITY
                   for t in evaluate(fit(series, quarters, s, replicates))]
    full = fit(series, quarters, FULL_SAMPLE, replicates)
    return Study(primary, evaluate(primary), sensitivity, describe(primary, full, bayes),
                 reproduced, provenance or {})


def build_study(data_dir: Path, elasticities_csv: Path, bayes_csv: Path) -> Study:
    audit = audit_snapshot(data_dir)
    base = _read_panel(data_dir / "processed" / "eu27_quarterly.csv")
    quarters = quarters_for()
    return study_from_series(load_series(base, quarters), quarters, elasticities_csv,
                             read_bayes(bayes_csv),
                             provenance={"eurostat_manifest_sha256":
                                         audit["snapshot_manifest_sha256"]})


def _band(point: float, low: float, high: float) -> str:
    return f"{point:+.2f} [{low:+.2f}, {high:+.2f}]"


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def render(st: Study, command: str) -> str:
    d = st.description
    lines = [
        "# A common-correlated-effects mean-group panel", "",
        "Generated by `eu-okun cce` under [CCE_PROTOCOL.md](../CCE_PROTOCOL.md), registered "
        "before any CCE estimate was computed. Each country's two-lag regression of the "
        "unemployment change on GDP growth, on the headline sample (2009-Q3 to 2026-Q2, "
        "windows touching 2020-2021 excluded), is augmented with the cross-country averages "
        "of GDP growth (lags 0-2) and of the unemployment change (Pesaran 2006). Country sums "
        "are averaged without weights. P1 uses the mean-group t statistic; intervals and P2 "
        f"use a synchronised block bootstrap over quarters (block 4, B = {REPLICATES}, seed "
        f"{SEED}). Associations only.", "",
        f"**Reproduction.** The {st.reproduced} plain OLS sums equal the case study's "
        "unemployment elasticities.", "",
        "## Registered tests", "",
        "| Test | Statistic | Countries | Estimate [95% bootstrap] | Draws dropped | p | "
        "Holm p | Verdict |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for t in st.tests:
        lines.append(f"| {t.key} | {t.statistic} | {t.countries} | "
                     f"{_band(t.estimate, t.low, t.high)} | {t.dropped:.1%} | {_p(t.p_value)} | "
                     f"{_p(t.holm)} | {t.verdict} |")
    lines += [
        "", f"Plain mean group: {d['mean_ols']:+.2f}; CCE mean group: {d['mean_cce']:+.2f} "
        f"(mean-group standard error {d['mean_group_se']:.3f}).", "",
        "## Sensitivity (fixed in advance)", "",
        "| Setting | Test | Countries | Estimate [95%] | Draws dropped | Holm p | Verdict |",
        "| --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for t in st.sensitivity:
        lines.append(f"| {t.setting} | {t.key} | {t.countries} | "
                     f"{_band(t.estimate, t.low, t.high)} | {t.dropped:.1%} | {_p(t.holm)} | "
                     f"{t.verdict} |")
    fs = d["full_sample"]
    lines += [
        "", "## Every country (descriptive)", "",
        f"- CCE interval below zero: {len(d['cce_below_zero'])} of {len(d['countries'])} "
        f"countries (OLS: {len(d['ols_below_zero'])}).",
        f"- CCE sum closer to zero than the OLS sum: {len(d['closer_to_zero'])} countries.",
        f"- Rank correlation of the CCE sums with OLS: {d['spearman_ols']:+.2f}; with the "
        f"Bayesian primary medians: {d['spearman_bayes']:+.2f}.",
        f"- Full sample, pandemic included: plain mean group {fs['mean_ols']:+.2f}, CCE "
        f"{fs['mean_cce']:+.2f}.", "",
        "| Country | Rows | OLS [95%] | CCE [95%] |",
        "| --- | ---: | --- | --- |",
    ]
    for r in country_rows(st.primary):
        lines.append(f"| {r['geo']} | {r['rows']} | {_band(r['ols'], r['ols_low'], r['ols_high'])} "
                     f"| {_band(r['cce'], r['cce_low'], r['cce_high'])} |")
    lines += [
        "", "## Reading", "",
        "- A CCE sum is the association with a country's own growth beyond what the "
        "cross-country averages absorb. Like the Bayesian factor model's sum, it is a "
        "different estimand from the plain sum, and neither is causal.",
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


def write_report(data_dir: Path, elasticities_csv: Path, bayes_csv: Path, output_dir: Path,
                 report: Path | None = None) -> Study:
    st = build_study(data_dir, elasticities_csv, bayes_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        with (output_dir / name).open("w", encoding="utf-8", newline="\n") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write("countries.csv", country_rows(st.primary))
    write("tests.csv", [asdict(t) for t in st.tests])
    write("sensitivity.csv", [asdict(t) for t in st.sensitivity])
    summary = {"reproduced": st.reproduced, "seed": SEED, "replicates": REPLICATES,
               **st.description, "verdicts": {t.key: t.verdict for t in st.tests},
               "provenance": st.provenance}
    (output_dir / "summary.json").write_text(json.dumps(_json(summary), indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    if report is not None:
        command = (f"poetry run eu-okun cce --data-dir {data_dir.as_posix()} --case-study "
                   f"{elasticities_csv.as_posix()} --bayes {bayes_csv.as_posix()} "
                   f"--output-dir {output_dir.as_posix()} --report {report.as_posix()}")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render(st, command), encoding="utf-8", newline="\n")
    return st
