"""Asymmetry between expansions and contractions (``ASYMMETRY_PROTOCOL.md``).

The change in the unemployment rate is regressed on GDP growth and on its negative part,
each with lags, on the headline sample. The primary statistic is a common extra response
to negative growth, pooled over countries that keep their own intercepts and Okun slopes;
the secondary one is the mean of country-by-country asymmetries. Both use the case study's
synchronised block bootstrap over quarters. Every result is an association.
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
SEED = 20261027
MIN_NEGATIVE = 8
MAX_DROPPED = 0.05


@dataclass(frozen=True)
class Settings:
    """Choices that the sensitivity analyses vary (protocol, section 7)."""

    key: str = "primary"
    lags: int = 2
    block: int = 4
    threshold: str = "zero"            # "zero" or "mean"
    exclude: tuple[str, ...] = ()
    last: str = LAST


PRIMARY = Settings()
SENSITIVITY: tuple[Settings, ...] = (
    Settings("lags_1", lags=1), Settings("lags_4", lags=4),
    Settings("mean_threshold", threshold="mean"), Settings("block_8", block=8),
    Settings("without_IE_LU", exclude=("IE", "LU")), Settings("without_EL", exclude=("EL",)),
    Settings("pre_2020", last=PRE_LAST))


# ----------------------------------------------------------------------------------
# Data and designs
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Series:
    g: np.ndarray
    du: np.ndarray


def load_series(base: Mapping[tuple[str, str], Mapping[str, float | None]],
                quarters: Sequence[str]) -> dict[str, Series]:
    out = {}
    for geo in EU27:
        def get(name: str, geo: str = geo) -> np.ndarray:
            values = [base.get((geo, q), {}).get(name) for q in quarters]
            return np.asarray([np.nan if v is None else float(v) for v in values])
        out[geo] = Series(get("gdp"), np.diff(get("unemployment"), prepend=np.nan))
    return out


@dataclass(frozen=True)
class Design:
    """One country's rows: V = [1, g lags, g_neg lags, du] and each row's quarter index."""

    v: np.ndarray
    q: np.ndarray
    negative: int                      # rows with g_neg[t] < 0


def design(s: Series, quarters: Sequence[str], settings: Settings) -> Design:
    rows = lag_rows(s.du, s.g, quarters, settings.lags, FIRST, settings.last)
    g_rows = rows.x[:, 1:]                                  # g[t], g[t-1], ...
    center = float(np.mean(g_rows[:, 0])) if settings.threshold == "mean" else 0.0
    g_neg = np.minimum(g_rows - center, 0.0)
    v = np.column_stack([rows.x, g_neg, rows.y])
    return Design(v, rows.q, int(np.sum(g_neg[:, 0] < 0)))


# ----------------------------------------------------------------------------------
# Weighted estimates for every draw
# ----------------------------------------------------------------------------------


def _grams(d: Design, counts: np.ndarray, first_index: int) -> np.ndarray:
    """Weighted Gram matrices of V: (draws + 1, p, p); draw 0 is the point estimate."""
    p = d.v.shape[1]
    weights = np.vstack([np.ones(len(d.q)), counts[:, d.q - first_index]])
    outer = np.einsum("ri,rj->rij", d.v, d.v).reshape(len(d.q), p * p)
    return np.asarray(weights @ outer).reshape(-1, p, p)


def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched solve; NaN where a matrix is singular."""
    out = np.full(b.shape, np.nan)
    ok = np.abs(np.linalg.det(a)) > 1e-10
    if ok.any():
        out[ok] = np.linalg.solve(a[ok], b[ok][..., None])[..., 0]
    return out


@dataclass
class Fit:
    """Point estimates (index 0) and draws of every statistic of one setting."""

    settings: Settings
    countries: list[str]
    eligible: list[str]
    negative: dict[str, int]
    pooled_c: np.ndarray                                   # (draws + 1,)
    pooled_theta_plus: np.ndarray                          # mean of sum b_i, pooled model
    theta_plus: dict[str, np.ndarray] = field(default_factory=dict)
    asymmetry: dict[str, np.ndarray] = field(default_factory=dict)
    symmetric: dict[str, float] = field(default_factory=dict)

    @property
    def mean_group(self) -> np.ndarray:
        return np.mean(np.vstack([self.asymmetry[g] for g in self.eligible]), axis=0)


def fit(series: Mapping[str, Series], quarters: Sequence[str], settings: Settings = PRIMARY,
        replicates: int = REPLICATES, seed: int = SEED) -> Fit:
    first_index = quarters.index(FIRST)
    n_quarters = quarters.index(settings.last) - first_index + 1
    counts = block_counts(n_quarters, settings.block, replicates, np.random.default_rng(seed))
    k = settings.lags + 1                                  # lag terms of each kind
    zc = 1 + k                                             # columns of Z = [1, g lags]
    countries = [g for g in EU27 if g not in settings.exclude]
    s_sum = np.zeros((replicates + 1, k, k))
    r_sum = np.zeros((replicates + 1, k))
    parts = {}
    out = Fit(settings, countries, [], {}, np.zeros(0), np.zeros(0))
    for geo in countries:
        d = design(series[geo], quarters, settings)
        out.negative[geo] = d.negative
        gram = _grams(d, counts, first_index)
        zz, zh = gram[:, :zc, :zc], gram[:, :zc, zc:zc + k]
        hh, zy, hy = gram[:, zc:zc + k, zc:zc + k], gram[:, :zc, -1], gram[:, zc:zc + k, -1]
        # Symmetric fit (c = 0) at the point: the reproduction check of section 9.
        out.symmetric[geo] = float(_solve(zz[:1], zy[:1])[0, 1:].sum())
        # Pooled model: eliminate the country's own intercept and slopes.
        inv_zh = np.stack([_solve(zz, zh[:, :, j]) for j in range(k)], axis=-1)
        inv_zy = _solve(zz, zy)
        s_sum += hh - np.einsum("dij,dik->djk", zh, inv_zh)
        r_sum += hy - np.einsum("dij,di->dj", zh, inv_zy)
        parts[geo] = (inv_zh, inv_zy)
        # Country-by-country model with its own asymmetry.
        full = _solve(gram[:, :zc + k, :zc + k], gram[:, :zc + k, -1])
        out.theta_plus[geo] = full[:, 1:zc].sum(axis=1)
        out.asymmetry[geo] = full[:, zc:].sum(axis=1)
    c = _solve(s_sum, r_sum)
    out.pooled_c = c.sum(axis=1)
    b_sums = [(parts[g][1] - np.einsum("dij,dj->di", parts[g][0], c))[:, 1:].sum(axis=1)
              for g in countries]
    out.pooled_theta_plus = np.mean(np.vstack(b_sums), axis=0)
    out.eligible = [g for g in countries if out.negative[g] >= MIN_NEGATIVE]
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


def _summary(values: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
    point, draws = float(values[0]), values[1:]
    finite = draws[np.isfinite(draws)]
    dropped = 1.0 - len(finite) / len(draws)
    low, high = (np.percentile(finite, [2.5, 97.5]) if len(finite) else (math.nan, math.nan))
    return point, float(low), float(high), dropped, finite


def evaluate(f: Fit) -> list[Test]:
    raw = []
    for key, statistic, n, values in (
            ("P1", "common asymmetry C (pooled)", len(f.countries), f.pooled_c),
            ("P2", "mean-group asymmetry", len(f.eligible), f.mean_group)):
        point, low, high, dropped, finite = _summary(values)
        raw.append((key, statistic, n, point, low, high, dropped, two_sided_p(finite)))
    adjusted = holm([r[7] for r in raw])
    out = []
    for (key, statistic, n, point, low, high, dropped, p), h in zip(raw, adjusted,
                                                                    strict=True):
        if dropped > MAX_DROPPED:
            verdict = "unreliable"
        else:
            verdict = ("supported" if h < ALPHA and point < 0 else
                       "contradicted" if h < ALPHA else "not determined")
        out.append(Test(f.settings.key, key, statistic, n, point, low, high, dropped, p, h,
                        verdict))
    return out


def country_rows(f: Fit) -> list[dict[str, Any]]:
    out = []
    for geo in f.countries:
        row: dict[str, Any] = {"geo": geo, "negative_quarters": f.negative[geo],
                               "eligible": geo in f.eligible}
        plus = f.theta_plus[geo]
        minus = plus + f.asymmetry[geo]
        for name, values in (("theta_plus", plus), ("theta_minus", minus),
                             ("asymmetry", f.asymmetry[geo])):
            point, low, high, dropped, _ = _summary(values)
            row.update({name: point, f"{name}_low": low, f"{name}_high": high})
        row["dropped"] = _summary(f.asymmetry[geo])[3]
        out.append(row)
    return out


def check_reproduction(f: Fit, elasticities_csv: Path) -> int:
    """The symmetric part equals the case study's unemployment elasticities (section 9)."""
    with elasticities_csv.open(encoding="utf-8") as stream:
        committed = {r["geo"]: float(r["elasticity"]) for r in csv.DictReader(stream)
                     if r["measure"] == "du"}
    for geo, value in committed.items():
        if abs(f.symmetric[geo] - value) > 1e-9:
            raise ValueError(f"{geo}: {f.symmetric[geo]!r} does not reproduce the case "
                             f"study's {value!r} (protocol, section 9); the run stops.")
    return len(committed)


# ----------------------------------------------------------------------------------
# Study, report and files
# ----------------------------------------------------------------------------------


@dataclass
class Study:
    primary: Fit
    tests: list[Test]
    sensitivity: list[Test]
    reproduced: int
    provenance: dict[str, Any]


def study_from_series(series: Mapping[str, Series], quarters: Sequence[str],
                      elasticities_csv: Path, replicates: int = REPLICATES,
                      provenance: dict[str, Any] | None = None) -> Study:
    primary = fit(series, quarters, PRIMARY, replicates)
    reproduced = check_reproduction(primary, elasticities_csv)
    sensitivity = [t for s in SENSITIVITY for t in evaluate(fit(series, quarters, s,
                                                                 replicates))]
    return Study(primary, evaluate(primary), sensitivity, reproduced, provenance or {})


def build_study(data_dir: Path, elasticities_csv: Path) -> Study:
    audit = audit_snapshot(data_dir)
    base = _read_panel(data_dir / "processed" / "eu27_quarterly.csv")
    quarters = quarters_for()
    return study_from_series(load_series(base, quarters), quarters, elasticities_csv,
                             provenance={"eurostat_manifest_sha256":
                                         audit["snapshot_manifest_sha256"]})


def _band(point: float, low: float, high: float) -> str:
    if not math.isfinite(point):
        return "n/a"
    return f"{point:+.2f} [{low:+.2f}, {high:+.2f}]"


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def render(st: Study, command: str) -> str:
    f = st.primary
    plus, *_ = _summary(f.pooled_theta_plus)
    excluded = [g for g in f.countries if g not in f.eligible]
    lines = [
        "# Asymmetry between expansions and contractions", "",
        "Generated by `eu-okun asymmetry` under [ASYMMETRY_PROTOCOL.md](../ASYMMETRY_PROTOCOL.md), "
        "registered before any asymmetric estimate was computed. The change in the "
        "unemployment rate on GDP growth and its negative part, each with lags 0-2, on the "
        "headline sample (2009-Q3 to 2026-Q2, windows touching 2020-2021 excluded). 95% "
        f"synchronised block-bootstrap intervals (block 4, B = {REPLICATES}, seed {SEED}). "
        "A negative asymmetry means unemployment responds more strongly to falling output. "
        "Associations only.", "",
        f"**Reproduction.** With the asymmetry fixed at zero, the {st.reproduced} country "
        "sums equal the case study's unemployment elasticities.", "",
        "## Registered tests", "",
        "| Test | Statistic | Countries | Estimate [95%] | Draws dropped | p | Holm p | "
        "Verdict |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for t in st.tests:
        lines.append(f"| {t.key} | {t.statistic} | {t.countries} | "
                     f"{_band(t.estimate, t.low, t.high)} | {t.dropped:.1%} | {_p(t.p_value)} | "
                     f"{_p(t.holm)} | {t.verdict} |")
    lines += [
        "", f"In the pooled model the mean response to positive growth is {plus:+.2f}; "
        f"to negative growth it is {plus + float(f.pooled_c[0]):+.2f}. The mean group uses "
        f"the {len(f.eligible)} countries with at least {MIN_NEGATIVE} negative-growth "
        f"quarters (not: {', '.join(excluded) or 'none'}).", "",
        "## Sensitivity (fixed in advance)", "",
        "| Setting | Test | Countries | Estimate [95%] | Draws dropped | Holm p | Verdict |",
        "| --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for t in st.sensitivity:
        lines.append(f"| {t.setting} | {t.key} | {t.countries} | "
                     f"{_band(t.estimate, t.low, t.high)} | {t.dropped:.1%} | {_p(t.holm)} | "
                     f"{t.verdict} |")
    lines += [
        "", "## Every country (descriptive)", "",
        "| Country | Negative quarters | Positive growth | Negative growth | Asymmetry |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for r in country_rows(f):
        mark = "" if r["eligible"] else " *"
        lines.append(f"| {r['geo']}{mark} | {r['negative_quarters']} | "
                     f"{_band(r['theta_plus'], r['theta_plus_low'], r['theta_plus_high'])} | "
                     f"{_band(r['theta_minus'], r['theta_minus_low'], r['theta_minus_high'])} | "
                     f"{_band(r['asymmetry'], r['asymmetry_low'], r['asymmetry_high'])} |")
    lines += [
        "", f"\\* fewer than {MIN_NEGATIVE} negative-growth quarters: outside the mean group; "
        "their own estimates rest on very few quarters.", "",
        "## Reading", "",
        "- The contractions in this sample are concentrated in 2009-2013 and in the countries "
        "hit by the euro-area crisis, so an asymmetry can reflect different shocks as well as "
        "a different labour-market response. Nothing here identifies a mechanism.",
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


def write_report(data_dir: Path, elasticities_csv: Path, output_dir: Path,
                 report: Path | None = None) -> Study:
    st = build_study(data_dir, elasticities_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        with (output_dir / name).open("w", encoding="utf-8", newline="\n") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    f = st.primary
    write("countries.csv", country_rows(f))
    write("tests.csv", [asdict(t) for t in st.tests])
    write("sensitivity.csv", [asdict(t) for t in st.sensitivity])
    plus, plus_low, plus_high, _, _ = _summary(f.pooled_theta_plus)
    minus, minus_low, minus_high, _, _ = _summary(f.pooled_theta_plus + f.pooled_c)
    summary = {
        "reproduced": st.reproduced, "seed": SEED, "replicates": REPLICATES,
        "countries": f.countries, "eligible": f.eligible,
        "negative_quarters": f.negative,
        "pooled_theta_plus": {"point": plus, "interval": [plus_low, plus_high]},
        "pooled_theta_minus": {"point": minus, "interval": [minus_low, minus_high]},
        "verdicts": {t.key: t.verdict for t in st.tests},
        "sensitivity_verdicts": {f"{t.setting}:{t.key}": t.verdict for t in st.sensitivity},
        "provenance": st.provenance,
    }
    (output_dir / "summary.json").write_text(json.dumps(_json(summary), indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    if report is not None:
        command = (f"poetry run eu-okun asymmetry --data-dir {data_dir.as_posix()} "
                   f"--case-study {elasticities_csv.as_posix()} --output-dir "
                   f"{output_dir.as_posix()} --report {report.as_posix()}")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render(st, command), encoding="utf-8", newline="\n")
    return st
