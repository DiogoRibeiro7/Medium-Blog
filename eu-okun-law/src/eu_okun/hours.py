"""Okun's law in hours, for every country, with and without the pandemic
(``HOURS_PROTOCOL.md``).

The case study's measures, estimator and synchronised block bootstrap
(``heterogeneity``), applied to every country on the pandemic-excluded and the full
sample. The two samples share each resample, so their differences have a joint
distribution. The registered tests ask whether hours keep following output when 2020-2021
are included while unemployment does not, and whether adjustment moves from workers to
hours per worker. Every result is an association; nothing is identified.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
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
    PANDEMIC,
    CountryData,
    Estimate,
    Rows,
    block_counts,
    build_country,
    estimate,
    holm,
    lag_rows,
    quarters_for,
    spearman,
    two_sided_p,
)
from eu_okun.heterogeneity_data import audit_heterogeneity, load_heterogeneity

MEASURES: tuple[str, ...] = ("h", "du", "hpw", "e_dom", "prod")
MARGINS: tuple[str, ...] = ("prod", "hpw", "e_dom")
SAMPLES: tuple[str, ...] = ("excluded", "full")
FOCUS: tuple[str, ...] = ("ES", "PT", "NL", "LU")
REPLICATES = 1999
SEED = 20261026
PERMUTATIONS = 9999
TOLERANCE = 1e-9


@dataclass(frozen=True)
class Settings:
    """Choices that the sensitivity analyses vary (protocol, section 7)."""

    key: str = "primary"
    lags: int = 2
    block: int = 4
    exclude: tuple[str, ...] = ()


PRIMARY = Settings()
SENSITIVITY: tuple[Settings, ...] = (
    Settings("lags_1", lags=1), Settings("lags_4", lags=4), Settings("block_8", block=8),
    Settings("without_IE_LU", exclude=("IE", "LU")))


# ----------------------------------------------------------------------------------
# Estimates on both samples
# ----------------------------------------------------------------------------------


def full_rows(y: np.ndarray, g: np.ndarray, quarters: Sequence[str], lags: int) -> Rows:
    """y[t] on a constant and g[t..t-lags], pandemic windows included."""
    ys, xs, qs = [], [], []
    for t in range(lags, len(quarters)):
        if not FIRST <= quarters[t] <= LAST:
            continue
        regs = [g[t - j] for j in range(lags + 1)]
        if math.isfinite(y[t]) and all(math.isfinite(v) for v in regs):
            ys.append(y[t])
            xs.append([1.0, *regs])
            qs.append(t)
    return Rows(np.asarray(ys), np.asarray(xs).reshape(-1, lags + 2), np.asarray(qs, dtype=int))


def sample_rows(y: np.ndarray, g: np.ndarray, quarters: Sequence[str], lags: int,
                sample: str) -> Rows:
    if sample == "excluded":
        return lag_rows(y, g, quarters, lags, FIRST, LAST)
    return full_rows(y, g, quarters, lags)


def pandemic_rows(quarters: Sequence[str], lags: int) -> np.ndarray:
    """Rows from FIRST to LAST whose window [t-lags, t] touches 2020-2021."""
    first_index, last_index = quarters.index(FIRST), quarters.index(LAST)
    return np.asarray([any(PANDEMIC[0] <= q <= PANDEMIC[1]
                           for q in quarters[t - lags: t + 1])
                       for t in range(first_index, last_index + 1)])


def stratified_counts(pandemic: np.ndarray, block: int, replicates: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Circular block resamples within each stratum, in time order: the other rows first,
    then the pandemic rows (change log, amendment). Every resample keeps the observed
    number of pandemic rows."""
    counts = np.zeros((replicates, len(pandemic)))
    for stratum in (~pandemic, pandemic):
        positions = np.flatnonzero(stratum)
        counts[:, positions] = block_counts(len(positions), block, replicates, rng)
    return counts


@dataclass
class Run:
    """Every elasticity of one setting, both samples sharing one set of resamples."""

    settings: Settings
    estimates: dict[tuple[str, str, str], Estimate]     # (geo, measure, sample)
    bootstrap: str                                      # "registered" or "stratified"
    no_pandemic_share: float                            # resamples without a pandemic row

    def countries(self, measures: Sequence[str]) -> list[str]:
        return [g for g in EU27 if g not in self.settings.exclude
                and all((g, m, s) in self.estimates for m in measures for s in SAMPLES)]


def run(data: Mapping[str, CountryData], quarters: Sequence[str],
        settings: Settings = PRIMARY, replicates: int = REPLICATES, seed: int = SEED,
        stratified: bool = False) -> Run:
    first_index = quarters.index(FIRST)
    pandemic = pandemic_rows(quarters, settings.lags)
    rng = np.random.default_rng(seed)
    counts = (stratified_counts(pandemic, settings.block, replicates, rng) if stratified
              else block_counts(len(pandemic), settings.block, replicates, rng))
    none = counts[:, pandemic].sum(axis=1) == 0
    estimates = {}
    for geo, cd in data.items():
        for measure in MEASURES:
            for sample in SAMPLES:
                rows = sample_rows(cd.measures[measure], cd.g, quarters, settings.lags, sample)
                est = estimate(rows, counts, first_index)
                if est is not None:
                    estimates[(geo, measure, sample)] = est
            full, excl = (estimates.get((geo, measure, s)) for s in ("full", "excluded"))
            if full is not None and excl is not None:
                # Without a pandemic row the two regressions use the same rows and weights;
                # make them equal exactly, not up to rounding (change log).
                full.draws[none] = excl.draws[none]
                full.coef_draws[none] = excl.coef_draws[none]
    return Run(settings, estimates, "stratified" if stratified else "registered",
               float(np.mean(none)))


def check_reproduction(r: Run, elasticities_csv: Path) -> int:
    """The excluded-sample point estimates must equal the case study's (protocol, section 3)."""
    with elasticities_csv.open(encoding="utf-8") as stream:
        committed = {(row["geo"], row["measure"]): float(row["elasticity"])
                     for row in csv.DictReader(stream) if row["measure"] in MEASURES}
    ours = {(g, m): e.point for (g, m, s), e in r.estimates.items() if s == "excluded"}
    if set(ours) != set(committed):
        raise ValueError("The excluded-sample estimates do not cover the case study's "
                         "countries and measures (protocol, section 3); the run stops.")
    for key, value in committed.items():
        if abs(ours[key] - value) > TOLERANCE:
            raise ValueError(f"{key}: {ours[key]!r} does not reproduce the case study's "
                             f"{value!r} (protocol, section 3); the run stops.")
    return len(committed)


# ----------------------------------------------------------------------------------
# Registered tests (protocol, section 5)
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Statistic:
    point: float
    draws: np.ndarray

    def interval(self) -> tuple[float, float]:
        finite = self.draws[np.isfinite(self.draws)]
        return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def mean_beta(r: Run, measure: str, sample: str, geos: Sequence[str]) -> Statistic:
    ests = [r.estimates[(g, measure, sample)] for g in geos]
    return Statistic(float(np.mean([e.point for e in ests])),
                     np.mean(np.vstack([e.draws for e in ests]), axis=0))


def retention(r: Run, measure: str, geos: Sequence[str]) -> Statistic:
    full, excl = mean_beta(r, measure, "full", geos), mean_beta(r, measure, "excluded", geos)
    return Statistic(full.point / excl.point, full.draws / excl.draws)


def change(r: Run, measure: str, geos: Sequence[str]) -> Statistic:
    full, excl = mean_beta(r, measure, "full", geos), mean_beta(r, measure, "excluded", geos)
    return Statistic(full.point - excl.point, full.draws - excl.draws)


def country_change(r: Run, measure: str, geos: Sequence[str]) -> np.ndarray:
    return np.asarray([r.estimates[(g, measure, "full")].point
                       - r.estimates[(g, measure, "excluded")].point for g in geos])


def permutation_spearman(x: np.ndarray, y: np.ndarray, permutations: int = PERMUTATIONS,
                         seed: int = SEED) -> tuple[float, float]:
    """Two-sided permutation p of a Spearman correlation, by the formula of
    ``heterogeneity.correlations``."""
    rng = np.random.default_rng(seed)
    rho = spearman(x, y)
    perm = np.asarray([spearman(x, rng.permutation(y)) for _ in range(permutations)])
    return rho, float((1 + np.sum(np.abs(perm) >= abs(rho) - 1e-12)) / (permutations + 1))


@dataclass(frozen=True)
class Test:
    setting: str
    bootstrap: str
    key: str
    text: str
    predicted: int
    countries: int
    estimate: float
    low: float
    high: float
    p_value: float
    holm: float
    verdict: str


TESTS: tuple[tuple[str, str, int], ...] = (
    ("P1", "Hours keep more of their association than unemployment: R_h - R_du", 1),
    ("P2a", "Hours per worker respond more with the pandemic: change in mean beta_hpw", 1),
    ("P2b", "Persons respond less with the pandemic: change in mean beta_e_dom", -1),
    ("P3", "Unemployment is attenuated more where hours per worker take up more: "
           "Spearman of the two changes", 1),
)


def evaluate(r: Run, permutations: int = PERMUTATIONS) -> list[Test]:
    c_h, c_p = r.countries(("h", "du")), r.countries(("hpw", "e_dom", "du"))
    p1 = Statistic(retention(r, "h", c_h).point - retention(r, "du", c_h).point,
                   retention(r, "h", c_h).draws - retention(r, "du", c_h).draws)
    p2a, p2b = change(r, "hpw", c_p), change(r, "e_dom", c_p)
    rho, p3_p = permutation_spearman(country_change(r, "hpw", c_p),
                                     country_change(r, "du", c_p), permutations)
    raw: list[tuple[int, float, float, float, float]] = []
    for n, stat in ((len(c_h), p1), (len(c_p), p2a), (len(c_p), p2b)):
        low, high = stat.interval()
        raw.append((n, stat.point, low, high, two_sided_p(stat.draws)))
    raw.append((len(c_p), rho, math.nan, math.nan, p3_p))
    adjusted = holm([x[4] for x in raw])
    out = []
    for (key, text, sign), (n, point, low, high, p), h in zip(TESTS, raw, adjusted,
                                                            strict=True):
        verdict = ("supported" if h < ALPHA and np.sign(point) == sign else
                   "contradicted" if h < ALPHA else "not determined")
        out.append(Test(r.settings.key, r.bootstrap, key, text, sign, n, point, low, high, p,
                        h, verdict))
    return out


# ----------------------------------------------------------------------------------
# Descriptive (protocol, section 6)
# ----------------------------------------------------------------------------------


def country_rows(r: Run) -> list[dict[str, Any]]:
    out = []
    for geo in EU27:
        for measure in MEASURES:
            for sample in SAMPLES:
                e = r.estimates.get((geo, measure, sample))
                if e is None:
                    continue
                low, high = e.interval()
                out.append({"geo": geo, "measure": measure, "sample": sample,
                            "elasticity": e.point, "low": low, "high": high, "n": e.n})
    return out


def describe(r: Run) -> dict[str, Any]:
    c_h, c_p = r.countries(("h", "du")), r.countries(("hpw", "e_dom", "du"))

    def point(geo: str, measure: str, sample: str) -> float:
        return r.estimates[(geo, measure, sample)].point

    def above(sample: str) -> list[str]:
        return [g for g in c_h if r.estimates[(g, "h", sample)].interval()[0] > 0]

    def towards_zero(measure: str) -> list[str]:
        return [g for g in c_h if abs(point(g, measure, "full"))
                < abs(point(g, measure, "excluded"))]

    def ranks(sample: str) -> dict[str, int]:
        order = sorted(c_h, key=lambda g: -point(g, "h", sample))
        return {g: order.index(g) + 1 for g in FOCUS if g in order}

    return {
        "countries_hours": c_h, "countries_persons": c_p,
        "hours_above_zero": {s: above(s) for s in SAMPLES},
        "towards_zero": {m: towards_zero(m) for m in ("h", "du")},
        "rank_correlation_excluded": spearman(
            np.asarray([-point(g, "du", "excluded") for g in c_h]),
            np.asarray([point(g, "h", "excluded") for g in c_h])),
        "focus_rank": {s: ranks(s) for s in SAMPLES},
        "mean_hours": {s: mean_beta(r, "h", s, c_h).point for s in SAMPLES},
        "mean_du": {s: mean_beta(r, "du", s, c_h).point for s in SAMPLES},
        **intervals(r),
        "margin_mean": {m: {s: mean_beta(r, m, s, c_p).point for s in SAMPLES}
                        for m in MARGINS},
    }


# ----------------------------------------------------------------------------------
# The study, the report and the files
# ----------------------------------------------------------------------------------


def p_floor(r: Run) -> float:
    """Smallest two-sided p a full-minus-excluded difference can reach under ``r``'s
    resamples: those without a pandemic row give a difference of exactly zero."""
    replicates = len(next(iter(r.estimates.values())).draws)
    zero = round(r.no_pandemic_share * replicates)
    return min(1.0, 2 * (1 + zero) / (replicates + 1))


def intervals(r: Run) -> dict[str, Any]:
    """Retention ratios and margin changes with intervals under ``r``'s resamples."""
    c_h, c_p = r.countries(("h", "du")), r.countries(("hpw", "e_dom", "du"))
    return {"retention": {m: {"point": v.point, "interval": list(v.interval())}
                          for m in ("h", "du") for v in [retention(r, m, c_h)]},
            "margin_change": {m: {"point": v.point, "interval": list(v.interval())}
                              for m in MARGINS for v in [change(r, m, c_p)]}}


@dataclass
class Study:
    primary: Run
    tests: list[Test]
    sensitivity: list[Test]
    amended: list[Test]
    sensitivity_amended: list[Test]
    description: dict[str, Any]
    amended_intervals: dict[str, Any]
    floors: dict[str, dict[str, float]]
    reproduced: int
    provenance: dict[str, Any]


def study_from_data(data: Mapping[str, CountryData], quarters: Sequence[str],
                    elasticities_csv: Path, replicates: int = REPLICATES,
                    permutations: int = PERMUTATIONS,
                    provenance: dict[str, Any] | None = None) -> Study:
    primary = run(data, quarters, PRIMARY, replicates)
    reproduced = check_reproduction(primary, elasticities_csv)
    floors: dict[str, dict[str, float]] = {}
    sensitivity, sensitivity_amended = [], []
    for s in SENSITIVITY:
        r = run(data, quarters, s, replicates)
        floors[s.key] = {"no_pandemic_share": r.no_pandemic_share, "p_floor": p_floor(r)}
        sensitivity += evaluate(r, permutations)
        sensitivity_amended += evaluate(run(data, quarters, s, replicates, stratified=True),
                                        permutations)
    floors[PRIMARY.key] = {"no_pandemic_share": primary.no_pandemic_share,
                           "p_floor": p_floor(primary)}
    amended = run(data, quarters, PRIMARY, replicates, stratified=True)
    return Study(primary, evaluate(primary, permutations), sensitivity,
                 evaluate(amended, permutations), sensitivity_amended, describe(primary),
                 intervals(amended), floors, reproduced, provenance or {})


def build_study(base_dir: Path, snapshot: Path, elasticities_csv: Path) -> Study:
    audit = audit_snapshot(base_dir)
    manifest = audit_heterogeneity(snapshot)
    base = _read_panel(base_dir / "processed" / "eu27_quarterly.csv")
    extra, _ = load_heterogeneity(snapshot)
    quarters = quarters_for()
    data = {geo: build_country(geo, quarters, base, extra) for geo in EU27}
    provenance = {"eurostat_manifest_sha256": audit["snapshot_manifest_sha256"],
                  "heterogeneity_manifest": manifest["downloaded_at_utc"],
                  "hours_sha256": next(f["sha256"] for f in manifest["files"]
                                       if f["key"] == "hours"),
                  "persons_sha256": next(f["sha256"] for f in manifest["files"]
                                         if f["key"] == "emp_a10")}
    return study_from_data(data, quarters, elasticities_csv, provenance=provenance)


def _band(point: float, low: float, high: float) -> str:
    return f"{point:+.2f} [{low:+.2f}, {high:+.2f}]"


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _names(geos: Sequence[str]) -> str:
    return ", ".join(geos) if geos else "none"


def render(st: Study, command: str) -> str:
    d = st.description
    r = st.primary
    c_h, c_p = d["countries_hours"], d["countries_persons"]
    lines = [
        "# Okun's law in hours, for every country, with and without the pandemic", "",
        "Generated by `eu-okun hours` under [HOURS_PROTOCOL.md](../HOURS_PROTOCOL.md), "
        "registered before any full-sample elasticity was computed. The pandemic-excluded "
        "elasticities are the case study's and had been seen; they are descriptive here. "
        "Two-lag elasticities to GDP growth (sum of the coefficients on g[t], g[t-1], "
        "g[t-2]), quarters 2009-Q3 to 2026-Q2, 95% synchronised block-bootstrap intervals "
        f"(block 4, B = {len(r.estimates[(c_h[0], 'h', 'full')].draws)}, seed {SEED}). "
        "Associations only.", "",
        f"**Reproduction.** The {st.reproduced} pandemic-excluded point estimates of the case "
        "study (`docs/heterogeneity/elasticities.csv`) are reproduced.", "",
        "## Tests", "",
        f"Hours: {len(c_h)} countries ({_names([g for g in EU27 if g not in c_h])} missing). "
        f"Persons and hours per worker: {len(c_p)} countries. Holm over the four tests. "
        "Intervals and p-values of P1-P2b come from each bootstrap; P3 is a permutation "
        "test, the same in both.", "",
        "| Test | Prediction | Countries | Estimate | Registered: [95%], Holm p, verdict | "
        "Amended, stratified: [95%], Holm p, verdict |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for t, a in zip(st.tests, st.amended, strict=True):
        cells = []
        for x in (t, a):
            band = "" if math.isnan(x.low) else f"[{x.low:+.2f}, {x.high:+.2f}], "
            cells.append(f"{band}{_p(x.holm)}, {x.verdict}")
        lines.append(f"| {t.key} | {t.text} | {t.countries} | {t.estimate:+.2f} | "
                     + " | ".join(cells) + " |")
    floor = st.floors[PRIMARY.key]
    lines += [
        "", "**The registered tests could not support P1-P2b.** "
        f"{floor['no_pandemic_share']:.1%} of the registered resamples contain no pandemic "
        "row. In those the full and excluded estimates coincide, so the differences are "
        "exactly zero, and the two-sided p-value cannot fall below "
        f"{floor['p_floor']:.3f} (with block 8: {st.floors['block_8']['p_floor']:.3f}). "
        "The registered verdicts stand. The amendment, registered after the registered "
        "result had been seen and before it was computed (protocol, change log), holds the "
        "number of pandemic rows fixed in every resample. Its verdicts are secondary.",
    ]
    ret, ret_a = d["retention"], st.amended_intervals["retention"]
    lines += [
        "", "**Components** (intervals: registered; amended).", "",
        f"- Retention ratios: hours {_band(ret['h']['point'], *ret['h']['interval'])}; "
        f"[{ret_a['h']['interval'][0]:+.2f}, {ret_a['h']['interval'][1]:+.2f}]. "
        f"Unemployment {_band(ret['du']['point'], *ret['du']['interval'])}; "
        f"[{ret_a['du']['interval'][0]:+.2f}, {ret_a['du']['interval'][1]:+.2f}].",
        f"- Mean hours elasticity: {d['mean_hours']['excluded']:+.2f} without the pandemic, "
        f"{d['mean_hours']['full']:+.2f} with it. Mean unemployment elasticity: "
        f"{d['mean_du']['excluded']:+.2f} and {d['mean_du']['full']:+.2f}.",
        "- Mean change of each margin when the pandemic is included (they sum to zero): "
        + "; ".join(f"{m} {_band(v['point'], *v['interval'])}; [{a['interval'][0]:+.2f}, "
                    f"{a['interval'][1]:+.2f}]"
                    for (m, v), a in zip(d["margin_change"].items(),
                                         st.amended_intervals["margin_change"].values(),
                                         strict=True)) + ".",
        "", "## Sensitivity (fixed in advance)", "",
        "| Setting | Test | Countries | Estimate | Registered: Holm p, verdict | Amended: "
        "Holm p, verdict |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for t, a in zip(st.sensitivity, st.sensitivity_amended, strict=True):
        lines.append(f"| {t.setting} | {t.key} | {t.countries} | {t.estimate:+.2f} | "
                     f"{_p(t.holm)}, {t.verdict} | {_p(a.holm)}, {a.verdict} |")
    above = d["hours_above_zero"]
    towards = d["towards_zero"]
    lines += [
        "", "## Every country (descriptive)", "",
        f"- Hours interval above zero: {len(above['excluded'])} of {len(c_h)} countries "
        f"without the pandemic, {len(above['full'])} with it.",
        f"- Elasticity closer to zero with the pandemic: hours in {len(towards['h'])} of "
        f"{len(c_h)} countries, unemployment in {len(towards['du'])}.",
        f"- Rank correlation of the unemployment response (minus β_du) and the hours "
        f"response, without the pandemic: {d['rank_correlation_excluded']:+.2f}. Both are "
        "estimated on the same quarters, so their estimation errors are correlated.",
        "- Hours rank (1 = largest) of the focus countries: "
        + "; ".join(f"{g} {d['focus_rank']['excluded'][g]} without the pandemic, "
                    f"{d['focus_rank']['full'][g]} with it" for g in FOCUS) + ".",
        "",
        "| Country | Hours, without 2020-21 | Hours, full sample | Unemployment, without "
        "2020-21 | Unemployment, full sample |",
        "| --- | --- | --- | --- | --- |",
    ]
    for geo in EU27:
        cells = []
        for measure in ("h", "du"):
            for sample in SAMPLES:
                e = r.estimates.get((geo, measure, sample))
                cells.append("n/a" if e is None else _band(e.point, *e.interval()))
        lines.append(f"| {geo} | " + " | ".join(cells) + " |")
    lines += [
        "", "## Reading", "",
        "- Supported P1 and P2a would mean that, with the pandemic included, labour input in "
        "hours kept following output and the adjustment moved to hours per worker. That is "
        "the pattern job-retention schemes would produce; paid absence and closures with "
        "continued pay produce it too, and the 2021 survey redesign also falls in the "
        "window. Nothing here identifies a cause.",
        "- Hours and persons are national-accounts, domestic concepts; unemployment is a "
        "survey concept for residents.",
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


def write_report(base_dir: Path, snapshot: Path, elasticities_csv: Path, output_dir: Path,
                 report: Path | None = None) -> Study:
    st = build_study(base_dir, snapshot, elasticities_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        with (output_dir / name).open("w", encoding="utf-8", newline="\n") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write("countries.csv", country_rows(st.primary))
    write("tests.csv", [asdict(t) for t in [*st.tests, *st.amended]])
    write("sensitivity.csv", [asdict(t) for t in [*st.sensitivity, *st.sensitivity_amended]])
    summary = {"reproduced": st.reproduced, "seed": SEED, "replicates": REPLICATES,
               "permutations": PERMUTATIONS, **st.description,
               "verdicts": {t.key: t.verdict for t in st.tests},
               "verdicts_amended": {t.key: t.verdict for t in st.amended},
               "amended": st.amended_intervals, "p_floor": st.floors,
               "provenance": st.provenance}
    (output_dir / "summary.json").write_text(json.dumps(_json(summary), indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    if report is not None:
        command = (f"poetry run eu-okun hours --data-dir {base_dir.as_posix()} --snapshot "
                   f"{snapshot.as_posix()} --case-study {elasticities_csv.as_posix()} "
                   f"--output-dir {output_dir.as_posix()} --report {report.as_posix()}")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render(st, command), encoding="utf-8", newline="\n")
    return st
