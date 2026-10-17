"""Amendment 4 (``BAYES_PROTOCOL.md``, section 12): regime-specific and time-varying
coefficients.

Reads the saved fits (``fit.json``, ``draws.npz``, ``posterior.nc``) of
``pandemic_excluded`` and the five new specifications, applies the decision rules of
section 12.5 and writes the report ``docs/BAYES_REGIMES.md``. The recovery fit of
section 12.5 is the only sampling done here (``run_regime_recovery``). Every coefficient
is a descriptive association; a shift is not attributed to any cause.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.audit import audit_snapshot
from eu_okun.baseline import _read_panel
from eu_okun.bayes_artifacts import simulate_outcomes
from eu_okun.bayes_checks import (
    PRIOR_CHECK_DRAWS,
    LooComparison,
    LooResult,
    PriorPredictive,
    compare_loo,
    coverage_range,
    prior_predictive_check,
    recovery_truth,
)
from eu_okun.bayes_data import BayesData
from eu_okun.bayes_dynamics import cell_labels, common_cells, loo_on_cells
from eu_okun.bayes_fit import PROTOCOL_SETTINGS, SamplerSettings, fit_specification
from eu_okun.bayes_model import AMENDMENT_4, SPECS, ModelSpec, design_for, likelihood_mask
from eu_okun.bayes_report import PROBABILITY, LoadedFit, credible_interval, load_fit, sign_class
from eu_okun.review import read_flags
from eu_okun.stability import classical_weaker

COMPARED: tuple[str, ...] = ("pandemic_excluded", *AMENDMENT_4)
SHIFT_SPECS: tuple[str, ...] = ("regime_shift", "regime_shift_final_gdp", "regime_shift_dynamic")
# Section 12.5, in the registered order: (A, B) means "A against B".
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("regime_shift", "pandemic_excluded"),
    ("drifting_association", "pandemic_excluded"),
    ("drifting_association", "regime_shift"),
    ("regime_shift_dynamic", "pandemic_excluded_dynamic"),
)
MIN_POST_TERMS = 12          # evaluable country: at least this many post-2021 terms
PERIODS: dict[str, tuple[str, str]] = {
    "2009-2013": ("2009-Q2", "2013-Q4"),
    "2014-2019": ("2014-Q1", "2019-Q4"),
    "2022-2026": ("2022-Q3", "2026-Q2"),
}
RECOVERY_SEED = 20261021
RECOVERY_SHIFT: dict[str, float] = {"mu_delta": 0.15, "tau_delta": 0.10,
                                    "alpha_shift_sd": 0.05}
PRIOR_DRAWS = PRIOR_CHECK_DRAWS
VERDICT_SYMBOL = {"rises": "rises", "falls": "falls", "no credible change": "-",
                  "not evaluable": "n/e"}


# ----------------------------------------------------------------------------------
# Decision rules
# ----------------------------------------------------------------------------------


def shift_verdict(draws: np.ndarray) -> str:
    """Section 12.5: 'rises' / 'falls' at 0.95, otherwise 'no credible change'."""
    p_rise = float(np.mean(draws > 0))
    if p_rise >= PROBABILITY:
        return "rises"
    if p_rise <= 1.0 - PROBABILITY:
        return "falls"
    return "no credible change"


def post_terms(data: BayesData, spec: ModelSpec) -> np.ndarray:
    """Post-2021 terms in the specification's likelihood, per country."""
    used = likelihood_mask(data.mask, data.segment_start, spec)
    return np.asarray(used[:, data.post()].sum(axis=1), dtype=int)


@dataclass(frozen=True)
class Summary3:
    """Median and equal-tailed 90% interval."""

    median: float
    low: float
    high: float

    @classmethod
    def of(cls, values: np.ndarray) -> Summary3:
        low, high = credible_interval(values)
        return cls(float(np.median(values)), low, high)

    def text(self) -> str:
        return f"{self.median:+.3f} [{self.low:+.3f}, {self.high:+.3f}]"


@dataclass(frozen=True)
class CountryShift:
    """The shift of one country's GDP coefficient sum in one regime specification."""

    spec: str
    geo: str
    post_terms: int
    theta_pre: Summary3
    theta_post: Summary3
    delta: Summary3
    p_rise: float
    pre_sign: str
    post_sign: str
    verdict: str                      # rises / falls / no credible change / not evaluable

    @property
    def evaluable(self) -> bool:
        return self.post_terms >= MIN_POST_TERMS


def country_shifts(fit: LoadedFit, terms: np.ndarray) -> list[CountryShift]:
    theta, delta = fit.draws["theta"], fit.draws["delta"]
    post = theta + delta
    rows = []
    for i, geo in enumerate(fit.countries):
        evaluable = int(terms[i]) >= MIN_POST_TERMS
        rows.append(CountryShift(
            fit.name, geo, int(terms[i]), Summary3.of(theta[:, i]), Summary3.of(post[:, i]),
            Summary3.of(delta[:, i]), float(np.mean(delta[:, i] > 0)),
            sign_class(theta[:, i]), sign_class(post[:, i]) if evaluable else "not evaluable",
            shift_verdict(delta[:, i]) if evaluable else "not evaluable"))
    return rows


@dataclass(frozen=True)
class AverageShift:
    """``mu_delta`` and the cross-country means of the pre- and post-2021 sums."""

    spec: str
    mu_delta: Summary3
    p_rise: float
    verdict: str
    tau_delta: Summary3
    mean_pre: Summary3                # unweighted mean over evaluable countries
    mean_post: Summary3
    long_run_factor: Summary3 | None  # median of 1 / (1 - psi) across countries (ARDL)


def average_shift(fit: LoadedFit, evaluable: Sequence[bool]) -> AverageShift:
    mu = fit.draws["mu_delta"]
    keep = np.asarray(evaluable, dtype=bool)
    theta = fit.draws["theta"][:, keep]
    post = theta + fit.draws["delta"][:, keep]
    factor = None
    if "psi" in fit.draws:
        factor = Summary3.of(np.median(1.0 / (1.0 - fit.draws["psi"]), axis=1))
    return AverageShift(fit.name, Summary3.of(mu), float(np.mean(mu > 0)), shift_verdict(mu),
                        Summary3.of(fit.draws["tau_delta"]), Summary3.of(theta.mean(axis=1)),
                        Summary3.of(post.mean(axis=1)), factor)


@dataclass(frozen=True)
class Drift:
    """The common random walk ``kappa`` of ``drifting_association``."""

    tau_kappa: Summary3
    contrasts: dict[str, tuple[Summary3, float, str]]   # name -> summary, P(>0), verdict
    path: tuple[tuple[str, float, float, float], ...]    # quarter, low, median, high


def drift_summary(fit: LoadedFit, quarters: Sequence[str]) -> Drift:
    """D1 (2014-2019 minus 2009-2013) and D2 (2022-2026 minus 2014-2019), section 12.4.

    The path is the cross-country mean sum at each quarter, mean(theta) + kappa[t].
    """
    kappa = fit.draws["kappa"]

    def period_mean(name: str) -> np.ndarray:
        first, last = PERIODS[name]
        keep = np.asarray([first <= q <= last for q in quarters], dtype=bool)
        return np.asarray(kappa[:, keep].mean(axis=1))

    contrasts = {}
    for label, (later, earlier) in (("D1", ("2014-2019", "2009-2013")),
                                    ("D2", ("2022-2026", "2014-2019"))):
        values = period_mean(later) - period_mean(earlier)
        contrasts[label] = (Summary3.of(values), float(np.mean(values > 0)),
                            shift_verdict(values))
    level = fit.draws["theta"].mean(axis=1)[:, None] + kappa
    low, mid, high = np.quantile(level, [0.05, 0.5, 0.95], axis=0)
    path = tuple((q, float(a), float(b), float(c))
                 for q, a, b, c in zip(quarters, low, mid, high, strict=True))
    return Drift(Summary3.of(fit.draws["tau_kappa"]), contrasts, path)


# ----------------------------------------------------------------------------------
# Recovery (section 12.5)
# ----------------------------------------------------------------------------------


def regime_recovery_truth(spec: ModelSpec, data: BayesData, seed: int = RECOVERY_SEED,
                          ) -> dict[str, np.ndarray]:
    """Generating values: section 9.4's, plus the known shifts of section 12.5."""
    truth = {k: v[0] for k, v in recovery_truth(spec, data, seed).items()}
    rng = np.random.default_rng(seed + 2)
    n = data.n_countries
    truth["delta"] = (RECOVERY_SHIFT["mu_delta"]
                      + RECOVERY_SHIFT["tau_delta"] * np.asarray(rng.standard_normal(n)))
    truth["alpha_shift"] = np.asarray(rng.normal(0.0, RECOVERY_SHIFT["alpha_shift_sd"], n))
    truth["log_sigma_shift"] = np.asarray(0.0)
    truth["theta_post"] = truth["theta"] + truth["delta"]
    return {k: np.asarray(v)[None] for k, v in truth.items()}


def regime_recovery_data(spec: ModelSpec, data: BayesData, seed: int = RECOVERY_SEED,
                         ) -> tuple[BayesData, dict[str, np.ndarray]]:
    truth = regime_recovery_truth(spec, data, seed)
    y = simulate_outcomes(spec, data, truth, np.random.default_rng(seed + 1))[0]
    return replace(data, y=y), {k: v[0] for k, v in truth.items()}


@dataclass(frozen=True)
class RegimeRecovery:
    """Recovery of known shifts by one fit to synthetic data."""

    accepted: bool
    max_rhat: float
    delta_inside_90: int
    delta_total: int
    delta_inside_50: int
    theta_inside_90: int
    mu_delta: tuple[float, float, float, float, float]   # truth, low, median, high, P(>0)
    realized_mean_delta: float
    detected: bool                                        # average-shift rule says "rises"


def assess_regime_recovery(samples: Mapping[str, np.ndarray], truth: Mapping[str, np.ndarray],
                           evaluable: Sequence[bool], accepted: bool,
                           max_rhat: float) -> RegimeRecovery:
    keep = np.asarray(evaluable, dtype=bool)
    delta = samples["delta"].reshape(-1, keep.size)[:, keep]
    lo50, hi50, lo90, hi90 = np.quantile(delta, [0.25, 0.75, 0.05, 0.95], axis=0)
    true_delta = np.asarray(truth["delta"])[keep]
    theta = samples["theta"].reshape(-1, keep.size)
    t_lo, t_hi = np.quantile(theta, [0.05, 0.95], axis=0)
    mu = samples["mu_delta"].reshape(-1)
    low, high = credible_interval(mu)
    return RegimeRecovery(
        accepted, max_rhat,
        int(np.sum((true_delta >= lo90) & (true_delta <= hi90))), int(keep.sum()),
        int(np.sum((true_delta >= lo50) & (true_delta <= hi50))),
        int(np.sum((truth["theta"] >= t_lo) & (truth["theta"] <= t_hi))),
        (RECOVERY_SHIFT["mu_delta"], low, float(np.median(mu)), high, float(np.mean(mu > 0))),
        float(np.asarray(truth["delta"]).mean()), shift_verdict(mu) == "rises")


def run_regime_recovery(data_dir: Path, output_dir: Path,
                        settings: SamplerSettings = PROTOCOL_SETTINGS,
                        retry_warmup: int | None = 3000) -> RegimeRecovery:
    """Fit ``regime_shift`` to its synthetic data set and save ``recovery.json``."""
    audit_snapshot(data_dir)
    csv_path = data_dir / "processed" / "eu27_quarterly.csv"
    spec = SPECS["regime_shift"]
    design = design_for(spec, _read_panel(csv_path), read_flags(csv_path))
    data, truth = regime_recovery_data(spec, design)
    fit = fit_specification(data, spec, settings)
    if not fit.diagnostics.accepted and retry_warmup is not None:
        fit = fit_specification(data, spec, replace(settings, warmup=retry_warmup))
    evaluable = [bool(v) for v in post_terms(data, spec) >= MIN_POST_TERMS]
    result = assess_regime_recovery(fit.samples, truth, evaluable, fit.diagnostics.accepted,
                                    fit.diagnostics.max_rhat)
    target = output_dir / spec.name
    target.mkdir(parents=True, exist_ok=True)
    (target / "recovery.json").write_text(json.dumps({
        **asdict(result), "settings": asdict(fit.settings), "seed": RECOVERY_SEED,
        "shift_truth": RECOVERY_SHIFT, "runtime_seconds": fit.runtime_seconds,
    }, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return result


def load_regime_recovery(directory: Path) -> RegimeRecovery | None:
    path = directory / "regime_shift" / "recovery.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return RegimeRecovery(d["accepted"], d["max_rhat"], d["delta_inside_90"], d["delta_total"],
                          d["delta_inside_50"], d["theta_inside_90"], tuple(d["mu_delta"]),
                          d["realized_mean_delta"], d["detected"])


# ----------------------------------------------------------------------------------
# Review
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimesReplication:
    label: str
    summary: dict[str, Any]          # regimes.json of the other run


def load_regimes_replication(directory: Path) -> RegimesReplication:
    summary = json.loads((directory / "regimes.json").read_text(encoding="utf-8"))
    readme = directory / "README.md"
    label = (readme.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
             if readme.exists() else directory.name)
    return RegimesReplication(label, summary)


@dataclass
class RegimesReview:
    """Everything the report is generated from."""

    fits: dict[str, LoadedFit]
    terms: dict[str, np.ndarray]                 # spec -> post-2021 terms per country
    shifts: dict[str, list[CountryShift]]
    averages: dict[str, AverageShift]
    drift: Drift | None
    loo: dict[tuple[str, str], tuple[LooResult, LooResult]]
    comparisons: list[LooComparison]
    prior: dict[str, PriorPredictive]
    classical: dict[str, bool]
    quarters: list[str]
    recovery: RegimeRecovery | None = None
    replication: RegimesReplication | None = None

    def accepted(self, name: str) -> bool:
        return name in self.fits and self.fits[name].accepted

    def comparison(self, a: str, b: str) -> LooComparison | None:
        return next((c for c in self.comparisons if (c.model_a, c.model_b) == (a, b)), None)


def build_regimes(fits_dir: Path, data_dir: Path, review_dir: Path,
                  recovery_dir: Path | None = None,
                  replication_dir: Path | None = None) -> RegimesReview:
    """Load the fits, apply the rules of section 12.5 and compute LOO per comparison."""
    missing = [n for n in COMPARED if not (fits_dir / n / "fit.json").exists()]
    if missing:
        raise FileNotFoundError(f"No fit for {', '.join(missing)} in {fits_dir}.")
    csv_path = data_dir / "processed" / "eu27_quarterly.csv"
    panel, flags = _read_panel(csv_path), read_flags(csv_path)
    fits = {name: load_fit(fits_dir / name) for name in COMPARED}
    terms: dict[str, np.ndarray] = {}
    prior: dict[str, PriorPredictive] = {}
    quarters: list[str] = []
    for name, fit in fits.items():
        data = design_for(SPECS[name], panel, flags)
        if fit.summary["quarters"] != list(data.quarters):
            raise ValueError(f"Snapshot does not match the quarters of fit {name}.")
        terms[name] = post_terms(data, SPECS[name])
        prior[name] = prior_predictive_check(SPECS[name], data, draws=PRIOR_DRAWS)
        quarters = list(data.quarters)
    shifts = {n: country_shifts(fits[n], terms[n]) for n in SHIFT_SPECS}
    averages = {n: average_shift(fits[n], [r.evaluable for r in shifts[n]])
                for n in SHIFT_SPECS}
    drift = drift_summary(fits["drifting_association"], quarters)
    loo: dict[tuple[str, str], tuple[LooResult, LooResult]] = {}
    comparisons = []
    for a, b in COMPARISONS:
        paths = [fits_dir / n / "posterior.nc" for n in (a, b)]
        if not (fits[a].accepted and fits[b].accepted and all(p.exists() for p in paths)):
            continue
        cells = common_cells([cell_labels(p) for p in paths])
        pair = (loo_on_cells(paths[0], a, cells), loo_on_cells(paths[1], b, cells))
        loo[(a, b)] = pair
        comparisons.append(compare_loo(*pair))
    recovery = load_regime_recovery(recovery_dir) if recovery_dir is not None else None
    replication = (load_regimes_replication(replication_dir) if replication_dir is not None
                   else None)
    return RegimesReview(fits, terms, shifts, averages, drift, loo, comparisons, prior,
                         classical_weaker(review_dir), quarters, recovery, replication)


def summary_document(review: RegimesReview) -> dict[str, Any]:
    """The machine-readable results (``regimes.json``), also used for replication."""
    specs: dict[str, Any] = {}
    for name, fit in review.fits.items():
        d = fit.summary["diagnostics"]
        entry: dict[str, Any] = {"accepted": fit.accepted, "max_rhat": d["max_rhat"],
                                 "warmup": fit.summary["settings"]["warmup"]}
        if name in review.averages:
            avg = review.averages[name]
            entry.update(mu_delta=asdict(avg.mu_delta), mu_delta_p_rise=avg.p_rise,
                         average_verdict=avg.verdict,
                         countries={r.geo: {"verdict": r.verdict, "p_rise": r.p_rise,
                                            "delta_median": r.delta.median,
                                            "post_sign": r.post_sign}
                                    for r in review.shifts[name]})
        specs[name] = entry
    drift = None
    if review.drift is not None:
        drift = {"tau_kappa": asdict(review.drift.tau_kappa),
                 **{k: {"summary": asdict(s), "p_rise": p, "verdict": v}
                    for k, (s, p, v) in review.drift.contrasts.items()}}
    return {"specifications": specs, "drift": drift,
            "comparisons": [asdict(c) for c in review.comparisons],
            "recovery": None if review.recovery is None else asdict(review.recovery)}


def changed_verdicts(review: RegimesReview, other: str) -> list[str]:
    """Verdicts of ``regime_shift`` that differ in ``other`` (countries evaluable in both)."""
    base = {r.geo: r for r in review.shifts["regime_shift"]}
    out = []
    if review.averages["regime_shift"].verdict != review.averages[other].verdict:
        out.append(f"the average shift ({review.averages['regime_shift'].verdict} -> "
                   f"{review.averages[other].verdict})")
    for row in review.shifts[other]:
        first = base[row.geo]
        if row.evaluable and first.evaluable and row.verdict != first.verdict:
            out.append(f"{row.geo} ({first.verdict} -> {row.verdict})")
    return out


def unstable_verdicts(review: RegimesReview) -> list[str]:
    """Verdicts that differ between this run and the replication, in words."""
    if review.replication is None:
        return []
    theirs = review.replication.summary
    out: list[str] = []
    for name, fit in review.fits.items():
        t = theirs["specifications"].get(name)
        if t is None:
            continue
        if fit.accepted != bool(t["accepted"]):
            out.append(f"the acceptance of `{name}`")
        if fit.accepted and name in review.averages and "average_verdict" in t:
            avg = review.averages[name]
            if avg.verdict != t["average_verdict"]:
                out.append(f"the average shift in `{name}` ({avg.verdict} here at "
                           f"P = {avg.p_rise:.3f}, {t['average_verdict']} in the replication "
                           f"at P = {t['mu_delta_p_rise']:.3f})")
            for row in review.shifts[name]:
                there = t["countries"][row.geo]
                if row.verdict != there["verdict"]:
                    out.append(f"{row.geo} in `{name}` ({row.verdict} here at "
                               f"P = {row.p_rise:.3f}, {there['verdict']} in the replication "
                               f"at P = {there['p_rise']:.3f})")
    if review.drift is not None and theirs.get("drift"):
        for label, (_, p, verdict) in review.drift.contrasts.items():
            there = theirs["drift"][label]
            if verdict != there["verdict"]:
                out.append(f"{label} ({verdict} here at P = {p:.3f}, {there['verdict']} in "
                           f"the replication at P = {there['p_rise']:.3f})")
    for c in review.comparisons:
        t = next((x for x in theirs["comparisons"]
                  if (x["model_a"], x["model_b"]) == (c.model_a, c.model_b)), None)
        if t is not None and t["verdict"] != c.verdict:
            out.append(f"PSIS-LOO `{c.model_a}` against `{c.model_b}` ({c.verdict} here, "
                       f"{c.elpd_diff:+.1f} SE {c.se_diff:.1f}; {t['verdict']} in the "
                       f"replication, {t['elpd_diff']:+.1f} SE {t['se_diff']:.1f})")
    return out


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]], align: str = "") -> list[str]:
    align = align or "l" + "r" * (len(header) - 1)
    marks = ["---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(marks) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def _names(items: Sequence[str]) -> str:
    items = list(items)
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _code(name: str) -> str:
    return f"`{name}`"


def _failure(review: RegimesReview, name: str) -> str:
    """Why a fit failed the acceptance criteria, from its ``fit.json``."""
    failures = review.fits[name].summary["diagnostics"].get("failures") or ["see fit.json"]
    return "; ".join(str(f) for f in failures)


def _verdict_count(rows: Sequence[CountryShift]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"rises": [], "falls": [], "no credible change": [],
                                 "not evaluable": []}
    for row in rows:
        out[row.verdict].append(row.geo)
    return out


def render(review: RegimesReview, assets: str, command: str) -> str:
    """Assemble the Markdown report."""
    fits = review.fits
    base = review.shifts["regime_shift"]
    counts = _verdict_count(base)
    avg = review.averages["regime_shift"]
    lines = ["# Hierarchical Bayes: regime-specific and time-varying coefficients", "",
             "Generated by `eu-okun bayes-regimes` from the frozen snapshot "
             "`snapshots/eurostat-2026-10-15`. Amendment 4 of "
             "[BAYES_PROTOCOL.md](../BAYES_PROTOCOL.md) (section 12) registered the five "
             "specifications, their estimands and every decision rule below before any of "
             "them was fitted. They are reported alongside the earlier results, which they "
             "do not replace. Every coefficient is a descriptive conditional association. "
             "A shift after 2021 may reflect measurement (definitions, provisional data), "
             "a different mix of shocks (the energy shock, inflation) or a changed "
             "labour-market response; this study cannot separate them "
             "([IDENTIFICATION_RESULTS.md](IDENTIFICATION_RESULTS.md)).", ""]

    # Summary
    lines += ["## Summary", ""]
    for name in fits:
        if not review.accepted(name):
            lines.append(f"- {_code(name)} failed the acceptance criteria after the single "
                         f"retry ({_failure(review, name)}); its results are not used.")
    lines.append(
        f"- Average shift of the GDP coefficient sum after 2021 (`regime_shift`): mu_delta "
        f"{avg.mu_delta.text()}, P(> 0) = {avg.p_rise:.3f}: **{avg.verdict}**. The mean "
        f"sum over the {len(base) - len(counts['not evaluable'])} evaluable countries moves "
        f"from {avg.mean_pre.text()} (2009-2019) to {avg.mean_post.text()} (2022-2026).")
    lines.append(
        f"- Country shifts (`regime_shift`): rises in {len(counts['rises'])} "
        f"({_names(counts['rises'])}), falls in {len(counts['falls'])} "
        f"({_names(counts['falls'])}), no credible change in "
        f"{len(counts['no credible change'])}; not evaluable: "
        f"{_names(counts['not evaluable'])} (fewer than {MIN_POST_TERMS} post-2021 terms).")
    post_negative = [r.geo for r in base if r.post_sign == "negative"]
    lines.append(f"- After 2021 the sum is credibly negative in {len(post_negative)} "
                 f"evaluable countries ({_names(post_negative)}).")
    for other, intro, label in (
            ("regime_shift_final_gdp", "Without the terms that use provisional GDP",
             "provisional GDP"),
            ("regime_shift_dynamic", "With the ARDL and AR(1) factor dynamics", "the dynamics")):
        if not review.accepted(other):
            lines.append(f"- {intro}: not available, because {_code(other)} failed the "
                         "acceptance criteria.")
            continue
        countries = [c for c in changed_verdicts(review, other)
                     if not c.startswith("the average")]
        other_avg = review.averages[other]
        lines.append(f"- {intro}: average shift {other_avg.verdict} "
                     f"(P = {other_avg.p_rise:.3f}); "
                     + ("no country verdict changes." if not countries else
                        f"country verdicts that depend on {label}: {_names(countries)}."))
    if review.drift is not None:
        d1, d2 = review.drift.contrasts["D1"], review.drift.contrasts["D2"]
        lines.append(f"- Common drift (`drifting_association`): D1 (2014-2019 minus "
                     f"2009-2013) {d1[0].text()}, {d1[2]}; D2 (2022-2026 minus 2014-2019) "
                     f"{d2[0].text()}, {d2[2]}.")
    replicated = ({(x["model_a"], x["model_b"]): x
                   for x in review.replication.summary["comparisons"]}
                  if review.replication is not None else {})
    for c in review.comparisons:
        t = replicated.get((c.model_a, c.model_b))
        lines.append(f"- PSIS-LOO {_code(c.model_a)} against {_code(c.model_b)}: "
                     f"{c.elpd_diff:+.1f} (SE {c.se_diff:.1f}), {c.verdict}"
                     f"{'' if c.reliable else ' (unreliable: Pareto k)'}"
                     + ("" if t is None or t["verdict"] == c.verdict else
                        f"; the replication gives {t['elpd_diff']:+.1f} (SE "
                        f"{t['se_diff']:.1f}), {t['verdict']}, so this verdict is not stable")
                     + ".")
    if review.recovery is not None:
        rec = review.recovery
        lo, hi = coverage_range(rec.delta_total)
        lines.append(f"- Recovery of a known average shift of +{rec.mu_delta[0]:.2f}: "
                     f"{'detected' if rec.detected else 'not detected'} (P(mu_delta > 0) = "
                     f"{rec.mu_delta[4]:.3f}); {rec.delta_inside_90} of {rec.delta_total} "
                     f"country shifts inside their 90% intervals (consistent range "
                     f"{lo}-{hi}).")
    if review.replication is not None:
        unstable = unstable_verdicts(review)
        lines.append(f"- Replication ({review.replication.label}): "
                     + ("every verdict is the same." if not unstable else
                        (f"{len(unstable)} verdicts differ and are" if len(unstable) > 1
                         else "1 verdict differs and is")
                        + " not treated as established: "
                        + "; ".join(unstable) + ". Every other verdict is the same."))
    lines.append("")

    # 1. Convergence
    lines += ["## 1. Convergence", "",
              "Same sampler, acceptance criteria and single retry as sections 4 and 6 "
              "(4 chains x 1000 draws; the retry uses 3000 warm-up iterations).", ""]
    rows = []
    for name, fit in fits.items():
        d = fit.summary["diagnostics"]
        rows.append([_code(name), "yes" if fit.accepted else "**no**",
                     f"{d['max_rhat']:.4f} ({d['max_rhat_parameter']})",
                     f"{min(d['min_ess_bulk'], d['min_ess_tail']):.0f}", d["divergences"],
                     f"{d['min_bfmi']:.2f}", fit.summary["settings"]["warmup"]])
    lines += _table(["Specification", "Accepted", "Max R-hat", "Min ESS", "Divergences",
                     "Min E-BFMI", "Warm-up"], rows, "llrrrrr")

    # 2. Prior and posterior predictive checks
    lines += ["## 2. Prior and posterior predictive checks", "",
              f"Prior predictive (section 9.3, {PRIOR_DRAWS} draws): 5%, 50% and 95% "
              "quantiles of simulated statistics; the observed values are in the last row.",
              ""]
    keys = (("mean_sd", "Mean country SD (pp)"),
            ("max_abs_change", "Largest absolute change (pp)"),
            ("lag1_autocorrelation", "Lag-1 autocorrelation"))
    rows = []
    for name in fits:
        prior = review.prior[name]
        rows.append([_code(name), *(" / ".join(f"{v:.2f}" for v in prior.quantiles[k])
                                    for k, _ in keys), f"{prior.share_above_5pp:.1%}"])
    observed = review.prior["pandemic_excluded"].observed
    rows.append(["**observed**", *(f"{observed[k]:.2f}" for k, _ in keys), ""])
    lines += _table(["Specification", *(label for _, label in keys), "Cells beyond 5 pp"],
                    rows)
    lines += ["Posterior predictive checks written with each fit (p = share of replications "
              "at or above the observed statistic; one-step coverage over the likelihood "
              "cells):", ""]
    rows = []
    for name, fit in fits.items():
        pc = fit.summary["predictive_check"]
        rows.append([_code(name), f"{pc['p_value']['lag1_autocorrelation']:.3f}",
                     f"{pc['p_value']['cross_country_correlation']:.3f}",
                     f"{pc['p_value']['mean_sd']:.3f}", f"{pc['coverage_50']:.3f}",
                     f"{pc['coverage_90']:.3f}"])
    lines += _table(["Specification", "p, lag-1 autocorr.", "p, cross-country corr.",
                     "p, mean SD", "50% coverage", "90% coverage"], rows)

    # 3. Average shift
    lines += ["## 3. The average shift", "",
              "`mu_delta` is the pooled mean of the country shifts of the GDP coefficient "
              "sum; the means of the pre- and post-2021 sums are unweighted over the "
              "evaluable countries of each specification. In `regime_shift_dynamic` the "
              "sums are short-run; the long-run shift is `delta / (1 - psi)` and has the "
              "same sign.", ""]
    rows = []
    for name in SHIFT_SPECS:
        if not review.accepted(name):
            rows.append([_code(name), "failed acceptance; not used", "-", "-", "-", "-", "-",
                         "-"])
            continue
        shift = review.averages[name]
        rows.append([_code(name), shift.mu_delta.text(), f"{shift.p_rise:.3f}", shift.verdict,
                     shift.tau_delta.text(), shift.mean_pre.text(), shift.mean_post.text(),
                     "-" if shift.long_run_factor is None
                     else f"{shift.long_run_factor.median:.2f}"])
    lines += _table(["Specification", "mu_delta", "P(> 0)", "Verdict", "tau_delta",
                     "Mean sum 2009-2019", "Mean sum 2022-2026",
                     "Median 1 / (1 - psi)"], rows, "lrrlrrrr")

    # 4. Country shifts
    lines += ["## 4. Country shifts", "",
              f"`regime_shift`, by country: post-2021 terms in the likelihood (a country is "
              f"evaluable with at least {MIN_POST_TERMS}); the 2009-2019 and 2022-2026 sums "
              "and their difference `delta` (median, 90% interval); the verdict of section "
              "12.5 ('rises': P(delta > 0) >= 0.95); the same verdict in the two "
              "sensitivity specifications (n/e: not evaluable); and the classical review's "
              "weaker-after-2021 rule, recomputed from its tables.", ""]
    final = {r.geo: r for r in review.shifts["regime_shift_final_gdp"]}
    dynamic = {r.geo: r for r in review.shifts["regime_shift_dynamic"]}

    def sensitivity(name: str, row: CountryShift, terms: bool) -> str:
        if not review.accepted(name):
            return "failed"
        return VERDICT_SYMBOL[row.verdict] + (f" ({row.post_terms})" if terms else "")

    rows = []
    for r in base:
        rows.append([r.geo, r.post_terms, r.theta_pre.text(),
                     r.theta_post.text() if r.evaluable else "n/e",
                     r.delta.text() if r.evaluable else f"({r.delta.median:+.3f}, pooled)",
                     f"{r.p_rise:.3f}" if r.evaluable else "-",
                     VERDICT_SYMBOL[r.verdict],
                     sensitivity("regime_shift_final_gdp", final[r.geo], True),
                     sensitivity("regime_shift_dynamic", dynamic[r.geo], False),
                     "weaker" if review.classical.get(r.geo) else "-"])
    lines += _table(["Country", "Post terms", "Sum 2009-2019", "Sum 2022-2026", "delta",
                     "P(delta > 0)", "Verdict", "Final GDP only (terms)", "Dynamic",
                     "Classical"], rows, "lrrrrrllll")
    classical_set = sorted(g for g, weaker in review.classical.items() if weaker)
    evaluable_classical = [g for g in classical_set
                           if next(r for r in base if r.geo == g).evaluable]
    agree = [g for g in evaluable_classical
             if next(r for r in base if r.geo == g).verdict == "rises"]
    extra = [r.geo for r in base if r.verdict == "rises" and r.geo not in classical_set]
    lines += [f"- The classical rule marks {len(classical_set)} countries "
              f"({_names(classical_set)}); {len(evaluable_classical)} of them are evaluable "
              f"here, and the shift rises credibly in {len(agree)} ({_names(agree)}). "
              f"Countries that rise here but are not in the classical set: {_names(extra)}.",
              "- The two rules answer related but different questions, on different designs:",
              "  - classical: country-by-country OLS with K = 1, a post-2021 sample of 17 "
              "quarters from 2022-Q2, no common factor and no intercept shift, judged by "
              "whether every leave-one-out post-2021 sum lies above the 2009-2019 95% "
              "interval;",
              "  - here: K = 2, a post-2021 sample of 16 quarters from 2022-Q3, partial "
              "pooling of the shifts, a common factor and post-2021 intercept and scale "
              "shifts, judged by a posterior probability.",
              "- Which of these differences drives the disagreement is not tested here.", ""]
    lines += [f"![Country sums before and after 2021]({assets}/regime_countries.png)", ""]

    # 5. Measurement and dynamics
    lines += ["## 5. Provisional GDP and dynamics", ""]
    for other, label in (("regime_shift_final_gdp", "provisional GDP"),
                         ("regime_shift_dynamic", "the dynamics")):
        if not review.accepted(other):
            lines.append(f"- {_code(other)} failed the acceptance criteria after the single "
                         f"retry ({_failure(review, other)}), so the sensitivity to {label} "
                         "cannot be judged; none of its estimates is used.")
            continue
        changed = changed_verdicts(review, other)
        evaluable_both = [r.geo for r in review.shifts[other]
                          if r.evaluable and next(b for b in base if b.geo == r.geo).evaluable]
        lines.append(f"- {_code(other)}: {len(evaluable_both)} countries evaluable in both "
                     "specifications. " + ("No verdict changes." if not changed else
                                           f"Verdicts that depend on {label}: "
                                           f"{_names(changed)}."))
    lines += ["", f"![Country shifts in the accepted regime specifications]({assets}/"
              "regime_deltas.png)", ""]

    # 6. Drift
    lines += ["## 6. A gradual or an abrupt change?", ""]
    if review.drift is not None:
        drift = review.drift
        lines += ["`drifting_association` lets the GDP coefficient sum of every country move "
                  "with a common random walk `kappa[t]`. `D1` and `D2` contrast period means "
                  "of the walk; the random walk cannot place a change inside the unobserved "
                  "2020-2022 gap.", ""]
        rows = [[label, s.text(), f"{p:.3f}", v]
                for label, (s, p, v) in drift.contrasts.items()]
        rows.insert(0, ["tau_kappa (per quarter)", drift.tau_kappa.text(), "-", "-"])
        lines += _table(["Quantity", "Median [90%]", "P(> 0)", "Verdict"], rows, "lrrl")
        lines += [f"![Cross-country mean sum over time]({assets}/drift.png)", ""]

    # 7. LOO
    lines += ["## 7. PSIS-LOO", "",
              "Each comparison scores both fits on the cells common to them; 'better "
              "supported' needs an elpd difference above twice its standard error. "
              "`regime_shift_final_gdp` has fewer cells and is not compared.", ""]
    rows = []
    for (name_a, name_b), (loo_a, loo_b) in review.loo.items():
        comp = review.comparison(name_a, name_b)
        assert comp is not None
        rows.append([_code(name_a), _code(name_b), loo_a.cells, f"{comp.elpd_diff:+.1f}",
                     f"{comp.se_diff:.1f}", comp.verdict,
                     f"{loo_a.pareto_high} / {loo_b.pareto_high}",
                     "yes" if comp.reliable else "**no**"])
    lines += _table(["A", "B", "Cells", "elpd A - B", "SE", "Verdict", "Pareto k > 0.7",
                     "Reliable"], rows, "llrrrlrl")

    # 8. Recovery
    lines += ["## 8. Recovery of a known shift", ""]
    if review.recovery is None:
        lines += ["Not run in this report.", ""]
    else:
        rec = review.recovery
        lo, hi = coverage_range(rec.delta_total)
        truth, low, mid, high, p_rise = rec.mu_delta
        lines += [f"One data set simulated from `regime_shift` on the real design (seed "
                  f"{RECOVERY_SEED}) with mu_delta = +{truth:.2f} and tau_delta = "
                  f"{RECOVERY_SHIFT['tau_delta']:.2f}; the realised mean of the 27 shifts is "
                  f"{rec.realized_mean_delta:+.3f}. Fit "
                  f"{'accepted' if rec.accepted else 'NOT accepted'} (max R-hat "
                  f"{rec.max_rhat:.4f}).", "",
                  f"- mu_delta: median {mid:+.3f}, 90% interval [{low:+.3f}, {high:+.3f}], "
                  f"P(> 0) = {p_rise:.3f}: the average-shift rule "
                  f"{'detects' if rec.detected else 'does not detect'} the known shift.",
                  f"- Country shifts: {rec.delta_inside_90} of {rec.delta_total} evaluable "
                  f"true values inside their 90% intervals (consistent with calibration: "
                  f"{lo}-{hi}), {rec.delta_inside_50} inside their 50% intervals.",
                  f"- Pre-2020 sums: {rec.theta_inside_90} of 27 inside their 90% intervals.",
                  "- One data set is a check of the machinery, not a power analysis.", ""]

    # 9. Status
    lines += ["## 9. What this changes", "",
              "The new specifications are reported alongside the earlier results and do not "
              "replace them. The pre-registered judgements of HIERARCHICAL_BAYES.md, "
              "BAYES_ROBUSTNESS.md and BAYES_DYNAMICS.md keep their own specification sets. "
              "Spain and France have no post-2021 terms (definition change), so nothing here "
              "describes them after 2021.", ""]
    if review.replication is not None:
        lines += _replication_section(review)
    lines += ["## Reproduce", "", "```bash",
              "poetry install --extras \"bayes report\"",
              "poetry run eu-okun bayes-fit --data-dir snapshots/eurostat-2026-10-15 \\",
              "  --output-dir results/bayes --specs pandemic_excluded," + ",".join(AMENDMENT_4)
              + " --retry-warmup 3000",
              "poetry run eu-okun bayes-regime-recovery --data-dir snapshots/eurostat-2026-10-15 "
              "--output-dir results/bayes-regime-recovery",
              command, "```", ""]
    return "\n".join(lines)


def _replication_section(review: RegimesReview) -> list[str]:
    rep = review.replication
    assert rep is not None
    other = rep.summary["specifications"]
    lines = ["## 10. Replication on a second platform", "",
             f"{rep.label}: the same snapshot, lock file, seeds, sampler settings and single "
             "retry. Its results are in `replication-linux/`.", ""]
    rows = []
    for name, fit in review.fits.items():
        t = other.get(name)
        if t is None:
            continue
        row = [_code(name),
               f"{'yes' if fit.accepted else 'no'} / {'yes' if t['accepted'] else 'no'}",
               f"{fit.summary['settings']['warmup']} / {t['warmup']}"]
        if fit.accepted and name in review.averages and "mu_delta" in t:
            here = review.averages[name]
            largest = max(abs(r.delta.median - t["countries"][r.geo]["delta_median"])
                          for r in review.shifts[name])
            row += [f"{here.mu_delta.median:+.3f} / {t['mu_delta']['median']:+.3f}",
                    f"{largest:.3f}"]
        else:
            row += ["-", "-"]
        rows.append(row)
    lines += _table(["Specification", "Accepted (here / there)", "Warm-up (here / there)",
                     "mu_delta median (here / there)", "Largest delta median difference"],
                    rows, "lllrr")
    unstable = unstable_verdicts(review)
    lines += ["Verdicts that differ between the two runs: "
              + ("none." if not unstable else "; ".join(unstable) + "."), ""]
    return lines


def write_figures(review: RegimesReview, directory: Path, source: str) -> None:
    """Country sums before and after 2021, shifts by specification and the drift path."""
    from eu_okun.review_figures import (
        AQUA,
        BLUE,
        MUTED,
        ORANGE,
        SURFACE,
        _finish,
        _pyplot,
    )

    plt = _pyplot()
    from matplotlib.lines import Line2D

    directory.mkdir(parents=True, exist_ok=True)
    base = review.shifts["regime_shift"]
    order = sorted(range(len(base)), key=lambda j: base[j].geo)
    ys = np.arange(len(order))[::-1]

    def interval(ax: Any, s: Summary3, y: float, colour: str, hollow: bool = False) -> None:
        ax.plot([s.low, s.high], [y, y], color=colour, lw=1.5, solid_capstyle="round",
                alpha=0.45 if hollow else 1.0)
        ax.plot([s.median], [y], marker="o", ms=5.5, color=SURFACE if hollow else colour,
                mec=colour if hollow else SURFACE, mew=1.3, lw=0)

    fig, ax = plt.subplots(figsize=(7.8, 9.4))
    for y, i in zip(ys, order, strict=True):
        r = base[i]
        interval(ax, r.theta_pre, y + 0.17, BLUE)
        interval(ax, r.theta_post, y - 0.17, ORANGE, hollow=not r.evaluable)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([base[i].geo + ("" if base[i].evaluable else " (n/e)") for i in order])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Sum of GDP coefficients, lags 0-2 (pp per pp); 90% intervals")
    ax.set_title("regime_shift: country sums 2009-2019 and 2022-2026")
    ax.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, lw=1.5, ms=6, label="2009-2019"),
        Line2D([], [], marker="o", color=ORANGE, lw=1.5, ms=6, label="2022-2026"),
        Line2D([], [], marker="o", color=ORANGE, mfc=SURFACE, lw=1.5, ms=6, alpha=0.6,
               label="2022-2026, not evaluable (pooled)"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.07), fontsize=8.5, ncol=2)
    _finish(plt, fig, directory / "regime_countries.png", source, credit_y=-0.1)

    fig, ax = plt.subplots(figsize=(7.8, 9.4))
    colours = {"regime_shift": BLUE, "regime_shift_final_gdp": ORANGE,
               "regime_shift_dynamic": AQUA}
    shown = [n for n in SHIFT_SPECS if review.accepted(n)]
    for name, offset in zip(shown, np.linspace(0.24, -0.24, len(shown)), strict=True):
        rows = review.shifts[name]
        for y, i in zip(ys, order, strict=True):
            interval(ax, rows[i].delta, y + offset, colours[name], hollow=not rows[i].evaluable)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([base[i].geo for i in order])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("delta: change of the sum after 2021 (pp per pp); 90% intervals")
    ax.set_title("Country shifts in the accepted regime specifications (hollow: not "
                 "evaluable)")
    ax.legend(handles=[Line2D([], [], marker="o", color=colours[n], lw=1.5, ms=6, label=n)
                       for n in shown], loc="upper left", bbox_to_anchor=(0.0, -0.07),
              fontsize=8.5, ncol=2)
    _finish(plt, fig, directory / "regime_deltas.png", source, credit_y=-0.1)

    if review.drift is not None:
        path = review.drift.path
        quarters = [p[0] for p in path]
        x = np.arange(len(path))
        low = np.asarray([p[1] for p in path])
        mid = np.asarray([p[2] for p in path])
        high = np.asarray([p[3] for p in path])
        post = np.asarray([q >= "2022" for q in quarters], dtype=bool)
        fig, ax = plt.subplots(figsize=(10.4, 4.6))
        for keep in (~post, post):
            ax.fill_between(x[keep], low[keep], high[keep], color=BLUE, alpha=0.2, lw=0)
            ax.plot(x[keep], mid[keep], color=BLUE, lw=2)
        gap = int(np.argmax(post))
        ax.axvline(gap - 0.5, color=MUTED, lw=1, ls="--")
        ax.text(gap - 0.4, ax.get_ylim()[1], " 2020-2022 not sampled", color=MUTED,
                fontsize=8.5, va="top")
        ax.axhline(0, color=MUTED, lw=1)
        ticks = [j for j, q in enumerate(quarters) if q.endswith("Q1") or j == 0]
        ax.set_xticks(ticks)
        ax.set_xticklabels([quarters[j] for j in ticks], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean sum over countries (pp per pp)")
        ax.set_title("drifting_association: cross-country mean of theta[i] + kappa[t], "
                     "median and 90% interval")
        _finish(plt, fig, directory / "drift.png", source, credit_y=-0.08)


def write_regimes_report(fits_dir: Path, data_dir: Path, output_dir: Path,
                         review_dir: Path = Path("docs/review"), report: Path | None = None,
                         recovery_dir: Path | None = None, figures: bool = True,
                         replication_dir: Path | None = None) -> RegimesReview:
    """Build the review, write ``regimes.json``, the figures and the Markdown report."""
    review = build_regimes(fits_dir, data_dir, review_dir, recovery_dir, replication_dir)
    report = report or output_dir / "BAYES_REGIMES.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "regimes.json").write_text(
        json.dumps(summary_document(review), indent=1, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")
    if figures:
        source = ("Source: Eurostat, snapshot 2026-10-15; hierarchical Bayesian model "
                  "(BAYES_PROTOCOL.md, amendment 4). Descriptive, not causal.")
        write_figures(review, output_dir, source)
    assets = Path(os.path.relpath(output_dir, report.parent)).as_posix()
    command = (f"poetry run eu-okun bayes-regimes --fits-dir {fits_dir.as_posix()} "
               f"--data-dir {data_dir.as_posix()} --output-dir {output_dir.as_posix()} "
               f"--report {report.as_posix()}"
               + (f" --recovery-dir {recovery_dir.as_posix()}" if recovery_dir else "")
               + (f" --replication-dir {replication_dir.as_posix()}"
                  if replication_dir else ""))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(review, assets, command), encoding="utf-8", newline="\n")
    return review

