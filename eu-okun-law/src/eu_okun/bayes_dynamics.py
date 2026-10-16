"""Amendment 3 (``BAYES_PROTOCOL.md``, section 11): dynamics that address the misfits.

Reads the saved fits (``fit.json``, ``draws.npz``, ``posterior.nc``) of ``primary`` and
the three new specifications, applies the decision rules of section 11.4 and writes the
report ``docs/BAYES_DYNAMICS.md``. Fitting happens in ``bayes_fit``. The new
specifications are reported alongside the pre-registered results and never replace them.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.baseline import _read_panel
from eu_okun.bayes_checks import (
    PARETO_K_MAX,
    PRIOR_CHECK_DRAWS,
    LooComparison,
    LooResult,
    PriorPredictive,
    compare_loo,
    prior_predictive_check,
)
from eu_okun.bayes_data import build_bayes_data
from eu_okun.bayes_model import AMENDMENT_3, SPECS
from eu_okun.bayes_report import (
    PROBABILITY,
    Claim,
    FactorCheck,
    LoadedFit,
    credible_interval,
    evaluate_claims,
    factor_checks,
    load_fit,
    sign_class,
)
from eu_okun.review import read_flags

COMPARED: tuple[str, ...] = ("primary", *AMENDMENT_3)
# Section 11.4, in the registered order: (A, B) means "A against B".
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("dynamic_factor", "primary"),
    ("ardl", "primary"),
    ("ardl_dynamic_factor", "primary"),
    ("ardl_dynamic_factor", "ardl"),
)
PPC_LOW, PPC_HIGH = 0.05, 0.95       # the persistence misfit is fixed inside this range
RHO_PROBABILITY = 0.90               # interval of rho_f that must exclude zero
PRIOR_DRAWS = PRIOR_CHECK_DRAWS      # prior predictive check of section 9.3, same seed
STATISTICS = ("lag1_autocorrelation", "cross_country_correlation", "mean_sd")
LABELS = {"primary": "primary", "dynamic_factor": "AR(1) factor",
          "ardl": "ARDL", "ardl_dynamic_factor": "ARDL + AR(1) factor"}


# ----------------------------------------------------------------------------------
# Estimands and decision rules
# ----------------------------------------------------------------------------------


def total_association(fit: LoadedFit) -> tuple[str, np.ndarray]:
    """The estimand comparable with ``theta`` of ``primary`` (section 11.3), (draws, N).

    ``theta`` in the specifications without a lagged change; in the ARDL specifications
    the long-run multiplier ``theta_LR = theta / (1 - psi)``, from the joint draws.
    """
    if SPECS[fit.name].lagged_outcome:
        return "theta_LR", fit.draws["theta"] / (1.0 - fit.draws["psi"])
    return "theta", fit.draws["theta"]


@dataclass(frozen=True)
class Persistence:
    """The posterior predictive check of the lag-1 autocorrelation and its companions."""

    spec: str
    observed: float
    replicated_mean: float
    p_value: float
    coverage_50: float
    coverage_90: float
    other_p: dict[str, float]

    @property
    def fixed(self) -> bool:
        """Section 11.4: the p-value lies in [0.05, 0.95]."""
        return PPC_LOW <= self.p_value <= PPC_HIGH


def persistence(summary: Mapping[str, Any], spec: str) -> Persistence:
    """Read the check written with the fit (``fit.json``, ``predictive_check``)."""
    check = summary["predictive_check"]
    key = "lag1_autocorrelation"
    return Persistence(spec, float(check["observed"][key]), float(check["replicated_mean"][key]),
                       float(check["p_value"][key]), float(check["coverage_50"]),
                       float(check["coverage_90"]),
                       {k: float(check["p_value"][k]) for k in STATISTICS if k != key})


@dataclass(frozen=True)
class FactorDynamics:
    """``rho_f`` where the factor is dynamic, and the checks of the posterior factor."""

    spec: str
    modelled: bool
    rho_median: float
    rho_low: float
    rho_high: float
    check: FactorCheck

    @property
    def supported(self) -> bool:
        """Section 11.4: the 90% interval of ``rho_f`` excludes zero."""
        return self.modelled and (self.rho_low > 0.0 or self.rho_high < 0.0)


def factor_dynamics(fit: LoadedFit, mean_gdp: np.ndarray, mean_y: np.ndarray,
                    segment_start: np.ndarray) -> FactorDynamics:
    check = factor_checks(fit, mean_gdp, mean_y, segment_start)
    if "rho_f" not in fit.draws:
        return FactorDynamics(fit.name, False, math.nan, math.nan, math.nan, check)
    rho = fit.draws["rho_f"]
    low, high = credible_interval(rho, RHO_PROBABILITY)
    return FactorDynamics(fit.name, True, float(np.median(rho)), low, high, check)


@dataclass(frozen=True)
class CountryEstimate:
    """The comparable total association of one country in one fit."""

    spec: str
    geo: str
    estimand: str
    median: float
    low: float
    high: float
    p_negative: float
    sign: str
    psi_median: float            # NaN without a lagged change
    delayed_share_lr: float      # median of (theta_LR - beta0) / theta_LR; NaN otherwise


def country_estimates(fit: LoadedFit) -> list[CountryEstimate]:
    estimand, total = total_association(fit)
    psi = fit.draws.get("psi")
    rows = []
    for i, geo in enumerate(fit.countries):
        low, high = credible_interval(total[:, i])
        share = math.nan
        if psi is not None:
            share = float(np.median((total[:, i] - fit.draws["beta0"][:, i]) / total[:, i]))
        rows.append(CountryEstimate(
            fit.name, geo, estimand, float(np.median(total[:, i])), low, high,
            float(np.mean(total[:, i] < 0)), sign_class(total[:, i]),
            math.nan if psi is None else float(np.median(psi[:, i])), share))
    return rows


def dynamic_claims(fit: LoadedFit) -> list[Claim]:
    """The three earlier claims; Luxembourg and Spain on ``theta_LR`` in the ARDL fits.

    The ARDL reference for Luxembourg and Spain is the per-draw unweighted mean of
    ``theta_LR`` over the countries (protocol change log); the Netherlands rule keeps
    ``theta`` and ``beta0``.
    """
    claims = evaluate_claims(fit)
    if not SPECS[fit.name].lagged_outcome:
        return claims
    _, total = total_association(fit)
    long_run = evaluate_claims(LoadedFit(fit.name, fit.summary,
                                         {"theta": total, "beta0": fit.draws["beta0"]}))

    def relabel(claim: Claim, geo: str) -> Claim:
        detail = (claim.detail.replace(f"theta_{geo}", f"theta_LR_{geo}")
                  .replace("mu_theta", "mean theta_LR"))
        return replace(claim, detail=detail)

    return [relabel(long_run[0], "LU"), claims[1], relabel(long_run[2], "ES")]


def mean_association(fit: LoadedFit) -> tuple[float, float, float]:
    """Median and 90% interval of the per-draw unweighted mean of the estimand."""
    _, total = total_association(fit)
    means = total.mean(axis=1)
    low, high = credible_interval(means)
    return float(np.median(means)), low, high


def reference_check(fit: LoadedFit) -> dict[str, float]:
    """Descriptive check of the claim references (not a rule).

    Luxembourg and Spain against the per-draw mean of the country estimates, and the
    short-run ``theta`` against ``mu_theta`` as in section 7, in the same fit: this
    separates the effect of the reference from that of the estimand.
    """
    index = {geo: i for i, geo in enumerate(fit.countries)}
    _, total = total_association(fit)
    mean = total.mean(axis=1)
    theta, mu = fit.draws["theta"], fit.mu_theta()
    return {
        "lu_vs_mean": float(np.mean(total[:, index["LU"]] > mean)),
        "lu_short_run_vs_mu": float(np.mean(theta[:, index["LU"]] > mu)),
        "es_vs_mean": float(np.mean(total[:, index["ES"]] < mean)),
        "es_short_run_vs_mu": float(np.mean(theta[:, index["ES"]] < mu)),
    }


def claim_status(claim: Claim) -> str:
    if not claim.evaluable:
        return "not evaluable"
    if claim.holds:
        return "holds"
    return "contradicted" if claim.contradicted else "neither"


# ----------------------------------------------------------------------------------
# PSIS-LOO on the cells common to the compared fits
# ----------------------------------------------------------------------------------


def cell_labels(path: Path) -> list[str]:
    """Cell labels of the ``log_likelihood`` group of a NetCDF artifact."""
    import xarray as xr

    with xr.open_datatree(path, engine="h5netcdf") as tree:
        return [str(c) for c in tree["log_likelihood"].to_dataset()["cell"].to_numpy()]


def common_cells(labels: Sequence[Sequence[str]]) -> list[str]:
    """Cells present in every list, in the order of the first."""
    if not labels:
        return []
    shared = set(labels[0]).intersection(*map(set, labels[1:]))
    return [cell for cell in labels[0] if cell in shared]


def loo_on_cells(path: Path, spec: str, cells: Sequence[str]) -> LooResult:
    """PSIS-LOO of one fit restricted to ``cells`` (pointwise values in that order)."""
    import xarray as xr
    from arviz_stats.loo.loo import loo

    with xr.open_datatree(path, engine="h5netcdf") as tree:
        subset = xr.DataTree.from_dict({
            "posterior": tree["posterior"].to_dataset(),
            "log_likelihood": tree["log_likelihood"].to_dataset().sel(cell=list(cells)),
        })
        # ArviZ's loo carries no annotations.
        result: Any = loo(subset, pointwise=True, var_name="y")  # type: ignore[no-untyped-call]
        elpd_i = np.asarray(result.elpd_i).reshape(-1)
        pareto_k = np.asarray(result.pareto_k).reshape(-1)
        return LooResult(spec, float(result.elpd), float(result.se), float(result.p),
                         int(elpd_i.size), int(np.sum(pareto_k > PARETO_K_MAX)),
                         tuple(float(v) for v in elpd_i))


# ----------------------------------------------------------------------------------
# Review
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicsReplication:
    """The same analysis from an independent run (e.g. on another platform)."""

    label: str
    summary: dict[str, Any]          # dynamics.json of that run


def load_dynamics_replication(directory: Path) -> DynamicsReplication:
    summary = json.loads((directory / "dynamics.json").read_text(encoding="utf-8"))
    readme = directory / "README.md"
    label = (readme.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
             if readme.exists() else directory.name)
    return DynamicsReplication(label, summary)


@dataclass
class DynamicsReview:
    """Everything the report is generated from."""

    fits: dict[str, LoadedFit]                 # the compared specifications
    persistence: dict[str, Persistence]        # every fit found, for context
    factor: dict[str, FactorDynamics]
    estimates: dict[str, list[CountryEstimate]]
    claims: dict[str, list[Claim]]
    loo: dict[str, LooResult]
    comparisons: list[LooComparison]
    mean_gdp: np.ndarray
    quarters: list[str]
    prior: dict[str, PriorPredictive]
    replication: DynamicsReplication | None = None

    def accepted(self, name: str) -> bool:
        return name in self.fits and self.fits[name].accepted

    def negative_set(self, name: str) -> list[str]:
        return [row.geo for row in self.estimates[name] if row.sign == "negative"]

    def comparison(self, a: str, b: str) -> LooComparison | None:
        return next((c for c in self.comparisons if (c.model_a, c.model_b) == (a, b)), None)


def build_dynamics(fits_dir: Path, data_dir: Path,
                   replication_dir: Path | None = None) -> DynamicsReview:
    """Load the fits, apply the rules of section 11.4 and compute LOO on common cells."""
    missing = [n for n in COMPARED if not (fits_dir / n / "fit.json").exists()]
    if missing:
        raise FileNotFoundError(f"No fit for {', '.join(missing)} in {fits_dir}.")
    fits = {name: load_fit(fits_dir / name) for name in COMPARED}
    context = {name: persistence(json.loads((fits_dir / name / "fit.json").read_text(
        encoding="utf-8")), name) for name in SPECS if (fits_dir / name / "fit.json").exists()}
    csv_path = data_dir / "processed" / "eu27_quarterly.csv"
    primary = SPECS["primary"]
    data = build_bayes_data(_read_panel(csv_path), read_flags(csv_path), primary.windows,
                            primary.lags)
    for name, fit in fits.items():
        if fit.summary["quarters"] != list(data.quarters):
            raise ValueError(f"Snapshot does not match the quarters of fit {name}.")
    mean_gdp, mean_y = data.g.mean(axis=0), data.y.mean(axis=0)
    factor = {name: factor_dynamics(fit, mean_gdp, mean_y, data.segment_start)
              for name, fit in fits.items()}
    paths = {name: fits_dir / name / "posterior.nc" for name, fit in fits.items()
             if fit.accepted and (fits_dir / name / "posterior.nc").exists()}
    cells = common_cells([cell_labels(path) for path in paths.values()])
    loo = {name: loo_on_cells(path, name, cells) for name, path in paths.items()}
    comparisons = [compare_loo(loo[a], loo[b]) for a, b in COMPARISONS if a in loo and b in loo]
    prior = {name: prior_predictive_check(SPECS[name], data, draws=PRIOR_DRAWS)
             for name in COMPARED}
    replication = (load_dynamics_replication(replication_dir) if replication_dir is not None
                   else None)
    return DynamicsReview(
        fits, context, factor, {name: country_estimates(fit) for name, fit in fits.items()},
        {name: dynamic_claims(fit) for name, fit in fits.items()}, loo, comparisons,
        mean_gdp, list(data.quarters), prior, replication)


def summary_document(review: DynamicsReview) -> dict[str, Any]:
    """The machine-readable results (``dynamics.json``), also used for replication."""
    out: dict[str, Any] = {"specifications": {}, "comparisons": [asdict(c) for c in
                                                                 review.comparisons]}
    for name, fit in review.fits.items():
        p, f = review.persistence[name], review.factor[name]
        diagnostics = fit.summary["diagnostics"]
        out["specifications"][name] = {
            "accepted": fit.accepted, "max_rhat": diagnostics["max_rhat"],
            "warmup": fit.summary["settings"]["warmup"],
            "prior_share_above_5pp": review.prior[name].share_above_5pp,
            "lag1_p_value": p.p_value, "lag1_replicated_mean": p.replicated_mean,
            "persistence_fixed": p.fixed,
            "rho_f": None if not f.modelled else [f.rho_median, f.rho_low, f.rho_high],
            "factor_lag1": f.check.lag1_autocorrelation,
            "factor_corr_mean_gdp": f.check.corr_mean_gdp,
            "estimand": review.estimates[name][0].estimand,
            "medians": {r.geo: r.median for r in review.estimates[name]},
            "credibly_negative": review.negative_set(name),
            "claims": {c.claim: claim_status(c) for c in review.claims[name]},
            "loo": None if name not in review.loo else {
                k: v for k, v in asdict(review.loo[name]).items() if k != "pointwise"},
        }
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


def _estimate(row: CountryEstimate) -> str:
    mark = {"negative": "", "positive": " (+)", "uncertain": " (?)"}[row.sign]
    return f"{row.median:+.3f} [{row.low:+.3f}, {row.high:+.3f}]{mark}"


def _replicated(review: DynamicsReview, name: str) -> dict[str, Any] | None:
    if review.replication is None:
        return None
    found: dict[str, Any] | None = review.replication.summary["specifications"].get(name)
    return found


def _replicated_comparison(review: DynamicsReview, a: str, b: str) -> dict[str, Any] | None:
    if review.replication is None:
        return None
    return next((c for c in review.replication.summary["comparisons"]
                 if (c["model_a"], c["model_b"]) == (a, b)), None)


def unstable_verdicts(review: DynamicsReview) -> list[str]:
    """Verdicts that differ between this run and the replication, in words."""
    out: list[str] = []
    for name, fit in review.fits.items():
        theirs = _replicated(review, name)
        if theirs is None:
            continue
        if fit.accepted != bool(theirs["accepted"]):
            out.append(f"the acceptance of {_code(name)}")
        p, f = review.persistence[name], review.factor[name]
        if p.fixed != bool(theirs["persistence_fixed"]):
            out.append(f"the persistence verdict of {_code(name)} (p = {p.p_value:.3f} here, "
                       f"{theirs['lag1_p_value']:.3f} in the replication)")
        if f.modelled and theirs["rho_f"] is not None:
            _, low, high = theirs["rho_f"]
            if f.supported != (low > 0.0 or high < 0.0):
                out.append(f"the factor-dynamics verdict of {_code(name)}")
        if set(review.negative_set(name)) != set(theirs["credibly_negative"]):
            out.append(f"the credibly negative set of {_code(name)}")
        for claim in review.claims[name]:
            there = theirs["claims"].get(claim.claim)
            if there is not None and claim_status(claim) != there:
                out.append(f'"{claim.claim}" in {_code(name)} ({claim_status(claim)} here at '
                           f"{claim.probability:.3f}, {there} in the replication)")
    for c in review.comparisons:
        t = _replicated_comparison(review, c.model_a, c.model_b)
        if t is not None and t["verdict"] != c.verdict:
            out.append(f"PSIS-LOO {_code(c.model_a)} against {_code(c.model_b)} ({c.verdict} "
                       f"here, {c.elpd_diff:+.1f} SE {c.se_diff:.1f}; {t['verdict']} in the "
                       f"replication, {t['elpd_diff']:+.1f} SE {t['se_diff']:.1f})")
    return out


def _verdict_sentence(review: DynamicsReview, name: str) -> str:
    """Section 11.4, conclusions: what the specification fixes and how LOO judges it.

    A verdict that the replication does not reproduce is marked as not stable.
    """
    if not review.accepted(name):
        return (f"{_code(name)} failed the acceptance criteria, so none of its results is "
                "used.")
    p, f = review.persistence[name], review.factor[name]
    parts = [("fixes the persistence misfit" if p.fixed else
              "does not fix the persistence misfit")
             + f" (p = {p.p_value:.3f})"]
    theirs = _replicated(review, name)
    if theirs is not None and p.fixed != bool(theirs["persistence_fixed"]):
        parts[0] += (f", but the replication gives p = {theirs['lag1_p_value']:.3f}, so this "
                     "verdict is not stable")
    if f.modelled:
        parts.append("its factor dynamics are " + ("supported" if f.supported else
                                                   "not supported")
                     + f" (rho_f {f.rho_median:+.2f}, 90% [{f.rho_low:+.2f}, {f.rho_high:+.2f}])")
    else:
        parts.append("it keeps an i.i.d. factor, whose posterior lag-1 autocorrelation is "
                     f"{f.check.lag1_autocorrelation:+.2f}")
    against = review.comparison(name, "primary")
    if against is not None:
        if against.verdict == f"{name} better supported":
            parts.append("it is better supported than `primary` by PSIS-LOO")
        elif against.verdict == "primary better supported":
            parts.append("`primary` is better supported by PSIS-LOO")
        else:
            parts.append("PSIS-LOO does not distinguish it from `primary`")
        if not against.reliable:
            parts[-1] += " (unreliable: Pareto k)"
        t = _replicated_comparison(review, name, "primary")
        if t is not None and t["verdict"] != against.verdict:
            parts[-1] += f" (the replication: {t['verdict']}; not stable)"
    return f"{_code(name)} " + "; ".join(parts) + "."


def render(review: DynamicsReview, assets: str, command: str) -> str:
    """Assemble the Markdown report."""
    fits = review.fits
    new = [n for n in AMENDMENT_3 if n in fits]
    lines = ["# Hierarchical Bayes: dynamics that address the misfits", "",
             "Generated by `eu-okun bayes-dynamics` from the frozen snapshot "
             "`snapshots/eurostat-2026-10-15`. Amendment 3 of "
             "[BAYES_PROTOCOL.md](../BAYES_PROTOCOL.md) (section 11) registered the three "
             "specifications, their estimands and every decision rule below before any of "
             "them was fitted. They are reported **alongside** the pre-registered results "
             "in [HIERARCHICAL_BAYES.md](HIERARCHICAL_BAYES.md), which they do not replace. "
             "Every coefficient is a descriptive conditional association; no specification "
             "identifies a causal GDP shock ([IDENTIFICATION_RESULTS.md]"
             "(IDENTIFICATION_RESULTS.md)).", ""]

    lines += ["## Summary", ""]
    for name in new:
        lines.append(f"- {_verdict_sentence(review, name)}")
    against = review.comparison("ardl_dynamic_factor", "ardl")
    if against is not None:
        t = _replicated_comparison(review, "ardl_dynamic_factor", "ardl")
        lines.append(f"- `ardl_dynamic_factor` against `ardl`: elpd difference "
                     f"{against.elpd_diff:+.1f} (SE {against.se_diff:.1f}), {against.verdict}"
                     f"{'' if against.reliable else ' (unreliable: Pareto k)'}"
                     + ("" if t is None or t["verdict"] == against.verdict else
                        f"; the replication gives {t['elpd_diff']:+.1f} (SE {t['se_diff']:.1f}),"
                        f" {t['verdict']}, so this verdict is not stable") + ".")
    if review.replication is not None:
        unstable = unstable_verdicts(review)
        lines.append(f"- Replication ({review.replication.label}): "
                     + ("every verdict is the same." if not unstable else
                        f"{len(unstable)} verdicts differ and are not treated as established: "
                        + "; ".join(unstable) + ". Every other verdict is the same."))
    primary_negative = review.negative_set("primary")
    for name in new:
        if not review.accepted(name):
            continue
        negative = review.negative_set(name)
        gained = [g for g in negative if g not in primary_negative]
        lost = [g for g in primary_negative if g not in negative]
        lines.append(f"- Credibly negative in {_code(name)}: {len(negative)} of 27 countries "
                     f"(`primary`: {len(primary_negative)}); added: {_names(gained)}; "
                     f"dropped: {_names(lost)}.")
    averages = []
    for n in fits:
        mid, low, high = mean_association(fits[n])
        averages.append(f"{_code(n)} {mid:+.3f} [{low:+.3f}, {high:+.3f}]")
    lines.append("- Mean of the 27 country estimates (`theta`, or `theta_LR` in the ARDL fits; "
                 f"median, 90% interval): {'; '.join(averages)}.")
    corr = [f"{_code(n)} {review.factor[n].check.corr_mean_gdp:+.2f}" for n in fits]
    lines += [f"- The posterior factor stays correlated with mean GDP growth (median "
              f"correlation: {', '.join(corr)}), as section 11.4 anticipated: "
              "`theta` remains an association beyond the common European cycle.", ""]

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
                     f"{d['min_bfmi']:.2f}", fit.summary["settings"]["warmup"],
                     f"{fit.summary['runtime_seconds'] / 60:.1f}"])
    lines += _table(["Specification", "Accepted", "Max R-hat", "Min ESS", "Divergences",
                     "Min E-BFMI", "Warm-up", "Minutes"], rows, "llrrrrrr")

    # 2. Prior predictive
    lines += ["## 2. Prior predictive check", "",
              f"The check of section 9.3 with the new priors: {PRIOR_DRAWS} prior draws per "
              "specification, each propagated to a simulated data set on the real design "
              "(the ARDL simulations start from the observed first quarter of each segment, "
              "on which their likelihood conditions). 5%, 50% and 95% quantiles; the "
              "observed value is in the last row.", ""]
    keys = (("mean_sd", "Mean country SD (pp)"),
            ("max_abs_change", "Largest absolute change (pp)"),
            ("lag1_autocorrelation", "Lag-1 autocorrelation"),
            ("cross_country_correlation", "Cross-country correlation"))
    rows = []
    for name in fits:
        prior = review.prior[name]
        rows.append([_code(name), *(" / ".join(f"{v:.2f}" for v in prior.quantiles[k])
                                    for k, _ in keys), f"{prior.share_above_5pp:.1%}"])
    observed = review.prior["primary"].observed
    rows.append(["**observed**", *(f"{observed[k]:.2f}" for k, _ in keys), ""])
    lines += _table(["Specification", *(label for _, label in keys), "Cells beyond 5 pp"],
                    rows)

    # 3. Persistence
    lines += ["## 3. The persistence misfit", "",
              "Posterior predictive check of the mean lag-1 autocorrelation of unemployment "
              "changes (400 replicated data sets per fit; p = share of replications at or "
              "above the observed value). Fixed when p lies in [0.05, 0.95].", ""]
    rows = []
    for name in fits:
        p = review.persistence[name]
        rows.append([_code(name), f"{p.observed:.3f}", f"{p.replicated_mean:.3f}",
                     f"{p.p_value:.3f}", "fixed" if p.fixed else "not fixed",
                     f"{p.other_p['cross_country_correlation']:.3f}",
                     f"{p.other_p['mean_sd']:.3f}", f"{p.coverage_50:.3f}",
                     f"{p.coverage_90:.3f}"])
    lines += _table(["Specification", "Observed", "Replicated mean", "p", "Misfit",
                     "p, cross-country corr.", "p, mean SD", "50% coverage", "90% coverage"],
                    rows, "lrrrlrrrr")
    earlier = [p for n, p in review.persistence.items() if n not in COMPARED]
    if earlier:
        means = [p.replicated_mean for p in earlier]
        pvals = [p.p_value for p in earlier]
        lines += [f"For context, the other {len(earlier)} fitted specifications replicate a "
                  f"mean lag-1 autocorrelation of {min(means):.3f} to {max(means):.3f} "
                  f"(p = {min(pvals):.3f} to {max(pvals):.3f}). Coverage is one-step-ahead, "
                  "over the cells of each likelihood (the ARDL fits condition on the first "
                  "quarter).", ""]
    lines += [f"![Lag-1 autocorrelation, observed and replicated]({assets}/persistence.png)",
              ""]

    # 4. Factor
    lines += ["## 4. The common factor", "",
              "`rho_f` is the AR(1) coefficient of the factor (90% interval). The last "
              "columns check the posterior factor itself: its lag-1 autocorrelation and its "
              "correlation with the cross-country mean of GDP growth (medians over draws, "
              "90% intervals).", ""]
    rows = []
    for name in fits:
        f = review.factor[name]
        rho = (f"{f.rho_median:+.3f} [{f.rho_low:+.3f}, {f.rho_high:+.3f}]" if f.modelled
               else "i.i.d. (not modelled)")
        verdict = ("supported" if f.supported else "not supported") if f.modelled else "-"
        rows.append([_code(name), rho, verdict,
                     f"{f.check.lag1_autocorrelation:+.2f} [{f.check.lag1_low:+.2f}, "
                     f"{f.check.lag1_high:+.2f}]",
                     f"{f.check.corr_mean_gdp:+.2f} [{f.check.corr_mean_gdp_low:+.2f}, "
                     f"{f.check.corr_mean_gdp_high:+.2f}]"])
    lines += _table(["Specification", "rho_f", "Factor dynamics", "Factor lag-1 autocorr.",
                     "Corr. with mean GDP growth"], rows, "lrlrr")
    lines += [f"![rho_f and psi]({assets}/dynamics_parameters.png)", ""]

    # 5. LOO
    lines += ["## 5. PSIS-LOO on common cells", ""]
    if review.loo:
        cells = next(iter(review.loo.values())).cells
        lines += [f"Each accepted fit is scored on the {cells} cells that every compared "
                  "likelihood contains: the unmasked cells without each segment's first "
                  "quarter, which the ARDL fits condition on. The terms are one-step "
                  "conditional densities in every fit, so they are comparable. 'Better "
                  "supported' needs an elpd difference above twice its standard error.", ""]
        rows = [[_code(r.spec), f"{r.elpd:.1f}", f"{r.se:.1f}", f"{r.p_loo:.1f}",
                 f"{r.pareto_high}", "yes" if r.reliable else "**no**"]
                for r in review.loo.values()]
        lines += _table(["Specification", "elpd", "SE", "p_loo", f"Pareto k > {PARETO_K_MAX}",
                         "Reliable"], rows, "lrrrrl")
        rows = [[_code(c.model_a), _code(c.model_b), f"{c.elpd_diff:+.1f}",
                 f"{c.se_diff:.1f}", c.verdict, "yes" if c.reliable else "**no**"]
                for c in review.comparisons]
        lines += _table(["A", "B", "elpd A - B", "SE", "Verdict", "Reliable"], rows,
                        "llrrll")
    else:
        lines += ["No accepted fit has a NetCDF artifact, so LOO was not computed.", ""]

    # 6. Country estimates
    lines += ["## 6. Country estimates", "",
              "Posterior median and 90% interval of the total association with GDP growth, "
              "2009-2019: `theta` (sum of the GDP coefficients at lags 0-2) for `primary` "
              "and `dynamic_factor`, the long-run multiplier `theta_LR = theta / (1 - psi)` "
              "for the ARDL fits. (?) marks a sign that is not credible at 0.95 and (+) a "
              "credibly positive one. Because `1 - psi > 0`, `theta_LR` has the sign of "
              "`theta` in every draw.", ""]
    names = list(fits)
    header = ["Country", *[f"{_code(n)} ({review.estimates[n][0].estimand})" for n in names]]
    rows = []
    for i, geo in enumerate(fits["primary"].countries):
        rows.append([geo, *[_estimate(review.estimates[n][i]) for n in names]])
    lines += _table(header, rows, "l" + "r" * len(names))
    for n in new:
        pairs = zip(review.estimates["primary"], review.estimates[n], strict=True)
        base, row = max(pairs, key=lambda p: abs(p[1].median - p[0].median))
        lines.append(f"- Largest change of a median against `primary` in {_code(n)}: "
                     f"{row.geo}, {base.median:+.3f} to {row.median:+.3f}.")
    lines.append("")
    ardl = [n for n in new if SPECS[n].lagged_outcome]
    if ardl:
        lines += ["The lagged-change coefficient `psi` (median) and the descriptive long-run "
                  "delayed share `(theta_LR - beta0) / theta_LR` (median; not a decision "
                  "rule):", ""]
        rows = []
        for i, geo in enumerate(fits["primary"].countries):
            rows.append([geo, *[f"{review.estimates[n][i].psi_median:+.2f}" for n in ardl],
                         *[f"{review.estimates[n][i].delayed_share_lr:.2f}" for n in ardl]])
        lines += _table(["Country", *[f"psi, {_code(n)}" for n in ardl],
                         *[f"Delayed share, {_code(n)}" for n in ardl]], rows,
                        "l" + "r" * 2 * len(ardl))
        for n in ardl:
            mu_psi = np.tanh(fits[n].draws["mu_psi"])
            low, high = credible_interval(mu_psi)
            lines.append(f"- {_code(n)}: tanh(mu_psi) {np.median(mu_psi):+.3f} "
                         f"[{low:+.3f}, {high:+.3f}], tau_psi "
                         f"{np.median(fits[n].draws['tau_psi']):.3f}.")
        lines.append("")
    lines += [f"![Country estimates]({assets}/country_estimates.png)", ""]
    rows = [[_code(n), len(review.negative_set(n)), _names(review.negative_set(n))]
            for n in names]
    lines += ["Countries credibly negative (P(estimate < 0) >= 0.95):", ""]
    lines += _table(["Specification", "Count", "Countries"], rows, "lrl")

    # 7. Claims
    lines += ["## 7. The three earlier claims", "",
              "The rules of section 7. In the ARDL fits, Luxembourg and Spain use `theta_LR` "
              "against the per-draw mean of `theta_LR` over the 27 countries; the Netherlands "
              "rule keeps `theta` and `beta0` (protocol change log).", ""]
    rows = []
    for name in names:
        for claim in review.claims[name]:
            rows.append([claim.claim, _code(name), f"{claim.probability:.3f}",
                         claim_status(claim), claim.detail])
    lines += _table(["Claim", "Specification", "Probability", "Status", "Detail"], rows,
                    "llrll")
    checks = {n: reference_check(fits[n]) for n in names}
    lines += ["Descriptive check, not a rule: the same probabilities against the per-draw "
              "mean of the country estimates, and with the short-run `theta` against "
              "`mu_theta` (the section 7 rule) in the same fit. The first isolates the "
              "change of reference, the second the change of estimand.", ""]
    rows = [[_code(n), f"{c['lu_vs_mean']:.3f}", f"{c['lu_short_run_vs_mu']:.3f}",
             f"{c['es_vs_mean']:.3f}", f"{c['es_short_run_vs_mu']:.3f}"]
            for n, c in checks.items()]
    lines += _table(["Specification", "LU above the mean", "LU short run above mu_theta",
                     "ES below the mean", "ES short run below mu_theta"], rows, "lrrrr")
    ref = checks["primary"]
    same = (ref["lu_vs_mean"] >= PROBABILITY) == (ref["lu_short_run_vs_mu"] >= PROBABILITY)
    lines.append(f"- Reference: in `primary`, Luxembourg gives {ref['lu_vs_mean']:.3f} against "
                 f"the mean of the country estimates and {ref['lu_short_run_vs_mu']:.3f} "
                 "against `mu_theta`, so the change of reference "
                 + ("does not change the status." if same else "changes the status."))
    for n in ardl:
        c = checks[n]
        lu = review.estimates[n][fits[n].countries.index("LU")]
        positive = sum(r.psi_median > 0 for r in review.estimates[n] if r.geo != "LU")
        if c["lu_vs_mean"] >= PROBABILITY > c["lu_short_run_vs_mu"]:
            lines.append(
                f"- Estimand: in {_code(n)} the short-run `theta_LU` gives "
                f"{c['lu_short_run_vs_mu']:.3f}, so the Luxembourg claim holds because of "
                f"the long-run multiplier: Luxembourg's median `psi` is {lu.psi_median:+.2f}, "
                f"which shrinks `theta_LR` towards zero, while {positive} of the other "
                f"{len(review.estimates[n]) - 1} countries have a positive median `psi`.")
        else:
            lines.append(f"- Estimand: in {_code(n)} the short-run `theta_LU` gives "
                         f"{c['lu_short_run_vs_mu']:.3f} and `theta_LR` "
                         f"{c['lu_vs_mean']:.3f}.")
    lines.append("")

    # 8. Status
    lines += ["## 8. What this changes", "",
              "Section 11.4 fixes the status of these specifications: a fit that fixes "
              "the misfits and is better supported is described as such, and it does not "
              "retroactively replace the primary specification of sections 3-7. The "
              "robustness and claim judgements of HIERARCHICAL_BAYES.md and "
              "BAYES_ROBUSTNESS.md keep their own specification sets.", ""]
    for name in new:
        lines.append(f"- {_verdict_sentence(review, name)}")
    if review.replication is not None and unstable_verdicts(review):
        lines.append("- A verdict that the replication does not reproduce sits on its "
                     "threshold; it is reported with both values and is not treated as "
                     "established.")
    lines.append("")

    if review.replication is not None:
        lines += _replication_section(review)

    lines += ["## Reproduce", "", "```bash",
              "poetry install --extras \"bayes report\"",
              "poetry run eu-okun bayes-fit --data-dir snapshots/eurostat-2026-10-15 \\",
              "  --output-dir results/bayes --specs primary,dynamic_factor,ardl,"
              "ardl_dynamic_factor --retry-warmup 3000",
              command, "```", ""]
    return "\n".join(lines)


def _replication_section(review: DynamicsReview) -> list[str]:
    rep = review.replication
    assert rep is not None
    other = rep.summary["specifications"]
    lines = ["## 9. Replication on a second platform", "",
             f"{rep.label}: the same snapshot, lock file, seeds, sampler settings and single "
             "retry. Its results are in `replication-linux/`.", ""]
    rows = []
    for name, fit in review.fits.items():
        if name not in other:
            continue
        theirs = other[name]
        p = review.persistence[name]
        largest = max(abs(r.median - theirs["medians"][r.geo]) for r in review.estimates[name])
        same_set = set(review.negative_set(name)) == set(theirs["credibly_negative"])
        same_claims = all(claim_status(c) == theirs["claims"][c.claim]
                          for c in review.claims[name])
        rows.append([_code(name),
                     f"{'yes' if fit.accepted else 'no'} / {'yes' if theirs['accepted'] else 'no'}",
                     f"{fit.summary['settings']['warmup']} / {theirs['warmup']}",
                     f"{p.p_value:.3f} / {theirs['lag1_p_value']:.3f}",
                     f"{largest:.3f}", "yes" if same_set else "**no**",
                     "yes" if same_claims else "**no**"])
    lines += _table(["Specification", "Accepted (here / there)", "Warm-up (here / there)",
                     "Lag-1 p (here / there)", "Largest median difference",
                     "Same credibly negative set", "Same claim statuses"], rows, "lllrrll")
    lines.append("A warm-up of 3000 means the first attempt failed the acceptance criteria "
                 "and the single retry of section 6 was used.")
    lines.append("")
    theirs_c = {(c["model_a"], c["model_b"]): c for c in rep.summary["comparisons"]}
    for c in review.comparisons:
        t = theirs_c.get((c.model_a, c.model_b))
        if t is not None:
            lines.append(f"- LOO `{c.model_a}` vs `{c.model_b}`: {c.elpd_diff:+.1f} "
                         f"({c.se_diff:.1f}), {c.verdict} here; {t['elpd_diff']:+.1f} "
                         f"({t['se_diff']:.1f}), {t['verdict']} there.")
    unstable = unstable_verdicts(review)
    lines += ["", "Verdicts that differ between the two runs: "
              + ("none." if not unstable else "; ".join(unstable) + ".")]
    return lines + [""]


def write_figures(review: DynamicsReview, directory: Path, source: str) -> None:
    """Persistence check, rho_f and psi, and the country estimates."""
    from eu_okun.review_figures import (
        AQUA,
        BLUE,
        MUTED,
        NEUTRAL,
        ORANGE,
        SURFACE,
        _finish,
        _pyplot,
    )

    plt = _pyplot()
    from matplotlib.lines import Line2D

    colours = {"primary": NEUTRAL, "dynamic_factor": BLUE, "ardl": ORANGE,
               "ardl_dynamic_factor": AQUA}
    directory.mkdir(parents=True, exist_ok=True)

    # Persistence: replicated mean of every fit against the observed value.
    items = sorted(review.persistence.values(), key=lambda p: (p.spec in COMPARED, p.spec))
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(items) + 1.6))
    ys = np.arange(len(items))
    for y, p in zip(ys, items, strict=True):
        if p.spec in colours:
            ax.plot([p.replicated_mean], [y], marker="o", ms=7, color=colours[p.spec],
                    mec=SURFACE, mew=1.2, lw=0)
        else:  # earlier specifications, for context: hollow markers
            ax.plot([p.replicated_mean], [y], marker="o", ms=6.5, color=SURFACE, mec=MUTED,
                    mew=1.4, lw=0)
    observed = items[0].observed
    ax.axvline(observed, color=MUTED, lw=1.4, ls="--")
    ax.text(observed, len(items) - 0.4, f" observed {observed:.2f}", color=MUTED, fontsize=8.5,
            va="bottom")
    ax.set_yticks(ys)
    ax.set_yticklabels([p.spec for p in items], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(items) + 0.2)
    ax.set_xlabel("Mean lag-1 autocorrelation of unemployment changes (replicated mean)")
    ax.set_title("Persistence: replicated against observed, every fitted specification")
    _finish(plt, fig, directory / "persistence.png", source, credit_y=-0.04)

    # rho_f and psi.
    dynamic = [n for n in AMENDMENT_3 if review.factor[n].modelled]
    ardl = [n for n in AMENDMENT_3 if SPECS[n].lagged_outcome]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    bins = np.linspace(-1, 1, 81)
    for n in dynamic:
        axes[0].hist(review.fits[n].draws["rho_f"], bins=bins, color=colours[n], alpha=0.55,
                     label=n)
    axes[0].axvline(0, color=MUTED, lw=1)
    axes[0].set_xlabel("rho_f (AR(1) coefficient of the common factor)")
    axes[0].set_ylabel("Posterior draws")
    axes[0].set_title("Factor persistence")
    axes[0].legend(fontsize=8.5)
    countries = review.fits["primary"].countries
    xs = np.arange(len(countries))
    for offset, n in zip((-0.15, 0.15), ardl, strict=False):
        psi = review.fits[n].draws["psi"]
        lo, med, hi = np.quantile(psi, [0.05, 0.5, 0.95], axis=0)
        axes[1].vlines(xs + offset, lo, hi, color=colours[n], lw=1.4)
        axes[1].plot(xs + offset, med, "o", ms=4.5, color=colours[n], mec=SURFACE, mew=1,
                     label=n)
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(countries, rotation=90, fontsize=8)
    axes[1].set_ylabel("psi (median, 90% interval)")
    axes[1].set_title("Lagged unemployment change, by country")
    axes[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.subplots_adjust(wspace=0.28)
    _finish(plt, fig, directory / "dynamics_parameters.png", source, credit_y=-0.2)

    # Country estimates.
    names = list(review.fits)
    order = sorted(range(len(countries)), key=lambda j: countries[j])
    ys = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(7.8, 10.0))
    offsets = np.linspace(-0.3, 0.3, len(names))
    for name, offset in zip(names, offsets, strict=True):
        rows = review.estimates[name]
        for y, i in zip(ys, order, strict=True):
            ax.plot([rows[i].low, rows[i].high], [y + offset] * 2, color=colours[name], lw=1.5,
                    solid_capstyle="round")
            ax.plot([rows[i].median], [y + offset], marker="o", ms=5, color=colours[name],
                    mec=SURFACE, mew=1.1, lw=0)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([countries[i] for i in order])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Total association with GDP growth (pp per pp), 2009-2019; 90% intervals")
    ax.set_title("theta (primary, AR(1) factor) and theta_LR (ARDL fits)")
    ax.legend(handles=[Line2D([], [], marker="o", color=colours[n], lw=1.5, ms=6,
                              label=f"{n} ({review.estimates[n][0].estimand})")
                       for n in names], loc="upper left", bbox_to_anchor=(0.0, -0.07),
              fontsize=8.5, ncol=2)
    _finish(plt, fig, directory / "country_estimates.png", source, credit_y=-0.1)


def write_dynamics_report(fits_dir: Path, data_dir: Path, output_dir: Path,
                          report: Path | None = None, figures: bool = True,
                          replication_dir: Path | None = None) -> DynamicsReview:
    """Build the review, write ``dynamics.json``, the figures and the Markdown report."""
    review = build_dynamics(fits_dir, data_dir, replication_dir)
    report = report or output_dir / "BAYES_DYNAMICS.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dynamics.json").write_text(
        json.dumps(summary_document(review), indent=1, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")
    if figures:
        source = ("Source: Eurostat, snapshot 2026-10-15; hierarchical Bayesian model "
                  "(BAYES_PROTOCOL.md, amendment 3). Descriptive, not causal.")
        write_figures(review, output_dir, source)
    assets = Path(os.path.relpath(output_dir, report.parent)).as_posix()
    command = (f"poetry run eu-okun bayes-dynamics --fits-dir {fits_dir.as_posix()} "
               f"--data-dir {data_dir.as_posix()} --output-dir {output_dir.as_posix()} "
               f"--report {report.as_posix()}"
               + (f" --replication-dir {replication_dir.as_posix()}"
                  if replication_dir else ""))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(review, assets, command), encoding="utf-8", newline="\n")
    return review
