"""Model checks of amendment 1 (``BAYES_PROTOCOL.md``, section 9).

Additive shared quarter effects, the with/without comparison by PSIS-LOO, prior
predictive checks and posterior recovery on synthetic data, with the report
``docs/BAYES_MODEL_CHECKS.md``. Fitting happens in ``bayes_fit``; this module reads the
saved fits (``fit.json``, ``draws.npz``, ``posterior.nc``) except for the recovery run.
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

from eu_okun.audit import audit_snapshot
from eu_okun.baseline import _read_panel
from eu_okun.bayes_artifacts import prior_draws, simulate_outcomes
from eu_okun.bayes_data import BayesData, build_bayes_data
from eu_okun.bayes_fit import (
    PROTOCOL_SETTINGS,
    SamplerSettings,
    _masked_stats,
    fit_specification,
    interval,
)
from eu_okun.bayes_model import AMENDMENT_1, SPECS, ModelSpec, design_for
from eu_okun.contract import EurostatDataError
from eu_okun.review import read_flags

# Fits on the primary window with K = 2 share their unmasked cells, so their LOO values
# are comparable; the comparisons are those of section 9.2.
LOO_SPECS: tuple[str, ...] = (
    "primary", "no_common_factor", "participation", *AMENDMENT_1,
)
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("time_effects", "no_common_factor"),
    ("time_effects_participation", "participation_no_common_factor"),
    ("time_effects", "primary"),
    ("time_effects_participation", "time_effects"),
)
PARETO_K_MAX = 0.7
PARETO_SHARE_MAX = 0.01
PRIOR_CHECK_DRAWS = 1000
PRIOR_CHECK_SEED = 20261018
RECOVERY_SPECS: tuple[str, ...] = ("primary", "time_effects", "gaussian_iid")
RECOVERY_SEED = 20261019
# Generating values of the recovery data sets (section 9.4), fixed before any fit.
RECOVERY_HYPER: dict[str, float] = {"mu_theta": -0.30, "tau_theta": 0.15, "nu": 8.0,
                                    "tau_quarter": 0.15}


# ----------------------------------------------------------------------------------
# Prior predictive checks (section 9.3)
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorPredictive:
    """Quantiles (5%, 50%, 95%) of simulated statistics next to the observed values."""

    spec: str
    draws: int
    quantiles: dict[str, tuple[float, float, float]]
    observed: dict[str, float]
    share_above_5pp: float


def series_statistics(data: BayesData, y: np.ndarray) -> dict[str, float]:
    """Statistics of one data set over the unmasked cells."""
    stats = _masked_stats(data, y)
    stats["max_abs_change"] = float(np.max(np.abs(y[data.mask])))
    return stats


def prior_predictive_check(spec: ModelSpec, data: BayesData, draws: int = PRIOR_CHECK_DRAWS,
                           seed: int = PRIOR_CHECK_SEED) -> PriorPredictive:
    """Simulate one data set per prior draw on the real design and summarise them."""
    prior = prior_draws(spec, data, draws, seed)
    simulated = simulate_outcomes(spec, data, prior, np.random.default_rng(seed))
    rows = [series_statistics(data, simulated[s]) for s in range(draws)]
    quantiles = {}
    for key in rows[0]:
        values = np.asarray([r[key] for r in rows])
        values = values[np.isfinite(values)]
        q = np.quantile(values, [0.05, 0.5, 0.95])
        quantiles[key] = (float(q[0]), float(q[1]), float(q[2]))
    cells = simulated[:, data.mask]
    return PriorPredictive(spec.name, draws, quantiles, series_statistics(data, data.y),
                           float(np.mean(np.abs(cells) > 5.0)))


# ----------------------------------------------------------------------------------
# PSIS-LOO (section 9.2)
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class LooResult:
    """PSIS-LOO of one fit over its unmasked cells."""

    spec: str
    elpd: float
    se: float
    p_loo: float
    cells: int
    pareto_high: int
    pointwise: tuple[float, ...]

    @property
    def reliable(self) -> bool:
        return self.pareto_high <= PARETO_SHARE_MAX * self.cells


def loo_of(path: Path, spec: str) -> LooResult:
    """PSIS-LOO from the ``log_likelihood`` group of a NetCDF artifact."""
    import xarray as xr
    from arviz_stats.loo.loo import loo

    with xr.open_datatree(path, engine="h5netcdf") as tree:
        # ArviZ's loo carries no annotations.
        result: Any = loo(tree, pointwise=True, var_name="y")  # type: ignore[no-untyped-call]
        elpd_i = np.asarray(result.elpd_i).reshape(-1)
        pareto_k = np.asarray(result.pareto_k).reshape(-1)
        return LooResult(spec, float(result.elpd), float(result.se), float(result.p),
                         int(elpd_i.size), int(np.sum(pareto_k > PARETO_K_MAX)),
                         tuple(float(v) for v in elpd_i))


@dataclass(frozen=True)
class LooComparison:
    """elpd difference A - B with the standard error of the pointwise differences."""

    model_a: str
    model_b: str
    elpd_diff: float
    se_diff: float
    verdict: str
    reliable: bool


def compare_loo(a: LooResult, b: LooResult) -> LooComparison:
    """Section 9.2: 'better supported' when |difference| > 2 SE, else not distinguishable."""
    if a.cells != b.cells:
        raise ValueError(f"{a.spec} and {b.spec} were scored on different cells.")
    diff = np.asarray(a.pointwise) - np.asarray(b.pointwise)
    elpd_diff = float(diff.sum())
    se_diff = float(math.sqrt(diff.size * np.var(diff)))
    if abs(elpd_diff) > 2.0 * se_diff:
        verdict = f"{a.spec if elpd_diff > 0 else b.spec} better supported"
    else:
        verdict = "not distinguishable"
    return LooComparison(a.spec, b.spec, elpd_diff, se_diff, verdict,
                         a.reliable and b.reliable)


# ----------------------------------------------------------------------------------
# Posterior recovery on synthetic data (section 9.4)
# ----------------------------------------------------------------------------------


def recovery_truth(spec: ModelSpec, data: BayesData, seed: int = RECOVERY_SEED,
                   ) -> dict[str, np.ndarray]:
    """Generating parameters (one draw, leading axis of length 1), fixed by the seed."""
    rng = np.random.default_rng(seed)
    n, t = data.y.shape
    h = RECOVERY_HYPER
    theta = h["mu_theta"] + h["tau_theta"] * np.asarray(rng.standard_normal(n))
    b = np.asarray(rng.normal(0.05, 0.05, (n, data.lags)))
    truth: dict[str, np.ndarray] = {
        "theta": theta, "alpha": np.asarray(rng.normal(0.0, 0.05, n)), "b": b,
        "beta0": theta - b.sum(axis=1),
        "sigma": np.exp(np.asarray(rng.normal(math.log(0.2), 0.4, n))),
    }
    if spec.ar1:
        truth["phi"] = np.asarray(rng.uniform(0.0, 0.5, n))
    if spec.student_t:
        truth["nu"] = np.asarray(h["nu"])
    if spec.common_factor:
        truth["lam"] = np.asarray(rng.uniform(0.0, 0.2, n))
        truth["f"] = np.asarray(rng.standard_normal(t))
    if spec.time_effects:
        effect = np.asarray(rng.normal(0.0, h["tau_quarter"], t))
        truth["quarter_effect"] = effect - effect.mean()
    if spec.participation:
        truth["gamma"] = np.asarray(rng.normal(0.0, 0.1, n))
    return {k: np.asarray(v)[None] for k, v in truth.items()}


def recovery_data(spec: ModelSpec, data: BayesData, seed: int = RECOVERY_SEED,
                  ) -> tuple[BayesData, dict[str, np.ndarray]]:
    """The real design with the outcome replaced by a simulation from known parameters."""
    truth = recovery_truth(spec, data, seed)
    y = simulate_outcomes(spec, data, truth, np.random.default_rng(seed + 1))[0]
    return replace(data, y=y), {k: v[0] for k, v in truth.items()}


@dataclass(frozen=True)
class Recovery:
    """Interval coverage of known parameters by one fit to synthetic data."""

    spec: str
    accepted: bool
    max_rhat: float
    coverage: dict[str, tuple[int, int, int]]          # name -> (in 50%, in 90%, total)
    hyper: dict[str, tuple[float, float, float, bool]]  # name -> (truth, low, high, inside)
    theta: tuple[tuple[float, float, float, float], ...]  # (truth, median, low90, high90)


def assess_recovery(spec: str, samples: Mapping[str, np.ndarray],
                    truth: Mapping[str, np.ndarray], accepted: bool,
                    max_rhat: float) -> Recovery:
    """Coverage of the 50% and 90% equal-tailed intervals, and hyperparameter checks."""
    coverage: dict[str, tuple[int, int, int]] = {}
    for name in ("theta", "beta0", "alpha", "sigma", "phi", "lam", "f", "quarter_effect",
                 "gamma"):
        if name not in samples or name not in truth:
            continue
        draws = samples[name].reshape(-1, *samples[name].shape[2:])
        lo50, hi50, lo90, hi90 = np.quantile(draws, [0.25, 0.75, 0.05, 0.95], axis=0)
        value = truth[name]
        coverage[name] = (int(np.sum((value >= lo50) & (value <= hi50))),
                          int(np.sum((value >= lo90) & (value <= hi90))), int(value.size))
    hyper: dict[str, tuple[float, float, float, bool]] = {}
    for name, true_value in RECOVERY_HYPER.items():
        if name in samples:
            low, high = interval(samples[name].reshape(-1))
            hyper[name] = (true_value, low, high, low <= true_value <= high)
    theta = samples["theta"].reshape(-1, samples["theta"].shape[-1])
    lo, med, hi = np.quantile(theta, [0.05, 0.5, 0.95], axis=0)
    points = tuple((float(t), float(m), float(a), float(b))
                   for t, m, a, b in zip(truth["theta"], med, lo, hi, strict=True))
    return Recovery(spec, accepted, max_rhat, coverage, hyper, points)


def run_recovery(data_dir: Path, output_dir: Path, names: Sequence[str] = RECOVERY_SPECS,
                 settings: SamplerSettings = PROTOCOL_SETTINGS,
                 retry_warmup: int | None = 3000) -> list[Recovery]:
    """Fit each specification to its synthetic data set and save ``recovery.json``."""
    unknown = [n for n in names if n not in SPECS]
    if unknown:
        raise ValueError(f"Unknown specifications: {', '.join(unknown)}.")
    audit_snapshot(data_dir)
    csv_path = data_dir / "processed" / "eu27_quarterly.csv"
    panel, flags = _read_panel(csv_path), read_flags(csv_path)
    results = []
    for name in names:
        spec = SPECS[name]
        design = design_for(spec, panel, flags)
        data, truth = recovery_data(spec, design)
        fit = fit_specification(data, spec, settings)
        if not fit.diagnostics.accepted and retry_warmup is not None:
            fit = fit_specification(data, spec, replace(settings, warmup=retry_warmup))
        result = assess_recovery(name, fit.samples, truth, fit.diagnostics.accepted,
                                 fit.diagnostics.max_rhat)
        target = output_dir / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "recovery.json").write_text(json.dumps({
            **asdict(result), "settings": asdict(fit.settings), "seed": RECOVERY_SEED,
            "hyper_truth": RECOVERY_HYPER, "runtime_seconds": fit.runtime_seconds,
        }, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        results.append(result)
    if not results:
        raise EurostatDataError("No recovery fit was run.")
    return results


def realized_hyperparameters(truth: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Sample analogues of the hyperparameters in one simulated data set.

    A posterior conditions on the 27 simulated country sums (and 43 quarter effects), so
    it centres on their realised mean and spread rather than on the generating values.
    """
    theta = np.asarray(truth["theta"]).reshape(-1)
    out = {"mu_theta": float(theta.mean()), "tau_theta": float(theta.std(ddof=1))}
    if "quarter_effect" in truth:
        effect = np.asarray(truth["quarter_effect"]).reshape(-1)
        # Sum-to-zero values: sum(x^2) / (T - 1) estimates the ZeroSumNormal scale squared.
        out["tau_quarter"] = float(np.sqrt(np.sum(effect**2) / (effect.size - 1)))
    return out


def load_recovery(directory: Path) -> list[Recovery]:
    """Read the ``recovery.json`` files written by ``run_recovery``."""
    out = []
    for name in RECOVERY_SPECS:
        path = directory / name / "recovery.json"
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            out.append(Recovery(d["spec"], d["accepted"], d["max_rhat"],
                                {k: tuple(v) for k, v in d["coverage"].items()},
                                {k: tuple(v) for k, v in d["hyper"].items()},
                                tuple(tuple(v) for v in d["theta"])))
    return out


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChecksReplication:
    """Results of an independent run of the same fits (e.g. on another platform)."""

    label: str
    comparisons: dict[tuple[str, str], tuple[float, float, str]]  # elpd diff, SE, verdict
    recovery: dict[str, tuple[int, int, bool]]                    # theta in 90%, total, ok
    accepted: dict[str, tuple[bool, float]]                       # spec -> accepted, R-hat


def load_checks_replication(directory: Path) -> ChecksReplication:
    """Read ``model_checks.json`` and ``fits/<spec>/fit.json`` of another run."""
    checks = json.loads((directory / "model_checks.json").read_text(encoding="utf-8"))
    comparisons = {(c["model_a"], c["model_b"]): (c["elpd_diff"], c["se_diff"], c["verdict"])
                   for c in checks["comparisons"]}
    recovery = {r["spec"]: (r["coverage"]["theta"][1], r["coverage"]["theta"][2],
                            bool(r["accepted"])) for r in checks["recovery"]}
    accepted = {}
    for path in sorted((directory / "fits").glob("*/fit.json")):
        diagnostics = json.loads(path.read_text(encoding="utf-8"))["diagnostics"]
        accepted[path.parent.name] = (bool(diagnostics["accepted"]),
                                      float(diagnostics["max_rhat"]))
    readme = directory / "README.md"
    label = (readme.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
             if readme.exists() else directory.name)
    return ChecksReplication(label, comparisons, recovery, accepted)


@dataclass
class ChecksReview:
    """Everything the report is generated from."""

    fits: dict[str, dict[str, Any]]                 # spec -> fit.json
    draws: dict[str, dict[str, np.ndarray]]         # spec -> saved draws (chain, draw, ...)
    data: BayesData                                 # primary-window design
    loo: dict[str, LooResult]
    comparisons: list[LooComparison]
    prior: list[PriorPredictive]
    recovery: list[Recovery]
    realized: dict[str, dict[str, float]]           # recovery spec -> sample analogues
    replication: ChecksReplication | None = None


def _load(directory: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    summary = json.loads((directory / "fit.json").read_text(encoding="utf-8"))
    with np.load(directory / "draws.npz") as stored:
        return summary, {name: stored[name] for name in stored.files}


def build_checks(fits_dir: Path, data_dir: Path, recovery_dir: Path | None = None,
                 replication_dir: Path | None = None) -> ChecksReview:
    """Load the fits, compute LOO, the comparisons and the prior predictive checks."""
    missing = [n for n in AMENDMENT_1 if not (fits_dir / n / "fit.json").exists()]
    if missing:
        raise FileNotFoundError(f"No fit for {', '.join(missing)} in {fits_dir}.")
    csv_path = data_dir / "processed" / "eu27_quarterly.csv"
    panel, flags = _read_panel(csv_path), read_flags(csv_path)
    fits: dict[str, dict[str, Any]] = {}
    draws: dict[str, dict[str, np.ndarray]] = {}
    for name in SPECS:
        if (fits_dir / name / "fit.json").exists():
            fits[name], draws[name] = _load(fits_dir / name)
    data = build_bayes_data(panel, flags, SPECS["primary"].windows, 2)
    loo: dict[str, LooResult] = {}
    for name in LOO_SPECS:
        path = fits_dir / name / "posterior.nc"
        if name in fits and fits[name]["diagnostics"]["accepted"] and path.exists():
            loo[name] = loo_of(path, name)
    comparisons = [compare_loo(loo[a], loo[b]) for a, b in COMPARISONS if a in loo and b in loo]
    prior = []
    for spec in SPECS.values():
        design = design_for(spec, panel, flags)
        prior.append(prior_predictive_check(spec, design, draws=PRIOR_CHECK_DRAWS))
    recovery = load_recovery(recovery_dir) if recovery_dir is not None else []
    realized = {}
    for rec in recovery:
        spec = SPECS[rec.spec]
        design = design_for(spec, panel, flags)
        realized[rec.spec] = realized_hyperparameters(
            {k: v[0] for k, v in recovery_truth(spec, design).items()})
    replication = (load_checks_replication(replication_dir) if replication_dir is not None
                   else None)
    return ChecksReview(fits, draws, data, loo, comparisons, prior, recovery, realized,
                        replication)


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]], align: str = "") -> list[str]:
    align = align or "l" + "r" * (len(header) - 1)
    marks = ["---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(marks) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def _num(value: float) -> str:
    """Two decimals, or three significant digits for very large values."""
    return f"{value:.2f}" if abs(value) < 1000 else f"{value:.3g}"


def coverage_range(total: int, level: float = 0.9) -> tuple[int, int]:
    """Central 95% range of the number of true values inside ``level`` intervals.

    Binomial(total, level) quantiles: the smallest counts whose cumulative probability
    reaches 2.5% and 97.5%.
    """
    cdf = 0.0
    low: int | None = None
    for k in range(total + 1):
        cdf += math.comb(total, k) * level**k * (1.0 - level) ** (total - k)
        if low is None and cdf >= 0.025:
            low = k
        if cdf >= 0.975:
            return (k if low is None else low), k
    return (0 if low is None else low), total


def _median_interval(values: np.ndarray) -> str:
    low, high = interval(values)
    return f"{np.median(values):+.3f} [{low:+.3f}, {high:+.3f}]"


def _flat(draws: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    values = draws[name]
    return values.reshape(-1, *values.shape[2:])


def render(review: ChecksReview, assets: str, command: str) -> str:
    """Assemble the Markdown report."""
    lines = ["# Hierarchical Bayes: quarter effects, prior predictive checks and recovery", "",
             "Generated by `eu-okun bayes-checks` from the frozen snapshot "
             "`snapshots/eurostat-2026-10-15`. Amendment 1 of "
             "[BAYES_PROTOCOL.md](../BAYES_PROTOCOL.md) (section 9) registered these checks "
             "and the three new specifications before any of them was fitted. The "
             "pre-registered results in [HIERARCHICAL_BAYES.md](HIERARCHICAL_BAYES.md) are "
             "unchanged: their robustness and claim judgements use the nine original "
             "specifications only. Everything here is a descriptive conditional "
             "association; no specification identifies a causal GDP shock.", ""]
    fits, draws = review.fits, review.draws
    accepted = {n: bool(f["diagnostics"]["accepted"]) for n, f in fits.items()}

    # Summary
    lines += ["## Summary", ""]
    te = draws["time_effects"]
    lines.append(f"- `time_effects`: {'accepted' if accepted['time_effects'] else 'FAILED'}; "
                 f"mu_theta {_median_interval(_flat(te, 'mu_theta'))}, tau of the quarter "
                 f"effects {_median_interval(_flat(te, 'tau_quarter'))} pp.")
    for c in review.comparisons:
        lines.append(f"- LOO, `{c.model_a}` vs `{c.model_b}`: elpd difference "
                     f"{c.elpd_diff:+.1f} (SE {c.se_diff:.1f}), {c.verdict}"
                     f"{'' if c.reliable else ' (unreliable: Pareto k)'}.")
    implausible = [p for p in review.prior if p.share_above_5pp > 0.01]
    primary_prior = next(p for p in review.prior if p.spec == "primary")
    observed_max = primary_prior.observed["max_abs_change"]
    lines.append(
        f"- Prior predictive: simulated changes beyond 5 pp in absolute value exceed 1% of "
        f"cells for {len(implausible)} of {len(review.prior)} specifications (primary: "
        f"{primary_prior.share_above_5pp:.1%}; median largest simulated change "
        f"{primary_prior.quantiles['max_abs_change'][1]:.1f} pp against {observed_max:.1f} pp "
        "observed). The weakly informative priors put visible mass on implausibly large "
        "quarterly changes; they are reported, not changed.")
    for r in review.recovery:
        cov = r.coverage["theta"]
        lines.append(f"- Recovery, `{r.spec}`: {cov[1]} of {cov[2]} true country sums inside "
                     f"their 90% intervals ({'accepted' if r.accepted else 'not accepted'} "
                     f"fit, max R-hat {r.max_rhat:.4f}).")
    if review.replication is not None:
        same = all(review.replication.comparisons.get((c.model_a, c.model_b), (0, 0, ""))[2]
                   == c.verdict for c in review.comparisons)
        lines.append(f"- An independent run ({review.replication.label}) gives "
                     f"{'the same' if same else 'different'} LOO verdicts; see section 6.")
    lines.append("")

    # 1. Quarter-effect specifications
    lines += ["## 1. Additive shared quarter effects", "",
              "`delta[t]` is a quarter effect common to all countries, constrained to sum to "
              "zero over the 43 quarters so that the country intercepts carry the level. "
              "With it, `theta[i]` measures the association with the country's GDP growth "
              "beyond the quarter's common effect, and `mu_theta` is identified only from "
              "cross-country differences in growth within quarters.", ""]
    rows = []
    for name in ("primary", "no_common_factor", "participation", *AMENDMENT_1):
        if name not in fits:
            continue
        d = fits[name]["diagnostics"]
        rows.append([f"`{name}`", SPECS[name].description,
                     "accepted" if d["accepted"] else "**failed**: " + "; ".join(d["failures"]),
                     f"{d['max_rhat']:.3f}", f"{min(d['min_ess_bulk'], d['min_ess_tail']):.0f}",
                     d["divergences"], _median_interval(_flat(draws[name], "mu_theta"))])
    lines += _table(["Specification", "Description", "Convergence", "Max R-hat", "Min ESS",
                     "Divergences", "mu_theta, median [90%]"], rows, "lllrrrr")
    countries = list(review.data.countries)
    rows = []
    for i in sorted(range(len(countries)), key=lambda j: countries[j]):
        row = [countries[i]]
        for name in ("primary", "no_common_factor", "time_effects",
                     "time_effects_participation"):
            row.append(_median_interval(_flat(draws[name], "theta")[:, i]) if name in draws
                       else "")
        rows.append(row)
    lines += ["Country sums `theta[i]`, posterior median and 90% interval (alphabetical; no "
              "ranking):", ""]
    lines += _table(["Country", "primary (factor)", "no_common_factor",
                     "time_effects", "time_effects_participation"], rows)
    delta = np.median(_flat(te, "quarter_effect"), axis=0)
    mean_g = review.data.g.mean(axis=0)
    mean_y = review.data.y.mean(axis=0)
    lines += [
        f"The posterior median quarter effects correlate {np.corrcoef(delta, mean_g)[0, 1]:+.2f} "
        f"with average GDP growth and {np.corrcoef(delta, mean_y)[0, 1]:+.2f} with the average "
        "unemployment change across countries. A strong correlation with GDP growth means "
        "the quarter effects absorb part of the common GDP association, as the "
        "identification argument above anticipates.", "",
        f"![Quarter effects and average GDP growth]({assets}/quarter_effects.png)", "",
        f"![Country sums under the factor, no shared effect and quarter effects]({assets}/"
        "theta_shared_effects.png)", "",
    ]

    # 2. With versus without
    lines += ["## 2. With versus without shared quarter effects", "",
              "PSIS-LOO over the unmasked cells of the 2009-Q2..2019-Q4 window (identical "
              "for every specification listed). Leave-one-cell-out; for AR(1) errors each "
              "term conditions on the previous quarter, so this is not temporal validation.",
              ""]
    rows = []
    for name, res in review.loo.items():
        rows.append([f"`{name}`", f"{res.elpd:.1f}", f"{res.se:.1f}", f"{res.p_loo:.1f}",
                     res.cells, res.pareto_high, "yes" if res.reliable else "no"])
    lines += _table(["Specification", "elpd LOO", "SE", "p_loo", "Cells",
                     f"Pareto k > {PARETO_K_MAX}", "Reliable"], rows)
    rows = [[f"`{c.model_a}`", f"`{c.model_b}`", f"{c.elpd_diff:+.1f}", f"{c.se_diff:.1f}",
             c.verdict, "yes" if c.reliable else "no"] for c in review.comparisons]
    lines += ["Registered comparisons (A minus B; 'better supported' when the difference "
              "exceeds twice its standard error):", ""]
    lines += _table(["A", "B", "elpd difference", "SE", "Verdict", "Reliable"], rows, "llrrll")
    rows = []
    for name in ("primary", "no_common_factor", "participation", *AMENDMENT_1):
        if name in fits:
            pc = fits[name]["predictive_check"]
            rows.append([f"`{name}`", f"{pc['observed']['cross_country_correlation']:.3f}",
                         f"{pc['replicated_mean']['cross_country_correlation']:.3f}",
                         f"{pc['p_value']['cross_country_correlation']:.2f}",
                         f"{pc['coverage_90']:.2f}"])
    lines += ["Posterior predictive check of the average cross-country correlation "
              "(p-value: share of replications at least as large as observed):", ""]
    lines += _table(["Specification", "Observed", "Replicated mean", "p-value",
                     "One-step 90% coverage"], rows)

    # 3. Prior predictive
    lines += ["## 3. Prior predictive checks", "",
              f"{review.prior[0].draws} prior draws per specification, each propagated to a "
              "simulated data set on the real design. 5%, 50% and 95% quantiles; the "
              "observed value is in the last row. The priors were fixed in advance and are "
              "not changed by this check.", ""]
    keys = (("mean_sd", "Mean country SD (pp)"), ("max_abs_change", "Largest |change| (pp)"),
            ("lag1_autocorrelation", "Lag-1 autocorrelation"),
            ("cross_country_correlation", "Cross-country correlation"))
    rows = []
    for p in review.prior:
        rows.append([f"`{p.spec}`", *(" / ".join(_num(v) for v in p.quantiles[k])
                                      for k, _ in keys), f"{p.share_above_5pp:.1%}"])
    observed = review.prior[0].observed
    rows.append(["**observed (primary window)**", *(f"{observed[k]:.2f}" for k, _ in keys), ""])
    lines += _table(["Specification", *(label for _, label in keys), "Cells beyond 5 pp"], rows)

    # 4. Recovery
    lines += ["## 4. Posterior recovery on synthetic data", ""]
    if review.recovery:
        lo27, hi27 = coverage_range(27)
        lines += ["One data set per specification, simulated on the real design from "
                  "parameters fixed in code (mu_theta = -0.30, tau_theta = 0.15), fitted with "
                  f"the protocol settings. With 27 countries, {lo27} to {hi27} true values "
                  "inside the 90% intervals is consistent with calibration (central 95% of "
                  "the binomial distribution); the last column gives that range for each row. "
                  "A single data set is a recovery check, not simulation-based calibration.",
                  ""]
        rows = []
        below = []
        for rec in review.recovery:
            for name, (in50, in90, total) in rec.coverage.items():
                low, high = coverage_range(total)
                flag = "" if low <= in90 <= high else " **outside**"
                if flag:
                    below.append(f"`{rec.spec}` {name} ({in90}/{total})")
                rows.append([f"`{rec.spec}`", name, total, f"{in50} ({in50 / total:.0%})",
                             f"{in90} ({in90 / total:.0%})", f"{low}-{high}{flag}"])
        lines += _table(["Specification", "Parameter", "Values", "Inside 50%", "Inside 90%",
                         "Consistent range"], rows, "llrrrl")
        lines += [("90% coverage outside the range expected under calibration: "
                   + "; ".join(below) + ". These intervals are too narrow for those "
                   "parameters in this data set." if below else
                   "Every 90% coverage count lies within the range expected under "
                   "calibration."), ""]
        rows = []
        for rec in review.recovery:
            sample = review.realized.get(rec.spec, {})
            for name, (t, lo, hi, inside) in rec.hyper.items():
                realised = sample.get(name)
                rows.append([f"`{rec.spec}`", name, f"{t:+.3f}",
                             "" if realised is None else f"{realised:+.3f}",
                             f"[{lo:+.3f}, {hi:+.3f}]", "yes" if inside else "**no**"])
        lines += ["Hyperparameters: the generating value, its realised sample analogue in "
                  "the simulated data (the posterior conditions on the simulated country "
                  "sums and quarter effects, so it centres on these), and the 90% interval. "
                  "The registered check is whether the generating value lies inside.", ""]
        lines += _table(["Specification", "Hyperparameter", "Generating value",
                         "Realised in the data", "90% interval", "Inside"], rows, "llrrrl")
        lines += [f"![Recovery of the country sums]({assets}/recovery.png)", ""]
    else:
        lines += ["No recovery run was supplied (`--recovery-dir`).", ""]

    # 5. Artifacts and limitations
    lines += [
        "## 5. Posterior artifacts", "",
        "Every fit directory holds `posterior.nc`, an ArviZ-compatible NetCDF file (xarray "
        "DataTree) with the groups `posterior`, `prior`, `prior_predictive`, "
        "`observed_data`, `constant_data` and `log_likelihood`, plus `sample_stats` for fits "
        "run after this amendment. Root attributes record the sampler settings and seed, "
        "software versions, snapshot hashes and diagnostics. The nine original fits were "
        "exported from their saved draws (`eu-okun bayes-export`), so their files have no "
        "`sample_stats` group.", "",
        "```python",
        "import xarray as xr",
        "tree = xr.open_datatree(\"results/bayes/primary/posterior.nc\", engine=\"h5netcdf\")",
        "```", "",
    ]
    number = 6
    if review.replication is not None:
        lines += _replication_section(review)
        number += 1
    lines += [
        f"## {number}. Limitations", "",
        "- Quarter effects absorb every common movement, including the part of GDP growth "
        "that all countries share, so `theta` under quarter effects is a within-quarter "
        "association and not comparable one-to-one with the primary estimand.",
        "- PSIS-LOO leaves out single cells; it rewards fitting the in-sample quarter "
        "effects, which are informed by the other 26 countries in the same quarter. "
        "Temporal predictive performance is assessed separately (PREDICTIVE_PROTOCOL.md).",
        "- One synthetic data set per specification checks recovery, not calibration.",
        "- None of these models identifies a structural GDP shock.", "",
        f"## {number + 1}. Reproduction", "",
        "```bash",
        "poetry run eu-okun bayes-fit --data-dir snapshots/eurostat-2026-10-15 "
        "--output-dir results/bayes --specs " + ",".join(AMENDMENT_1) + " --retry-warmup 3000",
        "poetry run eu-okun bayes-export --fits-dir results/bayes "
        "--data-dir snapshots/eurostat-2026-10-15",
        "poetry run eu-okun bayes-recovery --data-dir snapshots/eurostat-2026-10-15 "
        "--output-dir results/bayes-recovery",
        command,
        "```", "",
    ]
    return "\n".join(lines)


def _replication_section(review: ChecksReview) -> list[str]:
    rep_run = review.replication
    assert rep_run is not None
    lines = ["## 6. Replication on a second platform", "",
             f"{rep_run.label}: the same snapshot, lock file, seeds, sampler settings and "
             "single retry. Its results are in `replication-linux/`.", ""]
    rows = []
    for c in review.comparisons:
        diff, se, verdict = rep_run.comparisons[(c.model_a, c.model_b)]
        rows.append([f"`{c.model_a}`", f"`{c.model_b}`", f"{c.elpd_diff:+.1f} ({c.se_diff:.1f})",
                     c.verdict, f"{diff:+.1f} ({se:.1f})", verdict])
    lines += _table(["A", "B", "elpd difference (SE), this run", "Verdict",
                     "elpd difference (SE), replication", "Verdict"], rows, "llrlrl")
    differ = [f"`{name}` (this run {'accepted' if fit['diagnostics']['accepted'] else 'failed'}, "
              f"replication {'accepted' if rep_run.accepted[name][0] else 'failed'}, R-hat "
              f"{fit['diagnostics']['max_rhat']:.3f} / {rep_run.accepted[name][1]:.3f})"
              for name, fit in review.fits.items()
              if name in rep_run.accepted
              and bool(fit["diagnostics"]["accepted"]) != rep_run.accepted[name][0]]
    lines.append("- Convergence: acceptance " + ("is the same for every specification."
                                                 if not differ else
                                                 f"differs for {'; '.join(differ)}."))
    for rec in review.recovery:
        if rec.spec in rep_run.recovery:
            in90, total, ok = rep_run.recovery[rec.spec]
            lines.append(f"- Recovery, `{rec.spec}`: {rec.coverage['theta'][1]}/"
                         f"{rec.coverage['theta'][2]} country sums inside 90% intervals here "
                         f"({'accepted' if rec.accepted else 'not accepted'}), {in90}/{total} "
                         f"in the replication ({'accepted' if ok else 'not accepted'}).")
    return lines + [""]


def write_figures(review: ChecksReview, directory: Path, source: str) -> None:
    """Quarter effects, country sums under the three treatments and recovery."""
    from eu_okun.review_figures import AQUA, BLUE, INK2, MUTED, ORANGE, SURFACE, _finish, _pyplot

    plt = _pyplot()
    from matplotlib.lines import Line2D

    data = review.data
    effect = _flat(review.draws["time_effects"], "quarter_effect")
    low, mid, high = np.quantile(effect, [0.05, 0.5, 0.95], axis=0)
    x = np.arange(data.n_quarters)
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 6.2), sharex=True)
    axes[0].fill_between(x, low, high, color=BLUE, alpha=0.2, lw=0)
    axes[0].plot(x, mid, color=BLUE, lw=2)
    axes[0].axhline(0, color=MUTED, lw=1)
    axes[0].set_ylabel("Quarter effect (pp)")
    axes[0].set_title("Shared quarter effects (time_effects): posterior median and 90% interval")
    axes[1].plot(x, data.g.mean(axis=0), color=ORANGE, lw=2)
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set_ylabel("Mean GDP growth (%)")
    axes[1].set_title("Unweighted mean of quarterly GDP growth across the 27 countries")
    axes[1].set_xticks(x[::4])
    axes[1].set_xticklabels(list(data.quarters)[::4], rotation=45, ha="right", fontsize=8)
    fig.subplots_adjust(hspace=0.3)
    _finish(plt, fig, directory / "quarter_effects.png", source, credit_y=-0.05)

    countries = list(data.countries)
    order = sorted(range(len(countries)), key=lambda j: countries[j])
    ys = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 9.2))
    for name, colour, offset in (("primary", BLUE, -0.24), ("no_common_factor", AQUA, 0.0),
                                 ("time_effects", ORANGE, 0.24)):
        theta = _flat(review.draws[name], "theta")
        lo, med, hi = np.quantile(theta, [0.05, 0.5, 0.95], axis=0)
        for y, i in zip(ys, order, strict=True):
            ax.plot([lo[i], hi[i]], [y + offset] * 2, color=colour, lw=1.6,
                    solid_capstyle="round")
            ax.plot([med[i]], [y + offset], marker="o", ms=5.5, color=colour, mec=SURFACE,
                    mew=1.2, lw=0)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([countries[i] for i in order])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Sum of GDP coefficients, lags 0-2 (pp per pp), 2009-2019; 90% intervals")
    ax.set_title("Country sums under three treatments of shared quarterly movements")
    ax.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, lw=1.6, ms=6, label="Common factor (primary)"),
        Line2D([], [], marker="o", color=AQUA, lw=1.6, ms=6, label="No shared effect"),
        Line2D([], [], marker="o", color=ORANGE, lw=1.6, ms=6,
               label="Additive quarter effects"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.075), fontsize=8.5)
    _finish(plt, fig, directory / "theta_shared_effects.png", source, credit_y=-0.1)

    if review.recovery:
        fig, axes = plt.subplots(1, len(review.recovery), figsize=(4.2 * len(review.recovery),
                                                                     4.2), squeeze=False)
        for ax, r in zip(axes[0], review.recovery, strict=True):
            points = np.asarray(r.theta)
            ax.vlines(points[:, 0], points[:, 2], points[:, 3], color=BLUE, lw=1.4, alpha=0.7)
            ax.plot(points[:, 0], points[:, 1], "o", ms=5, color=BLUE, mec=SURFACE, mew=1)
            ax.set_title(f"{r.spec}: {r.coverage['theta'][1]}/{r.coverage['theta'][2]} "
                         "inside 90%", fontsize=9.5)
            ax.set_xlabel("True country sum")
            ax.axline((0, 0), slope=1, color=MUTED, lw=1)
        axes[0][0].set_ylabel("Posterior median and 90% interval")
        fig.text(0.5, 1.02, "Recovery of known country sums on simulated data", ha="center",
                 color=INK2, fontsize=10)
        _finish(plt, fig, directory / "recovery.png", source, credit_y=-0.08)


def write_checks_report(fits_dir: Path, data_dir: Path, output_dir: Path,
                        report: Path | None = None, recovery_dir: Path | None = None,
                        figures: bool = True,
                        replication_dir: Path | None = None) -> ChecksReview:
    """Build the checks, write their tables, figures and the Markdown report."""
    review = build_checks(fits_dir, data_dir, recovery_dir, replication_dir)
    report = report or output_dir / "BAYES_MODEL_CHECKS.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_checks.json").write_text(json.dumps({
        "loo": {k: {key: val for key, val in asdict(v).items() if key != "pointwise"}
                for k, v in review.loo.items()},
        "comparisons": [asdict(c) for c in review.comparisons],
        "prior_predictive": [asdict(p) for p in review.prior],
        "recovery": [asdict(r) for r in review.recovery],
    }, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    if figures:
        source = ("Source: Eurostat, snapshot 2026-10-15; hierarchical Bayesian model "
                  "(BAYES_PROTOCOL.md, amendment 1). Descriptive, not causal.")
        write_figures(review, output_dir, source)
    assets = Path(os.path.relpath(output_dir, report.parent)).as_posix()
    command = (f"poetry run eu-okun bayes-checks --fits-dir {fits_dir.as_posix()} "
               f"--data-dir {data_dir.as_posix()} --output-dir {output_dir.as_posix()} "
               f"--report {report.as_posix()}"
               + (f" --recovery-dir {recovery_dir.as_posix()}" if recovery_dir else "")
               + (f" --replication-dir {replication_dir.as_posix()}"
                  if replication_dir else ""))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(review, assets, command), encoding="utf-8", newline="\n")
    return review
