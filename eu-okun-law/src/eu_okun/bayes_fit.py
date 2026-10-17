"""Sampling, convergence diagnostics and posterior predictive checks.

Expensive: a full run of the nine pre-registered specifications takes minutes to tens
of minutes on a CPU. It is run by ``eu-okun bayes-fit`` and the separate Bayesian
workflow, never by the ordinary pull-request checks.
"""

from __future__ import annotations

import json
import platform
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.audit import audit_snapshot
from eu_okun.baseline import _read_panel, fit_hac_covariance
from eu_okun.bayes_data import BayesData, ar1_path, ardl_path, regime_terms
from eu_okun.bayes_model import (
    SPECS,
    ModelSpec,
    design_for,
    likelihood_mask,
    model_inputs,
    okun_model,
)
from eu_okun.comparison import cumulative_gdp
from eu_okun.contract import EurostatDataError
from eu_okun.review import read_flags

RHAT_MAX = 1.01
ESS_MIN = 400.0
BFMI_MIN = 0.3
SAVED = ("theta", "beta0", "b", "alpha", "phi", "sigma", "lam", "nu", "gamma", "mu_theta",
         "tau_theta", "mu_b", "tau_b", "f", "quarter_effect", "tau_quarter", "psi", "mu_psi",
         "tau_psi", "rho_f", "delta", "mu_delta", "tau_delta", "theta_post", "alpha_shift",
         "mu_alpha_shift", "tau_alpha_shift", "log_sigma_shift", "kappa", "tau_kappa")
REGIME_KEYS = ("theta", "alpha", "sigma", "delta", "kappa", "alpha_shift", "log_sigma_shift")


@dataclass(frozen=True)
class SamplerSettings:
    """NUTS settings fixed in the protocol (section 6)."""

    chains: int = 4
    warmup: int = 1000
    draws: int = 1000
    target_accept: float = 0.95
    max_tree_depth: int = 10
    seed: int = 20261016


PROTOCOL_SETTINGS = SamplerSettings()


@dataclass(frozen=True)
class Diagnostics:
    """Convergence summary and the pre-registered acceptance decision."""

    max_rhat: float
    max_rhat_parameter: str
    min_ess_bulk: float
    min_ess_tail: float
    min_ess_parameter: str
    divergences: int
    min_bfmi: float
    tree_depth_saturated: int
    draws_total: int

    @property
    def accepted(self) -> bool:
        return (self.max_rhat <= RHAT_MAX and self.min_ess_bulk >= ESS_MIN
                and self.min_ess_tail >= ESS_MIN and self.divergences == 0
                and self.min_bfmi >= BFMI_MIN)

    def failures(self) -> list[str]:
        """Which acceptance criteria failed, in words."""
        problems = []
        if self.max_rhat > RHAT_MAX:
            problems.append(f"R-hat {self.max_rhat:.3f} > {RHAT_MAX} ({self.max_rhat_parameter})")
        if min(self.min_ess_bulk, self.min_ess_tail) < ESS_MIN:
            problems.append(f"ESS {min(self.min_ess_bulk, self.min_ess_tail):.0f} < {ESS_MIN:.0f} "
                            f"({self.min_ess_parameter})")
        if self.divergences:
            problems.append(f"{self.divergences} divergent transitions")
        if self.min_bfmi < BFMI_MIN:
            problems.append(f"E-BFMI {self.min_bfmi:.2f} < {BFMI_MIN}")
        return problems


def _flatten(samples: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Split vector/matrix parameters into scalar series of shape (chain, draw)."""
    out: dict[str, np.ndarray] = {}
    for name, values in samples.items():
        if values.ndim == 2:
            out[name] = values
            continue
        for index in np.ndindex(values.shape[2:]):
            out[f"{name}[{','.join(map(str, index))}]"] = values[(slice(None), slice(None), *index)]
    return out


def compute_diagnostics(
    samples: Mapping[str, np.ndarray], extra: Mapping[str, np.ndarray], max_tree_depth: int,
) -> Diagnostics:
    """R-hat (rank-normalised), bulk and tail ESS, divergences, E-BFMI, tree depth."""
    import arviz as az

    series = _flatten(samples)
    if not series:
        raise ValueError("No posterior samples to diagnose.")
    names = list(series)
    stacked = np.stack([series[n] for n in names], axis=-1)
    rhat = np.asarray(az.rhat(stacked, chain_axis=0, draw_axis=1))
    bulk = np.asarray(az.ess(stacked, method="bulk", chain_axis=0, draw_axis=1))
    tail = np.asarray(az.ess(stacked, method="tail", prob=(0.05, 0.95), chain_axis=0,
                             draw_axis=1))
    # ESS is undefined for a parameter fixed by construction; R-hat is undefined with a
    # single chain, which therefore never passes acceptance.
    finite = np.isfinite(bulk) & np.isfinite(tail)
    if not finite.any():
        raise ValueError("Diagnostics are undefined for every parameter.")
    rhat_defined = np.isfinite(rhat)
    if rhat_defined.any():
        worst_rhat = int(np.argmax(np.where(rhat_defined, rhat, -np.inf)))
        max_rhat, rhat_name = float(rhat[worst_rhat]), names[worst_rhat]
    else:
        max_rhat, rhat_name = float("inf"), "undefined (single chain)"
    ess_floor = np.minimum(bulk, tail)
    worst_ess = int(np.argmin(np.where(finite, ess_floor, np.inf)))
    energy = np.asarray(extra["energy"])
    steps = np.asarray(extra["num_steps"])
    return Diagnostics(
        max_rhat=max_rhat, max_rhat_parameter=rhat_name,
        min_ess_bulk=float(np.min(bulk[finite])), min_ess_tail=float(np.min(tail[finite])),
        min_ess_parameter=names[worst_ess],
        divergences=int(np.sum(extra["diverging"])),
        min_bfmi=float(np.min(np.asarray(az.bfmi(energy)))),
        tree_depth_saturated=int(np.sum(steps >= 2**max_tree_depth - 1)),
        draws_total=int(energy.size),
    )


@dataclass
class FitResult:
    """Posterior draws (chain, draw, ...) and diagnostics for one specification."""

    spec: ModelSpec
    data: BayesData
    settings: SamplerSettings
    samples: dict[str, np.ndarray]
    diagnostics: Diagnostics
    runtime_seconds: float
    sample_stats: dict[str, np.ndarray] = field(default_factory=dict)


def fit_specification(
    data: BayesData, spec: ModelSpec, settings: SamplerSettings = PROTOCOL_SETTINGS,
) -> FitResult:
    """Run NUTS for one specification on prepared data (float64 throughout)."""
    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS

    numpyro.enable_x64()
    if data.lags != spec.lags:
        raise ValueError("Data and specification use different GDP lag orders.")
    kernel = NUTS(okun_model, target_accept_prob=settings.target_accept,
                  max_tree_depth=settings.max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=settings.warmup, num_samples=settings.draws,
                num_chains=settings.chains, chain_method="sequential", progress_bar=False)
    started = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(settings.seed), model_inputs(data), spec,
             extra_fields=("diverging", "energy", "num_steps"))
    runtime = time.perf_counter() - started
    raw = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
    diagnostics = compute_diagnostics(raw, extra, settings.max_tree_depth)
    kept = {k: v for k, v in raw.items() if k in SAVED}
    return FitResult(spec, data, settings, kept, diagnostics, runtime, extra)


# ----------------------------------------------------------------------------------
# Posterior predictive checks (NumPy, from saved draws)
# ----------------------------------------------------------------------------------


def _draws(samples: Mapping[str, np.ndarray], name: str) -> np.ndarray | None:
    values = samples.get(name)
    return None if values is None else values.reshape(-1, *values.shape[2:])


def _regime_draw(data: BayesData, s: Mapping[str, np.ndarray], d: int,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """theta, alpha and sigma of draw ``d`` by country and quarter (amendment 4)."""
    return regime_terms(data, {k: s[k][d] for k in REGIME_KEYS if k in s})


def _mean_without_factor(data: BayesData, s: Mapping[str, np.ndarray], d: int) -> np.ndarray:
    theta, alpha, _ = _regime_draw(data, s, d)
    mean = alpha + theta * data.g + np.einsum("ntk,nk->nt", data.z, s["b"][d])
    if "gamma" in s:
        mean = mean + s["gamma"][d][:, None] * data.dp
    return np.asarray(mean, dtype=float)


def _innovation(spec: ModelSpec, nu: float | None, size: tuple[int, ...],
                rng: np.random.Generator) -> np.ndarray:
    if spec.student_t and nu is not None:
        return rng.standard_t(nu, size=size)
    return rng.standard_normal(size)


def _masked_stats(data: BayesData, y: np.ndarray) -> dict[str, float]:
    """Mean lag-1 autocorrelation, mean pairwise correlation and mean SD, unmasked terms."""
    mask = data.mask
    pairs = mask[:, 1:] & mask[:, :-1] & ~data.segment_start[None, 1:]
    ac = []
    for i in range(data.n_countries):
        a, b = y[i, 1:][pairs[i]], y[i, :-1][pairs[i]]
        if a.size > 3 and a.std() > 0 and b.std() > 0:
            ac.append(float(np.corrcoef(a, b)[0, 1]))
    sds = [float(y[i][mask[i]].std()) for i in range(data.n_countries)]
    common = mask.all(axis=0)
    corr = np.corrcoef(y[:, common])
    off = corr[~np.eye(data.n_countries, dtype=bool)]
    return {"lag1_autocorrelation": float(np.mean(ac)), "cross_country_correlation":
            float(np.mean(off)), "mean_sd": float(np.mean(sds))}


@dataclass(frozen=True)
class PredictiveCheck:
    """Calibration and replicated-statistic checks for one fit."""

    coverage_50: float
    coverage_90: float
    observed: dict[str, float]
    replicated_mean: dict[str, float]
    p_value: dict[str, float]
    draws_used: int


def predictive_checks(fit: FitResult, max_draws: int = 400, seed: int = 7) -> PredictiveCheck:
    """One-step-ahead interval coverage and replicated summary statistics.

    One-step predictions condition on the observed previous residual (AR(1)) or change
    (ARDL) and the posterior factor value; replications simulate whole series with fresh
    innovations and fresh factor values (an AR(1) path when the factor is dynamic). ARDL
    replications start from the observed first quarter of each segment, on which the
    likelihood conditions. Coverage, near 0.5 and 0.9 when intervals are calibrated, is
    taken over the cells of the likelihood.
    """
    rng = np.random.default_rng(seed)
    data, spec = fit.data, fit.spec
    flat = {k: v for k in fit.samples if (v := _draws(fit.samples, k)) is not None}
    total = flat["theta"].shape[0]
    chosen = np.linspace(0, total - 1, min(max_draws, total)).astype(int)
    n, t = data.y.shape
    one_step = np.empty((chosen.size, n, t))
    stats: dict[str, list[float]] = {"lag1_autocorrelation": [], "cross_country_correlation": [],
                                     "mean_sd": []}
    y_lag = data.y_lag()
    for row, d in enumerate(chosen):
        mean = _mean_without_factor(data, flat, d)
        if spec.common_factor:
            mean_obs = mean + flat["lam"][d][:, None] * flat["f"][d][None, :]
        else:
            mean_obs = mean
        if spec.time_effects:
            mean_obs = mean_obs + flat["quarter_effect"][d][None, :]
        if spec.lagged_outcome:
            mean_obs = mean_obs + flat["psi"][d][:, None] * y_lag
        sigma = _regime_draw(data, flat, d)[2]
        nu = float(flat["nu"][d]) if "nu" in flat else None
        resid = data.y - mean_obs
        phi_vector = np.asarray(flat["phi"][d], dtype=float) if spec.ar1 else np.zeros(n)
        phi = phi_vector[:, None]
        if spec.ar1:
            lagged = np.concatenate([np.zeros((n, 1)), resid[:, :-1]], axis=1)
            start = data.segment_start[None, :]
            centre = np.where(start, mean_obs, mean_obs + phi * lagged)
            scale = np.where(start, sigma / np.sqrt(1 - phi**2), sigma)
        else:
            centre, scale = mean_obs, np.broadcast_to(sigma, (n, t))
        one_step[row] = centre + scale * _innovation(spec, nu, (n, t), rng)
        # Whole-series replication with fresh factor values and innovations.
        noise = sigma * _innovation(spec, nu, (n, t), rng)
        if spec.ar1:
            e = np.empty((n, t))
            for j in range(t):
                if data.segment_start[j]:
                    e[:, j] = noise[:, j] / np.sqrt(1 - phi_vector**2)
                else:
                    e[:, j] = phi_vector * e[:, j - 1] + noise[:, j]
        else:
            e = noise
        replicated = mean + e
        if spec.common_factor:
            fresh_factor = rng.standard_normal((t,))
            if spec.factor_ar:
                fresh_factor = ar1_path(fresh_factor, float(flat["rho_f"][d]),
                                        data.segment_start)
            replicated = replicated + flat["lam"][d][:, None] * fresh_factor[None, :]
        if spec.time_effects:
            # Fresh quarter effects from their ZeroSumNormal(tau) distribution.
            fresh = np.asarray(rng.normal(0.0, float(flat["tau_quarter"][d]), t))
            replicated = replicated + (fresh - fresh.mean())[None, :]
        if spec.lagged_outcome:
            replicated = ardl_path(replicated, flat["psi"][d], data.y, data.segment_start)
        for key, value in _masked_stats(data, replicated).items():
            stats[key].append(value)
    lower50, upper50 = np.quantile(one_step, [0.25, 0.75], axis=0)
    lower90, upper90 = np.quantile(one_step, [0.05, 0.95], axis=0)
    inside50 = (data.y >= lower50) & (data.y <= upper50)
    inside90 = (data.y >= lower90) & (data.y <= upper90)
    observed = _masked_stats(data, data.y)
    cells = likelihood_mask(data.mask, data.segment_start, spec)
    return PredictiveCheck(
        coverage_50=float(inside50[cells].mean()),
        coverage_90=float(inside90[cells].mean()),
        observed=observed,
        replicated_mean={k: float(np.mean(v)) for k, v in stats.items()},
        p_value={k: float(np.mean(np.asarray(v) >= observed[k])) for k, v in stats.items()},
        draws_used=int(chosen.size),
    )


# ----------------------------------------------------------------------------------
# Classical comparison on identical rows
# ----------------------------------------------------------------------------------


def classical_on_same_rows(data: BayesData) -> dict[str, dict[str, float]]:
    """OLS with calendar-aware HAC (bandwidth 4) on the unmasked rows of each country."""
    design = data.gdp_design()
    out: dict[str, dict[str, float]] = {}
    for i, geo in enumerate(data.countries):
        rows = data.mask[i]
        x, y = design[i][rows], data.y[i][rows]
        periods = [q for q, keep in zip(data.quarters, rows, strict=True) if keep]
        beta, covariance, _, _ = fit_hac_covariance(x, y, maxlags=4, periods=periods)
        total, se, low, high = cumulative_gdp(beta, covariance, data.lags + 1)
        out[geo] = {"theta": total, "se": se, "ci_low": low, "ci_high": high, "n": float(y.size)}
    return out


# ----------------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------------


def environment_versions() -> dict[str, str]:
    """Library and platform versions that sampler output depends on."""
    import arviz
    import jax
    import numpyro

    return {"python": platform.python_version(), "platform": platform.platform(),
            "jax": jax.__version__, "numpyro": numpyro.__version__,
            "arviz": arviz.__version__, "numpy": np.__version__}


def save_fit(fit: FitResult, check: PredictiveCheck, directory: Path,
             snapshot: Mapping[str, Any], netcdf: bool = True) -> None:
    """Write draws (``draws.npz``), a JSON summary and the NetCDF artifact of one fit."""
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = dict(fit.samples)
    np.savez_compressed(directory / "draws.npz", **arrays)
    document = {
        "spec": asdict(fit.spec),
        "settings": asdict(fit.settings),
        "countries": list(fit.data.countries),
        "quarters": list(fit.data.quarters),
        "masked_terms": int((~fit.data.mask).sum()),
        "likelihood_terms": int(likelihood_mask(fit.data.mask, fit.data.segment_start,
                                                fit.spec).sum()),
        "masked_flagged_quarters": fit.data.masked_reasons,
        "diagnostics": {**asdict(fit.diagnostics), "accepted": fit.diagnostics.accepted,
                        "failures": fit.diagnostics.failures()},
        "predictive_check": asdict(check),
        "classical_same_rows": classical_on_same_rows(fit.data),
        "runtime_seconds": fit.runtime_seconds,
        "environment": environment_versions(),
        "snapshot": dict(snapshot),
    }
    (directory / "fit.json").write_text(
        json.dumps(document, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    if netcdf:
        from eu_okun.bayes_artifacts import write_posterior_netcdf

        write_posterior_netcdf(directory / "posterior.nc", fit.spec, fit.data, fit.samples,
                               fit.sample_stats, {
                                   "settings": document["settings"],
                                   "environment": document["environment"],
                                   "snapshot": document["snapshot"],
                                   "diagnostics": document["diagnostics"]})


def run_specifications(
    data_dir: Path, output_dir: Path, names: Sequence[str],
    settings: SamplerSettings = PROTOCOL_SETTINGS, retry_warmup: int | None = None,
) -> list[tuple[str, Diagnostics]]:
    """Audit the snapshot, then fit and save each named specification.

    With ``retry_warmup``, a specification that fails the acceptance criteria is fitted
    once more with that many warm-up iterations (protocol, section 6: more warm-up is
    allowed, a different model or prior is not). The first attempt is kept under
    ``<output_dir>-first-attempt/<name>``.
    """
    unknown = [name for name in names if name not in SPECS]
    if unknown:
        raise ValueError(f"Unknown specifications: {', '.join(unknown)}.")
    audit = audit_snapshot(data_dir)
    csv_path = data_dir / "processed" / "eu27_quarterly.csv"
    panel, flags = _read_panel(csv_path), read_flags(csv_path)
    snapshot = {"manifest_sha256": audit["snapshot_manifest_sha256"],
                "processed_csv_sha256": audit["processed_csv_sha256"]}
    results = []
    for name in names:
        spec = SPECS[name]
        data = design_for(spec, panel, flags)
        fit = fit_specification(data, spec, settings)
        save_fit(fit, predictive_checks(fit), output_dir / name, snapshot)
        if not fit.diagnostics.accepted and retry_warmup is not None:
            first = output_dir.parent / f"{output_dir.name}-first-attempt" / name
            if first.exists():
                shutil.rmtree(first)
            first.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(output_dir / name), str(first))
            fit = fit_specification(data, spec, replace(settings, warmup=retry_warmup))
            save_fit(fit, predictive_checks(fit), output_dir / name, snapshot)
        results.append((name, fit.diagnostics))
    if not results:
        raise EurostatDataError("No specification was fitted.")
    return results


def interval(values: np.ndarray, probability: float = 0.9) -> tuple[float, float]:
    """Equal-tailed credible interval."""
    tail = (1.0 - probability) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)

