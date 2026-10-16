"""Hierarchical Bayesian one-quarter-ahead forecasting models (``BAYES_FORECAST_PROTOCOL.md``).

Partially pooled regressions of the unemployment-rate change on predictors dated t-1
or earlier, refitted at every origin of the PR #149 forecasting framework. Expensive:
run by ``eu-okun forecast-bayes-fit`` and the separate sampling workflow, never by the
ordinary pull-request checks. Each origin is written to its own directory, so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.audit import audit_snapshot
from eu_okun.baseline import _read_panel
from eu_okun.bayes_fit import (
    PROTOCOL_SETTINGS,
    Diagnostics,
    SamplerSettings,
    compute_diagnostics,
    environment_versions,
)
from eu_okun.contract import EurostatDataError
from eu_okun.forecast_comparison import (
    ForecastPanel,
    bayes_forecast_predictions,
    build_forecast_panel,
)
from eu_okun.predictive import write_predictions

LOG_SIGMA_CENTRE = float(np.log(0.3))
REPLICATES = 4
RETRY_WARMUP = 3000
SAVED = ("alpha", "beta", "sigma", "nu", "lam", "mu_alpha", "tau_alpha", "mu_beta",
         "tau_beta", "mu_s", "tau_s")


@dataclass(frozen=True)
class ForecastSpec:
    """One pre-registered Bayesian forecasting model (protocol, section 3)."""

    name: str
    features: tuple[str, ...]
    common_factor: bool = False


FORECAST_SPECS: dict[str, ForecastSpec] = {
    s.name: s for s in (
        ForecastSpec("bayes_ar1", ("du_lag1",)),
        ForecastSpec("bayes_gdp_ar1", ("du_lag1", "gdp_lag1", "gdp_lag2")),
        ForecastSpec("bayes_gdp_ar1_factor", ("du_lag1", "gdp_lag1", "gdp_lag2"),
                     common_factor=True),
    )
}


def forecast_model(x: Any, y: Any, common_factor: bool) -> None:
    """Partially pooled Student-t regression; ``x`` (N, T, K), ``y`` (N, T) or None."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    n, t, k = x.shape
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0.0, 0.25))
    tau_alpha = numpyro.sample("tau_alpha", dist.HalfNormal(0.25))
    alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0.0, 1.0).expand((n,)).to_event(1))
    alpha = numpyro.deterministic("alpha", mu_alpha + tau_alpha * alpha_raw)
    mu_beta = numpyro.sample("mu_beta", dist.Normal(0.0, 0.5).expand((k,)).to_event(1))
    tau_beta = numpyro.sample("tau_beta", dist.HalfNormal(0.25).expand((k,)).to_event(1))
    beta_raw = numpyro.sample("beta_raw", dist.Normal(0.0, 1.0).expand((n, k)).to_event(2))
    beta = numpyro.deterministic("beta", mu_beta[None, :] + tau_beta[None, :] * beta_raw)
    mu_s = numpyro.sample("mu_s", dist.Normal(LOG_SIGMA_CENTRE, 1.0))
    tau_s = numpyro.sample("tau_s", dist.HalfNormal(0.5))
    s_raw = numpyro.sample("s_raw", dist.Normal(0.0, 1.0).expand((n,)).to_event(1))
    sigma = numpyro.deterministic("sigma", jnp.exp(mu_s + tau_s * s_raw))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    mean = alpha[:, None] + jnp.einsum("ntk,nk->nt", x, beta)
    if common_factor:
        lam = numpyro.sample("lam", dist.HalfNormal(0.2).expand((n,)).to_event(1))
        f = numpyro.sample("f", dist.Normal(0.0, 1.0).expand((t,)).to_event(1))
        mean = mean + lam[:, None] * f[None, :]
    numpyro.sample("y", dist.StudentT(nu, mean, sigma[:, None]).to_event(2), obs=y)


@dataclass
class ForecastFit:
    """Posterior draws (chain, draw, ...) and diagnostics of one refit."""

    samples: dict[str, np.ndarray]
    diagnostics: Diagnostics
    settings: SamplerSettings
    runtime_seconds: float


def fit_forecast_model(x: np.ndarray, y: np.ndarray, spec: ForecastSpec,
                       settings: SamplerSettings = PROTOCOL_SETTINGS) -> ForecastFit:
    """NUTS with the protocol settings (float64); diagnostics on every sampled site."""
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS

    numpyro.enable_x64()
    if x.ndim != 3 or y.shape != x.shape[:2] or not np.isfinite(x).all() \
            or not np.isfinite(y).all():
        raise ValueError("Training data must be finite, x (N, T, K) and y (N, T).")
    kernel = NUTS(forecast_model, target_accept_prob=settings.target_accept,
                  max_tree_depth=settings.max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=settings.warmup, num_samples=settings.draws,
                num_chains=settings.chains, chain_method="sequential", progress_bar=False)
    started = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(settings.seed), jnp.asarray(x), jnp.asarray(y),
             spec.common_factor, extra_fields=("diverging", "energy", "num_steps"))
    runtime = time.perf_counter() - started
    raw = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
    diagnostics = compute_diagnostics(raw, extra, settings.max_tree_depth)
    return ForecastFit({k: v for k, v in raw.items() if k in SAVED}, diagnostics, settings,
                       runtime)


def flat_draws(samples: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Merge chains: (chain, draw, ...) -> (chain * draw, ...)."""
    return {name: values.reshape(-1, *values.shape[2:]) for name, values in samples.items()}


def _attempt(fit: ForecastFit) -> dict[str, Any]:
    return {**asdict(fit.diagnostics), "accepted": fit.diagnostics.accepted,
            "failures": fit.diagnostics.failures(), "warmup": fit.settings.warmup,
            "runtime_seconds": fit.runtime_seconds}


def refit_origin(fpanel: ForecastPanel, column: int, spec: ForecastSpec,
                 settings: SamplerSettings, retry_warmup: int | None,
                 ) -> tuple[ForecastFit, list[dict[str, Any]]]:
    """Fit on every quarter strictly before ``column``; one retry with more warm-up."""
    x, y = fpanel.design(spec.features)[:, :column], fpanel.y[:, :column]
    fit = fit_forecast_model(x, y, spec, settings)
    attempts = [_attempt(fit)]
    if not fit.diagnostics.accepted and retry_warmup is not None:
        fit = fit_forecast_model(x, y, spec, replace(settings, warmup=retry_warmup))
        attempts.append(_attempt(fit))
    return fit, attempts


def run_forecast_refits(
    data_dir: Path, output_dir: Path, models: Sequence[str],
    settings: SamplerSettings = PROTOCOL_SETTINGS, retry_warmup: int | None = RETRY_WARMUP,
    targets: Sequence[str] | None = None, resume: bool = True,
) -> list[dict[str, Any]]:
    """Refit each model at each origin; write ``predictions.csv`` and ``origin.json``.

    ``origin.json`` is written last, so its presence marks a finished origin.
    """
    unknown = [m for m in models if m not in FORECAST_SPECS]
    if unknown:
        raise ValueError(f"Unknown forecasting models: {', '.join(unknown)}.")
    audit = audit_snapshot(data_dir)
    end = audit["analysis_period"]["end"]
    if not isinstance(end, str):
        raise EurostatDataError("Invalid study endpoint.")
    fpanel = build_forecast_panel(_read_panel(data_dir / "processed" / "eu27_quarterly.csv"),
                                  end)
    snapshot = {"manifest_sha256": audit["snapshot_manifest_sha256"],
                "processed_csv_sha256": audit["processed_csv_sha256"]}
    environment = environment_versions()
    records: list[dict[str, Any]] = []
    for model in models:
        spec = FORECAST_SPECS[model]
        for column in fpanel.origins:
            quarter = fpanel.quarters[column]
            if targets is not None and quarter not in targets:
                continue
            directory = output_dir / model / quarter
            done = directory / "origin.json"
            if resume and done.exists():
                records.append(json.loads(done.read_text(encoding="utf-8")))
                continue
            fit, attempts = refit_origin(fpanel, column, spec, settings, retry_warmup)
            rows = bayes_forecast_predictions(model, fpanel, column, flat_draws(fit.samples),
                                              spec.features, spec.common_factor, REPLICATES,
                                              settings.seed + column)
            write_predictions(directory / "predictions.csv", rows)
            record = {
                "model": model, "target": quarter, "features": list(spec.features),
                "common_factor": spec.common_factor,
                "training_first": fpanel.quarters[0], "training_last": fpanel.quarters[column - 1],
                "training_quarters": column, "countries": len(fpanel.countries),
                "accepted": fit.diagnostics.accepted, "attempts": attempts,
                "settings": asdict(fit.settings), "predictive_seed": settings.seed + column,
                "replicates": REPLICATES, "environment": environment, "snapshot": snapshot,
            }
            done.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n",
                            encoding="utf-8", newline="\n")
            records.append(record)
    if not records:
        raise EurostatDataError("No origin was fitted.")
    return records
