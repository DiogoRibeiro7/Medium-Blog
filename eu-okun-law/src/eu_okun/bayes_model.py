"""Hierarchical Bayesian Okun model in NumPyro (see ``BAYES_PROTOCOL.md``).

The likelihood is written as an explicit function of the parameters so that it can
be checked against an independent SciPy implementation. JAX and NumPyro are imported
here only; the rest of the package does not need the ``bayes`` extra.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from eu_okun.bayes_data import PANDEMIC_EXCLUDED_WINDOWS, PRIMARY_WINDOWS, BayesData

LOG_SIGMA_CENTRE = math.log(0.3)


@dataclass(frozen=True)
class ModelSpec:
    """One pre-registered specification (``BAYES_PROTOCOL.md``, section 5)."""

    name: str
    description: str
    pooling: bool = True
    common_factor: bool = True
    ar1: bool = True
    student_t: bool = True
    participation: bool = False
    prior_scale: float = 1.0
    tau_theta_scale: float = 0.25
    lags: int = 2
    windows: tuple[tuple[str, str], ...] = PRIMARY_WINDOWS
    time_effects: bool = False   # additive quarter effects, sum to zero (amendment 1)


PRIMARY = ModelSpec(
    "primary",
    "Partial pooling, K=2, 2009-Q2..2019-Q4, Student-t AR(1) errors and a common factor",
)
SPECS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        PRIMARY,
        replace(PRIMARY, name="wide_priors", description="All location and scale priors "
                "widened four-fold", prior_scale=4.0),
        replace(PRIMARY, name="strong_pooling", description="tau_theta ~ HalfNormal(0.05)",
                tau_theta_scale=0.05),
        replace(PRIMARY, name="no_pooling", description="Independent country parameters, "
                "no hyperparameters", pooling=False),
        replace(PRIMARY, name="no_common_factor", description="No common quarterly factor",
                common_factor=False),
        replace(PRIMARY, name="gaussian_iid", description="Normal errors, no AR(1), no "
                "common factor (closest to OLS)", common_factor=False, ar1=False,
                student_t=False),
        replace(PRIMARY, name="participation", description="Adds the participation change, "
                "partially pooled", participation=True),
        replace(PRIMARY, name="lags_4", description="GDP lags 0..4", lags=4),
        replace(PRIMARY, name="pandemic_excluded", description="2009-Q2..2019-Q4 and "
                "2022-Q3..2026-Q2 as two segments", windows=PANDEMIC_EXCLUDED_WINDOWS),
    )
}
# The nine specifications of BAYES_PROTOCOL.md, section 5: the robustness and claim
# judgements of section 7 use these only.
PREREGISTERED: tuple[str, ...] = tuple(SPECS)
# Amendment 1 (section 9): additive shared quarter effects, with and without
# participation, registered before any fit.
AMENDMENT_1: tuple[str, ...] = (
    "time_effects", "time_effects_participation", "participation_no_common_factor",
)
SPECS.update({
    spec.name: spec
    for spec in (
        replace(PRIMARY, name="time_effects", description="Additive shared quarter effects "
                "(sum to zero) instead of the common factor", common_factor=False,
                time_effects=True),
        replace(PRIMARY, name="time_effects_participation", description="Additive shared "
                "quarter effects plus the participation change", common_factor=False,
                time_effects=True, participation=True),
        replace(PRIMARY, name="participation_no_common_factor", description="The "
                "participation change, no common factor and no quarter effects",
                common_factor=False, participation=True),
    )
})


def model_inputs(data: BayesData) -> dict[str, Any]:
    """Convert the NumPy arrays of ``BayesData`` to JAX arrays for the model."""
    return {
        "y": jnp.asarray(data.y), "g": jnp.asarray(data.g), "z": jnp.asarray(data.z),
        "dp": jnp.asarray(data.dp), "mask": jnp.asarray(data.mask),
        "segment_start": jnp.asarray(data.segment_start),
    }


def conditional_mean(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    """alpha + theta*g + sum_k b_k*(g_{t-k} - g_t) [+ gamma*dp] [+ lambda*f] [+ delta_t].

    Shape (N, T). ``quarter_effect`` (delta) is the additive shared quarter effect.
    """
    mean = (
        params["alpha"][:, None]
        + params["theta"][:, None] * inputs["g"]
        + jnp.einsum("ntk,nk->nt", inputs["z"], params["b"])
    )
    if "gamma" in params:
        mean = mean + params["gamma"][:, None] * inputs["dp"]
    if "lam" in params:
        mean = mean + params["lam"][:, None] * params["f"][None, :]
    if "quarter_effect" in params:
        mean = mean + params["quarter_effect"][None, :]
    return mean


def innovations(
    inputs: Mapping[str, Any], params: Mapping[str, Any], ar1: bool,
) -> tuple[Any, Any]:
    """Innovations ``u`` and their scales after removing the mean and the AR(1) part.

    With e = y - mean, u[t] = e[t] - phi*e[t-1] inside a segment. At a segment's first
    quarter u = e with the stationary scale sigma / sqrt(1 - phi^2). The map from y to
    u is triangular with a unit diagonal, so its Jacobian is one.
    """
    resid = inputs["y"] - conditional_mean(inputs, params)
    sigma = params["sigma"][:, None]
    if not ar1:
        return resid, jnp.broadcast_to(sigma, resid.shape)
    phi = params["phi"][:, None]
    lagged = jnp.concatenate([jnp.zeros_like(resid[:, :1]), resid[:, :-1]], axis=1)
    start = inputs["segment_start"][None, :]
    u = jnp.where(start, resid, resid - phi * lagged)
    scale = jnp.where(start, sigma / jnp.sqrt(1.0 - phi**2), sigma)
    return u, scale


def pointwise_log_likelihood(
    inputs: Mapping[str, Any], params: Mapping[str, Any], spec: ModelSpec,
) -> Any:
    """Log density of each (country, quarter) term, before masking; shape (N, T)."""
    u, scale = innovations(inputs, params, spec.ar1)
    if spec.student_t:
        return dist.StudentT(params["nu"], 0.0, scale).log_prob(u)
    return dist.Normal(0.0, scale).log_prob(u)


def _pooled(
    name: str, shape: tuple[int, ...], location: float, scale: float, spec: ModelSpec,
    location_centre: float = 0.0,
) -> Any:
    """Non-centred partial pooling: name = mu_name + tau_name * name_raw."""
    s = spec.prior_scale
    mu = numpyro.sample(f"mu_{name}", dist.Normal(location_centre, location * s))
    tau = numpyro.sample(f"tau_{name}", dist.HalfNormal(scale * s))
    raw = numpyro.sample(f"{name}_raw", dist.Normal(0.0, 1.0).expand(shape).to_event(len(shape)))
    return numpyro.deterministic(name, mu + tau * raw)


def okun_model(inputs: Mapping[str, Any], spec: ModelSpec) -> None:
    """The hierarchical model of ``BAYES_PROTOCOL.md`` for one specification."""
    n, t = inputs["y"].shape
    k = inputs["z"].shape[2]
    s = spec.prior_scale
    params: dict[str, Any] = {}
    if spec.pooling:
        mu_theta = numpyro.sample("mu_theta", dist.Normal(0.0, 0.5 * s))
        tau_theta = numpyro.sample("tau_theta", dist.HalfNormal(spec.tau_theta_scale * s))
        eta = numpyro.sample("theta_raw", dist.Normal(0.0, 1.0).expand((n,)).to_event(1))
        params["theta"] = numpyro.deterministic("theta", mu_theta + tau_theta * eta)
        params["alpha"] = _pooled("alpha", (n,), 0.25, 0.25, spec)
        mu_b = numpyro.sample("mu_b", dist.Normal(0.0, 0.5 * s).expand((k,)).to_event(1))
        tau_b = numpyro.sample("tau_b", dist.HalfNormal(0.25 * s))
        xi = numpyro.sample("b_raw", dist.Normal(0.0, 1.0).expand((n, k)).to_event(2))
        params["b"] = numpyro.deterministic("b", mu_b[None, :] + tau_b * xi)
        if spec.participation:
            params["gamma"] = _pooled("gamma", (n,), 0.5, 0.25, spec)
        if spec.ar1:
            raw_phi = _pooled("phi_logit", (n,), 0.5, 0.5, spec)
            params["phi"] = numpyro.deterministic("phi", jnp.tanh(raw_phi))
        log_sigma = _pooled("log_sigma", (n,), 1.0, 0.5, spec, LOG_SIGMA_CENTRE)
    else:
        params["theta"] = numpyro.sample(
            "theta", dist.Normal(0.0, 0.5 * s).expand((n,)).to_event(1))
        params["alpha"] = numpyro.sample(
            "alpha", dist.Normal(0.0, 0.5 * s).expand((n,)).to_event(1))
        params["b"] = numpyro.sample(
            "b", dist.Normal(0.0, 0.5 * s).expand((n, k)).to_event(2))
        if spec.participation:
            params["gamma"] = numpyro.sample(
                "gamma", dist.Normal(0.0, 0.5 * s).expand((n,)).to_event(1))
        if spec.ar1:
            raw_phi = numpyro.sample(
                "phi_logit", dist.Normal(0.0, 0.5 * s).expand((n,)).to_event(1))
            params["phi"] = numpyro.deterministic("phi", jnp.tanh(raw_phi))
        log_sigma = numpyro.sample(
            "log_sigma", dist.Normal(LOG_SIGMA_CENTRE, 1.0 * s).expand((n,)).to_event(1))
    params["sigma"] = numpyro.deterministic("sigma", jnp.exp(log_sigma))
    if spec.common_factor and spec.time_effects:
        raise ValueError("Quarter effects and the common factor are never combined.")
    if spec.common_factor:
        params["lam"] = numpyro.sample(
            "lam", dist.HalfNormal(0.2 * s).expand((n,)).to_event(1))
        params["f"] = numpyro.sample("f", dist.Normal(0.0, 1.0).expand((t,)).to_event(1))
    if spec.time_effects:
        # Sum to zero over the sample quarters: the intercepts carry the level, so the
        # quarter effects are identified (BAYES_PROTOCOL.md, section 9.1).
        tau_quarter = numpyro.sample("tau_quarter", dist.HalfNormal(0.25 * s))
        params["quarter_effect"] = numpyro.sample(
            "quarter_effect", dist.ZeroSumNormal(tau_quarter, event_shape=(t,)))
    if spec.student_t:
        params["nu"] = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    numpyro.deterministic(
        "beta0", params["theta"] - jnp.sum(params["b"], axis=1)
    )
    terms = pointwise_log_likelihood(inputs, params, spec)
    numpyro.factor("log_likelihood", jnp.sum(jnp.where(inputs["mask"], terms, 0.0)))
