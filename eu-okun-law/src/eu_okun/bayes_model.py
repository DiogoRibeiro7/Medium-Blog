"""Hierarchical Bayesian Okun model in NumPyro (see ``BAYES_PROTOCOL.md``).

The likelihood is written as an explicit function of the parameters so that it can
be checked against an independent SciPy implementation. JAX and NumPyro are imported
here only; the rest of the package does not need the ``bayes`` extra.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

from eu_okun.bayes_data import (
    PANDEMIC_EXCLUDED_WINDOWS,
    PRIMARY_WINDOWS,
    BayesData,
    build_bayes_data,
)
from eu_okun.contract import EU27
from eu_okun.review import Flags, Panel

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
    factor_ar: bool = False      # the common factor follows a stationary AR(1) (amendment 3)
    lagged_outcome: bool = False  # psi[i] * y[i, t-1] in the mean, an ARDL (amendment 3)
    regime_shift: bool = False   # post-2021 shift delta[i] of the GDP sum (amendment 4)
    drift: bool = False          # common random walk kappa[t] in the GDP sum (amendment 4)
    final_gdp_only: bool = False  # mask terms whose GDP inputs are provisional (amendment 4)

    @property
    def post_shifts(self) -> bool:
        """Post-2021 intercept and error-scale shifts come with a shift or a drift."""
        return self.regime_shift or self.drift


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
# Amendment 2 (section 10): the lag order K = 1, registered before its fit.
AMENDMENT_2: tuple[str, ...] = ("lags_1",)
SPECS["lags_1"] = replace(PRIMARY, name="lags_1", description="GDP lags 0..1", lags=1)
# Amendment 3 (section 11): dynamics that address the misfits, registered before any fit.
AMENDMENT_3: tuple[str, ...] = ("dynamic_factor", "ardl", "ardl_dynamic_factor")
SPECS.update({
    spec.name: spec
    for spec in (
        replace(PRIMARY, name="dynamic_factor", description="The common factor follows a "
                "stationary AR(1)", factor_ar=True),
        replace(PRIMARY, name="ardl", description="Lagged unemployment change psi*y[t-1] "
                "instead of the AR(1) error; i.i.d. factor", ar1=False, lagged_outcome=True),
        replace(PRIMARY, name="ardl_dynamic_factor", description="Lagged unemployment change "
                "and an AR(1) common factor", ar1=False, lagged_outcome=True, factor_ar=True),
    )
})
# Amendment 4 (section 12): regime-specific and time-varying coefficients on the
# pandemic-excluded windows, registered before any fit.
AMENDMENT_4: tuple[str, ...] = (
    "regime_shift", "regime_shift_final_gdp", "pandemic_excluded_dynamic",
    "regime_shift_dynamic", "drifting_association",
)
_PANDEMIC_EXCLUDED = SPECS["pandemic_excluded"]
SPECS.update({
    spec.name: spec
    for spec in (
        replace(_PANDEMIC_EXCLUDED, name="regime_shift", description="Post-2021 shifts in the "
                "GDP coefficient sum, the intercept and the error scale", regime_shift=True),
        replace(_PANDEMIC_EXCLUDED, name="regime_shift_final_gdp", description="regime_shift "
                "without terms whose GDP inputs are provisional", regime_shift=True,
                final_gdp_only=True),
        replace(_PANDEMIC_EXCLUDED, name="pandemic_excluded_dynamic", description="The "
                "pandemic-excluded windows with the dynamics of ardl_dynamic_factor",
                ar1=False, lagged_outcome=True, factor_ar=True),
        replace(_PANDEMIC_EXCLUDED, name="regime_shift_dynamic", description="regime_shift "
                "with the dynamics of ardl_dynamic_factor", ar1=False, lagged_outcome=True,
                factor_ar=True, regime_shift=True),
        replace(_PANDEMIC_EXCLUDED, name="drifting_association", description="A common random "
                "walk in the GDP coefficient sum, with the post-2021 intercept and scale "
                "shifts", drift=True),
    )
})


def design_for(spec: ModelSpec, panel: Panel, flags: Flags,
               countries: Sequence[str] = EU27) -> BayesData:
    """The arrays a specification is fitted on (windows, lag order and masks)."""
    return build_bayes_data(panel, flags, spec.windows, spec.lags, countries,
                            provisional_gdp=not spec.final_gdp_only)


def model_inputs(data: BayesData) -> dict[str, Any]:
    """Convert the NumPy arrays of ``BayesData`` to JAX arrays for the model."""
    return {
        "y": jnp.asarray(data.y), "g": jnp.asarray(data.g), "z": jnp.asarray(data.z),
        "dp": jnp.asarray(data.dp), "mask": jnp.asarray(data.mask),
        "segment_start": jnp.asarray(data.segment_start), "y_lag": jnp.asarray(data.y_lag()),
        "post": jnp.asarray(data.post().astype(float)),
        "gap": jnp.asarray(data.calendar_gaps()),
    }


def likelihood_mask(mask: Any, segment_start: Any, spec: ModelSpec) -> Any:
    """Cells whose term enters the likelihood (NumPy or JAX arrays).

    The ARDL specifications condition on each segment's first quarter, whose lagged
    change lies outside the window (section 11.2); the others use every unmasked cell.
    """
    if spec.lagged_outcome:
        return mask & ~segment_start[None, :]
    return mask


def conditional_mean(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    """alpha + theta*g + sum_k b_k*(g_{t-k} - g_t) [+ gamma*dp] [+ lambda*f] [+ delta_t]
    [+ psi*y_{t-1}].

    Shape (N, T). ``quarter_effect`` (delta) is the additive shared quarter effect;
    ``psi`` multiplies the lagged unemployment change of the ARDL specifications. In
    amendment 4, theta and alpha shift after 2021 (``delta``, ``alpha_shift``) and theta
    drifts with the common random walk ``kappa``.
    """
    alpha = params["alpha"][:, None]
    theta = params["theta"][:, None]
    if "delta" in params:
        theta = theta + params["delta"][:, None] * inputs["post"][None, :]
    if "kappa" in params:
        theta = theta + params["kappa"][None, :]
    if "alpha_shift" in params:
        alpha = alpha + params["alpha_shift"][:, None] * inputs["post"][None, :]
    mean = alpha + theta * inputs["g"] + jnp.einsum("ntk,nk->nt", inputs["z"], params["b"])
    if "gamma" in params:
        mean = mean + params["gamma"][:, None] * inputs["dp"]
    if "lam" in params:
        mean = mean + params["lam"][:, None] * params["f"][None, :]
    if "quarter_effect" in params:
        mean = mean + params["quarter_effect"][None, :]
    if "psi" in params:
        mean = mean + params["psi"][:, None] * inputs["y_lag"]
    return mean


class SegmentAR1(dist.Distribution):  # type: ignore[misc]
    """Stationary AR(1) with unit variance, restarted at each segment's first quarter.

    f[t] = rho*f[t-1] + sqrt(1 - rho^2)*e[t] with e[t] ~ N(0, 1) inside a segment, and
    f[t] ~ N(0, 1) at a segment's first quarter (section 11.2). The density is that of
    ``f`` itself (the centred form), so the factor values are the sampled parameters.
    """

    arg_constraints = {"rho": constraints.open_interval(-1.0, 1.0)}
    support = constraints.real_vector

    def __init__(self, rho: Any, segment_start: Any, *, validate_args: bool | None = None):
        self.rho = rho
        self.segment_start = jnp.asarray(segment_start, dtype=bool)
        super().__init__(batch_shape=jnp.shape(rho), event_shape=self.segment_start.shape,
                         validate_args=validate_args)

    def sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        e = jax.random.normal(key, sample_shape + self.batch_shape + self.event_shape)
        rho = jnp.broadcast_to(self.rho, sample_shape + self.batch_shape)
        scale = jnp.sqrt(1.0 - rho**2)

        def step(previous: Any, inputs: tuple[Any, Any]) -> tuple[Any, Any]:
            e_t, start = inputs
            current = jnp.where(start, e_t, rho * previous + scale * e_t)
            return current, current

        _, path = jax.lax.scan(step, jnp.zeros_like(rho),
                               (jnp.moveaxis(e, -1, 0), self.segment_start))
        return jnp.moveaxis(path, 0, -1)

    def log_prob(self, value: Any) -> Any:
        rho = jnp.expand_dims(self.rho, -1)
        lagged = jnp.concatenate([jnp.zeros_like(value[..., :1]), value[..., :-1]], axis=-1)
        centre = jnp.where(self.segment_start, 0.0, rho * lagged)
        scale = jnp.where(self.segment_start, 1.0, jnp.sqrt(1.0 - rho**2))
        return jnp.sum(dist.Normal(centre, scale).log_prob(value), axis=-1)


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
    if "log_sigma_shift" in params:
        sigma = sigma * jnp.exp(params["log_sigma_shift"] * inputs["post"][None, :])
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
    if spec.common_factor and spec.time_effects:
        raise ValueError("Quarter effects and the common factor are never combined.")
    if spec.factor_ar and not spec.common_factor:
        raise ValueError("An AR(1) factor needs the common factor.")
    if spec.lagged_outcome and spec.ar1:
        raise ValueError("The lagged change replaces the AR(1) error; never both.")
    if spec.post_shifts and not spec.pooling:
        raise ValueError("Regime shifts and drifts are only defined with partial pooling.")
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
        if spec.lagged_outcome:
            # psi = tanh(mu_psi + tau_psi * omega), so |psi| < 1 (section 11.2).
            mu_psi = numpyro.sample("mu_psi", dist.Normal(0.0, 0.5 * s))
            tau_psi = numpyro.sample("tau_psi", dist.HalfNormal(0.5 * s))
            omega = numpyro.sample("psi_raw", dist.Normal(0.0, 1.0).expand((n,)).to_event(1))
            params["psi"] = numpyro.deterministic("psi", jnp.tanh(mu_psi + tau_psi * omega))
        log_sigma = _pooled("log_sigma", (n,), 1.0, 0.5, spec, LOG_SIGMA_CENTRE)
        if spec.regime_shift:
            # delta = mu_delta + tau_delta * raw, on the scales of theta (section 12.3).
            mu_delta = numpyro.sample("mu_delta", dist.Normal(0.0, 0.5 * s))
            tau_delta = numpyro.sample("tau_delta", dist.HalfNormal(spec.tau_theta_scale * s))
            raw_delta = numpyro.sample("delta_raw",
                                       dist.Normal(0.0, 1.0).expand((n,)).to_event(1))
            params["delta"] = numpyro.deterministic("delta", mu_delta + tau_delta * raw_delta)
            numpyro.deterministic("theta_post", params["theta"] + params["delta"])
        if spec.post_shifts:
            params["alpha_shift"] = _pooled("alpha_shift", (n,), 0.25, 0.25, spec)
            params["log_sigma_shift"] = numpyro.sample("log_sigma_shift",
                                                       dist.Normal(0.0, 0.5 * s))
        if spec.drift:
            # Random walk over calendar quarters (variance grows with the gap between
            # sample quarters), centred so that theta[i] is the time average.
            tau_kappa = numpyro.sample("tau_kappa", dist.HalfNormal(0.05 * s))
            steps = numpyro.sample("kappa_raw",
                                   dist.Normal(0.0, 1.0).expand((t - 1,)).to_event(1))
            walk = jnp.concatenate([jnp.zeros(1), jnp.cumsum(
                tau_kappa * jnp.sqrt(inputs["gap"][1:]) * steps)])
            params["kappa"] = numpyro.deterministic("kappa", walk - walk.mean())
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
        if spec.lagged_outcome:
            raw_psi = numpyro.sample(
                "psi_logit", dist.Normal(0.0, 0.5 * s).expand((n,)).to_event(1))
            params["psi"] = numpyro.deterministic("psi", jnp.tanh(raw_psi))
        log_sigma = numpyro.sample(
            "log_sigma", dist.Normal(LOG_SIGMA_CENTRE, 1.0 * s).expand((n,)).to_event(1))
    params["sigma"] = numpyro.deterministic("sigma", jnp.exp(log_sigma))
    if spec.common_factor:
        params["lam"] = numpyro.sample(
            "lam", dist.HalfNormal(0.2 * s).expand((n,)).to_event(1))
        if spec.factor_ar:
            rho_raw = numpyro.sample("rho_f_raw", dist.Normal(0.0, 1.0 * s))
            rho_f = numpyro.deterministic("rho_f", jnp.tanh(rho_raw))
            params["f"] = numpyro.sample("f", SegmentAR1(rho_f, inputs["segment_start"]))
        else:
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
    used = likelihood_mask(inputs["mask"], inputs["segment_start"], spec)
    numpyro.factor("log_likelihood", jnp.sum(jnp.where(used, terms, 0.0)))
