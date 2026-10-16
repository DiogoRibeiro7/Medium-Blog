"""ArviZ-compatible NetCDF artifacts of the Bayesian fits (``BAYES_PROTOCOL.md``, 9.5).

Each file is an xarray DataTree in the ArviZ group layout:

- ``posterior``: the saved draws, dims (chain, draw, ...);
- ``sample_stats``: divergences, energy and tree steps, when the sampler kept them;
- ``prior`` and ``prior_predictive``: draws from the prior and data simulated from them;
- ``observed_data`` and ``constant_data``: the outcome, regressors, mask and segments;
- ``log_likelihood``: pointwise log densities of the unmasked cells (for PSIS-LOO).

The root attributes record the specification, seeds, sampler settings, software versions
and snapshot hashes. Open with ``xarray.open_datatree(path, engine="h5netcdf")`` or
``arviz_base.from_netcdf(path)``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.bayes_data import BayesData
from eu_okun.bayes_model import ModelSpec, model_inputs, okun_model, pointwise_log_likelihood

PRIOR_DRAWS = 1000
PRIOR_SEED = 20261017
# Dimensions beyond (chain, draw) of every saved parameter.
DIMS: dict[str, tuple[str, ...]] = {
    "theta": ("country",), "beta0": ("country",), "alpha": ("country",),
    "phi": ("country",), "sigma": ("country",), "lam": ("country",), "gamma": ("country",),
    "b": ("country", "lag"), "mu_b": ("lag",), "f": ("quarter",),
    "quarter_effect": ("quarter",), "nu": (), "mu_theta": (), "tau_theta": (), "tau_b": (),
    "tau_quarter": (),
}
LIKELIHOOD_PARAMETERS = ("theta", "alpha", "b", "phi", "sigma", "lam", "f", "nu", "gamma",
                         "quarter_effect")


def _coords(data: BayesData) -> dict[str, list[str]]:
    return {"country": list(data.countries), "quarter": list(data.quarters),
            "lag": [f"lag{k + 1}" for k in range(data.lags)]}


# ----------------------------------------------------------------------------------
# Prior draws, simulated outcomes and pointwise log likelihood
# ----------------------------------------------------------------------------------


def prior_draws(spec: ModelSpec, data: BayesData, draws: int = PRIOR_DRAWS,
                seed: int = PRIOR_SEED) -> dict[str, np.ndarray]:
    """Parameter draws from the prior of ``spec`` on the design of ``data``, (draws, ...)."""
    import jax
    import numpyro
    from numpyro.infer import Predictive

    numpyro.enable_x64()
    predictive = Predictive(okun_model, num_samples=draws)
    sampled = predictive(jax.random.PRNGKey(seed), model_inputs(data), spec)
    return {name: np.asarray(value) for name, value in sampled.items() if name in DIMS}


def simulate_outcomes(spec: ModelSpec, data: BayesData, draws: Mapping[str, np.ndarray],
                      rng: np.random.Generator) -> np.ndarray:
    """One simulated data set per parameter draw, shape (S, N, T).

    Uses each draw's own shared terms (factor values or quarter effects) and fresh
    Student-t or Normal innovations through the AR(1) error, with the stationary scale
    at the first quarter of each segment.
    """
    s = draws["theta"].shape[0]
    n, t = data.y.shape
    mean = (draws["alpha"][:, :, None] + draws["theta"][:, :, None] * data.g[None]
            + np.einsum("ntk,snk->snt", data.z, draws["b"]))
    if spec.participation:
        mean = mean + draws["gamma"][:, :, None] * data.dp[None]
    if spec.common_factor:
        mean = mean + draws["lam"][:, :, None] * draws["f"][:, None, :]
    if spec.time_effects:
        mean = mean + draws["quarter_effect"][:, None, :]
    if spec.student_t:
        eps = rng.standard_t(np.asarray(draws["nu"]).reshape(s, 1, 1), size=(s, n, t))
    else:
        eps = rng.standard_normal((s, n, t))
    noise = draws["sigma"][:, :, None] * eps
    if not spec.ar1:
        return np.asarray(mean + noise)
    phi = draws["phi"]
    e = np.empty_like(noise)
    for j in range(t):
        if data.segment_start[j]:
            e[:, :, j] = noise[:, :, j] / np.sqrt(1.0 - phi**2)
        else:
            e[:, :, j] = phi * e[:, :, j - 1] + noise[:, :, j]
    return np.asarray(mean + e)


def pointwise_log_likelihood_draws(spec: ModelSpec, data: BayesData,
                                   samples: Mapping[str, np.ndarray]) -> np.ndarray:
    """log density of every (country, quarter) term for every draw: (chain, draw, N, T)."""
    import jax
    import jax.numpy as jnp
    import numpyro

    numpyro.enable_x64()
    inputs = model_inputs(data)
    chains, draws = samples["theta"].shape[:2]
    flat = {name: jnp.asarray(samples[name].reshape(chains * draws, *samples[name].shape[2:]))
            for name in LIKELIHOOD_PARAMETERS if name in samples}
    values = jax.vmap(lambda params: pointwise_log_likelihood(inputs, params, spec))(flat)
    return np.asarray(values).reshape(chains, draws, *data.y.shape)


# ----------------------------------------------------------------------------------
# DataTree
# ----------------------------------------------------------------------------------


def _draws_dataset(values: Mapping[str, np.ndarray], coords: Mapping[str, list[str]]) -> Any:
    """A (chain, draw, ...) dataset of the parameters in ``DIMS``."""
    import xarray as xr

    variables: dict[str, Any] = {}
    used: set[str] = set()
    for name, array in values.items():
        if name not in DIMS:
            continue
        dims = DIMS[name]
        if array.ndim != 2 + len(dims):
            raise ValueError(f"{name} has shape {array.shape}; expected dims {dims}.")
        variables[name] = (("chain", "draw", *dims), np.asarray(array))
        used.update(dims)
    first = next(iter(variables.values()))[1]
    return xr.Dataset(variables, coords={
        "chain": np.arange(first.shape[0]), "draw": np.arange(first.shape[1]),
        **{dim: coords[dim] for dim in used}})


def build_datatree(spec: ModelSpec, data: BayesData, samples: Mapping[str, np.ndarray],
                   sample_stats: Mapping[str, np.ndarray] | None = None,
                   prior: Mapping[str, np.ndarray] | None = None,
                   prior_predictive: np.ndarray | None = None,
                   log_likelihood: np.ndarray | None = None,
                   attrs: Mapping[str, Any] | None = None) -> Any:
    """Assemble the ArviZ group layout. ``prior`` and ``prior_predictive`` are (draws, ...)."""
    import xarray as xr

    coords = _coords(data)
    nodes: dict[str, Any] = {"posterior": _draws_dataset(samples, coords)}
    if sample_stats:
        renamed = {"diverging": "diverging", "energy": "energy", "num_steps": "n_steps"}
        nodes["sample_stats"] = xr.Dataset(
            {renamed[k]: (("chain", "draw"), np.asarray(v)) for k, v in sample_stats.items()
             if k in renamed},
            coords={"chain": np.arange(samples["theta"].shape[0]),
                    "draw": np.arange(samples["theta"].shape[1])})
    if prior is not None:
        nodes["prior"] = _draws_dataset({k: v[None] for k, v in prior.items()}, coords)
    if prior_predictive is not None:
        nodes["prior_predictive"] = xr.Dataset(
            {"y": (("chain", "draw", "country", "quarter"), prior_predictive[None])},
            coords={"chain": [0], "draw": np.arange(prior_predictive.shape[0]),
                    "country": coords["country"], "quarter": coords["quarter"]})
    cell = ("country", "quarter")
    nodes["observed_data"] = xr.Dataset({"y": (cell, data.y)}, coords={
        "country": coords["country"], "quarter": coords["quarter"]})
    nodes["constant_data"] = xr.Dataset(
        {"g": (cell, data.g), "z": ((*cell, "lag"), data.z), "dp": (cell, data.dp),
         "mask": (cell, data.mask.astype(np.int8)),
         "segment_start": (("quarter",), data.segment_start.astype(np.int8))},
        coords=coords)
    if log_likelihood is not None:
        rows, columns = np.nonzero(data.mask)
        labels = [f"{data.countries[i]} {data.quarters[j]}" for i, j in zip(rows, columns,
                                                                             strict=True)]
        nodes["log_likelihood"] = xr.Dataset(
            {"y": (("chain", "draw", "cell"), log_likelihood[:, :, rows, columns])},
            coords={"chain": np.arange(log_likelihood.shape[0]),
                    "draw": np.arange(log_likelihood.shape[1]), "cell": labels})
    tree = xr.DataTree.from_dict(nodes)
    tree.attrs.update({k: v if isinstance(v, str | int | float) else json.dumps(v)
                       for k, v in (attrs or {}).items()})
    return tree


def write_posterior_netcdf(path: Path, spec: ModelSpec, data: BayesData,
                           samples: Mapping[str, np.ndarray],
                           sample_stats: Mapping[str, np.ndarray] | None = None,
                           attrs: Mapping[str, Any] | None = None,
                           prior_seed: int = PRIOR_SEED) -> None:
    """Prior draws, prior predictive and pointwise log likelihood, then the NetCDF file."""
    prior = prior_draws(spec, data, PRIOR_DRAWS, prior_seed)
    simulated = simulate_outcomes(spec, data, prior, np.random.default_rng(prior_seed))
    tree = build_datatree(
        spec, data, samples, sample_stats, prior, simulated,
        pointwise_log_likelihood_draws(spec, data, samples),
        {**(attrs or {}), "specification": spec.name, "prior_seed": prior_seed,
         "prior_draws": PRIOR_DRAWS, "inference_library": "numpyro"})
    encoding = {
        f"/{name}": {var: {"zlib": True, "complevel": 4} for var in node.data_vars}
        for name, node in tree.children.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.to_netcdf(path, engine="h5netcdf", encoding=encoding)


def export_fit(directory: Path, spec: ModelSpec, data: BayesData) -> Path:
    """Write ``posterior.nc`` for a fit saved before NetCDF artifacts existed.

    Its sampler statistics were not kept, so the file has no ``sample_stats`` group.
    """
    summary = json.loads((directory / "fit.json").read_text(encoding="utf-8"))
    if summary["quarters"] != list(data.quarters) or summary["countries"] != list(
            data.countries):
        raise ValueError(f"The data do not match the fit in {directory}.")
    with np.load(directory / "draws.npz") as stored:
        samples = {name: stored[name] for name in stored.files}
    target = directory / "posterior.nc"
    write_posterior_netcdf(target, spec, data, samples, None, {
        "settings": summary["settings"], "environment": summary["environment"],
        "snapshot": summary["snapshot"], "diagnostics": summary["diagnostics"],
        "exported_from": "draws.npz"})
    return target
