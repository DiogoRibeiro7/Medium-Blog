"""Out-of-sample predictive comparison of the Bayesian and classical models.

Evaluation sets, expanding-window training data, the classical predictive distributions,
the Bayesian posterior predictive (from saved draws), proper scores and the panel
loss-differential test of ``PREDICTIVE_PROTOCOL.md``. NumPy and SciPy only: refitting
the Bayesian models lives in ``bayes_predictive`` and needs the ``bayes`` extra.

Every prediction for target quarter t uses observations up to t-1 plus GDP growth in
quarter t. It is a conditional prediction on revised data, not a real-time forecast.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import astuple, dataclass, fields, replace
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy import special, stats

from eu_okun.baseline import _quarter_before
from eu_okun.bayes_data import (
    EXCLUDED_INPUTS,
    PANDEMIC_EXCLUDED_WINDOWS,
    PRIMARY_WINDOWS,
    BayesData,
    build_bayes_data,
)
from eu_okun.contract import EU27, EurostatDataError, quarter_sequence
from eu_okun.review import BREAK_FLAGS, ENDPOINT_FLAGS, Flags, Panel
from eu_okun.stability import holm_adjust

LAGS = 2                     # GDP lags of every compared model (protocol, section 3)
NW_LAGS = 2                  # Newey-West bandwidth of the loss-differential test
ALPHA = 0.05                 # familywise level of the Holm-adjusted family
MIXTURE_STEP = 0.05          # grid step in log w for the factor integral of the log score
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass(frozen=True)
class EvaluationSet:
    """Target quarters and the data windows that hold their training samples."""

    name: str
    label: str
    windows: tuple[tuple[str, str], ...]
    first_target: str
    last_target: str


EVALUATION_SETS: dict[str, EvaluationSet] = {
    s.name: s for s in (
        EvaluationSet("E1", "pre-pandemic", PRIMARY_WINDOWS, "2016-Q1", "2019-Q4"),
        EvaluationSet("E2", "post-pandemic", PANDEMIC_EXCLUDED_WINDOWS, "2022-Q3", "2026-Q2"),
    )
}

# Model name -> pre-registered Bayesian specification (BAYES_PROTOCOL.md, section 5).
BAYES_MODELS: dict[str, str] = {
    "bayes_primary": "primary",
    "bayes_no_common_factor": "no_common_factor",
    "bayes_gaussian_iid": "gaussian_iid",
}
CLASSICAL_MODELS: tuple[str, ...] = (
    "ols_dl", "ardl", "zero_change", "historical_mean", "unemployment_ar1",
)
MODELS: tuple[str, ...] = (*BAYES_MODELS, *CLASSICAL_MODELS)
LAGGED_CHANGE_MODELS = frozenset({"ardl", "unemployment_ar1"})
MODEL_LABELS: dict[str, str] = {
    "bayes_primary": "Bayes primary",
    "bayes_no_common_factor": "Bayes, no common factor",
    "bayes_gaussian_iid": "Bayes, Gaussian i.i.d. errors",
    "ols_dl": "OLS distributed lag",
    "ardl": "ARDL (OLS + lagged change)",
    "zero_change": "Zero change",
    "historical_mean": "Historical mean",
    "unemployment_ar1": "AR(1) of the change",
}
# The pre-registered family (protocol, section 6): (A, B) in each evaluation set.
FAMILY: tuple[tuple[str, str], ...] = (
    ("bayes_primary", "ols_dl"),
    ("bayes_gaussian_iid", "ols_dl"),
    ("bayes_primary", "bayes_gaussian_iid"),
    ("bayes_primary", "ardl"),
    ("bayes_primary", "zero_change"),
)


# ----------------------------------------------------------------------------------
# Data: the full grid of an evaluation set and expanding-window training slices
# ----------------------------------------------------------------------------------


def _previous_change(panel: Panel, flags: Flags, geo: str, quarter: str,
                     gaps: set[str]) -> float | None:
    """The unemployment change in ``quarter`` if it may serve as a lagged regressor.

    It needs both quarters outside the excluded period, observed values and no
    labour-force flag that invalidates the change (the rule of ``flagged_change``).
    """
    before = _quarter_before(quarter, 1)
    if quarter in gaps or before in gaps:
        return None
    now_row, before_row = panel.get((geo, quarter)), panel.get((geo, before))
    if now_row is None or before_row is None:
        return None
    now_u, before_u = now_row["unemployment"], before_row["unemployment"]
    if now_u is None or before_u is None:
        return None
    now_flag = flags.get((geo, quarter), {}).get("lfs", "")
    before_flag = flags.get((geo, before), {}).get("lfs", "")
    if now_flag in BREAK_FLAGS or now_flag in ENDPOINT_FLAGS or before_flag in ENDPOINT_FLAGS:
        return None
    return now_u - before_u


@dataclass(frozen=True)
class PredictiveData:
    """The full grid of one evaluation set and the columns of its target quarters.

    ``y_prev[i, t]`` is the change in quarter t-1 and ``prev_ok`` says whether it may be
    used: both quarters outside the excluded period, observed, and no labour-force flag
    that invalidates the change. At a segment start the change comes from the quarter
    before the window.
    """

    evaluation: EvaluationSet
    data: BayesData
    y_prev: np.ndarray
    prev_ok: np.ndarray
    targets: tuple[int, ...]

    @property
    def target_quarters(self) -> list[str]:
        return [self.data.quarters[j] for j in self.targets]

    @property
    def scored(self) -> np.ndarray:
        """Cells scored for every model, shape (N, number of targets)."""
        columns = list(self.targets)
        return np.asarray(self.data.mask[:, columns] & self.prev_ok[:, columns])

    def training(self, column: int) -> BayesData:
        """Training data for target ``column``: every column strictly before it."""
        if not 0 < column < self.data.n_quarters:
            raise ValueError("A target needs at least one earlier quarter.")
        d = self.data
        last = d.quarters[column]
        return replace(
            d, quarters=d.quarters[:column], y=d.y[:, :column], g=d.g[:, :column],
            z=d.z[:, :column], dp=d.dp[:, :column], mask=d.mask[:, :column],
            segment_start=d.segment_start[:column],
            masked_reasons={geo: [q for q in qs if q < last]
                            for geo, qs in d.masked_reasons.items() if any(q < last for q in qs)},
        )


def build_predictive_data(
    panel: Panel, flags: Flags, evaluation: EvaluationSet, countries: Sequence[str] = EU27,
) -> PredictiveData:
    """Assemble the grid of an evaluation set (targets included) and lagged changes."""
    data = build_bayes_data(panel, flags, evaluation.windows, LAGS, countries)
    quarters = list(data.quarters)
    if evaluation.first_target not in quarters or evaluation.last_target not in quarters:
        raise EurostatDataError(f"Targets of {evaluation.name} lie outside its windows.")
    first = quarters.index(evaluation.first_target)
    last = quarters.index(evaluation.last_target)
    if first < 1 or last < first:
        raise ValueError(f"Invalid target range for {evaluation.name}.")
    gaps = {q for start, end in EXCLUDED_INPUTS for q in quarter_sequence(start, end)}
    n, t = data.y.shape
    y_prev = np.full((n, t), np.nan)
    prev_ok = np.zeros((n, t), dtype=bool)
    for j in range(t):
        previous = _quarter_before(quarters[j], 1)
        for i, geo in enumerate(countries):
            value = _previous_change(panel, flags, geo, previous, gaps)
            if value is not None:
                y_prev[i, j], prev_ok[i, j] = value, True
    return PredictiveData(evaluation, data, y_prev, prev_ok, tuple(range(first, last + 1)))


# ----------------------------------------------------------------------------------
# Predictive distributions and scores
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentT:
    """Location-scale Student-t predictive distribution (df > 1)."""

    loc: float
    scale: float
    df: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.loc) and math.isfinite(self.scale) and self.scale > 0
                and self.df > 1):
            raise ValueError("Invalid Student-t predictive distribution.")


def ols_predictive(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> StudentT:
    """Exact OLS predictive distribution under Gaussian i.i.d. errors.

    Location ``x_new'b``, scale ``s * sqrt(1 + x_new'(X'X)^{-1} x_new)`` and n - k
    degrees of freedom: the frequentist prediction interval, equivalently the Bayesian
    predictive under the reference prior.
    """
    if x.ndim != 2 or y.shape != (x.shape[0],) or x_new.shape != (x.shape[1],):
        raise ValueError("Incompatible design, response and prediction row.")
    n, k = x.shape
    if n < max(12, 3 * k):
        raise ValueError("Insufficient training rows.")
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(x_new).all()):
        raise ValueError("Non-finite inputs are not allowed.")
    if np.linalg.matrix_rank(x) != k or np.linalg.cond(x) > 1e10:
        raise ValueError("Rank-deficient or ill-conditioned training regression.")
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    s2 = float(residuals @ residuals) / (n - k)
    leverage = float(x_new @ np.linalg.solve(x.T @ x, x_new))
    return StudentT(float(x_new @ beta), math.sqrt(s2 * (1.0 + leverage)), float(n - k))


def zero_change_predictive(y: np.ndarray) -> StudentT:
    """No change, with the scale of past changes: Student-t, n df, scale sqrt(mean y^2)."""
    if y.ndim != 1 or y.size < 12 or not np.isfinite(y).all():
        raise ValueError("Zero-change scale needs at least 12 finite training changes.")
    return StudentT(0.0, float(np.sqrt(np.mean(y**2))), float(y.size))


def crps_student_t(y: float, dist: StudentT) -> float:
    """Closed-form CRPS of a location-scale Student-t (Jordan, Krueger and Lerch, 2019)."""
    nu, z = dist.df, (y - dist.loc) / dist.scale
    # 2 sqrt(nu) B(1/2, nu - 1/2) / ((nu - 1) B(1/2, nu/2)^2), on the log scale for large nu.
    constant = 2.0 * math.sqrt(nu) / (nu - 1.0) * math.exp(
        special.betaln(0.5, nu - 0.5) - 2.0 * special.betaln(0.5, nu / 2.0))
    cdf, pdf = float(stats.t.cdf(z, nu)), float(stats.t.pdf(z, nu))
    return dist.scale * (z * (2.0 * cdf - 1.0) + 2.0 * pdf * (nu + z * z) / (nu - 1.0) - constant)


def crps_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS of the empirical distribution of each column: E|X - y| - E|X - X'| / 2.

    ``samples`` has shape (m, n) and ``y`` shape (n,). For sorted samples,
    E|X - X'| = (2 / m^2) * sum_i (2i - m - 1) x_(i).
    """
    if samples.ndim != 2 or y.shape != (samples.shape[1],):
        raise ValueError("Samples must be (m, n) with one observation per column.")
    x = np.sort(samples, axis=0)
    m = x.shape[0]
    first = np.mean(np.abs(x - y[None, :]), axis=0)
    weights = 2.0 * np.arange(1, m + 1) - m - 1.0
    spread = 2.0 * (weights[:, None] * x).sum(axis=0) / m**2
    return np.asarray(first - spread / 2.0)


@dataclass(frozen=True)
class Prediction:
    """One model's predictive distribution for one scored cell, with its scores."""

    evaluation: str
    model: str
    geo: str
    quarter: str
    actual: float
    mean: float
    median: float
    q05: float
    q25: float
    q75: float
    q95: float
    crps: float
    log_score: float
    pit: float


def predict_student_t(evaluation: str, model: str, geo: str, quarter: str, actual: float,
                      dist: StudentT) -> Prediction:
    """Summaries and scores of a Student-t predictive distribution."""
    q = stats.t.ppf(QUANTILES, dist.df, dist.loc, dist.scale)
    return Prediction(
        evaluation, model, geo, quarter, actual, dist.loc, float(q[2]), float(q[0]),
        float(q[1]), float(q[3]), float(q[4]), crps_student_t(actual, dist),
        float(stats.t.logpdf(actual, dist.df, dist.loc, dist.scale)),
        float(stats.t.cdf(actual, dist.df, dist.loc, dist.scale)),
    )


# ----------------------------------------------------------------------------------
# Classical models (country by country, no pooling)
# ----------------------------------------------------------------------------------


def _regressors(pdata: PredictiveData, design: np.ndarray, model: str, i: int,
                j: int) -> np.ndarray:
    """Regressor row of ``model`` for country i in column j (no current-quarter outcome)."""
    gdp = design[i, j]                      # 1, g[t], g[t-1], g[t-2]
    if model == "ols_dl":
        return np.asarray(gdp)
    if model == "ardl":
        return np.asarray([1.0, pdata.y_prev[i, j], *gdp[1:]])
    if model == "unemployment_ar1":
        return np.asarray([1.0, pdata.y_prev[i, j]])
    if model == "historical_mean":
        return np.ones(1)
    raise ValueError(f"No regression design for {model}.")


def classical_predictions(pdata: PredictiveData, model: str) -> list[Prediction]:
    """Refit ``model`` for every target and scored cell, on rows strictly before it."""
    if model not in CLASSICAL_MODELS:
        raise ValueError(f"Unknown classical model: {model}.")
    d = pdata.data
    design = d.gdp_design()
    scored = pdata.scored
    out: list[Prediction] = []
    for column_index, j in enumerate(pdata.targets):
        quarter = d.quarters[j]
        for i, geo in enumerate(d.countries):
            if not scored[i, column_index]:
                continue
            rows = d.mask[i, :j].copy()
            if model in LAGGED_CHANGE_MODELS:
                rows &= pdata.prev_ok[i, :j]
            y = d.y[i, :j][rows]
            try:
                if model == "zero_change":
                    dist = zero_change_predictive(y)
                else:
                    x = np.vstack([_regressors(pdata, design, model, i, int(c))
                                   for c in np.flatnonzero(rows)])
                    dist = ols_predictive(x, y, _regressors(pdata, design, model, i, j))
            except ValueError as exc:
                raise EurostatDataError(f"{model} failed for {geo}/{quarter}: {exc}") from exc
            out.append(predict_student_t(pdata.evaluation.name, model, geo, quarter,
                                         float(d.y[i, j]), dist))
    return out


# ----------------------------------------------------------------------------------
# Bayesian posterior predictive from saved draws
# ----------------------------------------------------------------------------------


class ErrorModel(Protocol):
    """The parts of a Bayesian specification that shape its predictive distribution."""

    @property
    def ar1(self) -> bool: ...
    @property
    def student_t(self) -> bool: ...
    @property
    def common_factor(self) -> bool: ...
    @property
    def participation(self) -> bool: ...


@dataclass(frozen=True)
class TargetInputs:
    """What a prediction for one target column may use (nothing from the target's y)."""

    g: np.ndarray            # (N,) GDP growth in the target quarter
    z: np.ndarray            # (N, K) g[t-k] - g[t]
    segment_start: bool      # first quarter of a segment: no AR(1) predecessor
    y_last: np.ndarray       # (N,) outcome in the previous column (training data)
    g_last: np.ndarray       # (N,) GDP growth in the previous column
    z_last: np.ndarray       # (N, K)


def target_inputs(pdata: PredictiveData, column: int) -> TargetInputs:
    """Inputs for target ``column``; the previous column belongs to the training data."""
    d = pdata.data
    previous = column - 1
    return TargetInputs(
        g=d.g[:, column], z=d.z[:, column], segment_start=bool(d.segment_start[column]),
        y_last=d.y[:, previous], g_last=d.g[:, previous], z_last=d.z[:, previous],
    )


@dataclass(frozen=True)
class PredictiveComponents:
    """Per draw and country: centre and scale of the innovation, factor loading, nu."""

    centre: np.ndarray               # (S, N)
    scale: np.ndarray                # (S, N)
    loading: np.ndarray | None       # (S, N) lambda, when the factor is in the model
    nu: np.ndarray | None            # (S,) Student-t degrees of freedom


def predictive_components(draws: Mapping[str, np.ndarray], spec: ErrorModel,
                          target: TargetInputs) -> PredictiveComponents:
    """Conditional mean and scale of y[t] given each draw, before the unknown factor.

    ``draws`` holds flattened posterior draws, shape (S, ...). The previous residual uses
    each draw's in-sample factor value for the last training quarter (``f[:, -1]``).
    """
    if spec.participation:
        raise ValueError("Participation in the target quarter is not in the information set.")
    alpha, theta, b, sigma = draws["alpha"], draws["theta"], draws["b"], draws["sigma"]
    mean_now = alpha + theta * target.g[None, :] + np.einsum("nk,snk->sn", target.z, b)
    loading = draws["lam"] if spec.common_factor else None
    if spec.ar1 and not target.segment_start:
        mean_last = alpha + theta * target.g_last[None, :] + np.einsum(
            "nk,snk->sn", target.z_last, b)
        if loading is not None:
            mean_last = mean_last + loading * draws["f"][:, -1][:, None]
        phi = draws["phi"]
        centre = mean_now + phi * (target.y_last[None, :] - mean_last)
        scale = sigma
    elif spec.ar1:
        centre, scale = mean_now, sigma / np.sqrt(1.0 - draws["phi"] ** 2)
    else:
        centre, scale = mean_now, sigma
    nu = np.asarray(draws["nu"]).reshape(-1) if spec.student_t else None
    return PredictiveComponents(np.asarray(centre), np.asarray(scale), loading, nu)


def predictive_samples(components: PredictiveComponents, replicates: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Samples of y[t], shape (S * replicates, N); one factor value per draw, shared."""
    s, n = components.centre.shape
    out = np.empty((replicates, s, n))
    for r in range(replicates):
        if components.nu is not None:
            eps = rng.standard_t(components.nu[:, None], size=(s, n))
        else:
            eps = rng.standard_normal((s, n))
        y = components.centre + components.scale * eps
        if components.loading is not None:
            factor = np.asarray(rng.standard_normal(s))
            y = y + components.loading * factor[:, None]
        out[r] = y
    return out.reshape(replicates * s, n)


def _t_factor_log_density(r: np.ndarray, scale: np.ndarray, loading: np.ndarray,
                          nu: np.ndarray) -> np.ndarray:
    """log density of ``r = loading*f + scale*eps``, f ~ N(0, 1), eps ~ Student-t(nu).

    Shapes: ``r``, ``scale``, ``loading`` (S, N); ``nu`` (S,). Writing eps as a
    Gamma(nu/2, nu/2) scale mixture of normals integrates f analytically:
    ``p(r) = E_w[N(r; 0, loading^2 + scale^2 / w)]``. The expectation is a smooth
    integral over v = log w, computed with the trapezoid rule on a uniform grid that
    covers both tails (agreement with adaptive quadrature to about 1e-12 in the tests,
    far-tail outcomes included).
    """
    nu_min, nu_max = float(nu.min()), float(nu.max())
    step = min(MIXTURE_STEP, math.sqrt(2.0 / nu_max) / 4.0)
    # The integrand peaks near v = -2 log|r / scale| for far-tail outcomes; extend the
    # grid below that, and above to where the Gamma density is negligible.
    extreme = float(np.max(np.abs(r) / scale))
    lower = -40.0 - 2.0 * math.log1p(extreme)
    v = np.arange(lower, math.log(1.0 + 100.0 / nu_min) + step, step)
    w = np.exp(v)
    a = nu / 2.0
    log_gamma = ((a * np.log(a) - special.gammaln(a))[:, None] + a[:, None] * v[None, :]
                 - a[:, None] * w[None, :])                     # log Gamma(w) + log w, (S, V)
    out = np.empty_like(r)
    for i in range(r.shape[1]):
        variance = loading[:, i, None] ** 2 + scale[:, i, None] ** 2 / w[None, :]
        log_kernel = (log_gamma - 0.5 * np.log(2.0 * math.pi * variance)
                      - r[:, i, None] ** 2 / (2.0 * variance))
        peak = log_kernel.max(axis=1, keepdims=True)
        out[:, i] = peak[:, 0] + np.log(np.trapezoid(np.exp(log_kernel - peak), v, axis=1))
    return out


def predictive_log_density(components: PredictiveComponents, y: np.ndarray) -> np.ndarray:
    """log p(y[i] | training data) per country: a mixture over posterior draws.

    The unknown factor value f ~ N(0, 1) is integrated out exactly for Gaussian
    innovations and through the Student-t scale-mixture representation otherwise.
    """
    centre, scale = components.centre, components.scale
    loading, nu = components.loading, components.nu
    r = y[None, :] - centre
    if loading is None:
        per_draw = (stats.norm.logpdf(r, 0.0, scale) if nu is None
                    else stats.t.logpdf(r, nu[:, None], 0.0, scale))
    elif nu is None:
        per_draw = stats.norm.logpdf(r, 0.0, np.sqrt(scale**2 + loading**2))
    else:
        per_draw = _t_factor_log_density(r, scale, loading, nu)
    return np.asarray(special.logsumexp(per_draw, axis=0) - math.log(centre.shape[0]))


def bayes_predictions(evaluation: str, model: str, pdata: PredictiveData, column: int,
                      draws: Mapping[str, np.ndarray], spec: ErrorModel, replicates: int,
                      seed: int) -> list[Prediction]:
    """Posterior predictive summaries and scores for the scored cells of one target."""
    components = predictive_components(draws, spec, target_inputs(pdata, column))
    samples = predictive_samples(components, replicates, np.random.default_rng(seed))
    actual = pdata.data.y[:, column]
    crps = crps_samples(samples, actual)
    log_score = predictive_log_density(components, actual)
    quantiles = np.quantile(samples, QUANTILES, axis=0)
    pit = np.mean(samples <= actual[None, :], axis=0)
    mean = samples.mean(axis=0)
    scored = pdata.scored[:, pdata.targets.index(column)]
    quarter = pdata.data.quarters[column]
    return [
        Prediction(evaluation, model, geo, quarter, float(actual[i]), float(mean[i]),
                   float(quantiles[2, i]), float(quantiles[0, i]), float(quantiles[1, i]),
                   float(quantiles[3, i]), float(quantiles[4, i]), float(crps[i]),
                   float(log_score[i]), float(pit[i]))
        for i, geo in enumerate(pdata.data.countries) if scored[i]
    ]


# ----------------------------------------------------------------------------------
# Panel loss-differential test (protocol, section 6)
# ----------------------------------------------------------------------------------


def newey_west_variance(d: np.ndarray, lags: int = NW_LAGS) -> float:
    """Long-run variance of a series with Bartlett weights (Newey and West, 1987)."""
    if d.ndim != 1 or lags < 0 or lags >= d.size:
        raise ValueError("Invalid series or bandwidth.")
    x = d - d.mean()
    t = x.size
    variance = float(x @ x) / t
    for lag in range(1, lags + 1):
        variance += 2.0 * (1.0 - lag / (lags + 1.0)) * float(x[lag:] @ x[:-lag]) / t
    return variance


@dataclass(frozen=True)
class PanelTest:
    """Panel loss differential of model A against model B in one evaluation set."""

    evaluation: str
    model_a: str
    model_b: str
    score: str
    quarters: int
    cells: int
    mean_a: float
    mean_b: float
    ratio: float
    mean_difference: float
    standard_error: float
    statistic: float
    p_value: float
    p_holm: float = float("nan")
    verdict: str = ""


def quarterly_differentials(loss_a: np.ndarray, loss_b: np.ndarray,
                            scored: np.ndarray) -> np.ndarray:
    """d[t]: mean over the countries scored in quarter t of loss A minus loss B."""
    if loss_a.shape != loss_b.shape or loss_a.shape != scored.shape:
        raise ValueError("Loss arrays and the scored mask must have one shape.")
    if not scored.any(axis=0).all():
        raise ValueError("Every target quarter needs at least one scored cell.")
    return np.asarray([np.mean(loss_a[scored[:, t], t] - loss_b[scored[:, t], t])
                       for t in range(scored.shape[1])])


def panel_test(evaluation: str, model_a: str, model_b: str, loss_a: np.ndarray,
               loss_b: np.ndarray, scored: np.ndarray, score: str = "crps",
               lags: int = NW_LAGS, horizon: int = 1) -> PanelTest:
    """Diebold-Mariano test on cross-country mean loss differentials.

    Averaging within a quarter first makes the test valid under any cross-country
    dependence; serial dependence is handled with a Newey-West variance. The statistic
    has the Harvey-Leybourne-Newbold small-sample correction and is compared with a
    t distribution with T - 1 degrees of freedom (two-sided). ``score`` names the loss
    (a key of ``LOSSES``) and is only recorded.
    """
    d = quarterly_differentials(loss_a, loss_b, scored)
    t = d.size
    if t < 8:
        raise ValueError("At least 8 target quarters are needed.")
    mean_a = float(np.mean([loss_a[scored[:, j], j].mean() for j in range(t)]))
    mean_b = float(np.mean([loss_b[scored[:, j], j].mean() for j in range(t)]))
    variance = newey_west_variance(d, lags)
    mean = float(d.mean())
    if variance <= 0.0:
        if mean != 0.0:
            raise ValueError("Constant non-zero loss differential: the test is undefined.")
        statistic, p_value, se = 0.0, 1.0, 0.0
    else:
        se = math.sqrt(variance / t)
        correction = math.sqrt((t + 1 - 2 * horizon + horizon * (horizon - 1) / t) / t)
        statistic = mean / se * correction
        p_value = float(2.0 * stats.t.sf(abs(statistic), t - 1))
    return PanelTest(evaluation, model_a, model_b, score, t, int(scored.sum()), mean_a, mean_b,
                     mean_a / mean_b, mean, se, statistic, p_value)


def adjust_family(tests: Sequence[PanelTest], alpha: float = ALPHA) -> list[PanelTest]:
    """Holm-adjust the family and attach the pre-registered verdicts."""
    adjusted = holm_adjust([test.p_value for test in tests])
    out = []
    for test, p in zip(tests, adjusted, strict=True):
        if p < alpha:
            winner = test.model_a if test.mean_difference < 0 else test.model_b
            verdict = f"{winner} better"
        else:
            verdict = "not distinguishable"
        out.append(replace(test, p_holm=p, verdict=verdict))
    return out


# Losses (lower is better): CRPS, negative log predictive density, squared error of the
# predictive mean and absolute error of the predictive median.
LOSSES: dict[str, str] = {
    "crps": "CRPS",
    "neg_log_score": "Negative log predictive density",
    "squared_error": "Squared error of the predictive mean",
    "absolute_error": "Absolute error of the predictive median",
}


def loss_value(p: Prediction, loss: str) -> float:
    """One loss of one prediction."""
    if loss == "crps":
        return p.crps
    if loss == "neg_log_score":
        return -p.log_score
    if loss == "squared_error":
        return (p.mean - p.actual) ** 2
    if loss == "absolute_error":
        return abs(p.median - p.actual)
    raise ValueError(f"Unknown loss: {loss}.")


def loss_matrix(predictions: Sequence[Prediction], pdata: PredictiveData,
                loss: str) -> np.ndarray:
    """Arrange one model's losses as (N, targets), NaN where a cell is not scored.

    Raises if the predictions do not cover exactly the scored cells.
    """
    countries = list(pdata.data.countries)
    quarters = pdata.target_quarters
    out = np.full((len(countries), len(quarters)), np.nan)
    for p in predictions:
        i, j = countries.index(p.geo), quarters.index(p.quarter)
        if np.isfinite(out[i, j]):
            raise ValueError(f"Duplicate prediction for {p.model} {p.geo}/{p.quarter}.")
        out[i, j] = loss_value(p, loss)
    if not np.array_equal(np.isfinite(out), pdata.scored):
        raise ValueError("Predictions do not cover exactly the scored cells.")
    return out


# ----------------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------------

PREDICTION_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Prediction))


def write_predictions(path: Path, rows: Sequence[Prediction]) -> None:
    """Write predictions as CSV (full float precision, LF line endings)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(PREDICTION_FIELDS)
        for row in rows:
            writer.writerow([repr(v) if isinstance(v, float) else v for v in astuple(row)])


def read_predictions(path: Path) -> list[Prediction]:
    """Read predictions written by ``write_predictions``."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != PREDICTION_FIELDS:
            raise ValueError(f"Unexpected prediction columns in {path}.")
        return [
            Prediction(r["evaluation"], r["model"], r["geo"], r["quarter"],
                       *(float(r[name]) for name in PREDICTION_FIELDS[4:]))
            for r in reader
        ]
