"""Small-sample inference for the post-2021 stability tests (``SMALL_SAMPLE_PROTOCOL.md``).

The published tests (``stability.fit_stability``) compare the GDP coefficient sum of
2009-2019 with that of 2022 onwards using a calendar-aware Newey-West t-statistic and a
normal reference distribution. This module keeps the data, design and statistic and
replaces only the reference distribution: null-imposed sieve and moving-block bootstraps
of the HAC-studentised statistic, a Monte Carlo size study, and bootstrap-t intervals for
the post-2021 sums. NumPy only; everything reads the frozen snapshot.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.baseline import _quarter_before, _read_panel
from eu_okun.contract import EU27, EurostatDataError
from eu_okun.stability import _eligible_periods, classical_weaker, holm_adjust

Z_975 = 1.959963984540054
ALPHA = 0.05
PRIMARY_BANDWIDTH = 4
BANDWIDTHS = (0, 2, 4, 8)
B_MAIN = 999
B_SIZE = 199
R_SIZE = 200
BLOCK = 4
RHO_CLIP = 0.95
SEED_TESTS = 20261017
SEED_SIZE = 20261018
SEED_POST = 20261019
# Not pre-registered (change log): B = 999 cannot resolve p-values below 1/1000, while
# Holm over 54 tests needs the smallest p <= 0.05/54. The check repeats both schemes.
B_RESOLUTION = 9999


# ----------------------------------------------------------------------------------
# Batched HAC t-statistics
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class HacDesign:
    """A fixed design, contrast and bandwidth, precomputed for many outcomes.

    With ``a = (X'X)^{-1} c``, the HAC variance of ``c'beta`` is ``a' meat a``, which
    depends on the residuals only through ``h_t = (x_t'a) e_t``. That makes the
    statistic cheap for a batch of outcome vectors.
    """

    x: np.ndarray                     # (n, k)
    projector: np.ndarray             # (k, n) = (X'X)^{-1} X'
    contrast: np.ndarray              # (k,)
    leverage: np.ndarray              # (n,) = X (X'X)^{-1} c
    pairs: tuple[tuple[float, np.ndarray, np.ndarray], ...]  # (weight, current, previous)
    restrict: np.ndarray              # (k,) = (X'X)^{-1} c / (c'(X'X)^{-1} c)


def hac_design(x: np.ndarray, periods: Sequence[str], contrast: np.ndarray,
               bandwidth: int) -> HacDesign:
    """Precompute the pieces of the calendar-aware Bartlett HAC t-statistic."""
    if len(periods) != x.shape[0] or list(periods) != sorted(set(periods)):
        raise ValueError("Periods must be unique, sorted and match the rows.")
    xtx_inv = np.linalg.inv(x.T @ x)
    a = xtx_inv @ contrast
    index = {p: i for i, p in enumerate(periods)}
    pairs = []
    for lag in range(1, bandwidth + 1):
        current = [i for i, p in enumerate(periods) if _quarter_before(p, lag) in index]
        previous = [index[_quarter_before(periods[i], lag)] for i in current]
        pairs.append((1.0 - lag / (bandwidth + 1.0), np.asarray(current, dtype=int),
                      np.asarray(previous, dtype=int)))
    return HacDesign(x, xtx_inv @ x.T, contrast, x @ a, tuple(pairs),
                     a / float(contrast @ a))


def hac_t(design: HacDesign, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate ``c'beta``, its HAC standard error and t for each column of ``y``."""
    outcomes = y[:, None] if y.ndim == 1 else y
    beta = design.projector @ outcomes
    resid = outcomes - design.x @ beta
    h = design.leverage[:, None] * resid
    variance = np.sum(h * h, axis=0)
    for weight, current, previous in design.pairs:
        variance = variance + 2.0 * weight * np.sum(h[current] * h[previous], axis=0)
    estimate = design.contrast @ beta
    se = np.sqrt(np.maximum(variance, 0.0))
    return estimate, se, estimate / se


def restricted_fit(design: HacDesign, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares under ``c'beta = 0``: coefficients and residuals."""
    beta = design.projector @ y
    beta_r = beta - design.restrict * float(design.contrast @ beta)
    return beta_r, y - design.x @ beta_r


# ----------------------------------------------------------------------------------
# Bootstrap error processes
# ----------------------------------------------------------------------------------


def segment_starts(periods: Sequence[str]) -> np.ndarray:
    """True where a row has no calendar predecessor among the rows."""
    present = set(periods)
    return np.asarray([_quarter_before(p, 1) not in present for p in periods], dtype=bool)


@dataclass(frozen=True)
class Sieve:
    """AR(1) error model per period (protocol, section 3.1)."""

    starts: np.ndarray                # (n,) True at the first quarter of a period
    rho: np.ndarray                   # (n,) AR coefficient of each row's period
    innovations: np.ndarray           # (n,) centred u_t (start rows: e_t itself)
    period_rho: tuple[float, ...]


def fit_sieve(resid: np.ndarray, starts: np.ndarray) -> Sieve:
    """AR(1) by least squares within each period, clipped to [-0.95, 0.95]."""
    n = resid.size
    segment = np.cumsum(starts) - 1
    rho = np.zeros(n)
    innovations = resid.astype(float).copy()
    rhos = []
    for s in range(int(segment.max()) + 1):
        rows = np.flatnonzero(segment == s)
        lagged, current = resid[rows[:-1]], resid[rows[1:]]
        denominator = float(lagged @ lagged)
        value = float(lagged @ current) / denominator if denominator > 0 else 0.0
        value = float(np.clip(value, -RHO_CLIP, RHO_CLIP))
        rho[rows] = value
        u = current - value * lagged
        innovations[rows[1:]] = u - u.mean()
        rhos.append(value)
    return Sieve(starts, rho, innovations, tuple(rhos))


def sieve_errors(sieve: Sieve, start_values: np.ndarray, draws: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Wild AR(1) errors, shape (n, draws): e*_t = rho e*_{t-1} + v_t u_t."""
    n = sieve.starts.size
    v = rng.choice(np.asarray([-1.0, 1.0]), size=(n, draws))
    errors = np.empty((n, draws))
    for t in range(n):
        if sieve.starts[t]:
            errors[t] = v[t] * start_values[t]
        else:
            errors[t] = sieve.rho[t] * errors[t - 1] + v[t] * sieve.innovations[t]
    return errors


def block_errors(resid: np.ndarray, starts: np.ndarray, draws: int,
                 rng: np.random.Generator, block: int = BLOCK) -> np.ndarray:
    """Moving blocks of residuals, drawn and concatenated within each period."""
    segment = np.cumsum(starts) - 1
    errors = np.empty((resid.size, draws))
    for s in range(int(segment.max()) + 1):
        rows = np.flatnonzero(segment == s)
        length = rows.size
        size = min(block, length)
        count = math.ceil(length / size)
        first = rng.integers(0, length - size + 1, size=(draws, count))
        index = (first[:, :, None] + np.arange(size)).reshape(draws, -1)[:, :length]
        errors[rows] = resid[rows][index].T
    return errors


def bootstrap_p(observed: float, simulated: np.ndarray) -> float:
    """(1 + #{|t*| >= |t|}) / (B + 1)."""
    return float((1 + np.sum(np.abs(simulated) >= abs(observed))) / (simulated.size + 1))


def null_sieve_test(design: HacDesign, y: np.ndarray, starts: np.ndarray, draws: int,
                    rng: np.random.Generator) -> tuple[float, np.ndarray, Sieve]:
    """Observed t, bootstrap t* under the null (sieve scheme) and the fitted sieve."""
    _, _, t_obs = hac_t(design, y)
    beta_r, resid = restricted_fit(design, y)
    sieve = fit_sieve(resid, starts)
    y_star = (design.x @ beta_r)[:, None] + sieve_errors(sieve, resid, draws, rng)
    return float(t_obs[0]), hac_t(design, y_star)[2], sieve


def null_block_test(design: HacDesign, y: np.ndarray, starts: np.ndarray, draws: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Bootstrap t* under the null (moving-block scheme)."""
    beta_r, resid = restricted_fit(design, y)
    y_star = (design.x @ beta_r)[:, None] + block_errors(resid, starts, draws, rng)
    return hac_t(design, y_star)[2]


# ----------------------------------------------------------------------------------
# The study
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """One of the 54 published stability tests."""

    index: int
    geo: str
    participation: bool
    x: np.ndarray                     # pooled interacted design
    y: np.ndarray
    periods: list[str]
    x_post: np.ndarray
    y_post: np.ndarray
    post_periods: list[str]


def stability_specs(panel: dict[tuple[str, str], dict[str, float | None]], end: str,
                    countries: Sequence[str] = EU27) -> list[Spec]:
    """The published designs, in the published row order (country, then participation)."""
    pre, pre_dates, post, post_dates = _eligible_periods(panel, end, countries)
    specs: list[Spec] = []
    for geo in countries:
        for control in (False, True):
            x_pre, y_pre = pre[(geo, 1, control)]
            x_post, y_post = post[(geo, 1, control)]
            x = np.vstack((np.hstack((x_pre, np.zeros_like(x_pre))),
                           np.hstack((x_post, x_post))))
            specs.append(Spec(len(specs), geo, control, x, np.concatenate((y_pre, y_post)),
                              pre_dates + post_dates, x_post, y_post, list(post_dates)))
    return specs


def change_contrast(width: int) -> np.ndarray:
    """Picks the two GDP interaction coefficients (post-minus-pre change of the sum)."""
    c = np.zeros(2 * width)
    c[width + 1: width + 3] = 1.0
    return c


def sum_contrast(width: int) -> np.ndarray:
    c = np.zeros(width)
    c[1:3] = 1.0
    return c


@dataclass(frozen=True)
class ChangeTest:
    """One stability test under the asymptotic and bootstrap reference distributions."""

    geo: str
    participation: bool
    change: float
    se: float
    t: float
    p_asymptotic: float
    p_sieve: float
    p_block: float
    asymptotic_size_under_null: float   # share of |t*| > 1.96 in the sieve draws
    rho_pre: float
    rho_post: float
    p_sieve_bandwidth: dict[int, float]
    p_asymptotic_holm: float = 1.0
    p_sieve_holm: float = 1.0
    p_block_holm: float = 1.0

    @property
    def survives(self) -> bool:
        return self.p_sieve_holm <= ALPHA and self.p_block_holm <= ALPHA

    @property
    def scheme_dependent(self) -> bool:
        return (self.p_sieve_holm <= ALPHA) != (self.p_block_holm <= ALPHA)


def change_tests(specs: Sequence[Spec], draws: int = B_MAIN) -> list[ChangeTest]:
    """Section 3: every published test under both null-imposed bootstraps."""
    rows = []
    for spec in specs:
        width = spec.x.shape[1] // 2
        starts = segment_starts(spec.periods)
        by_bandwidth: dict[int, float] = {}
        primary: tuple[float, np.ndarray, Sieve] | None = None
        for bandwidth in BANDWIDTHS:
            design = hac_design(spec.x, spec.periods, change_contrast(width), bandwidth)
            rng = np.random.default_rng([SEED_TESTS, spec.index, bandwidth])
            t_obs, t_star, sieve = null_sieve_test(design, spec.y, starts, draws, rng)
            by_bandwidth[bandwidth] = bootstrap_p(t_obs, t_star)
            if bandwidth == PRIMARY_BANDWIDTH:
                primary = (t_obs, t_star, sieve)
                estimate, se, _ = hac_t(design, spec.y)
                block_rng = np.random.default_rng([SEED_TESTS, spec.index, 100])
                t_block = null_block_test(design, spec.y, starts, draws, block_rng)
                change, change_se = float(estimate[0]), float(se[0])
        assert primary is not None
        t_obs, t_star, sieve = primary
        rows.append(ChangeTest(
            spec.geo, spec.participation, change, change_se, t_obs,
            math.erfc(abs(t_obs) / math.sqrt(2.0)), by_bandwidth[PRIMARY_BANDWIDTH],
            bootstrap_p(t_obs, t_block), float(np.mean(np.abs(t_star) > Z_975)),
            sieve.period_rho[0], sieve.period_rho[-1], by_bandwidth))
    holm_asym = holm_adjust([r.p_asymptotic for r in rows])
    holm_sieve = holm_adjust([r.p_sieve for r in rows])
    holm_block = holm_adjust([r.p_block for r in rows])
    return [ChangeTest(**{**_fields(r), "p_asymptotic_holm": a, "p_sieve_holm": s,
                          "p_block_holm": b})
            for r, a, s, b in zip(rows, holm_asym, holm_sieve, holm_block, strict=True)]


def _fields(row: ChangeTest) -> dict[str, Any]:
    return {k: getattr(row, k) for k in row.__dataclass_fields__}


@dataclass(frozen=True)
class Resolution:
    """Both bootstraps at bandwidth 4 with B = 9999 (not pre-registered)."""

    p_sieve: tuple[float, ...]
    p_block: tuple[float, ...]
    holm_sieve: tuple[float, ...]
    holm_block: tuple[float, ...]

    @property
    def survivors(self) -> int:
        return sum(a <= ALPHA and b <= ALPHA
                   for a, b in zip(self.holm_sieve, self.holm_block, strict=True))


def resolution_check(specs: Sequence[Spec], draws: int = B_RESOLUTION) -> Resolution:
    """Repeat the primary tests with finer p-value resolution and fresh seeds."""
    sieve_p, block_p = [], []
    for spec in specs:
        width = spec.x.shape[1] // 2
        design = hac_design(spec.x, spec.periods, change_contrast(width), PRIMARY_BANDWIDTH)
        starts = segment_starts(spec.periods)
        rng = np.random.default_rng([SEED_TESTS, spec.index, 200])
        t_obs, t_star, _ = null_sieve_test(design, spec.y, starts, draws, rng)
        sieve_p.append(bootstrap_p(t_obs, t_star))
        block_rng = np.random.default_rng([SEED_TESTS, spec.index, 201])
        block_p.append(bootstrap_p(t_obs, null_block_test(design, spec.y, starts, draws,
                                                          block_rng)))
    return Resolution(tuple(sieve_p), tuple(block_p), tuple(holm_adjust(sieve_p)),
                      tuple(holm_adjust(block_p)))


def bandwidth_counts(rows: Sequence[ChangeTest]) -> dict[int, dict[str, int]]:
    """Holm-significant counts per bandwidth: asymptotic (published) and sieve bootstrap."""
    out = {}
    for bandwidth in BANDWIDTHS:
        holm = holm_adjust([r.p_sieve_bandwidth[bandwidth] for r in rows])
        out[bandwidth] = {"sieve": int(sum(p <= ALPHA for p in holm))}
    return out


@dataclass(frozen=True)
class SizeResult:
    """Monte Carlo rejection rates at 5% under the fitted null DGP of one test."""

    geo: str
    participation: bool
    asymptotic: float
    bootstrap: float


def size_study(specs: Sequence[Spec], replications: int = R_SIZE,
               draws: int = B_SIZE) -> list[SizeResult]:
    """Section 4: the asymptotic and bootstrap tests on data simulated under the null."""
    out = []
    for spec in specs:
        width = spec.x.shape[1] // 2
        design = hac_design(spec.x, spec.periods, change_contrast(width), PRIMARY_BANDWIDTH)
        starts = segment_starts(spec.periods)
        rng = np.random.default_rng([SEED_SIZE, spec.index])
        beta_r, resid = restricted_fit(design, spec.y)
        sieve = fit_sieve(resid, starts)
        simulated = (design.x @ beta_r)[:, None] + sieve_errors(sieve, resid, replications,
                                                                 rng)
        t_sim = hac_t(design, simulated)[2]
        rejected = 0
        for r in range(replications):
            t_obs, t_star, _ = null_sieve_test(design, simulated[:, r], starts, draws, rng)
            rejected += bootstrap_p(t_obs, t_star) <= ALPHA
        out.append(SizeResult(spec.geo, spec.participation,
                              float(np.mean(np.abs(t_sim) > Z_975)), rejected / replications))
    return out


@dataclass(frozen=True)
class PostSum:
    """The post-2021 GDP sum with asymptotic and bootstrap-t 95% intervals."""

    geo: str
    participation: bool
    theta_post: float
    se: float
    normal_low: float
    normal_high: float
    boot_low: float
    boot_high: float

    @property
    def sign(self) -> str:
        if self.boot_high < 0:
            return "negative"
        if self.boot_low > 0:
            return "positive"
        return "not reportable"


def post_sums(specs: Sequence[Spec], draws: int = B_MAIN) -> list[PostSum]:
    """Section 5: bootstrap-t intervals from the unrestricted sieve of the post regression."""
    out = []
    for spec in specs:
        width = spec.x_post.shape[1]
        design = hac_design(spec.x_post, spec.post_periods, sum_contrast(width),
                            PRIMARY_BANDWIDTH)
        estimate, se, _ = hac_t(design, spec.y_post)
        theta, s = float(estimate[0]), float(se[0])
        beta = design.projector @ spec.y_post
        resid = spec.y_post - design.x @ beta
        starts = segment_starts(spec.post_periods)
        sieve = fit_sieve(resid, starts)
        rng = np.random.default_rng([SEED_POST, spec.index])
        y_star = (design.x @ beta)[:, None] + sieve_errors(sieve, resid, draws, rng)
        e_star, se_star, _ = hac_t(design, y_star)
        q_low, q_high = np.quantile((e_star - theta) / se_star, [0.025, 0.975])
        out.append(PostSum(spec.geo, spec.participation, theta, s, theta - Z_975 * s,
                           theta + Z_975 * s, theta - float(q_high) * s,
                           theta - float(q_low) * s))
    return out


# ----------------------------------------------------------------------------------
# Review, tables and report
# ----------------------------------------------------------------------------------


def binomial_range(total: int, level: float = ALPHA) -> tuple[float, float]:
    """Central 95% range of a rejection rate with ``total`` independent trials."""
    cdf, low = 0.0, None
    for k in range(total + 1):
        cdf += math.exp(math.lgamma(total + 1) - math.lgamma(k + 1) - math.lgamma(total - k + 1)
                        + k * math.log(level) + (total - k) * math.log1p(-level))
        if low is None and cdf >= 0.025:
            low = k
        if cdf >= 0.975:
            return (0 if low is None else low) / total, k / total
    return (0 if low is None else low) / total, 1.0


@dataclass
class SmallSampleReview:
    tests: list[ChangeTest]
    bandwidths: dict[int, dict[str, int]]
    published_holm: dict[int, int]
    size: list[SizeResult]
    post: list[PostSum]
    classical: dict[str, bool]
    flags: dict[str, tuple[bool, float]]       # geo -> definition flagged, provisional share
    periods: tuple[str, str, str, str]
    resolution: Resolution


def build_review(data_dir: Path, review_dir: Path) -> SmallSampleReview:
    """Run the whole study on the frozen snapshot."""
    from eu_okun.audit import audit_snapshot
    from eu_okun.stability import fit_stability

    audit = audit_snapshot(data_dir)
    end = audit["analysis_period"]["end"]
    if not isinstance(end, str):
        raise EurostatDataError("Invalid final quarter in audit.")
    panel = _read_panel(data_dir / "processed" / "eu27_quarterly.csv")
    specs = stability_specs(panel, end)
    tests = change_tests(specs)
    published_holm = {}
    for bandwidth in BANDWIDTHS:
        rows, _ = fit_stability(panel, end, hac_bandwidth=bandwidth)
        published_holm[bandwidth] = sum(r.p_change_holm <= ALPHA for r in rows)
        if bandwidth == PRIMARY_BANDWIDTH:
            for published, ours in zip(rows, tests, strict=True):
                if (not math.isclose(published.p_change, ours.p_asymptotic, rel_tol=1e-8,
                                     abs_tol=1e-12)
                        or not math.isclose(published.change_post_minus_pre, ours.change,
                                            rel_tol=1e-10, abs_tol=1e-12)):
                    raise ValueError(f"Published test of {published.geo} not reproduced.")
    flags = {}
    with (review_dir / "stability_review.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["participation"] == "False":
                flags[row["geo"]] = (row["post_definition_flagged"] == "True",
                                     float(row["post_provisional_share"]))
    first_pre, last_pre = specs[0].periods[0], specs[0].periods[42]
    return SmallSampleReview(
        tests, bandwidth_counts(tests), published_holm, size_study(specs), post_sums(specs),
        classical_weaker(review_dir), flags,
        (first_pre, last_pre, specs[0].post_periods[0], specs[0].post_periods[-1]),
        resolution_check(specs))


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


def _p(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def _label(geo: str, participation: bool) -> str:
    return f"{geo}{' (+dp)' if participation else ''}"


def render(review: SmallSampleReview, assets: str, command: str) -> str:
    tests = review.tests
    asym = [r for r in tests if r.p_asymptotic_holm <= ALPHA]
    sieve = [r for r in tests if r.p_sieve_holm <= ALPHA]
    block = [r for r in tests if r.p_block_holm <= ALPHA]
    survivors = [r for r in tests if r.survives]
    dependent = [r for r in tests if r.scheme_dependent]
    size = review.size
    pooled_asym = float(np.mean([s.asymptotic for s in size]))
    pooled_boot = float(np.mean([s.bootstrap for s in size]))
    low_r, high_r = binomial_range(R_SIZE)
    low_pool, high_pool = binomial_range(R_SIZE * len(size))
    over = [s for s in size if s.asymptotic > high_r]
    boot_over = [s for s in size if s.bootstrap > high_r]
    adequate = low_pool <= pooled_boot <= high_pool
    post = review.post
    reportable = [p for p in post if p.sign != "not reportable"]
    plain = [p for p in reportable if not p.participation]
    first_pre, last_pre, first_post, last_post = review.periods
    res = review.resolution
    lines = [
        "# Small-sample inference for the post-2021 window", "",
        "Generated by `eu-okun small-sample` from the frozen snapshot "
        "`snapshots/eurostat-2026-10-15`, following "
        "[SMALL_SAMPLE_PROTOCOL.md](../SMALL_SAMPLE_PROTOCOL.md), which was registered "
        "before any bootstrap was computed. Data, samples, design, statistic and "
        "multiplicity family are those of the published stability tests (`eu-okun "
        f"infer`): {first_pre}..{last_pre} against {first_post}..{last_post}, one GDP "
        "lag, a calendar-aware Newey-West t-statistic for the change in the GDP "
        "coefficient sum, Holm over 54 tests. Only the reference distribution changes. "
        "Every result is descriptive coefficient stability, not causal evidence.", "",
        "## Summary", "",
        f"- Holm-significant changes (bandwidth {PRIMARY_BANDWIDTH}): {len(asym)} with the "
        f"published normal reference, {len(sieve)} with the null-imposed sieve bootstrap, "
        f"{len(block)} with the moving-block bootstrap. **{len(survivors)} survive both "
        f"bootstraps** ({_names([_label(r.geo, r.participation) for r in survivors])}); "
        f"{len(dependent)} depend on the scheme"
        + (f" ({_names([_label(r.geo, r.participation) for r in dependent])})."
           if dependent else "."),
        f"- Resolution: with B = {B_MAIN} the smallest attainable p-value is "
        f"{1 / (B_MAIN + 1):.3f}, while Holm over {len(tests)} tests needs the smallest "
        f"p at or below {ALPHA / len(tests):.5f}. The smallest observed sieve p-value is "
        f"{min(r.p_sieve for r in tests):.3f}. A repeat with B = {B_RESOLUTION} (not "
        f"pre-registered, section 1.1) gives {res.survivors} surviving changes "
        f"(sieve Holm: {sum(h <= ALPHA for h in res.holm_sieve)}, block Holm: "
        f"{sum(h <= ALPHA for h in res.holm_block)}).",
        f"- Size under the fitted null DGPs ({R_SIZE} simulated data sets per test, "
        f"{len(size) * R_SIZE} in all): the nominal 5% normal test rejects "
        f"{pooled_asym:.1%} of the time overall and exceeds the binomial upper bound "
        f"({high_r:.3f}) in {len(over)} of {len(size)} tests; the sieve bootstrap test "
        f"rejects {pooled_boot:.1%} (binomial 95% range for the pooled rate "
        f"{low_pool:.3f}-{high_pool:.3f}): "
        + ("**adequate size**." if adequate else "**size outside the binomial range**."),
        f"- Post-2021 sums (bootstrap-t 95% intervals, 17 quarters): the sign is "
        f"reportable in {len(reportable)} of {len(post)} specifications "
        f"({_names([_label(p.geo, p.participation) for p in reportable])}); chance alone "
        f"would give about {0.05 * len(post):.1f}. Without the participation variants, "
        f"which are not independent of them: {len(plain)} of {len(post) // 2} countries "
        f"({_names([f'{p.geo} {p.sign}' for p in plain])}), against about "
        f"{0.05 * len(post) / 2:.1f} by chance.", ""]

    # 1. Change tests
    lines += ["## 1. Pre/post change tests", "",
              "Change = post-minus-pre GDP coefficient sum (pp per pp). p-values: normal "
              "(published), null-imposed sieve bootstrap and moving-block bootstrap "
              f"(B = {B_MAIN}); Holm over the 54 tests. 'Null size' is the share of the "
              "sieve bootstrap draws in which the normal test would reject at 5%: the "
              "size of the published test under that test's own fitted null. Flags: "
              "post-2021 definition change (d) and share of provisional GDP.", ""]
    rows = []
    for r in tests:
        definition, provisional = review.flags.get(r.geo, (False, 0.0))
        flag = ("d" if definition else "") + (f" p {provisional:.0%}" if provisional else "")
        rows.append([_label(r.geo, r.participation), f"{r.change:+.3f}", f"{r.se:.3f}",
                     _p(r.p_asymptotic), _p(r.p_asymptotic_holm), _p(r.p_sieve),
                     _p(r.p_sieve_holm), _p(r.p_block), _p(r.p_block_holm),
                     f"{r.asymptotic_size_under_null:.3f}",
                     "**yes**" if r.survives else ("scheme" if r.scheme_dependent else "-"),
                     flag.strip() or "-"])
    lines += _table(["Test", "Change", "HAC s.e.", "p normal", "Holm", "p sieve", "Holm",
                     "p block", "Holm", "Null size", "Survives", "Flags"], rows,
                    "lrrrrrrrrrll")
    lines += [f"![p-values]({assets}/p_values.png)", ""]
    rows = [[str(b), str(review.published_holm[b]), str(review.bandwidths[b]["sieve"])]
            for b in BANDWIDTHS]
    lines += ["Holm-significant counts by HAC bandwidth (normal reference as published; "
              "sieve bootstrap with the same bandwidth):", ""]
    lines += _table(["Bandwidth", "Normal", "Sieve bootstrap"], rows)
    weaker = sorted(g for g, w in review.classical.items() if w)
    surviving_geo = sorted({r.geo for r in survivors})
    lines += [f"- The review's descriptive weaker-after-2021 set ({_names(weaker)}) and the "
              f"countries with a surviving change ({_names(surviving_geo)}) overlap in "
              f"{_names(sorted(set(weaker) & set(surviving_geo)))}.",
              "- A surviving change says the GDP sum differs between the two periods under "
              "a valid reference distribution; it does not say why. The post-2021 window "
              "contains the energy shock, high inflation, a definition change in ES and FR "
              "and provisional GDP.", ""]

    # 1.1 Resolution
    lines += ["### 1.1 Resolution of the bootstrap p-values (not pre-registered)", "",
              f"The registered B = {B_MAIN} cannot produce a p-value below "
              f"{1 / (B_MAIN + 1):.3f}, yet Holm over {len(tests)} tests rejects the "
              f"smallest p only at or below {ALPHA / len(tests):.5f}: a test at the "
              "resolution floor could never survive. Both schemes were therefore repeated "
              f"with B = {B_RESOLUTION} and fresh seeds. This check was added after the "
              "registered run and is labelled as such in the protocol change log.", ""]
    order = sorted(range(len(tests)), key=lambda i: tests[i].p_sieve)[:8]
    rows = [[_label(tests[i].geo, tests[i].participation), _p(tests[i].p_sieve),
             f"{res.p_sieve[i]:.4f}", f"{res.holm_sieve[i]:.3f}", _p(tests[i].p_block),
             f"{res.p_block[i]:.4f}", f"{res.holm_block[i]:.3f}"] for i in order]
    lines += ["The eight smallest sieve p-values:", ""]
    lines += _table(["Test", f"p sieve, B = {B_MAIN}", f"p sieve, B = {B_RESOLUTION}", "Holm",
                     f"p block, B = {B_MAIN}", f"p block, B = {B_RESOLUTION}", "Holm"], rows)
    lines += [f"- Surviving changes with B = {B_RESOLUTION}: {res.survivors}.", ""]

    # 2. Size
    lines += ["## 2. Size of the tests", "",
              f"For each test, {R_SIZE} data sets are simulated from its fitted null sieve "
              "DGP (restricted coefficients, AR(1) errors per period, wild innovations); "
              "on each, the normal test and the full bootstrap test (B = "
              f"{B_SIZE}, everything re-estimated) are applied at 5%. With {R_SIZE} "
              f"replications a correctly sized test lies in {low_r:.3f}-{high_r:.3f} "
              "95% of the time.", ""]
    rates_a = np.asarray([s.asymptotic for s in size])
    rates_b = np.asarray([s.bootstrap for s in size])
    rows = [["Normal (published)", f"{np.median(rates_a):.3f}", f"{rates_a.min():.3f}",
             f"{rates_a.max():.3f}", f"{pooled_asym:.3f}", str(len(over))],
            ["Sieve bootstrap", f"{np.median(rates_b):.3f}", f"{rates_b.min():.3f}",
             f"{rates_b.max():.3f}", f"{pooled_boot:.3f}", str(len(boot_over))]]
    lines += _table(["Test", "Median rate", "Min", "Max", "Pooled", "Tests above "
                     f"{high_r:.3f}"], rows)
    lines += [f"![Size study]({assets}/size.png)", ""]

    # 3. Post sums
    lines += ["## 3. Post-2021 sums", "",
              "Post-only regression on the 17 quarters, GDP sum with a normal 95% interval "
              f"and a bootstrap-t 95% interval (unrestricted sieve, B = {B_MAIN}). The sign "
              "is reportable when the bootstrap-t interval excludes zero (no multiplicity "
              "adjustment).", ""]
    rows = [[_label(p.geo, p.participation), f"{p.theta_post:+.3f}",
             f"[{p.normal_low:+.3f}, {p.normal_high:+.3f}]",
             f"[{p.boot_low:+.3f}, {p.boot_high:+.3f}]", p.sign] for p in post]
    lines += _table(["Specification", "Post-2021 sum", "Normal 95%", "Bootstrap-t 95%",
                     "Sign"], rows, "lrrrl")
    lines += [f"![Post-2021 sums]({assets}/post_sums.png)", ""]

    lines += ["## 4. Limitations", "",
              "- The bootstraps condition on the GDP regressors and model the errors as "
              "AR(1) (sieve) or short-range dependent (blocks of 4) within each period; "
              "dependence beyond that is not reproduced.",
              "- The size study uses the same sieve DGP family as the bootstrap, so it "
              "checks the bootstrap test where its own assumptions hold, and the normal "
              "test under realistic nulls; it is not a check against every alternative "
              "error process.",
              "- The post-2021 window is 17 quarters of one snapshot; the roadmap's re-test "
              "with more quarters needs a new snapshot.", "",
              "## Reproduce", "", "```bash", command, "```", ""]
    return "\n".join(lines)


def write_tables(review: SmallSampleReview, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    def write(name: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        with (directory / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)

    write("change_tests.csv",
          ["geo", "participation", "change", "se", "t", "p_asymptotic", "p_asymptotic_holm",
           "p_sieve", "p_sieve_holm", "p_block", "p_block_holm", "null_size", "rho_pre",
           "rho_post", *[f"p_sieve_bw{b}" for b in BANDWIDTHS], "survives"],
          [[r.geo, r.participation, repr(r.change), repr(r.se), repr(r.t),
            repr(r.p_asymptotic), repr(r.p_asymptotic_holm), repr(r.p_sieve),
            repr(r.p_sieve_holm), repr(r.p_block), repr(r.p_block_holm),
            repr(r.asymptotic_size_under_null), repr(r.rho_pre), repr(r.rho_post),
            *[repr(r.p_sieve_bandwidth[b]) for b in BANDWIDTHS], r.survives]
           for r in review.tests])
    write("size_study.csv", ["geo", "participation", "asymptotic", "bootstrap"],
          [[s.geo, s.participation, repr(s.asymptotic), repr(s.bootstrap)]
           for s in review.size])
    write("post_sums.csv", ["geo", "participation", "theta_post", "se", "normal_low",
                            "normal_high", "boot_low", "boot_high", "sign"],
          [[p.geo, p.participation, repr(p.theta_post), repr(p.se), repr(p.normal_low),
            repr(p.normal_high), repr(p.boot_low), repr(p.boot_high), p.sign]
           for p in review.post])
    (directory / "summary.json").write_text(json.dumps({
        "settings": {"B": B_MAIN, "B_size": B_SIZE, "R_size": R_SIZE, "block": BLOCK,
                     "bandwidths": list(BANDWIDTHS), "seeds": [SEED_TESTS, SEED_SIZE,
                                                               SEED_POST]},
        "published_holm": review.published_holm,
        "resolution": {"B": B_RESOLUTION, "survivors": review.resolution.survivors,
                       "p_sieve": list(review.resolution.p_sieve),
                       "p_block": list(review.resolution.p_block)},
        "bandwidth_counts": review.bandwidths,
        "survivors": [[r.geo, r.participation] for r in review.tests if r.survives],
        "scheme_dependent": [[r.geo, r.participation] for r in review.tests
                             if r.scheme_dependent],
        "size_pooled": {"asymptotic": float(np.mean([s.asymptotic for s in review.size])),
                        "bootstrap": float(np.mean([s.bootstrap for s in review.size]))},
        "post_reportable": [[p.geo, p.participation, p.sign] for p in review.post
                            if p.sign != "not reportable"],
        "tests": [{**asdict(r), "survives": r.survives} for r in review.tests],
    }, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_figures(review: SmallSampleReview, directory: Path, source: str) -> None:
    from eu_okun.review_figures import BLUE, MUTED, ORANGE, SURFACE, _finish, _pyplot

    plt = _pyplot()
    from matplotlib.lines import Line2D

    tests = review.tests
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    for r in tests:
        colour = ORANGE if r.survives else BLUE
        ax.plot([r.p_asymptotic], [r.p_sieve], marker="o" if not r.participation else "s",
                ms=5.5, color=colour, mec=SURFACE, mew=1.0, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axline((1e-4, 1e-4), (1.0, 1.0), color=MUTED, lw=1)
    ax.axhline(1.0 / (B_MAIN + 1), color=MUTED, lw=1, ls=":")
    ax.set_xlabel("p-value, normal reference (published)")
    ax.set_ylabel("p-value, null-imposed sieve bootstrap")
    ax.set_title("Pre/post change tests: normal against bootstrap p-values")
    ax.legend(handles=[
        Line2D([], [], marker="o", color=ORANGE, lw=0, label="survives both bootstraps"),
        Line2D([], [], marker="o", color=BLUE, lw=0, label="does not survive"),
        Line2D([], [], marker="s", color=MUTED, lw=0, label="with participation"),
        Line2D([], [], color=MUTED, ls=":", label=f"smallest bootstrap p, 1/{B_MAIN + 1}"),
    ], fontsize=8.5, loc="lower right")
    _finish(plt, fig, directory / "p_values.png", source, credit_y=-0.04)

    size = review.size
    low_r, high_r = binomial_range(R_SIZE)
    order = sorted(range(len(size)), key=lambda i: (size[i].geo, size[i].participation))
    ys = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 11.0))
    ax.axvspan(low_r, high_r, color=MUTED, alpha=0.15, lw=0)
    for y, i in zip(ys, order, strict=True):
        ax.plot([size[i].asymptotic], [y], marker="o", ms=5, color=ORANGE, lw=0)
        ax.plot([size[i].bootstrap], [y], marker="o", ms=5, color=BLUE, lw=0)
    ax.axvline(ALPHA, color=MUTED, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([_label(size[i].geo, size[i].participation) for i in order],
                       fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Rejection rate at nominal 5% under the fitted null")
    ax.set_title("Size of the stability tests (grey band: binomial 95% range)")
    ax.legend(handles=[Line2D([], [], marker="o", color=ORANGE, lw=0, label="normal"),
                       Line2D([], [], marker="o", color=BLUE, lw=0, label="sieve bootstrap")],
              loc="upper left", bbox_to_anchor=(0.0, -0.05), fontsize=8.5, ncol=2)
    _finish(plt, fig, directory / "size.png", source, credit_y=-0.07)

    post = [p for p in review.post if not p.participation]
    ys = np.arange(len(post))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 9.0))
    for y, p in zip(ys, post, strict=True):
        ax.plot([p.normal_low, p.normal_high], [y + 0.15] * 2, color=MUTED, lw=1.4)
        ax.plot([p.boot_low, p.boot_high], [y - 0.15] * 2, color=BLUE, lw=1.6)
        ax.plot([p.theta_post], [y], marker="o", ms=5, color=BLUE, mec=SURFACE, mew=1.0)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([p.geo for p in post])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Post-2021 GDP coefficient sum, K = 1 (pp per pp); 95% intervals")
    ax.set_title("Post-2021 sums: normal (grey) and bootstrap-t (blue) intervals")
    _finish(plt, fig, directory / "post_sums.png", source, credit_y=-0.06)


def write_report(data_dir: Path, output_dir: Path, report: Path | None = None,
                 review_dir: Path = Path("docs/review"), figures: bool = True,
                 ) -> SmallSampleReview:
    """Run the study and write its tables, figures and Markdown report."""
    review = build_review(data_dir, review_dir)
    report = report or output_dir / "SMALL_SAMPLE_INFERENCE.md"
    write_tables(review, output_dir)
    if figures:
        source = ("Source: Eurostat, snapshot 2026-10-15; SMALL_SAMPLE_PROTOCOL.md. "
                  "Descriptive coefficient stability, not causal.")
        write_figures(review, output_dir, source)
    assets = Path(os.path.relpath(output_dir, report.parent)).as_posix()
    command = (f"poetry run eu-okun small-sample --data-dir {data_dir.as_posix()} "
               f"--output-dir {output_dir.as_posix()} --report {report.as_posix()} "
               f"--review-dir {review_dir.as_posix()}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(review, assets, command), encoding="utf-8", newline="\n")
    return review
