"""Sign-restricted structural VAR, robust Bayesian inference (SIGN_RESTRICTIONS_PROTOCOL.md).

A pooled panel VAR in GDP growth, GDP-deflator inflation and the changes in the
unemployment and participation rates, with country fixed effects. Demand, technology and
labour-supply shocks are set-identified by impact sign restrictions. For every posterior
draw of the reduced form the identified set of each target is approximated by the range
over admissible Haar rotations (an inner approximation), and summarised as in Giacomini
and Kitagawa (2021): the set of posterior means and a robust credible region, which do not
depend on a prior over rotations. The single uniform-rotation posterior is computed only
as a labelled contrast, and the 24 recursive orderings only as a labelled, non-causal
measure of sensitivity to ordering.

Nothing here is point-identified. The restrictions are untestable except jointly, through
the probability that the identified set is non-empty.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.acquisition import fetch_dataset
from eu_okun.contract import EU27, EurostatDataError, SeriesSpec, quarter_sequence
from eu_okun.jsonstat import decode_series

VARIABLES: tuple[str, ...] = ("g", "pi", "du", "dp")
G, PI, DU, DP = range(4)
CUMULATIVE = (G, DU, DP)                       # growth and changes: cumulated to levels
SHOCKS: tuple[str, ...] = ("demand", "technology", "labour_supply")
SHOCK_LABELS = {"demand": "Demand", "technology": "Technology",
                "labour_supply": "Labour supply"}
HORIZONS = (0, 4, 8)
RATIO_HORIZONS = (4, 8)
FEVD_HORIZON = 8
PANDEMIC = ("2020-Q1", "2021-Q4")
SEED = 20261018

DEFLATOR = SeriesSpec("deflator", "namq_10_gdp", "gdp_deflator_growth_pct", {
    "freq": "Q", "unit": "PD_PCH_PRE_NAC", "s_adj": "SCA", "na_item": "B1GQ"})

Restrictions = dict[str, dict[int, int]]        # shock -> {variable index: sign}
PRIMARY_RESTRICTIONS: Restrictions = {
    "demand": {G: 1, PI: 1, DU: -1},
    "technology": {G: 1, PI: -1, DU: -1},
    "labour_supply": {PI: -1, DU: 1, DP: 1},
}


@dataclass(frozen=True)
class Spec:
    """One registered specification (protocol, sections 2, 3 and 7)."""

    key: str
    label: str
    restrictions: Restrictions = field(default_factory=lambda: PRIMARY_RESTRICTIONS)
    restrict_h1: bool = False
    lags: int = 2
    exclude: tuple[str, ...] = ()
    exclude_pandemic: bool = True
    pandemic_margin: int = 0     # quarters added before t-lags in the pandemic window


PRIMARY = Spec("primary", "Primary")
VARIANTS: tuple[Spec, ...] = (
    replace(PRIMARY, key="tech_no_du", label="Technology without the unemployment sign",
            restrictions={**PRIMARY_RESTRICTIONS, "technology": {G: 1, PI: -1}}),
    replace(PRIMARY, key="ls_output_up", label="Labour supply with output up",
            restrictions={**PRIMARY_RESTRICTIONS,
                          "labour_supply": {G: 1, PI: -1, DU: 1, DP: 1}}),
    replace(PRIMARY, key="h01", label="Restrictions at h = 0 and 1", restrict_h1=True),
    replace(PRIMARY, key="lags4", label="Four lags", lags=4),
    replace(PRIMARY, key="no_ie", label="Without Ireland", exclude=("IE",)),
    replace(PRIMARY, key="pandemic", label="Pandemic included", exclude_pandemic=False),
)


# ----------------------------------------------------------------------------------
# Deflator snapshot
# ----------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _deflator_rows(payload: Mapping[str, Any], start: str, end: str) -> list[str]:
    cells = decode_series(payload, DEFLATOR)
    lines = ["geo,quarter,gdp_deflator_growth_pct,flag"]
    for geo in EU27:
        for q in quarter_sequence(start, end):
            cell = cells.get((geo, q))
            value = "" if cell is None or cell.value is None else repr(cell.value)
            lines.append(f"{geo},{q},{value},{'' if cell is None else cell.status}")
    return lines


def fetch_deflator(output_dir: Path, start: str = "2008-Q1", end: str = "2026-Q2",
                   ) -> dict[str, Any]:
    """Download the quarterly GDP deflator growth, freeze raw bytes and a processed CSV."""
    raw, payload, url = fetch_dataset(DEFLATOR, start, end)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    (output_dir / "processed").mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw" / "namq_10_gdp_deflator.json"
    raw_path.write_bytes(raw)
    processed = output_dir / "processed" / "eu27_deflator.csv"
    processed.write_text("\n".join(_deflator_rows(payload, start, end)) + "\n",
                         encoding="utf-8", newline="\n")
    manifest = {
        "description": "GDP deflator growth for SIGN_RESTRICTIONS_PROTOCOL.md",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "dataset": DEFLATOR.dataset, "filters": dict(DEFLATOR.filters), "url": url,
        "updated": str(payload.get("updated", "")),
        "period": {"start": start, "end": end},
        "raw_sha256": _sha256(raw_path), "processed_sha256": _sha256(processed),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                              encoding="utf-8", newline="\n")
    return manifest


def audit_deflator(directory: Path) -> dict[str, Any]:
    """Check hashes and that the processed CSV re-derives from the raw response."""
    manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text(
        encoding="utf-8"))
    raw_path = directory / "raw" / "namq_10_gdp_deflator.json"
    processed = directory / "processed" / "eu27_deflator.csv"
    if _sha256(raw_path) != manifest["raw_sha256"]:
        raise EurostatDataError("Deflator raw response changed.")
    if _sha256(processed) != manifest["processed_sha256"]:
        raise EurostatDataError("Deflator CSV does not match its manifest hash.")
    lines = _deflator_rows(json.loads(raw_path.read_bytes()), manifest["period"]["start"],
                           manifest["period"]["end"])
    if processed.read_text(encoding="utf-8").splitlines() != lines:
        raise EurostatDataError("Deflator CSV does not re-derive from the raw response.")
    return manifest


def load_deflator(directory: Path) -> dict[tuple[str, str], float]:
    out = {}
    for line in (directory / "processed" / "eu27_deflator.csv").read_text(
            encoding="utf-8").splitlines()[1:]:
        geo, quarter, value, _ = line.split(",")
        if value:
            out[(geo, quarter)] = float(value)
    return out


# ----------------------------------------------------------------------------------
# Reduced form
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class VarData:
    """x[i, t, :] = (g, pi, du, dp), NaN where missing."""

    countries: tuple[str, ...]
    quarters: tuple[str, ...]
    x: np.ndarray


def var_data(panel: Mapping[tuple[str, str], Mapping[str, float | None]],
             deflator: Mapping[tuple[str, str], float], first: str, last: str,
             countries: Sequence[str] = EU27) -> VarData:
    quarters = quarter_sequence(first, last)
    x = np.full((len(countries), len(quarters), 4), np.nan)
    for i, geo in enumerate(countries):
        for t, q in enumerate(quarters):
            row = panel.get((geo, q), {})
            prev = panel.get((geo, quarters[t - 1]), {}) if t else {}
            g, u, p = row.get("gdp"), row.get("unemployment"), row.get("participation")
            u0, p0 = prev.get("unemployment"), prev.get("participation")
            x[i, t, G] = np.nan if g is None else g
            x[i, t, PI] = deflator.get((geo, q), np.nan)
            x[i, t, DU] = np.nan if u is None or u0 is None else u - u0
            x[i, t, DP] = np.nan if p is None or p0 is None else p - p0
    return VarData(tuple(countries), tuple(quarters), x)


def var_rows(data: VarData, spec: Spec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Y (n, 4), X (n, 4 lags) and country index of every usable country-quarter."""
    ys, xs, groups = [], [], []
    for i, geo in enumerate(data.countries):
        if geo in spec.exclude:
            continue
        for t in range(spec.lags, len(data.quarters)):
            window = data.quarters[max(0, t - spec.lags - spec.pandemic_margin): t + 1]
            if spec.exclude_pandemic and any(PANDEMIC[0] <= q <= PANDEMIC[1] for q in window):
                continue
            block = data.x[i, t - spec.lags: t + 1]
            if not np.all(np.isfinite(block)):
                continue
            ys.append(block[-1])
            xs.append(np.concatenate([block[-1 - lag] for lag in range(1, spec.lags + 1)]))
            groups.append(i)
    return np.asarray(ys), np.asarray(xs), np.asarray(groups)


def demean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.array(values, dtype=float, copy=True)
    for g in np.unique(groups):
        rows = groups == g
        out[rows] -= out[rows].mean(axis=0)
    return out


@dataclass(frozen=True)
class ReducedForm:
    a_hat: np.ndarray            # (4 lags, 4): x_t' = sum_l x_{t-l}' A_l
    s: np.ndarray                # residual cross-product
    xtx_inv: np.ndarray
    n: int
    k: int
    groups: int
    lags: int

    @property
    def dof(self) -> int:
        return self.n - self.k - self.groups

    @property
    def sigma_hat(self) -> np.ndarray:
        return np.asarray(self.s / self.dof)


def fit_var(y: np.ndarray, x: np.ndarray, groups: np.ndarray, lags: int) -> ReducedForm:
    """Within (fixed-effects) OLS of the pooled panel VAR."""
    yd, xd = demean(y, groups), demean(x, groups)
    xtx_inv = np.linalg.inv(xd.T @ xd)
    a_hat = xtx_inv @ xd.T @ yd
    resid = yd - xd @ a_hat
    return ReducedForm(a_hat, resid.T @ resid, xtx_inv, len(y), x.shape[1],
                       len(np.unique(groups)), lags)


def companion_stable(a: np.ndarray, lags: int) -> bool:
    n = a.shape[1]
    top = np.hstack([a[j * n:(j + 1) * n].T for j in range(lags)])
    if lags == 1:
        comp = top
    else:
        comp = np.vstack([top, np.hstack([np.eye(n * (lags - 1)),
                                          np.zeros((n * (lags - 1), n))])])
    return bool(np.max(np.abs(np.linalg.eigvals(comp))) < 1.0)


def ma_coefficients(a: np.ndarray, lags: int, horizon: int) -> np.ndarray:
    """Psi_0..Psi_h with x_t = sum_h Psi_h eta_{t-h}: (h + 1, 4, 4)."""
    n = a.shape[1]
    phi = [a[j * n:(j + 1) * n].T for j in range(lags)]
    psi = [np.eye(n)]
    for h in range(1, horizon + 1):
        psi.append(sum((phi[j - 1] @ psi[h - j] for j in range(1, min(h, lags) + 1)),
                       np.zeros((n, n))))
    return np.asarray(psi)


@dataclass(frozen=True)
class Draw:
    a: np.ndarray
    sigma: np.ndarray


def posterior_draws(rf: ReducedForm, count: int, rng: np.random.Generator,
                    ) -> tuple[list[Draw], int]:
    """Flat-prior Normal-inverse-Wishart posterior; explosive draws are discarded."""
    from scipy.stats import invwishart

    chol_x = np.linalg.cholesky(rf.xtx_inv)
    draws: list[Draw] = []
    discarded = 0
    while len(draws) < count:
        sigma = np.asarray(invwishart(df=rf.dof, scale=rf.s).rvs(random_state=rng))
        z = rng.standard_normal(rf.a_hat.shape)
        a = rf.a_hat + chol_x @ z @ np.linalg.cholesky(sigma).T
        if companion_stable(a, rf.lags):
            draws.append(Draw(a, sigma))
        else:
            discarded += 1
        if discarded > 10 * count:
            raise ValueError("Posterior draws are almost all explosive.")
    return draws, discarded


# ----------------------------------------------------------------------------------
# Rotations and admissible sets
# ----------------------------------------------------------------------------------


def haar_rotations(m: int, rng: np.random.Generator, n: int = 4) -> np.ndarray:
    """m orthonormal matrices from the Haar distribution (QR with sign normalisation)."""
    z = rng.standard_normal((m, n, n))
    q, r = np.linalg.qr(z)
    signs = np.sign(np.diagonal(r, axis1=1, axis2=2))
    signs[signs == 0] = 1.0
    return np.asarray(q * signs[:, None, :])


def admissible(responses: Sequence[np.ndarray], restrictions: Restrictions,
               ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Valid assignments of rotation columns (with signs) to the restricted shocks.

    ``responses`` are the period responses to be restricted, each (m, 4 variables,
    4 columns). Returns, per valid column permutation, the rotation indices, the chosen
    columns (one per shock, in SHOCKS order) and their signs.
    """
    names = list(restrictions)
    ok: dict[str, dict[int, np.ndarray]] = {}
    for shock, signs in restrictions.items():
        ok[shock] = {}
        for sign in (1, -1):
            cond = np.ones(responses[0].shape[0:1] + (4,), dtype=bool)
            for resp in responses:
                for var, s in signs.items():
                    cond &= sign * s * resp[:, var, :] > 0
            ok[shock][sign] = cond                       # (m, columns)
    out = []
    for columns in itertools.permutations(range(4), len(names)):
        valid = np.ones(responses[0].shape[0], dtype=bool)
        chosen_signs = []
        for shock, col in zip(names, columns, strict=True):
            pos, neg = ok[shock][1][:, col], ok[shock][-1][:, col]
            valid &= pos | neg
            chosen_signs.append(np.where(pos, 1.0, -1.0))
        idx = np.nonzero(valid)[0]
        if idx.size:
            out.append((idx, np.asarray(columns),
                        np.column_stack(chosen_signs)[idx]))
    return out


@dataclass
class DrawSets:
    """Identified-set bounds and the uniform-prior pick of one posterior draw."""

    nonempty: bool
    admissible_count: int
    lower: dict[str, float]
    upper: dict[str, float]
    single: dict[str, float]


def target_names() -> list[str]:
    names = []
    for shock in SHOCKS:
        for h in HORIZONS:
            names += [f"{shock}:gdp:{h}", f"{shock}:unemployment:{h}",
                      f"{shock}:participation:{h}"]
        names.append(f"{shock}:fevd_du:{FEVD_HORIZON}")
    for shock in ("demand", "technology"):
        names += [f"{shock}:okun:{h}" for h in RATIO_HORIZONS]
    return names


def identified_sets(draw: Draw, spec: Spec, rotations: np.ndarray) -> DrawSets:
    """Bounds of every target over the admissible rotations of one posterior draw."""
    horizon = max(max(HORIZONS), FEVD_HORIZON)
    psi = ma_coefficients(draw.a, spec.lags, horizon)
    chol = np.linalg.cholesky(draw.sigma)
    impact = chol @ rotations                                       # (m, 4, 4)
    restricted = [impact]
    if spec.restrict_h1:
        restricted.append(psi[1] @ impact)
    assignments = admissible(restricted, spec.restrictions)
    names = target_names()
    if not assignments:
        nan = {n: math.nan for n in names}
        return DrawSets(False, 0, dict(nan), dict(nan), dict(nan))
    # Structural impact columns of every admissible (rotation, assignment): (a, 4, 3).
    cols = np.concatenate([impact[idx][:, :, columns] * signs[:, None, :]
                           for idx, columns, signs in assignments])
    order = np.argsort(np.concatenate([idx for idx, _, _ in assignments]), kind="stable")
    cols = cols[order]                     # rotation order: the first is the uniform pick
    period = psi[None] @ cols[:, None]                              # (a, h, 4, 3)
    cumulative = np.cumsum(period, axis=1)
    total_var = np.einsum("hij,jk,hlk->il", psi[:FEVD_HORIZON + 1], draw.sigma,
                          psi[:FEVD_HORIZON + 1])[DU, DU]
    values: dict[str, np.ndarray] = {}
    for s, shock in enumerate(SHOCKS):
        for h in HORIZONS:
            values[f"{shock}:gdp:{h}"] = cumulative[:, h, G, s]
            values[f"{shock}:unemployment:{h}"] = cumulative[:, h, DU, s]
            values[f"{shock}:participation:{h}"] = cumulative[:, h, DP, s]
        values[f"{shock}:fevd_du:{FEVD_HORIZON}"] = (
            np.sum(period[:, :FEVD_HORIZON + 1, DU, s] ** 2, axis=1) / total_var)
    lower, upper, single = {}, {}, {}
    for name, v in values.items():
        lower[name], upper[name], single[name] = float(v.min()), float(v.max()), float(v[0])
    for s, shock in enumerate(("demand", "technology")):
        for h in RATIO_HORIZONS:
            gdp = cumulative[:, h, G, s]
            ratio = cumulative[:, h, DU, s] / gdp
            name = f"{shock}:okun:{h}"
            single[name] = float(ratio[0])
            if np.all(gdp > 0) or np.all(gdp < 0):
                lower[name], upper[name] = float(ratio.min()), float(ratio.max())
            else:
                lower[name], upper[name] = -math.inf, math.inf
    return DrawSets(True, int(cols.shape[0]), lower, upper, single)


# ----------------------------------------------------------------------------------
# Robust Bayes summaries
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Summary:
    """Robust Bayes summary of one target, and the uniform-prior contrast."""

    target: str
    mean_lower: float
    mean_upper: float
    region_lower: float
    region_upper: float
    single_median: float
    single_low68: float
    single_high68: float
    single_low90: float
    single_high90: float

    @property
    def bounded(self) -> bool:
        return math.isfinite(self.region_lower) and math.isfinite(self.region_upper)


def summarise(sets: Sequence[DrawSets]) -> list[Summary]:
    """Set of posterior means, robust 90% credible region and uniform-prior bands."""
    kept = [s for s in sets if s.nonempty]
    out = []
    for name in target_names():
        lo = np.asarray([s.lower[name] for s in kept])
        hi = np.asarray([s.upper[name] for s in kept])
        single = np.asarray([s.single[name] for s in kept])
        q = np.percentile(single, [50, 16, 84, 5, 95])
        mean_lo = float(np.mean(lo)) if np.all(np.isfinite(lo)) else -math.inf
        mean_hi = float(np.mean(hi)) if np.all(np.isfinite(hi)) else math.inf
        # Order statistics, not interpolation: infinite bounds must stay infinite, and
        # the region is widened rather than narrowed.
        out.append(Summary(name, mean_lo, mean_hi,
                           float(np.percentile(lo, 5, method="lower")),
                           float(np.percentile(hi, 95, method="higher")),
                           *(float(v) for v in q)))
    return out


@dataclass
class SpecResults:
    """Everything estimated for one specification."""

    spec: Spec
    n: int
    countries: int
    draws: int
    discarded: int
    rotations: int
    nonempty_share: float
    mean_admissible: float
    summaries: list[Summary]
    convergence: dict[str, float] = field(default_factory=dict)

    def summary(self, target: str) -> Summary:
        return next(s for s in self.summaries if s.target == target)


def estimate(data: VarData, spec: Spec, draws: int = 1000, rotations: int = 20000,
             seed: int = SEED, check_draws: int = 0) -> SpecResults:
    """Posterior draws, identified sets per draw and robust Bayes summaries."""
    y, x, groups = var_rows(data, spec)
    rf = fit_var(y, x, groups, spec.lags)
    rng = np.random.default_rng(seed)
    posterior, discarded = posterior_draws(rf, draws, rng)
    sets = []
    for d, draw in enumerate(posterior):
        rot = haar_rotations(rotations, np.random.default_rng([seed, d]))
        sets.append(identified_sets(draw, spec, rot))
    nonempty = [s for s in sets if s.nonempty]
    if not nonempty:
        raise ValueError(f"No admissible rotation in any draw for {spec.key}.")
    convergence: dict[str, float] = {}
    if check_draws:
        moves = []
        for d, draw in enumerate(posterior[:check_draws]):
            if not sets[d].nonempty:
                continue
            wide = identified_sets(draw, spec, haar_rotations(
                2 * rotations, np.random.default_rng([seed, d, 2])))
            for name in target_names():
                if ":okun:" in name or not math.isfinite(sets[d].lower[name]):
                    continue
                width = sets[d].upper[name] - sets[d].lower[name]
                if width > 0:
                    moves.append((wide.upper[name] - wide.lower[name]) / width - 1)
        convergence = {"median_width_increase": float(np.median(moves)),
                       "max_width_increase": float(np.max(moves))}
    return SpecResults(spec, len(y), len(np.unique(groups)), len(posterior), discarded,
                       rotations, len(nonempty) / len(sets),
                       float(np.mean([s.admissible_count for s in nonempty])),
                       summarise(sets), convergence)


# ----------------------------------------------------------------------------------
# Recursive orderings (orthogonalised, not causal)
# ----------------------------------------------------------------------------------


def recursive_okun(draw: Draw, lags: int, ordering: Sequence[int],
                   horizons: Sequence[int] = RATIO_HORIZONS) -> dict[int, float]:
    """Okun ratio of the orthogonalised g-innovation under one Cholesky ordering."""
    perm = list(ordering)
    chol = np.linalg.cholesky(draw.sigma[np.ix_(perm, perm)])
    b = np.zeros((4, 4))
    b[np.ix_(perm, perm)] = chol                     # impact matrix in the original order
    column = b[:, G]                                 # the shock of the g equation
    psi = ma_coefficients(draw.a, lags, max(horizons))
    cum = np.cumsum(psi @ column, axis=0)
    return {h: float(cum[h, DU] / cum[h, G]) for h in horizons}


def recursive_orderings(data: VarData, spec: Spec, draws: int = 1000,
                        seed: int = SEED) -> list[dict[str, Any]]:
    """Posterior median of the orthogonalised Okun ratio for each of the 24 orderings."""
    y, x, groups = var_rows(data, spec)
    rf = fit_var(y, x, groups, spec.lags)
    posterior, _ = posterior_draws(rf, draws, np.random.default_rng(seed))
    rows = []
    for ordering in itertools.permutations(range(4)):
        values = [recursive_okun(d, spec.lags, ordering) for d in posterior]
        rows.append({"ordering": " > ".join(VARIABLES[v] for v in ordering),
                     **{f"okun_h{h}": float(np.median([v[h] for v in values]))
                        for h in RATIO_HORIZONS}})
    return rows
