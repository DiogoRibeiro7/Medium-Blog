"""Country-heterogeneity case study (``HETEROGENEITY_PROTOCOL.md``).

Builds the margins of labour-input adjustment from official data (productivity per hour,
hours per worker, domestic and resident employment, the commuting margin, unemployment,
participation, temporary and permanent employees), estimates their two-lag elasticities to
GDP growth country by country, and evaluates the pre-registered predictions with a
synchronised circular block bootstrap over quarters. Every result is an association or a
pattern consistent with a mechanism; nothing here is an identified causal effect.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eu_okun.contract import EU27, quarter_sequence

FOCUS: tuple[str, ...] = ("ES", "PT", "NL", "LU")
PANDEMIC = ("2020-Q1", "2021-Q4")
DATA_START, FIRST, LAST, PRE_LAST = "2008-Q1", "2009-Q3", "2026-Q2", "2019-Q4"
LAGS = 2
TIMING_LAGS = 4
REPLICATES = 1999
BLOCK = 4
SEED = 20261018
PERMUTATIONS = 9999
ALPHA = 0.05
MEASURES: tuple[str, ...] = ("du", "dp", "h", "e_dom", "hpw", "prod", "e_dom_pe", "e_res",
                             "comm")
YY_MEASURES: tuple[str, ...] = ("temp", "perm")
LABELS = {"du": "Unemployment rate (pp)", "dp": "Participation rate (pp)",
          "h": "Hours worked (%)", "e_dom": "Domestic employment (%)",
          "hpw": "Hours per worker (%)", "prod": "Output per hour (%)",
          "e_dom_pe": "Domestic employment, namq_10_pe (%)",
          "e_res": "Resident employment (%)", "comm": "Commuting margin (%)",
          "temp": "Temporary employees, y/y (%)", "perm": "Permanent employees, y/y (%)",
          "neg_du": "Minus unemployment rate (pp)"}


@dataclass(frozen=True)
class Settings:
    """Choices that the sensitivity analyses vary (protocol, section 7)."""

    key: str = "primary"
    lags: int = LAGS
    last: str = LAST
    block: int = BLOCK


PRIMARY = Settings()
SENSITIVITY: tuple[Settings, ...] = (
    Settings("lags_1", lags=1), Settings("lags_4", lags=4),
    Settings("pre_2020", last=PRE_LAST), Settings("block_8", block=8))


# ----------------------------------------------------------------------------------
# Measures
# ----------------------------------------------------------------------------------


def _log_change(x: np.ndarray, step: int = 1) -> np.ndarray:
    out = np.full_like(x, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[step:] = 100.0 * (np.log(x[step:]) - np.log(x[:-step]))
    return out


@dataclass(frozen=True)
class CountryData:
    """Quarterly measures of one country, aligned on ``quarters``."""

    g: np.ndarray
    measures: dict[str, np.ndarray]
    yy: dict[str, np.ndarray]                 # year-on-year changes, NaN where flagged
    levels: dict[str, np.ndarray]


def _flagged(flags: Sequence[str]) -> np.ndarray:
    return np.asarray([("b" in f) or ("u" in f) for f in flags])


def build_country(geo: str, quarters: Sequence[str],
                  base: Mapping[tuple[str, str], Mapping[str, float | None]],
                  extra: Mapping[tuple[str, str], Mapping[str, Any]]) -> CountryData:
    """Measures of section 3 of the protocol for one country."""
    def series(source: Mapping[tuple[str, str], Mapping[str, Any]], name: str) -> np.ndarray:
        values = [source.get((geo, q), {}).get(name) for q in quarters]
        return np.asarray([np.nan if v is None else float(v) for v in values])

    g = series(base, "gdp")
    u, p = series(base, "unemployment"), series(base, "participation")
    hours, emp = series(extra, "hours"), series(extra, "emp_a10")
    emp_dc, emp_nc = series(extra, "emp_dc"), series(extra, "emp_nc")
    h, e_dom = _log_change(hours), _log_change(emp)
    e_dom_pe, e_res = _log_change(emp_dc), _log_change(emp_nc)
    measures = {
        "du": np.diff(u, prepend=np.nan), "dp": np.diff(p, prepend=np.nan),
        "h": h, "e_dom": e_dom, "hpw": h - e_dom, "prod": g - h,
        "e_dom_pe": e_dom_pe, "e_res": e_res, "comm": e_dom_pe - e_res,
    }
    temp, employees = series(extra, "temp"), series(extra, "employees")
    perm = employees - temp
    flag_temp = _flagged([extra.get((geo, q), {}).get("temp_flag", "") or "" for q in quarters])
    flag_emp = _flagged([extra.get((geo, q), {}).get("employees_flag", "") or ""
                         for q in quarters])
    yy = {}
    for name, level, flags in (("temp", temp, flag_temp), ("perm", perm, flag_temp | flag_emp)):
        change = _log_change(level, 4)
        for t in range(len(quarters)):
            if t < 4 or flags[max(0, t - 4): t + 1].any():
                change[t] = np.nan
        yy[name] = change
    levels = {"emp_dc": emp_dc, "emp_nc": emp_nc,
              "temp_share": series(extra, "temp_share"), "pt_share": series(extra, "pt_share")}
    return CountryData(g, measures, yy, levels)


# ----------------------------------------------------------------------------------
# Elasticities with a synchronised block bootstrap
# ----------------------------------------------------------------------------------


def _touches(quarters: Sequence[str]) -> bool:
    return any(PANDEMIC[0] <= q <= PANDEMIC[1] for q in quarters)


@dataclass(frozen=True)
class Rows:
    """Regression rows of one series: y, X (constant first) and the quarter index."""

    y: np.ndarray
    x: np.ndarray
    q: np.ndarray


def lag_rows(y: np.ndarray, g: np.ndarray, quarters: Sequence[str], lags: int,
             first: str, last: str) -> Rows:
    """y[t] on a constant and g[t..t-lags]; window [t-lags, t] clear of 2020-2021."""
    ys, xs, qs = [], [], []
    for t in range(lags, len(quarters)):
        if not first <= quarters[t] <= last or _touches(quarters[t - lags: t + 1]):
            continue
        regs = [g[t - j] for j in range(lags + 1)]
        if math.isfinite(y[t]) and all(math.isfinite(v) for v in regs):
            ys.append(y[t])
            xs.append([1.0, *regs])
            qs.append(t)
    return Rows(np.asarray(ys), np.asarray(xs).reshape(-1, lags + 2), np.asarray(qs, dtype=int))


def yy_rows(y: np.ndarray, g: np.ndarray, quarters: Sequence[str], first: str,
            last: str) -> Rows:
    """Year-on-year change on the sum of four quarterly GDP growth rates."""
    ys, xs, qs = [], [], []
    for t in range(4, len(quarters)):
        if not first <= quarters[t] <= last or _touches(quarters[t - 4: t + 1]):
            continue
        g4 = float(np.sum(g[t - 3: t + 1]))
        if math.isfinite(y[t]) and math.isfinite(g4):
            ys.append(y[t])
            xs.append([1.0, g4])
            qs.append(t)
    return Rows(np.asarray(ys), np.asarray(xs).reshape(-1, 2), np.asarray(qs, dtype=int))


def block_counts(n_quarters: int, block: int, replicates: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Multiplicity of each quarter in circular block resamples: (replicates, n_quarters)."""
    n_blocks = math.ceil(n_quarters / block)
    starts = rng.integers(0, n_quarters, size=(replicates, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n_quarters
    idx = idx.reshape(replicates, -1)[:, :n_quarters]
    counts = np.zeros((replicates, n_quarters))
    np.add.at(counts, (np.repeat(np.arange(replicates), n_quarters), idx.ravel()), 1.0)
    return counts


def weighted_ols(rows: Rows, weights: np.ndarray) -> np.ndarray:
    """Coefficients for each row of ``weights`` (replicates, rows): (replicates, k)."""
    k = rows.x.shape[1]
    outer = np.einsum("ri,rj->rij", rows.x, rows.x).reshape(len(rows.y), k * k)
    xty = rows.x * rows.y[:, None]
    xtx = (weights @ outer).reshape(-1, k, k)
    rhs = weights @ xty
    out = np.full((weights.shape[0], k), np.nan)
    det_ok = np.abs(np.linalg.det(xtx)) > 1e-10
    if det_ok.any():
        out[det_ok] = np.linalg.solve(xtx[det_ok], rhs[det_ok][:, :, None])[:, :, 0]
    return out


@dataclass
class Estimate:
    """Point estimate and bootstrap draws of a slope sum (and of its components)."""

    point: float
    draws: np.ndarray
    n: int
    coef: np.ndarray = field(default_factory=lambda: np.zeros(0))
    coef_draws: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def interval(self) -> tuple[float, float]:
        finite = self.draws[np.isfinite(self.draws)]
        return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def estimate(rows: Rows, counts: np.ndarray, first_index: int) -> Estimate | None:
    """Sum of slope coefficients, at the point and in every bootstrap replicate."""
    if len(rows.y) < rows.x.shape[1] + 5:
        return None
    point = np.linalg.lstsq(rows.x, rows.y, rcond=None)[0]
    weights = counts[:, rows.q - first_index]
    coef = weighted_ols(rows, weights)
    return Estimate(float(point[1:].sum()), coef[:, 1:].sum(axis=1), len(rows.y), point, coef)


# ----------------------------------------------------------------------------------
# Running every estimate
# ----------------------------------------------------------------------------------


@dataclass
class Run:
    """All elasticities of one setting, sharing one set of bootstrap counts."""

    settings: Settings
    estimates: dict[tuple[str, str], Estimate]       # (geo, measure) -> estimate
    timing: dict[str, Estimate]                       # geo -> K = 4 du regression
    replicates: int

    def get(self, geo: str, measure: str) -> Estimate | None:
        if measure == "neg_du":
            e = self.estimates.get((geo, "du"))
            return None if e is None else Estimate(-e.point, -e.draws, e.n)
        return self.estimates.get((geo, measure))


def run(data: Mapping[str, CountryData], quarters: Sequence[str],
        settings: Settings = PRIMARY, replicates: int = REPLICATES,
        seed: int = SEED) -> Run:
    """Every elasticity of the protocol under one setting."""
    first_index = quarters.index(FIRST)
    last_index = quarters.index(settings.last)
    n_quarters = last_index - first_index + 1
    counts = block_counts(n_quarters, settings.block, replicates, np.random.default_rng(seed))
    estimates: dict[tuple[str, str], Estimate] = {}
    timing: dict[str, Estimate] = {}
    for geo, cd in data.items():
        for measure in MEASURES:
            rows = lag_rows(cd.measures[measure], cd.g, quarters, settings.lags, FIRST,
                            settings.last)
            est = estimate(rows, counts, first_index)
            if est is not None:
                estimates[(geo, measure)] = est
        for measure in YY_MEASURES:
            rows = yy_rows(cd.yy[measure], cd.g, quarters, FIRST, settings.last)
            est = estimate(rows, counts, first_index)
            if est is not None:
                estimates[(geo, measure)] = est
        rows = lag_rows(cd.measures["du"], cd.g, quarters, TIMING_LAGS, FIRST, settings.last)
        est = estimate(rows, counts, first_index)
        if est is not None:
            timing[geo] = est
    return Run(settings, estimates, timing, replicates)


# ----------------------------------------------------------------------------------
# Predictions
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    key: str
    hypothesis: str
    text: str
    statistic: str


PREDICTIONS: tuple[Prediction, ...] = (
    Prediction("P1.1", "H1", "LU: domestic minus resident employment elasticity > 0 "
               "(equivalently, the commuting margin responds to GDP; P1.2 is the same test)",
               "beta_e_dom_pe(LU) - beta_e_res(LU)"),
    Prediction("P1.3", "H1", "LU's domestic-resident gap exceeds the median gap of the other "
               "countries", "gap(LU) - median gap(others)"),
    Prediction("P2.1a", "H2", "NL hours-per-worker elasticity > ES", "beta_hpw(NL) - beta_hpw(ES)"),
    Prediction("P2.1b", "H2", "NL hours-per-worker elasticity > PT", "beta_hpw(NL) - beta_hpw(PT)"),
    Prediction("P2.2", "H2", "NL output-per-hour elasticity > ES", "beta_prod(NL) - beta_prod(ES)"),
    Prediction("P2.3a", "H2", "NL participation elasticity > ES", "beta_dp(NL) - beta_dp(ES)"),
    Prediction("P2.3b", "H2", "NL participation elasticity > PT", "beta_dp(NL) - beta_dp(PT)"),
    Prediction("P3.1a", "H3", "NL lagged share of the unemployment response > ES",
               "share(NL) - share(ES)"),
    Prediction("P3.1b", "H3", "NL lagged share of the unemployment response > PT",
               "share(NL) - share(PT)"),
    Prediction("P4.1 ES", "H4", "ES temporary employment more GDP-elastic than permanent",
               "beta_temp(ES) - beta_perm(ES)"),
    Prediction("P4.1 PT", "H4", "PT temporary employment more GDP-elastic than permanent",
               "beta_temp(PT) - beta_perm(PT)"),
    Prediction("P4.1 NL", "H4", "NL temporary employment more GDP-elastic than permanent",
               "beta_temp(NL) - beta_perm(NL)"),
    Prediction("P5.1", "H5", "ES domestic-employment elasticity > PT",
               "beta_e_dom(ES) - beta_e_dom(PT)"),
    Prediction("P5.2", "H5", "PT output-per-hour elasticity > ES", "beta_prod(PT) - beta_prod(ES)"),
)


@dataclass
class Evaluation:
    """Verdict of one prediction."""

    key: str
    hypothesis: str
    text: str
    statistic: str
    estimate: float
    low: float
    high: float
    sd: float
    p_value: float
    holm: float
    verdict: str
    note: str = ""


def two_sided_p(draws: np.ndarray) -> float:
    finite = draws[np.isfinite(draws)]
    n = len(finite)
    below = (1 + np.sum(finite <= 0)) / (n + 1)
    above = (1 + np.sum(finite >= 0)) / (n + 1)
    return float(min(1.0, 2 * min(below, above)))


def holm(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.zeros(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return [float(a) for a in adjusted]


def _diff(r: Run, a: tuple[str, str], b: tuple[str, str]) -> tuple[float, np.ndarray] | None:
    ea, eb = r.get(*a), r.get(*b)
    if ea is None or eb is None:
        return None
    return ea.point - eb.point, ea.draws - eb.draws


def lagged_share(e: Estimate) -> tuple[float, np.ndarray, bool]:
    """Share of the K = 4 sum from lags 1-4; determinable if the sum's interval excludes 0."""
    total, total_draws = e.coef[1:].sum(), e.coef_draws[:, 1:].sum(axis=1)
    lo, hi = np.percentile(total_draws[np.isfinite(total_draws)], [2.5, 97.5])
    share = e.coef[2:].sum() / total
    share_draws = e.coef_draws[:, 2:].sum(axis=1) / total_draws
    return float(share), share_draws, bool(lo > 0 or hi < 0)


def gap_statistic(r: Run) -> tuple[float, np.ndarray] | None:
    """gap(LU) - median of the other countries' gaps (domestic minus resident)."""
    gaps = {}
    for geo in EU27:
        d = _diff(r, (geo, "e_dom_pe"), (geo, "e_res"))
        if d is not None:
            gaps[geo] = d
    if "LU" not in gaps or len(gaps) < 3:
        return None
    others = [g for g in gaps if g != "LU"]
    point = gaps["LU"][0] - float(np.median([gaps[g][0] for g in others]))
    draws = gaps["LU"][1] - np.median(np.column_stack([gaps[g][1] for g in others]), axis=1)
    return point, draws


def statistic(r: Run, key: str) -> tuple[float, np.ndarray, str] | None:
    """Point estimate and bootstrap draws of a prediction's statistic, and any note."""
    pairs: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
        "P1.1": (("LU", "e_dom_pe"), ("LU", "e_res")),
        "P2.1a": (("NL", "hpw"), ("ES", "hpw")), "P2.1b": (("NL", "hpw"), ("PT", "hpw")),
        "P2.2": (("NL", "prod"), ("ES", "prod")),
        "P2.3a": (("NL", "dp"), ("ES", "dp")), "P2.3b": (("NL", "dp"), ("PT", "dp")),
        "P4.1 ES": (("ES", "temp"), ("ES", "perm")), "P4.1 PT": (("PT", "temp"), ("PT", "perm")),
        "P4.1 NL": (("NL", "temp"), ("NL", "perm")),
        "P5.1": (("ES", "e_dom"), ("PT", "e_dom")), "P5.2": (("PT", "prod"), ("ES", "prod")),
    }
    if key in pairs:
        d = _diff(r, *pairs[key])
        return None if d is None else (d[0], d[1], "")
    if key == "P1.3":
        d = gap_statistic(r)
        return None if d is None else (d[0], d[1], "")
    if key in ("P3.1a", "P3.1b"):
        other = "ES" if key == "P3.1a" else "PT"
        if "NL" not in r.timing or other not in r.timing:
            return None
        nl, ot = lagged_share(r.timing["NL"]), lagged_share(r.timing[other])
        if not (nl[2] and ot[2]):
            missing = [g for g, s in (("NL", nl), (other, ot)) if not s[2]]
            return (math.nan, np.full(r.replicates, np.nan),
                    f"not determinable: the K = 4 sum of {' and '.join(missing)} is not "
                    "distinguishable from zero")
        return nl[0] - ot[0], nl[1] - ot[1], ""
    raise KeyError(key)


def evaluate(r: Run) -> list[Evaluation]:
    """Every prediction, with Holm adjustment over the determinable ones."""
    raw = [(pred, statistic(r, pred.key)) for pred in PREDICTIONS]
    determinable: dict[int, float] = {}
    for i, (_, stat) in enumerate(raw):
        if stat is not None and math.isfinite(stat[0]):
            determinable[i] = two_sided_p(stat[1])
    adjusted = dict(zip(determinable, holm(list(determinable.values())), strict=True))
    out = []
    for i, (pred, stat) in enumerate(raw):
        if stat is None or not math.isfinite(stat[0]):
            note = "data not available" if stat is None else stat[2]
            out.append(Evaluation(pred.key, pred.hypothesis, pred.text, pred.statistic,
                                  math.nan, math.nan, math.nan, math.nan, math.nan, math.nan,
                                  "not determined", note))
            continue
        point, draws, note = stat
        finite = draws[np.isfinite(draws)]
        low, high = np.percentile(finite, [2.5, 97.5])
        p = two_sided_p(draws)
        h = adjusted[i]
        verdict = ("supported" if h < ALPHA and point > 0 else
                   "contradicted" if h < ALPHA and point < 0 else "not determined")
        out.append(Evaluation(pred.key, pred.hypothesis, pred.text, pred.statistic,
                              float(point), float(low), float(high),
                              float(np.std(finite, ddof=1)), p, h, verdict, note))
    return out


# ----------------------------------------------------------------------------------
# Do country differences remain under other measures? (protocol, section 6)
# ----------------------------------------------------------------------------------

PAIRS: tuple[tuple[str, str], ...] = (("ES", "PT"), ("ES", "NL"), ("ES", "LU"), ("PT", "NL"),
                                      ("PT", "LU"), ("NL", "LU"))
PAIR_MEASURES: tuple[str, ...] = ("neg_du", "e_res", "e_dom", "h")


@dataclass(frozen=True)
class PairResult:
    a: str
    b: str
    measure: str
    estimate: float
    low: float
    high: float
    status: str


def pair_results(r: Run) -> list[PairResult]:
    out = []
    for a, b in PAIRS:
        base = _diff(r, (a, "neg_du"), (b, "neg_du"))
        base_sign = 0
        if base is not None:
            lo, hi = np.percentile(base[1][np.isfinite(base[1])], [2.5, 97.5])
            base_sign = 1 if lo > 0 else (-1 if hi < 0 else 0)
        for measure in PAIR_MEASURES:
            d = _diff(r, (a, measure), (b, measure))
            if d is None:
                out.append(PairResult(a, b, measure, math.nan, math.nan, math.nan,
                                      "not available"))
                continue
            lo, hi = np.percentile(d[1][np.isfinite(d[1])], [2.5, 97.5])
            sign = 1 if lo > 0 else (-1 if hi < 0 else 0)
            if measure == "neg_du":
                status = "different" if sign else "not distinguishable"
            elif base_sign == 0:
                status = "no difference under unemployment to compare"
            elif sign == base_sign:
                status = "remains"
            elif sign == 0:
                status = "disappears"
            else:
                status = "reverses"
            out.append(PairResult(a, b, measure, float(d[0]), float(lo), float(hi), status))
    return out


# ----------------------------------------------------------------------------------
# Institutions across countries (H6)
# ----------------------------------------------------------------------------------

INDICATORS: tuple[tuple[str, str, int], ...] = (
    ("temp_share", "Temporary share, 2009-2019 (%)", -1),
    ("pt_share", "Part-time share, 2009-2019 (%)", 1),
    ("epl_r", "EPL regular contracts, 2008 (0-6)", 1),
    ("epl_t", "EPL temporary contracts, 2008 (0-6)", 1),
    ("beta_hpw", "Hours-per-worker elasticity", 1),
    ("commuting", "Domestic / resident employment, 2009-2019", 1),
)


def indicators(data: Mapping[str, CountryData], quarters: Sequence[str],
               epl: Mapping[str, Mapping[str, float]], r: Run) -> dict[str, dict[str, float]]:
    """Country-level indicators of H6 (missing where a source does not cover a country)."""
    window = [i for i, q in enumerate(quarters) if "2009-Q1" <= q <= "2019-Q4"]
    out: dict[str, dict[str, float]] = {}
    for geo, cd in data.items():
        row: dict[str, float] = {}
        for name in ("temp_share", "pt_share"):
            values = cd.levels[name][window]
            if np.isfinite(values).any():
                row[name] = float(np.nanmean(values))
        ratio = cd.levels["emp_dc"][window] / cd.levels["emp_nc"][window]
        if np.isfinite(ratio).any():
            row["commuting"] = float(np.nanmean(ratio))
        if geo in epl:
            row.update(epl[geo])
        hpw = r.get(geo, "hpw")
        if hpw is not None:
            row["beta_hpw"] = hpw.point
        out[geo] = row
    return out


def rank(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    for v in np.unique(values):
        tied = values == v
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


@dataclass(frozen=True)
class Correlation:
    variant: str
    indicator: str
    label: str
    predicted: int
    n: int
    rho: float
    p_value: float
    holm: float
    verdict: str


def correlations(theta: Mapping[str, float], ind: Mapping[str, Mapping[str, float]],
                 variant: str, exclude: Sequence[str] = (),
                 permutations: int = PERMUTATIONS, seed: int = SEED) -> list[Correlation]:
    """Spearman correlations of theta with each indicator, permutation p, Holm."""
    rng = np.random.default_rng(seed)
    raw = []
    for key, label, sign in INDICATORS:
        geos = [g for g in EU27 if g in theta and key in ind.get(g, {}) and g not in exclude]
        x = np.asarray([ind[g][key] for g in geos])
        y = np.asarray([theta[g] for g in geos])
        if len(geos) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
            rho, p = math.nan, 1.0              # no variation: nothing to correlate
        else:
            rho = spearman(x, y)
            perm = np.asarray([spearman(x, rng.permutation(y)) for _ in range(permutations)])
            p = float((1 + np.sum(np.abs(perm) >= abs(rho) - 1e-12)) / (permutations + 1))
        raw.append((key, label, sign, len(geos), rho, p))
    adjusted = holm([r[5] for r in raw])
    out = []
    for (key, label, sign, n, rho, p), h in zip(raw, adjusted, strict=True):
        verdict = ("supported" if h < ALPHA and np.sign(rho) == sign else
                   "contradicted" if h < ALPHA else "not determined")
        out.append(Correlation(variant, key, label, sign, n, rho, p, h, verdict))
    return out


# ----------------------------------------------------------------------------------
# Crisis decomposition (descriptive)
# ----------------------------------------------------------------------------------

CRISIS = ("2008-Q2", "2013-Q1")
CRISIS_MEASURES: tuple[str, ...] = ("prod", "hpw", "e_dom", "du", "dp")


def crisis(data: Mapping[str, CountryData], quarters: Sequence[str],
           geos: Sequence[str] = ("ES", "PT")) -> list[dict[str, Any]]:
    """Cumulative changes over the crisis window, from the first available quarter."""
    out = []
    for geo in geos:
        cd = data[geo]
        idx = [i for i, q in enumerate(quarters) if CRISIS[0] <= q <= CRISIS[1]]
        row: dict[str, Any] = {"geo": geo, "g": float(np.nansum(cd.g[idx]))}
        for m in CRISIS_MEASURES:
            values = cd.measures[m][idx]
            have = [quarters[i] for i, v in zip(idx, values, strict=True) if math.isfinite(v)]
            # An unpublished series is missing, not a zero change.
            row[m] = float(np.nansum(values)) if have else math.nan
            row[f"{m}_from"] = have[0] if have else ""
        out.append(row)
    return out


def quarters_for(first: str = DATA_START, last: str = LAST) -> list[str]:
    return quarter_sequence(first, last)
