"""Reading the national YRBS files, the grid of specifications, and the estimators.

Everything here is plain numpy. The survey design (strata, primary sampling units, weights) is handled
by Taylor linearisation of weighted least squares: each coefficient is written as a sum of
per-student influence values, summed to PSU totals and centred within strata. The same PSU totals give
the design-based standard error of every specification and, stacked, the joint distribution of all
specifications of one survey wave, which the multiplier bootstrap in ``joint_null`` draws from.
"""

import itertools
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("YRBS_DATA", HERE / "data"))

BASE = "https://www.cdc.gov/yrbs/files"
FILES = {
    2017: ("2017/XXH2017_YRBS_Data.dat", "2017/2017XXH_SAS_Input_Program.sas"),
    2019: ("2019/XXH2019_YRBS_Data.dat", "2019/2019XXH-SAS-Input-Program.sas"),
    2021: ("2021/XXH2021_YRBS_Data.dat", "2021/2021XXH-SAS-Input-Program.sas"),
    2023: ("2023/XXH2023_YRBS_Data.dat", "2023/2023XXH-SAS-Input-Program.sas"),
}
WAVES = tuple(FILES)


# --------------------------------------------------------------------------------------------------
# Fixed-width layout from the SAS input program
# --------------------------------------------------------------------------------------------------


def parse_sas_input(text):
    """Variable name -> (first column, last column), 1-based and inclusive, from a SAS INPUT block.

    Handles the two layouts CDC has used: column input (``Q1 $ 17-17``, ``weight 431-440``) and
    pointer input (``@17 Q1 $1.``, ``@431 weight 10.4``).
    """
    m = re.search(r"\binput\b(.*?);", text, flags=re.IGNORECASE | re.DOTALL)
    if m is None:
        raise ValueError("no INPUT statement found")
    block = m.group(1)
    layout = {}
    for name, a, b in re.findall(r"\b([A-Za-z_]\w*)\s+\$?\s*(\d+)\s*-\s*(\d+)", block):
        layout[name.lower()] = (int(a), int(b))
    for a, name, w in re.findall(r"@\s*(\d+)\s+([A-Za-z_]\w*)\s+\$?\s*(\d+)\.\d*", block):
        layout[name.lower()] = (int(a), int(a) + int(w) - 1)
    if not layout:
        raise ValueError("INPUT statement has no recognisable variables")
    return layout


def read_fixed(path, layout, names):
    """Read the named variables of a fixed-width file as float arrays; blanks and '.' become NaN."""
    names = [n.lower() for n in names]
    cols = {n: [] for n in names}
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("latin-1").rstrip("\r\n")
            for n in names:
                a, b = layout[n]
                cols[n].append(line[a - 1 : b].strip())
    out = {}
    for n, vals in cols.items():
        arr = np.full(len(vals), np.nan)
        for i, v in enumerate(vals):
            if v not in ("", "."):
                try:
                    arr[i] = float(v)
                except ValueError:
                    arr[i] = np.nan
        out[n] = arr
    return out


# --------------------------------------------------------------------------------------------------
# Items by wave. Question numbers change from wave to wave; the map below is taken from Appendix A of
# each year's Data User's Guide. tests/test_data.py checks it against the files when they are present.
# --------------------------------------------------------------------------------------------------

_COMMON = {"sex": "q2", "grade": "q3", "raceeth": "raceeth", "weight": "weight", "stratum": "stratum", "psu": "psu"}
ITEMS = {
    2017: _COMMON
    | {"sexid": "q67", "bullied": "q23", "ebullied": "q24", "sad": "q25", "consider": "q26", "plan": "q27"}
    | {"attempt": "q28", "active": "q79", "tv": "q80", "games": "q81", "sleep": "q88"},
    2019: _COMMON
    | {"sexid": "q66", "bullied": "q23", "ebullied": "q24", "sad": "q25", "consider": "q26", "plan": "q27"}
    | {"attempt": "q28", "active": "q78", "tv": "q79", "games": "q80", "sleep": "q88"},
    2021: _COMMON
    | {"sexid": "q65", "bullied": "q23", "ebullied": "q24", "sad": "q25", "consider": "q26", "plan": "q27"}
    | {"attempt": "q28", "active": "q77", "screen": "q78", "poor_mh": "q85", "sleep": "q86"},
    2023: _COMMON
    | {"sexid": "q64", "bullied": "q24", "ebullied": "q25", "sad": "q26", "consider": "q27", "plan": "q28"}
    | {"attempt": "q29", "active": "q76", "social": "q80", "poor_mh": "q84", "sleep": "q85"},
}
# Answers to the sexual-identity item that count: 2017 and 2019 offer four options (heterosexual, gay
# or lesbian, bisexual, not sure); 2021 and 2023 add "some other way" and "questioning" (codes 4, 5)
# and "I do not know what this question is asking" (6), which is treated as missing.
SEXID_VALID = {2017: (1, 2, 3, 4), 2019: (1, 2, 3, 4), 2021: (1, 2, 3, 4, 5), 2023: (1, 2, 3, 4, 5)}


def load_wave(wave, data=None):
    """Recoded analysis variables of one wave, read with the column layout of its own SAS program."""
    data = Path(data) if data is not None else DATA
    dat, sas = (data / rel.split("/")[1] for rel in FILES[wave])
    layout = parse_sas_input(sas.read_text(encoding="latin-1"))
    names = sorted(set(ITEMS[wave].values()))
    missing = [n for n in names if n not in layout]
    if missing:
        raise KeyError(f"{wave}: variables not in the SAS layout: {missing}")
    return recode(read_fixed(dat, layout, names), wave)


# Response codes of the raw questions, as CDC numbers the options (1 = A, 2 = B, ...).
TV_HOURS = {1: 0.0, 2: 0.5, 3: 1.0, 4: 2.0, 5: 3.0, 6: 4.0, 7: 5.0}  # "I do not ...", "<1 hour", 1..4, "5+"
SCREEN_HOURS = {1: 0.5, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0, 6: 5.0}  # "<1 hour", 1..4, "5+" (2021, 2023)
SOCIAL_LEVELS = {k: k - 1.0 for k in range(1, 9)}  # 0 = does not use ... 7 = more than once an hour
SOCIAL_LABELS = (
    "Does not use",
    "A few times a month",
    "About once a week",
    "A few times a week",
    "About once a day",
    "Several times a day",
    "About once an hour",
    "More than once an hour",
)
SLEEP_HOURS = {1: 4.0, 2: 5.0, 3: 6.0, 4: 7.0, 5: 8.0, 6: 9.0, 7: 10.0}


def _map(arr, table):
    out = np.full(arr.shape, np.nan)
    for k, v in table.items():
        out[arr == k] = v
    return out


def recode(raw, wave):
    """Analysis variables of one wave from its raw columns (dict of float arrays)."""
    it = ITEMS[wave]
    r = {k: raw[v] for k, v in it.items()}
    n = len(next(iter(r.values())))
    d = {"wave": np.full(n, wave)}

    def yes(x):
        return np.where(np.isnan(x), np.nan, (x == 1).astype(float))

    d["sad"] = yes(r["sad"])
    d["consider"] = yes(r["consider"])
    d["plan"] = yes(r["plan"])
    d["attempt"] = np.where(np.isnan(r["attempt"]), np.nan, (r["attempt"] >= 2).astype(float))
    if "poor_mh" in r:
        d["poor_mh"] = np.where(np.isnan(r["poor_mh"]), np.nan, (r["poor_mh"] >= 4).astype(float))
    for k in ("tv", "games"):
        if k in r:
            d[k] = _map(r[k], TV_HOURS)
    if "screen" in r:
        d["screen"] = _map(r["screen"], SCREEN_HOURS)
    if "social" in r:
        d["social"] = _map(r["social"], SOCIAL_LEVELS)

    d["female"] = np.where(np.isin(r["sex"], (1, 2)), (r["sex"] == 1).astype(float), np.nan)
    d["grade"] = np.where(np.isin(r["grade"], (1, 2, 3, 4)), r["grade"], np.nan)
    d["raceeth"] = np.where(np.isin(r["raceeth"], range(1, 9)), r["raceeth"], np.nan)
    # Sexual identity: heterosexual (1) against every other identity; "I do not know what this
    # question is asking" (2023 only) is treated as missing.
    sid = r["sexid"]
    valid = np.isin(sid, SEXID_VALID[wave])
    d["minority"] = np.where(valid, (sid != 1).astype(float), np.nan)
    d["sleep"] = _map(r["sleep"], SLEEP_HOURS)
    d["bullied"] = yes(r["bullied"])
    d["ebullied"] = yes(r["ebullied"])
    d["active"] = np.where(np.isin(r["active"], range(1, 9)), r["active"] - 1.0, np.nan)
    d["weight"] = r["weight"]
    d["stratum"] = r["stratum"]
    d["psu"] = r["psu"]
    return d


# --------------------------------------------------------------------------------------------------
# The grid of specifications
# --------------------------------------------------------------------------------------------------

MEASURES = {  # wave -> the technology items asked that year; each wave asks a different question
    2017: ("tv", "games"),
    2019: ("tv", "games"),
    2021: ("screen",),
    2023: ("social",),
}
OUTCOMES = ("sad", "consider", "plan", "attempt", "poor_mh")
CODINGS = ("linear", "high_vs_rest", "high_vs_light", "top_vs_rest", "top_vs_light")
CONTROLS = ("none", "demographic", "demographic_identity", "pathways")
SAMPLES = ("all", "girls", "boys")
WEIGHTINGS = ("survey", "unweighted")
FORKS = ("measure", "coding", "outcome", "controls", "sample", "weighting")


@dataclass(frozen=True)
class Spec:
    wave: int
    measure: str
    coding: str
    outcome: str
    controls: str
    sample: str
    weighting: str

    def key(self):
        return "|".join(str(getattr(self, f)) for f in ("wave",) + FORKS)

    @property
    def wave_measure(self):
        return f"{self.measure} {self.wave}"


def grid(available):
    """All specifications. ``available`` maps wave -> set of outcome names present that year."""
    specs = []
    for wave in WAVES:
        for measure, coding, outcome, controls, sample, weighting in itertools.product(
            MEASURES[wave], CODINGS, OUTCOMES, CONTROLS, SAMPLES, WEIGHTINGS
        ):
            if outcome not in available[wave]:
                continue
            specs.append(Spec(wave, measure, coding, outcome, controls, sample, weighting))
    return specs


def exposure(d, measure, coding):
    """Exposure vector and the mask of students who enter the comparison.

    Hours items: high = 3 hours or more, top = 5 hours or more; the light reference group is 1 to 2
    hours. Social media: high = several times a day or more (CDC's "frequent use"), top = about
    once an hour or more; the light reference group is a few times a week to about once a day.
    ``*_vs_rest`` compares against everyone below the threshold, ``*_vs_light`` only against the
    light group.
    """
    x = d[measure]
    ok = ~np.isnan(x)
    if coding == "linear":
        return x, ok
    if measure == "social":
        thr = 5.0 if coding.startswith("high") else 6.0
        light = (x >= 3.0) & (x <= 4.0)
    else:
        thr = 3.0 if coding.startswith("high") else 5.0
        light = (x >= 1.0) & (x <= 2.0)
    hi = x >= thr
    xb = np.where(ok, hi.astype(float), np.nan)
    if coding.endswith("vs_rest"):
        return xb, ok
    return xb, ok & (hi | light)


def design_matrix(d, spec, mask):
    """Intercept, exposure and the control columns of a specification, for rows in ``mask``."""
    cols = []
    names = []
    if spec.controls != "none":
        if spec.sample == "all":
            cols.append(d["female"])
            names.append("female")
        for g in (2, 3, 4):
            cols.append((d["grade"] == g).astype(float))
            names.append(f"grade{g}")
        # Race and ethnicity: Black, Hispanic or Latino (any race), Asian, multiple races
        # (non-Hispanic), other; White is the reference.
        re_ = d["raceeth"]
        for lab, codes in (("black", (3,)), ("hispanic", (6, 7)), ("asian", (2,)), ("multi", (8,)), ("other", (1, 4))):
            cols.append(np.isin(re_, codes).astype(float))
            names.append(lab)
    if spec.controls in ("demographic_identity", "pathways"):
        cols.append(d["minority"])
        names.append("minority")
    if spec.controls == "pathways":
        for k in ("sleep", "bullied", "ebullied", "active"):
            cols.append(d[k])
            names.append(k)
    need = [d["grade"], d["raceeth"], d["female"]] if spec.controls != "none" else []
    for c in cols:
        need.append(c)
    m = mask.copy()
    for c in need:
        m &= ~np.isnan(c)
    X = np.column_stack([np.ones(int(m.sum()))] + [c[m] for c in cols]) if cols else np.ones((int(m.sum()), 1))
    return X, names, m


# --------------------------------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------------------------------


@dataclass
class Fit:
    b: float  # risk difference per unit of exposure (linear probability model)
    se_design: float
    se_naive: float
    beta: float  # standardised coefficient, b * sd(x) / sd(y)
    n: int
    p_ref: float  # adjusted risk at exposure 0, for binary exposures
    rr: float  # (p_ref + b) / p_ref for binary exposures, NaN otherwise
    z: np.ndarray  # centred, scaled PSU totals of the influence of b (length = PSUs of the wave)
    sx: float
    sy: float


def psu_index(stratum, psu):
    """Integer id of each (stratum, PSU) pair, the stratum of each id, and the stratum sizes."""
    pairs = np.column_stack([stratum, psu])
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    strata_of = uniq[:, 0]
    _, s_inv, s_cnt = np.unique(strata_of, return_inverse=True, return_counts=True)
    return inv.ravel(), s_inv, s_cnt


def wls(X, y, w, cid, s_of, s_cnt, n_psu, j=1):
    """Weighted least squares, coefficient ``j``, with design-based and naive standard errors.

    Design-based variance (Taylor linearisation, PSUs sampled with replacement within strata):
    V = sum_h n_h/(n_h - 1) sum_i (z_hi - mean_h z)^2, where z_hi is the PSU total of the influence
    values u_k = [(X'WX)^-1 x_k]_j w_k e_k. A stratum with a single PSU is centred on the grand mean
    of PSU totals (the "adjust" convention). ``n_psu`` is the number of PSUs of the whole wave, so a
    subgroup's estimate keeps the zero totals of PSUs where it has no members (domain estimation).
    """
    XtW = X.T * w
    A = XtW @ X
    Ainv = np.linalg.inv(A)
    coef = Ainv @ (XtW @ y)
    e = y - X @ coef
    a = X @ Ainv[j]
    u = a * w * e
    tot = np.bincount(cid, weights=u, minlength=n_psu)
    n_strata = len(s_cnt)
    s_mean = np.bincount(s_of, weights=tot, minlength=n_strata) / s_cnt
    centre = np.where(s_cnt[s_of] > 1, s_mean[s_of], tot.mean())
    scale = np.where(s_cnt[s_of] > 1, np.sqrt(s_cnt[s_of] / np.maximum(s_cnt[s_of] - 1, 1)), 1.0)
    z = (tot - centre) * scale
    var_design = float(z @ z)
    wn = w / w.mean()
    k = X.shape[1]
    s2 = float((wn * e * e).sum() / (wn.sum() - k))
    var_naive = s2 * float(np.linalg.inv((X.T * wn) @ X)[j, j])
    return coef, np.sqrt(var_design), np.sqrt(var_naive), z


def wmean(v, w):
    return float((v * w).sum() / w.sum())


def wsd(v, w):
    m = wmean(v, w)
    return float(np.sqrt(((v - m) ** 2 * w).sum() / w.sum()))


def fit_spec(d, spec, design):
    """Estimate one specification on one wave's recoded data ``d``; ``design`` from psu_index."""
    cid_all, s_of, s_cnt = design
    y_all = d[spec.outcome]
    x_all, mask = exposure(d, spec.measure, spec.coding)
    mask = mask & ~np.isnan(y_all) & ~np.isnan(d["weight"])
    if spec.sample == "girls":
        mask &= d["female"] == 1
    elif spec.sample == "boys":
        mask &= d["female"] == 0
    Xc, _, m = design_matrix(d, spec, mask)
    x = x_all[m]
    y = y_all[m]
    w = d["weight"][m] if spec.weighting == "survey" else np.ones(int(m.sum()))
    X = np.column_stack([Xc[:, :1], x, Xc[:, 1:]])
    coef, se_d, se_n, z = wls(X, y, w, cid_all[m], s_of, s_cnt, len(s_of), j=1)
    b = float(coef[1])
    sx, sy = wsd(x, w), wsd(y, w)
    binary = spec.coding != "linear"
    if binary:
        p_ref = wmean(y, w) - b * wmean(x, w)
        rr = (p_ref + b) / p_ref if p_ref > 0 else np.nan
    else:
        p_ref, rr = np.nan, np.nan
    return Fit(b, se_d, se_n, b * sx / sy, int(m.sum()), p_ref, rr, z, sx, sy)


# --------------------------------------------------------------------------------------------------
# Joint inference and the decomposition of the curve
# --------------------------------------------------------------------------------------------------


def joint_null(z_by_wave, se, scale, draws, rng):
    """Draws of every specification's estimate under the null of no association anywhere.

    ``z_by_wave``: wave -> (indices of its specifications, matrix of their PSU totals, specs x PSUs).
    A draw multiplies each PSU's centred total by an independent standard normal; the estimates of
    one wave share the multipliers, so their correlation (same students, same PSUs) is kept, and
    waves are independent samples. Returns (draws x specs) matrices of standardised estimates and of
    t statistics. This is the multiplier (wild) bootstrap of the linearised estimators; with an
    exactly linear statistic its covariance equals the design-based covariance.
    """
    n_specs = len(se)
    bstar = np.zeros((draws, n_specs))
    for _wave, (idx, Z) in z_by_wave.items():
        G = rng.standard_normal((draws, Z.shape[1]))
        bstar[:, idx] = G @ Z.T
    return bstar * scale, bstar / se


def shapley_r2(y, factors):
    """Share of the variance of ``y`` attributed to each categorical factor (Shapley values of R^2).

    ``factors``: name -> array of labels. R^2 of every subset of factors is computed with dummy
    regressions; a factor's value is its average marginal R^2 over all orderings. The values sum to
    the R^2 of the full model.
    """
    names = list(factors)
    k = len(names)
    dummies = {}
    for nme in names:
        labs = np.asarray(factors[nme])
        levels = np.unique(labs)
        dummies[nme] = np.column_stack([(labs == lv).astype(float) for lv in levels[1:]])
    yc = y - y.mean()
    sst = float(yc @ yc)
    cache = {}

    def r2(subset):
        if subset in cache:
            return cache[subset]
        if not subset:
            cache[subset] = 0.0
            return 0.0
        X = np.column_stack([np.ones(len(y))] + [dummies[names[i]] for i in subset])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ coef
        cache[subset] = 1.0 - float(res @ res) / sst
        return cache[subset]

    from math import factorial

    phi = {}
    for i in range(k):
        others = [j for j in range(k) if j != i]
        val = 0.0
        for size in range(k):
            for comb in itertools.combinations(others, size):
                wgt = factorial(size) * factorial(k - size - 1) / factorial(k)
                with_i = tuple(sorted(comb + (i,)))
                val += wgt * (r2(with_i) - r2(tuple(sorted(comb))))
        phi[names[i]] = val
    return phi, r2(tuple(range(k)))


def e_value(rr):
    """VanderWeele and Ding's E-value for a risk ratio (point estimate); 1 for rr <= 1 is mirrored."""
    rr = np.asarray(rr, dtype=float)
    r = np.where(rr >= 1, rr, 1.0 / rr)
    return r + np.sqrt(r * (r - 1.0))


def besd(r):
    """Rosenthal and Rubin's binomial effect size display: 50% +/- r/2 'success' rates."""
    return 0.5 - r / 2.0, 0.5 + r / 2.0


def weighted_prevalence(y, w, cid, s_of, s_cnt):
    """Weighted mean of a 0/1 variable and its design-based standard error (ratio linearisation)."""
    p = wmean(y, w)
    u = w * (y - p) / w.sum()
    tot = np.bincount(cid, weights=u, minlength=len(s_of))
    n_strata = len(s_cnt)
    s_mean = np.bincount(s_of, weights=tot, minlength=n_strata) / s_cnt
    centre = np.where(s_cnt[s_of] > 1, s_mean[s_of], tot.mean())
    scale = np.where(s_cnt[s_of] > 1, np.sqrt(s_cnt[s_of] / np.maximum(s_cnt[s_of] - 1, 1)), 1.0)
    zz = (tot - centre) * scale
    return p, float(np.sqrt(zz @ zz))
