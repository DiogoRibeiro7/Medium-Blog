"""Part 3 of the regression-to-the-mean series: real data.

Every number quoted in the article is printed by this script. Run:

    python real_data_numbers.py > output.txt

Data downloaded on first run, checked against SHA-256 digests and cached in data/:
  - NHANESraw from the R package NHANES 2.1.4 (Pruim), read from a pinned commit of the CRAN mirror on
    GitHub: the US National Health and Nutrition Examination Survey, 2009-2010 and 2011-2012, with three
    consecutive systolic blood-pressure readings per examined participant, the examination weights and the
    design variables. Only the columns in NHANES_COLUMNS are kept after reading; nothing else is used.
  - Lahman::Batting, Lahman::People, Lahman::AwardsPlayers and Lahman::Teams (Lahman Baseball Database,
    seasons to 2025) from a pinned commit of the Rdatasets repository.

Sections
1. NHANES: the sample, the three readings and the population drift between them.
2. Selecting on the first reading: drift, regression and what is left, with design-based standard errors.
3. Why the first reading is different: variances, covariances, noise by level and by age.
4. The next visit: a scenario with day-to-day variation taken from an outside estimate.
5. Held-out check for blood pressure: the line from 2009-2010 predicts the selections of 2011-2012.
6. Batting: the reliability of a season computed from the sampling model, component by component.
7. Rookies and Rookie of the Year winners: the sophomore slump against its prediction.
8. Teams: run differential and the Pythagorean residual.
9. Held-out check for baseball, and the predicted-against-observed summary.
"""

from __future__ import annotations

import hashlib
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

SEED = 1886
HERE = Path(__file__).parent
DATA = HERE / "data"
RD = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/1dcc2bf5f955cc1224a3e1307256e1fe86b68dae/csv/"
NH = "https://raw.githubusercontent.com/cran/NHANES/105a4cfb64514937462db86293f0dac78860f004/data/"
FILES = {
    "NHANESraw.rda": NH + "NHANESraw.rda",
    "Batting.csv": RD + "Lahman/Batting.csv",
    "People.csv": RD + "Lahman/People.csv",
    "AwardsPlayers.csv": RD + "Lahman/AwardsPlayers.csv",
    "Teams.csv": RD + "Lahman/Teams.csv",
}
DIGESTS = {
    "NHANESraw.rda": "f93a21ad0e492f5bc492ac5b96ac81cdb9171b190ac149981f6ce8f9ebdea2e4",
    "Batting.csv": "a4083e2d4bacf411c352686fd7c0696eaa4bc8719b718db5e9c45ac90fd95b35",
    "People.csv": "33aacfdea45b8e19166e4d725191ac6b874b334260584d2999107b42cc746a81",
    "AwardsPlayers.csv": "c3e767e56047c24b3130c2be01274eedcdc55dee3ced287fccddc4184ff4ac1b",
    "Teams.csv": "9d1c390df5d2a422175f6bcfb56d099cfb9c5805586ba5e3f8c895a9952423ad",
}
NHANES_COLUMNS = ["SurveyYr", "Gender", "Age", "BPSys1", "BPSys2", "BPSys3", "BPSysAve",
                  "WTMEC2YR", "SDMVPSU", "SDMVSTRA"]

# Muntner et al. (2011), NHANES III: mean across participants of the standard deviation of systolic pressure
# over three visits, each visit summarised by the mean of its second and third readings.
MUNTNER_MEAN_SD = 7.7
SHORT_SEASON = 2020          # 60-game season: no regular reaches 300 plate appearances
MIN_PA = 300                 # a regular season for a batter
ROOKIE_AB = 130              # a player is a rookie until his career at-bats before the season exceed this
FIRST, LAST = 1955, 2024     # first seasons of the batting pairs (sacrifice flies recorded from 1954)


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fetch(name: str) -> Path:
    path = DATA / name
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FILES[name], timeout=300) as r:
            path.write_bytes(r.read())
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != DIGESTS[name]:
        raise RuntimeError(f"{name}: SHA-256 {got} does not match the pinned digest {DIGESTS[name]}")
    return path


# ---------------------------------------------------------------------------
# Survey tools: weighted moments and the delete-one-PSU jackknife
# ---------------------------------------------------------------------------

def wmean(x, w):
    return float(np.sum(w * x) / np.sum(w))


def wcov(x, y, w):
    mx, my = wmean(x, w), wmean(y, w)
    return float(np.sum(w * (x - mx) * (y - my)) / np.sum(w))


def wquantile(x, w, q):
    """Smallest x at which the weighted cumulative share reaches q."""
    order = np.argsort(x, kind="stable")
    cw = np.cumsum(w[order]) / np.sum(w)
    return float(x[order][min(np.searchsorted(cw, q), len(x) - 1)])


def jk_replicates(strata, psu, w):
    """Replicate weights for the stratified jackknife (JKn): drop one PSU, reweight its stratum."""
    reps = []
    for h in np.unique(strata):
        inside = strata == h
        units = np.unique(psu[inside])
        nh = len(units)
        for j in units:
            wr = w.copy()
            wr[inside] *= nh / (nh - 1)
            wr[inside & (psu == j)] = 0.0
            reps.append(((nh - 1) / nh, wr))
    return reps


def jk(stat, w, reps):
    """Estimate and jackknife standard error of stat(weights); stat may return an array."""
    est = np.asarray(stat(w), dtype=float)
    var = sum(f * (np.asarray(stat(wr), dtype=float) - est) ** 2 for f, wr in reps)
    return est, np.sqrt(var)


# ---------------------------------------------------------------------------
# Blood pressure
# ---------------------------------------------------------------------------

def load_nhanes() -> pd.DataFrame:
    import pyreadr  # reads the .rda file without R

    raw = pyreadr.read_r(str(fetch("NHANESraw.rda")))["NHANESraw"]
    d = raw[NHANES_COLUMNS].copy()
    del raw
    d["SurveyYr"] = d["SurveyYr"].astype(str)
    d["Gender"] = d["Gender"].astype(str)
    return d


def nhanes_adults(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    adults = d[d.Age >= 18]
    full = adults.dropna(subset=["BPSys1", "BPSys2", "BPSys3"]).copy()
    full["m23"] = (full.BPSys2 + full.BPSys3) / 2
    full["w4"] = full.WTMEC2YR / 2          # four-year examination weight for the two cycles combined
    return adults, full.reset_index(drop=True)


def selection_decomposition(x1, m23, w, sel):
    """Fall of a group selected on x1, split into population drift, predicted regression and a residual."""
    mu1, mu23 = wmean(x1, w), wmean(m23, w)
    b = wcov(x1, m23, w) / wcov(x1, x1, w)
    xs, ys = wmean(x1[sel], w[sel]), wmean(m23[sel], w[sel])
    pred = mu23 + b * (xs - mu1)
    return {
        "share": float(w[sel].sum() / w.sum()),
        "first": xs, "later": ys, "fall": xs - ys,
        "drift": mu1 - mu23, "regression": (1 - b) * (xs - mu1),
        "predicted": pred, "residual": ys - pred,
        "kept": (ys - mu23) / (xs - mu1), "slope": b,
    }


def visit_scenario(x1, m23, w, x2, x3, sigma_visit, cut):
    """Next-visit mean of readings 2-3 under a normal model, for groups selected on x1 or on m23.

    sigma_visit: standard deviation of a person's visit mean (of readings 2 and 3) from visit to visit.
    Var(U), the variance of usual pressure, is Var(m23) - sigma_visit^2; a measurement in a sitting covaries
    with the next visit only through U.
    """
    mu1, mu23 = wmean(x1, w), wmean(m23, w)
    v1, v23 = wcov(x1, x1, w), wcov(m23, m23, w)
    vu = v23 - sigma_visit ** 2
    out = {}
    for label, x, vx, mx in (("first", x1, v1, mu1), ("mean23", m23, v23, mu23)):
        sel = x >= cut
        b = vu / vx
        cm = mu23 + b * (x - mx)
        csd = math.sqrt(v23 - vu ** 2 / vx)
        p_below = stats.norm.cdf((cut - cm) / csd)
        out[label] = {
            "slope": b,
            "mean_next": wmean(cm[sel], w[sel]),
            "share_below": wmean(p_below[sel], w[sel]),
            "selected_mean": wmean(x[sel], w[sel]),
        }
    return out


def usual_prevalence(df, sigma_visit, cut):
    """Share with usual pressure >= cut by rescaling deviations within age-sex groups (NRC 1986 method)."""
    w = df.w4.to_numpy()
    band = np.clip((df.Age.to_numpy() // 10) * 10, 20, 70)   # 18-29, 30-39, ..., 60-69, 70-80
    group = df.Gender.to_numpy() + band.astype(str)
    adj = np.empty(len(df))
    m = df.m23.to_numpy()
    for g in np.unique(group):
        k = group == g
        mu, v = wmean(m[k], w[k]), wcov(m[k], m[k], w[k])
        ratio = math.sqrt(max(v - sigma_visit ** 2, 0.0) / v)
        adj[k] = mu + ratio * (m[k] - mu)
    return wmean((adj >= cut).astype(float), w)


def bp_conditional_curve(x1, m23, w, cut, sigma_visit, grid):
    """P(m23 >= cut | x1 = v): within the sitting (normal model) and at the next visit."""
    mu1, mu23 = wmean(x1, w), wmean(m23, w)
    v1, v23, c = wcov(x1, x1, w), wcov(m23, m23, w), wcov(x1, m23, w)
    same_sd = math.sqrt(v23 - c ** 2 / v1)
    same = stats.norm.sf((cut - (mu23 + c / v1 * (grid - mu1))) / same_sd)
    vu = v23 - sigma_visit ** 2
    next_sd = math.sqrt(v23 - vu ** 2 / v1)
    nxt = stats.norm.sf((cut - (mu23 + vu / v1 * (grid - mu1))) / next_sd)
    return same, nxt


# ---------------------------------------------------------------------------
# Baseball
# ---------------------------------------------------------------------------

CATS = ["OUT", "X1B", "X2B", "X3B", "HR", "BB", "HBP", "SF", "SH"]
RATES = {"BABIP": "BIP", "K": "PA", "BBr": "PA", "HRr": "PA"}
LABELS = {"OPS": "OPS", "BABIP": "BABIP", "K": "strikeout rate", "BBr": "walk rate", "HRr": "home-run rate"}


def ops_from_counts(c: dict) -> np.ndarray:
    ab = c["OUT"] + c["X1B"] + c["X2B"] + c["X3B"] + c["HR"]
    h = c["X1B"] + c["X2B"] + c["X3B"] + c["HR"]
    with np.errstate(invalid="ignore", divide="ignore"):   # seasons with no at-bats give NaN
        obp = (h + c["BB"] + c["HBP"]) / (ab + c["BB"] + c["HBP"] + c["SF"])
        slg = (c["X1B"] + 2 * c["X2B"] + 3 * c["X3B"] + 4 * c["HR"]) / ab
    return obp + slg


def load_batting() -> pd.DataFrame:
    bat = pd.read_csv(fetch("Batting.csv"))
    people = pd.read_csv(fetch("People.csv"), usecols=["playerID", "birthYear", "birthMonth"])
    num = ["AB", "H", "X2B", "X3B", "HR", "BB", "SO", "HBP", "SH", "SF"]
    b = bat.groupby(["playerID", "yearID"], as_index=False)[num].sum(min_count=1).fillna(0)
    b["PA"] = b.AB + b.BB + b.HBP + b.SF + b.SH
    b["X1B"] = b.H - b.X2B - b.X3B - b.HR
    b["OUT"] = b.AB - b.H
    b["OPS"] = ops_from_counts({k: b[k].to_numpy(float) for k in CATS})
    b["BIP"] = b.AB - b.SO - b.HR + b.SF
    with np.errstate(invalid="ignore", divide="ignore"):
        b["BABIP"] = (b.H - b.HR) / b.BIP
        b["K"] = b.SO / b.PA
        b["BBr"] = b.BB / b.PA
        b["HRr"] = b.HR / b.PA
    b = b.sort_values(["playerID", "yearID"])
    b["AB_before"] = b.groupby("playerID").AB.cumsum() - b.AB
    b = b.merge(people, on="playerID", how="left")
    # Age on 30 June of the season, the usual baseball convention.
    b["age"] = b.yearID - b.birthYear - (b.birthMonth >= 7).astype(int)
    return b.reset_index(drop=True)


def season_deviations(b: pd.DataFrame) -> pd.DataFrame:
    """Each statistic as a deviation from the mean of that season's regulars (>= 300 PA)."""
    reg = b[(b.PA >= MIN_PA) & (b.yearID != SHORT_SEASON)]
    means = reg.groupby("yearID")[["OPS"] + list(RATES)].mean()
    out = b.copy()
    for s in ["OPS"] + list(RATES):
        out[s + "_d"] = out[s] - out.yearID.map(means[s])
    return out


def pairs_next(b: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    nxt = b[["playerID", "yearID", "PA"] + cols].copy()
    nxt["yearID"] -= 1
    return b.merge(nxt, on=["playerID", "yearID"], suffixes=("", "_n"), how="left")


def sampling_variance(reg: pd.DataFrame, rng, draws: int = 400) -> pd.DataFrame:
    """Sampling variance of each statistic in a season, holding the player's outcome probabilities fixed."""
    cnt = reg[CATS].to_numpy(float)
    n = cnt.sum(1).astype(np.int64)
    p = cnt / n[:, None]
    sims = rng.multinomial(n, p, size=(draws, len(n)))
    c = {k: sims[..., i].astype(float) for i, k in enumerate(CATS)}
    out = pd.DataFrame(index=reg.index)
    out["OPS_sv"] = ops_from_counts(c).var(0, ddof=1)
    for s, den in RATES.items():
        out[s + "_sv"] = reg[s] * (1 - reg[s]) / reg[den]
    return out


def lindsey_score(x, bins: int = 80, degree: int = 7):
    """d/dx log f(x) for the density of x, from Lindsey's method: a Poisson regression of histogram counts
    on a polynomial in the bin centre (Efron 2011). Returns a function, clipped to the central range."""
    lo, hi = np.quantile(x, [0.0005, 0.9995])
    edges = np.linspace(lo, hi, bins + 1)
    mid = (edges[:-1] + edges[1:]) / 2
    y, _ = np.histogram(x, edges)
    c, s = mid.mean(), mid.std()
    Z = np.vander((mid - c) / s, degree + 1, increasing=True)
    beta = np.zeros(degree + 1)
    beta[0] = math.log(y.mean())
    for _ in range(200):                      # iteratively reweighted least squares
        eta = Z @ beta
        mu = np.exp(eta)
        new = np.linalg.solve(Z.T @ (mu[:, None] * Z), Z.T @ (mu * (eta + (y - mu) / mu)))
        done = np.max(np.abs(new - beta)) < 1e-10
        beta = new
        if done:
            break

    def score(v):
        u = (np.clip(v, lo, hi) - c) / s
        return sum(beta[j] * j * u ** (j - 1) for j in range(1, degree + 1)) / s

    return score


def tweedie(x, noise_var, score):
    """Tweedie's formula: E[true | observed x] = x + noise variance * d/dx log f(x)."""
    return x + noise_var * score(x)


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.diag(cov))


def slope(x, y):
    return float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))


def load_teams() -> pd.DataFrame:
    t = pd.read_csv(fetch("Teams.csv"), usecols=["yearID", "franchID", "G", "W", "L", "R", "RA"])
    t = t[(t.yearID >= 1961) & (t.yearID != SHORT_SEASON)].copy()
    t["wp"] = t.W / (t.W + t.L)
    return t.reset_index(drop=True)


def pythagorean(t: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    f = lambda k: float(np.sum((t.wp - t.R ** k / (t.R ** k + t.RA ** k)) ** 2))
    k = optimize.minimize_scalar(f, bounds=(1.0, 3.0), method="bounded").x
    t = t.copy()
    t["pyth"] = t.R ** k / (t.R ** k + t.RA ** k)
    t["dev"] = t.wp - 0.5
    t["pdev"] = t.pyth - 0.5
    t["luck"] = t.wp - t.pyth
    nxt = t[["franchID", "yearID", "dev", "pdev", "luck"]].copy()
    nxt["yearID"] -= 1
    p = t.merge(nxt, on=["franchID", "yearID"], suffixes=("", "_n"))
    p = p[p.yearID != SHORT_SEASON - 1]
    return p.reset_index(drop=True), k


def kept_share(first, second, sel):
    """Share of the selected group's distance from the mean on the first occasion still present on the second."""
    return float((np.mean(second[sel]) - np.mean(second)) / (np.mean(first[sel]) - np.mean(first)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    pd.set_option("display.width", 120)
    calibration = []   # (domain, label, predicted kept share, observed kept share, n)

    # -----------------------------------------------------------------------
    header("1. NHANES 2009-2012: the sample, the three readings and the drift between them")
    d = load_nhanes()
    adults, a = nhanes_adults(d)
    x1, x2, x3, m23 = (a[c].to_numpy(float) for c in ("BPSys1", "BPSys2", "BPSys3", "m23"))
    w = a.w4.to_numpy()
    reps = jk_replicates(a.SDMVSTRA.to_numpy(), a.SDMVPSU.to_numpy(), w)
    print(f"participants in the file: {len(d)}; aged 18 or over: {len(adults)}")
    any_bp = adults[["BPSys1", "BPSys2", "BPSys3"]].notna().any(axis=1).sum()
    print(f"adults with at least one systolic reading: {any_bp}; with all three: {len(a)}")
    print(f"strata: {a.SDMVSTRA.nunique()}, PSUs: {a.groupby(['SDMVSTRA', 'SDMVPSU']).ngroups}, "
          f"jackknife replicates: {len(reps)}")
    w_adults = adults.WTMEC2YR.to_numpy() / 2
    print(f"weighted population represented (millions): {w.sum() / 1e6:.1f}; by all examined adults "
          f"{w_adults.sum() / 1e6:.1f}; share without three readings {1 - w.sum() / w_adults.sum():.3f}")
    even = np.mean(np.concatenate([x1, x2, x3]) % 2 == 0)
    print(f"share of readings that are even numbers: {even:.4f}")
    digits = np.bincount((x1 % 10).astype(int), minlength=10)[::2] / len(x1)
    print("last digit of the first reading, shares of 0/2/4/6/8:", " ".join(f"{v:.3f}" for v in digits))
    print(f"largest |BPSysAve - mean of readings 2 and 3|: {np.max(np.abs(a.BPSysAve - m23)):.3f}")
    for name, v in (("reading 1", x1), ("reading 2", x2), ("reading 3", x3), ("mean of 2 and 3", m23)):
        print(f"  {name:16s} weighted mean {wmean(v, w):7.2f}  SD {math.sqrt(wcov(v, v, w)):6.2f}")
    est, se = jk(lambda ww: [wmean(x2, ww) - wmean(x1, ww), wmean(x3, ww) - wmean(x1, ww),
                             wmean(m23, ww) - wmean(x1, ww)], w, reps)
    print(f"drift from reading 1 to reading 2: {est[0]:+.2f} (SE {se[0]:.2f}); to reading 3: {est[1]:+.2f} "
          f"(SE {se[1]:.2f}); to the mean of 2 and 3: {est[2]:+.2f} (SE {se[2]:.2f})")
    C = np.array([[wcov(u, v, w) for v in (x1, x2, x3)] for u in (x1, x2, x3)])
    print("weighted covariance matrix of the three readings:")
    for row in C:
        print("   " + "  ".join(f"{v:7.2f}" for v in row))
    R = np.array([[C[i, j] / math.sqrt(C[i, i] * C[j, j]) for j in range(3)] for i in range(3)])
    print(f"correlations: r12 {R[0, 1]:.4f}, r13 {R[0, 2]:.4f}, r23 {R[1, 2]:.4f}; "
          f"r(reading 1, mean of 2 and 3) {wcov(x1, m23, w) / math.sqrt(wcov(x1, x1, w) * wcov(m23, m23, w)):.4f}")
    b_all = wcov(x1, m23, w) / wcov(x1, x1, w)
    print(f"weighted slope of the mean of readings 2-3 on reading 1: {b_all:.4f} "
          f"(unweighted {slope(x1, m23):.4f})")

    # -----------------------------------------------------------------------
    header("2. Selecting on the first reading: drift, regression and what is left")
    for cut, side in ((130, "high"), (140, "high"), (160, "high"), (100, "low")):
        sel = x1 >= cut if side == "high" else x1 < cut
        r = selection_decomposition(x1, m23, w, sel)
        f = lambda ww: [selection_decomposition(x1, m23, ww, sel)[k] for k in ("fall", "residual", "kept")]
        _, se = jk(f, w, reps)
        rel = ">=" if side == "high" else "<"
        print(f"first reading {rel} {cut}: n {sel.sum()}, weighted share {r['share']:.3f}")
        print(f"   first {r['first']:.2f} -> mean of 2-3 {r['later']:.2f}: change {-r['fall']:+.2f} (SE {se[0]:.2f})")
        print(f"   population drift {-r['drift']:+.2f}, predicted regression {-r['regression']:+.2f}, "
              f"predicted mean {r['predicted']:.2f}, residual {r['residual']:+.2f} (SE {se[1]:.2f})")
        print(f"   share of the deviation kept {r['kept']:.3f} (SE {se[2]:.3f}) against the slope {r['slope']:.3f}")
    for cut in (130, 140):
        pos1 = x1 >= cut
        pos23 = m23 >= cut
        f = lambda ww: [wmean(pos1, ww), wmean(pos23, ww), wmean(~pos23[pos1], ww[pos1]),
                        wmean(pos23[~pos1], ww[~pos1])]
        est, se = jk(f, w, reps)
        print(f"threshold {cut}: share at or above on reading 1 {est[0]:.3f} (SE {se[0]:.3f}); on the mean of 2-3 "
              f"{est[1]:.3f} (SE {se[1]:.3f})")
        print(f"   of those at or above on reading 1, below on the mean of 2-3: {est[2]:.3f} (SE {se[2]:.3f}); "
              f"of those below on reading 1, at or above on the mean of 2-3: {est[3]:.3f} (SE {se[3]:.3f})")
        print(f"   people moved down per person moved up: "
              f"{est[2] * est[0] / (est[3] * (1 - est[0])):.2f}")
        shift = wmean(x1, w) - wmean(m23, w)
        pos23d = (m23 + shift) >= cut
        down = wmean(~pos23d[pos1], w[pos1])
        up = wmean(pos23d[~pos1], w[~pos1])
        print(f"   with the population drift added back to the later mean: below {down:.3f}, above {up:.3f}, "
              f"moved down per person moved up {down * est[0] / (up * (1 - est[0])):.2f}")
    cross = (wmean(m23, w) - b_all * wmean(x1, w)) / (1 - b_all)
    print(f"first reading at which drift and regression cancel (the line crosses zero): {cross:.1f}")
    print("share with the mean of readings 2-3 at or above 140, by first reading (weighted):")
    print("   " + "  ".join(f"{v}: {wmean((m23 >= 140)[x1 == v], w[x1 == v]):.2f}" for v in range(134, 160, 2)))
    print("binned means of the change from reading 1 to the mean of readings 2-3 (weighted):")
    edges = [80, 100, 110, 120, 130, 140, 150, 160, 170, 180, 230]
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (x1 >= lo) & (x1 < hi)
        ch = m23[k] - x1[k]
        pred = wmean(m23, w) + b_all * (wmean(x1[k], w[k]) - wmean(x1, w)) - wmean(x1[k], w[k])
        print(f"   first reading {lo}-{hi - 2}: n {k.sum():5d}  change {wmean(ch, w[k]):+6.2f}  line {pred:+6.2f}")

    # -----------------------------------------------------------------------
    header("3. Why the first reading is different")
    f = lambda ww: [wcov(x1, x1, ww) - wcov(x2, x2, ww), wcov(x2, x2, ww) - wcov(x3, x3, ww),
                    wcov(x1, x2, ww) - wcov(x2, x3, ww), wcov(x1, x3, ww) - wcov(x2, x3, ww)]
    est, se = jk(f, w, reps)
    print(f"variance of reading 1 minus reading 2: {est[0]:.2f} (SE {se[0]:.2f}); reading 2 minus 3: {est[1]:.2f} "
          f"(SE {se[1]:.2f})")
    print(f"cov12 - cov23: {est[2]:.2f} (SE {se[2]:.2f}); cov13 - cov23: {est[3]:.2f} (SE {se[3]:.2f})")
    noise = {j: math.sqrt(C[j, j] - C[1, 2]) for j in range(3)}
    print("variance of each reading not shared with the others (Var_j - cov23), as an SD: "
          + ", ".join(f"reading {j + 1} {noise[j]:.2f}" for j in range(3)))
    oldham = wcov(x1 - x2, x1 + x2, w) / math.sqrt(wcov(x1 - x2, x1 - x2, w) * wcov(x1 + x2, x1 + x2, w))
    print(f"weighted correlation of (reading 1 - reading 2) with their sum (Oldham): {oldham:.3f}")
    print("within-sitting noise by level: SD of (reading 2 - reading 3)/sqrt(2) in bins of the mean of 2-3:")
    for lo, hi in ((0, 110), (110, 120), (120, 130), (130, 140), (140, 160), (160, 300)):
        k = (m23 >= lo) & (m23 < hi)
        print(f"   {lo:3d}-{hi:3d}: n {k.sum():5d}  noise SD {math.sqrt(wcov(x2[k] - x3[k], x2[k] - x3[k], w[k]) / 2):.2f}")
    print("by age group:")
    age = a.Age.to_numpy()
    for lo, hi in ((18, 40), (40, 60), (60, 81)):
        k = (age >= lo) & (age < hi)
        bk = wcov(x1[k], m23[k], w[k]) / wcov(x1[k], x1[k], w[k])
        nz = math.sqrt(wcov(x2[k] - x3[k], x2[k] - x3[k], w[k]) / 2)
        sk = k & (x1 >= 140)
        r = selection_decomposition(x1[k], m23[k], w[k], x1[k] >= 140)
        v1k = wcov(x1[k], x1[k], w[k])
        print(f"   {lo}-{hi - 1}: n {k.sum()}, mean reading 1 {wmean(x1[k], w[k]):.2f}, SD {math.sqrt(v1k):.2f}, "
              f"noise SD {nz:.2f}, slope {bk:.3f}, drift {wmean(m23[k], w[k]) - wmean(x1[k], w[k]):+.2f}")
        print(f"      slope if within-sitting noise were the only difference {(v1k - nz ** 2) / v1k:.3f}; variance of "
              f"reading 1 minus reading 3 {v1k - wcov(x3[k], x3[k], w[k]):.1f}")
        _, se_r = jk(lambda ww: selection_decomposition(x1[k], m23[k], ww[k], x1[k] >= 140)["residual"], w, reps)
        pooled = wmean(m23, w) + b_all * (r["first"] - wmean(x1, w))
        print(f"      selected at 140: n {sk.sum()}, share of the age group {r['share']:.3f}, first {r['first']:.2f} -> "
              f"{r['later']:.2f}; drift {-r['drift']:+.2f}, regression {-r['regression']:+.2f}, residual {r['residual']:+.2f} "
              f"(SE {float(se_r):.2f})")
        print(f"      fall {r['fall']:.2f}; fall net of drift {r['fall'] - r['drift']:.2f}; predicted later mean: own line "
              f"{r['predicted']:.2f}, pooled line {pooled:.2f}")
    sel = x1 >= 140
    print(f"mean age of those at or above 140 on reading 1: {wmean(age[sel], w[sel]):.1f}; of all adults "
          f"{wmean(age, w):.1f}")

    # -----------------------------------------------------------------------
    header("4. The next visit: a scenario with day-to-day variation from an outside estimate")
    c4 = math.sqrt(2 / 2) * math.gamma(3 / 2) / math.gamma(1)
    sigma_v = MUNTNER_MEAN_SD / c4
    within_avg = math.sqrt(wcov(x2 - x3, x2 - x3, w) / 4)
    day = math.sqrt(sigma_v ** 2 - within_avg ** 2)
    print(f"Muntner et al. mean SD over three visits {MUNTNER_MEAN_SD}; c4(3) = {c4:.4f}; implied SD of a visit mean "
          f"{sigma_v:.2f}")
    print(f"within-sitting noise of the mean of readings 2-3: SD {within_avg:.2f}; implied day-to-day SD {day:.2f}")
    vu = wcov(m23, m23, w) - sigma_v ** 2
    print(f"variance of the mean of 2-3 {wcov(m23, m23, w):.1f}; of usual pressure under the scenario {vu:.1f}; "
          f"reliability of one visit {vu / wcov(m23, m23, w):.3f}")
    # Model check at the same sitting: normal model for (reading 1, mean of 2-3) against what was observed.
    v1, c1 = wcov(x1, x1, w), wcov(x1, m23, w)
    cm = wmean(m23, w) + c1 / v1 * (x1 - wmean(x1, w))
    csd = math.sqrt(wcov(m23, m23, w) - c1 ** 2 / v1)
    model_same = wmean(stats.norm.cdf((140 - cm[sel]) / csd), w[sel])
    print(f"model check, same sitting: share of those >= 140 on reading 1 below 140 on 2-3, normal model "
          f"{model_same:.3f}, observed {wmean((m23 < 140)[sel], w[sel]):.3f}")
    for s_label, sv in (("within-sitting noise only", within_avg),
                        ("day-to-day SD 4", math.sqrt(within_avg ** 2 + 16)),
                        ("Muntner-based", sigma_v),
                        ("day-to-day SD 12", math.sqrt(within_avg ** 2 + 144))):
        out = visit_scenario(x1, m23, w, x2, x3, sv, 140)
        f1, f2 = out["first"], out["mean23"]
        print(f"{s_label:26s} (visit SD {sv:5.2f}): selected on reading 1: slope {f1['slope']:.3f}, "
              f"{f1['selected_mean']:.2f} -> next visit {f1['mean_next']:.2f}, below 140 {f1['share_below']:.3f}")
        print(f"{'':26s}  selected on mean of 2-3: slope {f2['slope']:.3f}, {f2['selected_mean']:.2f} -> next visit "
              f"{f2['mean_next']:.2f}, below 140 {f2['share_below']:.3f}")
    for cut in (130, 140):
        single = wmean((m23 >= cut).astype(float), w)
        print(f"share at or above {cut}: one visit (mean of 2-3) {single:.3f}; usual pressure, NRC rescaling within "
              f"age-sex groups: Muntner-based {usual_prevalence(a, sigma_v, cut):.3f}, "
              f"within-sitting noise only {usual_prevalence(a, within_avg, cut):.3f}")

    # -----------------------------------------------------------------------
    header("5. Held-out check: the line from 2009-2010 predicts the selections of 2011-2012")
    tr = (a.SurveyYr == "2009_10").to_numpy()
    te = ~tr
    w2 = a.WTMEC2YR.to_numpy()
    b_tr = wcov(x1[tr], m23[tr], w2[tr]) / wcov(x1[tr], x1[tr], w2[tr])
    a_tr = wmean(m23[tr], w2[tr]) - b_tr * wmean(x1[tr], w2[tr])
    b_te = wcov(x1[te], m23[te], w2[te]) / wcov(x1[te], x1[te], w2[te])
    print(f"2009-2010: n {tr.sum()}, line {a_tr:.2f} + {b_tr:.4f} x; 2011-2012: n {te.sum()}, slope {b_te:.4f}")
    reps2 = jk_replicates(a.SDMVSTRA.to_numpy(), a.SDMVPSU.to_numpy(), w2)

    def cycle_stats(ww):
        out = []
        for k in (tr, te):
            out += [wmean(m23[k], ww[k]) - wmean(x1[k], ww[k]), wcov(x1[k], x1[k], ww[k]) - wcov(x3[k], x3[k], ww[k]),
                    wcov(x1[k], x2[k], ww[k]) - wcov(x2[k], x3[k], ww[k])]
        return out + [out[3] - out[0], out[4] - out[1], out[5] - out[2]]

    est, se = jk(cycle_stats, w2, reps2)
    names = ("drift", "var1 - var3", "cov12 - cov23")
    for i, lab in enumerate(("2009-2010", "2011-2012")):
        print(f"   {lab}: " + ", ".join(f"{names[j]} {est[3 * i + j]:+.2f} (SE {se[3 * i + j]:.2f})" for j in range(3)))
    print("   2011-2012 minus 2009-2010: " + ", ".join(f"{names[j]} {est[6 + j]:+.2f} (SE {se[6 + j]:.2f})"
                                                     for j in range(3)))
    for lab, k in (("2009-2010", tr), ("2011-2012", te)):
        xx1, xx2, xx3, mm, ww = x1[k], x2[k], x3[k], m23[k], w2[k]
        print(f"   {lab}: mean reading 1 {wmean(xx1, ww):.2f}, SD {math.sqrt(wcov(xx1, xx1, ww)):.2f}, drift to the mean "
              f"of 2-3 {wmean(mm, ww) - wmean(xx1, ww):+.2f}, noise SD from readings 2-3 "
              f"{math.sqrt(wcov(xx2 - xx3, xx2 - xx3, ww) / 2):.2f}, from readings 1-2 "
              f"{math.sqrt(wcov(xx1 - xx2, xx1 - xx2, ww) / 2):.2f}, variance of reading 1 minus reading 3 "
              f"{wcov(xx1, xx1, ww) - wcov(xx3, xx3, ww):.1f}")
    xt, mt, wt = x1[te], m23[te], w2[te]
    mu1t, mu23t = wmean(xt, wt), wmean(mt, wt)
    for label, q, side in (("top 1%", 0.99, "high"), ("top 5%", 0.95, "high"), ("top 10%", 0.90, "high"),
                           ("top 25%", 0.75, "high"), ("bottom 10%", 0.10, "low"), ("bottom 25%", 0.25, "low")):
        thr = wquantile(xt, wt, q)
        sel = xt >= thr if side == "high" else xt <= thr
        xs, ys = wmean(xt[sel], wt[sel]), wmean(mt[sel], wt[sel])
        obs_kept = (ys - mu23t) / (xs - mu1t)
        print(f"   {label:10s} (n {sel.sum():4d}): first {xs:6.2f}, observed {ys:6.2f}, predicted {a_tr + b_tr * xs:6.2f}; "
              f"kept {obs_kept:.3f} vs {b_tr:.3f}")
        calibration.append(("blood pressure", label, b_tr, obs_kept, int(sel.sum())))

    # -----------------------------------------------------------------------
    header("6. Batting: the reliability of a season from the sampling model")
    b = season_deviations(load_batting())
    bad = (b.X1B < 0) | (b.OUT < 0)
    print(f"player-seasons in Batting: {len(b)}; with inconsistent counts (dropped): {int(bad.sum())}")
    b = b[~bad].copy()
    regm = (b.PA >= MIN_PA) & (b.yearID >= FIRST) & (b.yearID <= LAST + 1) & (b.yearID != SHORT_SEASON)
    reg = b[regm].copy()
    sv = sampling_variance(reg, np.random.default_rng(SEED + 6))
    reg = reg.join(sv)
    print(f"regular seasons ({MIN_PA}+ PA), {FIRST}-{LAST + 1} without {SHORT_SEASON}: {len(reg)}")
    rel = {}
    for s in ["OPS"] + list(RATES):
        V = reg[s + "_d"].var()
        S = reg[s + "_sv"].mean()
        rel[s] = 1 - S / V
        print(f"   {LABELS[s]:15s} SD across players {math.sqrt(V):.4f}, sampling SD {math.sqrt(S):.4f}, "
              f"reliability of one season {rel[s]:.3f}")
    cols = [s + "_d" for s in ["OPS"] + list(RATES)]
    pr = pairs_next(reg, cols)
    pr = pr[pr.PA_n.notna() & (pr.yearID <= LAST) & (pr.yearID != SHORT_SEASON - 1)]
    print(f"consecutive pairs of regular seasons: {len(pr)}")
    yty = {}
    for s in ["OPS"] + list(RATES):
        r = np.corrcoef(pr[s + "_d"], pr[s + "_d_n"])[0, 1]
        yty[s] = r
        sel_hi = pr[s + "_d"] >= pr[s + "_d"].quantile(0.9)
        kept = kept_share(pr[s + "_d"].to_numpy(), pr[s + "_d_n"].to_numpy(), sel_hi.to_numpy())
        print(f"   {LABELS[s]:15s} year-to-year r {r:.3f}, slope {slope(pr[s + '_d'], pr[s + '_d_n']):.3f}, "
              f"one-season reliability {rel[s]:.3f}, implied correlation of true rates {r / rel[s]:.3f}; "
              f"top 10% keep {kept:.3f}")
    allnext = b[["playerID", "yearID", "PA"]].copy()
    allnext["yearID"] -= 1
    surv = reg[(reg.yearID <= LAST) & (reg.yearID != SHORT_SEASON - 1)].merge(
        allnext, on=["playerID", "yearID"], suffixes=("", "_n"))
    lo_q, hi_q = surv.OPS_d.quantile(0.1), surv.OPS_d.quantile(0.9)
    print(f"regular seasons followed by any next season: {len(surv)}; share reaching {MIN_PA} PA the next season: "
          f"bottom tenth by OPS {(surv.PA_n[surv.OPS_d <= lo_q] >= MIN_PA).mean():.3f}, "
          f"top tenth {(surv.PA_n[surv.OPS_d >= hi_q] >= MIN_PA).mean():.3f}")
    babip_share = reg.BIP.mean() / reg.PA.mean()
    print(f"balls in play per plate appearance among regulars: {babip_share:.3f}; mean PA {reg.PA.mean():.0f}")

    # -----------------------------------------------------------------------
    header("7. Rookies and Rookie of the Year winners")
    awards = pd.read_csv(fetch("AwardsPlayers.csv"))
    roy_all = awards[awards.awardID == "Rookie of the Year"]
    in_window = (roy_all.yearID >= FIRST) & (roy_all.yearID <= LAST) & ~roy_all.yearID.isin([SHORT_SEASON - 1, SHORT_SEASON])
    n_awards = roy_all[in_window][["yearID", "lgID"]].drop_duplicates().shape[0]
    roy = roy_all[["playerID", "yearID"]].drop_duplicates()
    roy_years = roy[(roy.yearID >= FIRST) & (roy.yearID <= LAST) & ~roy.yearID.isin([SHORT_SEASON - 1, SHORT_SEASON])]
    ties = roy_all[in_window].groupby(["yearID", "lgID"]).size()
    print(f"awards (league-seasons) {n_awards}; shared awards: " + ", ".join(f"{y} {l}" for (y, l), c in ties.items() if c > 1))
    bb = pairs_next(b, cols + ["OPS"])
    bb = bb.merge(roy.assign(roy=1), on=["playerID", "yearID"], how="left")
    bb["roy"] = bb.roy.fillna(0).astype(int)
    window = (bb.yearID >= FIRST) & (bb.yearID <= LAST) & ~bb.yearID.isin([SHORT_SEASON - 1, SHORT_SEASON])
    roy_hit = bb[window & (bb.roy == 1) & (bb.PA >= MIN_PA)]
    print(f"Rookie of the Year winners {FIRST}-{LAST} (without {SHORT_SEASON - 1} and {SHORT_SEASON}): {len(roy_years)}; "
          f"winners with {MIN_PA}+ PA in the award season: {len(roy_hit)}; of these with more than {ROOKIE_AB} career AB "
          f"before it: {int((roy_hit.AB_before > ROOKIE_AB).sum())}")
    rk = bb[window & (bb.PA >= MIN_PA) & (bb.AB_before <= ROOKIE_AB)].copy()
    rk["PA_n"] = rk.PA_n.fillna(0)
    win = rk.roy == 1
    print(f"rookie seasons ({MIN_PA}+ PA, at most {ROOKIE_AB} AB before): {len(rk)}, winners among them {int(win.sum())}")
    print(f"mean age: rookies who did not win {rk.age[~win].mean():.2f}, winners {rk.age[win].mean():.2f}")
    for thr in (1, 100, 300):
        print(f"   share with {thr}+ PA the next season: non-winners {(rk.PA_n[~win] >= thr).mean():.3f}, "
              f"winners {(rk.PA_n[win] >= thr).mean():.3f}")
    print(f"winners' raw OPS: rookie season {rk.OPS[win].mean():.3f}, next season {rk.OPS_n[win].mean():.3f}")
    roy_rows = {}
    for thr in (1, 100, 300):
        r = rk[rk.PA_n >= thr]
        nw, ww_ = r[r.roy == 0], r[r.roy == 1]
        X = np.column_stack([np.ones(len(nw)), nw.OPS_d, nw.age - 24])
        beta, se = ols(X, nw.OPS_d_n.to_numpy())
        pred = beta[0] + beta[1] * ww_.OPS_d + beta[2] * (ww_.age - 24)
        res = ww_.OPS_d_n - pred
        roy_rows[thr] = (beta, ww_, pred)
        print(f"next-season PA >= {thr}: non-winners {len(nw)}, line {beta[0]:+.4f} + {beta[1]:.3f} d "
              f"{beta[2]:+.4f} (age - 24) (slope SE {se[1]:.3f})")
        print(f"   winners ({len(ww_)}): OPS deviation {ww_.OPS_d.mean():+.4f} -> {ww_.OPS_d_n.mean():+.4f}; "
              f"predicted {pred.mean():+.4f}; difference {res.mean():+.4f} (SE {res.std(ddof=1) / math.sqrt(len(res)):.4f})")
        q = r.OPS_d.quantile(0.9)
        top = r.OPS_d >= q
        b_r = slope(r.OPS_d, r.OPS_d_n)
        a_r = r.OPS_d_n.mean() - b_r * r.OPS_d.mean()
        print(f"   top 10% of rookies by OPS ({int(top.sum())}, winners {r.roy[top].mean():.3f}): "
              f"{r.OPS_d[top].mean():+.4f} -> {r.OPS_d_n[top].mean():+.4f}, line predicts "
              f"{a_r + b_r * r.OPS_d[top].mean():+.4f}")
    main_thr = 100
    beta, ww_, pred = roy_rows[main_thr]
    fall_obs = ww_.OPS_d.mean() - ww_.OPS_d_n.mean()
    fall_pred = ww_.OPS_d.mean() - pred.mean()
    print(f"winners' fall in OPS deviation: observed {fall_obs:.4f}, predicted {fall_pred:.4f}, "
          f"share of the observed fall predicted {fall_pred / fall_obs:.2f}")
    r = rk[rk.PA_n >= main_thr]
    print("winners' components, rookie season -> next season (deviations from the season mean of regulars):")
    for s in RATES:
        print(f"   {LABELS[s]:15s} {r[s + '_d'][r.roy == 1].mean():+.4f} -> {r[s + '_d_n'][r.roy == 1].mean():+.4f}")
    X = np.column_stack([np.ones(len(r)), r.OPS_d, r.BABIP_d, r.age - 24])
    beta2, se2 = ols(X, r.OPS_d_n.to_numpy())
    print(f"rookies, next-season OPS deviation on rookie OPS, rookie BABIP and age: OPS {beta2[1]:.3f} "
          f"(SE {se2[1]:.3f}), BABIP {beta2[2]:+.3f} (SE {se2[2]:.3f}), age {beta2[3]:+.4f} (SE {se2[3]:.4f})")
    print(f"   so two rookies with the same OPS, one with BABIP 0.030 higher: next-season OPS lower by "
          f"{-0.030 * beta2[2]:.4f}")

    # -----------------------------------------------------------------------
    header("8. Teams: run differential and the Pythagorean residual")
    teams = load_teams()
    tp, k_exp = pythagorean(teams)
    print(f"team seasons 1961-2025 without {SHORT_SEASON}: {len(teams)}; consecutive pairs (franchise): {len(tp)}")
    print(f"fitted Pythagorean exponent {k_exp:.3f}")
    print(f"SD of winning percentage {teams.wp.std():.4f}; of the Pythagorean expectation {tp.pyth.std():.4f}; "
          f"of the residual {tp.luck.std():.4f}; corr(expectation, residual) {np.corrcoef(tp.pdev, tp.luck)[0, 1]:+.3f}")
    print(f"binomial SD of a winning percentage over 162 games at 0.5: {math.sqrt(0.25 / 162):.4f}")
    bs = {}
    for s, lab in (("dev", "winning percentage"), ("pdev", "Pythagorean expectation"), ("luck", "residual")):
        bs[s] = slope(tp[s], tp[s + "_n"])
        print(f"   {lab:24s} year-to-year r {np.corrcoef(tp[s], tp[s + '_n'])[0, 1]:.3f}, slope {bs[s]:.3f}")
    for label, sel in (("top quarter by winning percentage", tp.dev >= tp.dev.quantile(0.75)),
                       ("top tenth by residual", tp.luck >= tp.luck.quantile(0.9)),
                       ("bottom quarter by winning percentage", tp.dev <= tp.dev.quantile(0.25))):
        g = tp[sel]
        pred_c = bs["pdev"] * g.pdev.mean() + bs["luck"] * g.luck.mean()
        pred_1 = bs["dev"] * g.dev.mean()
        print(f"   {label} ({len(g)}): surplus {g.dev.mean():+.4f} = expectation {g.pdev.mean():+.4f} + residual "
              f"{g.luck.mean():+.4f}")
        print(f"      next season {g.dev_n.mean():+.4f} = {g.pdev_n.mean():+.4f} + {g.luck_n.mean():+.4f}; kept "
              f"{g.dev_n.mean() / g.dev.mean():.3f}; predicted by components {pred_c:+.4f}, by one slope {pred_1:+.4f}")
        print(f"      in wins over 162 games: {162 * g.dev.mean():+.1f} -> {162 * g.dev_n.mean():+.1f}; "
              f"next-season expectation {162 * g.pdev_n.mean():+.1f} wins (SE {162 * g.pdev_n.std(ddof=1) / math.sqrt(len(g)):.1f})")

    # -----------------------------------------------------------------------
    header("9. Held-out check for baseball (slopes from 1955-1989, selections from 1990 on)")
    trn, tst = pr[pr.yearID <= 1989], pr[pr.yearID >= 1990]
    print(f"batting pairs: training {len(trn)}, test {len(tst)}")
    print("two predictions: one slope for everyone, and a player-specific reliability from the season's sampling")
    print("variance (R_i = V_true / (V_true + v_i)) times a persistence fitted in the training period")
    unit_rows = []
    for s in ["OPS"] + list(RATES):
        b_s = slope(trn[s + "_d"], trn[s + "_d_n"])
        v_true = trn[s + "_d"].var() - trn[s + "_sv"].mean()
        r_tr = v_true / (v_true + trn[s + "_sv"])
        rho = slope(r_tr * trn[s + "_d"], trn[s + "_d_n"])
        x, y = tst[s + "_d"].to_numpy(), tst[s + "_d_n"].to_numpy()
        r_te = (v_true / (v_true + tst[s + "_sv"])).to_numpy()
        shrunk = r_te * x
        xm, ym = x.mean(), y.mean()
        print(f"   {LABELS[s]:15s} training slope {b_s:.3f}; true variance {v_true:.6f}; persistence {rho:.3f}; "
              f"player reliabilities in the test period from {np.quantile(r_te, 0.1):.3f} to {np.quantile(r_te, 0.9):.3f} "
              f"(10th-90th percentile)")
        sels = (("top 10%", x >= np.quantile(x, 0.9)), ("bottom 10%", x <= np.quantile(x, 0.1))) if s != "OPS" else \
            (("top 1%", x >= np.quantile(x, 0.99)), ("top 5%", x >= np.quantile(x, 0.95)),
             ("top 10%", x >= np.quantile(x, 0.9)), ("top 25%", x >= np.quantile(x, 0.75)),
             ("bottom 10%", x <= np.quantile(x, 0.1)))
        for label, sel in sels:
            obs = (y[sel].mean() - ym) / (x[sel].mean() - xm)
            unit = rho * (shrunk[sel].mean() - shrunk.mean()) / (x[sel].mean() - xm)
            print(f"      {label:10s} n {sel.sum():4d}: kept {obs:.3f}; one slope {b_s:.3f}; player-specific {unit:.3f}; "
                  f"mean PA {tst.PA.to_numpy()[sel].mean():.0f}, sampling SD {np.sqrt(tst[s + '_sv'].to_numpy()[sel]).mean():.4f}")
            calibration.append(("batting", f"{LABELS[s]}, {label}", b_s, obs, int(sel.sum())))
            unit_rows.append((f"{LABELS[s]}, {label}", b_s, unit, obs))
    e1 = np.array([o - p for _, p, _, o in unit_rows])
    e2 = np.array([o - u for _, _, u, o in unit_rows])
    print(f"batting selections, mean absolute difference: one slope {np.mean(np.abs(e1)):.3f}, "
          f"player-specific {np.mean(np.abs(e2)):.3f}")
    # Tweedie's formula for OPS: the shape of the distribution of first seasons sets the regression.
    first_train = reg[reg.yearID <= 1989].OPS_d.to_numpy()
    print(f"OPS deviations of regular seasons to 1989: skewness {stats.skew(first_train):.3f}; "
          f"sampling variance {trn.OPS_sv.mean():.6f}")
    x, y = tst.OPS_d.to_numpy(), tst.OPS_d_n.to_numpy()
    xm, ym = x.mean(), y.mean()
    b_ops = slope(trn.OPS_d, trn.OPS_d_n)
    v_true = trn.OPS_d.var() - trn.OPS_sv.mean()
    r_tr = v_true / (v_true + trn.OPS_sv)
    rho_u = slope(r_tr * trn.OPS_d, trn.OPS_d_n)
    shrunk = (v_true / (v_true + tst.OPS_sv)).to_numpy() * x
    ops_rows = []
    for degree in (5, 7, 9):
        score = lindsey_score(first_train, degree=degree)
        post_tr = tweedie(trn.OPS_d.to_numpy(), trn.OPS_sv.mean(), score)
        rho_t = slope(post_tr, trn.OPS_d_n)
        post = tweedie(x, trn.OPS_sv.mean(), score)
        line = []
        for label, sel in (("top 1%", x >= np.quantile(x, 0.99)), ("top 5%", x >= np.quantile(x, 0.95)),
                           ("top 10%", x >= np.quantile(x, 0.9)), ("top 25%", x >= np.quantile(x, 0.75)),
                           ("bottom 25%", x <= np.quantile(x, 0.25)), ("bottom 10%", x <= np.quantile(x, 0.1)),
                           ("bottom 1%", x <= np.quantile(x, 0.01))):
            dx = x[sel].mean() - xm
            obs = (y[sel].mean() - ym) / dx
            tw = rho_t * (post[sel].mean() - post.mean()) / dx
            un = rho_u * (shrunk[sel].mean() - shrunk.mean()) / dx
            line.append((label, obs, b_ops, un, tw))
        if degree == 7:
            ops_rows = line
            print(f"Tweedie with Lindsey's method (degree 7, persistence {rho_t:.3f}), OPS selections in the test period:")
            for label, obs, b1, un, tw in line:
                print(f"   {label:10s} observed {obs:.3f}; one slope {b1:.3f}; player-specific {un:.3f}; Tweedie {tw:.3f}")
            for k, name in ((2, "one slope"), (3, "player-specific"), (4, "Tweedie")):
                print(f"   mean absolute difference, {name}: {np.mean([abs(r[1] - r[k]) for r in line]):.3f}")
        else:
            print(f"   Tweedie with degree {degree}: " + ", ".join(f"{r[0]} {r[4]:.3f}" for r in line))
    print("Tweedie with one noise variance for everyone, applied to the rates (degree 7), test period:")
    for s in RATES:
        first_s = reg[reg.yearID <= 1989][s + "_d"].dropna().to_numpy()
        score = lindsey_score(first_s)
        nv = trn[s + "_sv"].mean()
        rho_t = slope(tweedie(trn[s + "_d"].to_numpy(), nv, score), trn[s + "_d_n"])
        x, y = tst[s + "_d"].to_numpy(), tst[s + "_d_n"].to_numpy()
        post = tweedie(x, nv, score)
        parts = []
        for label, sel in (("top 10%", x >= np.quantile(x, 0.9)), ("bottom 10%", x <= np.quantile(x, 0.1))):
            dx = x[sel].mean() - x.mean()
            parts.append(f"{label} observed {(y[sel].mean() - y.mean()) / dx:.3f}, Tweedie "
                         f"{rho_t * (post[sel].mean() - post.mean()) / dx:.3f}")
        print(f"   {LABELS[s]:15s} " + "; ".join(parts))
    ttr, tte = tp[tp.yearID <= 1989], tp[tp.yearID >= 1990]
    _, k_tr = pythagorean(teams[teams.yearID <= 1989])
    print(f"Pythagorean exponent fitted on 1961-1989 only: {k_tr:.3f} (all seasons {k_exp:.3f}); team kept shares are "
          f"measured from .500")
    bt = {s: slope(ttr[s], ttr[s + "_n"]) for s in ("dev", "pdev", "luck")}
    print(f"team pairs: training {len(ttr)}, test {len(tte)}; training slopes: winning percentage {bt['dev']:.3f}, "
          f"expectation {bt['pdev']:.3f}, residual {bt['luck']:.3f}")
    for label, sel in (("top 10% by wins", tte.dev >= tte.dev.quantile(0.9)),
                       ("top 25% by wins", tte.dev >= tte.dev.quantile(0.75)),
                       ("bottom 25% by wins", tte.dev <= tte.dev.quantile(0.25)),
                       ("top 10% by residual", tte.luck >= tte.luck.quantile(0.9))):
        g = tte[sel]
        pred = (bt["pdev"] * g.pdev.mean() + bt["luck"] * g.luck.mean()) / g.dev.mean()
        obs = g.dev_n.mean() / g.dev.mean()
        print(f"   {label:20s} n {len(g):3d}: kept {obs:.3f}, predicted by components {pred:.3f}, by one slope "
              f"{bt['dev']:.3f}")
        calibration.append(("teams", label, pred, obs, len(g)))
    rk_tr, rk_te = rk[(rk.yearID <= 1989) & (rk.PA_n >= main_thr)], rk[(rk.yearID >= 1990) & (rk.PA_n >= main_thr)]
    nw = rk_tr[rk_tr.roy == 0]
    X = np.column_stack([np.ones(len(nw)), nw.OPS_d, nw.age - 24])
    beta3, _ = ols(X, nw.OPS_d_n.to_numpy())
    wt_ = rk_te[rk_te.roy == 1]
    pr3 = beta3[0] + beta3[1] * wt_.OPS_d + beta3[2] * (wt_.age - 24)
    print(f"Rookie of the Year winners from 1990 ({len(wt_)}), line from non-winning rookies before 1990: "
          f"{wt_.OPS_d.mean():+.4f} -> observed {wt_.OPS_d_n.mean():+.4f}, predicted {pr3.mean():+.4f}")
    calibration.append(("batting", "Rookie of the Year winners", pr3.mean() / wt_.OPS_d.mean(),
                        wt_.OPS_d_n.mean() / wt_.OPS_d.mean(), len(wt_)))
    print("\npredicted against observed share of the deviation kept, held-out selections:")
    errs = []
    for dom, label, p, o, n in calibration:
        errs.append(o - p)
        print(f"   {dom:15s} {label:32s} n {n:5d}  predicted {p:.3f}  observed {o:.3f}  difference {o - p:+.3f}")
    errs = np.array(errs)
    print(f"mean absolute difference {np.mean(np.abs(errs)):.3f}; largest {errs[np.argmax(np.abs(errs))]:+.3f}")
    return {
        "nhanes": a, "sigma_visit": sigma_v, "within_avg": within_avg, "slope_bp": b_all,
        "regulars": reg, "pairs": pr, "reliability": rel, "year_to_year": yty,
        "rookies": rk, "rookie_line": roy_rows[main_thr][0], "teams": tp,
        "calibration": calibration, "ops_rows": ops_rows,
    }


if __name__ == "__main__":
    main()
