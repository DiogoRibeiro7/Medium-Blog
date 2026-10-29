"""Part 2 of the regression-to-the-mean series: the mechanism.

Every number quoted in the article is printed by this script. Run:

    python mechanism_numbers.py > output.txt

Data downloaded on first run from a pinned commit of the Rdatasets repository, checked against SHA-256
digests and cached in data/:
  - nycflights13::weather: hourly weather at the three New York airports in 2013 (daily means at JFK used);
  - datasets::EuStockMarkets: daily closing prices of four European stock indices, 1991-1998 (DAX used);
  - MASS::anorexia: weights of 72 young women with anorexia before and after one of three treatments.

Sections
1. The measurement model: a stable component plus noise, and the regression it implies.
2. Three sources of imperfect correlation, and a process with no mean to return to.
3. How much regression a selection produces: the closed form, a simulation check, and designs that reduce it.
4. Beyond the normal distribution: skewed, heavy-tailed and mixed populations.
5. Shrinkage: Kelley's formula, its error and its rankings.
6. Real series: daily temperature anomalies in New York, and an equity index in levels and in returns.
7. Separating regression from an effect: designs compared by simulation, and the anorexia trial.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import math
import urllib.request
from pathlib import Path

import numpy as np
from scipy import stats

SEED = 1886
HERE = Path(__file__).parent
DATA = HERE / "data"
RD = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/1dcc2bf5f955cc1224a3e1307256e1fe86b68dae/csv/"
FILES = {
    "weather.csv": RD + "nycflights13/weather.csv",
    "EuStockMarkets.csv": RD + "datasets/EuStockMarkets.csv",
    "anorexia.csv": RD + "MASS/anorexia.csv",
}
DIGESTS = {
    "weather.csv": "e69580cc3c18070692b6bf2f52e890803a313507c23283b15f06dbbae86451cf",
    "EuStockMarkets.csv": "aa5513f3ca55daf51294accae9a3f946e957ba53f405dc4fafb4e125e5c03818",
    "anorexia.csv": "6f44d7ac59e879a4b206c801a985630f21aef7feb025a94d0f98444a93d80ccd",
}


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


def read(name: str) -> list[dict[str, str]]:
    with open(fetch(name)) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Tools quoted in the article
# ---------------------------------------------------------------------------

def inverse_mills(z):
    """E[Z | Z > z] for a standard normal Z: phi(z) / (1 - Phi(z))."""
    return stats.norm.pdf(z) / stats.norm.sf(z)


def expected_regression(rho: float, top_share: float, sd: float = 1.0) -> tuple[float, float]:
    """For two measurements with equal spread sd and correlation rho, select the units whose first
    measurement is in the top `top_share` of the population. Returns the selected group's mean deviation on
    the first measurement and the amount by which its second measurement is expected to fall back, both in
    the units of sd: sd * lambda(z) and sd * (1 - rho) * lambda(z), with z the selection cut-off."""
    z = stats.norm.isf(top_share)
    lam = inverse_mills(z)
    return sd * lam, sd * (1 - rho) * lam


def kelley(x, reliability: float, mean: float):
    """Kelley's estimate of a true score from an observed score: the observed deviation shrunk towards the
    group mean by the reliability."""
    return mean + reliability * (np.asarray(x) - mean)


def spearman_brown(reliability: float, k: int) -> float:
    """Reliability of the average of k parallel measurements."""
    return k * reliability / (1 + (k - 1) * reliability)


def ols(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    cov = np.linalg.inv(X.T @ X) * (resid @ resid) / (n - k)
    return beta, np.sqrt(np.diag(cov))


def local_linear_jump(x, y, cutoff, bandwidth):
    """Difference at the cut-off between linear fits to the points within `bandwidth` on each side
    (uniform kernel), with its standard error."""
    left = (x < cutoff) & (x >= cutoff - bandwidth)
    right = (x >= cutoff) & (x < cutoff + bandwidth)
    fits = []
    for m in (left, right):
        b, se = ols(np.column_stack([np.ones(m.sum()), x[m] - cutoff]), y[m])
        fits.append((b[0], se[0]))
    return fits[1][0] - fits[0][0], math.sqrt(fits[0][1] ** 2 + fits[1][1] ** 2)


SCREENING = (140.0, 15.0, 0.6, 160.0, -5.0)  # population mean, SD, reliability, selection cut-off, true effect


def three_sources(rng, n: int = 20_000, T: int = 12, phi: float = 0.7):
    """Panels of n units over T occasions under three models, each with unit variance, and a random walk."""
    trait = rng.normal(0, 1, n)
    state = np.zeros((n, T))
    state[:, 0] = rng.normal(0, 1, n)
    for k in range(1, T):
        state[:, k] = phi * state[:, k - 1] + rng.normal(0, math.sqrt(1 - phi ** 2), n)
    noise = rng.normal(0, 1, (n, T))
    walk = np.cumsum(rng.normal(0, 1, (n, T + 50)), axis=1)[:, 50:]
    models = {
        "stable trait + measurement error (reliability 0.5)": (trait[:, None] + noise) / math.sqrt(2),
        "transient state, AR(1) with coefficient 0.7": state,
        "trait (0.4) + AR(1) state (0.4) + error (0.2)": math.sqrt(0.4) * trait[:, None] + math.sqrt(0.4) * state
                                                          + math.sqrt(0.2) * noise,
    }
    return models, walk


def nonnormal_pairs(rng, n: int = 2_000_000):
    """Pairs of measurements sharing a component plus independent normal noise of equal variance (reliability
    0.5): a lognormal shared component, a Student t(3) shared component, a normal component with t(3) noise,
    and two groups with means -2 and +2."""
    out = {}
    for name, s in (("lognormal (skewed)", np.exp(rng.normal(0, 0.8, n))),
                    ("Student t with 3 df (heavy tails)", rng.standard_t(3, n))):
        out[name] = (s + rng.normal(0, s.std(), n), s + rng.normal(0, s.std(), n))
    # the reverse: a normal shared component with heavy-tailed noise of the same variance (t with 3 df has variance 3)
    s = rng.normal(0, 1, n)
    out["normal component, heavy-tailed noise"] = (s + rng.standard_t(3, n) / math.sqrt(3), s + rng.standard_t(3, n) / math.sqrt(3))
    g = rng.random(n) < 0.5
    t = np.where(g, 2.0, -2.0) + rng.normal(0, 1, n)
    out["two groups"] = (t + rng.normal(0, 1, n), t + rng.normal(0, 1, n), g)
    return out


def screening_designs(rng, reps: int, n: int):
    """Screen n people on one measurement, select those above the cut-off, and estimate the effect of a
    treatment given to the selected with five designs; returns the estimates from each replication."""
    mu, sd, rel, cut, effect = SCREENING
    res = {k: [] for k in ("naive", "rct", "predicted", "second baseline", "rd")}
    for _ in range(reps):
        t = mu + rng.normal(0, sd * math.sqrt(rel), n)
        m1 = t + rng.normal(0, sd * math.sqrt(1 - rel), n)
        m2 = t + rng.normal(0, sd * math.sqrt(1 - rel), n)
        m3 = t + rng.normal(0, sd * math.sqrt(1 - rel), n)
        sel = m1 > cut
        # naive: treat everyone selected, compare the follow-up with the screening value
        res["naive"].append(np.mean(m3[sel] + effect) - np.mean(m1[sel]))
        # randomised: half of the selected treated, the other half followed up untreated
        arm = rng.random(n) < 0.5
        res["rct"].append(np.mean(m3[sel & arm] + effect) - np.mean(m3[sel & ~arm]))
        # predicted regression from a known reliability, subtracted from the naive change
        res["predicted"].append(np.mean(m3[sel] + effect) - (mu + rel * (np.mean(m1[sel]) - mu)))
        # second baseline: compare the follow-up with a second, independent pre-treatment measurement
        res["second baseline"].append(np.mean(m3[sel] + effect) - np.mean(m2[sel]))
        # regression discontinuity: everyone followed up, treatment above the cut-off
        y = m3 + np.where(sel, effect, 0.0)
        res["rd"].append(local_linear_jump(m1, y, cut, 10.0)[0])
    return {k: np.array(v) for k, v in res.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEED + 1)

    header("1. THE MEASUREMENT MODEL: STABLE COMPONENT PLUS NOISE")
    n = 1_000_000
    for rel in (0.5, 0.8):
        t = rng.normal(0, math.sqrt(rel), n)
        x = t + rng.normal(0, math.sqrt(1 - rel), n)
        y = t + rng.normal(0, math.sqrt(1 - rel), n)
        r = np.corrcoef(x, y)[0, 1]
        b = np.polyfit(x, y, 1)[0]
        m = x > 2
        print(f"  reliability {rel}: simulated correlation of two measurements {r:.4f}; slope of second on first {b:.4f};"
              f" units with first measurement above 2 SD ({m.mean():.4f} of them): first {x[m].mean():.4f},"
              f" second {y[m].mean():.4f}, true component {t[m].mean():.4f}")

    header("2. THREE SOURCES OF IMPERFECT CORRELATION, AND A RANDOM WALK")
    phi = 0.7
    models, walk = three_sources(np.random.default_rng(SEED + 2))
    print("  correlation between occasion 1 and occasion 1 + k:")
    for name, z in models.items():
        cors = [np.corrcoef(z[:, 0], z[:, k])[0, 1] for k in (1, 2, 3, 5, 10)]
        print(f"    {name:<50} k = 1, 2, 3, 5, 10: " + ", ".join(f"{c:.3f}" for c in cors))
    z = models["trait (0.4) + AR(1) state (0.4) + error (0.2)"]
    c1, c2, c3 = (np.corrcoef(z[:, 0], z[:, k])[0, 1] for k in (1, 2, 3))
    phi_hat = (c3 - c2) / (c2 - c1)
    s_hat = (c2 - c1) / (phi_hat ** 2 - phi_hat)
    tau_hat = c1 - s_hat * phi_hat
    print(f"    fitting trait share + state share x phi^k to the lag-1, 2 and 3 correlations of the mixture: phi {phi_hat:.3f},"
          f" state share {s_hat:.3f}, trait share {tau_hat:.3f}, error share {1 - s_hat - tau_hat:.3f} (true 0.7, 0.4, 0.4, 0.2);"
          f" relative error of the trait share {(tau_hat - 0.4) / 0.4:+.3f}")
    print(f"    theory for the mixture: 0.4 + 0.4 x 0.7^k = " + ", ".join(f"{0.4 + 0.4 * phi ** k:.3f}" for k in (1, 2, 3, 5, 10)))
    print("  random walk observed from step 51 on (no stationary mean):")
    for k in (1, 5, 10):
        b_fwd = np.polyfit(walk[:, 0], walk[:, k], 1)[0]
        b_bwd = np.polyfit(walk[:, k], walk[:, 0], 1)[0]
        print(f"    k = {k:>2}: slope of later on earlier {b_fwd:.4f}; slope of earlier on later {b_bwd:.4f};"
              f" SD earlier {walk[:, 0].std():.3f}, later {walk[:, k].std():.3f}; correlation {np.corrcoef(walk[:, 0], walk[:, k])[0, 1]:.4f}")

    rng = np.random.default_rng(SEED + 3)
    header("3. HOW MUCH REGRESSION A SELECTION PRODUCES (standard normal scale, equal spreads)")
    print("  selected share: mean of the selected on the first measurement; expected fall on the second for correlation"
          " 0.3 / 0.5 / 0.7 / 0.9")
    for q in (0.5, 0.2, 0.1, 0.05, 0.01):
        sel, _ = expected_regression(0.5, q)
        falls = [expected_regression(r, q)[1] for r in (0.3, 0.5, 0.7, 0.9)]
        print(f"    top {q:.0%}: selected mean {sel:.3f} SD; fall " + " / ".join(f"{f:.3f}" for f in falls)
              + " SD; share of the selected deviation lost " + " / ".join(f"{1 - r:.1f}" for r in (0.3, 0.5, 0.7, 0.9)))
    sel5, fall5 = expected_regression(0.3, 0.05)
    print(f"  selection on a score, outcome correlated 0.3 with it, top 5%: score deviation {sel5:.3f} SD; expected outcome"
          f" deviation {sel5 - fall5:.3f} SD; shortfall {fall5:.3f} SD")
    n = 2_000_000
    rho = 0.6
    x = rng.normal(0, 1, n)
    y = rho * x + math.sqrt(1 - rho ** 2) * rng.normal(0, 1, n)
    for q in (0.2, 0.05):
        c = np.quantile(x, 1 - q)
        m = x > c
        sel, fall = expected_regression(rho, q)
        print(f"  simulation, correlation {rho}, top {q:.0%}: selected mean {x[m].mean():.4f} (formula {sel:.4f});"
              f" fall {x[m].mean() - y[m].mean():.4f} (formula {fall:.4f})")
    print("  example in natural units: a measurement with SD 15 and test-retest correlation 0.6, top 10% selected:"
          f" expected fall {expected_regression(0.6, 0.10, 15)[1]:.2f} units from a selected mean"
          f" {expected_regression(0.6, 0.10, 15)[0]:.2f} above the population mean")
    print("  reducing the regression by design (reliability of a single measurement 0.6, top 10% selected):")
    lam = inverse_mills(stats.norm.isf(0.10))
    for k in (1, 2, 3, 5):
        rk = spearman_brown(0.6, k)
        sel_dev = lam * math.sqrt(0.6 / rk)
        print(f"    select on the mean of {k} measurements: reliability {rk:.3f}; selected deviation {sel_dev:.3f} SD of a single"
              f" measurement; expected fall on a new measurement {(1 - rk) * sel_dev:.3f} SD ({1 - rk:.3f} of the selected deviation)")
    t = rng.normal(0, math.sqrt(0.6), n)
    mean3 = t + rng.normal(0, math.sqrt(0.4), (3, n)).mean(axis=0)
    new = t + rng.normal(0, math.sqrt(0.4), n)
    s3 = mean3 > np.quantile(mean3, 0.9)
    print(f"    simulation, mean of 3: selected {mean3[s3].mean():.4f}, new measurement {new[s3].mean():.4f},"
          f" fall {mean3[s3].mean() - new[s3].mean():.4f}")
    # two-baseline design: select on measurement 1, use measurement 2 as the baseline, compare with 3
    rel = 0.6
    t = rng.normal(0, math.sqrt(rel), n)
    m1, m2, m3 = (t + rng.normal(0, math.sqrt(1 - rel), n) for _ in range(3))
    sel = m1 > np.quantile(m1, 0.9)
    print(f"    select on measurement 1 (top 10%): measurement 1 {m1[sel].mean():.4f}, measurement 2 {m2[sel].mean():.4f},"
          f" measurement 3 {m3[sel].mean():.4f}; change 1 to 3 {m3[sel].mean() - m1[sel].mean():+.4f},"
          f" change 2 to 3 {m3[sel].mean() - m2[sel].mean():+.4f}")

    print("  correlation between the first measurement and the change, with no change in anyone (equal spreads):")
    for rel in (0.5, 0.8):
        t = rng.normal(0, math.sqrt(rel), n)
        a = t + rng.normal(0, math.sqrt(1 - rel), n)
        b = t + rng.normal(0, math.sqrt(1 - rel), n)
        print(f"    reliability {rel}: corr(first, change) {np.corrcoef(a, b - a)[0, 1]:+.4f} (theory {-math.sqrt((1 - rel) / 2):+.4f});"
              f" Oldham's corr(mean of the two, change) {np.corrcoef((a + b) / 2, b - a)[0, 1]:+.4f}")

    header("4. BEYOND THE NORMAL: SKEWED, HEAVY-TAILED AND MIXED POPULATIONS")
    rng = np.random.default_rng(SEED + 4)
    pairs = nonnormal_pairs(rng)
    for name in ("lognormal (skewed)", "Student t with 3 df (heavy tails)", "normal component, heavy-tailed noise"):
        x, y = pairs[name]
        for q in (0.1, 0.01):
            c = np.quantile(x, 1 - q)
            m = x > c
            print(f"  {name:<33} top {q:.0%} on the first: first {x[m].mean() - x.mean():+.3f}, second {y[m].mean() - y.mean():+.3f}"
                  f" (deviations from the means; kept {(y[m].mean() - y.mean()) / (x[m].mean() - x.mean()):.3f}); correlation {np.corrcoef(x, y)[0, 1]:.3f}")
    x, y = pairs["lognormal (skewed)"]
    m = x < np.quantile(x, 0.1)
    print(f"  lognormal (skewed)                bottom 10% on the first: first {x[m].mean() - x.mean():+.3f}, second {y[m].mean() - y.mean():+.3f}"
          f" (kept {(y[m].mean() - y.mean()) / (x[m].mean() - x.mean()):.3f})")
    # two groups with different means: regression towards the group mean
    x, y, g = pairs["two groups"]
    print(f"  two groups with means -2 and +2 (within-group reliability 0.5), overall correlation {np.corrcoef(x, y)[0, 1]:.3f}:")
    for lo, hi in ((1.5, 2.5), (3.5, 4.5), (-0.5, 0.5)):
        m = (x > lo) & (x < hi)
        print(f"    first measurement in ({lo}, {hi}): first {x[m].mean():+.3f}, second {y[m].mean():+.3f};"
              f" share from the upper group {g[m].mean():.3f}")
    b = np.polyfit(x, y, 1)
    for lo, hi in ((1.5, 2.5), (3.5, 4.5)):
        m = (x > lo) & (x < hi)
        print(f"    single line's prediction for first measurement in ({lo}, {hi}): {b[1] + b[0] * x[m].mean():+.3f}")
    b = b[0]
    print(f"    single-line slope {b:.3f}; within-group slope "
          f"{np.polyfit(x[g] - 2, y[g] - 2, 1)[0]:.3f}")

    rng = np.random.default_rng(SEED + 5)
    header("5. SHRINKAGE: KELLEY'S FORMULA")
    n = 1_000
    rel = 0.5
    reps = 2_000
    mse_raw, mse_k, overlap_raw, overlap_k = [], [], [], []
    for _ in range(reps):
        t = rng.normal(0, math.sqrt(rel), n)
        x = t + rng.normal(0, math.sqrt(1 - rel), n)
        k = kelley(x, rel, 0.0)
        mse_raw.append(np.mean((x - t) ** 2))
        mse_k.append(np.mean((k - t) ** 2))
        top_true = set(np.argsort(-t)[:50])
        overlap_raw.append(len(top_true & set(np.argsort(-x)[:50])) / 50)
        overlap_k.append(len(top_true & set(np.argsort(-k)[:50])) / 50)
    print(f"  reliability {rel}, {n} units, {reps} simulations: mean squared error of the observed score as an estimate of"
          f" the true score {np.mean(mse_raw):.4f}; of Kelley's estimate {np.mean(mse_k):.4f} (theory {1 - rel:.4f} and"
          f" {rel * (1 - rel):.4f}); ratio {np.mean(mse_k) / np.mean(mse_raw):.3f}")
    print(f"  share of the true top 50 found in the observed top 50: {np.mean(overlap_raw):.3f}; in the top 50 by Kelley's"
          f" estimate: {np.mean(overlap_k):.3f}")
    # two groups with different means and the same reliability: shrinking to the wrong mean
    g = rng.random(200_000) < 0.5
    mu = np.where(g, 1.0, -1.0)
    t = mu + rng.normal(0, 1, 200_000)
    x = t + rng.normal(0, 1, 200_000)
    within_rel = 0.5
    overall_rel = np.var(t) / np.var(x)
    k_overall = kelley(x, overall_rel, x.mean())
    k_group = np.where(g, kelley(x, within_rel, 1.0), kelley(x, within_rel, -1.0))
    print(f"  two groups with means -1 and +1: overall reliability {overall_rel:.3f}; mean squared error shrinking to the"
          f" overall mean {np.mean((k_overall - t) ** 2):.4f}, to each unit's group mean {np.mean((k_group - t) ** 2):.4f},"
          f" raw {np.mean((x - t) ** 2):.4f}")
    sel = x > np.quantile(x, 0.95)
    print(f"    top 5% on the observed score: share from the upper group {g[sel].mean():.3f}; mean true score {t[sel].mean():.3f};"
          f" Kelley to overall mean predicts {k_overall[sel].mean():.3f}, to group means {k_group[sel].mean():.3f}")

    print("  rates from units of different sizes: 2,000 units, true rates Beta(10, 90) (mean 0.10), sizes log-uniform 20 to 2,000,")
    print("  two periods with independent binomial counts; the top 5% of units on the period-1 rate:")
    k_units = 2_000
    p_true = rng.beta(10, 90, k_units)
    size = np.round(np.exp(rng.uniform(np.log(20), np.log(2000), k_units))).astype(int)
    r1 = rng.binomial(size, p_true) / size
    r2 = rng.binomial(size, p_true) / size
    top = r1 >= np.quantile(r1, 0.95)
    small = size < np.quantile(size, 0.25)
    print(f"    top 5% ({top.sum()} units): share from the smallest quarter of units {small[top].mean():.3f}, from the largest quarter"
          f" {(size >= np.quantile(size, 0.75))[top].mean():.3f}; median size {np.median(size[top]):.0f} against {np.median(size):.0f} overall")
    print(f"    their period-1 rate {r1[top].mean():.4f}, true rate {p_true[top].mean():.4f}, period-2 rate {r2[top].mean():.4f};"
          f" population mean {p_true.mean():.4f}")
    var_true = np.var(p_true)  # the between-unit variance is taken as known here (in practice it is estimated)
    rel_u = var_true / (var_true + r1.mean() * (1 - r1.mean()) / size)
    shrunk = r1.mean() + rel_u * (r1 - r1.mean())
    top_s = shrunk >= np.quantile(shrunk, 0.95)
    print(f"    shrinking each unit by its own reliability (from {rel_u.min():.2f} for the smallest to {rel_u.max():.2f} for the largest):"
          f" top 5% by shrunk rate has median size {np.median(size[top_s]):.0f}, true rate {p_true[top_s].mean():.4f},"
          f" period-2 rate {r2[top_s].mean():.4f}; share of the true top 5% found {np.mean(top_s & (p_true >= np.quantile(p_true, 0.95))) / 0.05:.3f}"
          f" against {np.mean(top & (p_true >= np.quantile(p_true, 0.95))) / 0.05:.3f} by raw rate")

    header("6. REAL SERIES: NEW YORK TEMPERATURE ANOMALIES AND THE DAX")
    w = [r for r in read("weather.csv") if r["origin"] == "JFK" and r["temp"] not in ("", "NA")]
    daily: dict[tuple[int, int], list[float]] = {}
    for r in w:
        daily.setdefault((int(r["month"]), int(r["day"])), []).append(float(r["temp"]))
    keys = sorted(daily)
    temp = np.array([np.mean(daily[k]) for k in keys])
    doy = np.array([datetime.date(2013, m, d).timetuple().tm_yday for m, d in keys], float)
    X = np.column_stack([np.ones(len(doy))] + [f(2 * np.pi * h * doy / 365) for h in (1, 2) for f in (np.sin, np.cos)])
    beta, _ = ols(X, temp)
    anom = temp - X @ beta
    consecutive = np.diff(doy) == 1
    print(f"  JFK, 2013: {len(temp)} days with readings (days of the year {int(doy.min())}-{int(doy.max())}, {int((~consecutive).sum())} gaps);"
          f" daily mean temperature minus a two-harmonic seasonal curve; SD of the anomaly {anom.std():.2f} F")
    acf = []
    for k in range(1, 8):
        ok = np.array([doy[i + k] - doy[i] == k for i in range(len(doy) - k)])
        a0, ak = anom[:-k][ok], anom[k:][ok]
        acf.append(np.corrcoef(a0, ak)[0, 1])
    print("  correlation of the anomaly with itself k days later, k = 1..7: " + ", ".join(f"{c:+.3f}" for c in acf))
    print("  AR(1) prediction from the lag-1 value, k = 1..7: " + ", ".join(f"{acf[0] ** k:+.3f}" for k in range(1, 8)))
    hot = anom > np.quantile(anom, 0.95)
    idx = np.where(hot)[0]
    print(f"  the hottest 5% of days relative to the season ({hot.sum()} days): mean anomaly {anom[hot].mean():+.2f} F;"
          " mean anomaly k days later (k = 1..5): " + ", ".join(
              f"{np.mean([anom[i + k] for i in idx if i + k < len(anom) and doy[i + k] - doy[i] == k]):+.2f}" for k in range(1, 6)))
    cold = anom < np.quantile(anom, 0.05)
    idx = np.where(cold)[0]
    print(f"  the coldest 5% ({cold.sum()} days): mean anomaly {anom[cold].mean():+.2f} F; k days later (k = 1..5): " + ", ".join(
              f"{np.mean([anom[i + k] for i in idx if i + k < len(anom) and doy[i + k] - doy[i] == k]):+.2f}" for k in range(1, 6)))
    e = read("EuStockMarkets.csv")
    dax = np.array([float(r["DAX"]) for r in e])
    b_lvl = np.polyfit(dax[:-1], dax[1:], 1)
    ret = np.diff(np.log(dax))
    b_ret = np.polyfit(ret[:-1], ret[1:], 1)
    big = ret[:-1] > np.quantile(ret[:-1], 0.95)
    print(f"  DAX, {len(dax)} trading days 1991-1998: slope of the next close on today's close {b_lvl[0]:.4f};"
          f" lag-1 correlation of daily log returns {np.corrcoef(ret[:-1], ret[1:])[0, 1]:+.4f}, slope {b_ret[0]:+.4f}")
    nxt = ret[1:][big]
    print(f"    after the 5% largest daily gains: mean gain that day {100 * ret[:-1][big].mean():+.3f}%, next day"
          f" {100 * nxt.mean():+.3f}% (standard error {100 * nxt.std(ddof=1) / math.sqrt(len(nxt)):.3f}%, {len(nxt)} days);"
          f" mean daily return overall {100 * ret.mean():+.3f}%")
    hi_lvl = dax[:-20] > np.quantile(dax[:-20], 0.9)
    print(f"    days (among the first {len(dax) - 20}) with the close in its top 10%: mean close {dax[:-20][hi_lvl].mean():.0f};"
          f" 20 trading days later {dax[20:][hi_lvl].mean():.0f}; series mean {dax.mean():.0f}")

    header("7. SEPARATING REGRESSION FROM AN EFFECT")
    # Screening simulation: a measurement with mean 140, SD 15 and reliability 0.6; units above 160 are selected.
    mu, sd, rel, cut, effect = SCREENING
    reps, n = 1_000, 20_000
    res = screening_designs(np.random.default_rng(SEED + 7), reps, n)
    print(f"  screening simulation: mean {mu}, SD {sd}, reliability {rel}, selected above {cut} ({stats.norm.sf((cut - mu) / sd):.3f}"
          f" of the population), true effect of treatment {effect}; {reps} replications of {n:,} people")
    for k, v in res.items():
        v = np.array(v)
        print(f"    {k:<16} mean estimate {v.mean():+.3f} (SD across replications {v.std():.3f}; Monte Carlo SE of the mean"
              f" {v.std() / math.sqrt(len(v)):.3f}); ratio to the true effect {v.mean() / effect:.2f}")
    sel_mean = mu + sd * inverse_mills((cut - mu) / sd)
    print(f"    formula: selected mean {sel_mean:.2f}, {sel_mean - mu:.2f} above the population mean; expected regression"
          f" {(1 - rel) * (sel_mean - mu):.3f}; an error of 0.05 in the assumed reliability shifts the predicted regression by"
          f" {0.05 * (sel_mean - mu):.2f}")

    print("  subtracting a predicted regression with a reliability estimated from a pilot of m people measured twice:")
    prng = np.random.default_rng(SEED + 8)
    for m_pilot in (50, 100, 500, 2000):
        est = []
        for _ in range(500):
            t = mu + prng.normal(0, sd * math.sqrt(rel), 20_000)
            m1 = t + prng.normal(0, sd * math.sqrt(1 - rel), 20_000)
            m3 = t + prng.normal(0, sd * math.sqrt(1 - rel), 20_000)
            sel = m1 > cut
            tp = mu + prng.normal(0, sd * math.sqrt(rel), m_pilot)
            p1 = tp + prng.normal(0, sd * math.sqrt(1 - rel), m_pilot)
            p2 = tp + prng.normal(0, sd * math.sqrt(1 - rel), m_pilot)
            r_hat = np.corrcoef(p1, p2)[0, 1]
            est.append(np.mean(m3[sel] + effect) - (mu + r_hat * (np.mean(m1[sel]) - mu)))
        est = np.array(est)
        print(f"    pilot of {m_pilot:>4}: mean estimate {est.mean():+.3f}, SD across 500 replications {est.std():.3f}")
    an = read("anorexia.csv")
    treat = np.array([r["Treat"] for r in an])
    pre = np.array([float(r["Prewt"]) for r in an])
    post = np.array([float(r["Postwt"]) for r in an])
    print(f"  anorexia trial (Hand et al., 1994; MASS::anorexia): {len(pre)} patients, weights in pounds")
    for g in ("Cont", "CBT", "FT"):
        m = treat == g
        b, se = ols(np.column_stack([np.ones(m.sum()), pre[m]]), post[m] - pre[m])
        print(f"    {g:<4} n = {m.sum()}: before {pre[m].mean():.2f}, after {post[m].mean():.2f}, change {np.mean(post[m] - pre[m]):+.2f};"
              f" correlation before-after {np.corrcoef(pre[m], post[m])[0, 1]:+.3f}; slope of change on before {b[1]:+.3f} (SE {se[1]:.3f});"
              f" corr(before, change) {np.corrcoef(pre[m], post[m] - pre[m])[0, 1]:+.3f}; Oldham's corr(mean, change)"
              f" {np.corrcoef((pre[m] + post[m]) / 2, post[m] - pre[m])[0, 1]:+.3f}; SD before {pre[m].std(ddof=1):.2f}, after {post[m].std(ddof=1):.2f}"
              f" (ratio {post[m].std(ddof=1) / pre[m].std(ddof=1):.2f}); slope of after on before {b[1] + 1:.3f}")
    for g in ("CBT", "FT"):
        m = np.isin(treat, ["Cont", g])
        d = (treat[m] == g).astype(float)
        b_post, se_post = ols(np.column_stack([np.ones(m.sum()), d]), post[m])
        b_chg, se_chg = ols(np.column_stack([np.ones(m.sum()), d]), post[m] - pre[m])
        b_anc, se_anc = ols(np.column_stack([np.ones(m.sum()), d, pre[m] - pre[m].mean()]), post[m])
        print(f"    {g} against control: after only {b_post[1]:+.2f} (SE {se_post[1]:.2f}); change scores {b_chg[1]:+.2f} (SE {se_chg[1]:.2f});"
              f" ANCOVA {b_anc[1]:+.2f} (SE {se_anc[1]:.2f}), coefficient on the baseline {b_anc[2]:+.3f}")


if __name__ == "__main__":
    main()
