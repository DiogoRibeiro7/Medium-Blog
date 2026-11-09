"""Part 4 of the statistical-process-control series: designing a line's charts around a false-alarm budget.

Every number quoted in the article is printed by this script. Run:

    python budget_numbers.py > output.txt

The methods live in the companion module ../spcbudget (tested in ../tests). The script checks each method's
guarantee where it can be checked exactly, by simulation with normal and non-normal data, and then applies
the methods to the two public lines audited in Part 3. Their data are downloaded from pinned commits of
GitHub mirrors, checked against SHA-256 hashes (after converting CRLF line endings to LF) and cached in
part4/data/, which is not committed.

Sections
1. One chart, guaranteed: the limit width that guarantees a conditional in-control false-alarm probability
   with a stated probability, checked by simulation; its price in detection; the parametric bootstrap.
2. When the data are not normal: how often each guarantee holds under four distributions.
3. A line of charts: per-chart guarantees, a joint guarantee for the line, and Sidak limits.
4. How much in-control data a distribution-free guarantee needs.
5. The continuous-flow line: a budget of one false alarm per hour for 21 measurement series.
6. The SECOM line: a budget of one false alarm per 100 runs for 428 sensors.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import special, stats

HERE = Path(__file__).parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE.parent))
import spcbudget as sb  # noqa: E402

SEED = 20261109
CHILDREN = np.random.SeedSequence(SEED).spawn(16)
RNG = {name: np.random.default_rng(CHILDREN[i]) for i, name in enumerate(
    ["check", "ooc", "boot", "nonnormal", "npboot", "orderstat", "line", "line_ooc", "liveline",
     "secom", "spare1", "spare2", "spare3", "spare4", "spare5", "spare6"])}

FIG: dict = {}
P0 = 2 * stats.norm.sf(3.0)   # 0.0026998: nominal rate of a 3-sigma chart, in-control ARL 370.4


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def q3(x, qs=(10, 50, 90)) -> str:
    v = np.percentile(np.asarray(x, dtype=float), qs)
    return " / ".join(f"{t:.4g}" for t in v)


def fap_shift(z, s, k, delta):
    """Signal probability of limits mu_hat +/- k sigma_hat after a mean shift of delta sigma."""
    return special.ndtr(-(k * s + z - delta)) + special.ndtr(z - delta - k * s)


SOURCES = {
    "continuous_factory_process.csv": {
        "sha256": "5d1358bc03b3bc68d239eeccd9a5e5d5d48360b15b8ddacea369e1244647b02d",
        "urls": [
            "https://raw.githubusercontent.com/LeonardoMdSACode/Multi-stage_continuous-flow_manufacturing_process/"
            "97c447cfc87b4183609710513805843f0cfbe8b8/continuous_factory_process.csv",
            "https://raw.githubusercontent.com/hjeffreywang/"
            "Advanced-Prediction-of-Multi-stage-continuous-flow-manufacturing-process/"
            "52e34be815907b6700d2c2975e8f1aca4b508775/data/continuous_factory_process.csv",
        ],
    },
    "secom.data": {
        "sha256": "4503bdb42e63687d1c7586bdc1796df9a7cb5ae9cc9c0443d84957a446de3aaf",
        "urls": [
            "https://raw.githubusercontent.com/Aylin1/SECOM_Project/"
            "c6e3cc0a06b60155aafb6c06bb7cb39b7cf96d14/data/secom.data",
            "https://raw.githubusercontent.com/halinh1512/Predictive-Model-Semiconductor-SECOM/"
            "a4aff0133c2e2dce5311721fdb3874f7c3b956ba/Data/secom.data",
        ],
    },
    "secom_labels.data": {
        "sha256": "2fb2e8b5dfe77ba96ecf0960ff00f2ba9311795383a8342d4588a7cb37c816ef",
        "urls": [
            "https://raw.githubusercontent.com/Aylin1/SECOM_Project/"
            "c6e3cc0a06b60155aafb6c06bb7cb39b7cf96d14/data/secom_labels.data",
            "https://raw.githubusercontent.com/halinh1512/Predictive-Model-Semiconductor-SECOM/"
            "a4aff0133c2e2dce5311721fdb3874f7c3b956ba/Data/secom_labels.data",
        ],
    },
}


def fetch(name: str) -> Path:
    """Return the cached, hash-checked copy of a source file, downloading it if needed."""
    spec = SOURCES[name]
    DATA.mkdir(exist_ok=True)
    path = DATA / name
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]:
        return path
    for url in spec["urls"]:
        try:
            raw = urllib.request.urlopen(url, timeout=120).read()
        except Exception as exc:  # try the next mirror
            print(f"# download failed from {url}: {exc}", file=sys.stderr)
            continue
        body = raw.replace(b"\r\n", b"\n")
        if hashlib.sha256(body).hexdigest() == spec["sha256"]:
            path.write_bytes(body)
            return path
        print(f"# hash mismatch for {url}", file=sys.stderr)
    raise SystemExit(f"could not obtain a verified copy of {name}")


# ---------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------


def load_liveline():
    path = fetch("continuous_factory_process.csv")
    with path.open(newline="") as f:
        reader = csv.reader(f)
        cols = next(reader)
        rows = list(reader)
    stamps = [datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S") for r in rows]
    vals = np.array([[float(v) for v in r[1:]] for r in rows])
    t = np.array([(s - stamps[0]).total_seconds() for s in stamps])
    return cols, stamps, t, vals


def load_secom():
    X = np.loadtxt(fetch("secom.data"))
    labels, stamps = [], []
    for line in fetch("secom_labels.data").read_text().splitlines():
        if not line.strip():
            continue
        lab, stamp = re.match(r'\s*(-?1)\s+"([^"]+)"', line).groups()
        labels.append(int(lab))
        stamps.append(datetime.strptime(stamp, "%d/%m/%Y %H:%M:%S"))
    return X, np.array(labels), stamps



class Line:
    """The continuous-flow line, with dropouts (readings of exactly 0) and the two pauses set to NaN."""

    def __init__(self):
        self.cols, self.stamps, self.t, self.vals = load_liveline()
        c = {name: i - 1 for i, name in enumerate(self.cols)}   # vals has the time stamp removed
        self.c = c
        sp = self.vals[:, c["Stage1.Output.Measurement0.U.Setpoint"]]
        self.running = sp != 0
        self.minute = (self.t // 60).astype(int)
        self.nmin = int(self.minute.max()) + 1

    def raw(self, name: str) -> np.ndarray:
        return self.vals[:, self.c[name]]

    def meas(self, stage: int, k: int) -> np.ndarray:
        x = self.raw(f"Stage{stage}.Output.Measurement{k}.U.Actual").copy()
        x[(x == 0) | ~self.running] = np.nan
        return x

    def var(self, name: str) -> np.ndarray:
        x = self.raw(name).copy()
        x[~self.running] = np.nan
        return x

    def minute_means(self, x: np.ndarray, min_count: int = 10):
        mean = np.full(self.nmin, np.nan)
        sd = np.full(self.nmin, np.nan)
        cnt = np.zeros(self.nmin, dtype=int)
        for j in range(self.nmin):
            v = x[self.minute == j]
            v = v[~np.isnan(v)]
            cnt[j] = len(v)
            if len(v) >= min_count:
                mean[j] = v.mean()
                sd[j] = v.std(ddof=1)
        return mean, sd, cnt



# ---------------------------------------------------------------------------------------------------
# Section 1: one chart, guaranteed
# ---------------------------------------------------------------------------------------------------

KNOWN_ARL = {d: 1 / (stats.norm.sf(3 - d) + stats.norm.cdf(-3 - d)) for d in (0, 1, 2, 3)}


def phase1_draws(rng, m, n, nu=None):
    nu = m - 1 if nu is None else nu
    z = rng.normal(0.0, 1.0 / math.sqrt(m), n)
    s = np.sqrt(rng.chisquare(nu, n) / nu)
    return z, s


def section_one_chart() -> None:
    header("1. One chart, guaranteed")
    print("Individuals chart with limits xbar +/- k S from m in-control normal values. Target: conditional")
    print(f"false-alarm probability <= p0 = {P0:.6f}, i.e. conditional in-control ARL >= 370.4, with probability 1 - eps.")
    for m in (25, 50, 100, 250, 1000):
        k10, k05 = sb.guaranteed_k(m, P0, 0.10), sb.guaranteed_k(m, P0, 0.05)
        print(f"m {m:5d}: k(eps 0.10) {k10:.4f}, k(eps 0.05) {k05:.4f}; k = 3 meets the target with probability "
              f"{sb.guarantee_probability(3.0, m, P0):.3f}")
    kx = sb.guaranteed_k(25, P0, 0.10, nu=100)
    print(f"X-bar chart, 25 subgroups of 5, pooled S (nu = 100): k(eps 0.10) {kx:.4f}; k = 3 meets the target with "
          f"probability {sb.guarantee_probability(3.0, 25, P0, nu=100):.3f}")
    print("cross-check against Howe's (1969) approximation to the two-sided normal tolerance factor:")
    for n, cont, conf in ((10, 0.99, 0.95), (20, 0.95, 0.95), (100, 0.99, 0.95), (100, 1 - P0, 0.90)):
        howe = math.sqrt((n - 1) * (1 + 1 / n) * stats.norm.isf((1 - cont) / 2) ** 2 / stats.chi2.ppf(1 - conf, n - 1))
        ex = sb.guaranteed_k(n, 1 - cont, 1 - conf)
        print(f"   n {n}, content {cont:.4f}, confidence {conf:.2f}: exact {ex:.4f}, Howe {howe:.4f}, ratio {ex / howe:.4f}")
    for m in (50, 100):
        k = sb.guaranteed_k(m, P0, 0.10)
        z, s = phase1_draws(RNG["check"], m, 400_000)
        print(f"simulation check, m {m}, 400,000 Phase I samples: share meeting the target "
              f"{np.mean(sb.conditional_fap(z, s, k) <= P0):.4f} (designed 0.9000)")

    print("\nprice in detection: conditional ARL after a mean shift of delta sigma, median and 10/90% over")
    print("200,000 Phase I samples; known parameters: " + ", ".join(f"delta {d}: {KNOWN_ARL[d]:.1f}" for d in (0, 1, 2, 3)))
    fig_cost = {}
    for m in (50, 100, 250):
        z, s = phase1_draws(RNG["ooc"], m, 200_000)
        for label, k in (("k = 3", 3.0), ("guaranteed, eps 0.10", sb.guaranteed_k(m, P0, 0.10)),
                         ("guaranteed, eps 0.05", sb.guaranteed_k(m, P0, 0.05))):
            parts = []
            for d in (0, 1, 2, 3):
                arl = 1 / fap_shift(z, s, k, d)
                parts.append(f"delta {d}: {np.median(arl):.1f} ({np.percentile(arl, 10):.1f}-{np.percentile(arl, 90):.1f})")
            print(f"m {m:4d}, {label:21s} k {k:.3f}: " + "; ".join(parts))
    grid = np.unique(np.round(np.geomspace(20, 2000, 25)).astype(int))
    k10 = [sb.guaranteed_k(int(m), P0, 0.10) for m in grid]
    k05 = [sb.guaranteed_k(int(m), P0, 0.05) for m in grid]
    med1 = {"k3": [], "g10": []}
    for m, kg in zip(grid, k10):
        z, s = phase1_draws(RNG["ooc"], int(m), 50_000)
        med1["k3"].append(float(np.median(1 / fap_shift(z, s, 3.0, 1))))
        med1["g10"].append(float(np.median(1 / fap_shift(z, s, kg, 1))))
        fig_cost.setdefault("ic_k3_p10", []).append(float(np.percentile(1 / sb.conditional_fap(z, s, 3.0), 10)))
        fig_cost.setdefault("ic_g10_p10", []).append(float(np.percentile(1 / sb.conditional_fap(z, s, kg), 10)))
    FIG["one_chart"] = {"m": grid.tolist(), "k10": k10, "k05": k05, "med1_k3": med1["k3"], "med1_g10": med1["g10"],
                        "known1": KNOWN_ARL[1], **fig_cost}

    print("\nparametric bootstrap (Gandy and Kvaloy 2013) on simulated normal Phase I samples, m = 100, eps 0.10:")
    rng = RNG["boot"]
    ks, cover = [], []
    for _ in range(300):
        x = rng.normal(0.0, 1.0, 100)
        k = sb.bootstrap_guaranteed_k(x, P0, 0.10, rng, B=4000)
        ks.append(k)
        cover.append(sb.conditional_fap((x.mean()) / 1.0, x.std(ddof=1), k) <= P0)
    print(f"   300 samples, B = 4,000: k 10/50/90% {q3(ks)}; share meeting the target {np.mean(cover):.3f}; "
          f"exact k {sb.guaranteed_k(100, P0, 0.10):.4f}")


# ---------------------------------------------------------------------------------------------------
# Section 2: when the data are not normal
# ---------------------------------------------------------------------------------------------------

LN_SIGMA = 0.5
LN_MEAN = math.exp(LN_SIGMA ** 2 / 2)
LN_SD = math.sqrt((math.exp(LN_SIGMA ** 2) - 1) * math.exp(LN_SIGMA ** 2))
DISTS = {
    "normal": (lambda rng, size: rng.standard_normal(size), stats.norm.cdf),
    "Student t5": (lambda rng, size: rng.standard_t(5, size) * math.sqrt(3 / 5),
                   lambda x: stats.t.cdf(np.asarray(x) / math.sqrt(3 / 5), 5)),
    "Laplace": (lambda rng, size: rng.laplace(0.0, 1 / math.sqrt(2), size),
                lambda x: stats.laplace.cdf(x, scale=1 / math.sqrt(2))),
    "lognormal": (lambda rng, size: (np.exp(LN_SIGMA * rng.standard_normal(size)) - LN_MEAN) / LN_SD,
                  lambda x: stats.norm.cdf(np.log(np.maximum(LN_MEAN + LN_SD * np.asarray(x), 1e-300)) / LN_SIGMA)),
}


def section_nonnormal() -> None:
    header("2. When the data are not normal")
    print("Each distribution is standardized to mean 0 and variance 1. 'share' = share of Phase I samples whose")
    print(f"limits have a conditional false-alarm probability <= p0 under the true distribution (target 0.90).")
    print("Tail probability beyond 3 SD with known parameters: "
          + ", ".join(f"{n} {F(-3.0) + 1 - F(3.0):.5f}" for n, (_, F) in DISTS.items()))
    fig = {}
    for m in (100, 250, 1000):
        kg = sb.guaranteed_k(m, P0, 0.10)
        line = []
        for name, (draw, F) in DISTS.items():
            x = draw(RNG["nonnormal"], (20_000, m))
            mu, sd = x.mean(axis=1), x.std(axis=1, ddof=1)
            p3 = F(mu - 3 * sd) + 1 - F(mu + 3 * sd)
            pg = F(mu - kg * sd) + 1 - F(mu + kg * sd)
            line.append(f"{name}: k=3 {np.mean(p3 <= P0):.3f}, guaranteed {np.mean(pg <= P0):.3f} (median p {np.median(pg):.5f})")
            fig.setdefault(name, {})[f"normal_theory_{m}"] = float(np.mean(pg <= P0))
        print(f"m {m}, normal-theory guaranteed k {kg:.4f}, 20,000 samples: " + "; ".join(line))

    print("\nnonparametric bootstrap (resampling the Phase I values), m = 100, eps 0.10, B = 1,000, 300 samples:")
    for name, (draw, F) in DISTS.items():
        ks, cov = [], []
        for _ in range(300):
            x = draw(RNG["npboot"], 100)
            k = sb.bootstrap_guaranteed_k(x, P0, 0.10, RNG["npboot"], B=1000, kind="nonparametric")
            ks.append(k)
            mu, sd = x.mean(), x.std(ddof=1)
            cov.append(F(mu - k * sd) + 1 - F(mu + k * sd) <= P0)
        print(f"   {name}: k 10/50/90% {q3(ks)}; share meeting the target {np.mean(cov):.3f}")
        fig[name]["np_bootstrap_100"] = float(np.mean(cov))

    n2 = sb.min_calibration_size(P0, 0.10, rank=2)
    print(f"\ndistribution-free two-sided limits: the smallest and largest of n in-control values. The probability")
    print(f"outside them is Beta(2, n - 1) for any continuous distribution; the smallest n meeting the target is {n2}.")
    for name, (draw, F) in DISTS.items():
        x = draw(RNG["orderstat"], (20_000, n2))
        p = F(x.min(axis=1)) + 1 - F(x.max(axis=1))
        print(f"   {name}: share meeting the target {np.mean(p <= P0):.3f} (20,000 samples of {n2})")
        fig[name][f"order_stat_{n2}"] = float(np.mean(p <= P0))
    FIG["nonnormal"] = {"cover": fig, "n2": n2}


# ---------------------------------------------------------------------------------------------------
# Section 3: a line of charts
# ---------------------------------------------------------------------------------------------------


def section_line() -> None:
    header("3. A line of charts")
    m, eps, aL = 100, 0.10, P0
    print(f"K independent individuals charts, each with its own Phase I of m = {m}; line budget: false-alarm")
    print(f"probability of the line per time point <= {aL:.6f} (line ARL >= 370.4) with probability {1 - eps:.2f}.")
    print("Designs: Sidak limits with known-parameter widths; a guarantee for every chart at the Sidak rate with")
    print("eps_chart = 1 - (1 - eps)^(1/K); and a single joint guarantee for the line (Monte Carlo, 4,000 lines).")
    rows = {}
    zo, so = phase1_draws(RNG["line_ooc"], m, 200_000)
    for K in (1, 10, 50, 100, 500):
        rng = RNG["line"]
        z = rng.normal(0.0, 1.0 / math.sqrt(m), (4000, K))
        s = np.sqrt(rng.chisquare(m - 1, (4000, K)) / (m - 1))
        k_sidak = sb.sidak_k(aL, K)
        k_each = sb.guaranteed_k(m, sb.sidak_alpha(aL, K), 1 - (1 - eps) ** (1 / K))
        k_line = sb.line_guaranteed_k(m, K, aL, eps, RNG["line"], reps=4000)
        out = []
        for label, k in (("Sidak", k_sidak), ("every chart", k_each), ("joint line", k_line)):
            pl = -np.expm1(np.sum(np.log1p(-sb.conditional_fap(z, s, k)), axis=1))
            arl2 = np.median(1 / fap_shift(zo, so, k, 2.0))
            out.append(f"{label} k {k:.3f}: share meeting {np.mean(pl <= aL):.3f}, mean line rate / budget "
                       f"{np.mean(pl) / aL:.2f}, median ARL of a chart after a 2-sigma shift {arl2:.2f}")
            rows.setdefault(label, {"K": [], "k": [], "arl2": []})
            rows[label]["K"].append(K)
            rows[label]["k"].append(float(k))
            rows[label]["arl2"].append(float(arl2))
        print(f"K {K:4d}:")
        for o in out:
            print("   " + o)
    FIG["line"] = rows


# ---------------------------------------------------------------------------------------------------
# Section 4: how much data a distribution-free guarantee needs
# ---------------------------------------------------------------------------------------------------


def section_sizes() -> None:
    header("4. In-control data needed for a distribution-free guarantee")
    print("Threshold = the largest of n exchangeable in-control scores (one-sided). Its exceedance probability is")
    print("Beta(1, n), so it is <= alpha with probability 1 - eps once n >= log(eps) / log(1 - alpha) (Wilks 1941).")
    cases = [
        ("one 3-sigma-equivalent chart, p0", P0),
        ("a line budget of 1 alarm in 100 runs", 0.01),
        ("one of 428 sensors under a 1-in-100 line budget (Sidak)", sb.sidak_alpha(0.01, 428)),
        ("a line budget of 1 alarm per hour, scored once a minute", 1 / 60),
        ("one of 21 series under a 1-per-hour line budget, once a minute (Sidak)", sb.sidak_alpha(1 / 60, 21)),
    ]
    for label, a in cases:
        print(f"   {label}: alpha {a:.4g}; n for eps 0.10: {sb.min_calibration_size(a, 0.10)}, eps 0.05: "
              f"{sb.min_calibration_size(a, 0.05)}; with the second largest as threshold (eps 0.10): "
              f"{sb.min_calibration_size(a, 0.10, rank=2)}")
    print("\nworked example: 50 charts, one point every 10 minutes, Phase I of 100 points per chart; budget one false")
    print("alarm per 8-hour shift for the line, i.e. a line false-alarm probability of 1/48 per sampling time")
    aL, K, m = 1 / 48, 50, 100
    ac = sb.sidak_alpha(aL, K)
    k_s = sb.sidak_k(aL, K)
    k_e = sb.guaranteed_k(m, ac, 1 - 0.9 ** (1 / K))
    k_j = sb.line_guaranteed_k(m, K, aL, 0.10, RNG["spare1"], reps=4000)
    zo, so = phase1_draws(RNG["spare2"], m, 200_000)
    print(f"   per-chart Sidak rate {ac:.3e}; widths: Sidak {k_s:.3f}, every chart guaranteed {k_e:.3f}, joint line guarantee {k_j:.3f}")
    for label, k in (("Sidak", k_s), ("every chart", k_e), ("joint line", k_j)):
        arl = np.median(1 / fap_shift(zo, so, k, 2.0))
        print(f"   {label}: median ARL of one chart after a 2-sigma shift {arl:.1f} points = {arl / 6:.1f} hours")
    nl, nc = sb.min_calibration_size(aL, 0.10), sb.min_calibration_size(ac, 0.10)
    print(f"   distribution-free: one line score needs {nl} in-control points ({nl / 48:.1f} shifts); each chart separately "
          f"needs {nc} ({nc / 144:.0f} days at 144 points a day)")
    n, r = 1000, 10
    print(f"\na validation-quantile alert threshold: the {r}th largest of {n} in-control scores (the empirical 99th")
    print(f"percentile). Exceedance probability ~ Beta({r}, {n - r + 1}): mean {r / (n + 1):.5f}; P(exceedance > 1%) = "
          f"{stats.beta.sf(0.01, r, n - r + 1):.3f}; P(exceedance > 1.5%) = {stats.beta.sf(0.015, r, n - r + 1):.3f}")
    alphas = np.geomspace(1e-5, 0.1, 60)
    FIG["sizes"] = {"alpha": alphas.tolist(),
                    "n10": [sb.min_calibration_size(a, 0.10) for a in alphas],
                    "n05": [sb.min_calibration_size(a, 0.05) for a in alphas],
                    "marks": [(label, a, sb.min_calibration_size(a, 0.10)) for label, a in cases]}


# ---------------------------------------------------------------------------------------------------
# Section 5: the continuous-flow line
# ---------------------------------------------------------------------------------------------------

LIVE_P1 = (30, 90)       # Phase I minutes [30, 90): the hour after the start-up transient documented in Part 3
GAMMA = 0.005            # ACI step, the value used by Gibbs and Candes (2021) in their experiments


def aci_run(scores, alpha, gamma, history):
    a = sb.AdaptiveThreshold(alpha=alpha, gamma=gamma, window=10 ** 9, history=list(history))
    return np.array([a.update(v) if np.isfinite(v) else False for v in scores])


def section_liveline() -> None:
    header("5. The continuous-flow line: one false alarm per hour for 21 measurement series")
    L = Line()
    series = []
    for stage in (1, 2):
        for k in range(15):
            if np.mean(L.raw(f"Stage{stage}.Output.Measurement{k}.U.Actual") == 0) < 0.10:
                series.append((stage, k))
    MM = np.array([L.minute_means(L.meas(st, k))[0] for st, k in series])   # series x minutes
    J, T = MM.shape
    p1 = np.arange(LIVE_P1[0], LIVE_P1[1])
    ev = np.arange(LIVE_P1[1], T)
    mu = np.nanmean(MM[:, p1], axis=1)
    sd = np.nanstd(MM[:, p1], axis=1, ddof=1)
    Z = (MM - mu[:, None]) / sd[:, None]
    aL = 1 / 60
    a_series = sb.sidak_alpha(aL, J)
    print(f"{J} series of minute means; Phase I = minutes {LIVE_P1[0]}-{LIVE_P1[1] - 1} ({len(p1)} minutes, after the "
          f"start-up); evaluation = minutes {ev[0]}-{ev[-1]} ({len(ev)} minutes, {len(ev) / 60:.2f} h)")
    print(f"budget: one false alarm per hour for the line = {aL:.5f} per minute; Sidak share per series {a_series:.3e}")
    k_A = sb.sidak_k(aL, J)
    k_B = sb.line_guaranteed_k(len(p1), J, aL, 0.10, RNG["liveline"], reps=4000)
    print(f"design A, Sidak limits, known-parameter width: k = {k_A:.3f}")
    print(f"design B, joint line guarantee for normal data (m = {len(p1)}, eps 0.10): k = {k_B:.3f}")
    n_series = sb.min_calibration_size(a_series, 0.10)
    n_line = sb.min_calibration_size(aL, 0.10)
    print(f"distribution-free: per series needs {n_series} in-control minutes ({n_series / 60:.1f} h); a single line "
          f"score needs {n_line} minutes beyond the Phase I hour; the record has {len(ev)}")
    score = np.where(np.all(np.isnan(Z), axis=0), np.nan, np.nanmax(np.abs(np.nan_to_num(Z, nan=0.0)), axis=0))
    print(f"design C, adaptive threshold on the line score max_j |z_j| (alpha {aL:.5f}, gamma {GAMMA}), history seeded "
          f"with the {len(p1)} Phase I scores")
    bound = lambda TT: (max(aL, 1 - aL) + GAMMA) / (GAMMA * TT)
    print(f"   Gibbs-Candes bound on |alarm frequency - alpha| over T steps: T = {len(ev)}: {bound(len(ev)):.3f}; "
          f"one week of minutes (10,080): {bound(10080):.4f}; four weeks: {bound(40320):.4f}")

    alarms = {}
    over = np.abs(Z[:, ev])
    alarms["A"] = np.nan_to_num(over, nan=0.0) > k_A
    alarms["B"] = np.nan_to_num(over, nan=0.0) > k_B
    aci = aci_run(score[ev], aL, GAMMA, score[p1][np.isfinite(score[p1])])
    hours = len(ev) / 60
    for d in ("A", "B"):
        line = alarms[d].any(axis=0)
        print(f"design {d}: minutes with a line alarm {int(line.sum())} of {len(ev)} ({line.sum() / hours:.1f} per hour; "
              f"budget 1); series alarming at least once {int(alarms[d].any(axis=1).sum())} of {J}; "
              f"alarms per series-minute {alarms[d].mean():.4f} vs Sidak share {a_series:.2e}")
    print(f"design C: minutes with a line alarm {int(aci.sum())} of {len(ev)} ({aci.sum() / hours:.2f} per hour; budget 1); "
          f"alarm minutes {[int(t) for t in ev[aci]]}")
    FIG["live_rates"] = {"A": float(alarms["A"].any(axis=0).sum() / hours), "B": float(alarms["B"].any(axis=0).sum() / hours),
                         "C": float(aci.sum() / hours)}

    # injected shifts
    unit = np.nanstd(MM[:, LIVE_P1[0]:], axis=1, ddof=1)
    print("\ninjected faults: a shift added to one series from minute t0 (every 5 minutes from 90 to 215), in units of")
    print("that series' SD of minute means from minute 30 on; a step, or a ramp reaching the size after 60 minutes.")
    print("detection = an alarm attributed to that series within 15 minutes (step) or 60 minutes (ramp) of t0;")
    print("excess = detection with the fault minus the same count on the unaltered data.")
    starts = list(range(90, 216, 5))
    fig = {}
    for shape, W in (("step", 15), ("ramp", 60)):
        for size in (0.5, 1.0, 2.0, 4.0):
            hits = {"A": [], "B": [], "C": []}
            base = {"A": [], "B": [], "C": []}
            for j in range(J):
                for t0 in starts:
                    w = np.arange(t0, min(t0 + W, T))
                    add = np.zeros(T)
                    ramp = np.clip((np.arange(T) - t0 + 1) / 60, 0, 1) if shape == "ramp" else (np.arange(T) >= t0).astype(float)
                    add = size * unit[j] * ramp
                    Zj = Z.copy()
                    Zj[j] = (MM[j] + add - mu[j]) / sd[j]
                    for d, k in (("A", k_A), ("B", k_B)):
                        hits[d].append(bool(np.any(np.abs(np.nan_to_num(Zj[j, w])) > k)))
                        base[d].append(bool(np.any(np.abs(np.nan_to_num(Z[j, w])) > k)))
                    sc = np.nanmax(np.abs(np.nan_to_num(Zj, nan=0.0)), axis=0)
                    arg = np.argmax(np.abs(np.nan_to_num(Zj, nan=0.0)), axis=0)
                    al = aci_run(sc[ev], aL, GAMMA, score[p1][np.isfinite(score[p1])])
                    idx = w - ev[0]
                    hits["C"].append(bool(np.any(al[idx] & (arg[w] == j))))
                    arg0 = np.argmax(np.abs(np.nan_to_num(Z, nan=0.0)), axis=0)
                    base["C"].append(bool(np.any(aci[idx] & (arg0[w] == j))))
            parts = []
            for d in ("A", "B", "C"):
                h, b = np.mean(hits[d]), np.mean(base[d])
                parts.append(f"{d}: detected {h:.3f}, without fault {b:.3f}, excess {h - b:.3f}")
                fig.setdefault(shape, {}).setdefault(d, []).append(float(h - b))
            print(f"{shape} of {size:3.1f} SD ({len(hits['A'])} injections): " + "; ".join(parts))
    FIG["live_detect"] = {"sizes": [0.5, 1.0, 2.0, 4.0], **fig}


# ---------------------------------------------------------------------------------------------------
# Section 6: the SECOM line
# ---------------------------------------------------------------------------------------------------


def section_secom() -> None:
    header("6. The SECOM line: one false alarm per 100 runs for 428 sensors")
    X, y, stamps = load_secom()
    fail = y == 1
    passi = np.flatnonzero(~fail)
    days = (stamps[-1] - stamps[0]).total_seconds() / 86400
    print(f"{len(y)} runs over {days:.1f} days: {len(y) / days:.1f} runs per day")
    aL, eps = 0.01, 0.10
    n_cal = sb.min_calibration_size(aL, eps)
    p1 = passi[:100]
    cal = passi[100:100 + n_cal]
    test = np.arange(len(y)) > cal[-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(X[p1], axis=0)
        sd = np.nanstd(X[p1], axis=0, ddof=1)
    use = (np.sum(~np.isnan(X[p1]), axis=0) >= 80) & (sd > 0)
    K = int(use.sum())
    Zabs = np.abs((X[:, use] - mu[use]) / sd[use])
    score = np.nanmax(np.nan_to_num(Zabs, nan=0.0), axis=1)
    print(f"Phase I = first 100 passing runs (to {stamps[p1[-1]]}); calibration = next {n_cal} passing runs (to "
          f"{stamps[cal[-1]]}); test = the {int(test.sum())} runs after that ({int((test & ~fail).sum())} passing, "
          f"{int((test & fail).sum())} failing); charted sensors K = {K}")
    print(f"budget: at most {aL:.0%} of in-control runs with an alarm; line score = max_j |z_j| over the K sensors")
    n_sensor = sb.min_calibration_size(sb.sidak_alpha(aL, K), eps)
    print(f"distribution-free per sensor (Sidak share {sb.sidak_alpha(aL, K):.3e}) needs {n_sensor} in-control runs, "
          f"{n_sensor / (len(y) / days) / 365.25:.1f} years at this rate; the line score needs {n_cal}")

    tp, tf = test & ~fail, test & fail
    designs = {}
    k_A = sb.sidak_k(aL, K)
    designs["A Sidak, known-parameter width"] = score > k_A
    k_B = sb.line_guaranteed_k(100, K, aL, eps, RNG["secom"], reps=2000)
    designs["B joint line guarantee, normal theory"] = score > k_B
    ct = sb.calibrated_threshold(score[cal], aL, eps)
    designs["C calibrated threshold, fixed"] = score > ct.threshold
    roll = np.zeros(len(y), dtype=bool)
    for i in np.flatnonzero(test):
        prev = passi[(passi < i)][-n_cal:]
        roll[i] = score[i] > sb.calibrated_threshold(score[prev], aL, eps).threshold
    designs["D calibrated threshold, rolling"] = roll
    aci = np.zeros(len(y), dtype=bool)
    a = sb.AdaptiveThreshold(alpha=aL, gamma=GAMMA, window=n_cal, history=list(score[cal]))
    for i in np.flatnonzero(test):
        aci[i] = a.update(score[i])
    designs["E adaptive threshold"] = aci
    pv_min = 1 / (n_cal + 1)
    need = math.ceil(pv_min * K / aL)
    bh = np.zeros(len(y), dtype=bool)
    ti = np.flatnonzero(test)
    PV = np.full((len(ti), K), np.nan)
    for j in range(K):
        c = Zabs[cal, j]
        c = c[~np.isnan(c)]
        v = Zabs[ti, j]
        ok = ~np.isnan(v)
        PV[ok, j] = sb.conformal_pvalues(c, v[ok])
    min_p = []
    for r, i in enumerate(ti):
        pv = PV[r][~np.isnan(PV[r])]
        bh[i] = sb.benjamini_hochberg(pv, aL).any()
        min_p.append(pv.min())
    designs["F BH on per-sensor conformal p-values"] = bh
    print(f"design A k = {k_A:.3f}; design B k = {k_B:.3f}; design C threshold = {ct.threshold:.3f} (the {ct.rank}"
          f"{'st' if ct.rank == 1 else 'th'} largest of {ct.n}, guarantee {ct.guarantee:.3f})")
    print(f"design D recalibrates on the {n_cal} passing runs before each test run (labels assumed known by then);"
          f" design E: gamma {GAMMA}, window {n_cal}, updated on every test run")
    print(f"design F: smallest possible conformal p-value 1/{n_cal + 1} = {pv_min:.4f}; with {K} sensors BH at q = {aL}"
          f" needs at least {need} sensors at that p-value in the same run to reject anything")
    fig = {}
    for name, al in designs.items():
        a_p, a_f = int(al[tp].sum()), int(al[tf].sum())
        pf = stats.fisher_exact([[a_f, int(tf.sum()) - a_f], [a_p, int(tp.sum()) - a_p]]).pvalue
        print(f"{name}: passing runs alarmed {a_p}/{int(tp.sum())} = {a_p / tp.sum():.3f}; failing {a_f}/{int(tf.sum())}"
              f" = {a_f / tf.sum():.3f}; Fisher p {pf:.3g}")
        fig[name] = {"pass": a_p / float(tp.sum()), "fail": a_f / float(tf.sum())}
    print(f"   design F: smallest p-value in a test run: minimum {min(min_p):.6f}, maximum {max(min_p):.6f}")
    at_min = np.sum(np.isclose(PV, pv_min), axis=1)
    le01 = np.sum(PV <= aL, axis=1)
    print(f"   design F: sensors at the minimum p-value in one test run: median {np.median(at_min):.0f}, maximum "
          f"{at_min.max()}; sensors with p <= {aL}: median {np.median(le01):.0f}, maximum {le01.max()} (BH needs {need})")
    print(f"   Benjamini-Yekutieli factor for arbitrary dependence across {K} sensors: sum 1/i = {np.sum(1 / np.arange(1, K + 1)):.3f}")
    cols = np.flatnonzero(use)
    j = int(np.nanargmax(Zabs[cal[np.argmax(score[cal])]]))
    c = cols[j]
    i_max = cal[np.argmax(score[cal])]
    v = X[:, c]
    print(f"   design C threshold comes from sensor column {c} (0-based) in calibration run {i_max} ({stamps[i_max]}): "
          f"value {v[i_max]:.2f}; Phase I mean {mu[c]:.4f}, SD {sd[c]:.4f}, distinct Phase I values "
          f"{len(np.unique(v[p1][~np.isnan(v[p1])]))}")
    big = np.flatnonzero(np.abs(v - mu[c]) > 1000 * sd[c])
    print(f"   runs where that sensor is more than 1,000 Phase I SDs from its Phase I mean: "
          + ", ".join(f"{i} ({stamps[i].date()}, {v[i]:.2f})" for i in big))
    print(f"   design C alarms: runs {[int(i) for i in np.flatnonzero(designs['C calibrated threshold, fixed'] & test)]}")
    late = test & ~fail
    months = np.array([s.strftime("%Y-%m") for s in stamps])
    for mo in np.unique(months[test]):
        sel = late & (months == mo)
        print(f"   {mo}: passing test runs {int(sel.sum())}; alarmed by C {designs['C calibrated threshold, fixed'][sel].mean():.3f}, "
              f"by D {designs['D calibrated threshold, rolling'][sel].mean():.3f}, by E {designs['E adaptive threshold'][sel].mean():.3f}")
    print("\nother line scores, same Phase I, calibration and test runs. AUC = probability that a failing test run")
    print("scores above a passing one (0.5 = no information about failure).")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(X[p1][:, use], axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(X[p1][:, use] - med), axis=0)
    okm = mad > 0
    Rob = np.abs((X[:, use][:, okm] - med[okm]) / mad[okm])
    Z1 = np.nan_to_num((X[p1][:, use] - mu[use]) / sd[use])
    _, sv, Vt = np.linalg.svd(Z1 - Z1.mean(axis=0), full_matrices=False)
    lam = sv ** 2 / (len(p1) - 1)
    Za = np.nan_to_num((X[:, use] - mu[use]) / sd[use])
    T5 = Za @ Vt[:5].T
    others = {
        "max |z|": score,
        "number of sensors with |z| > 3": np.sum(np.nan_to_num(Zabs, nan=0.0) > 3, axis=1).astype(float),
        f"number of sensors with robust |z| > 3.5 ({int(okm.sum())} with MAD > 0)": np.sum(np.nan_to_num(Rob, nan=0.0) > 3.5, axis=1).astype(float),
        "Hotelling T^2 on 5 principal components": np.sum(T5 * T5 / lam[:5], axis=1),
    }
    auc_fig = {}
    for name, sc in others.items():
        roll_o = np.zeros(len(y), dtype=bool)
        for i in np.flatnonzero(test):
            prev = passi[(passi < i)][-n_cal:]
            roll_o[i] = sc[i] > sb.calibrated_threshold(sc[prev], aL, eps).threshold
        a = sb.AdaptiveThreshold(alpha=aL, gamma=GAMMA, window=n_cal, history=list(sc[cal]))
        aci_o = np.zeros(len(y), dtype=bool)
        for i in np.flatnonzero(test):
            aci_o[i] = a.update(sc[i])
        auc = stats.mannwhitneyu(sc[tf], sc[tp]).statistic / (tf.sum() * tp.sum())
        print(f"   {name}: rolling calibrated, passing {roll_o[tp].mean():.3f} failing {roll_o[tf].mean():.3f} "
              f"({int(roll_o[tf].sum())}/{int(tf.sum())}); adaptive, passing {aci_o[tp].mean():.3f} failing "
              f"{aci_o[tf].mean():.3f} ({int(aci_o[tf].sum())}/{int(tf.sum())}); AUC {auc:.3f}")
        auc_fig[name] = float(auc)
    FIG["secom"] = {"designs": fig, "budget": aL, "threshold_C": ct.threshold, "k_A": k_A, "k_B": k_B, "auc": auc_fig}


def main() -> None:
    print("Statistical process control, Part 4: designing a line's charts around a false-alarm budget")
    print(f"seed {SEED}; numpy {np.__version__}")
    for name in SOURCES:
        fetch(name)
    print("data files verified: " + ", ".join(f"{n} sha256 {SOURCES[n]['sha256'][:12]}" for n in SOURCES))
    section_one_chart()
    section_nonnormal()
    section_line()
    section_sizes()
    section_liveline()
    section_secom()
    (HERE / "figure_data.json").write_text(json.dumps(FIG))


if __name__ == "__main__":
    main()
