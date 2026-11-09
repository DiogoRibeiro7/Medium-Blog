"""Drift detectors read as control charts: the numbers.

Every number quoted in the article is printed by this script. Run:

    python drift_numbers.py > output.txt

Two public data sets are downloaded from pinned commits of GitHub mirrors, checked against SHA-256 hashes
(after converting CRLF line endings to LF) and cached in data/, which is not committed. The detectors are
river's (pinned in requirements.txt); the batch scores and thresholds of Evidently and NannyML are replicated
in drifttools.py from their source code. The budgeted designs use the spcbudget module of the
statistical-process-control companion, installed from a pinned commit.

Sections
1. Data.
2. The translation: Page-Hinkley computed as a CUSUM, and the defaults as installed.
3. Stream detectors on simulated in-control data: scale, autocorrelation and runs of errors.
4. Batch scores on simulated in-control data: PSI, normed Wasserstein, KS, NannyML's threshold, many columns.
5. Bike Sharing: a demand model's residuals and inputs, 2011 replayed in control and 2012 out of control.
6. Elec2: a classifier's error stream, with and without the order inside each day.
7. The budgets side by side.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import urllib.request  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from river import drift  # noqa: E402
from river.drift import binary  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import drifttools as dt  # noqa: E402
import spcbudget as sb  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
SEED = 20261110
_NAMES = ["ph_scale", "ar", "kswin", "ddm", "psi", "wass", "ks", "nanny", "bike_sur", "bike_gk",
          "bike_shift", "elec_perm", "elec_gk", "spare1", "spare2", "spare3"]
RNG = {n: np.random.default_rng(s) for n, s in zip(_NAMES, np.random.SeedSequence(SEED).spawn(len(_NAMES)))}
FIG: dict = {}

T_STREAM = 10_000      # length of each simulated stream
DAY_BUDGET = 1 / 365   # budget for the daily charts: one false alarm a year
EPS = 0.10             # the guarantee holds with probability 0.9 over the reference data


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def f(x, d=3):
    return f"{x:.{d}f}"


# ---------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------

SOURCES = {
    "hour.csv": {
        "sha256": "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f",
        "urls": [
            "https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/"
            "83d69f31047c0daa0d24c500eb3ba850c3380f89/project-bikesharing/Bike-Sharing-Dataset/hour.csv",
            "https://raw.githubusercontent.com/udacity/deep-learning/"
            "dd93380579bb192ab0a778e75596eec68141b4cc/first-neural-network/Bike-Sharing-Dataset/hour.csv",
        ],
    },
    "elec.csv": {
        "sha256": "cdf901433885f29eca6911f70c0eeafb50d90596c879c30c5b99f5a2e8e734ff",
        "urls": ["https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/"
                 "f1eb9f068c3450233721c3beac6fb0e880e9e7d7/elec.csv"],
    },
    "elec2_data.dat": {
        "sha256": "f96f3b096a5318c1dab465f9c64b50824f751d651a13b864137cfefac6a6bef2",
        "urls": ["https://raw.githubusercontent.com/vlosing/driftDatasets/"
                 "01d08dec716870a36ea793e26e7e896ba921fe5d/realWorld/Elec2/elec2_data.dat"],
    },
    "elec2_label.dat": {
        "sha256": "769805ed2f942698b72bb6398d1f685ddd2c8b01213334658df709466890c154",
        "urls": ["https://raw.githubusercontent.com/vlosing/driftDatasets/"
                 "01d08dec716870a36ea793e26e7e896ba921fe5d/realWorld/Elec2/elec2_label.dat"],
    },
}


def fetch(name: str) -> bytes | None:
    """Return the file's bytes (LF line endings) if a mirror serves it with the expected hash."""
    spec = SOURCES[name]
    path = DATA / name
    if path.exists():
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() == spec["sha256"]:
            return raw
    for url in spec["urls"]:
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                raw = r.read().replace(b"\r\n", b"\n")
        except OSError as exc:
            print(f"  could not fetch {url}: {exc}")
            continue
        if hashlib.sha256(raw).hexdigest() == spec["sha256"]:
            DATA.mkdir(exist_ok=True)
            path.write_bytes(raw)
            return raw
        print(f"  hash mismatch for {url}")
    return None


def load_bike() -> pd.DataFrame:
    raw = fetch("hour.csv")
    if raw is None:
        raise SystemExit("no mirror of hour.csv matched its hash")
    return pd.read_csv(io.BytesIO(raw), parse_dates=["dteday"])


def load_elec() -> tuple[pd.DataFrame, dict]:
    """Elec2 (MOA's normalized version, 45,312 half-hours from 7 May 1996). Two independent copies: one with
    the label as 1 = UP and no date or day columns, one with the day of week and the label as 0 = UP."""
    cols = ["period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
    a = fetch("elec.csv")
    bd, bl = fetch("elec2_data.dat"), fetch("elec2_label.dat")
    info = {"copy_a": a is not None, "copy_b": bd is not None and bl is not None}
    frames = []
    if a is not None:
        df = pd.read_csv(io.BytesIO(a))
        frames.append(df[cols + ["class"]].rename(columns={"class": "up"}))
    if info["copy_b"]:
        x = np.loadtxt(io.BytesIO(bd))
        lab = np.loadtxt(io.BytesIO(bl))
        dfb = pd.DataFrame(x[:, 2:], columns=cols)
        dfb["up"] = (1 - lab).astype(int)
        frames.append(dfb)
    if not frames:
        raise SystemExit("no copy of Elec2 matched its hashes")
    df = frames[0].copy()
    n = len(df)
    # Day of week from the row index: 48 half-hours a day, the first day (7 May 1996) is coded 2.
    df["day"] = ((np.arange(n) // 48 + 1) % 7) + 1
    df["dayno"] = np.arange(n) // 48
    if len(frames) == 2:
        info["max_abs_diff"] = float(np.abs(frames[0][cols].values - frames[1][cols].values).max())
        info["labels_agree"] = bool((frames[0]["up"].values == frames[1]["up"].values).all())
        info["day_agrees"] = bool((df["day"].values == x[:, 1]).all())
    return df, info


def acf(x, lag):
    x = np.asarray(x, float)
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


# ---------------------------------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------------------------------

header("1. Data")
bike = load_bike()
elec, einfo = load_elec()
ts = bike.dteday + pd.to_timedelta(bike.hr, unit="h")
print(f"Bike Sharing hour.csv: {len(bike)} hours, {bike.dteday.min().date()} to {bike.dteday.max().date()}, "
      f"{bike.dteday.nunique()} days, {24 * bike.dteday.nunique() - len(bike)} hours absent")
mm = bike.groupby(bike.dteday.dt.to_period("M")).cnt.mean()
print("  mean rentals per hour by month, 2011:", " ".join(f"{v:.0f}" for v in mm.values[:12]))
print("  mean rentals per hour by month, 2012:", " ".join(f"{v:.0f}" for v in mm.values[12:]))
print(f"  2012 over 2011, same month: {np.min(mm.values[12:] / mm.values[:12]):.2f} to "
      f"{np.max(mm.values[12:] / mm.values[:12]):.2f} times")
print(f"Elec2: {len(elec)} half-hours, {elec.dayno.max() + 1} days; copies present: "
      f"A (scikit-multiflow) {einfo['copy_a']}, B (vlosing) {einfo['copy_b']}")
if "max_abs_diff" in einfo:
    print(f"  copies A and B agree: largest feature difference {einfo['max_abs_diff']:.1e}, labels agree "
          f"{einfo['labels_agree']}, day of week rebuilt from the row index matches B's column "
          f"{einfo['day_agrees']}")
up = elec.up.values
print(f"  share UP {up.mean():.4f}; always-DOWN accuracy {1 - up.mean():.4f}; "
      f"accuracy of predicting the previous label {np.mean(up[1:] == up[:-1]):.4f}")
for c in ["vicprice", "vicdemand", "transfer"]:
    v = elec[c].values
    first = int(np.argmax(v != v[0]))
    print(f"  {c}: constant at {v[0]} for the first {first} rows ({first / 48:.0f} days)")
const_rows = int(np.argmax(elec.vicprice.values != elec.vicprice.values[0]))


# ---------------------------------------------------------------------------------------------------
# 2. The translation
# ---------------------------------------------------------------------------------------------------

header("2. The translation: Page-Hinkley is a CUSUM; the defaults as installed")
ph = drift.PageHinkley()
print(f"river PageHinkley: min_instances={ph.min_instances}, delta={ph.delta}, threshold={ph.threshold}, "
      f"alpha={ph.alpha}, mode={ph.mode!r}")
ad = drift.ADWIN()
print(f"river ADWIN: delta={ad.delta}, clock={ad.clock}, max_buckets={ad.max_buckets}, "
      f"min_window_length={ad.min_window_length}, grace_period={ad.grace_period}")
kw = drift.KSWIN()
print(f"river KSWIN: alpha={kw.alpha}, window_size={kw.window_size}, stat_size={kw.stat_size}")
dd = binary.DDM()
print(f"river DDM: warm_start={dd.warm_start}, warning_threshold={dd.warning_threshold}, "
      f"drift_threshold={dd.drift_threshold}")

print(f"reference points: a 3-sigma Shewhart chart with known parameters has in-control ARL "
      f"{1 / (2 * stats.norm.sf(3)):.1f}; a two-sided CUSUM with k=0.5 has in-control ARL {dt.cusum_arl(0.5, 4):.0f} "
      f"at h=4 and {dt.cusum_arl(0.5, 5):.0f} at h=5 (Markov chain, Brook and Evans 1972)")

# Check: river's upper Page-Hinkley signal times equal those of the CUSUM written in drifttools.
x = 1.0 * RNG["ph_scale"].standard_normal(3000) + np.r_[np.zeros(1500), 0.8 * np.ones(1500)]
stat = dt.page_hinkley_statistic(x)
mine = [t for t in range(len(x)) if t + 1 >= 30 and stat[t] > 50.0][:1]
det = drift.PageHinkley(mode="up")
theirs = dt.run_detector(det, x)[:1]
print(f"upper Page-Hinkley on a stream with a +0.8 sigma step at t=1500: first signal by river at "
      f"t={theirs[0]}, by the CUSUM of deviations from the running mean at t={mine[0]}")


# ---------------------------------------------------------------------------------------------------
# 3. Stream detectors under the null
# ---------------------------------------------------------------------------------------------------

header("3. Stream detectors on simulated in-control streams (alarms per 10,000 observations)")
rng = RNG["ph_scale"]
sigmas = [0.01, 0.03, 0.1, 0.36, 1, 3, 10, 30, 100]
ph_scale = []
ad_scale = []
for s in sigmas:
    ph_scale.append(np.mean([len(dt.run_detector(drift.PageHinkley(), s * rng.standard_normal(T_STREAM)))
                             for _ in range(20)]))
    ad_scale.append(np.mean([len(dt.run_detector(drift.ADWIN(), s * rng.standard_normal(T_STREAM)))
                             for _ in range(20)]))
print("i.i.d. normal, standard deviation sigma; 20 streams each")
for s, a, b in zip(sigmas, ph_scale, ad_scale):
    print(f"  sigma {s:>6}: Page-Hinkley {a:7.2f}   ADWIN {b:5.2f}")
print(f"  a stream that signals at every opportunity after the 30-observation warm-up signals "
      f"{T_STREAM / 30:.0f} times")
FIG["ph_scale"] = {"sigma": sigmas, "ph": ph_scale}

rng = RNG["ar"]
phis = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
ar_res = {"ADWIN": [], "Page-Hinkley": [], "KSWIN": []}
for phi in phis:
    ar_res["ADWIN"].append(np.mean([len(dt.run_detector(drift.ADWIN(), dt.ar1(phi, T_STREAM, rng)))
                                    for _ in range(20)]))
    ar_res["Page-Hinkley"].append(np.mean([len(dt.run_detector(drift.PageHinkley(),
                                                               dt.ar1(phi, T_STREAM, rng)))
                                           for _ in range(20)]))
kr = RNG["kswin"]
for phi in phis:
    ar_res["KSWIN"].append(np.mean([len(dt.run_detector(drift.KSWIN(seed=int(kr.integers(2**31))),
                                                        dt.ar1(phi, T_STREAM, kr))) for _ in range(6)]))
print("Gaussian AR(1) with unit marginal variance, lag-1 autocorrelation phi (20 streams; KSWIN 6)")
for i, phi in enumerate(phis):
    print(f"  phi {phi:.1f}: ADWIN {ar_res['ADWIN'][i]:6.2f}   Page-Hinkley {ar_res['Page-Hinkley'][i]:6.2f}   "
          f"KSWIN {ar_res['KSWIN'][i]:6.2f}")
FIG["ar"] = {"phi": phis, **ar_res}

rng = RNG["ddm"]
ps = [0.05, 0.1, 0.2, 0.3, 0.5]
print("DDM on i.i.d. Bernoulli(p) errors (20 streams each)")
ddm_p = []
for p in ps:
    ddm_p.append(np.mean([len(dt.run_detector(binary.DDM(), (rng.random(T_STREAM) < p).astype(int)))
                          for _ in range(20)]))
    print(f"  p {p:.2f}: {ddm_p[-1]:6.2f}")
rhos = [0.0, 0.2, 0.4, 0.6]
ddm_rho = []
print("DDM on errors with rate 0.3 arriving in runs (two-state Markov chain, lag-1 autocorrelation rho)")
for rho in rhos:
    ddm_rho.append(np.mean([len(dt.run_detector(binary.DDM(), dt.markov_bernoulli(0.3, rho, T_STREAM, rng)))
                            for _ in range(20)]))
    print(f"  rho {rho:.1f}: {ddm_rho[-1]:6.2f}")
FIG["ddm"] = {"rho": rhos, "ddm": ddm_rho}

# p_min selection: how far below the true rate DDM's reference p_min + s_min ends up after 10,000 errors
sel = []
for _ in range(200):
    e = (rng.random(T_STREAM) < 0.3).astype(float)
    n = np.arange(1, T_STREAM + 1)
    pi = np.cumsum(e) / n
    si = np.sqrt(pi * (1 - pi) / n)
    ok = n > 30
    j = np.argmin(np.where(ok, pi + si, np.inf))
    sel.append((pi[j] - 0.3) / math.sqrt(0.3 * 0.7 / n[j]))
print(f"DDM's reference p_min is the running error rate at its lowest point: on i.i.d. errors with rate 0.3 "
      f"(200 streams of 10,000) it sits a median {np.median(sel):.2f} standard errors below the true rate")


# ---------------------------------------------------------------------------------------------------
# 4. Batch scores under the null
# ---------------------------------------------------------------------------------------------------

header("4. Batch scores on simulated in-control windows")
rng = RNG["psi"]
n_ref = 5000
ncur = [24, 50, 100, 200, 500, 1000, 5000]
psi_rows = []
print(f"PSI, reference n={n_ref} and current window n from one N(0,1); 400 pairs each")
for nc in ncur:
    q = np.array([dt.psi_quantile_bins(rng.standard_normal(n_ref), rng.standard_normal(nc)) for _ in range(400)])
    e = np.array([dt.psi_evidently(rng.standard_normal(n_ref), rng.standard_normal(nc)) for _ in range(400)])
    psi_rows.append((nc, q.mean(), (q >= 0.1).mean(), (q >= 0.25).mean(), e.mean(), (e >= 0.1).mean(),
                     (e >= 0.25).mean()))
    print(f"  n={nc:>5}: deciles mean {q.mean():.4f} (large-sample {dt.psi_null_mean(10, n_ref, nc):.4f}), "
          f"P(>=0.1) {(q >= 0.1).mean():.3f}, P(>=0.25) {(q >= 0.25).mean():.3f}; "
          f"Evidently bins mean {e.mean():.4f}, P(>=0.1) {(e >= 0.1).mean():.3f}, P(>=0.25) {(e >= 0.25).mean():.3f}")
for nc in [24, 100, 500]:
    c01 = stats.chi2.sf(0.1 / (1 / n_ref + 1 / nc), 9)
    print(f"  chi-square approximation, deciles, n={nc}: P(PSI >= 0.1) = {c01:.3f}")
FIG["psi"] = {"n": ncur, "deciles_01": [r[2] for r in psi_rows], "evidently_01": [r[5] for r in psi_rows]}

rng = RNG["wass"]
print(f"Normed Wasserstein (Evidently's default for a numerical column when the reference has more than "
      f"1,000 rows), reference n={n_ref}; P(score >= 0.1); 400 pairs each")
w_iid, w_ar = [], []
for nc in ncur:
    a = np.array([dt.wasserstein_normed(rng.standard_normal(n_ref), rng.standard_normal(nc)) for _ in range(400)])
    b = np.array([dt.wasserstein_normed(dt.ar1(0.5, n_ref, rng), dt.ar1(0.5, nc, rng)) for _ in range(400)])
    w_iid.append((a >= 0.1).mean())
    w_ar.append((b >= 0.1).mean())
    print(f"  n={nc:>5}: i.i.d. {w_iid[-1]:.3f}   AR(1) phi=0.5 {w_ar[-1]:.3f}")
FIG["wass"] = {"n": ncur, "iid": w_iid, "ar05": w_ar}

rng = RNG["ks"]
print("Two-sample KS at p < 0.05 (Evidently's default when the reference has at most 1,000 rows), reference "
      "n=1000, current n=200, both AR(1); 1,000 pairs each")
ks_rows = []
for phi in [0.0, 0.2, 0.4, 0.6, 0.8]:
    pv = np.array([stats.ks_2samp(dt.ar1(phi, 1000, rng), dt.ar1(phi, 200, rng)).pvalue for _ in range(1000)])
    ks_rows.append((pv < 0.05).mean())
    print(f"  phi {phi:.1f}: rejection rate {(pv < 0.05).mean():.3f}")

rng = RNG["nanny"]
chunk = 500
print(f"NannyML's default alert threshold for the KS statistic: 10 reference chunks of {chunk}, mean + 3 sd "
      f"(ddof=0) of their scores against the full reference; rate per analysis chunk from the same "
      f"distribution, 400 references x 50 analysis chunks")
rates, rates_oos, ratio = [], [], []
pool = []
for _ in range(400):
    ref = rng.standard_normal(10 * chunk)
    ins = np.array([dt.ks_statistic(ref, ref[i * chunk:(i + 1) * chunk]) for i in range(10)])
    oos = np.array([dt.ks_statistic(np.delete(ref, np.s_[i * chunk:(i + 1) * chunk]),
                                    ref[i * chunk:(i + 1) * chunk]) for i in range(10)])
    an = np.array([dt.ks_statistic(ref, rng.standard_normal(chunk)) for _ in range(50)])
    pool.append(an)
    rates.append(np.mean(an > dt.nannyml_upper_threshold(ins)))
    rates_oos.append(np.mean(an > dt.nannyml_upper_threshold(oos)))
    ratio.append(ins.mean() / an.mean())
rates, rates_oos = np.array(rates), np.array(rates_oos)
pool = np.concatenate(pool)
known = np.mean(pool > pool.mean() + 3 * pool.std())
print(f"  rate per analysis chunk: mean {rates.mean():.4f}, median {np.median(rates):.4f}, 90th percentile "
      f"{np.quantile(rates, 0.9):.4f}; share of references with a rate above 1%: {(rates > 0.01).mean():.3f}")
print(f"  P(at least one alert among 10 analysis chunks), averaged over references: "
      f"{np.mean(1 - (1 - rates) ** 10):.3f}")
print(f"  reference chunks scored against a reference that contains them: mean score {np.mean(ratio):.3f} of an "
      f"analysis chunk's (algebra for 10 chunks: 0.9 x sqrt(1 + 1/9) / sqrt(1 + 1/10) = "
      f"{0.9 * math.sqrt(1 + 1 / 9) / math.sqrt(1 + 1 / 10):.3f})")
print(f"  scoring each reference chunk against the other nine instead: mean rate {rates_oos.mean():.4f}, "
      f"median {np.median(rates_oos):.4f}")
print(f"  mean + 3 sd of the analysis-chunk score with both known (KS is right-skewed): rate {known:.4f}; "
      f"normal theory, one-sided: {stats.norm.sf(3):.5f}")
FIG["nanny"] = {"rates": rates.tolist()}

print("Many columns at once, each tested at its own default rate, independent columns")
for K in [4, 9, 20, 50]:
    print(f"  K={K:>2}: P(any column flagged) at 0.05 each {1 - 0.95**K:.3f}; P(at least half flagged, "
          f"Evidently's dataset-drift rule) {stats.binom.sf(math.ceil(K / 2) - 1, K, 0.05):.2e}; Sidak rate "
          f"per column for a 1% window budget {sb.sidak_alpha(0.01, K):.5f}")


# ---------------------------------------------------------------------------------------------------
# 5. Bike Sharing
# ---------------------------------------------------------------------------------------------------

header("5. Bike Sharing: a demand model's residuals and inputs")
F_BIKE = ["hr", "weekday", "workingday", "holiday", "season", "weathersit", "temp", "atemp", "hum", "windspeed"]
WEATHER = ["temp", "atemp", "hum", "windspeed"]
bike["y"] = np.log1p(bike.cnt)
bike["month"] = bike.dteday.dt.month
y11 = bike[bike.yr == 0].reset_index(drop=True)
y12 = bike[bike.yr == 1].reset_index(drop=True)


def hgb():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, early_stopping=False, random_state=0)


oof = np.zeros(len(y11))
for m_ in range(1, 13):
    tr = (y11.month != m_).values
    oof[~tr] = hgb().fit(y11.loc[tr, F_BIKE], y11.loc[tr, "y"]).predict(y11.loc[~tr, F_BIKE])
y11["r"] = y11.y - oof
y11["rc"] = y11.cnt - np.expm1(oof)
full_model = hgb().fit(y11[F_BIKE], y11.y)
y12["r"] = y12.y - full_model.predict(y12[F_BIKE])
print("Model: gradient boosting on log(1 + rentals) with hour, weekday, working day, holiday, season, weather "
      "and the four weather measurements; no year or month. 2011 residuals out of fold (each month predicted "
      "by a model fitted to the other eleven); 2012 residuals from the model fitted to all of 2011.")
mres11 = y11.groupby("month").r.mean()
mres12 = y12.groupby("month").r.mean()
print("  mean residual by month, 2011 (out of fold):", " ".join(f"{v:+.2f}" for v in mres11.values))
print("  mean residual by month, 2012:", " ".join(f"{v:+.2f}" for v in mres12.values))
r = y11.r.values
print(f"  2011 hourly residual: sd {r.std(ddof=1):.3f}; autocorrelation lag 1 {acf(r, 1):.3f}, lag 2 "
      f"{acf(r, 2):.3f}, lag 24 {acf(r, 24):.3f}; on the count scale sd {y11.rc.std(ddof=1):.1f} rentals")
d11 = y11.groupby("dteday").r.mean()
d12 = y12.groupby("dteday").r.mean()
print(f"  daily mean residual, 2011: sd {d11.std(ddof=1):.3f} against {r.std(ddof=1) / math.sqrt(24):.3f} if the "
      f"24 hours of a day were independent (ratio {d11.std(ddof=1) / (r.std(ddof=1) / math.sqrt(24)):.2f}); "
      f"lag-1 autocorrelation {acf(d11.values, 1):.3f}; skewness {stats.skew(d11.values):.2f}, excess kurtosis "
      f"{stats.kurtosis(d11.values):.2f}")
print("  most negative days:", ", ".join(f"{k.date()} {v:+.2f}" for k, v in d11.sort_values().head(6).items()))
for day in ["2011-01-01", "2011-01-27", "2011-03-10", "2011-03-21", "2011-10-29", "2011-11-24", "2011-12-07",
            "2011-12-25", "2011-11-27"]:
    g = y11[y11.dteday == day]
    print(f"    {day} ({pd.Timestamp(day).day_name()}): {len(g)} hours recorded, light rain or snow or worse "
          f"(weather category 3-4) in {int((g.weathersit >= 3).sum())}, holiday flag {int(g.holiday.iloc[0])}")

# Inputs: each 2011 day against the 2011 reference, Evidently's default for > 1,000 reference rows.
flag = {c: [] for c in WEATHER}
for _, g in y11.groupby("dteday"):
    for c in WEATHER:
        flag[c].append(dt.wasserstein_normed(y11[c].values, g[c].values) >= 0.1)
share = np.mean([np.mean([flag[c][i] for c in WEATHER]) >= 0.5 for i in range(len(flag["temp"]))])
print("Inputs, each 2011 day (24 hours) against all of 2011, normed Wasserstein >= 0.1: share of days flagged "
      + ", ".join(f"{c} {np.mean(flag[c]):.3f}" for c in WEATHER) + f"; dataset drift (half the columns) {share:.3f}")
wk = {c: [] for c in WEATHER}
for _, g in y11.groupby((y11.dteday - y11.dteday.min()).dt.days // 7):
    for c in WEATHER:
        wk[c].append(dt.wasserstein_normed(y11[c].values, g[c].values) >= 0.1)
print("  the same with weeks as the current window: " + ", ".join(f"{c} {np.mean(wk[c]):.3f}" for c in WEATHER))
ks_cols, ref_rows = [], []
for m_ in range(1, 13):
    a_ = y11[y11.month == m_]
    b_ = y12[y12.month == m_]
    ref_rows.append(len(a_))
    ks_cols.append(sum(stats.ks_2samp(a_[c].values, b_[c].values).pvalue <= 0.05 for c in WEATHER))
print(f"  each 2012 month against the same 2011 month ({min(ref_rows)}-{max(ref_rows)} reference rows, so "
      f"Evidently's default is KS at p <= 0.05): columns flagged of 4, January to December: "
      + " ".join(str(v) for v in ks_cols))
print(f"  months with dataset drift (at least half the columns flagged): {sum(v >= 2 for v in ks_cols)} of 12")
tC = y11.groupby("month").temp.mean() * 47 - 8
tC12 = y12.groupby("month").temp.mean() * 47 - 8
print(f"  mean temperature in March (dataset scaling -8 to 39 C): 2011 {tC[3]:.1f} C, 2012 {tC12[3]:.1f} C; "
      f"July 2011 {tC[7]:.1f} C, 2012 {tC12[7]:.1f} C")

# Stream detectors with their defaults on the 2011 hourly residual stream.
days11 = y11.dteday.nunique()
DEFAULTS = {
    "ADWIN": (lambda: drift.ADWIN(), "r"),
    "KSWIN": (lambda: drift.KSWIN(seed=7), "r"),
    "Page-Hinkley (log)": (lambda: drift.PageHinkley(), "r"),
    "Page-Hinkley (counts)": (lambda: drift.PageHinkley(), "rc"),
}
dates11 = y11.dteday.values
print(f"  hourly temperature: lag-1 autocorrelation {acf(y11.temp.values, 1):.3f}, lag 24 {acf(y11.temp.values, 24):.3f}")
print("Stream detectors with default settings on the 2011 hourly residuals (out of fold), replayed in order")
real_rate = {}
adwin11 = []
for name, (mk, col) in DEFAULTS.items():
    al = dt.run_detector(mk(), y11[col].values)
    real_rate[name] = len(al) / days11 * 30
    if name == "ADWIN":
        adwin11 = sorted({str(pd.Timestamp(dates11[i]).date()) for i in al})
    print(f"  {name}: {len(al)} signals, {real_rate[name]:.2f} per 30 days")
print("  ADWIN signal days:", ", ".join(adwin11))

BLOCK = 7


def block_surrogate(days_list, n_days, rng):
    """In-control surrogate: whole days in blocks of BLOCK consecutive days, block starts drawn at random
    (circularly) from the reference days. Stationary by construction; keeps the hours of a day and the
    dependence between neighbouring days."""
    n = len(days_list)
    out = []
    while len(out) < n_days:
        s0 = int(rng.integers(0, n))
        out += [days_list[(s0 + j) % n] for j in range(BLOCK)]
    return np.concatenate(out[:n_days])


rng = RNG["bike_sur"]
days_list = [g.r.values for _, g in y11.groupby("dteday")]
print(f"  the same detectors on two in-control surrogates of the 2011 residuals, signals per 30 days (mean of 20 "
      f"surrogate years; KSWIN 5): whole days in blocks of {BLOCK} consecutive days, resampled | hours shuffled")
sur = {}
for name in ["ADWIN", "KSWIN", "Page-Hinkley (log)"]:
    mk = DEFAULTS[name][0]
    B = 5 if name == "KSWIN" else 20
    db = [len(dt.run_detector(mk(), block_surrogate(days_list, days11, rng))) for _ in range(B)]
    hs = [len(dt.run_detector(mk(), rng.permutation(y11.r.values))) for _ in range(B)]
    sur[name] = (np.mean(db) / days11 * 30, np.mean(hs) / days11 * 30)
    print(f"    {name}: {sur[name][0]:.2f} | {sur[name][1]:.2f}")


def whiten(d_phase1: np.ndarray, consecutive: np.ndarray):
    """AR(1) fitted to Phase I daily means; returns mean, phi and the Phase I one-step residuals."""
    mu = float(d_phase1.mean())
    a_, b_ = d_phase1[:-1][consecutive], d_phase1[1:][consecutive]
    phi = float(np.sum((a_ - mu) * (b_ - mu)) / np.sum((a_ - mu) ** 2))
    e = (b_ - mu) - phi * (a_ - mu)
    return mu, phi, e


def budgeted_design(d_phase1: np.ndarray, consecutive: np.ndarray, rng, B: int = 200):
    mu, phi, e = whiten(d_phase1, consecutive)
    m = len(e)
    k_sh = sb.guaranteed_k(m, DAY_BUDGET, EPS)
    h_p = dt.guaranteed_cusum_h(e, 0.5, 1 / DAY_BUDGET, EPS, rng, B=B, kind="parametric")
    h_np = dt.guaranteed_cusum_h(e, 0.5, 1 / DAY_BUDGET, EPS, rng, B=B, kind="nonparametric")
    return {"mu": mu, "phi": phi, "em": float(e.mean()), "es": float(e.std(ddof=1)), "m": m, "k": k_sh,
            "h_p": h_p, "h_np": h_np}


def monitor(design, d_prev: float, d_new: np.ndarray, which: str, sides: bool = False):
    """Indices of the monitored days that signal, on the standardized one-step residuals of the AR(1);
    with sides=True, (index, +1 or -1) pairs."""
    full = np.r_[d_prev, d_new]
    e = (full[1:] - design["mu"]) - design["phi"] * (full[:-1] - design["mu"])
    z = (e - design["em"]) / design["es"]
    if which == "shewhart":
        out = [(i, 1 if v > 0 else -1) for i, v in enumerate(z) if abs(v) > design["k"]]
        return out if sides else [i for i, _ in out]
    return dt.cusum_alarms(z, 0.5, design["h_np" if which == "cusum_np" else "h_p"], sides=sides)


ADWIN_GRID = [2e-3, 2e-4, 2e-5, 2e-6, 2e-8, 2e-10, 2e-12, 2e-15]
PH_GRID = [50, 75, 100, 150, 200, 300, 400, 600]


def calibrate(kind: str, days_list, rng, n_sur: int = 20):
    """Most sensitive grid value whose mean signal rate on block surrogates of the reference days is at most
    the daily budget (an expected rate, with no guarantee over the reference)."""
    surs = [block_surrogate(days_list, 364, rng) for _ in range(n_sur)]
    grid = ADWIN_GRID if kind == "ADWIN" else PH_GRID
    for g in grid:
        mk = (lambda: drift.ADWIN(delta=g)) if kind == "ADWIN" else (lambda: drift.PageHinkley(threshold=g))
        rate = np.mean([len(dt.run_detector(mk(), x_)) for x_ in surs]) / 364
        if rate <= DAY_BUDGET:
            return g, rate
    return grid[-1], rate


print(f"Budget: an in-control ARL of {1 / DAY_BUDGET:.0f} days (alarm probability {DAY_BUDGET:.5f} a day); for the "
      f"daily charts guaranteed with probability {1 - EPS:.1f} over the Phase I data")
print(f"  a distribution-free (Wilks) threshold for that budget needs {sb.min_calibration_size(DAY_BUDGET, EPS)} "
      f"in-control days; with {days11} the smallest guaranteeable daily rate is "
      f"{1 - EPS ** (1 / days11):.5f} (one alarm per {1 / (1 - EPS ** (1 / days11)):.0f} days)")
rng = RNG["bike_gk"]
dvals = d11.values
dindex = d11.index
consec_all = (np.diff(dindex.values).astype("timedelta64[D]").astype(int) == 1)
des11 = budgeted_design(dvals, consec_all, rng)
print(f"  Phase I all of 2011 ({des11['m']} one-step residuals of an AR(1) on the daily mean residual): phi "
      f"{des11['phi']:.3f}, sd {des11['es']:.3f}; Shewhart k {des11['k']:.3f} (known-parameter value "
      f"{stats.norm.isf(DAY_BUDGET / 2):.3f}); two-sided CUSUM with k=0.5, h by parametric bootstrap "
      f"{des11['h_p']:.2f}, nonparametric {des11['h_np']:.2f} (known-parameter value "
      f"{dt.cusum_h_for_arl(0.5, 1 / DAY_BUDGET):.2f})")
_, _, e11 = whiten(dvals, consec_all)
print(f"  one-step residuals: skewness {stats.skew(e11):.2f}, excess kurtosis {stats.kurtosis(e11):.2f}; "
      f"standard error of phi about {math.sqrt((1 - des11['phi'] ** 2) / des11['m']):.3f}")
print(f"  a two-sided distribution-free threshold (the minimum and maximum of the reference) needs "
      f"{sb.min_calibration_size(DAY_BUDGET, EPS, rank=2)} in-control days for the same budget")
for dlt in [0.1, 0.2, 0.3, float(mres12.values[0])]:
    w_ = (1 - des11["phi"]) * dlt / des11["es"]
    print(f"  a sustained step of {dlt:.2f} in the daily mean is {w_:.2f} sd of the one-step residual after its first "
          f"day; the Shewhart chart then signals on a given day with probability "
          f"{stats.norm.sf(des11['k'] - w_):.3f}; a CUSUM aimed at it would use k = {w_ / 2:.2f}")
cal11 = {k: calibrate(k, days_list, rng) for k in ["ADWIN", "Page-Hinkley"]}
print(f"  stream detectors calibrated to the budget on block surrogates of all of 2011: ADWIN delta "
      f"{cal11['ADWIN'][0]:g} (surrogate rate {cal11['ADWIN'][1]:.5f} a day), Page-Hinkley threshold "
      f"{cal11['Page-Hinkley'][0]:g} ({cal11['Page-Hinkley'][1]:.5f})")

# Leave one quarter out: Phase I the other three quarters; in-control replay and injected shifts.
rng = RNG["bike_shift"]
y11["q"] = (y11.month - 1) // 3
dq = (dindex.month - 1) // 3
shifts = [0.1, 0.2, 0.3]
starts = [14, 21, 28, 35, 42, 49, 56]
METHODS = ["ADWIN, default", "Page-Hinkley, default", "ADWIN, calibrated", "Page-Hinkley, calibrated",
           "Shewhart", "CUSUM"]
delays = {m: {s: [] for s in shifts} for m in METHODS}
low_first = {m: {s: [] for s in shifts} for m in METHODS}
incontrol = {m: [] for m in METHODS}
qinfo = []
for q_ in range(4):
    keep = (dq != q_)
    c1 = np.diff(dindex[keep].values).astype("timedelta64[D]").astype(int) == 1
    des = budgeted_design(dvals[keep], c1, rng)
    dl = [days_list[i] for i in np.flatnonzero(keep)]
    cal = {k: calibrate(k, dl, rng)[0] for k in ["ADWIN", "Page-Hinkley"]}
    qinfo.append((des["k"], des["h_np"], cal["ADWIN"], cal["Page-Hinkley"]))
    hq = y11[y11.q == q_].reset_index(drop=True)
    qdays = list(pd.unique(hq.dteday))
    first = int(np.argmax(dq == q_))
    prev = dvals[first - 1] if first > 0 else des["mu"]
    makers = {"ADWIN, default": lambda: drift.ADWIN(), "Page-Hinkley, default": lambda: drift.PageHinkley(),
              "ADWIN, calibrated": lambda: drift.ADWIN(delta=cal["ADWIN"]),
              "Page-Hinkley, calibrated": lambda: drift.PageHinkley(threshold=cal["Page-Hinkley"])}

    def daily_of(xs):
        return pd.Series(xs).groupby(hq.dteday.values).mean().values

    def signals(name, xs):
        """(signal time in hours from the start of the quarter, side) pairs; daily charts signal at the end
        of the day and report the side (+1 high, -1 low); stream detectors do not report it (0)."""
        if name in makers:
            return [(i + 1, 0) for i in dt.run_detector(makers[name](), xs)]
        which = "shewhart" if name == "Shewhart" else "cusum_np"
        hours_end = np.cumsum(pd.Series(xs).groupby(hq.dteday.values).size().values)
        return [(int(hours_end[i]), sd_) for i, sd_ in monitor(des, prev, daily_of(xs), which, sides=True)]

    base = hq.r.values
    for name in METHODS:
        sig = signals(name, base)
        incontrol[name] += [str(pd.Timestamp(hq.dteday.values[min(t, len(base)) - 1]).date()) for t, _ in sig]
    for s_day in starts:
        t0 = int(np.argmax(hq.dteday.values == qdays[s_day]))
        for dlt in shifts:
            xs = base.copy()
            xs[t0:] += dlt
            for name in METHODS:
                after_ = [(t, sd_) for t, sd_ in signals(name, xs) if t > t0]
                delays[name][dlt].append((after_[0][0] - t0) / 24 if after_ else math.inf)
                low_first[name][dlt].append(bool(after_) and after_[0][1] < 0)
print("Leave one quarter out: Phase I the other three quarters of 2011 for the daily charts and for the "
      "calibration of the stream detectors; Shewhart k " + ", ".join(f"{q[0]:.3f}" for q in qinfo)
      + "; CUSUM h (nonparametric) " + ", ".join(f"{q[1]:.2f}" for q in qinfo)
      + "; ADWIN delta " + ", ".join(f"{q[2]:g}" for q in qinfo)
      + "; Page-Hinkley threshold " + ", ".join(f"{q[3]:g}" for q in qinfo))
print(f"  in-control replay of each held-out quarter, signals per 30 days (budget {30 * DAY_BUDGET:.3f}):")
for name in METHODS:
    real_rate[name] = len(incontrol[name]) / days11 * 30
    lst = ", ".join(incontrol[name]) if len(incontrol[name]) <= 12 else f"{len(incontrol[name])} days"
    print(f"    {name}: {len(incontrol[name])} signals, {real_rate[name]:.2f} per 30 days ({lst})")
print("  injected shifts: a step of delta log units added to the held-out quarter's hourly residuals from day 14, "
      "21, ..., 56 (28 runs per delta); days from the start of the step to the first signal after it, of either "
      "sign; daily charts signal at the end of a day; median [quartiles], 'never' when more of the runs than "
      "the quantile reached the end of the quarter without a signal")


def qfmt(v):
    v = np.asarray(v, float)
    big = np.where(np.isinf(v), 1e9, v)
    out = []
    for q in (0.25, 0.5, 0.75):
        x_ = float(np.quantile(big, q))
        out.append("never" if x_ > 1e3 else f"{x_:.1f}")
    return f"{out[1]} [{out[0]}, {out[2]}]"


for name in METHODS:
    print(f"    {name}: " + "; ".join(f"delta {dlt}: {qfmt(delays[name][dlt])}" for dlt in shifts))
    if name in ("Shewhart", "CUSUM"):
        print("      undetected, and first signals after the step that were low-side (opposite to the step): "
              + "; ".join(f"delta {dlt}: {int(np.sum(np.isinf(delays[name][dlt])))} and "
                          f"{int(np.sum(low_first[name][dlt]))} of 28" for dlt in shifts))
FIG["delays"] = {m: {str(s): [None if math.isinf(x) else x for x in v] for s, v in d.items()}
                 for m, d in delays.items()}
FIG["rates"] = {m: real_rate[m] for m in METHODS}

# 2012: Phase I is all of 2011.
print("2012, with all of 2011 as Phase I (stream detectors continue from the 2011 stream):")
first12 = {}
for which, lab in [("shewhart", "Shewhart"), ("cusum_np", "CUSUM")]:
    al = monitor(des11, dvals[-1], d12.values, which)
    first12[which] = al[0] + 1 if al else None
    print(f"  {lab}: first signal at the end of day {first12[which]} ({d12.index[al[0]].date() if al else '-'})")
n11 = len(y11)
rc12 = (y12.cnt - np.expm1(full_model.predict(y12[F_BIKE]))).values
for name, mk, col in [("ADWIN", lambda: drift.ADWIN(), "r"), ("KSWIN", lambda: drift.KSWIN(seed=7), "r"),
                      ("Page-Hinkley (log)", lambda: drift.PageHinkley(), "r"),
                      ("Page-Hinkley (counts)", lambda: drift.PageHinkley(), "rc"),
                      ("ADWIN, calibrated", lambda: drift.ADWIN(delta=cal11["ADWIN"][0]), "r"),
                      ("Page-Hinkley, calibrated", lambda: drift.PageHinkley(threshold=cal11["Page-Hinkley"][0]), "r")]:
    xs = np.r_[y11[col].values, y12.r.values if col == "r" else rc12]
    al = np.array(dt.run_detector(mk(), xs))
    post = al[al >= n11] - n11
    print(f"  {name}: first signal after {post[0] + 1 if len(post) else '-'} hours of 2012")
FIG["bike"] = {
    "dates": [str(x.date()) for x in list(d11.index) + list(d12.index)],
    "daily": [float(v) for v in list(d11.values) + list(d12.values)],
    "adwin11": adwin11, "cusum11": sorted(set(incontrol["CUSUM"])), "shewhart11": sorted(set(incontrol["Shewhart"])),
    "first12_cusum": str(d12.index[first12["cusum_np"] - 1].date()) if first12["cusum_np"] else None,
    "sd_hourly_iid": float(r.std(ddof=1) / math.sqrt(24)), "sd_daily": float(d11.std(ddof=1)), "mu": des11["mu"],
}


# ---------------------------------------------------------------------------------------------------
# 6. Elec2
# ---------------------------------------------------------------------------------------------------

header("6. Elec2: a classifier's error stream")
F_EL = ["period", "day", "nswprice", "nswdemand"]
train_days, ref_days = 28, 56
ntr = 48 * train_days
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(elec[F_EL].values[:ntr], up[:ntr])
err = (clf.predict(elec[F_EL].values) != up).astype(int)
mon = err[ntr:]
daily = err.reshape(-1, 48).mean(axis=1)
ndays = len(daily)
print(f"Model: logistic regression on period, day of week, NSW price and NSW demand, fitted to days 1-{train_days} "
      f"(the VIC columns are left out: constant for the first {const_rows // 48} days). Error rate in training "
      f"{err[:ntr].mean():.3f}; on days {train_days + 1}-{ndays} {mon.mean():.3f}")
print(f"  error stream after training: lag-1 autocorrelation {acf(mon, 1):.3f}, lag 48 {acf(mon, 48):.3f}")
dm = daily[train_days:]
pbar = mon.mean()
print(f"  daily error rate: sd {dm.std(ddof=1):.3f} against binomial {math.sqrt(pbar * (1 - pbar) / 48):.3f} "
      f"(ratio {dm.std(ddof=1) / math.sqrt(pbar * (1 - pbar) / 48):.2f}); lag-1 autocorrelation {acf(dm, 1):.3f}")
blocks = dm[: len(dm) // 28 * 28].reshape(-1, 28).mean(axis=1)
print("  error rate by 28-day block after training:", " ".join(f"{v:.2f}" for v in blocks))

rng = RNG["elec_perm"]
mon_days = mon[: len(mon) // 48 * 48]
years = len(mon_days) / 48 / 365


def within_day(e):
    x_ = e.reshape(-1, 48).copy()
    for row in x_:
        rng.shuffle(row)
    return x_.ravel()


print(f"Stream detectors with default settings on the error stream (days {train_days + 1}-{ndays}, "
      f"{years:.2f} years); signals per year: real order | order shuffled within each day (20 shuffles) | "
      f"fully shuffled (20)")
el_perm = {}
for name, mk in [("DDM", binary.DDM), ("ADWIN", drift.ADWIN), ("Page-Hinkley", drift.PageHinkley)]:
    real = len(dt.run_detector(mk(), mon_days)) / years
    wd = np.mean([len(dt.run_detector(mk(), within_day(mon_days))) for _ in range(20)]) / years
    fp = np.mean([len(dt.run_detector(mk(), rng.permutation(mon_days))) for _ in range(20)]) / years
    el_perm[name] = (real, wd, fp)
    print(f"  {name}: {real:.1f} | {wd:.1f} | {fp:.1f}")
ddm_days = sorted({i // 48 + train_days for i in dt.run_detector(binary.DDM(), mon_days)})
print(f"  DDM's first signal on day {ddm_days[0] + 1}; after each signal it restarts and learns a new reference "
      f"from the next {binary.DDM().warm_start} observations onwards")
FIG["elec_perm"] = el_perm

rng = RNG["elec_gk"]
ph1 = daily[train_days:train_days + ref_days]
mon2 = daily[train_days + ref_days:]
yrs2 = len(mon2) / 365
pb = ph1.mean()
sd_bin = math.sqrt(pb * (1 - pb) / 48)
pchart = int(np.sum(np.abs(mon2 - pb) > 3 * sd_bin))
pchart_ph1 = int(np.sum(np.abs(ph1 - pb) > 3 * sd_bin))
print(f"Daily error rate, Phase I days {train_days + 1}-{train_days + ref_days} (m={ref_days}), monitoring days "
      f"{train_days + ref_days + 1}-{ndays} ({yrs2:.2f} years)")
print(f"  p-chart with binomial sigma (the independence DDM assumes): Phase I mean {pb:.3f}, sigma {sd_bin:.4f}; "
      f"{pchart_ph1} of {ref_days} Phase I days already outside; {pchart} signals while monitoring "
      f"({pchart / yrs2:.0f} a year)")
des_el = budgeted_design(ph1, np.ones(ref_days - 1, dtype=bool), rng)
print(f"  between-day design: phi {des_el['phi']:.3f} (standard error about "
      f"{math.sqrt((1 - des_el['phi'] ** 2) / des_el['m']):.2f}), one-step sd {des_el['es']:.3f}, Shewhart k {des_el['k']:.3f}, "
      f"CUSUM h parametric {des_el['h_p']:.2f}, nonparametric {des_el['h_np']:.2f}")
el_alarm = {}
for which in ["shewhart", "cusum_p", "cusum_np"]:
    al = monitor(des_el, ph1[-1], mon2, which)
    el_alarm[which] = [i + train_days + ref_days for i in al]
    print(f"  {which}: {len(al)} signals ({len(al) / yrs2:.1f} a year); first on day "
          f"{al[0] + train_days + ref_days + 1 if al else '-'}")
print("  (daily charts see only each day's error rate, so shuffling the order inside days cannot change them)")
FIG["elec"] = {"daily": daily.tolist(), "train_days": train_days, "ref_days": ref_days,
               "ddm_days": ddm_days, "cusum_days": el_alarm["cusum_np"], "shewhart_days": el_alarm["shewhart"],
               "pb": pb, "sd_bin": sd_bin, "sd_day": float(ph1.std(ddof=1)), "k": des_el["k"]}

# The reference with constant VIC columns.
ref = elec.iloc[:ntr]
after = elec.iloc[const_rows:const_rows + ntr]
print(f"Batch scores with days 1-{train_days} as the reference and the first {train_days} days after the VIC "
      f"columns start to vary as the current window:")
for c in ["vicprice", "vicdemand", "transfer", "nswdemand"]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(f"  {c}: reference sd {ref[c].std(ddof=0):.4f}, normed Wasserstein "
              f"{dt.wasserstein_normed(ref[c].values, after[c].values):.2f}, KS {dt.ks_statistic(ref[c].values, after[c].values):.3f}")


# ---------------------------------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------------------------------

header("7. The budgets side by side")
print("Bike Sharing 2011 replay, signals per 30 days (budget for the daily charts: 0.082):")
for name in ["ADWIN", "KSWIN", "Page-Hinkley (log)", "Page-Hinkley (counts)"] + METHODS:
    print(f"  {name}: {real_rate[name]:.2f}")
print("Elec2 monitoring, signals per year: " + ", ".join(f"{k} {v[0]:.1f}" for k, v in el_perm.items())
      + f", p-chart {pchart / yrs2:.1f}, "
      + ", ".join(f"{k} {len(v) / yrs2:.1f}" for k, v in el_alarm.items()))

(HERE / "figure_data.json").write_text(json.dumps(FIG, indent=1))
