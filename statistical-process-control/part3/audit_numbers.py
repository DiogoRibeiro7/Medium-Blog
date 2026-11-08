"""Part 3 of the statistical-process-control series: two public production lines, audited.

Every number quoted in the article is printed by this script. Run:

    python audit_numbers.py > output.txt

The script downloads three public files from pinned commits of GitHub mirrors, checks their SHA-256
hashes (computed after converting CRLF line endings to LF, so that mirrors that differ only in line
endings agree), and caches them in part3/data/, which is not committed. Nothing else is downloaded.

A real line cannot be certified to have been in control, so no in-control ARL can be measured on it. What
can be measured is the realized alarm rate: build a chart from a Phase I stretch of data the way a
practitioner would, run it over the rest of the record, and count the points that fall outside its limits.
That is the observable counterpart of the conditional false-alarm probability of Parts 1 and 2. Each audit
below is set beside a reference computed for independent normal data with the same Phase I size and the
same number of evaluated points.

Sections
1. The data: a multistage continuous-flow line (Liveline Technologies, Kaggle 2020; 1 Hz, about four hours) and the
   SECOM semiconductor line (McCann and Johnston, 2008; 1,567 production runs, 590 sensors, pass/fail).
2. Which sigma? Time scales on the continuous line: held readings, autocorrelation, the semivariogram, and
   the realized alarm rate of four textbook chart designs.
3. Which half hour? The Phase I window lottery on the continuous line, against independent-normal and AR(1)
   references.
4. Machines in parallel: the same variable on three parallel machines.
5. Stations in series: regression adjustment of stage-2 measurements on stage-1 measurements, estimated in a
   Phase I window and used outside it.
6. Many sensors, one at a time: individuals charts on every usable SECOM sensor, line-level alarms, Sidak
   limits, and a decomposition of the excess into distribution shape, estimation and drift.
7. Many sensors at once: PCA-based Hotelling T^2 and squared prediction error (SPE) on SECOM, against a
   normal reference with the same correlation matrix.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
DATA = HERE / "data"

SEED = 20261108
CHILDREN = np.random.SeedSequence(SEED).spawn(12)
RNG = {name: np.random.default_rng(CHILDREN[i]) for i, name in enumerate(
    ["lottery_iid", "lottery_ar", "secom_ref", "secom_random", "pca_ref", "spare1", "spare2", "spare3",
     "spare4", "spare5", "spare6", "spare7"])}

FIG: dict = {}  # data for figures.py, written to figure_data.json

P0 = 2 * stats.norm.sf(3.0)   # 0.0026998, the nominal false-alarm probability of a 3-sigma chart
D2_2 = 2 / np.sqrt(np.pi)     # d2 for moving ranges of two, 1.128

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


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def q3(x, qs=(10, 50, 90)) -> str:
    v = np.percentile(np.asarray(x, dtype=float), qs)
    return " / ".join(f"{t:.4f}" for t in v)


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


# ---------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------


def lag1(x: np.ndarray) -> float:
    a, b = x[:-1], x[1:]
    ok = ~np.isnan(a) & ~np.isnan(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def mr_sigma(x: np.ndarray) -> float:
    d = np.abs(np.diff(x))
    return float(np.nanmean(d) / D2_2)


def semivariogram(x: np.ndarray, h: int) -> float:
    d = x[h:] - x[:-h]
    return float(0.5 * np.nanmean(d * d))


def p_cond(z, s, k: float = 3.0):
    """Conditional false-alarm probability of limits mu_hat +/- k sigma_hat (Part 2's p(z, s))."""
    return stats.norm.sf(z + k * s) + stats.norm.cdf(z - k * s)


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
# Section 1: the data
# ---------------------------------------------------------------------------------------------------


def section_data(L: Line, X, y, stamps) -> tuple[list, list]:
    header("1. The data")
    for name in SOURCES:
        p = DATA / name
        print(f"{name}: {p.stat().st_size} bytes, sha256 {SOURCES[name]['sha256'][:16]}... verified")

    print("\nContinuous-flow line (Liveline Technologies; posted on Kaggle in 2020; recorded on 6 March 2019)")
    print(f"rows {L.vals.shape[0]}, columns {len(L.cols)} (time stamp + {L.vals.shape[1]})")
    print(f"first {L.stamps[0]}, last {L.stamps[-1]}, span {L.t[-1] / 3600:.3f} h, minutes indexed {L.nmin}")
    gaps = np.diff(L.t)
    for g in np.unique(gaps):
        print(f"   interval {g:.0f} s: {int(np.sum(gaps == g))}")
    sp_cols = [c for c in L.cols if c.startswith("Stage1") and c.endswith("Setpoint")]
    sp = np.array([L.raw(c) for c in sp_cols])
    paused = ~L.running
    print(f"rows with stage-1 setpoints at 0: {int(paused.sum())}; all 15 stage-1 setpoints 0 on those rows: "
          f"{bool(np.all(sp[:, paused] == 0))}; constant otherwise: {bool(np.all(sp[:, ~paused] == sp[:, [np.argmax(~paused)]]))}")
    edges = np.flatnonzero(np.diff(paused.astype(int)))
    for e in edges:
        print(f"   setpoint change after row {e}: {L.stamps[e]} -> {L.stamps[e + 1]}")
    sp2 = np.array([L.raw(c) for c in L.cols if c.startswith("Stage2") and c.endswith("Setpoint")])
    print(f"stage-2 setpoints constant over the record: {bool(np.all(sp2 == sp2[:, :1]))}")

    use1, use2 = [], []
    print("\nshare of readings exactly 0 (sensor dropouts), by measurement location 0..14")
    for stage, use in ((1, use1), (2, use2)):
        shares = []
        for k in range(15):
            x = L.raw(f"Stage{stage}.Output.Measurement{k}.U.Actual")
            s = float(np.mean(x == 0))
            shares.append(s)
            if s < 0.10:
                use.append(k)
        print(f"   stage {stage}: " + " ".join(f"{s:.3f}" for s in shares))
    print(f"usable (dropout share < 0.10): stage 1 locations {use1} ({len(use1)}), stage 2 locations {use2} ({len(use2)})")
    starts = np.flatnonzero(np.diff(paused.astype(int)) == 1) + 1
    ends = np.flatnonzero(np.diff(paused.astype(int)) == -1)
    allmeans = [L.minute_means(L.meas(stage, k))[0] for stage, ks in ((1, use1), (2, use2)) for k in ks]

    def split_shift(m0, m1):
        j = np.arange(L.nmin)
        out = []
        for mean in allmeans:
            before = mean[(j >= m0 - 10) & (j < m0)]
            after = mean[(j > m1) & (j <= m1 + 10)]
            out.append(abs(np.nanmean(after) - np.nanmean(before)) / np.nanstd(mean, ddof=1))
        return np.array(out)

    base = np.array([np.median(split_shift(t, t)) for t in range(10, L.nmin - 10)])
    print(f"same statistic at every minute of the record (no pause; {len(base)} split points): median over series has "
          f"median {np.median(base):.2f}, 10/90% {np.percentile(base, 10):.2f} / {np.percentile(base, 90):.2f}")
    for a0, a1 in zip(starts, ends):
        m0, m1 = L.minute[a0], L.minute[a1]
        shifts = []
        for stage, ks in ((1, use1), (2, use2)):
            for k in ks:
                mean, _, _ = L.minute_means(L.meas(stage, k))
                j = np.arange(L.nmin)
                before = mean[(j >= m0 - 10) & (j < m0)]
                after = mean[(j > m1) & (j <= m1 + 10)]
                shifts.append((np.nanmean(after) - np.nanmean(before)) / np.nanstd(mean, ddof=1))
        shifts = np.abs(shifts)
        print(f"pause {L.stamps[a0].time()}-{L.stamps[a1].time()} ({a1 - a0 + 1} rows): change in the mean of the 10 minute means "
              f"after vs before, in SDs of the minute means, over the {len(shifts)} usable series: median {np.median(shifts):.2f}, max {shifts.max():.2f}")
    dup = np.array_equal(L.raw("Machine4.Pressure.C.Actual"), L.raw("Machine4.Temperature4.C.Actual"))
    print(f"Machine4.Pressure.C.Actual identical to Machine4.Temperature4.C.Actual: {dup}")

    print("\nSECOM semiconductor line (McCann and Johnston, 2008)")
    print(f"runs {X.shape[0]}, sensors {X.shape[1]}, fail {int(np.sum(y == 1))}, pass {int(np.sum(y == -1))}")
    t = np.array([s.timestamp() for s in stamps])
    print(f"first {stamps[0]}, last {stamps[-1]}, in time order: {bool(np.all(np.diff(t) >= 0))}")
    gaps_h = np.diff(t) / 3600
    print(f"hours between runs, 10/50/90%: {q3(gaps_h)}; longest {gaps_h.max():.2f}")
    months = np.array([s.strftime("%Y-%m") for s in stamps])
    for mo in np.unique(months):
        sel = months == mo
        print(f"   {mo}: runs {int(sel.sum())}, fail share {np.mean(y[sel] == 1):.3f}")
    miss = np.mean(np.isnan(X), axis=0)
    print(f"missing share per sensor, 50/90/99%: {q3(miss, (50, 90, 99))}; sensors >50% missing {int(np.sum(miss > 0.5))}")
    sdall = np.nanstd(X, axis=0)
    print(f"sensors constant over all runs: {int(np.sum(sdall == 0))}")
    return use1, use2


# ---------------------------------------------------------------------------------------------------
# Section 2: which sigma?
# ---------------------------------------------------------------------------------------------------

LAGS = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600]
PHASE1_MIN = 30


def design_rates(L: Line, x: np.ndarray, m0: int, m1: int) -> dict:
    """Realized alarm rates of four designs, Phase I = minutes [m0, m1), Phase II = every other minute."""
    inside = (L.minute >= m0) & (L.minute < m1)
    xi, xo = x[inside], x[~inside]
    xo = xo[~np.isnan(xo)]
    mu = np.nanmean(xi)
    d1 = float(np.mean(np.abs(xo - mu) > 3 * mr_sigma(xi)))
    mean, sd, cnt = L.minute_means(x)
    j = np.arange(L.nmin)
    win = (j >= m0) & (j < m1) & ~np.isnan(mean)
    out = ~((j >= m0) & (j < m1)) & ~np.isnan(mean)
    sw = np.sqrt(np.sum((cnt[win] - 1) * sd[win] ** 2) / np.sum(cnt[win] - 1))
    gm = mean[win].mean()
    d2 = float(np.mean(np.abs(mean[out] - gm) > 3 * sw / np.sqrt(cnt[out])))
    d3 = float(np.mean(np.abs(mean[out] - gm) > 3 * mr_sigma(mean[(j >= m0) & (j < m1)])))
    d4 = float(np.mean(np.abs(mean[out] - gm) > 3 * mean[win].std(ddof=1)))
    return {"D1": d1, "D2": d2, "D3": d3, "D4": d4}


def section_sigma(L: Line, use1, use2) -> list:
    header("2. Which sigma? Time scales on the continuous line")
    series = [(1, k) for k in use1] + [(2, k) for k in use2]
    print("held reading = consecutive 1 Hz readings that are identical; r1 = lag-1 autocorrelation at 1 Hz")
    print("semivariogram gamma(h)/variance at lags (s): " + " ".join(str(h) for h in LAGS))
    held, r1s, vg, ratio_mr, ratio_sub = [], [], [], [], []
    for stage, k in series:
        x = L.meas(stage, k)
        a, b = x[:-1], x[1:]
        ok = ~np.isnan(a) & ~np.isnan(b)
        h = float(np.mean(a[ok] == b[ok]))
        r = lag1(x)
        var = float(np.nanvar(x, ddof=1))
        g = [semivariogram(x, lag) / var for lag in LAGS]
        mean, sd, cnt = L.minute_means(x)
        okm = ~np.isnan(mean)
        sub = np.sqrt(np.sum((cnt[okm] - 1) * sd[okm] ** 2) / np.sum(cnt[okm] - 1)) / np.sqrt(np.median(cnt[okm]))
        rm = mr_sigma(x) / np.sqrt(var)
        rs = sub / np.nanstd(mean, ddof=1)
        held.append(h)
        r1s.append(r)
        vg.append(g)
        ratio_mr.append(rm)
        ratio_sub.append(rs)
        print(f"S{stage}M{k:<2} held {h:.3f} r1 {r:.3f} | " + " ".join(f"{v:.3f}" for v in g)
              + f" | MRsigma/SD {rm:.3f} | within-minute sigma/sqrt(n) over SD of minute means {rs:.3f}")
    vg = np.array(vg)
    n1 = len(use1)
    for label, sel in (("stage 1", slice(0, n1)), ("stage 2", slice(n1, None))):
        print(f"{label}: median held share {np.median(np.array(held)[sel]):.3f}; r1 range "
              f"{min(r1s[sel]):.3f} to {max(r1s[sel]):.3f}")
        print(f"   median normalized semivariogram by lag: " + " ".join(f"{v:.3f}" for v in np.median(vg[sel], axis=0)))
        print(f"   median MRsigma/SD {np.median(np.array(ratio_mr)[sel]):.3f}; median within-minute ratio "
              f"{np.median(np.array(ratio_sub)[sel]):.3f}")
    late_mr, late_g1 = [], []
    for stage, k in series:
        x = L.meas(stage, k).copy()
        x[L.minute < PHASE1_MIN] = np.nan
        v = float(np.nanvar(x, ddof=1))
        late_mr.append(mr_sigma(x) / np.sqrt(v))
        late_g1.append(semivariogram(x, 1) / v)
    late_mr, late_g1 = np.array(late_mr), np.array(late_g1)
    print(f"excluding the first {PHASE1_MIN} minutes: median MRsigma/SD stage 1 {np.median(late_mr[:n1]):.3f}, stage 2 "
          f"{np.median(late_mr[n1:]):.3f}; median gamma(1)/var stage 1 {np.median(late_g1[:n1]):.3f}, stage 2 {np.median(late_g1[n1:]):.3f}")
    print(f"all {len(series)}: gamma(1)/var range {vg[:, 0].min():.3f} to {vg[:, 0].max():.3f}, median {np.median(vg[:, 0]):.3f}; "
          f"gamma(3600)/var median {np.median(vg[:, -1]):.3f}")
    print(f"limits from MR at 1 Hz are narrower than 3 SD by a median factor {np.median(ratio_mr):.3f}; "
          f"3 x that factor = {3 * np.median(ratio_mr):.3f} SD, "
          f"normal tail beyond it {2 * stats.norm.sf(3 * np.median(ratio_mr)):.3f}")

    print(f"\nFour designs, Phase I = first {PHASE1_MIN} minutes, evaluated on every later point")
    print("D1 individuals at 1 Hz, sigma from the moving range; D2 X-bar of one-minute subgroups, sigma pooled")
    print("within minutes; D3 individuals on minute means, sigma from their moving range; D4 individuals on")
    print("minute means, sigma = SD of the Phase I minute means. Nominal rate for each: "
          f"{P0:.4f}")
    rates = {d: [] for d in ("D1", "D2", "D3", "D4")}
    spread = []
    for stage, k in series:
        x = L.meas(stage, k)
        r = design_rates(L, x, 0, PHASE1_MIN)
        for d in rates:
            rates[d].append(r[d])
        mean, _, _ = L.minute_means(x)
        blocks = [np.nanstd(mean[s:s + PHASE1_MIN], ddof=1) for s in range(PHASE1_MIN, L.nmin - PHASE1_MIN + 1, PHASE1_MIN)]
        spread.append(np.nanstd(mean[:PHASE1_MIN], ddof=1) / np.median(blocks))
        print(f"S{stage}M{k:<2} " + " ".join(f"{d} {r[d]:.3f}" for d in rates)
              + f" | SD of minute means, first 30 min / median later 30-min block {spread[-1]:.2f}")
    n1 = len(use1)
    print(f"first-30-minute spread ratio, median: stage 1 {np.median(spread[:n1]):.2f}, stage 2 {np.median(spread[n1:]):.2f}")
    for d, v in rates.items():
        v = np.array(v)
        print(f"{d}: median {np.median(v):.4f}, min {v.min():.4f}, max {v.max():.4f}, "
              f"median / nominal {np.median(v) / P0:.0f}, measurements above 10x nominal {int(np.sum(v > 10 * P0))} of {len(v)}")
    FIG["sigma"] = {
        "lags": LAGS, "labels": [f"S{s}M{k}" for s, k in series], "stage": [s for s, _ in series],
        "variogram": vg.tolist(), "designs": {d: list(map(float, v)) for d, v in rates.items()}, "p0": P0,
    }
    return series


# ---------------------------------------------------------------------------------------------------
# Section 3: which half hour?
# ---------------------------------------------------------------------------------------------------


def ar1_fit(prev, cur):
    X = np.column_stack([np.ones_like(prev), prev])
    b, *_ = np.linalg.lstsq(X, cur, rcond=None)
    r = cur - X @ b
    return b, float(np.sqrt(np.sum(r * r) / (len(cur) - 2)))


def lottery_real(L: Line, series, m: int, step: int = 5):
    out = {"SD": [], "MR": [], "AR1": []}
    phis = []
    for stage, k in series:
        mean, _, _ = L.minute_means(L.meas(stage, k))
        j = np.arange(L.nmin)
        for s0 in range(0, L.nmin - m + 1, step):
            w = (j >= s0) & (j < s0 + m)
            vw = mean[w]
            if np.sum(~np.isnan(vw)) < 0.9 * m:
                continue
            o = ~w & ~np.isnan(mean)
            mu = np.nanmean(vw)
            out["SD"].append(float(np.mean(np.abs(mean[o] - mu) > 3 * np.nanstd(vw, ddof=1))))
            out["MR"].append(float(np.mean(np.abs(mean[o] - mu) > 3 * mr_sigma(vw))))
            prev, cur = mean[:-1], mean[1:]
            pair_ok = ~np.isnan(prev) & ~np.isnan(cur)
            tr = pair_ok & w[:-1] & w[1:]
            te = pair_ok & ~w[:-1] & ~w[1:]
            b, sg = ar1_fit(prev[tr], cur[tr])
            phis.append(b[1])
            res = cur[te] - (b[0] + b[1] * prev[te])
            out["AR1"].append(float(np.mean(np.abs(res) > 3 * sg)))
    return {d: np.array(v) for d, v in out.items()}, np.array(phis)


def lottery_reference(rng, m: int, n_out: int, reps: int, design: str, phi: float = 0.0):
    rates = np.empty(reps)
    for r in range(reps):
        if design == "AR1":
            e = rng.standard_normal(m + n_out + 200)
            x = np.empty_like(e)
            x[0] = e[0] / np.sqrt(1 - phi * phi)
            for t in range(1, len(e)):
                x[t] = phi * x[t - 1] + e[t]
            x = x[200:]
            b, sg = ar1_fit(x[:m - 1], x[1:m])
            res = x[m + 1:] - (b[0] + b[1] * x[m:-1])
            rates[r] = np.mean(np.abs(res) > 3 * sg)
        else:
            x = rng.standard_normal(m + n_out)
            ph1, ph2 = x[:m], x[m:]
            sg = ph1.std(ddof=1) if design == "SD" else mr_sigma(ph1)
            rates[r] = np.mean(np.abs(ph2 - ph1.mean()) > 3 * sg)
    return rates


def summarize(v: np.ndarray) -> str:
    return (f"10/50/90% {q3(v)}; zero {np.mean(v == 0):.3f}; >1% {np.mean(v > 0.01):.3f}; "
            f">10% {np.mean(v > 0.10):.3f}")


def section_lottery(L: Line, series) -> None:
    header("3. Which half hour? The Phase I window lottery on minute means")
    print("Phase I = m consecutive minutes (window start every 5 minutes, at least 90% of minutes present);")
    print("the chart is evaluated on every other minute of the record. References: independent normal")
    print("minute means (SD and MR designs) and an AR(1) with phi = 0.7 (residual design), with the same m")
    print(f"and {L.nmin} - m evaluated minutes.")
    fig = {}
    for m in (30, 60, 120):
        real, phis = lottery_real(L, series, m)
        n_out = L.nmin - m
        print(f"\nm = {m} minutes: {len(real['SD'])} window x measurement pairs; fitted AR(1) phi median {np.median(phis):.3f}, "
              f"10/90% {np.percentile(phis, 10):.3f} / {np.percentile(phis, 90):.3f}")
        z = RNG["lottery_iid"].normal(0, 1 / np.sqrt(m), 400000)
        s = np.sqrt(RNG["lottery_iid"].chisquare(m - 1, 400000) / (m - 1))
        pc = p_cond(z, s)
        print(f"   conditional false-alarm probability, independent normal, sigma from SD: {q3(pc)}")
        fig[m] = {}
        for d in ("SD", "MR", "AR1"):
            ref = lottery_reference(RNG["lottery_ar"] if d == "AR1" else RNG["lottery_iid"], m, n_out,
                                    4000 if d == "AR1" else 20000, d, phi=0.7)
            print(f"   {d:3s} real:      {summarize(real[d])}")
            print(f"   {d:3s} reference: {summarize(ref)}")
            fig[m][d] = {"real": np.percentile(real[d], [10, 25, 50, 75, 90]).tolist(),
                         "ref": np.percentile(ref, [10, 25, 50, 75, 90]).tolist()}
    FIG["lottery"] = fig


# ---------------------------------------------------------------------------------------------------
# Section 4: machines in parallel
# ---------------------------------------------------------------------------------------------------

PARALLEL_VARS = ["Zone1Temperature.C.Actual", "Zone2Temperature.C.Actual", "MotorAmperage.U.Actual",
                 "MotorRPM.C.Actual", "MaterialPressure.U.Actual", "MaterialTemperature.U.Actual",
                 "ExitZoneTemperature.C.Actual", "RawMaterialFeederParameter.U.Actual"]


def section_parallel(L: Line) -> None:
    header("4. Machines in parallel: one variable, three machines")
    print("per machine: SD over the record, sigma from the 1 Hz moving range, share of held readings, lag-1")
    print(f"autocorrelation, and the realized alarm rate of a 1 Hz individuals chart (MR sigma, Phase I = first {PHASE1_MIN} min)")
    for v in PARALLEL_VARS:
        sds, mrs, rates = [], [], []
        parts = []
        for mach in (1, 2, 3):
            x = L.var(f"Machine{mach}.{v}")
            sd = float(np.nanstd(x, ddof=1))
            ms = mr_sigma(x)
            a, b = x[:-1], x[1:]
            ok = ~np.isnan(a) & ~np.isnan(b)
            held = float(np.mean(a[ok] == b[ok]))
            inside = L.minute < PHASE1_MIN
            xi, xo = x[inside], x[~inside & ~np.isnan(x)]
            si = mr_sigma(xi)
            rate = float(np.mean(np.abs(xo - np.nanmean(xi)) > 3 * si)) if si > 0 else float("nan")
            sds.append(sd)
            mrs.append(ms)
            rates.append(rate)
            parts.append(f"M{mach}: SD {sd:.4g} MRsig {ms:.4g} held {held:.3f} r1 {lag1(x):.3f} alarm {rate:.3f}")
        print(f"{v}")
        for p in parts:
            print("   " + p)
        print(f"   max/min across machines: SD {max(sds) / min(sds):.2f}, MR sigma {max(mrs) / min(mrs):.1f}")
    print("\nraw material property lots (distinct values over the record)")
    for mach in (1, 2, 3):
        nd = [len(np.unique(L.raw(f"Machine{mach}.RawMaterial.Property{p}"))) for p in (1, 2, 3, 4)]
        print(f"   machine {mach}: properties 1-4 distinct values {nd}")


# ---------------------------------------------------------------------------------------------------
# Section 5: stations in series
# ---------------------------------------------------------------------------------------------------


def section_series(L: Line, use1, use2) -> None:
    header("5. Stations in series: adjusting stage 2 for stage 1 with an estimated regression")
    mm1 = {k: L.minute_means(L.meas(1, k))[0] for k in use1}
    mm2 = {k: L.minute_means(L.meas(2, k))[0] for k in use2}
    both = [k for k in use1 if k in use2]
    print(f"locations usable at both stages: {both}")

    def corr_lag(a, b, lag):
        if lag > 0:
            a, b = a[:-lag], b[lag:]
        ok = ~np.isnan(a) & ~np.isnan(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    matched = [corr_lag(mm1[k], mm2[k], 0) for k in both]
    unmatched = [corr_lag(mm1[i], mm2[j], 0) for i in use1 for j in use2 if i != j]
    print("minute means, lag 0: matched corr " + " ".join(f"{v:.3f}" for v in matched))
    print(f"   median |corr| matched {np.median(np.abs(matched)):.3f}; unmatched pairs ({len(unmatched)}) "
          f"median |corr| {np.median(np.abs(unmatched)):.3f}, 90% {np.percentile(np.abs(unmatched), 90):.3f}, "
          f"max {np.max(np.abs(unmatched)):.3f}")
    for k in both:
        cs = [(corr_lag(mm1[k], mm2[k], lag), lag) for lag in range(0, 61)]
        best = max(cs, key=lambda t: abs(t[0]))
        print(f"   location {k}: lag (min) with largest |corr| over 0..60 = {best[1]} (corr {best[0]:.3f})")

    print("\nregression of the stage-2 minute mean on stage-1 minute means (lagged), fitted on a window of m")
    print("minutes (start every 10 min), used on every other minute. 'worse' = out-of-window RMS residual larger")
    print("than out-of-window RMS deviation from the window mean (no adjustment). Alarm = |residual| > 3 x in-window")
    print("residual SD (adjusted) or |deviation| > 3 x in-window SD (unadjusted).")
    j = np.arange(L.nmin)
    for spec in ("matched", "stage1_all"):
        for lag in (0, 5, 15):
            for m in (60, 120):
                stats_ = []
                locs = both if spec == "matched" else use2
                for k in locs:
                    yv = mm2[k]
                    preds = [mm1[k]] if spec == "matched" else [mm1[i] for i in use1]
                    Xv = np.column_stack([np.r_[np.full(lag, np.nan), p[:L.nmin - lag]] for p in preds])
                    ok = ~np.isnan(yv) & np.all(~np.isnan(Xv), axis=1)
                    first = int(j[ok].min())
                    for s0 in range(first, int(j[ok].max()) - m - 29, 10):
                        tr = ok & (j >= s0) & (j < s0 + m)
                        te = ok & ~((j >= s0) & (j < s0 + m))
                        if tr.sum() < 0.9 * m:
                            continue
                        A = np.column_stack([np.ones(tr.sum()), Xv[tr]])
                        B = np.column_stack([np.ones(te.sum()), Xv[te]])
                        b, *_ = np.linalg.lstsq(A, yv[tr], rcond=None)
                        p = A.shape[1] - 1
                        rin = yv[tr] - A @ b
                        s_in = np.sqrt(np.sum(rin * rin) / (tr.sum() - p - 1))
                        rout = yv[te] - B @ b
                        dev = yv[te] - yv[tr].mean()
                        r2 = 1 - np.sum(rin * rin) / np.sum((yv[tr] - yv[tr].mean()) ** 2)
                        stats_.append((r2, np.sqrt(np.mean(rout ** 2)) > np.sqrt(np.mean(dev ** 2)),
                                       np.mean(np.abs(rout) > 3 * s_in), np.mean(np.abs(dev) > 3 * yv[tr].std(ddof=1))))
                s = np.array(stats_, dtype=float)
                print(f"{spec:10s} lag {lag:2d} min, m {m:3d}, predictors {p}: pairs {len(s)}; in-window R^2 median {np.median(s[:, 0]):.2f}; "
                      f"worse {np.mean(s[:, 1]):.2f}; alarm median adjusted {np.median(s[:, 2]):.3f} vs unadjusted {np.median(s[:, 3]):.3f}")
    print("\nindependent-normal reference for the out-of-sample inflation of the residual SD: sqrt(1 + (p + 1)/m) for a")
    print("fixed design; sqrt((1 + 1/m)(m - 2)/(m - p - 2)) for Gaussian predictors")
    for p in (1, 8):
        for m in (60, 120):
            print(f"   p {p}, m {m}: {np.sqrt(1 + (p + 1) / m):.3f} / {np.sqrt((1 + 1 / m) * (m - 2) / (m - p - 2)):.3f}")


# ---------------------------------------------------------------------------------------------------
# Section 6: SECOM, one chart per sensor
# ---------------------------------------------------------------------------------------------------


def secom_limits(X, rows):
    P = X[rows]
    with warnings.catch_warnings():  # sensors entirely missing in Phase I give NaN, and are not charted
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(P, axis=0)
        sd = np.nanstd(P, axis=0, ddof=1)
    nobs = np.sum(~np.isnan(P), axis=0)
    usable = (nobs >= 0.8 * len(rows)) & (sd > 0)
    return mu, sd, usable, nobs


def secom_rates(X, mu, sd, usable, rows, k=3.0):
    Z = np.abs(X[np.ix_(rows, usable)] - mu[usable]) / sd[usable]
    A = Z > k
    V = ~np.isnan(Z)
    per = A.sum(axis=0) / np.maximum(V.sum(axis=0), 1)
    return per, A


def section_secom_univariate(X, y, stamps) -> None:
    header("6. Many sensors, one chart each: SECOM")
    fail = y == 1
    passi = np.flatnonzero(~fail)
    print("Phase I = the first m passing runs; Phase II = every run after the last Phase I run. A sensor is")
    print("charted if it is present in at least 80% of Phase I runs and not constant there. Limits = Phase I")
    print("mean +/- 3 Phase I SD. Missing readings raise no alarm.")
    for m in (50, 100, 200, 400):
        rows = passi[:m]
        last = rows[-1]
        mu, sd, usable, nobs = secom_limits(X, rows)
        const = (nobs >= 0.8 * m) & (sd == 0)
        later = np.arange(len(y)) > last
        varies = const & (np.nanstd(X[later], axis=0) > 0)
        K = int(usable.sum())
        pr, fr = np.flatnonzero(later & ~fail), np.flatnonzero(later & fail)
        per, A = secom_rates(X, mu, sd, usable, pr)
        perf, Af = secom_rates(X, mu, sd, usable, fr)
        ks = stats.norm.isf((1 - (1 - P0) ** (1 / K)) / 2)
        pers, As = secom_rates(X, mu, sd, usable, pr, ks)
        _, Afs = secom_rates(X, mu, sd, usable, fr, ks)
        print(f"\nm = {m}: Phase I ends {stamps[last]}; charted sensors K = {K}; constant in Phase I {int(const.sum())}, "
              f"of which vary later {int(varies.sum())}; Phase II passing {len(pr)}, failing {len(fr)}")
        print(f"   per-sensor realized rate on passing runs, 10/25/50/75/90%: {q3(per, (10, 25, 50, 75, 90))}; mean {per.mean():.4f}")
        print(f"   sensors above nominal {np.mean(per > P0):.3f}; above 10x nominal {np.mean(per > 10 * P0):.3f}")
        print(f"   runs with at least one alarm: passing {A.any(axis=1).mean():.3f}, failing {Af.any(axis=1).mean():.3f}; "
              f"nominal with known parameters 1-(1-p0)^K = {1 - (1 - P0) ** K:.3f}")
        print(f"   alarms per run, mean: passing {A.sum(axis=1).mean():.2f}, failing {Af.sum(axis=1).mean():.2f}; "
              f"median passing {np.median(A.sum(axis=1)):.0f}; nominal K x p0 = {K * P0:.2f}")
        print(f"   Sidak limits k = {ks:.3f} (line nominal {P0:.4f}): runs with an alarm, passing {As.any(axis=1).mean():.3f}, "
              f"failing {Afs.any(axis=1).mean():.3f}; per-sensor median rate {np.median(pers):.4f} vs nominal per sensor "
              f"{2 * stats.norm.sf(ks):.2e}")
        a_p, a_f = int(As.any(axis=1).sum()), int(Afs.any(axis=1).sum())
        p_fisher = stats.fisher_exact([[a_f, len(fr) - a_f], [a_p, len(pr) - a_p]]).pvalue
        print(f"   Sidak alarms: failing {a_f}/{len(fr)}, passing {a_p}/{len(pr)}; Fisher exact two-sided p = {p_fisher:.3g}")

    m = 100
    print(f"\nDecomposition at m = {m}: where does the excess come from?")
    rows = passi[:m]
    mu, sd, usable, _ = secom_limits(X, rows)
    pr = np.flatnonzero((np.arange(len(y)) > rows[-1]) & ~fail)
    first, A_first = secom_rates(X, mu, sd, usable, pr)
    K = int(usable.sum())
    n_eval = len(pr)
    rng = RNG["secom_ref"]
    ref = []
    ref_any = []
    for _ in range(20):
        ph1 = rng.standard_normal((m, K))
        ph2 = rng.standard_normal((n_eval, K))
        z = np.abs(ph2 - ph1.mean(axis=0)) > 3 * ph1.std(axis=0, ddof=1)
        ref.append(z.mean(axis=0))
        ref_any.append(z.any(axis=1).mean())
    ref = np.concatenate(ref)
    print(f"   (a) independent normal sensors, same m, K = {K}, {n_eval} evaluated runs (20 replicates):")
    print(f"       per-sensor 10/50/90% {q3(ref)}; runs with an alarm {np.mean(ref_any):.3f}")
    allpass = np.flatnonzero(~fail)
    mu_o, sd_o, use_o, _ = secom_limits(X, allpass)
    orc, A_o = secom_rates(X, mu_o, sd_o, use_o, allpass)
    print(f"   (b) oracle: limits from all {len(allpass)} passing runs, evaluated on the same runs, K = {int(use_o.sum())}:")
    print(f"       per-sensor 10/50/90% {q3(orc)}; runs with an alarm {A_o.any(axis=1).mean():.3f}")
    rnd, rnd_any = [], []
    rng = RNG["secom_random"]
    for _ in range(100):
        sel = np.sort(rng.choice(len(allpass), m, replace=False))
        rest = np.setdiff1d(np.arange(len(allpass)), sel)
        mu_r, sd_r, use_r, _ = secom_limits(X, allpass[sel])
        per_r, A_r = secom_rates(X, mu_r, sd_r, use_r, allpass[rest])
        rnd.append(per_r)
        rnd_any.append(A_r.any(axis=1).mean())
    rnd_all = np.concatenate(rnd)
    print(f"   (c) Phase I = {m} passing runs drawn at random from the whole period (100 draws), evaluated on the")
    print(f"       other passing runs: per-sensor 10/50/90% {q3(rnd_all)}; runs with an alarm, median over draws "
          f"{np.median(rnd_any):.3f}")
    print(f"   (d) Phase I = the first {m} passing runs: per-sensor 10/50/90% {q3(first)}; runs with an alarm "
          f"{A_first.any(axis=1).mean():.3f}")

    kurt = np.array([stats.kurtosis(c[~np.isnan(c)]) for c in X[np.ix_(allpass, usable)].T])
    nu = 2 / (2 / (m - 1) + kurt / m)
    print(f"\nexcess kurtosis of the {K} sensors charted at m = {m}, computed over all passing runs, 25/50/75/90%: "
          f"{q3(kurt, (25, 50, 75, 90))}")
    print(f"matching Var(S^2) to a chi-square (Shoemaker's adjusted degrees of freedom): nu = 2/(2/(m-1) + kurtosis/m) at m = {m}")
    nu_c = np.where(nu > 0, nu, np.inf)
    print(f"   10/50/90%: {q3(nu_c)}; share with nu <= 9: {np.mean(nu_c <= 9):.3f}; share with nu <= 2: {np.mean(nu_c <= 2):.3f}")
    print(f"   (negative excess kurtosis gives nu above m - 1: share {np.mean(kurt < 0):.3f})")
    P1 = X[np.ix_(rows, usable)]
    top2 = []
    for col in P1.T:
        v = col[~np.isnan(col)]
        d = np.sort((v - v.mean()) ** 2)[::-1]
        top2.append(d[:2].sum() / d.sum())
    top2 = np.array(top2)
    low = nu_c <= 2
    print(f"   share of the Phase I sum of squared deviations from the two most extreme of the first {m} runs:")
    print(f"   sensors with nu <= 2 ({int(low.sum())}): median {np.median(top2[low]):.3f}, share above one half {np.mean(top2[low] > 0.5):.3f}; "
          f"other sensors: median {np.median(top2[~low]):.3f}; independent normal with m = {m}, expected about "
          f"{np.mean([np.sort((z - z.mean()) ** 2)[::-1][:2].sum() / ((z - z.mean()) ** 2).sum() for z in RNG['spare1'].standard_normal((2000, m))]):.3f}")

    kgrid = np.round(np.arange(3.0, 8.01, 0.25), 2)
    any_real, any_nom = [], []
    for k in kgrid:
        _, Ak = secom_rates(X, mu, sd, usable, pr, k)
        any_real.append(float(Ak.any(axis=1).mean()))
        any_nom.append(float(1 - (1 - 2 * stats.norm.sf(k)) ** K))
    print("\nlimit width k against the share of passing runs with an alarm (first-100 limits):")
    print("   " + " ".join(f"{k:.2f}:{a:.3f}" for k, a in zip(kgrid, any_real)))
    FIG["secom_uni"] = {"ref": ref.tolist(), "oracle": orc.tolist(), "random": rnd_all.tolist(),
                        "first": first.tolist(), "p0": P0, "kgrid": kgrid.tolist(), "any_real": any_real,
                        "any_nom": any_nom, "K": K, "sidak": float(stats.norm.isf((1 - (1 - P0) ** (1 / K)) / 2))}


# ---------------------------------------------------------------------------------------------------
# Section 7: SECOM, PCA-based T^2 and SPE
# ---------------------------------------------------------------------------------------------------

KS = (2, 5, 10, 20, 50)


def pca_monitor(Z1, Zeval: dict, ks=KS, alpha=P0):
    """Fit PCA on standardized Phase I rows Z1; return alarm rates of T^2 and SPE on each row set."""
    m = Z1.shape[0]
    _, sv, Vt = np.linalg.svd(Z1 - Z1.mean(axis=0), full_matrices=False)
    lam = sv ** 2 / (m - 1)
    out = {}
    for k in ks:
        V = Vt[:k].T
        lim = k * (m + 1) * (m - 1) / (m * (m - k)) * stats.f.isf(alpha, k, m - k)
        th = [float(np.sum(lam[k:] ** i)) for i in (1, 2, 3)]
        h0 = 1 - 2 * th[0] * th[2] / (3 * th[1] ** 2)   # Jackson-Mudholkar exponent, reported only
        g, h = th[1] / th[0], th[0] ** 2 / th[1]        # Box (1954): SPE ~ g chi^2_h
        qlim = g * stats.chi2.isf(alpha, h)
        res = {"h0": h0}
        for name, Z in Zeval.items():
            T = Z @ V
            t2 = np.sum(T * T / lam[:k], axis=1)
            R = Z - T @ V.T
            spe = np.sum(R * R, axis=1)
            res[name] = (float(np.mean(t2 > lim)), float(np.mean(spe > qlim)))
        out[k] = res
    return out, lam, Vt


def standardize(X, mu, sd, use):
    return np.nan_to_num((X[:, use] - mu[use]) / sd[use])


def section_secom_pca(X, y) -> None:
    header("7. Many sensors at once: PCA-based T^2 and SPE on SECOM")
    print("Phase I = first m passing runs, standardized by their own means and SDs (missing set to the Phase I")
    print("mean), PCA on the Phase I correlation matrix, k components. T^2 limit: the Phase II F limit for k")
    print("variables with estimated mean and covariance; SPE limit: g chi^2_h with g = theta2/theta1 and")
    print("h = theta1^2/theta2 from the discarded Phase I eigenvalues (Box 1954). Both at alpha = "
          f"{P0:.4f}.")
    print("Rates: share of runs above the limit. h0 is the exponent of the Jackson-Mudholkar (1979)")
    print("approximation, which is undefined or unstable when h0 is near or below zero.")
    fail = y == 1
    passi = np.flatnonzero(~fail)
    fig = {"ks": list(KS), "real": {}, "ref": {}}
    allpass = passi
    mu_o, sd_o, use_o, _ = secom_limits(X, allpass)
    Zall = standardize(X[allpass], mu_o, sd_o, use_o)
    R = np.corrcoef(Zall, rowvar=False)
    w, Q = np.linalg.eigh(R)
    w = np.clip(w, 0, None)
    root = Q * np.sqrt(w)
    print(f"normal reference: Gaussian rows with the correlation matrix of all {len(allpass)} passing runs "
          f"(p = {int(use_o.sum())}), same m and number of evaluated rows, 10 replicates")
    for m in (100, 200, 400):
        rows = passi[:m]
        mu, sd, use, _ = secom_limits(X, rows)
        later = np.arange(len(y)) > rows[-1]
        Z1 = standardize(X[rows], mu, sd, use)
        Zp = standardize(X[later & ~fail], mu, sd, use)
        Zf = standardize(X[later & fail], mu, sd, use)
        out, lam, Vt = pca_monitor(Z1, {"phase1": Z1, "pass": Zp, "fail": Zf})
        print(f"\nm = {m}, p = {int(use.sum())} (Phase I rank at most {m - 1}); Phase II passing {Zp.shape[0]}, failing {Zf.shape[0]}")
        for k in KS:
            r = out[k]
            print(f"   k {k:2d}: T2 phase I {r['phase1'][0]:.3f} pass {r['pass'][0]:.3f} fail {r['fail'][0]:.3f} | "
                  f"SPE phase I {r['phase1'][1]:.3f} pass {r['pass'][1]:.3f} fail {r['fail'][1]:.3f} | "
                  f"explained {lam[:k].sum() / lam.sum():.2f} | h0 {r['h0']:.3f}")
        sc = (Zp - Z1.mean(axis=0)) @ Vt[:50].T
        ratio = np.median(sc * sc, axis=0) / stats.chi2.median(1) / lam[:50]
        print("   robust out-of-sample variance / Phase I eigenvalue, PCs 1 2 5 10 20 50: "
              + " ".join(f"{ratio[i]:.2f}" for i in (0, 1, 4, 9, 19, 49)))
        for k in (5, 20):
            V = Vt[:k].T
            c1, cp = Z1 - Z1.mean(axis=0), Zp - Z1.mean(axis=0)
            s1 = np.sum((c1 - c1 @ V @ V.T) ** 2, axis=1)
            s2 = np.sum((cp - cp @ V @ V.T) ** 2, axis=1)
            print(f"   k {k}: median SPE phase I {np.median(s1):.1f}, Phase II passing {np.median(s2):.1f}, ratio {np.median(s2) / np.median(s1):.2f}")
        fig["real"][m] = {k: {s: out[k][s] for s in ("phase1", "pass", "fail")} for k in KS}
        n_eval = Zp.shape[0]
        reps, ratios = [], []
        for _ in range(10):
            G1 = RNG["pca_ref"].standard_normal((m, root.shape[1])) @ root.T
            G2 = RNG["pca_ref"].standard_normal((n_eval, root.shape[1])) @ root.T
            g_mu, g_sd = G1.mean(axis=0), G1.std(axis=0, ddof=1)
            Z1g, Z2g = (G1 - g_mu) / g_sd, (G2 - g_mu) / g_sd
            o, lam_g, Vt_g = pca_monitor(Z1g, {"pass": Z2g})
            reps.append([[o[k]["pass"][0], o[k]["pass"][1]] for k in KS])
            scg = (Z2g - Z1g.mean(axis=0)) @ Vt_g[:50].T
            ratios.append(np.median(scg * scg, axis=0) / stats.chi2.median(1) / lam_g[:50])
        reps = np.array(reps)
        med = np.median(reps, axis=0)
        print("   normal reference (median of 10): " + " ".join(f"k {k}: T2 {med[i, 0]:.4f} SPE {med[i, 1]:.4f}" for i, k in enumerate(KS)))
        rg = np.median(np.array(ratios), axis=0)
        print("   normal reference, robust out-of-sample variance / eigenvalue, PCs 1 2 5 10 20 50: "
              + " ".join(f"{rg[i]:.2f}" for i in (0, 1, 4, 9, 19, 49)))
        fig["ref"][m] = med.tolist()
    m = 100
    print(f"\nPhase I = {m} passing runs drawn at random from the whole period (20 draws), evaluated on the other")
    print("passing runs: same procedure; medians over draws")
    rates, ratios = [], []
    for _ in range(20):
        sel = np.sort(RNG["spare2"].choice(len(passi), m, replace=False))
        rest = np.setdiff1d(np.arange(len(passi)), sel)
        mu, sd, use, _ = secom_limits(X, passi[sel])
        Z1 = standardize(X[passi[sel]], mu, sd, use)
        Zr = standardize(X[passi[rest]], mu, sd, use)
        o, lam, Vt = pca_monitor(Z1, {"pass": Zr})
        rates.append([[o[k]["pass"][0], o[k]["pass"][1]] for k in KS])
        sc = (Zr - Z1.mean(axis=0)) @ Vt[:50].T
        ratios.append(np.median(sc * sc, axis=0) / stats.chi2.median(1) / lam[:50])
    med = np.median(np.array(rates), axis=0)
    rr = np.median(np.array(ratios), axis=0)
    print("   " + " ".join(f"k {k}: T2 {med[i, 0]:.3f} SPE {med[i, 1]:.3f}" for i, k in enumerate(KS)))
    print("   robust out-of-sample variance / Phase I eigenvalue, PCs 1 2 5 10 20 50: "
          + " ".join(f"{rr[i]:.2f}" for i in (0, 1, 4, 9, 19, 49)))
    FIG["pca"] = fig


def main() -> None:
    print("Statistical process control, Part 3: two public production lines, audited")
    print(f"seed {SEED}; numpy {np.__version__}")
    L = Line()
    X, y, stamps = load_secom()
    use1, use2 = section_data(L, X, y, stamps)
    series = section_sigma(L, use1, use2)
    section_lottery(L, series)
    section_parallel(L)
    section_series(L, use1, use2)
    section_secom_univariate(X, y, stamps)
    section_secom_pca(X, y)
    (HERE / "figure_data.json").write_text(json.dumps(FIG))


if __name__ == "__main__":
    main()
