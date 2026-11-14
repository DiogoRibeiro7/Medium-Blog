"""Valuation models with and without the owner's nominal gain, evaluated out of time.

    python model.py --pairs /scratch/pairs.parquet --data data --out results

Every resale in the pairs file is valued at its sale date by three models.

- M0, the indexed valuation: the previous price carried forward by the postcode area's
  repeat-sales index. Its log estimate is lv = log(previous price) + m, where m is the change
  in the area index between the two sales: the nominal gain the index implies for this owner.
- M1, a gradient-boosted correction to M0 that sees the indexed value, the home's type,
  tenure, holding period, postcode area, its place in the area's price distribution and how
  its sector and district have done against the area index in the three previous years. It
  does not see m, so it cannot tell where the indexed value sits relative to the purchase
  price.
- M2, the same model with m added. With lv fixed, a different m means a different purchase
  price, so M2 can learn whether the price the owner paid matters for the price they get.

M1 and M2 predict the log error of M0, r = log(sale price) - lv. For each, a second pair of
models predicts the 10th and 90th percentiles of r, and conformalised quantile regression on a
calibration year turns them into 80% intervals, either with one correction for all homes
(global) or with one per band of m (Mondrian).

Three splits, each trained only on resales before its test period:

    test 2008-2012: train 1996-2006, calibrate 2007
    test 2017-2019: train 1996-2015, calibrate 2016
    test 2022-2025: train 1996-2020, calibrate 2021

Only aggregates are written (``results/*.csv``): metrics by group, binned curves, interval
coverage, partial dependence on m, feature importance and local-linear estimates of a jump or
kink in M0's error at m = 0 and at placebo values of m.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

SEED = 20261113
SPLITS = [("2008-2012", 2006, 2007, 2008, 2012),
          ("2017-2019", 2015, 2016, 2017, 2019),
          ("2022-2025", 2020, 2021, 2022, 2025)]
FIRST_TRAIN_YEAR = 1996
PERIODS = [("1996-2007", 1996, 2007), ("2008-2012", 2008, 2012), ("2013-2021", 2013, 2021),
           ("2022-2025", 2022, 2025)]
MBAND_EDGES = [-np.inf, -0.10, 0.0, 0.10, 0.25, np.inf]
MBAND_LABELS = ["below -10%", "-10% to 0", "0 to +10%", "+10% to +25%", "above +25%"]
HOLD_BANDS = [(0.5, 2, "0.5-2y"), (2, 5, "2-5y"), (5, 10, "5-10y"), (10, 20, "10-20y"),
              (20, 99, "20y+")]
CURVE_BIN = 0.01
CURVE_EDGE = 0.40
DRIFT_YEARS = 3
DRIFT_SHRINK = 20.0
ALPHA = 0.20                      # 80% intervals
MONDRIAN_MIN = 500                # smallest calibration band given its own correction
CUTS = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]
BANDWIDTHS = [0.05, 0.10, 0.15]
CATS = ["area", "ptype", "tenure"]
BASE_FEATURES = ["lv", "rel", "hold_y", "ptype", "tenure", "new0", "area", "drift_sec", "n_sec",
                 "drift_out", "n_out", "n_prior"]
GAIN_FEATURES = BASE_FEATURES + ["m"]
PARAMS = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=10.0,
              max_cat_to_onehot=4, deterministic=True, force_row_wise=True, verbose=-1)


# ----------------------------------------------------------------------------- features
def mband(m: np.ndarray) -> pd.Categorical:
    return pd.cut(m, MBAND_EDGES, labels=MBAND_LABELS, right=False)


def hold_band(h: np.ndarray) -> np.ndarray:
    out = np.full(len(h), "", dtype=object)
    for lo, hi, lab in HOLD_BANDS:
        out[(h >= lo) & (h < hi)] = lab
    return out


def add_drift(df: pd.DataFrame, key: str, name: str) -> pd.DataFrame:
    """Shrunk mean of M0's error in the same sector (or district) over the previous years.

    Uses only resales from the DRIFT_YEARS calendar years before the resale year, and only pairs
    whose first sale was not a new build, so it is known at the valuation date.
    """
    base = df.loc[~df.new0, [key, "year", "r"]]
    g = base.groupby([key, "year"], observed=True).r.agg(["sum", "count"]).reset_index()
    parts = []
    for k in range(1, DRIFT_YEARS + 1):
        parts.append(g.assign(year=g.year + k))
    roll = (pd.concat(parts).groupby([key, "year"], observed=True)[["sum", "count"]].sum()
            .reset_index().rename(columns={"sum": "_s", "count": "_n"}))
    out = df.merge(roll, on=[key, "year"], how="left")
    out["_s"] = out["_s"].fillna(0.0)
    out["_n"] = out["_n"].fillna(0).astype(np.int64)
    out[f"drift_{name}"] = out["_s"] / (out["_n"] + DRIFT_SHRINK)
    out[f"n_{name}"] = out["_n"]
    return out.drop(columns=["_s", "_n"])


def load(pairs: str, data: Path) -> pd.DataFrame:
    df = pd.read_parquet(pairs)
    idx = pd.read_csv(data / "area_index.csv", usecols=["area", "t", "log_index"])
    lookup = idx.set_index(["area", "t"]).log_index
    df = df[df.area.isin(idx.area.unique())].reset_index(drop=True)
    i1 = lookup.reindex(pd.MultiIndex.from_arrays([df.area, df.t1])).to_numpy()
    i0 = lookup.reindex(pd.MultiIndex.from_arrays([df.area, df.t0])).to_numpy()
    df["m"] = i1 - i0
    df["lp0"] = np.log(df.p0.to_numpy(float))
    df["lv"] = df.lp0 + df.m
    df["r"] = df.lr - df.m
    med = pd.read_csv(data / "area_year_median.csv")
    med = med.assign(year=med.year + 1)[["area", "year", "median_lp"]]
    df = df.merge(med, on=["area", "year"], how="left")
    df["rel"] = df.lv - df.median_lp
    df = add_drift(df, "sector", "sec")
    df = add_drift(df, "outward", "out")
    df["new0"] = df.new0.astype(bool)
    for c in CATS:
        df[c] = df[c].astype("category")
    df["mband"] = mband(df.m.to_numpy())
    df["hold"] = hold_band(df.hold_y.to_numpy())
    return df.drop(columns=["median_lp"])


# ----------------------------------------------------------------------------- models
def _dataset(x: pd.DataFrame, y: np.ndarray) -> lgb.Dataset:
    x = x.copy()
    x["new0"] = x.new0.astype(np.int8)
    return lgb.Dataset(x, y, categorical_feature=[c for c in CATS if c in x.columns],
                       free_raw_data=True)


def _xmat(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["new0"] = x.new0.astype(np.int8)
    return x


def fit(train: pd.DataFrame, features: list[str], objective: str, rounds: int, threads: int,
        alpha: float | None = None) -> lgb.Booster:
    params = dict(PARAMS, objective=objective, seed=SEED, num_threads=threads)
    if alpha is not None:
        params["alpha"] = alpha
    return lgb.train(params, _dataset(train[features], train.r.to_numpy()), num_boost_round=rounds)


def predict(model: lgb.Booster, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict(_xmat(df[features]), num_threads=0)


def conformal_q(scores: np.ndarray, alpha: float = ALPHA) -> float:
    n = len(scores)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(scores)[k - 1])


def calibrate(lo_c, hi_c, r_c, band_c, lo_t, hi_t, band_t):
    """Global and Mondrian (by band of m) conformalised quantile regression."""
    s = np.maximum(lo_c - r_c, r_c - hi_c)
    q_all = conformal_q(s)
    out = {"global": (lo_t - q_all, hi_t + q_all)}
    q_band = {}
    for b in MBAND_LABELS:
        sel = band_c == b
        q_band[b] = conformal_q(s[sel]) if sel.sum() >= MONDRIAN_MIN else q_all
    qt = np.array([q_band[b] for b in band_t.astype(str)]) if len(band_t) else np.array([])
    out["mondrian"] = (lo_t - qt, hi_t + qt)
    return out, q_all, q_band


# ----------------------------------------------------------------------------- metrics
def error_metrics(d: np.ndarray) -> dict:
    pe = np.expm1(d)
    return {"n": len(d), "mean_err": d.mean(), "median_err": np.median(d),
            "mae": np.abs(d).mean(), "mdae": np.median(np.abs(d)), "rmse": np.sqrt((d ** 2).mean()),
            "pe10": (np.abs(pe) <= 0.10).mean(), "over10": (pe > 0.10).mean(),
            "under10": (pe < -0.10).mean()}


def groups(df: pd.DataFrame):
    yield "all", "all", np.ones(len(df), bool)
    for kind, col in [("mband", "mband"), ("region", "region"), ("ptype", "ptype"),
                      ("hold", "hold"), ("year", "year")]:
        vals = df[col].astype(str).to_numpy()
        for v in pd.unique(vals):
            yield kind, v, vals == v
    nb = df.new0.to_numpy()
    yield "first_sale", "new build", nb
    yield "first_sale", "not new", ~nb
    reg = df.region.astype(str).to_numpy()
    mb = df.mband.astype(str).to_numpy()
    for v in pd.unique(reg):
        for b in MBAND_LABELS:
            sel = (reg == v) & (mb == b)
            if sel.any():
                yield "region_mband", f"{v} | {b}", sel


def curve_bins(m: np.ndarray) -> np.ndarray:
    b = np.floor(m / CURVE_BIN).astype(int)
    lim = int(round(CURVE_EDGE / CURVE_BIN))
    return np.clip(b, -lim - 1, lim)


def observed_curves(df: pd.DataFrame) -> pd.DataFrame:
    """M0's error and the outcome by 1% bin of m, by period, for pairs not first sold new."""
    rows = []
    base = df[~df.new0]
    for name, a, b in PERIODS + [("all", FIRST_TRAIN_YEAR, 2025)]:
        sub = base[(base.year >= a) & (base.year <= b)]
        sub = sub.assign(bin=curve_bins(sub.m.to_numpy()))
        g = sub.groupby("bin")
        q = g.r.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
        res = pd.DataFrame({
            "n": g.size(), "mean_m": g.m.mean(), "mean_r": g.r.mean(),
            "r_p10": q[0.1], "r_p25": q[0.25], "r_p50": q[0.5], "r_p75": q[0.75], "r_p90": q[0.9],
            "mean_lr": g.lr.mean(), "share_nominal_loss": g.lr.apply(lambda s: (s < 0).mean()),
            "share_exact": g.exact.mean(),
            "share_within_2pct_below": g.lr.apply(lambda s: ((s < 0) & (s >= -0.02)).mean()),
            "share_within_2pct_above": g.lr.apply(lambda s: ((s >= 0) & (s < 0.02)).mean()),
            "mean_hold_y": g.hold_y.mean()}).reset_index()
        res.insert(0, "period", name)
        rows.append(res)
    return pd.concat(rows, ignore_index=True)


def local_linear(m: np.ndarray, r: np.ndarray, cl: np.ndarray, c: float, h: float,
                 controls: np.ndarray | None = None) -> dict:
    """r = b x + D (jump + kink x) + controls, x = m - c, D = 1{m >= c}, |x| < h.

    ``controls`` holds integer codes of fixed effects (one column per set, e.g. holding band x
    resale year); without them an intercept is used. Standard errors are clustered by ``cl``.
    """
    sel = np.abs(m - c) < h
    x = m[sel] - c
    y = r[sel]
    d = (x >= 0).astype(float)
    nan = float("nan")
    if d.sum() < 200 or (1 - d).sum() < 200:
        return {"cut": c, "bw": h, "n": int(sel.sum()), "clusters": 0, "slope_left": nan,
                "jump": nan, "jump_se": nan, "kink": nan, "kink_se": nan}
    X = np.column_stack([x, d, d * x])
    if controls is None:
        X = np.column_stack([np.ones_like(x), X])
        k0 = 1
    else:
        # one set of fixed effects, partialled out by demeaning within groups (Frisch-Waugh)
        _, inv = np.unique(controls[sel], return_inverse=True)
        cnt = np.bincount(inv).astype(float)
        y = y - (np.bincount(inv, weights=y) / cnt)[inv]
        X = np.column_stack([X[:, j] - (np.bincount(inv, weights=X[:, j]) / cnt)[inv]
                             for j in range(X.shape[1])])
        k0 = 0
    xtx = X.T @ X
    beta = np.linalg.solve(xtx, X.T @ y)
    u = y - X @ beta
    cc, inv = np.unique(cl[sel], return_inverse=True)
    G = len(cc)
    sc = np.zeros((G, X.shape[1]))
    np.add.at(sc, inv, X * u[:, None])
    bread = np.linalg.inv(xtx)
    n, k = X.shape
    adj = G / (G - 1) * (n - 1) / (n - k) if G > 1 else 1.0
    V = adj * bread @ (sc.T @ sc) @ bread
    se = np.sqrt(np.diag(V))
    return {"cut": c, "bw": h, "n": int(sel.sum()), "clusters": G, "slope_left": beta[k0],
            "jump": beta[k0 + 1], "jump_se": se[k0 + 1], "kink": beta[k0 + 2],
            "kink_se": se[k0 + 2]}


def discontinuities(df: pd.DataFrame) -> pd.DataFrame:
    """Jump and change of slope in M0's error at m = c, for c = 0 and placebo values.

    Controls are holding band x resale year fixed effects, because short holdings cluster near
    m = 0 and may differ from long ones for reasons unrelated to the purchase price.
    """
    base = df[~df.new0]
    rows = []
    for name, a, b in PERIODS + [("all", FIRST_TRAIN_YEAR, 2025)]:
        sub = base[(base.year >= a) & (base.year <= b)]
        m = sub.m.to_numpy()
        r = sub.r.to_numpy()
        cl = pd.factorize(sub.area.astype(str) + "|" + sub.year.astype(str))[0]
        ctrl = pd.factorize(sub.hold.astype(str) + "|" + sub.year.astype(str))[0]
        for h in BANDWIDTHS:
            for c in CUTS:
                rows.append({"period": name, **local_linear(m, r, cl, c, h, ctrl)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- events
EVENTS = ["loss_over_2", "loss_within_2", "exact", "gain_within_2", "gain_over_2"]


def event_class(lr: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """Where the sale lands relative to the purchase price, in five classes (see EVENTS)."""
    c = np.full(len(lr), 4, dtype=np.int8)
    c[lr < 0.02] = 3
    c[exact] = 2
    c[lr < 0] = 1
    c[lr < -0.02] = 0
    return c


def dist_events(u_cal: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Class probabilities implied by a point model and the calibration distribution of its error.

    The sale's log change is lr = m + r, with r = prediction + u. ``shift`` is m + prediction.
    A continuous error distribution puts no mass on the exact old price.
    """
    us = np.sort(u_cal)
    n = len(us)

    def F(z):
        return np.searchsorted(us, z, side="left") / n

    a, b, c = F(-0.02 - shift), F(-shift), F(0.02 - shift)
    return np.column_stack([a, b - a, np.zeros_like(a), c - b, 1 - c])


def event_scores(p: np.ndarray, cls: np.ndarray) -> dict:
    out = {}
    for name, cols in [("loss", [0, 1]), ("loss_within_2", [1]), ("exact", [2]),
                       ("gain_within_2", [3])]:
        pr = p[:, cols].sum(1)
        obs = np.isin(cls, cols).astype(float)
        out[f"brier_{name}"] = float(((pr - obs) ** 2).mean())
        out[f"pred_{name}"] = float(pr.mean())
        out[f"obs_{name}"] = float(obs.mean())
    return out


# ----------------------------------------------------------------------------- run
def run_split(df, name, train_last, cal_year, t0, t1, cap, qcap, rounds, qrounds, threads, log):
    rng = np.random.default_rng(SEED)
    train = df[(df.year >= FIRST_TRAIN_YEAR) & (df.year <= train_last)]
    cal = df[df.year == cal_year].reset_index(drop=True)
    test = df[(df.year >= t0) & (df.year <= t1)].reset_index(drop=True)
    tr = train.iloc[rng.choice(len(train), min(cap, len(train)), replace=False)]
    trq = train.iloc[rng.choice(len(train), min(qcap, len(train)), replace=False)]
    info = {"split": name, "train_years": f"{FIRST_TRAIN_YEAR}-{train_last}", "cal_year": cal_year,
            "test_years": f"{t0}-{t1}", "n_train": len(train), "n_train_used": len(tr),
            "n_train_quantile": len(trq), "n_cal": len(cal), "n_test": len(test),
            "train_share_m_negative": float((train.m < 0).mean()),
            "train_share_m_minus10_0": float(((train.m >= -0.10) & (train.m < 0)).mean()),
            "test_share_m_negative": float((test.m < 0).mean()),
            "test_share_m_minus10_0": float(((test.m >= -0.10) & (test.m < 0)).mean())}
    r_cal, r_test = cal.r.to_numpy(), test.r.to_numpy()
    band_cal, band_test = cal.mband.astype(str).to_numpy(), test.mband.astype(str).to_numpy()
    preds, cal_preds = {"M0": np.zeros(len(test))}, {"M0": np.zeros(len(cal))}
    intervals, importance, qinfo, pdp_rows = {}, [], {}, []
    # M0 interval: training percentiles of r, conformalised on the calibration year
    lo0, hi0 = np.quantile(tr.r, [ALPHA / 2, 1 - ALPHA / 2])
    ints, q_all, q_band = calibrate(np.full(len(cal), lo0), np.full(len(cal), hi0), r_cal, band_cal,
                                    np.full(len(test), lo0), np.full(len(test), hi0), band_test)
    intervals["M0"] = ints
    qinfo["M0"] = {"q_global": q_all, **{f"q_{k}": v for k, v in q_band.items()}}
    for label, feats in [("M1", BASE_FEATURES), ("M2", GAIN_FEATURES)]:
        t = time.time()
        mean = fit(tr, feats, "regression", rounds, threads)
        lo = fit(trq, feats, "quantile", qrounds, threads, ALPHA / 2)
        hi = fit(trq, feats, "quantile", qrounds, threads, 1 - ALPHA / 2)
        preds[label] = predict(mean, test, feats)
        cal_preds[label] = predict(mean, cal, feats)
        ints, q_all, q_band = calibrate(predict(lo, cal, feats), predict(hi, cal, feats), r_cal,
                                        band_cal, predict(lo, test, feats), predict(hi, test, feats),
                                        band_test)
        intervals[label] = ints
        qinfo[label] = {"q_global": q_all, **{f"q_{k}": v for k, v in q_band.items()}}
        gain = mean.feature_importance("gain")
        for f, gval in zip(mean.feature_name(), gain):
            importance.append({"split": name, "model": label, "feature": f,
                               "gain_share": gval / gain.sum()})
        if label == "M2":
            samp = test.iloc[rng.choice(len(test), min(20000, len(test)), replace=False)]
            for gm in np.round(np.arange(-0.30, 0.3001, 0.01), 2):
                pdp_rows.append({"split": name, "m": gm,
                                 "mean_pred": float(predict(mean, samp.assign(m=gm), feats).mean())})
        log.append(f"{name} {label}: fitted in {time.time() - t:.0f}s")
    # where the sale lands relative to the purchase price: implied by M0/M1/M2, or learned (C2)
    t = time.time()
    cls_tr = event_class(trq.lr.to_numpy(), trq.exact.to_numpy())
    params = dict(PARAMS, objective="multiclass", num_class=len(EVENTS), seed=SEED,
                  num_threads=threads)
    clf = lgb.train(params, _dataset(trq[GAIN_FEATURES], cls_tr), num_boost_round=qrounds)
    cls_test = event_class(test.lr.to_numpy(), test.exact.to_numpy())
    m_test = test.m.to_numpy()
    probs = {"C2": clf.predict(_xmat(test[GAIN_FEATURES]), num_threads=0)}
    for label in ["M0", "M1", "M2"]:
        probs[f"{label}-dist"] = dist_events(r_cal - cal_preds[label], m_test + preds[label])
    # M2 with a separate error distribution for each holding band: short holdings vary less
    p_h = np.zeros_like(probs["M2-dist"])
    h_cal, h_test = cal.hold.to_numpy(), test.hold.to_numpy()
    u2 = r_cal - cal_preds["M2"]
    for hb in pd.unique(h_test):
        sel_t, sel_c = h_test == hb, h_cal == hb
        u = u2[sel_c] if sel_c.sum() >= MONDRIAN_MIN else u2
        p_h[sel_t] = dist_events(u, m_test[sel_t] + preds["M2"][sel_t])
    probs["M2-dist-h"] = p_h
    log.append(f"{name} event models in {time.time() - t:.0f}s")
    # metrics and coverage by group
    met, cov, ev = [], [], []
    for kind, g, sel in groups(test):
        r = r_test[sel]
        for label, p in preds.items():
            met.append({"split": name, "model": label, "group_type": kind, "group": g,
                        **error_metrics(p[sel] - r)})
            for method, (lo, hi) in intervals[label].items():
                cov.append({"split": name, "model": label, "method": method, "group_type": kind,
                            "group": g, "n": int(sel.sum()),
                            "coverage": float(((r >= lo[sel]) & (r <= hi[sel])).mean()),
                            "mean_width": float((hi[sel] - lo[sel]).mean()),
                            "below": float((r < lo[sel]).mean()),
                            "above": float((r > hi[sel]).mean())})
        for label, p in probs.items():
            ev.append({"split": name, "model": label, "group_type": kind, "group": g,
                       "n": int(sel.sum()), **event_scores(p[sel], cls_test[sel])})
    # binned predictions against the observed outcome, homes not first sold new
    keep = ~test.new0.to_numpy()
    b = curve_bins(m_test[keep])
    cdf = pd.DataFrame({"bin": b, "m": m_test[keep], "r": r_test[keep],
                        "M1": preds["M1"][keep], "M2": preds["M2"][keep]})
    for k, e in enumerate(EVENTS):
        cdf[f"obs_{e}"] = (cls_test[keep] == k).astype(float)
        for label, p in probs.items():
            cdf[f"{label}_{e}"] = p[keep, k]
    curve = cdf.groupby("bin").agg(n=("m", "size"), mean_m=("m", "mean"), mean_r=("r", "mean"),
                                   mean_M1=("M1", "mean"), mean_M2=("M2", "mean"),
                                   **{c: (c, "mean") for c in cdf.columns
                                      if c not in ("bin", "m", "r", "M1", "M2")}).reset_index()
    curve.insert(0, "split", name)
    info["calibration"] = qinfo
    return info, met, cov, ev, curve, importance, pdp_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cap", type=int, default=3_000_000, help="training rows for the mean models")
    ap.add_argument("--qcap", type=int, default=1_500_000, help="training rows for quantile models")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--qrounds", type=int, default=400)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--splits", default=",".join(s[0] for s in SPLITS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    t = time.time()
    df = load(args.pairs, Path(args.data))
    log.append(f"loaded {len(df):,} pairs in {time.time() - t:.0f}s")
    t = time.time()
    observed_curves(df).to_csv(out / "curve_observed.csv", index=False, float_format="%.6g")
    discontinuities(df).to_csv(out / "discontinuity.csv", index=False, float_format="%.6g")
    log.append(f"curves and discontinuities in {time.time() - t:.0f}s")
    infos, mets, covs, evs, curves, imps, pdps = [], [], [], [], [], [], []
    wanted = set(args.splits.split(","))
    for spec in SPLITS:
        if spec[0] not in wanted:
            continue
        info, met, cov, ev, curve, imp, pdp = run_split(df, *spec, args.cap, args.qcap, args.rounds,
                                                        args.qrounds, args.threads, log)
        infos.append(info)
        mets += met
        covs += cov
        evs += ev
        curves.append(curve)
        imps += imp
        pdps += pdp
    pd.DataFrame(mets).to_csv(out / "metrics.csv", index=False, float_format="%.6g")
    pd.DataFrame(covs).to_csv(out / "coverage.csv", index=False, float_format="%.6g")
    pd.DataFrame(evs).to_csv(out / "events.csv", index=False, float_format="%.6g")
    pd.concat(curves).to_csv(out / "curve_predicted.csv", index=False, float_format="%.6g")
    pd.DataFrame(imps).to_csv(out / "importance.csv", index=False, float_format="%.6g")
    pd.DataFrame(pdps).to_csv(out / "pdp_m.csv", index=False, float_format="%.6g")
    summary = {"n_pairs": len(df), "n_pairs_not_new": int((~df.new0).sum()),
               "years": [int(df.year.min()), int(df.year.max())], "splits": infos,
               "settings": {"cap": args.cap, "qcap": args.qcap, "rounds": args.rounds,
                            "qrounds": args.qrounds, "params": PARAMS, "seed": SEED}}
    (out / "model_run.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    (out / "model_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
