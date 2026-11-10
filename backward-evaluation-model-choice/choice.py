"""Model choice by forward backtest and by backward evaluation (PREREGISTRATION.md).

    python choice.py --domain test_web --shard 0 --of 8 --workers 4

For each series:

1. **Deployment.** Every candidate is refitted on all data to date every 91 days and forecasts the
   next 91 days one step ahead; its log loss is stored for every forecast day.
2. **Decisions** every 91 days from the first origin with a full year of deployed forecasts:
   * FWD: mean deployed log loss over the last 364 days;
   * BWD: fit on the last 364 days, mean log loss on every earlier day before a 28-day gap;
   * BWD1: the same fit, scored on the year before the gap only;
   * HYB: mean of FWD and BWD.
   Each procedure uses the deployed forecasts of the candidate it picked for the next 91 days.
3. **Outputs** per series and candidate set: total log loss and day count of every procedure
   (``FWD``, ``BWD``, ``BWD1``, ``HYB``, ``ENS``, ``ORACLE``) and of every candidate (``CAND:<name>``)
   over the evaluation days; one record per decision with scores, choices and realised losses.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

import common
from backward_eval import ARGaussian, ARStudentT
from models import MAX_LAG, T_DF, ETSModel, GBMModel, _hw_filter, design, gauss_scale, t_scale

L, GAP, REFIT = common.BLOCK, common.GAP, common.REFIT
SETS = {
    "A": ("AR7_t", "SLM_t", "ETS_t", "GBM_t"),
    "B": ("AR1_t", "AR7_t", "AR14_t", "AR28_t"),
    "A8": ("AR7_g", "SLM_g", "ETS_g", "GBM_g", "AR7_t", "SLM_t", "ETS_t", "GBM_t"),
}
CANDS = sorted({c for v in SETS.values() for c in v})
PROCS = ("FWD", "BWD", "BWD1", "HYB")


def _nll(z, m, s, dist):
    w = (z - m) / s
    lp = norm.logpdf(w) if dist == "g" else student_t.logpdf(w, T_DF)
    return -(lp - np.log(s))


def ar_design(y: np.ndarray, p: int) -> np.ndarray:
    n = len(y) - MAX_LAG
    lags = [y[MAX_LAG - k : MAX_LAG - k + n] for k in range(1, p + 1)]
    return np.column_stack([np.ones(n), *lags])


class Designs:
    def __init__(self, s: pd.Series):
        y = s.to_numpy(dtype=float)
        self.D = design(y, s.index)
        self.X = {"AR1": ar_design(y, 1), "AR7": self.D.X_ar, "AR14": ar_design(y, 14), "AR28": ar_design(y, 28),
                  "SLM": self.D.X_slm}
        self.z = self.D.z


def fit_score(base: str, Ds: Designs, a: int, b: int, rows: np.ndarray, from_start: int | None = None) -> dict[str, np.ndarray]:
    """Fit ``base`` (AR1/AR7/AR14/AR28/SLM/ETS/GBM) on [a, b); log loss on ``rows`` for 'g' and 't'.

    ``from_start``: for ETS, filter the states from this row (scoring rows before the fit window).
    """
    z = Ds.z
    zr = z[rows]
    if base in Ds.X:
        X = Ds.X[base]
        g = ARGaussian().fit(X[a:b], z[a:b])
        t = ARStudentT(df=T_DF).fit(X[a:b], z[a:b])
        mg, sg = g.predict(X[rows])
        mt, st = t.predict(X[rows])
        return {"g": _nll(zr, mg, sg, "g"), "t": _nll(zr, mt, st, "t")}
    if base == "ETS":
        m = ETSModel().fit(z, a, b)
        start = a if from_start is None else from_start
        f = _hw_filter(z, m.alpha, m.gamma, m.m, start, int(rows[-1]) + 1)[rows]
        return {"g": _nll(zr, f, gauss_scale(m.resid), "g"), "t": _nll(zr, f, t_scale(m.resid), "t")}
    if base == "GBM":
        m = GBMModel().fit(Ds.D.X_gbm, z, a, b)
        f = m.predict(Ds.D.X_gbm, rows)
        return {"g": _nll(zr, f, gauss_scale(m.resid), "g"), "t": _nll(zr, f, t_scale(m.resid), "t")}
    raise ValueError(base)


BASES = ("AR1", "AR7", "AR14", "AR28", "SLM", "ETS", "GBM")


def run_series(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    domain, key, s = args
    Ds = Designs(s)
    n = len(Ds.z)
    first = 2 * L + GAP
    dep_rows = np.arange(first, n)
    # 1. deployment: expanding window, refit every 91 days
    ll = {c: np.full(n, np.nan) for c in CANDS}
    for o in range(first, n, REFIT):
        rows = np.arange(o, min(o + REFIT, n))
        for base in BASES:
            out = fit_score(base, Ds, 0, o, rows)
            for dist in ("g", "t"):
                name = f"{base}_{dist}"
                if name in ll:
                    ll[name][rows] = out[dist]
    # 2. decisions
    d0 = first + L
    dec_origins = list(range(d0, n, REFIT))
    picks = {(st, p): {} for st in SETS for p in PROCS}
    recs = []
    for o in dec_origins:
        past = np.arange(MAX_LAG, o - L - GAP)
        past1 = np.arange(max(MAX_LAG, o - 2 * L - GAP), o - L - GAP)
        nxt = np.arange(o, min(o + REFIT, n))
        bwd, bwd1 = {}, {}
        for base in BASES:
            out = fit_score(base, Ds, o - L, o, past, from_start=0)
            for dist in ("g", "t"):
                name = f"{base}_{dist}"
                if name in ll:
                    bwd[name] = float(out[dist].mean())
                    bwd1[name] = float(out[dist][np.isin(past, past1)].mean())
        fwd = {c: float(np.mean(ll[c][o - L : o])) for c in CANDS}
        real = {c: float(np.mean(ll[c][nxt])) for c in CANDS}
        for st, cands in SETS.items():
            sc = {"FWD": fwd, "BWD": bwd, "BWD1": bwd1,
                  "HYB": {c: 0.5 * (fwd[c] + bwd[c]) for c in cands}}
            for p in PROCS:
                ch = min(cands, key=lambda c: sc[p][c])
                picks[(st, p)][o] = ch
                recs.append({"domain": domain, "series": key, "set": st, "date": f"{Ds.D.dates[o]:%Y-%m-%d}",
                             "proc": p, "choice": ch, "realised_choice": real[ch],
                             "realised_best": min(real[c] for c in cands),
                             "best": min(cands, key=lambda c: real[c]),
                             **{f"score:{c}": sc[p][c] for c in cands}, **{f"real:{c}": real[c] for c in cands}})
    # 3. procedure losses over the evaluation days
    ev = np.arange(d0, n)
    res = []
    for st, cands in SETS.items():
        for p in PROCS:
            tot = 0.0
            for o in dec_origins:
                rows = np.arange(o, min(o + REFIT, n))
                tot += float(ll[picks[(st, p)][o]][rows].sum())
            res.append((st, p, tot))
        dens = np.mean([np.exp(-ll[c][ev]) for c in cands], axis=0)
        res.append((st, "ENS", float(-np.log(np.maximum(dens, 1e-300)).sum())))
        orc = 0.0
        for o in dec_origins:
            rows = np.arange(o, min(o + REFIT, n))
            orc += min(float(ll[c][rows].sum()) for c in cands)
        res.append((st, "ORACLE", orc))
        for c in cands:
            res.append((st, f"CAND:{c}", float(ll[c][ev].sum())))
    per = pd.DataFrame(res, columns=["set", "proc", "sum_ll"])
    per.insert(0, "series", key)
    per.insert(0, "domain", domain)
    per["n"] = len(ev)
    return per, pd.DataFrame(recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=common.DOMAINS, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    data = common.load(args.domain)
    keys = sorted(data)[args.shard :: args.of]
    if args.limit:
        keys = keys[: args.limit]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(run_series, [(args.domain, k, data[k]) for k in keys]))
    out = common.OUT / "shards"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.domain}_{args.shard:02d}"
    pd.concat([r[0] for r in res], ignore_index=True).to_csv(out / f"{tag}_losses.csv", index=False, float_format="%.6f")
    pd.concat([r[1] for r in res], ignore_index=True).to_csv(out / f"{tag}_decisions.csv.gz", index=False,
                                                           float_format="%.5f", compression={"method": "gzip", "mtime": 0})
    print(f"{tag}: {len(keys)} series")


if __name__ == "__main__":
    main()
