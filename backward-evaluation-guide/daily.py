"""Worked examples on daily data: electricity load and Wikipedia page views.

    python daily.py --domain test_web --shard 0 --of 8 --workers 4

Data: the 25 electricity-load series and 304 Wikipedia articles of
``../backward-evaluation-generality`` and the 245 articles of
``../backward-evaluation-model-choice`` (log load; log(1 + views)). For every series:

1. **The matrix of each family.** Blocks of 364 days aligned to the end of the sample, a
   28-day gap, one-day-ahead Student-t(5) forecasts. Families: the autoregression (7 lags),
   the seasonal linear model and LightGBM of the earlier studies, additive Holt-Winters
   (through ``matrix_from_forecasts``: states filtered from the first day, the block's own
   model scored in sample), and ``GBMn``, LightGBM on the same features expressed relative
   to the mean of the previous 28 days, predicting the deviation from that mean.
2. **Monitoring**: the seasonal linear model fitted on the last 364 days, scored on the 364
   days before a 28-day gap, every 91 days.
3. **A forward check of GBMn against GBM**: both deployed as in the model-choice study
   (refitted on all data to date every 91 days), log loss over the same evaluation days.

Writes ``results/shards/<domain>_<shard>_{cells,monitor,forward}.csv.gz``.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("backward-evaluation-generality", "backward-evaluation-model-choice", "backward-evaluation"):
    if str(ROOT / sub) not in sys.path:
        sys.path.append(str(ROOT / sub))

from bwdiag import LinearModel, RegressorModel, blocks_from_end, matrix_from_forecasts, monitor, nlpd, transfer_matrix  # noqa: E402
from models import GBM_PARAMS, GBM_ROUNDS, T_DF, ETSModel, _hw_filter, design, t_scale  # noqa: E402

BLOCK, GAP, REFIT = 364, 28, 91
FAMILIES = ("AR", "SLM", "ETS", "GBM", "GBMn")
GEN = ROOT / "backward-evaluation-generality" / "data"
TEST = ROOT / "backward-evaluation-model-choice" / "data" / "wikipedia_test_daily.csv.gz"
OUT = HERE / "results" / "shards"
R28 = 11  # column of the 28-day mean in the gradient-boosting design


def load(domain: str) -> dict[str, pd.Series]:
    if domain == "elec":
        df = pd.read_csv(GEN / "electricity_daily.csv", parse_dates=["date"]).set_index("date")
        return {c: np.log(df[c].astype(float)).rename(c) for c in df.columns}
    path = GEN / "wikipedia_daily.csv.gz" if domain == "dev_web" else TEST
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return {c: np.log1p(df[c].astype(float)).rename(c) for c in df.columns}


class LGB:
    """LightGBM with the settings of the earlier studies, in scikit-learn form."""

    def fit(self, X, z):
        import lightgbm as lgb
        self.m = lgb.train(GBM_PARAMS, lgb.Dataset(X, z, free_raw_data=True), num_boost_round=GBM_ROUNDS)
        return self

    def predict(self, X):
        return self.m.predict(X)


class GBMNormalised:
    """LightGBM on level-free features: lags and the 7-day mean minus the 28-day mean,
    plus the calendar; it predicts the deviation of the day from the 28-day mean."""

    df = T_DF

    @staticmethod
    def _split(X):
        base = X[:, R28]
        return base, np.hstack([X[:, :R28] - base[:, None], X[:, R28 + 1:]])

    def fit(self, X, z):
        base, Xn = self._split(X)
        self.inner = RegressorModel(LGB, df=T_DF).fit(Xn, z - base)
        return self

    def predict(self, X):
        base, Xn = self._split(X)
        mean, scale = self.inner.predict(Xn)
        return base + mean, scale


def factories(D):
    return {
        "AR": (D.X_ar, lambda: LinearModel(df=T_DF, intercept=False)),
        "SLM": (D.X_slm, lambda: LinearModel(df=T_DF, intercept=False)),
        "GBM": (D.X_gbm, lambda: RegressorModel(LGB, df=T_DF)),
        "GBMn": (D.X_gbm, GBMNormalised),
    }


def ets_matrix(z, blocks, labels):
    fits = {}

    def forecast(i):
        if i not in fits:
            a, b = blocks[i]
            m = ETSModel().fit(z, a, b)
            mean = _hw_filter(z, m.alpha, m.gamma, m.m, 0, len(z))
            fits[i] = (mean, t_scale(m.resid))
        return fits[i]

    def own(j):
        a, b = blocks[j]
        mean, scale = forecast(j)
        return nlpd(z[a:b], mean[a:b], scale, T_DF)

    return matrix_from_forecasts(z, blocks, forecast, own, T_DF, GAP, labels)


def deployed(X, z, factory, first, n):
    ll = np.full(n, np.nan)
    for o in range(first, n, REFIT):
        rows = np.arange(o, min(o + REFIT, n))
        m = factory().fit(X[:o], z[:o])
        mean, scale = m.predict(X[rows])
        ll[rows] = nlpd(z[rows], mean, scale, T_DF)
    return ll


def run_series(args):
    domain, key, s = args
    D = design(s.to_numpy(float), s.index)
    z, n = D.z, len(D.z)
    blocks = blocks_from_end(n, BLOCK)
    labels = [f"{D.dates[a]:%Y-%m}" for a, b in blocks]
    cells = []
    mats = {"ETS": ets_matrix(z, blocks, labels)}
    for fam, (X, fac) in factories(D).items():
        mats[fam] = transfer_matrix(X, z, fac, BLOCK, gap=GAP, folds=5, seed=2026, labels=labels)
    for fam in FAMILIES:
        M = mats[fam]
        for i in range(M.k):
            for j in range(M.k):
                if i != j and np.isfinite(M.loss[i, j]):
                    cells.append(dict(domain=domain, series=key, family=fam, train=i, test=j, train_label=labels[i],
                                      test_label=labels[j], k=M.k, loss=M.loss[i, j], reference=M.reference[i, j],
                                      excess=M.excess[i, j], level=M.level[i, j], spread=M.spread[i, j],
                                      structure=M.structure[i, j], rows=int(M.rows[i, j])))
    X, fac = factories(D)["SLM"]
    mon = [dict(domain=domain, series=key, date=f"{D.dates[r['origin'] - 1]:%Y-%m-%d}", **r)
           for r in monitor(X, z, fac, BLOCK, REFIT, gap=GAP, folds=5, seed=2026)]
    first = 2 * BLOCK + GAP
    ev = np.arange(first + BLOCK, n)
    fwd = []
    for fam in ("GBM", "GBMn", "SLM"):
        X, fac = factories(D)[fam]
        ll = deployed(X, z, fac, first, n)
        fwd.append(dict(domain=domain, series=key, family=fam, sum_ll=float(ll[ev].sum()), n=len(ev)))
    return pd.DataFrame(cells), pd.DataFrame(mon), pd.DataFrame(fwd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=("elec", "dev_web", "test_web"), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    data = load(a.domain)
    keys = sorted(data)[a.shard::a.of]
    if a.limit:
        keys = keys[: a.limit]
    jobs = [(a.domain, k, data[k]) for k in keys]
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            out = list(ex.map(run_series, jobs))
    else:
        out = [run_series(j) for j in jobs]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.domain}_{a.shard:02d}"
    for i, name in enumerate(("cells", "monitor", "forward")):
        pd.concat([o[i] for o in out], ignore_index=True).to_csv(
            OUT / f"{tag}_{name}.csv.gz", index=False, float_format="%.6f", compression={"method": "gzip", "mtime": 0})
    print(f"{tag}: {len(keys)} series")


if __name__ == "__main__":
    main()
