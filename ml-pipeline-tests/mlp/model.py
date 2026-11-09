"""Model, evaluation protocol and serving-time feature frame.

Periods: train January-August, validation September, test October-November,
deployment December. The reviewed protocol fits on train, picks the decision
threshold on validation, reports on test, and is then judged on December with
features computed the way production can compute them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .features import CATEGORICAL, SHRINK, Variant

TRAIN, VALID, TEST, DEPLOY = (1, 8), (9, 9), (10, 11), (12, 12)


def months(df: pd.DataFrame, span: tuple[int, int]) -> pd.DataFrame:
    return df[(df["month"] >= span[0]) & (df["month"] <= span[1])]


def fit(train: pd.DataFrame, cols: list[str], seed: int = 0, max_iter: int = 300):
    model = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=50,
        categorical_features="from_dtype", early_stopping=False, random_state=seed,
    )
    model.fit(train[cols], train["disrupted"])
    return model


def scores(model, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return model.predict_proba(df[cols])[:, 1]


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.8, 76)
    f1s = [f1_score(y, p >= t) for t in grid]
    return float(grid[int(np.argmax(f1s))])


@dataclass
class Evaluation:
    auc: float
    ap: float
    f1: float
    brier: float
    mean_pred: float
    rate: float
    n: int


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> Evaluation:
    return Evaluation(
        auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
        f1=f1_score(y, p >= threshold), brier=float(np.mean((p - y) ** 2)),
        mean_pred=float(p.mean()), rate=float(y.mean()), n=len(y),
    )


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def split(df: pd.DataFrame, v: Variant, seed: int = 0) -> Split:
    if v.random_split:
        pool = df[df["month"] <= TEST[1]]
        u = np.random.default_rng(seed).random(len(pool))
        return Split(pool[u < 0.7], pool[(u >= 0.7) & (u < 0.8)], pool[u >= 0.8])
    return Split(months(df, TRAIN), months(df, VALID), months(df, TEST))


def serving_frame(con, honest: pd.DataFrame, v: Variant) -> pd.DataFrame:
    """December features as production can compute them two hours before
    departure: the reviewed features, with whatever the variant's model
    expects that production cannot know replaced by what it can."""
    dec = months(honest, DEPLOY).copy()
    if v.leak_dep_delay:
        dec["dep_delay"] = np.nan  # the flight has not left yet
    if v.global_encoding:
        # the encoding table recomputed with every label known before December
        rates = con.execute(f"""
            with prior as (select avg(disrupted) as p from base where month < 12)
            select origin, dest,
                   (sum(disrupted) + {SHRINK} * (select p from prior)) / (count(*) + {SHRINK}) as rr
            from base where month < 12 group by origin, dest""").df()
        tails = con.execute(f"""
            with prior as (select avg(disrupted) as p from base where month < 12)
            select tailnum,
                   (sum(disrupted) + {SHRINK} * (select p from prior)) / (count(*) + {SHRINK}) as tr
            from base where month < 12 and tailnum is not null group by tailnum""").df()
        ids = con.execute("select flight_id, tailnum from base where month = 12").df()
        dec = dec.merge(ids, on="flight_id", how="left")
        dec = dec.merge(rates, on=["origin", "dest"], how="left").merge(tails, on="tailnum", how="left")
        dec["route_rate"], dec["tail_rate"] = dec["rr"], dec["tr"]
        dec = dec.drop(columns=["rr", "tr", "tailnum"])
    for c in CATEGORICAL:
        dec[c] = pd.Categorical(dec[c].astype(str), categories=honest[c].cat.categories)
    return dec


@dataclass
class Result:
    variant: str
    threshold: float
    reported: Evaluation
    realized: Evaluation


def run(con, frames: dict, v: Variant, seed: int = 0, max_iter: int = 300) -> Result:
    """``frames`` caches feature tables by variant-relevant key."""
    df = frames[v]
    cols = v.feature_columns()
    s = split(df, v, seed)
    model = fit(s.train, cols, seed, max_iter)
    p_test = scores(model, s.test, cols)
    if v.threshold_on_test:
        thr = best_threshold(s.test["disrupted"].to_numpy(), p_test)
    else:
        thr = best_threshold(s.valid["disrupted"].to_numpy(), scores(model, s.valid, cols))
    reported = evaluate(s.test["disrupted"].to_numpy(), p_test, thr)
    dec = serving_frame(con, frames[Variant()], v)
    realized = evaluate(dec["disrupted"].to_numpy(), scores(model, dec, cols), thr)
    return Result(v.name, thr, reported, realized)


def day_bootstrap_auc(df: pd.DataFrame, p: np.ndarray, reps: int = 400, seed: int = 0):
    """95% interval for AUC, resampling whole days (delays cluster by day)."""
    rng = np.random.default_rng(seed)
    days = df["flight_date"].to_numpy()
    uniq = np.unique(days)
    idx_by_day = {d: np.flatnonzero(days == d) for d in uniq}
    y = df["disrupted"].to_numpy()
    out = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq))
        idx = np.concatenate([idx_by_day[d] for d in pick])
        out.append(roc_auc_score(y[idx], p[idx]))
    return tuple(np.percentile(out, [2.5, 97.5]))
