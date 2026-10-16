"""Does a learned classifier escape the drift, or only move it?

A TF-IDF logistic regression is trained on the hand labels two ways:

* within eras: repeated stratified K-fold over all labelled items, so every era is
  represented in training (the usual, optimistic evaluation);
* leave one era out: trained on every other era and tested on the one left out, which
  is what happens when a model built on one period is applied to another.

All metrics are weighted back to the population with the design weights N_s / n_s, so
they estimate precision and recall over every Chair speech in the era, not over the
sample. Intervals come from resampling items within strata with the predictions fixed;
they cover the evaluation noise, not the variability of retraining.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from .config import Config
from .dictionary import normalise
from .labels import final_labels


def labelled_items(cfg: Config) -> pd.DataFrame:
    design = pd.read_csv(cfg.labels_dir / "design.csv", dtype={"id": str})
    sample = pd.read_csv(cfg.labels_dir / "sample.csv", dtype=str, keep_default_na=False)[["id", "chair_text"]]
    labels, _ = final_labels(cfg)
    df = design.merge(labels, on="id").merge(sample, on="id")
    df["stratum"] = df["era"] + "|" + df["d_tag"].astype(str) + "|" + df["side"]
    n_lab = df.groupby("stratum")["id"].transform("size")
    df["w"] = df["N"] / n_lab
    df["norm"] = df["chair_text"].map(normalise)
    return df.reset_index(drop=True)


def _model(tr: dict) -> tuple[TfidfVectorizer, LogisticRegression]:
    vec = TfidfVectorizer(
        token_pattern=r"(?u)\b[a-z][a-z']*[a-z]\b|\b[a-z]\b",
        ngram_range=(1, int(tr["ngram_max"])),
        min_df=int(tr["min_df"]),
        sublinear_tf=True,
    )
    clf = LogisticRegression(C=float(tr["c"]), class_weight=tr.get("class_weight") or None, max_iter=2000)
    return vec, clf


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, tr: dict) -> np.ndarray:
    if train["discipline"].nunique() < 2:
        return np.full(len(test), int(train["discipline"].iloc[0]))
    vec, clf = _model(tr)
    x = vec.fit_transform(train["norm"])
    clf.fit(x, train["discipline"])
    return clf.predict(vec.transform(test["norm"]))


def weighted_pr(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    tp = float(np.sum(w * (yhat == 1) * (y == 1)))
    pp = float(np.sum(w * (yhat == 1)))
    pos = float(np.sum(w * (y == 1)))
    return (tp / pp if pp else np.nan, tp / pos if pos else np.nan)


def _boot(df: pd.DataFrame, pred: np.ndarray, reps: int, rng: np.random.Generator) -> dict:
    y = df["discipline"].to_numpy()
    w = df["w"].to_numpy()
    groups = [np.flatnonzero(df["stratum"].to_numpy() == s) for s in df["stratum"].unique()]
    p0, r0 = weighted_pr(y, pred, w)
    ps, rs = [], []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        p, r = weighted_pr(y[idx], pred[idx], w[idx])
        ps.append(p)
        rs.append(r)
    ps, rs = np.array(ps), np.array(rs)
    return {
        "precision": p0,
        "precision_lo": float(np.nanquantile(ps, 0.025)),
        "precision_hi": float(np.nanquantile(ps, 0.975)),
        "recall": r0,
        "recall_lo": float(np.nanquantile(rs, 0.025)),
        "recall_hi": float(np.nanquantile(rs, 0.975)),
    }


def transfer(cfg: Config) -> pd.DataFrame:
    tr = cfg.transfer
    df = labelled_items(cfg)
    rng = np.random.default_rng(int(tr["seed"]))
    eras = [e for e in cfg.era_names if (df["era"] == e).any()]

    # Within-era evaluation: out-of-fold predictions, majority vote over the repeats.
    votes = np.zeros(len(df))
    counts = np.zeros(len(df))
    cv = RepeatedStratifiedKFold(
        n_splits=int(tr["cv_folds"]), n_repeats=int(tr["cv_repeats"]), random_state=int(tr["seed"])
    )
    strat = df["era"] + "|" + df["discipline"].astype(str)
    for tr_idx, te_idx in cv.split(df, strat):
        votes[te_idx] += _fit_predict(df.iloc[tr_idx], df.iloc[te_idx], tr)
        counts[te_idx] += 1
    within = (votes / np.maximum(counts, 1) >= 0.5).astype(int)

    loeo = np.zeros(len(df), dtype=int)
    for e in eras:
        te = (df["era"] == e).to_numpy()
        loeo[te] = _fit_predict(df[~te], df[te], tr)

    rows = []
    for e in eras:
        sel = (df["era"] == e).to_numpy()
        sub = df[sel].reset_index(drop=True)
        for method, pred in (
            ("dictionary", sub["d_tag"].to_numpy()),
            ("model_within_eras", within[sel]),
            ("model_leave_era_out", loeo[sel]),
        ):
            rows.append(
                {"era": e, "method": method, "items": int(sel.sum()), **_boot(sub, pred, int(tr["bootstrap"]), rng)}
            )
    out = pd.DataFrame(rows)
    cfg.stage2_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.stage2_dir / "transfer.csv", index=False)

    vec, clf = _model(tr)
    clf.fit(vec.fit_transform(df["norm"]), df["discipline"])
    coef = pd.DataFrame({"term": vec.get_feature_names_out(), "coef": clf.coef_[0]}).sort_values("coef")
    pd.concat([coef.head(25), coef.tail(25)]).to_csv(cfg.stage2_dir / "model_terms.csv", index=False)
    return out
