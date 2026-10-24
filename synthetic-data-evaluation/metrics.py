"""Fidelity, utility and privacy measures for one synthetic table.

All functions take code matrices from `generators.Encoder` (real `R`,
synthetic `S`) or the decoded data frames, and return plain floats.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from generators import Encoder, distance_features
from inference import logit

# ---------------------------------------------------------------- fidelity


def marginal_distances(R: np.ndarray, S: np.ndarray, enc: Encoder) -> dict[str, float]:
    """Kolmogorov-Smirnov statistic for numeric columns, total variation
    distance for categorical ones. Both lie in [0, 1]."""
    out = {}
    for j, c in enumerate(enc.columns):
        if enc.categorical[c]:
            k = len(enc.levels[c])
            p = np.bincount(R[:, j].astype(int), minlength=k) / len(R)
            q = np.bincount(S[:, j].astype(int), minlength=k) / len(S)
            out[c] = 0.5 * float(np.abs(p - q).sum())
        else:
            out[c] = float(stats.ks_2samp(R[:, j], S[:, j]).statistic)
    return out


def _discretise(R: np.ndarray, X: np.ndarray, enc: Encoder, bins: int) -> np.ndarray:
    """Numeric columns cut at deciles of the real data; categories kept."""
    D = np.empty(X.shape, dtype=int)
    for j, c in enumerate(enc.columns):
        if enc.categorical[c]:
            D[:, j] = X[:, j].astype(int)
        else:
            edges = np.unique(np.quantile(R[:, j], np.linspace(0, 1, bins + 1)[1:-1]))
            D[:, j] = np.searchsorted(edges, X[:, j], side="right")
    return D


def pairwise_tvd(R: np.ndarray, S: np.ndarray, enc: Encoder, bins: int = 10) -> np.ndarray:
    """Total variation distance between every two-way table, real vs synthetic.

    Returns a symmetric p x p matrix (zeros on the diagonal)."""
    DR, DS = _discretise(R, R, enc, bins), _discretise(R, S, enc, bins)
    p = R.shape[1]
    out = np.zeros((p, p))
    for i, j in itertools.combinations(range(p), 2):
        ki = max(DR[:, i].max(), DS[:, i].max()) + 1
        kj = max(DR[:, j].max(), DS[:, j].max()) + 1
        a = np.bincount(DR[:, i] * kj + DR[:, j], minlength=ki * kj) / len(DR)
        b = np.bincount(DS[:, i] * kj + DS[:, j], minlength=ki * kj) / len(DS)
        out[i, j] = out[j, i] = 0.5 * np.abs(a - b).sum()
    return out


def discriminator(
    R: np.ndarray, S: np.ndarray, enc: Encoder, seed: int, pair_with: np.ndarray | None = None
) -> dict[str, float]:
    """Propensity-score test (Woo et al. 2009; Snoke et al. 2018).

    A gradient-boosting classifier is trained to tell real rows from synthetic
    ones; its probabilities are cross-fitted over five folds so that no row is
    scored by a model that saw it. With equal numbers of real and synthetic
    rows the share of synthetic rows is c = 1/2, pMSE = mean (p - c)^2 lies in
    [0, 1/4], and the AUC is 1/2 when the two tables are indistinguishable.

    If `pair_with` is given (for each synthetic row, the index of its nearest
    real row), a synthetic row and its nearest real row are always put in the
    same fold. Without this, a synthetic near-copy of a real row in the
    training folds teaches the classifier to call that real row synthetic,
    and the AUC can fall below 1/2.
    """
    X = np.vstack([R, S])
    y = np.r_[np.zeros(len(R)), np.ones(len(S))]
    p = np.empty(len(y))
    if pair_with is None:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = folds.split(X, y)
    else:
        groups = np.r_[np.arange(len(R)), pair_with]
        folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = folds.split(X, y, groups)
    for tr, te in splits:
        clf = HistGradientBoostingClassifier(
            categorical_features=enc.cat_mask,
            max_iter=200,
            learning_rate=0.1,
            early_stopping=False,
            random_state=seed,
        )
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    c = len(S) / len(y)
    return {"auc": float(roc_auc_score(y, p)), "pmse": float(np.mean((p - c) ** 2))}


def consistency(
    df: pd.DataFrame, education_map: dict[str, int], ranges: dict[str, tuple[float, float]]
) -> dict[str, float]:
    """Rules that hold (almost) exactly in the real data.

    education and education-num are two codings of the same variable; a
    husband is male and a wife female by the census definitions; and no
    numeric value falls outside the range seen in the real data (capital
    gains and losses, for instance, are never negative)."""
    mapped = df["education"].map(education_map).to_numpy()
    husband = df["relationship"].to_numpy() == "Husband"
    wife = df["relationship"].to_numpy() == "Wife"
    sex = df["sex"].to_numpy()
    outside = np.zeros(len(df), dtype=bool)
    for col, (lo, hi) in ranges.items():
        v = df[col].to_numpy(dtype=float)
        outside |= (v < lo) | (v > hi)
    return {
        "education_mismatch": float(np.mean(mapped != df["education-num"].to_numpy())),
        "female_husband": float(np.mean(sex[husband] == "Female")) if husband.any() else 0.0,
        "male_wife": float(np.mean(sex[wife] == "Male")) if wife.any() else 0.0,
        "out_of_range": float(np.mean(outside)),
    }


# ----------------------------------------------------------------- utility


def tstr(S: np.ndarray, H: np.ndarray, enc: Encoder, target: int, seed: int) -> dict[str, float]:
    """Train on synthetic, test on real (Esteban et al. 2017).

    A gradient-boosting classifier for income is fitted to the synthetic
    table, with early stopping on a 10% validation split of that same
    synthetic table (all the analyst would have), and scored on the real
    holdout."""
    keep = [j for j in range(S.shape[1]) if j != target]
    pos = list(enc.levels[enc.columns[target]]).index(">50K")
    clf = HistGradientBoostingClassifier(
        categorical_features=enc.cat_mask[keep],
        max_iter=500,
        learning_rate=0.05,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
    )
    clf.fit(S[:, keep], (S[:, target] == pos).astype(int))
    p = clf.predict_proba(H[:, keep])[:, 1]
    y = (H[:, target] == pos).astype(int)
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p))}


ANALYSIS_TERMS = ["intercept", "age", "education-num", "hours-per-week", "male", "married"]


def analysis_design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The analyst's model: logit P(income > 50K) on five terms."""
    X = np.column_stack(
        [
            np.ones(len(df)),
            df["age"].to_numpy(float),
            df["education-num"].to_numpy(float),
            df["hours-per-week"].to_numpy(float),
            (df["sex"].to_numpy() == "Male").astype(float),
            (df["marital-status"].to_numpy() == "Married-civ-spouse").astype(float),
        ]
    )
    y = (df["income"].to_numpy() == ">50K").astype(float)
    return X, y


def analysis_fit(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return logit(*analysis_design(df))


def ci_overlap(lo_r, hi_r, lo_s, hi_s) -> np.ndarray:
    """Interval overlap of Karr et al. (2006): the average of the shares of
    the real and of the synthetic interval covered by their intersection.
    1 means identical intervals, 0 disjoint ones."""
    inter = np.maximum(0.0, np.minimum(hi_r, hi_s) - np.maximum(lo_r, lo_s))
    return 0.5 * (inter / (hi_r - lo_r) + inter / (hi_s - lo_s))


# ----------------------------------------------------------------- privacy


def privacy(
    T: np.ndarray, H: np.ndarray, S: np.ndarray, enc: Encoder, rng: np.random.Generator,
    n_attack: int = 5000,
) -> dict[str, float]:
    """Distance-based privacy diagnostics against the training rows T.

    * exact copies: share of synthetic rows identical to some training row;
    * median DCR: median distance from a synthetic row to its closest
      training row (compare with the reference sample);
    * DCR share: share of synthetic rows whose closest real row is in the
      training set rather than the equally large holdout (Platzer and
      Reutterer 2021); 1/2 means the synthetic rows are no closer to the data
      the generator saw than to data it never saw;
    * membership inference: training rows (members) and holdout rows
      (non-members) are scored by their distance to the closest synthetic row;
      AUC of that score and the true-positive rate at a 1% false-positive rate
      (Carlini et al. 2022).
    """
    FT = distance_features(T, enc, T)
    FH = distance_features(H, enc, T)
    FS = distance_features(S, enc, T)

    def nearest(ref: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # The search uses the fast |a|^2 + |b|^2 - 2ab expansion, which leaves
        # rounding error of order 1e-4 on tiny distances; the distance to the
        # neighbour found is therefore recomputed exactly.
        idx = NearestNeighbors(n_neighbors=1).fit(ref).kneighbors(query, return_distance=False)[:, 0]
        return np.sqrt(((query - ref[idx]) ** 2).sum(axis=1)), idx

    d_train, i_train = nearest(FT, FS)
    d_hold, _ = nearest(FH, FS)
    share = np.mean(np.where(d_train == d_hold, 0.5, d_train < d_hold))

    mem = FT[rng.choice(len(FT), size=n_attack, replace=False)]
    non = FH[rng.choice(len(FH), size=n_attack, replace=False)]
    score = -np.r_[nearest(FS, mem)[0], nearest(FS, non)[0]]
    label = np.r_[np.ones(n_attack), np.zeros(n_attack)]
    fpr, tpr, _ = roc_curve(label, score)
    # exact copies by value (adding 0.0 turns any -0.0 into 0.0 before hashing)
    train_rows = {row.tobytes() for row in np.ascontiguousarray(T + 0.0)}
    copies = np.mean([row.tobytes() in train_rows for row in np.ascontiguousarray(S + 0.0)])
    return {
        "exact_copies": float(copies),
        "median_dcr": float(np.median(d_train)),
        "dcr_share": float(share),
        "mia_auc": float(roc_auc_score(label, score)),
        "mia_tpr_at_1pct": float(tpr[fpr <= 0.01].max()),
        "d_train": d_train,
        "nearest_train": i_train,
    }
