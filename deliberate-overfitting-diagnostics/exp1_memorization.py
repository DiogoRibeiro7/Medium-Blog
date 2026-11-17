"""Experiment 1 - what a deliberately overfit model can and cannot tell you about signal.

Part A  Memorisation grid on the simulation. For each label-noise rate eta (Bayes
        error = eta) and for real versus shuffled labels, four models are pushed to
        memorise the training set. Training accuracy, test accuracy and the effort
        each model needed to memorise (tree leaves at purity, boosting rounds to zero
        training error, network epochs to 100% training accuracy) are recorded.

Part B  Detection power. Three ways of deciding "there is signal" are run on many
        fresh data sets near the pure-noise end:
          * the memorisation rule ("if it can memorise, the signal is there");
          * a shuffle-calibrated memorisation test (leaves of a tree grown to purity
            on the real labels versus on K shuffled copies);
          * a held-out permutation test (test accuracy of a model fit on 70% of the
            same rows, against the accuracy distribution under permuted test labels).

Part C  Feature-set ranking on UCI Adult. Training accuracy of overfit models versus
        held-out accuracy, for feature sets that include an identifier and pure noise.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from common import RES, encode_adult, load_adult, memorizers, simulate, standardize

warnings.filterwarnings("ignore")
ETAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
N, N_TEST = 2000, 20_000


# ---------------------------------------------------------------- effort measures
def tree_leaves(x, y, seed=0) -> int:
    return int(DecisionTreeClassifier(random_state=seed).fit(x, y).get_n_leaves())


def xgb_rounds(x, y, cap=2000, seed=0) -> float:
    """Boosting rounds (depth-3 trees, no penalties) until training error is zero."""
    m = XGBClassifier(n_estimators=cap, max_depth=3, learning_rate=0.3, reg_lambda=0.0,
                      min_child_weight=0.0, gamma=0.0, eval_metric="error", tree_method="hist",
                      max_bin=1024, n_jobs=2, random_state=seed, verbosity=0)
    m.fit(x, y, eval_set=[(x, y)], verbose=False)
    err = np.asarray(m.evals_result()["validation_0"]["error"])
    hit = np.flatnonzero(err == 0)
    return float(hit[0] + 1) if hit.size else np.nan


def mlp_memorise(x, y, x_test, y_test, cap=1000, every=5, seed=0):
    """Train a 2x256 network until it fits every training label; return epochs and test acc."""
    m = MLPClassifier(hidden_layer_sizes=(256, 256), alpha=0.0, learning_rate_init=1e-3,
                      batch_size=200, random_state=seed)
    classes = np.array([0, 1])
    for ep in range(1, cap + 1):
        m.partial_fit(x, y, classes=classes)
        if ep % every == 0 and m.score(x, y) == 1.0:
            return ep, 1.0, m.score(x_test, y_test)
    return np.nan, m.score(x, y), m.score(x_test, y_test)


# ---------------------------------------------------------------- part A
def part_a(reps: int = 3) -> pd.DataFrame:
    rows = []
    for eta in ETAS:
        for r in range(reps):
            rng = np.random.default_rng(1000 * r + int(eta * 100))
            x, y = simulate(N, eta, rng)
            xt, yt = simulate(N_TEST, eta, rng)
            y_shuf = rng.permutation(y)
            xs, xts = standardize(x, xt)
            for labels, yy in (("real", y), ("shuffled", y_shuf)):
                for name, model in memorizers(seed=r).items():
                    if name.startswith("Neural"):
                        epochs, tr, te = mlp_memorise(xs, yy, xts, yt, seed=r)
                        rows.append(dict(eta=eta, rep=r, labels=labels, model=name,
                                         train_acc=tr, test_acc=te, effort=epochs,
                                         effort_unit="epochs to 100% train accuracy"))
                        continue
                    model.fit(x, yy)
                    rows.append(dict(eta=eta, rep=r, labels=labels, model=name,
                                     train_acc=model.score(x, yy), test_acc=model.score(xt, yt),
                                     effort=np.nan, effort_unit=""))
                rows.append(dict(eta=eta, rep=r, labels=labels, model="Decision tree (grown to purity)",
                                 train_acc=np.nan, test_acc=np.nan, effort=tree_leaves(x, yy, r),
                                 effort_unit="leaves at purity"))
                rows.append(dict(eta=eta, rep=r, labels=labels, model="XGBoost depth 3",
                                 train_acc=np.nan, test_acc=np.nan, effort=xgb_rounds(x, yy, seed=r),
                                 effort_unit="rounds to zero train error"))
            print(f"  A eta={eta} rep={r}", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- part B
B_ETAS = [0.30, 0.35, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]


def part_b(reps: int = 60, k_shuffle: int = 39, k_perm: int = 999) -> pd.DataFrame:
    rows = []
    for eta in B_ETAS:
        for r in range(reps):
            rng = np.random.default_rng(7_000_000 + 1000 * r + int(eta * 1000))
            x, y = simulate(N, eta, rng)
            # (1) memorisation rule: a tree grown to purity always fits continuous X
            memorised = DecisionTreeClassifier(random_state=r).fit(x, y).score(x, y) == 1.0
            # (2) shuffle-calibrated leaves test (fewer leaves = easier = signal)
            real = tree_leaves(x, y, r)
            null = np.array([tree_leaves(x, rng.permutation(y), r) for _ in range(k_shuffle)])
            p_leaves = (1 + np.sum(null <= real)) / (k_shuffle + 1)
            # (3) held-out permutation test on a 70/30 split of the same rows
            xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=r, stratify=y)
            rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=2,
                                        random_state=r).fit(xtr, ytr)
            pred = rf.predict(xte)
            acc = np.mean(pred == yte)
            null_acc = np.array([np.mean(pred == rng.permutation(yte)) for _ in range(k_perm)])
            p_hold = (1 + np.sum(null_acc >= acc)) / (k_perm + 1)
            rows.append(dict(eta=eta, rep=r, memorised=memorised, leaves_real=real,
                             leaves_null_mean=null.mean(), p_leaves=p_leaves,
                             holdout_acc=acc, p_holdout=p_hold))
        print(f"  B eta={eta}", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- part C
def part_c(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df, y = load_adult()
    n = len(df)
    random_id = rng.permutation(n).astype(float)
    noise = rng.standard_normal((n, 20))
    no_fnl = df.drop(columns=["fnlwgt"])
    sets = {
        "All 14 census columns": encode_adult(df),
        "13 columns (sampling weight dropped)": encode_adult(no_fnl),
        "13 columns + a random row ID": np.column_stack([encode_adult(no_fnl), random_id]),
        "13 columns + 20 pure-noise columns": np.column_stack([encode_adult(no_fnl), noise]),
        "Education and sex only": encode_adult(df[["education", "sex"]]),
        "Sampling weight only": df[["fnlwgt"]].to_numpy(float),
        "Random row ID only": random_id[:, None],
    }
    idx_tr, idx_te = train_test_split(np.arange(n), test_size=0.3, random_state=seed, stratify=y)
    base = max(y[idx_te].mean(), 1 - y[idx_te].mean())
    rows = []
    for name, X in sets.items():
        xtr, xte, ytr, yte = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]
        tree = DecisionTreeClassifier(random_state=seed).fit(xtr, ytr)
        deep = XGBClassifier(n_estimators=500, max_depth=12, learning_rate=0.3, reg_lambda=0.0,
                             min_child_weight=0.0, gamma=0.0, tree_method="hist", max_bin=1024,
                             n_jobs=2, random_state=seed, verbosity=0).fit(xtr, ytr)
        reg = XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                            n_jobs=2, random_state=seed, verbosity=0).fit(xtr, ytr)
        rows.append(dict(feature_set=name, n_columns=X.shape[1],
                         distinct_rows_share=len(np.unique(xtr, axis=0)) / len(xtr),
                         tree_train_acc=tree.score(xtr, ytr), tree_test_acc=tree.score(xte, yte),
                         deep_xgb_train_acc=deep.score(xtr, ytr), deep_xgb_test_acc=deep.score(xte, yte),
                         reg_xgb_train_acc=reg.score(xtr, ytr), reg_xgb_test_acc=reg.score(xte, yte),
                         majority_baseline=base))
        print(f"  C {name}", flush=True)
    return pd.DataFrame(rows)


def main(parts: str = "abc"):
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    if "c" in parts:
        part_c().to_csv(RES / "exp1c_feature_set_ranking.csv", index=False)
    if "b" in parts:
        part_b().to_csv(RES / "exp1b_detection_power.csv", index=False)
    if "a" in parts:
        part_a().to_csv(RES / "exp1a_memorisation_grid.csv", index=False)
    print(f"exp1 done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "abc")
