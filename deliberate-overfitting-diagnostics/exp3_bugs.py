"""Experiment 3 - which pipeline bugs does an overfitting check catch?

Six bugs are planted in a UCI Adult pipeline (one-hot encoding, 70/30 stratified
split, gradient boosting). Seven checks are run on each version:

  C1  overfit one batch: a 2x256 network must fit 64 training rows exactly
  C2  overfit the full training set: deep XGBoost must reach 99% training accuracy
  C3  shuffle-calibrated memorisation: a tree needs fewer leaves for the real labels
      than for shuffled labels (permutation p <= 0.05)
  C4  held-out signal: test accuracy beats the permuted-label null (p <= 0.05) and
      the majority-class baseline
  C5  single-column audit: no single column predicts the held-out label with AUC > 0.95
  C6  duplicate audit: under 1% of test rows have an exact copy in training
  C7  label audit: train and test prevalence within 5 points, both classes present

A check "flags" a version when it fails. The script also records the accuracy the
buggy pipeline would report and the accuracy the same model gets on clean data.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from common import ADULT_CAT, RES, encode_adult, load_adult, standardize

warnings.filterwarnings("ignore")
SEED = 0

BUGS = ["No bug",
        "Labels misaligned after a sort",
        "Every feature made constant",
        "Learning rate a million times too small",
        "Target leak in one column",
        "Minority class oversampled before the split",
        "Test labels never parsed"]
CHECKS = ["C1 overfit one batch", "C2 overfit full training set", "C3 shuffle-calibrated memorisation",
          "C4 held-out vs permuted labels", "C5 single-column audit", "C6 train-test duplicate audit",
          "C7 label audit"]


def floor_of(X: np.ndarray, y: np.ndarray) -> float:
    keys = pd.Series([r.tobytes() for r in np.ascontiguousarray(X)])
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return float(np.minimum(g["sum"], g["count"] - g["sum"]).sum() / len(y))


def build(bug: str):
    df, y = load_adult()
    rng = np.random.default_rng(SEED)
    raw = df.copy()
    if bug == "Target leak in one column":
        # an income band captured after the outcome is known, 3% transcription noise
        raw["income_band_followup"] = np.where(rng.random(len(y)) < 0.03, 1 - y, y)
    idx = np.arange(len(y))
    if bug == "Minority class oversampled before the split":
        pos = idx[y == 1]
        idx = np.concatenate([idx, pos, pos])          # minority rows tripled, then split
    yy = y[idx]
    itr, ite = train_test_split(np.arange(len(idx)), test_size=0.3, random_state=SEED, stratify=yy)
    enc = encode_adult(raw.drop(columns=[c for c in ["income_band_followup"] if c in raw]))
    if "income_band_followup" in raw:
        enc = np.column_stack([enc, raw["income_band_followup"].to_numpy(float)])
    X = enc[idx]
    Xtr, Xte, ytr, yte = X[itr], X[ite], yy[itr].copy(), yy[ite].copy()
    raw_tr, raw_te = raw.iloc[idx[itr]].reset_index(drop=True), raw.iloc[idx[ite]].reset_index(drop=True)
    lr_scale = 1.0
    if bug == "Labels misaligned after a sort":
        order = np.argsort(Xtr[:, 0], kind="stable")     # features sorted by age, labels not
        Xtr, raw_tr = Xtr[order], raw_tr.iloc[order].reset_index(drop=True)
    elif bug == "Every feature made constant":
        mu = Xtr.mean(0)
        Xtr, Xte = np.tile(mu, (len(Xtr), 1)), np.tile(mu, (len(Xte), 1))
    elif bug == "Learning rate a million times too small":
        lr_scale = 1e-6
    elif bug == "Test labels never parsed":
        yte = np.zeros_like(yte)                         # '>50K.' never matched '>50K'
    # clean, deployment-like evaluation set: original rows never seen in training
    seen = set(idx[itr])
    clean_rows = np.array([i for i in idx[ite] if i not in seen])
    clean_rows = np.unique(clean_rows)
    Xclean = enc[clean_rows].copy()
    if "income_band_followup" in raw:
        Xclean[:, -1] = 0.0                              # the follow-up field does not exist yet
    if bug == "Every feature made constant":
        Xclean = np.tile(Xtr[0], (len(clean_rows), 1))
    return dict(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, raw_tr=raw_tr, raw_te=raw_te,
                lr_scale=lr_scale, Xclean=Xclean, yclean=y[clean_rows],
                constant=bug == "Every feature made constant")


def deep_xgb(lr_scale):
    return XGBClassifier(n_estimators=500, max_depth=12, learning_rate=0.3 * lr_scale, reg_lambda=0.0,
                         min_child_weight=0.0, gamma=0.0, tree_method="hist", max_bin=1024,
                         n_jobs=2, random_state=SEED, verbosity=0)


def reg_xgb(lr_scale):
    return XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05 * lr_scale, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=5, tree_method="hist", n_jobs=2,
                         random_state=SEED, verbosity=0)


def run_checks(d) -> tuple[dict, dict]:
    rng = np.random.default_rng(SEED)
    Xtr, Xte, ytr, yte, s = d["Xtr"], d["Xte"], d["ytr"], d["yte"], d["lr_scale"]
    out, info = {}, {}

    # C1 overfit one batch
    b = rng.choice(len(ytr), 64, replace=False)
    xb = standardize(Xtr[b])
    mlp = MLPClassifier(hidden_layer_sizes=(256, 256), alpha=0.0, learning_rate_init=1e-3 * s,
                        batch_size=64, max_iter=500, tol=0.0, n_iter_no_change=10_000,
                        random_state=SEED).fit(xb, ytr[b])
    info["one_batch_acc"] = mlp.score(xb, ytr[b])
    out[CHECKS[0]] = info["one_batch_acc"] < 1.0

    # C2 overfit the full training set: the rule in its usual form ("it should reach ~100%").
    # The duplicate floor is recorded too; a model that only reaches the floor passes the
    # floor-aware version of the check, which is what happens when every row is identical.
    deep = deep_xgb(s).fit(Xtr, ytr)
    info["full_train_error"] = 1 - deep.score(Xtr, ytr)
    info["train_floor"] = floor_of(Xtr, ytr)
    out[CHECKS[1]] = info["full_train_error"] > 0.01

    # C3 shuffle-calibrated memorisation
    leaves = DecisionTreeClassifier(random_state=SEED).fit(Xtr, ytr).get_n_leaves()
    null = [DecisionTreeClassifier(random_state=SEED).fit(Xtr, rng.permutation(ytr)).get_n_leaves()
            for _ in range(39)]
    info["leaves_real"], info["leaves_shuffled_mean"] = int(leaves), float(np.mean(null))
    info["p_leaves"] = (1 + sum(v <= leaves for v in null)) / 40
    out[CHECKS[2]] = info["p_leaves"] > 0.05

    # C4 held-out accuracy against permuted test labels and the majority baseline
    model = reg_xgb(s).fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = float(np.mean(pred == yte))
    null_acc = np.array([np.mean(pred == rng.permutation(yte)) for _ in range(999)])
    base = max(yte.mean(), 1 - yte.mean())
    info["reported_test_acc"], info["majority_baseline"] = acc, float(base)
    info["p_holdout"] = (1 + np.sum(null_acc >= acc)) / 1000
    info["clean_acc"] = float(model.score(d["Xclean"], d["yclean"]))
    out[CHECKS[3]] = (info["p_holdout"] > 0.05) or (acc <= base)

    # C5 single-column audit on the raw columns
    best_auc, best_col = 0.5, ""
    if len(np.unique(yte)) == 2:
        n_tr = len(d["raw_tr"])
        for col in d["raw_tr"].columns:
            both = pd.concat([d["raw_tr"][col], d["raw_te"][col]], ignore_index=True)
            if col in ADULT_CAT:
                both = pd.get_dummies(both.astype(str), dtype=float).to_numpy()
            else:
                both = both.to_numpy(float)[:, None]
            if d["constant"]:
                both = np.zeros_like(both)
            enc_tr, enc_te = both[:n_tr], both[n_tr:]
            t = DecisionTreeClassifier(max_depth=4, random_state=SEED).fit(enc_tr, ytr)
            auc = roc_auc_score(yte, t.predict_proba(enc_te)[:, 1])
            if auc > best_auc:
                best_auc, best_col = auc, col
    info["best_single_auc"], info["best_single_col"] = float(best_auc), best_col
    out[CHECKS[4]] = best_auc > 0.95

    # C6 exact duplicates between test and train (all raw columns)
    tr_keys = set(d["raw_tr"].astype(str).agg("|".join, axis=1))
    te_keys = d["raw_te"].astype(str).agg("|".join, axis=1)
    info["test_rows_in_train"] = float(te_keys.isin(tr_keys).mean())
    out[CHECKS[5]] = info["test_rows_in_train"] > 0.01

    # C7 label audit
    info["prev_train"], info["prev_test"] = float(ytr.mean()), float(yte.mean())
    out[CHECKS[6]] = abs(ytr.mean() - yte.mean()) > 0.05 or len(np.unique(yte)) < 2
    return out, info


def main():
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    flags, infos = [], []
    for bug in BUGS:
        out, info = run_checks(build(bug))
        flags += [dict(bug=bug, check=c, flagged=bool(v)) for c, v in out.items()]
        infos.append(dict(bug=bug, **info))
        print(f"  {bug}: " + ", ".join(c.split()[0] for c, v in out.items() if v), flush=True)
    pd.DataFrame(flags).to_csv(RES / "exp3_detection_matrix.csv", index=False)
    pd.DataFrame(infos).to_csv(RES / "exp3_check_details.csv", index=False)
    print(f"exp3 done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
