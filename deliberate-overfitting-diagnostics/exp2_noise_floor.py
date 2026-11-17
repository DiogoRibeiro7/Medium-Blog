"""Experiment 2 - estimating the noise floor (Bayes error / residual variance).

Part A  Exact-duplicate floor on UCI Adult. Features are added one at a time. For
        each set: the plug-in floor (sum over feature cells of the minority count,
        divided by n), the training error of a tree grown to purity (which equals the
        plug-in floor), and two held-out errors (a cell lookup table with a model
        fallback, and a regularised XGBoost). For discrete X the plug-in floor is
        biased down and any held-out error is biased up, so they bracket the Bayes
        error of that feature set in expectation.

Part B  Cover-Hart nearest-neighbour bounds on the simulation (true Bayes error =
        eta), across sample sizes and numbers of irrelevant dimensions.

Part C  Learning curves (Cortes et al. 1994): test and training error of a
        regularised model against n, with a power-law fit a + b * n^(-c) whose
        asymptote a estimates the floor. Simulation (eta = 0.2) and Adult.

Part D  Regression: nearest-neighbour estimates of the residual variance
        E[Var(Y | X)] on a simulation with known noise and on California housing
        (log median house value), against cross-validated XGBoost error.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from common import RES, encode_adult, load_adult, load_housing, simulate, standardize

warnings.filterwarnings("ignore")

ADULT_ORDER = ["sex", "education", "marital_status", "occupation", "relationship", "race",
               "workclass", "native_country", "age", "hours_per_week", "capital_gain",
               "capital_loss", "fnlwgt"]


def reg_xgb_clf(seed=0, n_estimators=600):
    return XGBClassifier(n_estimators=n_estimators, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=5, tree_method="hist", n_jobs=2,
                         random_state=seed, verbosity=0)


def plug_in_floor(keys: pd.Series, y: np.ndarray) -> tuple[float, float, int]:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    minority = np.minimum(g["sum"], g["count"] - g["sum"]).sum()
    singleton_share = g.loc[g["count"] == 1, "count"].sum() / len(y)
    return minority / len(y), singleton_share, len(g)


def part_a(seed: int = 0) -> pd.DataFrame:
    df, y = load_adult()
    rows = []
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for k in range(1, len(ADULT_ORDER) + 1):
        cols = ADULT_ORDER[:k]
        sub = df[cols]
        keys = sub.astype(str).agg("|".join, axis=1)
        floor, singleton, cells = plug_in_floor(keys, y)
        X = encode_adult(sub)
        tree_err = 1 - DecisionTreeClassifier(random_state=seed).fit(X, y).score(X, y)
        lookup_err, xgb_err = [], []
        for tr, te in skf.split(X, y):
            model = reg_xgb_clf(seed).fit(X[tr], y[tr])
            p_model = model.predict(X[te])
            xgb_err.append(np.mean(p_model != y[te]))
            table = pd.DataFrame({"k": keys.iloc[tr].to_numpy(), "y": y[tr]}).groupby("k")["y"].mean()
            share = keys.iloc[te].map(table)
            p_lookup = np.where(share.isna(), p_model, (share.fillna(0) > 0.5).astype(int))
            lookup_err.append(np.mean(p_lookup != y[te]))
        rows.append(dict(step=k, added=cols[-1], n_features=k, cells=cells,
                         singleton_share=singleton, plug_in_floor=floor,
                         overfit_tree_train_error=tree_err,
                         lookup_cv_error=np.mean(lookup_err), xgb_cv_error=np.mean(xgb_err)))
        print(f"  A +{cols[-1]}: floor={floor:.4f} tree={tree_err:.4f} lookup={np.mean(lookup_err):.4f} "
              f"xgb={np.mean(xgb_err):.4f}", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ part B
def loo_1nn_error(x: np.ndarray, y: np.ndarray) -> float:
    nn = NearestNeighbors(n_neighbors=2).fit(x)
    _, idx = nn.kneighbors(x)
    own = np.arange(len(x))
    nb = np.where(idx[:, 0] == own, idx[:, 1], idx[:, 0])
    return float(np.mean(y[nb] != y))


def cover_hart_lower(r_nn: float) -> float:
    r = min(r_nn, 0.5)
    return (1 - np.sqrt(1 - 2 * r)) / 2


def part_b(reps: int = 4) -> pd.DataFrame:
    rows = []
    for eta in (0.1, 0.2, 0.3):
        for noise_dims in (0, 5, 20):
            for n in (500, 2000, 8000, 32000):
                for r in range(reps):
                    rng = np.random.default_rng(10_000 * r + 100 * noise_dims + int(eta * 10) + n)
                    x, y = simulate(n, eta, rng, d=5 + noise_dims)
                    r_nn = loo_1nn_error(x, y)
                    xt, yt = simulate(20_000, eta, rng, d=5 + noise_dims)
                    model = reg_xgb_clf(r, n_estimators=400).fit(x, y)
                    rows.append(dict(eta=eta, noise_dims=noise_dims, n=n, rep=r, r_1nn=r_nn,
                                     cover_hart_lower=cover_hart_lower(r_nn),
                                     xgb_test_error=1 - model.score(xt, yt)))
                print(f"  B eta={eta} noise={noise_dims} n={n}", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ part C
def power_law(n, a, b, c):
    return a + b * np.power(n, -c)


def fit_asymptote(ns: np.ndarray, err: np.ndarray) -> tuple[float, float, float, float]:
    p, cov = curve_fit(power_law, ns, err, p0=(err.min() * 0.9, 1.0, 0.5),
                       bounds=([0.0, 0.0, 0.01], [0.5, 100.0, 2.0]), maxfev=20_000)
    return p[0], float(np.sqrt(cov[0, 0])), p[1], p[2]


def fit_es_model(xtr, ytr, seed):
    """Regularised XGBoost with early stopping on 15% of the training rows."""
    xa, xv, ya, yv = train_test_split(xtr, ytr, test_size=0.15, random_state=seed, stratify=ytr)
    m = XGBClassifier(n_estimators=3000, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=3, tree_method="hist",
                      early_stopping_rounds=100, eval_metric="logloss", n_jobs=2,
                      random_state=seed, verbosity=0)
    m.fit(xa, ya, eval_set=[(xv, yv)], verbose=False)
    return m


def part_c(reps: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    # simulation
    for r in range(reps):
        rng = np.random.default_rng(555 + r)
        xt, yt = simulate(20_000, 0.2, rng)
        xall, yall = simulate(32_000, 0.2, rng)
        for n in (250, 500, 1000, 2000, 4000, 8000, 16000, 32000):
            x, y = xall[:n], yall[:n]
            m = fit_es_model(x, y, r)
            tree = DecisionTreeClassifier(random_state=r).fit(x, y)
            rows.append(dict(data="Simulation (Bayes error 0.20)", n=n, rep=r,
                             test_error=1 - m.score(xt, yt), train_error=1 - m.score(x, y),
                             overfit_train_error=1 - tree.score(x, y)))
        print(f"  C sim rep={r}", flush=True)
    # Adult, all 14 columns
    df, yA = load_adult()
    XA = encode_adult(df)
    for r in range(reps):
        itr, ite = train_test_split(np.arange(len(yA)), test_size=0.3, random_state=r, stratify=yA)
        rng = np.random.default_rng(r)
        itr = rng.permutation(itr)
        for n in (500, 1000, 2000, 4000, 8000, 16000, len(itr)):
            sel = itr[:n]
            m = fit_es_model(XA[sel], yA[sel], r)
            tree = DecisionTreeClassifier(random_state=r).fit(XA[sel], yA[sel])
            rows.append(dict(data="UCI Adult (14 columns)", n=n, rep=r,
                             test_error=1 - m.score(XA[ite], yA[ite]),
                             train_error=1 - m.score(XA[sel], yA[sel]),
                             overfit_train_error=1 - tree.score(XA[sel], yA[sel])))
        print(f"  C adult rep={r}", flush=True)
    lc = pd.DataFrame(rows)
    fits = []
    for name, g in lc.groupby("data"):
        mean = g.groupby("n")[["test_error", "train_error"]].mean().reset_index()
        a, se, b, c = fit_asymptote(mean["n"].to_numpy(float), mean["test_error"].to_numpy())
        fits.append(dict(data=name, asymptote=a, asymptote_se=se, b=b, c=c,
                         largest_n_test_error=mean["test_error"].iloc[-1],
                         largest_n_train_error=mean["train_error"].iloc[-1]))
    return lc, pd.DataFrame(fits)


# ------------------------------------------------------------------ part D
def nn_residual_variance(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two nearest-neighbour estimates of E[Var(Y|X)].

    product:    mean(Y^2) - mean(Y * Y_nn), Y centred (Devroye, Gyorfi, Lugosi & Walk 2018)
    difference: mean((Y - Y_nn)^2) / 2
    """
    nn = NearestNeighbors(n_neighbors=2).fit(x)
    _, idx = nn.kneighbors(x)
    own = np.arange(len(x))
    nb = np.where(idx[:, 0] == own, idx[:, 1], idx[:, 0])
    yc = y - y.mean()
    return float(np.mean(yc ** 2) - np.mean(yc * yc[nb])), float(np.mean((y - y[nb]) ** 2) / 2)


def f_reg(x):
    return np.sin(np.pi * x[:, 0]) + 0.5 * x[:, 1] ** 2 + x[:, 2] * x[:, 3]


def part_d(reps: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for sigma in (0.5, 1.0):
        for noise_dims in (0, 5):
            for n in (1000, 4000, 16000, 64000):
                for r in range(reps):
                    rng = np.random.default_rng(31 * r + n + int(sigma * 10) + noise_dims)
                    x = rng.standard_normal((n, 5 + noise_dims))
                    y = f_reg(x) + sigma * rng.standard_normal(n)
                    prod, diff = nn_residual_variance(x, y)
                    rows.append(dict(sigma2=sigma ** 2, noise_dims=noise_dims, n=n, rep=r,
                                     product_estimate=prod, difference_estimate=diff))
                print(f"  D sim sigma={sigma} noise={noise_dims} n={n}", flush=True)
    sim = pd.DataFrame(rows)

    X, y = load_housing()
    cols = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
            "population", "households", "median_income"]
    sets = {"All features": np.arange(X.shape[1]),
            "Location and income": np.array([0, 1, 7]),
            "Location only": np.array([0, 1])}
    hrows = []
    kf = KFold(5, shuffle=True, random_state=0)
    for name, idx in sets.items():
        Xs = standardize(X[:, idx])
        prod, diff = nn_residual_variance(Xs, y)
        mse, knn5, knn10 = [], [], []
        for tr, te in kf.split(Xs):
            m = XGBRegressor(n_estimators=1500, max_depth=6, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                             n_jobs=2, random_state=0, verbosity=0).fit(Xs[tr], y[tr])
            mse.append(np.mean((m.predict(Xs[te]) - y[te]) ** 2))
            for k, store in ((5, knn5), (10, knn10)):
                knn = KNeighborsRegressor(n_neighbors=k).fit(Xs[tr], y[tr])
                store.append(np.mean((knn.predict(Xs[te]) - y[te]) ** 2))
        tree = DecisionTreeRegressor(random_state=0).fit(Xs, y)
        hrows.append(dict(feature_set=name, n_columns=len(idx), var_y=float(np.var(y)),
                          product_estimate=prod, difference_estimate=diff,
                          xgb_cv_mse=float(np.mean(mse)), knn5_cv_mse=float(np.mean(knn5)),
                          knn10_cv_mse=float(np.mean(knn10)),
                          overfit_tree_train_mse=float(np.mean((tree.predict(Xs) - y) ** 2)),
                          distinct_rows=int(len(np.unique(X[:, idx], axis=0)))))
        print(f"  D housing {name}", flush=True)
    return sim, pd.DataFrame(hrows)


def main(parts: str = "abcd"):
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    if "a" in parts:
        part_a().to_csv(RES / "exp2a_adult_duplicate_floor.csv", index=False)
    if "d" in parts:
        sim, house = part_d()
        sim.to_csv(RES / "exp2d_residual_variance_sim.csv", index=False)
        house.to_csv(RES / "exp2d_residual_variance_housing.csv", index=False)
    if "c" in parts:
        lc, fits = part_c()
        lc.to_csv(RES / "exp2c_learning_curves.csv", index=False)
        fits.to_csv(RES / "exp2c_asymptotes.csv", index=False)
    if "b" in parts:
        part_b().to_csv(RES / "exp2b_cover_hart.csv", index=False)
    print(f"exp2 done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "abcd")
