"""One-day-ahead point forecasts (PLAN.md, "Algorithms")."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feed.data import Panel

ALGOS = ("naive7", "ma4w", "lgbm_l2", "lgbm_tweedie")
LAGS = (1, 2, 3, 4, 5, 6, 7, 14, 21, 28)
MAX_LAG = 28
CATEGORICAL = (
    "city_id",
    "store_id",
    "product_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
)
LGBM = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "random_state": 7,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 4,
    "verbose": -1,
}
OBJECTIVES = {
    "lgbm_l2": {"objective": "regression"},
    "lgbm_tweedie": {"objective": "tweedie", "tweedie_variance_power": 1.2},
}


def naive7(y: np.ndarray, t: int) -> np.ndarray:
    return y[:, t - 7]


def ma4w(y: np.ndarray, t: int) -> np.ndarray:
    return y[:, [t - 7, t - 14, t - 21, t - 28]].mean(axis=1)


def weekday(dates: np.ndarray) -> np.ndarray:
    return pd.to_datetime(pd.Series(dates)).dt.dayofweek.to_numpy()


def features(panel: Panel, y: np.ndarray, days: list[int]) -> pd.DataFrame:
    """One row per series and day t, built only from y up to day t - 1 and the planned covariates of day t."""
    dow = weekday(panel.dates)
    blocks = []
    for t in days:
        if t < MAX_LAG:
            raise ValueError(f"day {t} has no lag {MAX_LAG}")
        f = {f"lag{k}": y[:, t - k] for k in LAGS}
        f["mean7"] = y[:, t - 7 : t].mean(axis=1)
        f["mean14"] = y[:, t - 14 : t].mean(axis=1)
        f["mean28"] = y[:, t - 28 : t].mean(axis=1)
        f["std7"] = y[:, t - 7 : t].std(axis=1)
        f["ma4w"] = ma4w(y, t)
        f["dow"] = np.full(panel.n, dow[t])
        f["discount"] = panel.discount[:, t]
        f["holiday"] = panel.holiday[:, t]
        f["activity"] = panel.activity[:, t]
        for c in CATEGORICAL:
            f[c] = panel.ids[c]
        blocks.append(pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()}))
    df = pd.concat(blocks, ignore_index=True)
    for c in (*CATEGORICAL, "dow"):
        df[c] = df[c].astype(np.int32).astype("category")
    return df


def lgbm_forecasts(
    panel: Panel, y: np.ndarray, algo: str, fit_days: list[int], predict_days: list[int], settings: dict | None = None
) -> tuple[np.ndarray, object]:
    """Fit one global model on fit_days, return forecasts for predict_days (nan elsewhere) and the model."""
    import lightgbm as lgb

    params = {**LGBM, **OBJECTIVES[algo], **(settings or {})}
    train = features(panel, y, fit_days)
    target = np.concatenate([y[:, t] for t in fit_days])
    model = lgb.LGBMRegressor(**params)
    model.fit(train, target, categorical_feature=[*CATEGORICAL, "dow"])
    cats = {c: train[c].cat.categories for c in (*CATEGORICAL, "dow")}
    out = np.full(y.shape, np.nan)
    for t in predict_days:
        x = features(panel, y, [t])
        for c, levels in cats.items():
            x[c] = pd.Categorical(x[c].astype(np.int32), categories=levels)
        out[:, t] = np.maximum(model.predict(x), 0.0)
    return out, model


def forecasts(
    panel: Panel, y: np.ndarray, fit_days: list[int], predict_days: list[int], settings: dict | None = None
) -> dict[str, np.ndarray]:
    out = {}
    for name, rule in (("naive7", naive7), ("ma4w", ma4w)):
        f = np.full(y.shape, np.nan)
        for t in predict_days:
            f[:, t] = np.maximum(rule(y, t), 0.0)
        out[name] = f
    for name in ("lgbm_l2", "lgbm_tweedie"):
        out[name], _ = lgbm_forecasts(panel, y, name, fit_days, predict_days, settings)
    return out
