"""Classical baselines, fitted afresh on each context with statsforecast."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def make_model(spec: str):
    """Build a statsforecast model from the short specification used in the config."""
    from statsforecast.models import MSTL, AutoETS, SeasonalNaive

    spec = spec.replace(" ", "")
    if m := re.fullmatch(r"SeasonalNaive\((\d+)\)", spec):
        return SeasonalNaive(season_length=int(m.group(1)), alias="m")
    if m := re.fullmatch(r"AutoETS\((\d+)\)", spec):
        return AutoETS(season_length=int(m.group(1)), alias="m")
    if m := re.fullmatch(r"MSTL\(([\d,]+)\)\+AutoETS\(([AMNZ]{3})\)", spec):
        seasons = [int(s) for s in m.group(1).split(",")]
        return MSTL(season_length=seasons, trend_forecaster=AutoETS(model=m.group(2)), alias="m")
    raise ValueError(f"unknown baseline specification {spec!r}")


def levels_for(quantiles) -> list[int]:
    """Central interval levels whose bounds are the requested quantiles."""
    levels = sorted({int(round(abs(2 * q - 1) * 100)) for q in quantiles if q != 0.5})
    return levels


def column_for(q: float) -> str:
    if q == 0.5:
        return "m"
    level = int(round(abs(2 * q - 1) * 100))
    return f"m-lo-{level}" if q < 0.5 else f"m-hi-{level}"


def forecast(spec: str, contexts: np.ndarray, horizon: int, quantiles, n_jobs: int = -1) -> np.ndarray:
    """Quantile forecasts with shape (origins, horizon, quantiles).

    Each row of ``contexts`` is a separate series; the model is fitted on that row only.
    The point forecast of these models is the mean of a symmetric predictive distribution,
    so it is also the median.
    """
    from statsforecast import StatsForecast

    contexts = np.asarray(contexts, dtype=float)
    n, length = contexts.shape
    ids = np.repeat([f"o{i:06d}" for i in range(n)], length)
    df = pd.DataFrame({"unique_id": ids, "ds": np.tile(np.arange(1, length + 1), n), "y": contexts.ravel()})
    sf = StatsForecast(models=[make_model(spec)], freq=1, n_jobs=n_jobs)
    out = sf.forecast(df=df, h=horizon, level=levels_for(quantiles))
    if "unique_id" not in out.columns:
        out = out.reset_index()
    out = out.sort_values(["unique_id", "ds"])
    result = np.empty((n, horizon, len(quantiles)))
    for k, q in enumerate(quantiles):
        col = column_for(q)
        if col not in out.columns:
            raise KeyError(f"{spec}: missing column {col}; got {list(out.columns)}")
        result[:, :, k] = out[col].to_numpy(dtype=float).reshape(n, horizon)
    return result
