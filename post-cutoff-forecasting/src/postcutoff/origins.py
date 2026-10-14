"""Forecast origins and the matrices the models are fed."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data import Prepared


def build_origins(prep: Prepared, cfg: Config) -> pd.DataFrame:
    """One row per candidate origin, with whether it is scored and, if not, why."""
    n = prep.index.size
    C, H = cfg.context_length, cfg.horizon
    rows = []
    for w in cfg.windows:
        for origin in pd.date_range(w.first_origin, w.last_origin, freq="D"):
            pos = int(prep.index.get_indexer([origin])[0])
            status, reason, n_imputed = "scored", "", 0
            if pos < 0:
                status, reason = "excluded", "origin outside the snapshot"
            elif pos - C < 0:
                status, reason = "excluded", "context starts before the snapshot"
            elif pos + H > n:
                status, reason = "excluded", "forecast period ends after the snapshot"
            else:
                n_imputed = int(prep.imputed[pos - C : pos].sum())
                target_ok = prep.scorable[pos : pos + H]
                if not target_ok.all():
                    status = "excluded"
                    parts = []
                    if prep.masked[pos : pos + H].any():
                        parts.append("masked hours")
                    if prep.missing[pos : pos + H].any():
                        parts.append("missing hours")
                    reason = "forecast period contains " + " and ".join(parts)
            rows.append(
                {
                    "window": w.name,
                    "origin": origin,
                    "pos": pos,
                    "status": status,
                    "reason": reason,
                    "context_imputed_hours": n_imputed,
                }
            )
    frame = pd.DataFrame(rows)
    if frame["origin"].duplicated().any():
        raise ValueError("windows overlap: an origin belongs to more than one window")
    return frame


def contexts(prep: Prepared, positions: np.ndarray, length: int) -> np.ndarray:
    positions = np.asarray(positions, dtype=int)
    offsets = np.arange(-length, 0)
    return prep.inputs[positions[:, None] + offsets[None, :]].astype(np.float64)


def targets(prep: Prepared, positions: np.ndarray, horizon: int) -> np.ndarray:
    positions = np.asarray(positions, dtype=int)
    offsets = np.arange(horizon)
    return prep.observed[positions[:, None] + offsets[None, :]].astype(np.float64)


def window_status(cfg: Config, weights_dates: dict | None = None) -> pd.DataFrame:
    """For each window and model, where the window lies relative to what the model can have
    seen. The bound is the later of the public release date and the last commit of the
    weights file actually used (when a run has recorded it)."""
    weights_dates = weights_dates or {}
    rows = []
    for w in cfg.windows:
        for model, release in cfg.releases.items():
            bound = release
            if model in weights_dates:
                bound = max(bound, pd.Timestamp(weights_dates[model]).tz_convert("UTC"))
            if w.first_origin >= bound:
                status = "after release"
            elif w.last_origin + pd.Timedelta(hours=cfg.horizon) <= bound:
                status = "before release"
            else:
                status = "spans release"
            rows.append(
                {
                    "window": w.name,
                    "model": model,
                    "release": release.date().isoformat(),
                    "weights_last_commit": weights_dates.get(model),
                    "bound": bound.isoformat(),
                    "status": status,
                }
            )
    return pd.DataFrame(rows)
