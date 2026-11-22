"""Day-ahead forecasts issued at the local cutoff on the day before.

For a target day D in a country: the cutoff is cutoff_local_hour on D-1, local time; the
context is the `context` hours before the cutoff (masked and missing hours replaced by the
same hour one week earlier); the horizon runs from the cutoff to the end of D (29 to 31
hours, depending on daylight-saving changes) and only D's hours are scored.

A target day is skipped, and the reason recorded, when any of its hours is masked or
missing, or when the context would start before the data.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .calendar import date_range
from .config import DAY_TYPES, Config
from .data import Prepared
from .daytype import code
from .forecasters import Batch

FLAG_LEVELS = ("ordinary", "national_holiday")


@dataclass
class Origin:
    target: date
    cutoff: int  # index of the first hour not observed
    horizon: int  # hours from the cutoff to the end of the target day
    first_target: int  # offset of the target day's first hour within the horizon
    n_target: int


def origins(cfg: Config, prep: Prepared, first: date, last: date, context: int) -> tuple[list[Origin], list[dict]]:
    tz = ZoneInfo(prep.country.tz)
    position = {ts: i for i, ts in enumerate(prep.index)}
    kept, skipped = [], []
    for d in date_range(first, last):
        prev = d - timedelta(days=1)
        cutoff_ts = pd.Timestamp(
            datetime(prev.year, prev.month, prev.day, cfg.cutoff_local_hour, tzinfo=tz)
        ).tz_convert("UTC")
        target = np.flatnonzero(prep.local_date == d)
        if cutoff_ts not in position or target.size == 0:
            skipped.append({"date": d.isoformat(), "reason": "outside the data"})
            continue
        c = position[cutoff_ts]
        if c - context < 0:
            skipped.append({"date": d.isoformat(), "reason": "context starts before the data"})
            continue
        if not prep.scorable[target].all():
            n_masked = int(prep.masked[target].sum())
            skipped.append({"date": d.isoformat(), "reason": "masked hours" if n_masked else "missing hours"})
            continue
        end = int(target[-1]) + 1
        kept.append(Origin(d, c, end - c, int(target[0]) - c, int(target.size)))
    return kept, skipped


def hourly_codes(prep: Prepared, table: pd.DataFrame, variant: str) -> np.ndarray:
    """Day-type code of every hour, for a covariate variant."""
    if variant == "flag":
        by_day = {d: int(bool(n)) for d, n in table["national"].items()}
    elif variant == "oracle":
        by_day = {d: code(v, w) for d, v, w in zip(table.index, table["oracle"], table["weekday_rule"], strict=True)}
    elif variant.startswith("llm:"):
        col = variant
        by_day = {d: code(v, w) for d, v, w in zip(table.index, table[col], table["weekday_rule"], strict=True)}
    else:
        raise ValueError(f"variant {variant!r} has no calendar covariate")
    default = {
        d: DAY_TYPES.index("workday") if d.weekday() < 5 else (1 if d.weekday() == 5 else 2)
        for d in set(prep.local_date)
    }
    if variant == "flag":
        default = {d: 0 for d in default}
    return np.array([by_day.get(d, default[d]) for d in prep.local_date], dtype=np.int64)


def make_batch(
    prep: Prepared, items: list[Origin], context: int, horizon: int, variant: str, codes: np.ndarray | None
) -> Batch:
    """Contexts and covariates for a list of origins, padded to a common horizon."""
    contexts = np.stack([prep.inputs[o.cutoff - context : o.cutoff] for o in items])
    temperature = calendar = None
    if variant != "none":
        temperature = np.stack([_window(prep.temperature, o.cutoff - context, o.cutoff + horizon) for o in items])
    if variant not in ("none", "temp"):
        calendar = np.stack([_window(codes, o.cutoff - context, o.cutoff + horizon) for o in items])
    levels = FLAG_LEVELS if variant == "flag" else DAY_TYPES
    return Batch(contexts.astype(np.float64), horizon, temperature, calendar, levels)


def _window(values: np.ndarray, start: int, end: int) -> np.ndarray:
    """values[start:end], repeating the last value if the series ends before `end`."""
    out = values[start : min(end, values.size)]
    if out.size < end - start:
        out = np.concatenate([out, np.repeat(out[-1:], end - start - out.size)])
    return out


def forecast_path(cfg: Config, forecaster: str, variant: str, country: str) -> Path:
    return cfg.forecasts_dir / f"{forecaster}__{variant.replace(':', '-')}__{country}.parquet"


def run_one(
    cfg: Config, adapter, forecaster: str, variant: str, prep: Prepared, table: pd.DataFrame, force: bool = False
) -> Path:
    """Forecast every target day for one forecaster, variant and country; write a parquet file
    with the target hours and their quantiles."""
    spec = cfg.forecasters[forecaster]
    path = forecast_path(cfg, forecaster, variant, prep.country.key)
    if path.exists() and not force:
        return path
    context = min(cfg.context_hours, spec.max_context)
    items, skipped = origins(cfg, prep, cfg.forecast_first_day, cfg.forecast_last_day, context)
    codes = hourly_codes(prep, table, variant) if variant not in ("none", "temp") else None
    horizon = max(o.horizon for o in items)
    rows = []
    started = time.perf_counter()
    step = max(1, spec.batch_size)
    for b in range(0, len(items), step):
        chunk = items[b : b + step]
        batch = make_batch(prep, chunk, context, horizon, variant, codes)
        q = adapter.forecast(batch, cfg.quantiles)
        if q.shape != (len(chunk), horizon, len(cfg.quantiles)):
            raise ValueError(
                f"{forecaster}: forecast shape {q.shape}, expected {(len(chunk), horizon, len(cfg.quantiles))}"
            )
        if not np.isfinite(q).all():
            raise ValueError(f"{forecaster}|{variant}: non-finite forecasts for {chunk[0].target}")
        for o, qi in zip(chunk, q, strict=True):
            sl = slice(o.first_target, o.first_target + o.n_target)
            hours = prep.index[o.cutoff + o.first_target : o.cutoff + o.first_target + o.n_target]
            frame = pd.DataFrame(qi[sl], columns=[f"q{q:g}" for q in cfg.quantiles])
            frame.insert(0, "timestamp_utc", hours)
            frame.insert(0, "date", o.target)
            rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    meta = {
        "forecaster": forecaster,
        "variant": variant,
        "country": prep.country.key,
        "context_hours": context,
        "horizon_hours": horizon,
        "origins": len(items),
        "skipped": skipped,
        "seconds": round(time.perf_counter() - started, 1),
        "seconds_per_origin": round((time.perf_counter() - started) / max(1, len(items)), 4),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"{forecaster}|{variant}|{prep.country.key}: {len(items)} days in {meta['seconds']} s")
    return path


def read_forecast(cfg: Config, forecaster: str, variant: str, country: str) -> pd.DataFrame | None:
    path = forecast_path(cfg, forecaster, variant, country)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame
