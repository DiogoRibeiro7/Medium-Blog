"""Downloading, storing and checking the Portuguese load series.

The series is hourly, indexed by the start of each hour in UTC. Missing hours are kept as
NaN in the snapshot; nothing is filled in at that stage.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

HOUR = pd.Timedelta(hours=1)


# --------------------------------------------------------------------------- download


def parse_public_power(payload: dict, series: str) -> pd.Series:
    """Extract one named series from an Energy-Charts ``public_power`` response."""
    stamps = payload.get("unix_seconds")
    if not stamps:
        raise ValueError("response has no unix_seconds")
    names = [p.get("name") for p in payload.get("production_types", [])]
    matches = [p for p in payload["production_types"] if p.get("name") == series]
    if len(matches) != 1:
        raise ValueError(f"expected one series named {series!r}, found {names}")
    values = matches[0].get("data")
    if values is None or len(values) != len(stamps):
        raise ValueError(f"series {series!r} has {len(values or [])} values for {len(stamps)} timestamps")
    index = pd.to_datetime(np.asarray(stamps, dtype="int64"), unit="s", utc=True)
    out = pd.Series([np.nan if v is None else float(v) for v in values], index=index, name="load_mw")
    return out


def to_hourly(series: pd.Series) -> tuple[pd.Series, dict]:
    """Hourly means from a series at hourly or finer resolution.

    Each hour must be covered by evenly spaced stamps starting on the hour (one stamp, or
    four at :00, :15, :30 and :45, and so on) with every value present; otherwise the hour
    is missing. Values in MW are average power, so the hourly value is their mean.
    Returns the hourly series and how many hours came at each resolution.
    """
    if series.empty:
        raise ValueError("empty series")
    hour = series.index.floor("h")
    minutes = series.index.minute + series.index.second / 60.0
    frame = pd.DataFrame({"hour": hour, "minute": minutes, "value": series.to_numpy()})
    rows = {}
    resolution: dict[str, int] = {}
    for h, g in frame.groupby("hour", sort=True):
        n = len(g)
        expected = list(np.arange(n) * (60.0 / n)) if 60 % n == 0 else None
        regular = expected is not None and np.allclose(np.sort(g["minute"].to_numpy()), expected)
        complete = regular and not g["value"].isna().any()
        rows[h] = float(g["value"].mean()) if complete else np.nan
        key = f"{60 // n} min" if regular else "irregular"
        resolution[key] = resolution.get(key, 0) + 1
    out = pd.Series(list(rows.values()), index=pd.DatetimeIndex(list(rows.keys())).tz_convert("UTC"), name="load_mw")
    return out, resolution


def fetch_energy_charts(cfg: Config, today: pd.Timestamp | None = None, session=None) -> tuple[pd.Series, dict]:
    """Download the load series one calendar year at a time.

    Returns the hourly series up to the settle cutoff and a record of the requests.
    """
    import requests

    session = session or requests.Session()
    today = (today or pd.Timestamp.now(tz="UTC")).normalize()
    cutoff = today - pd.Timedelta(days=cfg.settle_days)
    requests_made = []
    parts = []
    years = range(cfg.first_day.year, cutoff.year + 1)
    for i, year in enumerate(years):
        start = max(cfg.first_day, pd.Timestamp(year=year, month=1, day=1, tz="UTC"))
        end = min(cutoff, pd.Timestamp(year=year, month=12, day=31, tz="UTC"))
        params = {
            "country": cfg.country,
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
        }
        payload = _get_json(session, cfg.endpoint, params)
        raw = parse_public_power(payload, cfg.series)
        part, resolution = to_hourly(raw)
        parts.append(part)
        requests_made.append(
            {
                "url": cfg.endpoint,
                "params": params,
                "stamps": int(raw.size),
                "hours": int(part.size),
                "hours_by_resolution": resolution,
                "first": raw.index.min().isoformat(),
                "last": raw.index.max().isoformat(),
            }
        )
        if i < len(years) - 1:
            time.sleep(cfg.pause_seconds)

    series = pd.concat(parts)
    # Where two requests return the same hour, keep the last value that is not missing.
    series = series.groupby(level=0).last().sort_index()
    # A plain-date request covers the local day, so the first hours of the following UTC day
    # can be present; keep only hours that start before the cutoff.
    series = series[(series.index >= cfg.first_day) & (series.index < cutoff)]
    series = regularise(series)
    record = {"requests": requests_made, "cutoff": cutoff.isoformat(), "fetched_at": _now()}
    return series, record


def _get_json(session, url: str, params: dict, attempts: int = 5) -> dict:
    last = None
    for attempt in range(attempts):
        response = session.get(url, params=params, timeout=120)
        if response.status_code == 200:
            return response.json()
        last = f"HTTP {response.status_code}: {response.text[:200]}"
        if response.status_code in (429, 500, 502, 503, 504):
            wait = float(response.headers.get("Retry-After", 30 * (attempt + 1)))
            time.sleep(min(wait, 300))
            continue
        break
    raise RuntimeError(f"request to {url} with {params} failed: {last}")


def regularise(series: pd.Series) -> pd.Series:
    """Put the series on a complete hourly grid; hours not returned become NaN."""
    if series.index.tz is None:
        raise ValueError("index must be timezone-aware UTC")
    series = series.sort_index()
    if (series.index != series.index.floor("h")).any():
        raise ValueError("timestamps must be on the hour; aggregate with to_hourly first")
    valid = series.dropna()
    if valid.empty:
        raise ValueError("series has no values")
    grid = pd.date_range(valid.index.min(), valid.index.max(), freq="h", tz="UTC")
    return series.reindex(grid).rename("load_mw")


# --------------------------------------------------------------------------- snapshot


def write_snapshot(series: pd.Series, cfg: Config, record: dict) -> dict:
    cfg.snapshot.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "timestamp_utc": series.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "load_mw": series.round(1).to_numpy(),
        }
    )
    frame.to_csv(cfg.snapshot, index=False, lineterminator="\n", na_rep="")
    digest = sha256(cfg.snapshot)
    manifest = {
        "source": "Energy-Charts public_power API",
        "endpoint": cfg.endpoint,
        "country": cfg.country,
        "series": cfg.series,
        "attribution": cfg.attribution,
        "licence": "CC BY 4.0",
        "timestamps": "start of each hour, UTC",
        "unit": "MW",
        "rows": int(series.size),
        "missing_hours": int(series.isna().sum()),
        "first_hour": series.index.min().isoformat(),
        "last_hour": series.index.max().isoformat(),
        "sha256": digest,
        **record,
    }
    cfg.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_snapshot(cfg: Config, verify: bool = True) -> pd.Series:
    if not cfg.snapshot.exists():
        raise FileNotFoundError(f"{cfg.snapshot} does not exist; run `postcutoff fetch` first")
    if verify and cfg.manifest.exists():
        expected = json.loads(cfg.manifest.read_text(encoding="utf-8")).get("sha256")
        actual = sha256(cfg.snapshot)
        if expected and expected != actual:
            raise ValueError(f"snapshot hash {actual} does not match the manifest ({expected})")
    frame = pd.read_csv(cfg.snapshot)
    index = pd.to_datetime(frame["timestamp_utc"], utc=True)
    series = pd.Series(frame["load_mw"].to_numpy(dtype=float), index=index, name="load_mw")
    if series.index.has_duplicates:
        raise ValueError("snapshot has duplicate timestamps")
    expected_grid = pd.date_range(series.index.min(), series.index.max(), freq="h", tz="UTC")
    if (series.index != series.index.floor("h")).any() or not series.index.equals(expected_grid):
        raise ValueError("snapshot is not on a complete hourly grid")
    return series


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- preparation


@dataclass
class Prepared:
    """The series as the models see it, and which hours may be scored.

    ``inputs`` has masked and missing hours replaced; ``observed`` is the raw series;
    ``scorable`` marks hours that are neither masked nor missing. (Anomalous hours that are
    not masked stop the run before this point.)
    """

    index: pd.DatetimeIndex
    observed: np.ndarray
    inputs: np.ndarray
    scorable: np.ndarray
    masked: np.ndarray
    missing: np.ndarray
    imputed: np.ndarray


def mask_hours(index: pd.DatetimeIndex, cfg: Config) -> np.ndarray:
    out = np.zeros(index.size, dtype=bool)
    for m in cfg.masks:
        out |= (index >= m.start) & (index < m.end)
    return out


def prepare(series: pd.Series, cfg: Config) -> Prepared:
    index = series.index
    observed = series.to_numpy(dtype=float)
    masked = mask_hours(index, cfg)
    missing = np.isnan(observed)
    replace = masked | missing
    inputs = observed.copy()
    inputs[replace] = np.nan
    imputed = np.zeros(index.size, dtype=bool)
    # Fill each hole with the same hour one week earlier, walking further back when needed;
    # this keeps the daily and weekly shape that every model relies on. Values are only ever
    # copied from earlier hours (including earlier filled ones), never from later ones.
    for lag in (168, 336, 504, 24, 48):
        holes = np.flatnonzero(np.isnan(inputs))
        if holes.size == 0:
            break
        src = holes - lag
        ok = src >= 0
        holes, src = holes[ok], src[ok]
        fill = inputs[src]
        good = ~np.isnan(fill)
        inputs[holes[good]] = fill[good]
        imputed[holes[good]] = True
    if np.isnan(inputs).any():
        # A hole with no earlier week or day to copy from (near the start of the series).
        # Filling it would need later values, so stop instead.
        holes = index[np.isnan(inputs)]
        raise ValueError(
            f"{holes.size} hours cannot be filled from earlier weeks (first {holes[0]}, last {holes[-1]}); "
            "shorten the snapshot or add a mask and investigate the gap"
        )
    scorable = ~(masked | missing)
    return Prepared(index, observed, inputs, scorable, masked, missing, imputed)


def anomalies(series: pd.Series, cfg: Config) -> pd.DataFrame:
    """Hours far from the median of the same hour in the previous four weeks."""
    import warnings

    y = series.to_numpy(dtype=float)
    lags = np.stack([np.roll(y, 168 * k) for k in range(1, 5)])
    for k in range(4):
        lags[k, : 168 * (k + 1)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN at the very start
        ref = np.nanmedian(lags, axis=0)
    ratio = y / ref
    flag = (ratio < cfg.anomaly_low) | (ratio > cfg.anomaly_high)
    flag &= ~np.isnan(ratio)
    masked = mask_hours(series.index, cfg)
    out = pd.DataFrame(
        {
            "timestamp_utc": series.index[flag],
            "load_mw": y[flag],
            "reference_mw": ref[flag],
            "ratio": ratio[flag],
            "masked": masked[flag],
        }
    )
    return out
