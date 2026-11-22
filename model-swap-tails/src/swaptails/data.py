"""Downloading, caching and preparing the series.

Three sources, all token-free:

- load: Energy-Charts `public_power`, series "Load", per country, at the resolution the API
  returns, aggregated to hourly means (MW);
- the grid operator's day-ahead load forecast: Energy-Charts `public_power_forecast`,
  production type "load", forecast type "day-ahead" (the reference forecast);
- temperature forecasts: Open-Meteo Previous Runs API, the value forecast two days earlier,
  averaged over a few large cities per country with equal weights.

Every series is hourly, indexed by the start of each hour in UTC. Nothing is committed: the
cache in data/cache is rebuilt by `swaptails fetch`, and data/manifest.json records every
request and the SHA-256 of every cached file.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, Country

AGENT = "model-swap-tails/0.1 (+https://github.com/DiogoRibeiro7/Medium-Blog)"


# --------------------------------------------------------------------------- parsing


def parse_public_power(payload: dict, series: str) -> pd.Series:
    """One named series from an Energy-Charts `public_power` response (columnar format)."""
    stamps = payload.get("unix_seconds")
    if not stamps:
        raise ValueError("response has no unix_seconds")
    types = payload.get("production_types", [])
    matches = [p for p in types if p.get("name") == series]
    if len(matches) != 1:
        raise ValueError(f"expected one series named {series!r}, found {[p.get('name') for p in types]}")
    values = matches[0].get("data")
    if values is None or len(values) != len(stamps):
        raise ValueError(f"series {series!r} has {len(values or [])} values for {len(stamps)} timestamps")
    index = pd.to_datetime(np.asarray(stamps, dtype="int64"), unit="s", utc=True)
    return pd.Series([np.nan if v is None else float(v) for v in values], index=index, name="value")


def parse_forecast(payload: dict) -> pd.Series:
    """A forecast series from an Energy-Charts `public_power_forecast` response.

    Accepts the columnar format (unix_seconds plus one array of values) and the v2 envelope
    (a `data` list of entries with `timestamp` and `values`).
    """
    stamps = payload.get("unix_seconds")
    if stamps:
        for key in ("forecast_values", "values", "load", "data"):
            values = payload.get(key)
            if isinstance(values, list) and len(values) == len(stamps):
                index = pd.to_datetime(np.asarray(stamps, dtype="int64"), unit="s", utc=True)
                return pd.Series([np.nan if v is None else float(v) for v in values], index=index, name="value")
        raise ValueError(f"no value array matching unix_seconds; keys {sorted(payload)}")
    rows = payload.get("data")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict) and "timestamp" in rows[0]:
        out = {}
        for row in rows:
            v = row.get("values")
            if isinstance(v, list):
                v = v[0] if v else None
            elif isinstance(v, dict):
                v = next(iter(v.values()), None)
            out[pd.Timestamp(row["timestamp"]).tz_convert("UTC")] = np.nan if v is None else float(v)
        return pd.Series(out, name="value").sort_index()
    raise ValueError(f"unrecognised forecast response; keys {sorted(payload)}")


def to_hourly(series: pd.Series) -> tuple[pd.Series, dict]:
    """Hourly means from a series at hourly or finer resolution.

    An hour counts only when it is covered by evenly spaced stamps starting on the hour, all
    present; otherwise it is missing. Values are average power (MW), so the hourly value is
    their mean. Returns the series and the number of hours at each resolution.
    """
    if series.empty:
        raise ValueError("empty series")
    hour = series.index.floor("h")
    minutes = series.index.minute + series.index.second / 60.0
    frame = pd.DataFrame({"hour": hour, "minute": minutes, "value": series.to_numpy()})
    rows, resolution = {}, {}
    for h, g in frame.groupby("hour", sort=True):
        n = len(g)
        expected = list(np.arange(n) * (60.0 / n)) if 60 % n == 0 else None
        regular = expected is not None and np.allclose(np.sort(g["minute"].to_numpy()), expected)
        complete = regular and not g["value"].isna().any()
        rows[h] = float(g["value"].mean()) if complete else np.nan
        key = f"{60 // n} min" if regular else "irregular"
        resolution[key] = resolution.get(key, 0) + 1
    out = pd.Series(list(rows.values()), index=pd.DatetimeIndex(list(rows.keys())), name="value")
    return out, resolution


def regularise(series: pd.Series, first: pd.Timestamp, last: pd.Timestamp) -> pd.Series:
    """Put a series on the complete hourly grid [first, last]; hours not returned are NaN."""
    if series.index.tz is None:
        raise ValueError("index must be timezone-aware UTC")
    series = series[~series.index.duplicated(keep="last")].sort_index()
    grid = pd.date_range(first, last, freq="h", tz="UTC")
    return series.reindex(grid).rename("value")


# --------------------------------------------------------------------------- download


def _get_json(session, url: str, params: dict, attempts: int = 5):
    last = None
    for attempt in range(attempts):
        response = session.get(url, params=params, timeout=180, headers={"User-Agent": AGENT})
        if response.status_code == 200:
            return response.json()
        last = f"HTTP {response.status_code}: {response.text[:200]}"
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(float(response.headers.get("Retry-After", 30 * (attempt + 1))), 300))
            continue
        break
    raise RuntimeError(f"request to {url} with {params} failed: {last}")


def grid_bounds(cfg: Config) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last hour (UTC) of the cached grid: the whole local days first_day..last_day
    in every country, padded by a day."""
    first = pd.Timestamp(cfg.first_day - timedelta(days=1), tz="UTC")
    last = pd.Timestamp(cfg.last_day + timedelta(days=2), tz="UTC") - pd.Timedelta(hours=1)
    return first, last


def fetch_load(cfg: Config, country: Country, session, pause: float) -> tuple[pd.Series, list]:
    first, last = grid_bounds(cfg)
    parts, record = [], []
    for year in range(cfg.first_day.year, cfg.last_day.year + 1):
        start = max(cfg.first_day - timedelta(days=1), date(year, 1, 1))
        end = min(cfg.last_day + timedelta(days=1), date(year, 12, 31))
        params = {"country": country.ec, "start": start.isoformat(), "end": end.isoformat()}
        payload = _get_json(session, cfg.load_endpoint, params)
        raw = parse_public_power(payload, cfg.load_series)
        hourly, resolution = to_hourly(raw)
        parts.append(hourly)
        record.append({"url": cfg.load_endpoint, "params": params, "stamps": int(raw.size), "resolution": resolution})
        time.sleep(pause)
    return regularise(pd.concat(parts), first, last), record


def fetch_tso(cfg: Config, country: Country, session, pause: float) -> tuple[pd.Series | None, list]:
    """The operator's day-ahead load forecast for the forecast period. Missing or unparsable
    responses are recorded and leave the reference out for that country."""
    first, last = grid_bounds(cfg)
    params = {
        "country": country.ec,
        "production_type": "load",
        "forecast_type": "day-ahead",
        "start": (cfg.forecast_first_day - timedelta(days=1)).isoformat(),
        "end": (cfg.forecast_last_day + timedelta(days=1)).isoformat(),
    }
    try:
        payload = _get_json(session, cfg.tso_endpoint, params)
        raw = parse_forecast(payload)
        hourly, resolution = to_hourly(raw)
        record = [{"url": cfg.tso_endpoint, "params": params, "stamps": int(raw.size), "resolution": resolution}]
        series = regularise(hourly, first, last)
    except (RuntimeError, ValueError) as e:
        record = [{"url": cfg.tso_endpoint, "params": params, "error": str(e)}]
        series = None
    time.sleep(pause)
    return series, record


def fetch_weather(cfg: Config, country: Country, session) -> tuple[pd.Series, list]:
    """Equal-weight mean of the cities' temperature forecasts made two days earlier."""
    first, last = grid_bounds(cfg)
    # The Previous Runs API keeps data from January 2024; earlier hours stay missing.
    start = max(cfg.first_day, date(2024, 1, 1))
    params = {
        "latitude": ",".join(f"{c[1]:.2f}" for c in country.cities),
        "longitude": ",".join(f"{c[2]:.2f}" for c in country.cities),
        "hourly": cfg.weather_variable,
        "start_date": start.isoformat(),
        "end_date": (cfg.last_day + timedelta(days=1)).isoformat(),
        "timezone": "GMT",
    }
    payload = _get_json(session, cfg.weather_endpoint, params)
    payload = payload if isinstance(payload, list) else [payload]
    if len(payload) != len(country.cities):
        raise ValueError(f"{country.key}: expected {len(country.cities)} locations, got {len(payload)}")
    columns = []
    for item in payload:
        hourly = item["hourly"]
        index = pd.to_datetime(hourly["time"]).tz_localize("UTC")
        values = [np.nan if v is None else float(v) for v in hourly[cfg.weather_variable]]
        columns.append(pd.Series(values, index=index))
    mean = pd.concat(columns, axis=1).mean(axis=1, skipna=False)
    record = [{"url": cfg.weather_endpoint, "params": params, "locations": len(payload)}]
    return regularise(mean, first, last), record


def cache_path(cfg: Config, kind: str, country: str) -> Path:
    return cfg.cache_dir / f"{kind}_{country}.parquet"


def write_series(path: Path, series: pd.Series) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"timestamp_utc": series.index, "value": series.to_numpy(dtype=float)})
    frame.to_parquet(path, index=False)
    return sha256(path)


def read_series(path: Path) -> pd.Series:
    frame = pd.read_parquet(path)
    index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp_utc"], utc=True))
    return pd.Series(frame["value"].to_numpy(dtype=float), index=index, name="value")


def fetch_all(cfg: Config, countries: list[str], session=None, pause: float | None = None) -> dict:
    import requests

    session = session or requests.Session()
    pause = cfg.pause_seconds if pause is None else pause
    manifest = read_manifest(cfg)
    manifest.setdefault("countries", {})
    for key in countries:
        country = cfg.countries[key]
        entry = {"fetched_at": _now()}
        load, rec = fetch_load(cfg, country, session, pause)
        entry["load"] = {"requests": rec, "sha256": write_series(cache_path(cfg, "load", key), load)}
        tso, rec = fetch_tso(cfg, country, session, pause)
        entry["tso"] = {"requests": rec}
        if tso is not None:
            entry["tso"]["sha256"] = write_series(cache_path(cfg, "tso", key), tso)
        weather, rec = fetch_weather(cfg, country, session)
        entry["weather"] = {"requests": rec, "sha256": write_series(cache_path(cfg, "temp", key), weather)}
        manifest["countries"][key] = entry
        write_manifest(cfg, manifest)
        print(f"{key}: load {int(load.notna().sum())} hours, tso {'yes' if tso is not None else 'no'}")
    manifest["attribution"] = [cfg.attribution_load, cfg.attribution_weather]
    write_manifest(cfg, manifest)
    return manifest


def read_manifest(cfg: Config) -> dict:
    if cfg.manifest.exists():
        return json.loads(cfg.manifest.read_text(encoding="utf-8"))
    return {}


def write_manifest(cfg: Config, manifest: dict) -> None:
    cfg.manifest.parent.mkdir(parents=True, exist_ok=True)
    cfg.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- local days


def local_dates(index: pd.DatetimeIndex, tz: str) -> np.ndarray:
    """The local calendar date of each UTC hour."""
    return np.asarray(index.tz_convert(tz).date)


def daily_mean(series: pd.Series, tz: str) -> pd.Series:
    """Mean of each local day's hours (MW); NaN unless every hour of the day is present.
    A mean rather than a sum, so days with 23 or 25 hours compare with the others."""
    days = local_dates(series.index, tz)
    frame = pd.DataFrame({"day": days, "value": series.to_numpy()})
    g = frame.groupby("day")["value"]
    out = g.mean().where(g.count() == g.size())
    out.index = pd.Index(out.index, name="date")
    return out


# --------------------------------------------------------------------------- preparation


@dataclass
class Prepared:
    """A country's load as the models see it, and which hours may be scored."""

    country: Country
    index: pd.DatetimeIndex
    observed: np.ndarray
    inputs: np.ndarray
    scorable: np.ndarray
    masked: np.ndarray
    temperature: np.ndarray
    tso: np.ndarray | None
    local_date: np.ndarray


def mask_hours(index: pd.DatetimeIndex, cfg: Config, country: str) -> np.ndarray:
    out = np.zeros(index.size, dtype=bool)
    for m in cfg.masks_for(country):
        out |= (index >= m.start) & (index < m.end)
    return out


def fill_from_earlier(values: np.ndarray, lags=(168, 336, 504, 24, 48)) -> tuple[np.ndarray, np.ndarray]:
    """Fill holes with the same hour one week earlier, walking further back when needed.
    Values are only ever copied from earlier hours. Holes that cannot be filled stay NaN."""
    out = values.astype(float).copy()
    filled = np.zeros(out.size, dtype=bool)
    for lag in lags:
        holes = np.flatnonzero(np.isnan(out))
        if holes.size == 0:
            break
        src = holes - lag
        ok = src >= 0
        holes, src = holes[ok], src[ok]
        good = ~np.isnan(out[src])
        out[holes[good]] = out[src[good]]
        filled[holes[good]] = True
    return out, filled


def fill_temperature(values: np.ndarray) -> np.ndarray:
    """Covariate values for hours the weather source does not cover: the same hour a day
    earlier, then a day later, then the series mean (only before the source starts)."""
    out = values.astype(float).copy()
    for lag in (24, 48, 168):
        holes = np.flatnonzero(np.isnan(out))
        src = holes - lag
        ok = src >= 0
        out[holes[ok]] = out[src[ok]]
    holes = np.isnan(out)
    if holes.all():
        return np.zeros_like(out)
    out[holes] = np.nanmean(out)
    return out


def prepare(cfg: Config, key: str, load: pd.Series, temp: pd.Series, tso: pd.Series | None) -> Prepared:
    country = cfg.countries[key]
    index = load.index
    observed = load.to_numpy(dtype=float)
    masked = mask_hours(index, cfg, key)
    raw = observed.copy()
    raw[masked] = np.nan
    inputs, _ = fill_from_earlier(raw)
    scorable = ~(masked | np.isnan(observed))
    temperature = fill_temperature(temp.reindex(index).to_numpy(dtype=float))
    tso_values = None if tso is None else tso.reindex(index).to_numpy(dtype=float)
    return Prepared(
        country, index, observed, inputs, scorable, masked, temperature, tso_values, local_dates(index, country.tz)
    )


def load_prepared(cfg: Config, key: str) -> Prepared:
    load = read_series(cache_path(cfg, "load", key))
    temp_path = cache_path(cfg, "temp", key)
    temp = read_series(temp_path) if temp_path.exists() else pd.Series(np.nan, index=load.index)
    tso_path = cache_path(cfg, "tso", key)
    tso = read_series(tso_path) if tso_path.exists() else None
    return prepare(cfg, key, load, temp, tso)


def anomalies(prep: Prepared, low: float = 0.5, high: float = 2.0) -> pd.DataFrame:
    """Hours far from the median of the same hour in the previous four weeks."""
    import warnings

    y = prep.observed
    lags = np.stack([np.roll(y, 168 * k) for k in range(1, 5)])
    for k in range(4):
        lags[k, : 168 * (k + 1)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanmedian(lags, axis=0)
    ratio = y / ref
    flag = ((ratio < low) | (ratio > high)) & ~np.isnan(ratio)
    return pd.DataFrame(
        {
            "timestamp_utc": prep.index[flag],
            "load_mw": y[flag],
            "reference_mw": ref[flag],
            "masked": prep.masked[flag],
        }
    )
