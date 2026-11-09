"""Shared loaders for the Portugal debt-management analysis."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"


def write(name: str, lines: list[str]) -> str:
    RESULTS.mkdir(exist_ok=True)
    text = "\n".join(lines) + "\n"
    (RESULTS / name).write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------- BPstat

def _bpstat_page(path: str) -> pd.DataFrame:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    ids, sizes = d["id"], d["size"]
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    pos = {}
    for dim in ids:
        index = d["dimension"][dim]["category"]["index"]
        if isinstance(index, list):
            pos[dim] = {str(c): k for k, c in enumerate(index)}
        else:
            pos[dim] = {str(c): k for c, k in index.items()}
    dates_index = d["dimension"]["reference_date"]["category"]["index"]
    dates = dates_index if isinstance(dates_index, list) else sorted(dates_index, key=dates_index.get)
    values = d["value"]
    get = (lambda k: values.get(str(k))) if isinstance(values, dict) else (lambda k: values[k] if k < len(values) else None)
    t_stride = strides[ids.index("reference_date")]
    rows = []
    for s in d.get("extension", {}).get("series", []):
        base = 0
        ok = True
        for dc in s["dimension_category"]:
            dim, cat = str(dc["dimension_id"]), str(dc["category_id"])
            if dim not in pos or cat not in pos[dim]:
                ok = False
                break
            base += pos[dim][cat] * strides[ids.index(dim)]
        if not ok:
            continue
        for t, date in enumerate(dates):
            v = get(base + t * t_stride)
            if v is not None:
                rows.append((s["id"], s["label"], date, float(v)))
    return pd.DataFrame(rows, columns=["series", "label", "date", "value"])


def bpstat(path: Path = DATA) -> pd.DataFrame:
    """Every BPstat series in the snapshot, long format: series id, label, date, value."""
    frames = [_bpstat_page(p) for p in sorted(glob.glob(str(path / "bpstat" / "*.json")))]
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["series", "date"])
    out["date"] = pd.to_datetime(out["date"])
    return out


def bp_series(ids: dict[int, str], path: Path = DATA) -> pd.DataFrame:
    """Selected BPstat series as columns, indexed by date."""
    b = bpstat(path)
    b = b[b["series"].isin(ids)]
    w = b.pivot_table(index="date", columns="series", values="value")
    return w.rename(columns=ids)


# ---------------------------------------------------------------------------- ECB, Eurostat, FRED

def ecb(name: str, path: Path = DATA) -> pd.DataFrame:
    d = pd.read_csv(path / f"{name}.csv", dtype={"TIME_PERIOD": str})
    d["OBS_VALUE"] = pd.to_numeric(d["OBS_VALUE"], errors="coerce")
    return d


def euribor(tenor: str = "3m", path: Path = DATA) -> pd.Series:
    d = ecb(f"euribor_{tenor}", path)
    s = d.set_index("TIME_PERIOD")["OBS_VALUE"]
    s.index = pd.PeriodIndex(s.index, freq="M")
    return s.sort_index()


def yields_10y(path: Path = DATA) -> pd.DataFrame:
    d = pd.read_csv(path / "yields_monthly.csv")
    w = d.pivot_table(index="time", columns="geo", values="value")
    w.index = pd.PeriodIndex(w.index, freq="M")
    return w.sort_index()


def fred(path: Path = DATA) -> pd.DataFrame:
    d = pd.read_csv(path / "fred_rates.csv")
    d["date"] = pd.PeriodIndex(pd.to_datetime(d["date"]), freq="M")
    return d.pivot_table(index="date", columns="series", values="value").sort_index()


# ---------------------------------------------------------------------------- SDR

# currency amounts in the SDR basket (Rule O-1), by valuation period
SDR_BASKETS = [
    ("2011-01", {"USD": 0.660, "EUR": 0.423, "JPY": 12.1, "GBP": 0.111}),
    ("2016-10", {"USD": 0.58252, "EUR": 0.38671, "CNY": 1.0174, "JPY": 11.900, "GBP": 0.085946}),
    ("2022-08", {"USD": 0.57813, "EUR": 0.37379, "CNY": 1.0993, "JPY": 13.452, "GBP": 0.08087}),
]
SDR_FLOOR = 0.05


def sdr(path: Path = DATA) -> pd.DataFrame:
    """Monthly SDR interest rate and euro value of the SDR, rebuilt from the basket.

    Exchange rates are ECB monthly averages. The three-month rates are the US Treasury bill
    (FRED TB3MS), the euro-area AAA government three-month spot rate (ECB yield curve, monthly mean
    of daily values) and OECD three-month interbank rates for the yen, pound and renminbi (FRED),
    carried forward when a series ends. The IMF uses Treasury-bill rates for the yen and pound and
    a three-month government benchmark for the renminbi, so those legs are approximations; they
    carry about a quarter of the basket's weight.
    """
    x = ecb("exr_monthly", path)
    x["ccy"] = x["KEY"].str.split(".").str[2]
    fx = x.pivot_table(index="TIME_PERIOD", columns="ccy", values="OBS_VALUE")  # currency per euro
    fx.index = pd.PeriodIndex(fx.index, freq="M")
    c = ecb("aaa_curve_daily", path)
    c = c[c["KEY"].str.endswith("SR_3M")].copy()
    c["m"] = pd.PeriodIndex(pd.to_datetime(c["TIME_PERIOD"]), freq="M")
    eur3m = c.groupby("m")["OBS_VALUE"].mean()
    f = fred(path)
    rates = pd.DataFrame({
        "USD": f["TB3MS"],
        "EUR": eur3m,
        "JPY": f["IR3TIB01JPM156N"],
        "GBP": f["IR3TIB01GBM156N"],
        "CNY": f["IR3TIB01CNM156N"],
    })
    idx = pd.period_range("2010-01", fx.index.max(), freq="M")
    rates = rates.reindex(idx).ffill()
    fx = fx.reindex(idx).ffill()
    usd_per = pd.DataFrame({"USD": 1.0, "EUR": fx["USD"], "JPY": fx["USD"] / fx["JPY"],
                            "GBP": fx["USD"] / fx["GBP"], "CNY": fx["USD"] / fx["CNY"]}, index=idx)
    rows = []
    for m in idx:
        basket = [b for start, b in SDR_BASKETS if m >= pd.Period(start, "M")]
        basket = basket[-1] if basket else SDR_BASKETS[0][1]
        value_usd = sum(a * usd_per.loc[m, k] for k, a in basket.items())
        w = {k: a * usd_per.loc[m, k] / value_usd for k, a in basket.items()}
        r = max(SDR_FLOOR, sum(w[k] * rates.loc[m, k] for k in basket))
        rows.append(dict(month=m, sdr_rate=r, eur_per_sdr=value_usd / usd_per.loc[m, "EUR"], usd_per_sdr=value_usd,
                         **{f"w_{k}": w.get(k, np.nan) for k in ("USD", "EUR", "CNY", "JPY", "GBP")}))
    return pd.DataFrame(rows).set_index("month")


# ---------------------------------------------------------------------------- Portuguese curve

PT_CURVE_SERIES = {12099465: 2.0, 12099460: 3.0, 12099461: 4.0, 12099462: 5.0, 12099463: 7.0, 12099464: 10.0}


def pt_curve(path: Path = DATA) -> pd.DataFrame:
    """Monthly Portuguese government yields by residual maturity (years as columns).

    BPstat fixed-rate Treasury bond yields at 2, 3, 4, 5, 7 and 10 years (monthly series), with
    twelve-month and three-month Euribor standing in for the 1-year and 3-month points.
    """
    b = bp_series({k: str(v) for k, v in PT_CURVE_SERIES.items()}, path)
    b.index = pd.PeriodIndex(b.index, freq="M")
    b.columns = [float(c) for c in b.columns]
    c = b.join(euribor("12m", path).rename(1.0), how="outer").join(euribor("3m", path).rename(0.25), how="outer")
    c = c[sorted(c.columns)].sort_index()
    return c.ffill(limit=2)


def pt_yield(curve: pd.DataFrame, month: pd.Period, years: float) -> float:
    """Linear interpolation on the curve; flat beyond its ends."""
    row = curve.loc[month].dropna()
    x, y = np.array(row.index, float), row.to_numpy(float)
    return float(np.interp(min(max(years, x.min()), x.max()), x, y))
