"""Shared loaders for the France 2026 / euro crisis 2011 comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

COUNTRIES = ("DE", "FR", "IT", "ES", "BE", "AT", "NL", "FI", "PT", "IE", "EL")
NAMES = {
    "DE": "Germany", "FR": "France", "IT": "Italy", "ES": "Spain", "BE": "Belgium", "AT": "Austria",
    "NL": "Netherlands", "FI": "Finland", "PT": "Portugal", "IE": "Ireland", "EL": "Greece",
}
ECB_TO_EU = {"GR": "EL"}


def monthly_yields(path: Path = DATA) -> pd.DataFrame:
    """Monthly 10-year yields (percent), one column per country, indexed by month.

    Eurostat's copy is the base; the ECB's copy of the same convergence-criterion series fills
    months that Eurostat has not yet published.
    """
    e = pd.read_csv(path / "yields_monthly.csv")
    e = e.pivot_table(index="time", columns="geo", values="value")
    e.index = pd.PeriodIndex(e.index, freq="M")
    ecb_file = path / "ecb_yields_monthly.csv"
    if ecb_file.exists():
        b = pd.read_csv(ecb_file)
        b["geo"] = b["REF_AREA"].replace(ECB_TO_EU)
        b["OBS_VALUE"] = pd.to_numeric(b["OBS_VALUE"], errors="coerce")
        b = b.pivot_table(index="TIME_PERIOD", columns="geo", values="OBS_VALUE")
        b.index = pd.PeriodIndex(b.index, freq="M")
        e = e.combine_first(b)
    return e.sort_index()[[c for c in COUNTRIES if c in e.columns]]


def daily_yields(path: Path = DATA) -> pd.DataFrame:
    """Daily 10-year benchmark yields (percent), one column per country, business days."""
    d = pd.read_csv(path / "yields_daily.csv.gz", parse_dates=["date"])
    d = d.pivot_table(index="date", columns="geo", values="value").sort_index()
    return d[[c for c in COUNTRIES if c in d.columns]]


def annual(path: Path = DATA) -> pd.DataFrame:
    """Annual government finance panel, one row per country-year.

    Columns: interest (D.41 payable, EUR m), balance (B.9), debt (Maastricht), gdp (current
    prices), avg_debt, implicit (interest over average debt, percent), int_gdp, debt_gdp,
    primary (primary balance, percent of GDP), balance_gdp, g (nominal GDP growth, percent),
    yield (annual average 10-year yield, percent).
    """
    g = pd.read_csv(path / "gov_main.csv").pivot_table(index=["geo", "time"], columns="na_item", values="value")
    debt = pd.read_csv(path / "debt.csv").pivot_table(index=["geo", "time"], values="value")["value"].rename("debt")
    gdp = pd.read_csv(path / "gdp.csv").pivot_table(index=["geo", "time"], values="value")["value"].rename("gdp")
    x = g.rename(columns={"D41PAY": "interest", "B9": "balance"})[["interest", "balance"]].join(debt).join(gdp).reset_index()
    x = x.rename(columns={"time": "year"}).sort_values(["geo", "year"])
    x["avg_debt"] = x.groupby("geo")["debt"].transform(lambda s: (s + s.shift()) / 2)
    x["implicit"] = 100 * x["interest"] / x["avg_debt"]
    x["int_gdp"] = 100 * x["interest"] / x["gdp"]
    x["debt_gdp"] = 100 * x["debt"] / x["gdp"]
    x["primary"] = 100 * (x["balance"] + x["interest"]) / x["gdp"]
    x["balance_gdp"] = 100 * x["balance"] / x["gdp"]
    x["g"] = 100 * x.groupby("geo")["gdp"].pct_change()
    m = monthly_yields(path)
    full = m.groupby(m.index.year).apply(lambda f: f.mean() if len(f) == 12 else f.mean() * np.nan)
    ya = full.stack().rename("yield").reset_index()
    ya.columns = ["year", "geo", "yield"]
    return x.merge(ya, on=["geo", "year"], how="left")


def quarterly(path: Path = DATA) -> pd.DataFrame:
    """Quarterly interest payable, balance, debt and GDP (EUR m, not seasonally adjusted)."""
    q = pd.read_csv(path / "gov_quarterly.csv").pivot_table(index=["geo", "time"], columns="na_item", values="value")
    q = q.rename(columns={"D41PAY": "interest", "B9": "balance"})
    debt = pd.read_csv(path / "debt_quarterly.csv").pivot_table(index=["geo", "time"], values="value")["value"].rename("debt")
    gdp = pd.read_csv(path / "gdp_quarterly.csv").pivot_table(index=["geo", "time"], values="value")["value"].rename("gdp")
    return q.join(debt).join(gdp).reset_index().rename(columns={"time": "quarter"})


def write(name: str, lines: list[str]) -> str:
    RESULTS.mkdir(exist_ok=True)
    text = "\n".join(lines) + "\n"
    (RESULTS / name).write_text(text, encoding="utf-8")
    return text
