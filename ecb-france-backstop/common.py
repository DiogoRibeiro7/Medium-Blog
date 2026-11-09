"""Shared loaders for the ECB backstop and France analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

COUNTRIES = ("DE", "FR", "IT", "ES", "BE", "AT", "NL", "FI", "PT", "IE", "EL")
# The spread panel: euro-area sovereigns priced against Germany. Greece is left out of the model
# (a default in 2012 and programmes from 2010 to 2018); it enters only as a robustness check.
PANEL = ("FR", "IT", "ES", "BE", "AT", "NL", "FI", "PT", "IE")
NAMES = {
    "DE": "Germany", "FR": "France", "IT": "Italy", "ES": "Spain", "BE": "Belgium", "AT": "Austria",
    "NL": "Netherlands", "FI": "Finland", "PT": "Portugal", "IE": "Ireland", "EL": "Greece",
}
ECB_TO_EU = {"GR": "EL"}


def write(name: str, lines: list[str]) -> str:
    RESULTS.mkdir(exist_ok=True)
    text = "\n".join(lines) + "\n"
    (RESULTS / name).write_text(text, encoding="utf-8")
    return text


def read(name: str, path: Path = DATA) -> pd.DataFrame:
    for p in (path / name, path / f"{name}.gz"):
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(name)


def monthly_yields(path: Path = DATA) -> pd.DataFrame:
    """Monthly 10-year yields (percent), one column per country; ECB copy fills Eurostat gaps."""
    e = read("yields_monthly.csv", path)
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


def monthly_spreads(path: Path = DATA) -> pd.DataFrame:
    """Monthly 10-year spreads to Germany (percentage points)."""
    y = monthly_yields(path)
    return y.drop(columns="DE").sub(y["DE"], axis=0)


def _quarter_index(s: pd.Series) -> pd.PeriodIndex:
    return pd.PeriodIndex(s.str.replace("-", ""), freq="Q")


def quarterly_panel(path: Path = DATA) -> pd.DataFrame:
    """Quarterly fundamentals and spreads, one row per country-quarter.

    Columns:
      debt      Maastricht debt, % of GDP over the last four quarters
      balance   net lending (B.9) over four quarters, % of GDP
      interest  interest payable (D.41) over four quarters, % of GDP
      growth    real GDP growth on the same quarter a year earlier, %
      ca        current account balance, % of GDP, four-quarter average
      spread    quarterly average 10-year spread to Germany, pp
      bund      quarterly average German 10-year yield, %
    and the same fundamentals relative to Germany with suffix ``_gap``.
    """
    g = read("gov_quarterly.csv", path)
    if "s_adj" in g:
        g = g[g["s_adj"] == "NSA"] if (g["s_adj"] == "NSA").any() else g
    g = g.pivot_table(index=["geo", "time"], columns="na_item", values="value").rename(columns={"D41PAY": "int_eur", "B9": "bal_eur"})
    d = read("debt_quarterly.csv", path).pivot_table(index=["geo", "time"], values="value")["value"].rename("debt_eur")
    y = read("gdp_quarterly.csv", path)
    if "s_adj" in y:
        y = y[y["s_adj"] == "NSA"] if (y["s_adj"] == "NSA").any() else y
    y = y.pivot_table(index=["geo", "time"], values="value")["value"].rename("gdp_eur")
    r = read("real_growth_quarterly.csv", path)
    if "s_adj" in r:
        pref = [s for s in ("SCA", "NSA", "CA", "SA") if (r["s_adj"] == s).any()]
        r = r[r["s_adj"] == pref[0]]
    r = r.pivot_table(index=["geo", "time"], values="value")["value"].rename("growth")
    x = pd.concat([g, d, y, r], axis=1).reset_index()
    x["quarter"] = _quarter_index(x["time"])
    x = x.sort_values(["geo", "quarter"])
    roll = lambda c: x.groupby("geo")[c].transform(lambda s: s.rolling(4, min_periods=4).sum())  # noqa: E731
    gdp4 = roll("gdp_eur")
    x["debt"] = 100 * x["debt_eur"] / gdp4
    x["balance"] = 100 * roll("bal_eur") / gdp4
    x["interest"] = 100 * roll("int_eur") / gdp4
    ca_file = [p for p in (path / "current_account.csv", path / "current_account.csv.gz") if p.exists()]
    if ca_file:
        c = pd.read_csv(ca_file[0])
        if "s_adj" in c and (c["s_adj"] == "NSA").any():
            c = c[c["s_adj"] == "NSA"]
        c = c.pivot_table(index=["geo", "time"], values="value")["value"].rename("ca_q").reset_index()
        c["quarter"] = _quarter_index(c["time"])
        x = x.merge(c[["geo", "quarter", "ca_q"]], on=["geo", "quarter"], how="left")
        x["ca"] = x.groupby("geo")["ca_q"].transform(lambda s: s.rolling(4, min_periods=4).mean())
    else:
        x["ca"] = np.nan
    m = monthly_yields(path)
    q = m.groupby(m.index.asfreq("Q")).mean()
    n = m.groupby(m.index.asfreq("Q")).count()
    q = q.where(n >= 2)  # a quarter needs at least two observed months
    sp = q.drop(columns="DE").sub(q["DE"], axis=0)
    long = sp.stack().rename("spread").reset_index()
    long.columns = ["quarter", "geo", "spread"]
    bund = q["DE"].rename("bund").reset_index().rename(columns={"index": "quarter"})
    bund.columns = ["quarter", "bund"]
    x = x.merge(long, on=["geo", "quarter"], how="outer").merge(bund, on="quarter", how="left")
    de = x[x["geo"] == "DE"].set_index("quarter")[["debt", "balance", "growth", "ca", "interest"]]
    for c in ("debt", "balance", "growth", "ca", "interest"):
        x[f"{c}_gap"] = x[c] - x["quarter"].map(de[c])
    keep = ["geo", "quarter", "debt", "balance", "interest", "growth", "ca", "spread", "bund",
            "debt_gap", "balance_gap", "growth_gap", "ca_gap", "interest_gap", "gdp_eur"]
    return x[keep].sort_values(["geo", "quarter"]).reset_index(drop=True)


def annual(path: Path = DATA) -> pd.DataFrame:
    """Annual government finance panel (interest, balance, debt, GDP, implicit rate, yields)."""
    g = read("gov_main.csv", path).pivot_table(index=["geo", "time"], columns="na_item", values="value")
    debt = read("debt.csv", path).pivot_table(index=["geo", "time"], values="value")["value"].rename("debt")
    gdp = read("gdp.csv", path).pivot_table(index=["geo", "time"], values="value")["value"].rename("gdp")
    x = g.rename(columns={"D41PAY": "interest", "B9": "balance"})[["interest", "balance"]].join(debt).join(gdp).reset_index()
    x = x.rename(columns={"time": "year"}).sort_values(["geo", "year"])
    x["avg_debt"] = x.groupby("geo")["debt"].transform(lambda s: (s + s.shift()) / 2)
    x["implicit"] = 100 * x["interest"] / x["avg_debt"]
    x["int_gdp"] = 100 * x["interest"] / x["gdp"]
    x["debt_gdp"] = 100 * x["debt"] / x["gdp"]
    x["primary"] = 100 * (x["balance"] + x["interest"]) / x["gdp"]
    x["g"] = 100 * x.groupby("geo")["gdp"].pct_change()
    m = monthly_yields(path)
    full = m.groupby(m.index.year).apply(lambda f: f.mean() if len(f) == 12 else f.mean() * np.nan)
    ya = full.stack().rename("yield").reset_index()
    ya.columns = ["year", "geo", "yield"]
    return x.merge(ya, on=["geo", "year"], how="left")


def quarterly_fiscal(path: Path = DATA) -> pd.DataFrame:
    """Quarterly interest payable and GDP for France's first-half checks (EUR m, NSA)."""
    g = read("gov_quarterly.csv", path)
    if "s_adj" in g and (g["s_adj"] == "NSA").any():
        g = g[g["s_adj"] == "NSA"]
    g = g.pivot_table(index=["geo", "time"], columns="na_item", values="value").rename(columns={"D41PAY": "interest", "B9": "balance"})
    y = read("gdp_quarterly.csv", path)
    if "s_adj" in y and (y["s_adj"] == "NSA").any():
        y = y[y["s_adj"] == "NSA"]
    y = y.pivot_table(index=["geo", "time"], values="value")["value"].rename("gdp")
    q = g.join(y).reset_index().rename(columns={"time": "quarter"})
    q["quarter"] = q["quarter"].str.replace("-", "-")
    return q
