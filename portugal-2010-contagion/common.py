"""Shared loaders for the Portugal 2010-2012 analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

COUNTRIES = ("DE", "FR", "IT", "ES", "BE", "AT", "NL", "FI", "PT", "IE", "EL")
# The estimation panel: euro-area sovereigns priced against Germany. Greece is not used to fit the
# models (a default in 2012 and programmes from 2010 to 2018); its warranted spread is predicted
# out of sample with the same models.
PANEL = ("FR", "IT", "ES", "BE", "AT", "NL", "FI", "PT", "IE")
PREDICT = PANEL + ("EL",)
TRIO = ("PT", "EL", "IE")
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
        if "freq" in c:
            c = c[c["freq"] == "Q"]
        if "s_adj" in c and (c["s_adj"] == "NSA").any():
            c = c[c["s_adj"] == "NSA"]
        c = c.pivot_table(index=["geo", "time"], values="value")["value"].rename("ca_q").reset_index()
        c["quarter"] = _quarter_index(c["time"])
        x = x.merge(c[["geo", "quarter", "ca_q"]], on=["geo", "quarter"], how="left")
        x["ca"] = x.groupby("geo")["ca_q"].transform(lambda s: s.rolling(4, min_periods=4).mean())
    else:
        x["ca"] = np.nan
    niip_file = [p for p in (path / "niip_annual.csv", path / "niip_annual.csv.gz") if p.exists()]
    if niip_file:
        n = pd.read_csv(niip_file[0])
        if "unit" in n and (n["unit"] == "PC_GDP").any():
            n = n[n["unit"] == "PC_GDP"]
        n = n.pivot_table(index=["geo", "time"], values="value")["value"].reset_index()
        # the end-year position is known only after the year ends: use it from the next year on
        rows = [(g, pd.Period(f"{int(t) + 1}Q{k}", "Q"), v) for g, t, v in n.itertuples(index=False) for k in (1, 2, 3, 4)]
        n = pd.DataFrame(rows, columns=["geo", "quarter", "niip"])
        x = x.merge(n, on=["geo", "quarter"], how="left")
    else:
        x["niip"] = np.nan
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
    de = x[x["geo"] == "DE"].set_index("quarter")[["debt", "balance", "growth", "ca", "interest", "niip"]]
    for c in ("debt", "balance", "growth", "ca", "interest", "niip"):
        x[f"{c}_gap"] = x[c] - x["quarter"].map(de[c])
    keep = ["geo", "quarter", "debt", "balance", "interest", "growth", "ca", "niip", "spread", "bund",
            "debt_gap", "balance_gap", "growth_gap", "ca_gap", "niip_gap", "interest_gap", "gdp_eur"]
    return x[keep].sort_values(["geo", "quarter"]).reset_index(drop=True)


def annual_panel(path: Path = DATA) -> pd.DataFrame:
    """Annual indicators by country and year (current Eurostat vintage).

    deficit and debt in % of GDP (EDP), real GDP growth, net international investment position and
    private-sector debt in % of GDP, unemployment rate.
    """
    out = []
    d = read("deficit_debt_annual.csv", path)
    d = d.pivot_table(index=["geo", "time"], columns="na_item", values="value").rename(columns={"B9": "balance", "GD": "debt"})
    out.append(d)
    g = read("growth_annual.csv", path).pivot_table(index=["geo", "time"], values="value")["value"].rename("growth")
    out.append(g)
    for name, col in (("niip_annual.csv", "niip"), ("private_debt.csv", "private_debt"), ("unemployment.csv", "unemployment")):
        try:
            f = read(name, path)
        except FileNotFoundError:
            continue
        if "unit" in f and (f["unit"] == "PC_GDP").any():
            f = f[f["unit"] == "PC_GDP"]
        if "sector" in f:
            pref = [s for s in ("S11_S14_S15", "S1_S2") if (f["sector"] == s).any()]
            if pref:
                f = f[f["sector"] == pref[0]]
        out.append(f.pivot_table(index=["geo", "time"], values="value")["value"].rename(col))
    a = pd.concat(out, axis=1).reset_index().rename(columns={"time": "year"})
    a["year"] = a["year"].astype(int)
    return a.sort_values(["geo", "year"]).reset_index(drop=True)
