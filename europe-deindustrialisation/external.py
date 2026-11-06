"""External balances and the manufacturing share, by member state.

    python external.py

- goods trade balance, % of GDP: (exports - imports of goods) / GDP, Eurostat national accounts, 1995-2025;
- current-account balance, % of GDP: AMECO, 1960-2025;
- net international investment position, % of GDP: Eurostat (tipsii10), from 1995 where available.

Two questions. (1) Did member states whose real manufacturing share fell run weaker external
balances, and did those that kept or grew it run stronger ones? (2) How far apart did the
current accounts of the first euro members drift, and when? Writes ``results/external_*.csv``
and ``results/external_summary.txt``. Correlations are descriptive: they say which countries
moved together, not why.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RES = HERE / "results"

EU27 = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"]
EURO12 = ["AT", "BE", "DE", "EL", "ES", "FI", "FR", "IE", "IT", "LU", "NL", "PT"]
AMECO_ISO = {
    "aut": "AT", "bel": "BE", "bgr": "BG", "cyp": "CY", "cze": "CZ", "deu": "DE", "dnk": "DK", "esp": "ES", "est": "EE", "fin": "FI",
    "fra": "FR", "grc": "EL", "hrv": "HR", "hun": "HU", "irl": "IE", "ita": "IT", "ltu": "LT", "lux": "LU", "lva": "LV", "mlt": "MT",
    "nld": "NL", "pol": "PL", "prt": "PT", "rom": "RO", "svk": "SK", "svn": "SI", "swe": "SE", "gbr": "UK", "eu27": "EU27",
}
OUTLIERS = ["IE", "LU", "MT", "CY"]  # multinational accounting and very small economies


def goods_balance() -> pd.DataFrame:
    d = pd.read_csv(DATA / "gdp_trade.csv")
    p = d.pivot_table(index=["geo", "time"], columns="na_item", values="value").reset_index()
    p["goods_balance"] = 100 * (p.P61 - p.P71) / p.B1GQ
    p["services_balance"] = 100 * (p.P62 - p.P72) / p.B1GQ
    p = p.rename(columns={"time": "year"})
    p["geo"] = p.geo.replace({"EU27_2020": "EU27"})
    return p[["geo", "year", "goods_balance", "services_balance"]]


def current_account() -> pd.DataFrame:
    a = pd.read_csv(DATA / "ameco_long.csv", low_memory=False)
    a = a[a.dataset == "UBCA"]
    dims = a.dimensions.map(json.loads)
    a["geo"] = dims.map(lambda d: AMECO_ISO.get(d.get("geo")))
    a["unit"] = dims.map(lambda d: d.get("unit"))
    a["year"] = pd.to_numeric(a.period.astype(str).str[:4], errors="coerce")
    a["value"] = pd.to_numeric(a.value, errors="coerce")
    a = a[a.geo.notna() & (a.unit == "percentage-of-gross-domestic-product-at-current-prices") & (a.year <= 2025)]
    return a.groupby(["geo", "year"]).value.mean().rename("current_account").reset_index()


def niip() -> pd.DataFrame:
    d = pd.read_csv(DATA / "niip.csv")
    d = d[d.partner == "WRL_REST"].rename(columns={"time": "year", "value": "niip"})
    d["geo"] = d.geo.replace({"EU27_2020": "EU27"})
    return d[["geo", "year", "niip"]]


def main() -> None:
    RES.mkdir(exist_ok=True)
    gb, ca, ni = goods_balance(), current_account(), niip()
    panel = gb.merge(ca, on=["geo", "year"], how="outer").merge(ni, on=["geo", "year"], how="outer").sort_values(["geo", "year"])
    panel.to_csv(RES / "external_panel.csv", index=False, float_format="%.6g")

    # dispersion of current accounts among the first euro members
    disp = ca[ca.geo.isin(EURO12)].groupby("year").current_account.agg(["std", "min", "max", "count"]).reset_index()
    disp = disp[disp["count"] == len(EURO12)]
    disp.to_csv(RES / "external_dispersion.csv", index=False, float_format="%.6g")

    # cross-section against the 2000-2024 decomposition
    dec = pd.read_csv(RES / "decomposition.csv")
    dec = dec[(dec.start == 2000) & dec.geo.isin(EU27)][["geo", "dlog_real_share", "dlog_emp_share", "dlog_nom_share"]]
    win = panel[(panel.year >= 2000) & (panel.year <= 2024)]
    agg = win.groupby("geo").agg(goods_balance_mean=("goods_balance", "mean"), current_account_mean=("current_account", "mean")).reset_index()
    first_last = panel[panel.year.isin([2000, 2007, 2024])].pivot_table(index="geo", columns="year", values=["goods_balance", "current_account", "niip"])
    first_last.columns = [f"{a}_{b}" for a, b in first_last.columns]
    x = dec.merge(agg, on="geo").merge(first_last.reset_index(), on="geo", how="left")
    x["d_goods_balance"] = x.goods_balance_2024 - x.goods_balance_2000
    x.to_csv(RES / "external_cross_section.csv", index=False, float_format="%.6g")

    L = []
    say = L.append

    def corr(col: str, data: pd.DataFrame, label: str) -> None:
        d = data[["dlog_real_share", col]].dropna()
        r, pr = stats.pearsonr(d.dlog_real_share, d[col])
        rho, ps = stats.spearmanr(d.dlog_real_share, d[col])
        slope = np.polyfit(d.dlog_real_share, d[col], 1)[0]
        say(f"  {label}: n={len(d)}, Pearson {r:+.2f} (p={pr:.3f}), Spearman {rho:+.2f} (p={ps:.3f}), slope {slope / 10:+.2f} points per 0.10 log points of real share")

    say("Change in the real manufacturing share, 2000-2024 (log points), against external balances")
    for label, data in (("all member states", x), ("without IE, LU, MT, CY", x[~x.geo.isin(OUTLIERS)])):
        say(f" {label}")
        corr("goods_balance_mean", data, "mean goods balance 2000-2024 (% GDP)")
        corr("d_goods_balance", data, "change in goods balance 2000-2024 (points)")
        corr("current_account_mean", data, "mean current account 2000-2024 (% GDP)")
        corr("niip_2024", data, "net international investment position 2024 (% GDP)")
    say("\nBy member state: real-share change 2000-2024, mean goods balance and current account 2000-2024, NIIP 2024 (% GDP)")
    for _, r in x.sort_values("dlog_real_share").iterrows():
        say(f"  {r.geo}: real share {r.dlog_real_share:+.2f}; goods balance {r.goods_balance_mean:+.1f}; current account {r.current_account_mean:+.1f}; NIIP 2024 {r.niip_2024:+.0f}")
    say("\nCurrent accounts of the 12 first euro members (% GDP): cross-country standard deviation, minimum and maximum")
    for _, r in disp[disp.year.isin([1992, 1995, 1999, 2003, 2007, 2008, 2012, 2016, 2020, 2024])].iterrows():
        say(f"  {int(r.year)}: sd {r['std']:.1f}, min {r['min']:+.1f}, max {r['max']:+.1f}")
    peak = disp.loc[disp["std"].idxmax()]
    say(f"  largest dispersion in {int(peak.year)}: sd {peak['std']:.1f}")
    for g in ["DE", "NL", "ES", "PT", "EL", "IT", "FR"]:
        s = ca[ca.geo == g].set_index("year").current_account
        say(f"  {g}: current account 1999 {s.get(1999, np.nan):+.1f}, 2007 {s.get(2007, np.nan):+.1f}, 2013 {s.get(2013, np.nan):+.1f}, 2024 {s.get(2024, np.nan):+.1f}")
    eu = panel[panel.geo == "EU27"].set_index("year")
    say(f"\nEU27 goods balance (% GDP): 2000 {eu.goods_balance.get(2000, np.nan):+.1f}, 2008 {eu.goods_balance.get(2008, np.nan):+.1f}, 2019 {eu.goods_balance.get(2019, np.nan):+.1f}, 2022 {eu.goods_balance.get(2022, np.nan):+.1f}, 2024 {eu.goods_balance.get(2024, np.nan):+.1f}")
    text = "\n".join(L) + "\n"
    (RES / "external_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
