"""Has the turn back to industry shown up in the data, and what would the 20% target take?

    python turn.py

1. Manufacturing output (industrial production index, NACE C, 2021 = 100), investment
   (gross fixed capital formation by manufacturing, 2020 prices) and employment, 2015-2025.
2. Manufacturing's share of GDP for the EU-27 (the measure used by the Commission's 20%
   target), from Eurostat national accounts.
3. The arithmetic of the target: raising the share from its 2024 level to 20% by 2035. The
   nominal share grows at the rate of the real share plus the rate of change of manufacturing's
   relative price, so the required real growth gap between manufacturing and the whole economy
   depends on what relative prices do. The required pace is compared with the fastest 11-year
   increases in the real manufacturing share that any member state achieved since 1995.

Writes ``results/turn_*.csv`` and ``results/turn_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RES = HERE / "results"

EU27 = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"]
TARGET, TARGET_YEAR, BASE_YEAR = 0.20, 2035, 2024
WINDOW = TARGET_YEAR - BASE_YEAR


def required_growth(s0: float, target: float, years: int) -> float:
    """Average log growth a year that takes a share from s0 to target in the given number of years."""
    return float(np.log(target / s0) / years)


def pivot(name: str, **filters) -> pd.DataFrame:
    d = pd.read_csv(DATA / f"{name}.csv")
    for k, v in filters.items():
        d = d[d[k] == v]
    d["geo"] = d.geo.replace({"EU27_2020": "EU27"})
    return d.pivot_table(index="time", columns="geo", values="value")


def main() -> None:
    RES.mkdir(exist_ok=True)
    L = []
    say = L.append

    ip = pivot("industrial_production")
    gf_c = pivot("gfcf_by_industry", nace_r2="C")
    gf_t = pivot("gfcf_by_industry", nace_r2="TOTAL")
    emp_c = pivot("employment_a64", nace_r2="C")
    gva_c = pivot("gva_a64_current", nace_r2="C")
    gdp = pd.read_csv(DATA / "gdp_trade.csv")
    gdp = gdp[gdp.na_item == "B1GQ"].assign(geo=lambda d: d.geo.replace({"EU27_2020": "EU27"})).pivot_table(index="time", columns="geo", values="value")
    share_gdp = gva_c / gdp

    rows = []
    for g in ["EU27"] + EU27:
        def ch(s: pd.DataFrame, a: int, b: int) -> float:
            try:
                return float(s.loc[b, g] / s.loc[a, g] - 1)
            except KeyError:
                return np.nan

        rows.append(
            {
                "geo": g,
                "ip_2018_2025": ch(ip, 2018, 2025),
                "ip_2021_2025": ch(ip, 2021, 2025),
                "gfcf_manuf_2019_2024": ch(gf_c, 2019, 2024),
                "gfcf_manuf_share_2019": float(gf_c.loc[2019, g] / gf_t.loc[2019, g]) if g in gf_c and g in gf_t and 2019 in gf_c.index else np.nan,
                "gfcf_manuf_share_2024": float(gf_c.loc[2024, g] / gf_t.loc[2024, g]) if g in gf_c and g in gf_t and 2024 in gf_c.index else np.nan,
                "emp_manuf_2019_2024": ch(emp_c, 2019, 2024),
                "manuf_share_gdp_2000": float(share_gdp.loc[2000, g]) if g in share_gdp and 2000 in share_gdp.index else np.nan,
                "manuf_share_gdp_2019": float(share_gdp.loc[2019, g]) if g in share_gdp and 2019 in share_gdp.index else np.nan,
                "manuf_share_gdp_2024": float(share_gdp.loc[2024, g]) if g in share_gdp and 2024 in share_gdp.index else np.nan,
            }
        )
    T = pd.DataFrame(rows)
    T.to_csv(RES / "turn_by_country.csv", index=False, float_format="%.6g")
    ip.to_csv(RES / "turn_industrial_production.csv", float_format="%.6g")

    eu = T[T.geo == "EU27"].iloc[0]
    say("EU-27 manufacturing since the late 2010s")
    say(f"  industrial production (C): 2018-2025 {eu.ip_2018_2025:+.1%}, 2021-2025 {eu.ip_2021_2025:+.1%}")
    say(f"  manufacturing investment (2020 prices): 2019-2024 {eu.gfcf_manuf_2019_2024:+.1%}; share of all investment 2019 {eu.gfcf_manuf_share_2019:.1%}, 2024 {eu.gfcf_manuf_share_2024:.1%}")
    say(f"  manufacturing employment: 2019-2024 {eu.emp_manuf_2019_2024:+.1%}")
    say(f"  manufacturing value added / GDP (current prices): 2000 {eu.manuf_share_gdp_2000:.1%}, 2019 {eu.manuf_share_gdp_2019:.1%}, 2024 {eu.manuf_share_gdp_2024:.1%}")
    say("\nIndustrial production, 2018-2025 and 2021-2025, by member state (sorted by 2018-2025)")
    for _, r in T[T.geo != "EU27"].sort_values("ip_2018_2025").iterrows():
        say(f"  {r.geo}: {r.ip_2018_2025:+.1%} (2021-25 {r.ip_2021_2025:+.1%}); investment 2019-24 {r.gfcf_manuf_2019_2024:+.1%}; jobs 2019-24 {r.emp_manuf_2019_2024:+.1%}")
    pos = T[(T.geo != "EU27") & T.ip_2021_2025.notna()]
    say(f"  member states with higher output in 2025 than in 2021: {int((pos.ip_2021_2025 > 0).sum())} of {len(pos)}")

    # the arithmetic of the target
    s0 = float(eu.manuf_share_gdp_2024)
    need = required_growth(s0, TARGET, WINDOW)
    sh = pd.read_csv(RES / "shares_long.csv")
    e = sh[sh.geo == "EU27"].set_index("year")
    hist_price = float(np.log(e.loc[2024, "rel_price"] / e.loc[2000, "rel_price"]) / 24)
    hist_real = float(np.log(e.loc[2024, "real_share"] / e.loc[2000, "real_share"]) / 24)
    say(f"\nThe 20% target: share of GDP {s0:.1%} in {BASE_YEAR} to {TARGET:.0%} in {TARGET_YEAR} ({WINDOW} years)")
    say(f"  required growth of the nominal share: {need:.2%} a year (log)")
    say(f"  EU-27 2000-2024: real share {hist_real:+.2%} a year, relative price {hist_price:+.2%} a year")
    for label, price in (("relative prices stop falling", 0.0), ("relative prices keep falling at the 2000-2024 pace", hist_price)):
        say(f"  if {label}: the real share must grow {need - price:.2%} a year faster than it did, i.e. manufacturing output about {need - price:.1%} a year faster than the whole economy")
    # the fastest 11-year increases in the real share, any member state, windows within 1995-2024
    best = []
    for g, d in sh[sh.geo.isin(EU27)].groupby("geo"):
        s = d.set_index("year").real_share.dropna()
        s = s[(s.index >= 1995) & (s.index <= 2024)]
        for a in s.index:
            b = a + WINDOW
            if b in s.index:
                best.append({"geo": g, "start": int(a), "end": int(b), "growth": float(np.log(s[b] / s[a]) / WINDOW)})
    B = pd.DataFrame(best)
    B.to_csv(RES / "turn_windows.csv", index=False, float_format="%.6g")
    top = B.sort_values("growth", ascending=False).drop_duplicates("geo").head(8)
    say(f"  fastest {WINDOW}-year growth of the real manufacturing share, by member state, 1995-2024:")
    for _, r in top.iterrows():
        say(f"    {r.geo} {r.start}-{r.end}: {r.growth:+.2%} a year")
    say(f"  member-state windows reaching {need:.2%} a year: {int((B.growth >= need).sum())} of {len(B)}; reaching {need - hist_price:.2%}: {int((B.growth >= need - hist_price).sum())}")
    big = ["DE", "FR", "IT", "ES", "NL", "PL"]
    say("  fastest window among the six largest economies: " + "; ".join(f"{g} {B[B.geo == g].growth.max():+.2%}" for g in big if g in set(B.geo)))
    text = "\n".join(L) + "\n"
    (RES / "turn_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
