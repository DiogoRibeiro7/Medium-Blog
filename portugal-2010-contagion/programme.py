"""Programme terms and the ECB's 2011 hikes, as Portugal experienced them.

    python programme.py

1. Growth forecasts against outturns. Real GDP growth projected when each programme was designed
   (European Commission, Occasional Papers 61, May 2010, for Greece and 79, June 2011, for
   Portugal) against Eurostat's current figures, and the cumulative level gap over the programme
   years.
2. Lending terms. The European share of Portugal's programme (52 billion euro: 26 from the EFSM,
   26 from the EFSF) at the rates simulated in Occasional Paper 79 from end-April 2011 market
   rates (about 5.5% EFSM, 6.1% EFSF; 7.5 years average maturity) against the terms of the
   21 July 2011 euro summit (EFSF rates equal to the balance-of-payments facility, about 3.5%;
   maturities of at least 15 and up to 30 years), applied to the full 52 billion. A full-amount,
   one-year illustration, not the interest Portugal paid: tranches were disbursed over 2011-2014
   and each priced on its own funding.
3. The 2011 hikes. The ECB's main refinancing rate, three-month Euribor and the rate on new
   house-purchase loans in Portugal and Germany. Portuguese housing loans are overwhelmingly
   indexed to Euribor (Banco de Portugal still put the share at about 98% in 2023), so the extra
   interest on the stock of Portuguese housing loans is approximated as the stock times the rise in
   three-month Euribor (an upper bound: loans reset every three to twelve months).

Writes ``results/programme_*.csv`` and ``results/programme_summary.txt``.
"""

from __future__ import annotations

import pandas as pd

from common import DATA, NAMES, RESULTS, annual_panel, write

FORECASTS = {
    # programme projections of real GDP growth, %
    "EL": {"source": "Occasional Paper 61, May 2010", "years": {2010: -4.0, 2011: -2.6, 2012: 1.1, 2013: 2.1}},
    "PT": {"source": "Occasional Paper 79, June 2011", "years": {2011: -2.2, 2012: -1.8, 2013: 1.2, 2014: 2.5}},
}
EU_LOANS_BN = {"EFSM": 26.0, "EFSF": 26.0}
RATES_APRIL_2011 = {"EFSM": 5.5, "EFSF": 6.1}
RATE_JULY_2011 = 3.5
MATURITY_BEFORE, MATURITY_AFTER = 7.5, (15, 30)
EURIBOR_FROM, EURIBOR_TO = "2010-12", "2011-07"


def ecb_monthly(name: str) -> pd.Series:
    d = pd.read_csv(DATA / f"{name}.csv", dtype={"TIME_PERIOD": str})
    s = d.set_index("TIME_PERIOD")["OBS_VALUE"].astype(float)
    s.index = pd.PeriodIndex(s.index, freq="M")
    return s


def forecasts() -> tuple[pd.DataFrame, list[str]]:
    a = annual_panel()
    rows, lines = [], ["Programme growth projections against outturns (real GDP, %):"]
    for g, f in FORECASTS.items():
        lvl_f = lvl_a = 1.0
        parts = []
        for y, fc in f["years"].items():
            out = float(a[(a["geo"] == g) & (a["year"] == y)]["growth"].iloc[0])
            lvl_f *= 1 + fc / 100
            lvl_a *= 1 + out / 100
            rows.append(dict(geo=g, year=y, forecast=fc, outturn=out, source=f["source"]))
            parts.append(f"{y} {fc:+.1f} vs {out:+.1f}")
        ys = list(f["years"])
        lines.append(f"  {NAMES[g]} ({f['source']}): " + "; ".join(parts) +
                     f"; cumulative {ys[0]}-{ys[-1]}: projected {100 * (lvl_f - 1):+.1f}%, outturn {100 * (lvl_a - 1):+.1f}%, level gap {100 * (lvl_a / lvl_f - 1):+.1f}%")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "programme_forecasts.csv", index=False, float_format="%.2f")
    return df, lines


def loan_terms() -> list[str]:
    total = sum(EU_LOANS_BN.values())
    before = sum(EU_LOANS_BN[k] * RATES_APRIL_2011[k] / 100 for k in EU_LOANS_BN)
    after = total * RATE_JULY_2011 / 100
    avg = 100 * before / total
    return ["Lending terms on the European share of Portugal's programme (illustration on the full amount):",
            f"  {total:.0f} billion euro (EFSM {EU_LOANS_BN['EFSM']:.0f}, EFSF {EU_LOANS_BN['EFSF']:.0f}); April 2011 simulated rates EFSM {RATES_APRIL_2011['EFSM']}%, EFSF {RATES_APRIL_2011['EFSF']}% "
            f"(average {avg:.2f}%), average maturity {MATURITY_BEFORE} years",
            f"  21 July 2011 terms: about {RATE_JULY_2011}% and {MATURITY_AFTER[0]}-{MATURITY_AFTER[1]} years",
            f"  Annual interest on the full amount: {before:.2f} billion at April terms, {after:.2f} at July terms; difference {before - after:.2f} billion a year "
            f"({avg - RATE_JULY_2011:.2f} pp)"]


def hikes() -> tuple[pd.DataFrame, list[str]]:
    mro = pd.read_csv(DATA / "mro_rate.csv", dtype={"TIME_PERIOD": str})[["TIME_PERIOD", "OBS_VALUE"]]
    eur = ecb_monthly("euribor_3m")
    stock = ecb_monthly("housing_loans_pt")
    lr = pd.read_csv(DATA / "loan_rates.csv", dtype={"TIME_PERIOD": str})
    house = lr[lr["KEY"].str.contains(r"\.A2C\.") & lr["KEY"].str.endswith(".N")]
    house = house.pivot(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE").rename(columns={"GR": "EL"})
    house.index = pd.PeriodIndex(house.index, freq="M")
    lines = ["The ECB's 2011 hikes:"]
    m11 = mro[(mro["TIME_PERIOD"] >= "2011-01-01") & (mro["TIME_PERIOD"] <= "2012-12-31")]
    lines.append("  Main refinancing rate changes (effective dates): " + ", ".join(f"{r.TIME_PERIOD} {r.OBS_VALUE:.2f}%" for r in m11.itertuples()))
    lines.append(f"  Three-month Euribor: {EURIBOR_FROM} {eur[EURIBOR_FROM]:.2f}%, {EURIBOR_TO} {eur[EURIBOR_TO]:.2f}% (+{eur[EURIBOR_TO] - eur[EURIBOR_FROM]:.2f} pp); peak {eur.loc['2011'].max():.2f}% in {eur.loc['2011'].idxmax()}; "
                 f"2010 average {eur.loc['2010'].mean():.2f}%, 2011 average {eur.loc['2011'].mean():.2f}%, 2012 average {eur.loc['2012'].mean():.2f}%")
    st = stock[EURIBOR_FROM] / 1000
    extra_peak = st * (eur[EURIBOR_TO] - eur[EURIBOR_FROM]) / 100
    extra_year = st * (eur.loc["2011"].mean() - eur.loc["2010"].mean()) / 100
    lines.append(f"  Portuguese housing loans outstanding, {EURIBOR_FROM}: {st:.1f} billion euro; extra annual interest at the July 2011 Euribor: about {extra_peak:.2f} billion; "
                 f"2011 against 2010 on annual averages: about {extra_year:.2f} billion")
    for g in ("PT", "ES", "IE", "IT", "DE"):  # Greece: not published from November 2010
        h = house[g]
        lines.append(f"  New house-purchase loan rate, {NAMES[g]}: Dec 2010 {h['2010-12']:.2f}%, Jul 2011 {h['2011-07']:.2f}% ({h['2011-07'] - h['2010-12']:+.2f}), "
                     f"Dec 2011 {h['2011-12']:.2f}% ({h['2011-12'] - h['2010-12']:+.2f})")
    out = pd.DataFrame({"euribor_3m": eur}).join(house, how="left").loc["2009-01":"2013-12"]
    out.index.name = "month"
    out.to_csv(RESULTS / "programme_rates.csv", float_format="%.4f")
    return out, lines


def run() -> str:
    _, l1 = forecasts()
    l2 = loan_terms()
    _, l3 = hikes()
    return write("programme_summary.txt", l1 + [""] + l2 + [""] + l3)


if __name__ == "__main__":
    print(run())
