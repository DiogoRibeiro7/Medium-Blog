"""Turn the pinned source files in .cache/ into the compact tables in data/.

    python fetch_data.py && python build_data.py

The tables are rebuilt from the pinned sources on every run and are not committed.

data/sp500_monthly.csv   S&P Composite, monthly, January 1871 to the last month with earnings
data/us_gdp.csv          United States real GDP (2011 international dollars) and population, 1870-2022
data/world.csv           World real GDP, primary energy (TWh), population and CO2, 1965-2024
data/countries.csv       Countries (not regional aggregates), 1990-2022: GDP, population, primary energy,
                         fossil fuel production (TWh), territorial and consumption-based CO2
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
DATA = HERE / "data"


def build_sp500():
    s = pd.read_csv(CACHE / "sp500.csv", parse_dates=["Date"])
    s = s.rename(
        columns={
            "Date": "date",
            "SP500": "price",
            "Dividend": "dividend",
            "Earnings": "earnings",
            "Consumer Price Index": "cpi",
            "PE10": "cape",
        }
    )[["date", "price", "dividend", "earnings", "cpi", "cape"]]
    last = s.loc[(s.earnings > 0) & (s.cpi > 0), "date"].max()
    s = s[s.date <= last].copy()
    s["cape"] = s["cape"].where(s["cape"] > 0)  # the mirror writes 0 for the first ten years
    s["date"] = s["date"].dt.strftime("%Y-%m")
    return s


def build_us_gdp(co2):
    u = co2[(co2.iso_code == "USA") & (co2.year.between(1870, 2022))]
    return u[["year", "gdp", "population"]].dropna().reset_index(drop=True)


def build_world(co2, energy):
    w_e = energy[energy.country == "World"][["year", "gdp", "primary_energy_consumption", "population"]]
    w_c = co2[co2.country == "World"][["year", "co2"]]
    w = w_e.merge(w_c, on="year", how="left")
    w = w[w.year.between(1965, 2024)].rename(columns={"primary_energy_consumption": "energy_twh"})
    return w.reset_index(drop=True)


def build_countries(co2, energy):
    real = co2.iso_code.notna() & ~co2.iso_code.astype(str).str.startswith("OWID")
    c = co2[real & co2.year.between(1990, 2022)][["iso_code", "country", "year", "gdp", "population", "co2", "consumption_co2"]]
    e = energy[energy.iso_code.notna()][
        ["iso_code", "year", "primary_energy_consumption", "coal_production", "gas_production", "oil_production"]
    ].copy()
    e["fossil_production_twh"] = e[["coal_production", "gas_production", "oil_production"]].sum(axis=1, min_count=1)
    e = e.drop(columns=["coal_production", "gas_production", "oil_production"])
    out = c.merge(e, on=["iso_code", "year"], how="left")
    out = out.rename(columns={"primary_energy_consumption": "energy_twh"})
    return out.sort_values(["iso_code", "year"]).reset_index(drop=True)


def main():
    DATA.mkdir(exist_ok=True)
    co2 = pd.read_csv(CACHE / "owid-co2-data.csv")
    energy = pd.read_csv(CACHE / "owid-energy-data.csv")
    tables = {
        "sp500_monthly.csv": build_sp500(),
        "us_gdp.csv": build_us_gdp(co2),
        "world.csv": build_world(co2, energy),
        "countries.csv": build_countries(co2, energy),
    }
    for name, df in tables.items():
        df.to_csv(DATA / name, index=False, float_format="%.6g")
        print(f"{name}: {len(df)} rows")


if __name__ == "__main__":
    main()
