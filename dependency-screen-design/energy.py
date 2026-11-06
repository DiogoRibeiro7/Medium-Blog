"""Natural gas as a known case: what the concentration index showed and what exposure shows, 1990-2024.

    python energy.py

From Eurostat's gas import and gas balance tables (snapshot in ../europe-deindustrialisation/data):
the Herfindahl index of the EU-27's extra-EU gas suppliers (by energy content), the largest
supplier's share of extra-EU imports, and the largest supplier's share of EU gas consumption
(gross inland consumption), with Russia's own share of consumption.

Writes ``results/energy_gas.csv`` and ``results/energy_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "europe-deindustrialisation" / "data"
RES = HERE / "results"
EU27 = {"AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"}


def gas() -> pd.DataFrame:
    g = pd.read_csv(SRC / "gas_imports_by_partner.csv", keep_default_na=False, na_values=[""])
    g["value"] = pd.to_numeric(g.value, errors="coerce")
    g = g[(g.geo == "EU27_2020") & ~g.partner.isin(EU27 | {"TOTAL", "NSP"}) & (g.value > 0)]
    # residual groupings such as "other Africa" are kept in the total but are not single suppliers
    single = g[g.partner.str.len() == 2]
    tot = g.groupby("time").value.sum()
    sh = single.set_index(["time", "partner"]).value / tot.reindex(single.time).to_numpy()
    b = pd.read_csv(SRC / "gas_supply.csv")
    bal = b[b.geo == "EU27_2020"].pivot_table(index="time", columns="nrg_bal", values="value")
    b = bal["IC_CAL"]
    prod = bal["IPRD"]
    rows = []
    for y, s in sh.groupby(level=0):
        s = s.droplevel(0).sort_values(ascending=False)
        top = s.index[0]
        imports_top = float(g[(g.time == y) & (g.partner == top)].value.sum())
        imports_ru = float(g[(g.time == y) & (g.partner == "RU")].value.sum())
        rows.append(
            {
                "year": int(y),
                "hhi": float((s**2).sum()),
                "top_partner": top,
                "top_share_of_imports": float(s.iloc[0]),
                "top_share_of_consumption": imports_top / float(b.get(y, float("nan"))),
                "russia_share_of_consumption": imports_ru / float(b.get(y, float("nan"))),
                "top_share_of_supply": imports_top / (float(prod.get(y, float("nan"))) + float(tot[y])),
                "russia_share_of_supply": imports_ru / (float(prod.get(y, float("nan"))) + float(tot[y])),
                "second_partner": s.index[1],
                "second_share_of_imports": float(s.iloc[1]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RES.mkdir(exist_ok=True)
    d = gas()
    d.to_csv(RES / "energy_gas.csv", index=False, float_format="%.6g")
    L = ["Natural gas, EU-27: concentration of extra-EU suppliers and exposure to the largest one"]
    for _, r in d[d.year.isin([1990, 1995, 2000, 2005, 2010, 2014, 2019, 2021, 2022, 2023, 2024])].iterrows():
        L.append(
            f"  {int(r.year)}: HHI {r.hhi:.2f}; largest {r.top_partner} {r.top_share_of_imports:.0%} of extra-EU imports, {r.top_share_of_consumption:.0%} of consumption; "
            f"{r.top_share_of_supply:.0%} of EU supply (production + extra-EU imports); second {r.second_partner} {r.second_share_of_imports:.0%}; Russia {r.russia_share_of_consumption:.0%} of consumption"
        )
    above = d[d.hhi > 0.4].year
    L.append(f"  years with HHI > 0.4: {', '.join(str(y) for y in above)}" if len(above) else "  HHI never above 0.4")
    L.append(f"  years in which the largest supplier had more than 25% of consumption: {', '.join(str(y) for y in d[d.top_share_of_consumption > 0.25].year)}")
    L.append(f"  years in which the largest supplier had more than 25% of EU supply: {', '.join(str(y) for y in d[d.top_share_of_supply > 0.25].year)}")
    (RES / "energy_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
