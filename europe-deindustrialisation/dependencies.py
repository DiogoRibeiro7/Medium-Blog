"""How concentrated are the EU's imports, and on whom does it depend? (Comext and energy balances)

    python dependencies.py

For every HS2 chapter and selected HS4 headings, imports into the EU-27 from outside the EU are
split by supplier country. For each product and year:

- HHI: the Herfindahl-Hirschman index of supplier shares (0 to 1; above 0.25 is usually read as
  highly concentrated),
- the share of China and the share and identity of the largest supplier.

Manufactured goods are HS chapters 28 to 96 excluding 71 (precious stones and metals) and 93
(arms); chapter 27 (mineral fuels) is reported separately. Energy: the share of Russia in the
EU's natural gas imports and the EU's import dependency by fuel. Writes
``results/dependencies_*.csv`` and ``results/dependencies_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RES = HERE / "results"

EU27 = {"AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "GR", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"}
MANUFACTURED = {f"{i:02d}" for i in range(28, 97)} - {"71", "93"}
YEARS = (2002, 2008, 2015, 2019, 2024)
STRATEGIC = {
    "2504": "natural graphite",
    "2805": "alkali, alkaline-earth and rare-earth metals",
    "2846": "rare-earth compounds",
    "2941": "antibiotics",
    "3004": "medicaments in measured doses",
    "8471": "computers",
    "8501": "electric motors and generators",
    "8504": "transformers and converters",
    "8505": "permanent magnets",
    "8507": "electric accumulators (batteries)",
    "8517": "telephone sets and network equipment",
    "8541": "semiconductor devices, incl. photovoltaic cells",
    "8542": "integrated circuits",
    "8703": "passenger cars",
    "8708": "parts of motor vehicles",
}


def load() -> pd.DataFrame:
    c = pd.read_csv(DATA / "comext_imports.csv", dtype={"product": str, "partner": str}, keep_default_na=False, na_values=[""], low_memory=False)
    c = c.rename(columns={"time": "year"})
    c["value"] = pd.to_numeric(c.value, errors="coerce")
    c = c[c.value.notna() & (c.value > 0)]
    extra = c[(c.partner.str.len() == 2) & ~c.partner.isin(EU27) & ~c.partner.str.startswith("Q")]
    return extra


def concentration(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (prod, year), g in c.groupby(["product", "year"]):
        tot = g.value.sum()
        sh = g.value / tot
        top = g.loc[g.value.idxmax()]
        rows.append(
            {
                "product": prod,
                "year": int(year),
                "imports_eur": tot,
                "hhi": float((sh**2).sum()),
                "china_share": float(g.loc[g.partner == "CN", "value"].sum() / tot),
                "top_partner": top.partner,
                "top_share": float(top.value / tot),
                "suppliers": int((sh >= 0.001).sum()),
            }
        )
    return pd.DataFrame(rows)


def energy() -> tuple[pd.DataFrame, pd.DataFrame]:
    g = pd.read_csv(DATA / "gas_imports_by_partner.csv", keep_default_na=False, na_values=[""])
    g["value"] = pd.to_numeric(g.value, errors="coerce")
    # extra-EU imports: every partner except EU member states, the total and "not specified"
    extra = g[~g.partner.isin(EU27 | {"TOTAL", "NSP"})]
    tot = extra.groupby("time").value.sum()
    shares = pd.DataFrame({p: g[g.partner == p].set_index("time").value / tot for p in ("RU", "NO", "DZ", "US", "QA", "UK")})
    shares.index.name = "year"
    d = pd.read_csv(DATA / "energy_dependency.csv")
    dep = d[d.geo == "EU27_2020"].pivot_table(index="time", columns="siec", values="value")[["TOTAL", "G3000", "O4000XBIO", "C0000X0350-0370"]]
    dep.columns = ["all_energy", "natural_gas", "oil", "solid_fossil"]
    dep.index.name = "year"
    return shares, dep


def partner_shares(name: str) -> pd.DataFrame:
    """Shares of extra-EU imports by partner, and the absolute imports, from an nrg_ti_* table."""
    g = pd.read_csv(DATA / f"{name}.csv", keep_default_na=False, na_values=[""])
    g["value"] = pd.to_numeric(g.value, errors="coerce")
    g = g[g.geo == "EU27_2020"]
    extra = g[~g.partner.isin(EU27 | {"TOTAL", "NSP"})]
    tot = extra.groupby("time").value.sum()
    ru = g[g.partner == "RU"].set_index("time").value
    return pd.DataFrame({"extra_imports": tot, "russia": ru, "russia_share": ru / tot})


def russia_energy() -> pd.DataFrame:
    """Russia's share of extra-EU imports of natural gas, crude oil and hard coal, and of EU gas use."""
    gas = partner_shares("gas_imports_by_partner")
    oil = partner_shares("oil_imports_by_partner")
    coal = partner_shares("coal_imports_by_partner")
    b = pd.read_csv(DATA / "gas_supply.csv")
    b = b[b.geo == "EU27_2020"].pivot_table(index="time", columns="nrg_bal", values="value")
    d = pd.DataFrame(
        {
            "gas_russia_share_of_extra_imports": gas.russia_share,
            "gas_russia_share_of_consumption": gas.russia / b["IC_CAL"],
            "gas_eu_production_share_of_consumption": b["IPRD"] / b["IC_CAL"],
            "oil_russia_share_of_extra_imports": oil.russia_share,
            "coal_russia_share_of_extra_imports": coal.russia_share,
        }
    )
    d.index.name = "year"
    return d[(d.index >= 1990) & (d.index <= 2024)]


def main() -> None:
    RES.mkdir(exist_ok=True)
    c = load()
    conc = concentration(c)
    conc.to_csv(RES / "dependencies_concentration.csv", index=False, float_format="%.6g")

    ch = conc[conc["product"].str.len() == 2]
    manu = ch[ch["product"].isin(MANUFACTURED)]
    agg = manu.groupby("year").apply(
        lambda g: pd.Series(
            {
                "imports_eur": g.imports_eur.sum(),
                "hhi_weighted": np.average(g.hhi, weights=g.imports_eur),
                "china_share": np.average(g.china_share, weights=g.imports_eur),
                "chapters_hhi_above_025": int((g.hhi > 0.25).sum()),
                "chapters_china_top": int((g.top_partner == "CN").sum()),
                "chapters": len(g),
            }
        ),
        include_groups=False,
    ).reset_index()
    agg.to_csv(RES / "dependencies_manufactured.csv", index=False, float_format="%.6g")
    gas, dep = energy()
    gas.to_csv(RES / "dependencies_gas_partners.csv", float_format="%.6g")
    dep.to_csv(RES / "dependencies_energy.csv", float_format="%.6g")

    L = []
    say = L.append
    say("EU-27 imports of manufactured goods from outside the EU (HS 28-96 without 71 and 93)")
    for _, r in agg[agg.year.isin(YEARS + (2025,))].iterrows():
        say(
            f"  {int(r.year)}: EUR {r.imports_eur / 1e9:,.0f} bn; China share {r.china_share:.3f}; import-weighted HHI across chapters {r.hhi_weighted:.3f}; "
            f"chapters with HHI > 0.25: {int(r.chapters_hhi_above_025)} of {int(r.chapters)}; chapters where China is the largest supplier: {int(r.chapters_china_top)}"
        )
    say("\nSelected headings: China share, HHI and largest supplier")
    for hs, name in STRATEGIC.items():
        g = conc[conc["product"] == hs].set_index("year")
        parts = []
        for y in YEARS:
            if y in g.index:
                r = g.loc[y]
                parts.append(f"{y} China {r.china_share:.2f}, HHI {r.hhi:.2f}, top {r.top_partner} {r.top_share:.2f}")
        say(f"  {hs} {name}: " + "; ".join(parts))
    say("\nChapters with the highest China share in 2024 (imports above EUR 5 bn)")
    top = ch[(ch.year == 2024) & (ch.imports_eur > 5e9)].sort_values("china_share", ascending=False).head(10)
    for _, r in top.iterrows():
        say(f"  HS {r['product']}: China {r.china_share:.2f}, HHI {r.hhi:.2f}, imports EUR {r.imports_eur / 1e9:,.0f} bn")
    say("\nNatural gas imports into the EU-27 from outside the EU, by partner (share, gross calorific value)")
    for y in (1990, 2000, 2010, 2019, 2021, 2022, 2023, 2024):
        if y in gas.index:
            r = gas.loc[y]
            say(f"  {y}: Russia {r.RU:.2f}, Norway {r.NO:.2f}, Algeria {r.DZ:.2f}, United States {r.US:.2f}, Qatar {r.QA:.2f}")
    say("\nEU-27 energy import dependency (% of gross available energy met by net imports)")
    for y in (2000, 2010, 2019, 2021, 2022, 2023, 2024):
        if y in dep.index:
            r = dep.loc[y]
            say(f"  {y}: all energy {r.all_energy:.1f}, natural gas {r.natural_gas:.1f}, oil {r.oil:.1f}, solid fossil fuels {r.solid_fossil:.1f}")
    ru = russia_energy()
    ru.to_csv(RES / "dependencies_russia_energy.csv", float_format="%.6g")
    say("\nRussia's share of EU-27 energy supply (gas in TJ GCV, crude oil and hard coal in tonnes)")
    for y in (1990, 2000, 2005, 2010, 2014, 2019, 2021, 2022, 2023, 2024):
        if y in ru.index:
            r = ru.loc[y]
            say(
                f"  {y}: gas, share of extra-EU imports {r.gas_russia_share_of_extra_imports:.2f}, of EU consumption {r.gas_russia_share_of_consumption:.2f} "
                f"(EU production {r.gas_eu_production_share_of_consumption:.2f} of consumption); crude oil {r.oil_russia_share_of_extra_imports:.2f}; hard coal {r.coal_russia_share_of_extra_imports:.2f}"
            )
    text = "\n".join(L) + "\n"
    (RES / "dependencies_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
