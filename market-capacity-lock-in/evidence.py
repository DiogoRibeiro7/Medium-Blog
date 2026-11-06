"""The Portuguese series behind the article, read from the pinned snapshots.

    python evidence.py

Writes ``results/evidence_summary.txt`` and ``results/evidence_*.csv``.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fetch import jsonstat_to_long

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ARCH = DATA / "archived"
RES = HERE / "results"


def eurostat_json(name: str) -> pd.DataFrame:
    df = jsonstat_to_long(json.loads((ARCH / name).read_text(encoding="utf-8")))
    df["time"] = df["time"].astype(int)
    return df


def ine_series(name: str, geo: str = "PT", dim3: str | None = "T") -> pd.Series:
    rows = []
    with gzip.open(ARCH / name, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["kind"] != "data":
                continue
            for block in json.loads(rec["body"]):
                for year, obs in block["Dados"].items():
                    for o in obs:
                        if o.get("geocod") != geo:
                            continue
                        if dim3 is not None and "dim_3" in o and o["dim_3"] != dim3:
                            continue
                        rows.append((int(year), float(o["valor"])))
    s = pd.Series(dict(rows)).sort_index()
    return s


def farm_age() -> pd.DataFrame:
    f = pd.read_csv(DATA / "farm_managers_age.csv", low_memory=False)
    f = f[(f.statinfo == "TOTAL") & (f.so_eur == "TOTAL") & (f.uaarea == "TOTAL") & (f.unit == "HLD")]
    p = f.pivot_table(index=["geo", "time"], columns="age", values="value")
    p["lt35_share"] = 100 * p[["Y_LT25", "Y25-34"]].sum(axis=1) / p["TOTAL"]
    # the 35-39 band exists from 2016; earlier waves only have 35-44
    lt40 = p[["Y_LT25", "Y25-34", "Y35-39"]].sum(axis=1, min_count=3)
    p["lt40_share"] = 100 * lt40 / p["TOTAL"]
    p["ge55_share"] = 100 * p[["Y55-64", "Y_GE65"]].sum(axis=1) / p["TOTAL"]
    p["ge65_share"] = 100 * p["Y_GE65"] / p["TOTAL"]
    p["managers_per_young"] = p["TOTAL"] / p[["Y_LT25", "Y25-34"]].sum(axis=1)
    return p


def agri_labour() -> pd.DataFrame:
    a = pd.read_csv(DATA / "agri_labour_input.csv")
    a = a.pivot_table(index="time", columns=["geo", "am_item"], values="value")
    out = pd.DataFrame(
        {
            "total": a[("PT", "AM400000")],
            "non_salaried": a[("PT", "AM410000")],
            "salaried": a[("PT", "AM420000")],
        }
    )
    out["salaried_share"] = out["salaried"] / out["total"]
    return out


def agri_output() -> pd.DataFrame:
    e = pd.read_csv(DATA / "agri_accounts_indices.csv", low_memory=False)
    o = e[(e.am_item == "AM180000") & (e.indic_agr == "PRD_BP") & (e.unit == "I15")]
    return o.pivot_table(index="time", columns="p_adj", values="value").rename(
        columns={"VOL": "output_volume", "RP": "output_real_price", "NP": "output_nominal_price"}
    )


def industry_panel() -> pd.DataFrame:
    a = pd.read_csv(ARCH / "nama_10_a10.csv")
    e = pd.read_csv(ARCH / "nama_10_a10_e.csv")
    a, e = a[a.geo == "PT"], e[(e.geo == "PT") & (e.unit == "THS_PER")]
    w = a[a.na_item == "D11"].pivot_table(index="year", columns="nace_r2", values="value")
    emp = e[e.na_item == "EMP_DC"].pivot_table(index="year", columns="nace_r2", values="value")
    sal = e[e.na_item == "SAL_DC"].pivot_table(index="year", columns="nace_r2", values="value")
    wpe = w / sal
    out = pd.DataFrame(
        {
            "emp_agri": emp["A"],
            "emp_constr": emp["F"],
            "emp_total": emp["TOTAL"],
            "sal_agri": sal["A"],
            "self_agri": emp["A"] - sal["A"],
            "wage_rel_agri": wpe["A"] / wpe["TOTAL"],
            "wage_rel_constr": wpe["F"] / wpe["TOTAL"],
            "wage_per_employee": wpe["TOTAL"],
        }
    )
    return out


def food_prices() -> pd.DataFrame:
    h = pd.read_csv(ARCH / "prc_hicp_aind.csv")
    p = h.pivot_table(index="year", columns="coicop", values="value")[["CP00", "CP011"]]
    p["food_relative"] = (p["CP011"] / p["CP00"]) / (p.loc[1996, "CP011"] / p.loc[1996, "CP00"])
    return p


def housing() -> pd.DataFrame:
    comp = ine_series("ine_0000076_2026-10-30.jsonl.gz")
    comp_long = ine_series("ine_0011725_2026-10-30.jsonl.gz", dim3=None)
    lic = ine_series("ine_0013235_2026-10-29.jsonl.gz")
    hpi = eurostat_json("eurostat_prc_hpi_a_PT_2026-10-29.json").set_index("time")["value"]
    permits = eurostat_json("eurostat_sts_cobp_a_PT_dwellings_2026-10-29.json").set_index("time")["value"] * 1000
    cpi = pd.read_csv(DATA / "construction_production.csv")
    cpi["time"] = cpi["time"].astype(int)
    cprod = cpi.set_index("time")["value"]
    ind = industry_panel()
    df = pd.DataFrame(
        {
            "completions": comp,
            "completions_long": comp_long,
            "licensed": lic,
            "permits_eurostat": permits,
            "hpi": hpi,
            "construction_production": cprod,
            "construction_employment": ind["emp_constr"],
            "wage_per_employee": ind["wage_per_employee"],
        }
    ).sort_index()
    w = df["wage_per_employee"] / df.loc[2015, "wage_per_employee"] * 100
    df["price_to_wage"] = df["hpi"] / w * 100
    return df


def overburden() -> pd.DataFrame:
    ob = eurostat_json("eurostat_ilc_lvho07a_PT_2026-10-29.json")
    ob = ob[ob.rskpovth == "TOTAL"].pivot_table(index="time", columns="age", values="value")
    lp = eurostat_json("eurostat_ilc_lvps08_PT_2026-10-29.json").pivot_table(index="time", columns="age", values="value")
    ten = eurostat_json("eurostat_ilc_lvho02_PT_2026-10-29.json")
    ten = ten[(ten.hhcomp == "TOTAL") & (ten.rskpovth == "TOTAL")].pivot_table(index="time", columns="tenure", values="value")
    out = pd.DataFrame(
        {
            "overburden_total": ob["TOTAL"],
            "overburden_25_29": ob["Y25-29"],
            "with_parents_25_34": lp["Y25-34"],
            "owner": ten["OWN"],
            "owner_no_loan": ten["OWN_NL"],
            "reduced_or_free_rent": ten["RENT_FR"],
        }
    )
    return out


def household_growth() -> tuple[float, float]:
    d = eurostat_json("eurostat_lfst_hhnhtych_PT_2026-10-29.json").set_index("time")["value"]
    g1 = float(np.log(d.loc[2019] / d.loc[2006]) / 13.0)
    g2 = float(np.log(d.loc[2025] / d.loc[2019]) / 6.0)
    return g1, g2


def gini() -> pd.Series:
    g = pd.read_csv(DATA / "gini.csv")
    g = g[g.age == "TOTAL"]
    return g.set_index(g.time.astype(int))["value"]


def load_all() -> dict:
    h = housing()
    return {
        "farm_age": farm_age(),
        "agri_labour": agri_labour(),
        "agri_output": agri_output(),
        "industry": industry_panel(),
        "food_prices": food_prices(),
        "housing_obs": h,
        "overburden": overburden(),
        "household_growth": household_growth(),
        "gini": gini(),
    }


def lograte(a: float, b: float, years: float) -> float:
    return float(np.log(b / a) / years)


def main() -> None:
    RES.mkdir(exist_ok=True)
    ev = load_all()
    L: list[str] = []
    say = L.append

    fa = ev["farm_age"]
    say("== Farm managers by age (Eurostat ef_m_farmang, holdings)")
    for geo in ("PT", "EU27_2020", "ES", "IT", "FR", "AT", "PL", "EL"):
        sub = fa.loc[geo]
        y = sub.index.max()
        r = sub.loc[y]
        say(
            f"{geo} {y}: managers {r.TOTAL:,.0f}; under 35 {r.lt35_share:.1f}%; under 40 {r.lt40_share:.1f}%; "
            f"55+ {r.ge55_share:.1f}%; 65+ {r.ge65_share:.1f}%; managers per manager under 35: {r.managers_per_young:.1f}"
        )
    pt = fa.loc["PT"]
    say(f"PT 65+ share: " + ", ".join(f"{y} {v:.1f}%" for y, v in pt.ge65_share.items()))
    say(f"PT under-35 share: " + ", ".join(f"{y} {v:.1f}%" for y, v in pt.lt35_share.items()))

    al = ev["agri_labour"]
    say("\n== Agricultural labour input, Portugal (aact_ali01, thousand annual work units)")
    for y in (1980, 1995, 2010, 2025):
        r = al.loc[y]
        say(f"{y}: total {r.total:.1f}, non-salaried {r.non_salaried:.1f}, salaried {r.salaried:.1f}, salaried share {r.salaried_share:.3f}")
    say(f"change 1980-2025: total {al.loc[2025, 'total'] / al.loc[1980, 'total'] - 1:+.3f}; non-salaried {al.loc[2025, 'non_salaried'] / al.loc[1980, 'non_salaried'] - 1:+.3f}")
    say(f"annual log rate 1980-2025, total: {lograte(al.loc[1980, 'total'], al.loc[2025, 'total'], 45):.4f}")
    say(f"salaried 2010 -> 2025: {al.loc[2010, 'salaried']:.1f} -> {al.loc[2025, 'salaried']:.1f}; non-salaried {al.loc[2010, 'non_salaried']:.1f} -> {al.loc[2025, 'non_salaried']:.1f}")

    ao = ev["agri_output"]
    say("\n== Agricultural output, Portugal (aact_eaa05, output of the agricultural industry, 2015 = 100)")
    for y in (1980, 1995, 2005, 2010, 2015, 2019, 2020, 2022, 2023, 2025):
        r = ao.loc[y]
        say(f"{y}: volume {r.output_volume:.1f}; real price {r.get('output_real_price', np.nan):.1f}")
    lp = (ao.output_volume / al.total)
    say(f"output volume per AWU, 1995 -> 2025: x{lp.loc[2025] / lp.loc[1995]:.3f}; 1980 -> 2025: x{lp.loc[2025] / lp.loc[1980]:.3f}")

    ind = ev["industry"]
    say("\n== Employment and pay by industry, Portugal (nama_10_a10_e, nama_10_a10)")
    for y in (1995, 2002, 2008, 2013, 2014, 2025):
        r = ind.loc[y]
        say(f"{y}: agriculture-forestry-fishing {r.emp_agri:.1f}k (self-employed {r.self_agri:.1f}k); construction {r.emp_constr:.1f}k")
    say(f"construction employment 2008 -> 2013: annual log rate {lograte(ind.loc[2008, 'emp_constr'], ind.loc[2013, 'emp_constr'], 5):.4f}")
    say(f"construction employment 2014 -> 2025: annual log rate {lograte(ind.loc[2014, 'emp_constr'], ind.loc[2025, 'emp_constr'], 11):.4f}")
    say(f"construction employment 2002 peak {ind.emp_constr.max():.1f}k ({int(ind.emp_constr.idxmax())}); trough {ind.emp_constr.loc[2008:].min():.1f}k ({int(ind.emp_constr.loc[2008:].idxmin())}); 2025 {ind.loc[2025, 'emp_constr']:.1f}k")
    for y in (1995, 2005, 2015, 2019, 2024, 2025):
        say(f"  pay per employee relative to the economy, {y}: agriculture {ind.loc[y, 'wage_rel_agri']:.3f}, construction {ind.loc[y, 'wage_rel_constr']:.3f}")

    fp = ev["food_prices"]
    say("\n== Food prices relative to all items (HICP CP011 / CP00, 1996 = 1)")
    for y in (1996, 2005, 2010, 2015, 2019, 2021, 2022, 2023, 2025):
        say(f"{y}: {fp.loc[y, 'food_relative']:.3f}")

    h = ev["housing_obs"]
    say("\n== Housing, Portugal")
    say(f"completions (INE 0000076): peak {h.completions.max():,.0f} in {int(h.completions.idxmax())}; trough {h.completions.loc[2005:].min():,.0f} in {int(h.completions.loc[2005:].idxmin())}; 2025 {h.loc[2025, 'completions']:,.0f}")
    say(f"  trough / peak {h.completions.loc[2005:].min() / h.completions.max():.3f}; 2025 / peak {h.loc[2025, 'completions'] / h.completions.max():.3f}")
    say(f"  mean 1995-2007 {h.completions.loc[1995:2007].mean():,.0f}; 2025 / that mean {h.loc[2025, 'completions'] / h.completions.loc[1995:2007].mean():.3f}")
    say(f"completions long series (INE 0011725): 1975 {h.loc[1975, 'completions_long']:,.0f}; 2025 {h.loc[2025, 'completions_long']:,.0f}")
    say(f"  years 1975-1987 below 2025 level? min {h.completions_long.loc[1975:1987].min():,.0f}")
    say(f"completions growth 2015 -> 2025: annual log rate {lograte(h.loc[2015, 'completions'], h.loc[2025, 'completions'], 10):.4f}; 2002 -> 2015 {lograte(h.loc[2002, 'completions'], h.loc[2015, 'completions'], 13):.4f}")
    say(f"licensed dwellings (INE 0013235): 2013 {h.loc[2013, 'licensed']:,.0f}; 2025 {h.loc[2025, 'licensed']:,.0f}")
    say(f"construction production (sts_copr_a, 2021 = 100): 2005 {h.loc[2005, 'construction_production']:.1f}; 2016 {h.loc[2016, 'construction_production']:.1f}; 2025 {h.loc[2025, 'construction_production']:.1f}")
    say(f"house price index (prc_hpi_a, 2015 = 100): 2008 {h.loc[2008, 'hpi']:.1f}; 2013 {h.loc[2013, 'hpi']:.1f}; 2025 {h.loc[2025, 'hpi']:.1f}")
    say(f"pay per employee 2015 -> 2025: x{h.loc[2025, 'wage_per_employee'] / h.loc[2015, 'wage_per_employee']:.3f}")
    say(f"house prices relative to pay per employee (2015 = 100): 2008 {h.loc[2008, 'price_to_wage']:.1f}; 2013 {h.loc[2013, 'price_to_wage']:.1f}; 2025 {h.loc[2025, 'price_to_wage']:.1f}")
    ob = ev["overburden"]
    for y in (2015, 2019, 2022, 2024, 2025):
        r = ob.loc[y]
        say(
            f"  {y}: housing-cost overburden {r.overburden_total:.1f}% (25-29: {r.overburden_25_29:.1f}%); aged 25-34 living with parents {r.with_parents_25_34:.1f}%; "
            f"population in owner-occupied homes {r.owner:.1f}% (no loan {r.owner_no_loan:.1f}%), reduced or free rent {r.reduced_or_free_rent:.1f}%"
        )
    g = ev["gini"]
    say(f"\nGini (ilc_di12): {int(g.index.max())} {g.iloc[-1]:.1f}")
    hg = ev["household_growth"]
    say(f"household growth (lfst_hhnhtych): 2006-2019 {hg[0]:.4f}/yr; 2019-2025 {hg[1]:.4f}/yr")

    for k in ("agri_labour", "agri_output", "industry", "food_prices", "housing_obs", "overburden"):
        ev[k].to_csv(RES / f"evidence_{k}.csv")
    fa.reset_index().to_csv(RES / "evidence_farm_age.csv", index=False)
    text = "\n".join(L) + "\n"
    (RES / "evidence_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
