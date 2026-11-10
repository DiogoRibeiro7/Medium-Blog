"""Measure Portugal's housebuilding sector layer by layer and write every number the article uses.

    python analysis.py

Reads the tidy tables written by ``fetch.py`` in ``data/`` and writes

- ``results/summary.txt``  every number quoted in the article, section by section
- ``results/*.csv``        the series behind each figure

The conjecture being measured has four parts, one section each:

A. skills: the people who build (age, occupation, training pipeline, the price of their time)
B. firms: the contractors and specialised trades that run building sites
C. promoters: the developers and the private individuals who commission new dwellings
D. sellers: the agencies, finders and traders of existing property

Section E puts every layer on one scale, the *recovered share of the loss*:

    R = (last - trough) / (reference - trough)

where the trough is the minimum of the series between the reference year and the last year.
R = 0 means nothing of the loss has come back, R = 1 means the series is back at its
reference level, R > 1 means it has passed it. The reference year is 2008 for every series
(the first year of the firm statistics) and, where the series starts earlier, also 2002 (the
peak of construction employment).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

MONTHS = {
    "Janeiro": 1, "Fevereiro": 2, "Março": 3, "Abril": 4, "Maio": 5, "Junho": 6,
    "Julho": 7, "Agosto": 8, "Setembro": 9, "Outubro": 10, "Novembro": 11, "Dezembro": 12,
}

BUILDERS = ("41100", "41200", "43")
SELLERS = ("68311", "68312")
TRADES = {
    "43210": "electrical installation",
    "43221": "plumbing",
    "43320": "carpentry and joinery",
    "43310": "plastering",
    "43330": "floor and wall covering",
    "43340": "painting and glazing",
    "43910": "roofing",
}
LABELS = {
    "41100": "developers (41.10)",
    "41200": "building contractors (41.20)",
    "43": "specialised trades (43)",
    "68100": "buying and selling own property (68.10)",
    "68200": "renting own property (68.20)",
    "68311": "real estate agencies (68.31.1)",
    "68312": "real estate finders (68.31.2)",
    "71110": "architects (71.11)",
    "71120": "engineering (71.12)",
    **{k: f"{v} ({k[:2]}.{k[2:4]})" for k, v in TRADES.items()},
}


# --------------------------------------------------------------------------- loading

def eurostat(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{name}.csv", dtype={"time": str})
    df["year"] = df["time"].str[:4].astype(int)
    return df


def ine(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv", dtype={"dim_3": str, "dim_4": str, "dim_5": str, "period": str})


def ine_annual(name: str) -> pd.Series:
    """Indicator without an extra dimension (or its total), indexed by year."""
    df = ine(name)
    if "dim_3" in df:
        df = df[df["dim_3"] == "T"]
    return df.set_index(df["period"].astype(int))["value"].sort_index()


def scie(kind: str) -> pd.DataFrame:
    """INE business statistics by CAE subclass: 2008-2022/23 from the NUTS 2013 indicator,
    then the NUTS 2024 indicator. Years in both must agree (checked in the tests)."""
    old = ine(f"ine_{kind}_2008")
    new = ine(f"ine_{kind}_2023")
    a = old.pivot_table(index="period", columns="dim_3", values="value")
    b = new.pivot_table(index="period", columns="dim_3", values="value")
    out = pd.concat([a, b.loc[~b.index.isin(a.index)]]).sort_index()
    out.index = out.index.astype(int)
    return out


def scie_overlap(kind: str) -> pd.DataFrame:
    old = ine(f"ine_{kind}_2008").pivot_table(index="period", columns="dim_3", values="value")
    new = ine(f"ine_{kind}_2023").pivot_table(index="period", columns="dim_3", values="value")
    years = sorted(set(old.index) & set(new.index))
    return pd.concat({"old": old.loc[years], "new": new.loc[years]})


def cost_index_annual() -> pd.DataFrame:
    df = ine("ine_cost_index")
    parts = df["period"].str.split(" de ", expand=True)
    df["year"] = parts[1].astype(int)
    df["month"] = parts[0].map(MONTHS)
    factor = {"T": "total", "1": "materials", "2": "labour"}
    df["factor"] = df["dim_3"].map(factor)
    monthly = df.pivot_table(index=["year", "month"], columns="factor", values="value").sort_index()
    full = monthly.groupby(level=0).size() == 12
    annual = monthly.groupby(level=0).mean()
    annual["complete_year"] = full
    return monthly, annual


# --------------------------------------------------------------------------- helpers

def recovered_share(s: pd.Series, ref_year: int, last_year: int | None = None) -> dict:
    s = s.dropna().sort_index()
    last_year = last_year or int(s.index.max())
    window = s.loc[ref_year:last_year]
    trough_year = int(window.idxmin())
    ref, trough, last = float(s.loc[ref_year]), float(window.min()), float(s.loc[last_year])
    loss = ref - trough
    return {
        "ref_year": ref_year, "ref": ref, "trough_year": trough_year, "trough": trough,
        "last_year": last_year, "last": last,
        "last_over_ref": last / ref,
        # with no fall after the reference year (less than 5% of it) a "recovered share" means nothing
        "R": (last - trough) / loss if loss > 0.05 * ref else np.nan,
    }


def fmt(x: float, nd: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:,.{nd}f}"


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def h(self, title: str) -> None:
        self.lines += ["", f"== {title}"]

    def __call__(self, text: str) -> None:
        self.lines.append(text)

    def r(self, label: str, d: dict, nd: int = 0) -> None:
        self.lines.append(
            f"{label}: {d['ref_year']} {fmt(d['ref'], nd)}; trough {d['trough_year']} {fmt(d['trough'], nd)}; "
            f"{d['last_year']} {fmt(d['last'], nd)}; last/ref {d['last_over_ref']:.3f}; recovered share R {fmt(d['R'], 2)}"
        )

    def write(self, path: Path) -> None:
        path.write_text("\n".join(self.lines).lstrip() + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- sections

def section_a(rep: Report, ratios: list[dict]) -> dict:
    out: dict = {}
    rep.h("A1. Employment in construction, national accounts (nama_10_a64_e, thousand persons)")
    na = eurostat("na_employment").pivot_table(index="year", columns="na_item", values="value")
    for y in (1995, 2002, 2008, 2014, 2019, 2025):
        rep(f"{y}: total {na.loc[y, 'EMP_DC']:.1f}; employees {na.loc[y, 'SAL_DC']:.1f}; self-employed {na.loc[y, 'SELF_DC']:.1f}")
    peak_year = int(na["EMP_DC"].idxmax())
    rep(f"peak {peak_year} {na['EMP_DC'].max():.1f}")
    for ref in (2002, 2008):
        d = recovered_share(na["EMP_DC"], ref)
        rep.r(f"total employment, ref {ref}", d, 1)
        ratios.append({"layer": "A skills", "series": "construction employment (national accounts)", **d})
    lost = na.loc[2002, "EMP_DC"] - na["EMP_DC"].loc[2002:].min()
    rep(f"jobs lost peak to trough: {lost:.1f} thousand ({lost / na.loc[2002, 'EMP_DC']:.1%} of the peak)")
    na.to_csv(RESULTS / "employment_national_accounts.csv")
    out["na"] = na

    rep.h("A2. Age of the construction workforce against all industries (LFS, thousand persons, shares of 15+)")
    c1 = eurostat("lfs_age_nace1").pivot_table(index="year", columns="age", values="value")
    n2 = eurostat("lfs_age_nace2")
    c2 = n2[n2.nace_r2 == "F"].pivot_table(index="year", columns="age", values="value")
    t2 = n2[n2.nace_r2 == "TOTAL"].pivot_table(index="year", columns="age", values="value")
    t1 = eurostat("lfs_age_total_nace1").pivot_table(index="year", columns="age", values="value")
    flags = n2[(n2.nace_r2 == "F") & (n2.age == "Y15-24")].set_index("year")["flag"].fillna("")
    # NACE Rev. 1.1 to 2007, NACE Rev. 2 from 2008; 2008 exists in both and is used to check the splice
    rep(f"2008 in both classifications: construction 15+ {c1.loc[2008, 'Y_GE15']:.1f} (F45) vs {c2.loc[2008, 'Y_GE15']:.1f} (F); "
        f"50+ share {c1.loc[2008, 'Y_GE50'] / c1.loc[2008, 'Y_GE15']:.3f} vs {c2.loc[2008, 'Y_GE50'] / c2.loc[2008, 'Y_GE15']:.3f}")
    flagged = {int(y): f for y, f in flags.items() if f}
    rep(f"Eurostat flags on the under-25 construction cell: {flagged if flagged else 'none'}")
    cons = pd.concat([c1.loc[:2007], c2])
    tot = pd.concat([t1.loc[:2007], t2])
    age = pd.DataFrame({
        "construction_total": cons["Y_GE15"],
        "construction_50plus": cons["Y_GE50"],
        "construction_under25": cons["Y15-24"],
        "construction_share_50plus": cons["Y_GE50"] / cons["Y_GE15"],
        "construction_share_under25": cons["Y15-24"] / cons["Y_GE15"],
        "all_share_50plus": tot["Y_GE50"] / tot["Y_GE15"],
        "all_share_under25": tot["Y15-24"] / tot["Y_GE15"],
    })
    age["construction_55plus"] = cons["Y55-64"] + cons["Y_GE65"].fillna(0) if "Y_GE65" in cons else np.nan
    for y in (1998, 2002, 2008, 2014, 2015, 2019, 2025):
        if y in age.index:
            r = age.loc[y]
            rep(f"{y}: construction {r.construction_total:.1f}k; 50+ {r.construction_50plus:.1f}k ({r.construction_share_50plus:.1%}); "
                f"under 25 {fmt(r.construction_under25, 1)}k ({r.construction_share_under25:.1%}); "
                f"all industries 50+ {r.all_share_50plus:.1%}, under 25 {r.all_share_under25:.1%}")
    cross = age.index[(age["construction_share_50plus"] > age["all_share_50plus"])]
    rep(f"first year the construction 50+ share exceeds the all-industry share: {int(cross.min())}")
    under50 = age["construction_total"] - age["construction_50plus"]
    rep(f"workers aged 50+ in construction: 2002 {age.loc[2002, 'construction_50plus']:.1f}k; 2008 {age.loc[2008, 'construction_50plus']:.1f}k; "
        f"2025 {age.loc[2025, 'construction_50plus']:.1f}k")
    rep(f"workers under 50: 2002 {under50.loc[2002]:.1f}k; 2008 {under50.loc[2008]:.1f}k; 2025 {under50.loc[2025]:.1f}k "
        f"({under50.loc[2025] / under50.loc[2002]:.1%} of 2002, {under50.loc[2025] / under50.loc[2008]:.1%} of 2008)")
    if "construction_55plus" in age and not np.isnan(age.loc[2025, "construction_55plus"]):
        rep(f"2025: construction workers aged 55+ {age.loc[2025, 'construction_55plus']:.1f}k ({age.loc[2025, 'construction_55plus'] / age.loc[2025, 'construction_total']:.1%})")
    for ref in (2002, 2008):
        d = recovered_share(age["construction_under25"].dropna(), ref)
        rep.r(f"construction workers under 25, ref {ref}", d, 1)
        ratios.append({"layer": "A skills", "series": "construction workers under 25 (LFS)", **d})
    age.to_csv(RESULTS / "workforce_age.csv")
    out["age"] = age

    rep.h("A3. Occupations in construction (LFS, ISCO-08, thousand persons)")
    occ = eurostat("lfs_occupation").pivot_table(index="year", columns="isco08", values="value")
    for y in (2008, 2011, 2014, 2019, 2025):
        r = occ.loc[y]
        rep(f"{y}: total {r.TOTAL:.1f}; craft (7) {r.OC7:.1f} ({r.OC7 / r.TOTAL:.1%}); elementary (9) {r.OC9:.1f} ({r.OC9 / r.TOTAL:.1%}); "
            f"professionals+technicians (2+3) {r.OC2 + r.OC3:.1f} ({(r.OC2 + r.OC3) / r.TOTAL:.1%}); managers (1) {r.OC1:.1f}")
    d = recovered_share(occ["OC7"], 2008)
    rep.r("craft and related trades workers, ref 2008", d, 1)
    ratios.append({"layer": "A skills", "series": "craft and trades workers (LFS)", **d})
    trough = int(occ["TOTAL"].loc[2008:].idxmin())
    dtot = occ.loc[2025, "TOTAL"] - occ.loc[trough, "TOTAL"]
    parts = {"craft (7)": "OC7", "elementary (9)": "OC9", "professionals and technicians (2+3)": ["OC2", "OC3"],
             "managers (1)": "OC1", "clerical (4)": "OC4", "machine operators (8)": "OC8"}
    rep(f"change {trough}->2025 in total: {dtot:+.1f}k, of which:")
    for lab, cols in parts.items():
        cols = cols if isinstance(cols, list) else [cols]
        dv = occ.loc[2025, cols].sum() - occ.loc[trough, cols].sum()
        rep(f"  {lab}: {dv:+.1f}k ({dv / dtot:.0%} of the change)")
    occ.to_csv(RESULTS / "occupations.csv")
    out["occ"] = occ

    rep.h("A4. Educational attainment of employees (LFS, % of employees 15-64)")
    ed = eurostat("lfs_education").pivot_table(index="year", columns=["nace_r2", "isced11"], values="value")
    for y in (2008, 2014, 2025):
        rep(f"{y}: construction at most lower secondary {ed.loc[y, ('F', 'ED0-2')]:.1f}% (all industries {ed.loc[y, ('TOTAL', 'ED0-2')]:.1f}%); "
            f"construction tertiary {ed.loc[y, ('F', 'ED5-8')]:.1f}% (all {ed.loc[y, ('TOTAL', 'ED5-8')]:.1f}%)")
    y = 2025
    rep(f"{y}: construction vocational upper secondary/post-secondary {ed.loc[y, ('F', 'ED35_45')]:.1f}% (all industries {ed.loc[y, ('TOTAL', 'ED35_45')]:.1f}%)")

    rep.h("A5. Graduates in architecture and construction (educ_uoe_grad02, ISCED-F 073, number)")
    g = eurostat("graduates_construction").pivot_table(index="year", columns="isced11", values="value")
    for y in g.index:
        rep(f"{y}: vocational upper/post-secondary (35+45) {fmt(g.loc[y].get('ED35_45', np.nan))}; tertiary (5-8) {fmt(g.loc[y].get('ED5-8', np.nan))}; "
            f"bachelor {fmt(g.loc[y].get('ED6', np.nan))}; master {fmt(g.loc[y].get('ED7', np.nan))}")
    voc = g["ED35_45"].dropna()
    rep(f"vocational graduates {int(voc.index.min())}-{int(voc.index.max())}: total {voc.sum():,.0f}; mean {voc.mean():,.0f} a year; "
        f"mean since 2017 {voc.loc[2017:].mean():,.0f} a year")
    craft25 = occ.loc[2025, "OC7"] * 1000
    rep(f"craft workers in construction in 2025 per vocational graduate in 2024: {craft25 / g.loc[2024, 'ED35_45']:,.0f}")
    g.to_csv(RESULTS / "graduates_construction.csv")
    out["grad"] = g

    rep.h("A6. Price of labour against materials (INE construction cost index for new housing, 2021 = 100, annual means)")
    monthly, annual = cost_index_annual()
    for y in (2000, 2008, 2015, 2019, 2025):
        rep(f"{y}: labour {annual.loc[y, 'labour']:.1f}; materials {annual.loc[y, 'materials']:.1f}; total {annual.loc[y, 'total']:.1f}")
    last_m = monthly.index[-1]
    rep(f"latest month {last_m[0]}-{last_m[1]:02d}: labour {monthly.iloc[-1]['labour']:.1f}; materials {monthly.iloc[-1]['materials']:.1f}")
    for a, b in ((2008, 2025), (2019, 2025), (2000, 2008), (2008, 2019)):
        rep(f"{a}->{b}: labour {annual.loc[b, 'labour'] / annual.loc[a, 'labour'] - 1:+.1%}; materials {annual.loc[b, 'materials'] / annual.loc[a, 'materials'] - 1:+.1%}")
    for y in (2019, 2022, 2023, 2024, 2025):
        yoy_l = annual.loc[y, "labour"] / annual.loc[y - 1, "labour"] - 1
        yoy_m = annual.loc[y, "materials"] / annual.loc[y - 1, "materials"] - 1
        rep(f"{y} annual change: labour {yoy_l:+.1%}; materials {yoy_m:+.1%}")
    lci = eurostat("labour_cost_index").pivot_table(index="year", columns="nace_r2", values="value")
    rep(f"Eurostat labour cost index (2020 = 100) 2025: construction {lci.loc[2025, 'F']:.1f}; business economy B-S {lci.loc[2025, 'B-S']:.1f}; "
        f"2019: {lci.loc[2019, 'F']:.1f} / {lci.loc[2019, 'B-S']:.1f}")
    rep(f"labour cost growth 2019->2025: construction {lci.loc[2025, 'F'] / lci.loc[2019, 'F'] - 1:+.1%}; business economy {lci.loc[2025, 'B-S'] / lci.loc[2019, 'B-S'] - 1:+.1%}")
    monthly.reset_index().to_csv(RESULTS / "cost_index_monthly.csv", index=False)
    annual.to_csv(RESULTS / "cost_index_annual.csv")
    out["cost_m"], out["cost_a"] = monthly, annual

    rep.h("A7. Pay per employee in construction relative to all industries (national accounts)")
    comp = eurostat("na_compensation").pivot_table(index="year", columns="nace_r2", values="value")
    emp_all = eurostat("na_employees_total").set_index("year")["value"]
    rel = (comp["F"] / na["SAL_DC"]) / (comp["TOTAL"] / emp_all)
    for y in (1995, 2002, 2008, 2014, 2019, 2025):
        rep(f"{y}: {rel.loc[y]:.3f}")
    rel.rename("relative_pay").to_csv(RESULTS / "relative_pay.csv")
    return out


def section_b(rep: Report, ratios: list[dict], firms: pd.DataFrame, persons: pd.DataFrame) -> dict:
    rep.h("B1. Contractors and specialised trades (INE SCIE: enterprises and persons employed)")
    last = int(firms.index.max())
    for code in ("41200", "43", *TRADES):
        rf = recovered_share(firms[code], 2008)
        rp = recovered_share(persons[code], 2008)
        rep.r(f"{LABELS[code]} enterprises", rf)
        rep.r(f"{LABELS[code]} persons employed", rp)
        if code in ("41200", "43"):
            ratios.append({"layer": "B firms", "series": f"{LABELS[code]}, persons employed", **rp})
    for y in (2008, last):
        rep(f"{y}: persons per building contractor {persons.loc[y, '41200'] / firms.loc[y, '41200']:.2f}; per trades firm {persons.loc[y, '43'] / firms.loc[y, '43']:.2f}")

    rep.h("B2. Size of construction firms (Eurostat SBS; 2008-2020 sbs_sc_con_r2, 2021- sbs_sc_ovw)")
    old = eurostat("sbs_size_construction")
    new = eurostat("sbs_size_construction_2021")
    new = new.rename(columns={"indic_sbs": "indic_sb"}).replace({"indic_sb": {"ENT_NR": "V11110", "EMP_NR": "V16110"}})
    sbs = pd.concat([old[old.year <= 2020], new[new.year >= 2021]])
    sbs = sbs[sbs.size_emp.isin(["TOTAL", "0-9", "10-19", "20-49", "50-249", "GE250"])]
    piv = sbs.pivot_table(index=["nace_r2", "indic_sb", "year"], columns="size_emp", values="value")
    rows = []
    for nace in ("F411", "F412", "F43"):
        ent = piv.loc[(nace, "V11110")]
        sme = ent[["10-19", "20-49", "50-249"]].sum(axis=1, min_count=3)
        med = ent["50-249"]
        for y in ent.index:
            rows.append({"nace": nace, "year": y, "enterprises": ent.loc[y, "TOTAL"], "micro_0_9": ent.loc[y, "0-9"],
                         "small_medium_10_249": sme.loc[y], "medium_50_249": med.loc[y]})
        d = recovered_share(sme, 2008)
        rep.r(f"{nace} enterprises with 10-249 persons", d)
        dm = recovered_share(med, 2008)
        rep.r(f"{nace} enterprises with 50-249 persons", dm)
        dmic = recovered_share(ent["0-9"], 2008)
        rep.r(f"{nace} enterprises with 0-9 persons", dmic)
        if nace in ("F412", "F43"):
            ratios.append({"layer": "B firms", "series": f"{'building contractors' if nace == 'F412' else 'specialised trades'} with 10-249 persons", **d})
        rep(f"{nace} share of enterprises with 0-9 persons: 2008 {ent.loc[2008, '0-9'] / ent.loc[2008, 'TOTAL']:.1%}; {int(ent.index.max())} "
            f"{ent.iloc[-1]['0-9'] / ent.iloc[-1]['TOTAL']:.1%}")
    rep("note: 2020 and 2021 come from different SBS regulations; the 2021 totals match the INE counts for the same year")
    size = pd.DataFrame(rows)
    size.to_csv(RESULTS / "firm_size.csv", index=False)
    return {"size": size}


def section_c(rep: Report, ratios: list[dict], firms, persons, production, cost_a) -> dict:
    out: dict = {}
    rep.h("C1. Developers (CAE 41.10, INE SCIE)")
    deflator = cost_a["total"] / cost_a.loc[2008, "total"]
    real = production["41100"] / deflator.reindex(production.index)
    for label, s in (("enterprises", firms["41100"]), ("persons employed", persons["41100"]),
                     ("production, EUR million (nominal)", production["41100"] / 1e6),
                     ("production, EUR million at 2008 construction costs", real / 1e6)):
        d = recovered_share(s, 2008)
        rep.r(f"developers {label}", d, 0 if "EUR" not in label else 0)
        if label in ("persons employed", "production, EUR million at 2008 construction costs"):
            ratios.append({"layer": "C promoters", "series": f"developers, {label.replace(', EUR million at 2008 construction costs', ' (real)')}", **d})
    rep(f"construction cost deflator 2024/2008: {deflator.loc[2024]:.3f}")
    rep(f"persons per developer: 2008 {persons.loc[2008, '41100'] / firms.loc[2008, '41100']:.2f}; {int(firms.index.max())} "
        f"{persons.iloc[-1]['41100'] / firms.iloc[-1]['41100']:.2f}")

    rep.h("C2. Who commissions new dwellings (INE 0013236, dwellings licensed in new construction for family housing)")
    lic = ine("ine_licensed_by_promoter").pivot_table(index="period", columns="dim_3", values="value")
    lic.index = lic.index.astype(int)
    lic = lic.rename(columns={"1": "individuals", "2": "public", "3": "companies", "4": "other", "T": "total"})
    for y in (2002, 2008, 2013, 2014, 2019, 2025):
        r = lic.loc[y]
        rep(f"{y}: total {r.total:,.0f}; private individuals {r.individuals:,.0f} ({r.individuals / r.total:.1%}); "
            f"private companies {r.companies:,.0f} ({r.companies / r.total:.1%}); public {r.public:,.0f}; other {r.other:,.0f}")
    for col in ("individuals", "companies", "total"):
        for ref in (2002, 2008):
            d = recovered_share(lic[col], ref)
            rep.r(f"licensed dwellings, {col}, ref {ref}", d)
            if col != "total":
                ratios.append({"layer": "C promoters", "series": f"dwellings licensed by private {col}", **d})
    rep(f"public bodies 2002-2007 mean {lic.loc[2002:2007, 'public'].mean():,.0f} a year; 2014-2025 mean {lic.loc[2014:2025, 'public'].mean():,.0f}")
    rep(f"cooperatives and other 2002-2007 mean {lic.loc[2002:2007, 'other'].mean():,.0f} a year; 2014-2025 mean {lic.loc[2014:2025, 'other'].mean():,.0f}")
    lic.to_csv(RESULTS / "licensed_by_promoter.csv")
    out["lic"] = lic

    rep.h("C3. Dwellings completed (INE 0013249 by promoter, 0012778 quarterly totals)")
    comp = ine("ine_completed_by_promoter").pivot_table(index="period", columns="dim_3", values="value")
    comp.index = comp.index.astype(int)
    comp = comp.rename(columns={"1": "individuals", "2": "public", "3": "companies", "4": "other", "T": "total"})
    for y in (2011, 2015, 2019, 2025):
        r = comp.loc[y]
        rep(f"{y}: total {r.total:,.0f}; individuals {r.individuals:,.0f} ({r.individuals / r.total:.1%}); companies {r.companies:,.0f} ({r.companies / r.total:.1%})")
    q = ine("ine_completed_quarterly")
    q["year"] = q["period"].str[-4:].astype(int)
    qa = q.groupby("year")["value"].agg(["sum", "count"])
    qa = qa[qa["count"] == 4]["sum"]
    for y in (2004, 2008, int(qa.idxmin()), int(qa.index.max())):
        rep(f"completed dwellings (sum of quarters) {y}: {qa.loc[y]:,.0f}")
    comp.to_csv(RESULTS / "completed_by_promoter.csv")
    qa.rename("completed_dwellings").to_csv(RESULTS / "completed_dwellings.csv")
    out["completed"] = qa

    rep.h("C4. What gets licensed: single-family houses and apartment buildings (INE 0013245, 0013244, 0013235)")
    houses = ine_annual("ine_licensed_houses")
    apt = ine_annual("ine_licensed_apartment_buildings")
    dwell = ine_annual("ine_licensed_dwellings")
    per_bld = (dwell - houses) / apt
    form = pd.DataFrame({"dwellings": dwell, "houses": houses, "apartment_buildings": apt, "dwellings_per_apartment_building": per_bld})
    for y in (2002, 2008, 2012, 2019, 2025):
        r = form.loc[y]
        rep(f"{y}: dwellings {r.dwellings:,.0f}; houses {r.houses:,.0f}; apartment buildings {r.apartment_buildings:,.0f}; "
            f"dwellings per apartment building {r.dwellings_per_apartment_building:.1f}")
    rep("dwellings per apartment building assumes one dwelling per licensed house (a single-family building)")
    for ref in (2002, 2008):
        d = recovered_share(houses, ref)
        rep.r(f"licensed single-family houses, ref {ref}", d)
    form.to_csv(RESULTS / "licensed_building_form.csv")
    out["form"] = form
    return out


def section_d(rep: Report, ratios: list[dict], firms, persons, production) -> dict:
    out: dict = {}
    rep.h("D1. Selling and trading layers (INE SCIE)")
    for code in ("68311", "68312", "68100", "68200", "71110", "71120"):
        rf = recovered_share(firms[code], 2008)
        rp = recovered_share(persons[code], 2008)
        rep.r(f"{LABELS[code]} enterprises", rf)
        rep.r(f"{LABELS[code]} persons employed", rp)
        if code in ("68311", "68312", "68100"):
            ratios.append({"layer": "D sellers", "series": f"{LABELS[code].split(' (')[0]}, persons employed", **rp})
        if code in ("71110", "71120"):
            ratios.append({"layer": "A skills", "series": f"{LABELS[code].split(' (')[0]} firms, persons employed", **rp})
    last = int(firms.index.max())
    for code in ("68311", "68312", "68100"):
        rep(f"{LABELS[code]}: enterprises {last}/2008 x{firms.loc[last, code] / firms.loc[2008, code]:.2f}; "
            f"vs their own trough x{firms.loc[last, code] / firms[code].loc[2008:].min():.2f}")

    rep.h("D2. Builders per seller (persons employed)")
    b = persons[list(BUILDERS)].sum(axis=1)
    s = persons[list(SELLERS)].sum(axis=1)
    ratio = b / s
    for y in (2008, 2013, 2019, last):
        rep(f"{y}: builders {b.loc[y]:,.0f}; agencies and finders {s.loc[y]:,.0f}; builders per seller {ratio.loc[y]:.1f}")
    bf = firms[list(BUILDERS)].sum(axis=1)
    sf = firms[list(SELLERS)].sum(axis=1)
    rep(f"by enterprises: 2008 {bf.loc[2008] / sf.loc[2008]:.1f} building firms per agency or finder; {last} {bf.loc[last] / sf.loc[last]:.1f}")

    rep.h("D3. Production of agencies and finders relative to developers (deflator-free)")
    pm = production[list(SELLERS)].sum(axis=1)
    pr = pm / production["41100"]
    for y in (2008, 2014, 2019, last):
        rep(f"{y}: agencies+finders EUR {pm.loc[y] / 1e6:,.0f}m; developers EUR {production.loc[y, '41100'] / 1e6:,.0f}m; ratio {pr.loc[y]:.2f}")
    rep(f"first year agencies and finders produce more than developers: {int(pr[pr > 1].index.min())}")
    rep(f"buying and selling own property: EUR {production.loc[2008, '68100'] / 1e6:,.0f}m (2008) -> EUR {production.loc[last, '68100'] / 1e6:,.0f}m ({last})")

    rep.h("D4. Agencies against the volume of sales (INE 0012787 transactions)")
    t = ine("ine_transactions")
    tt = t[(t.dim_4 == "T") & (t.dim_5 == "T")].pivot_table(index="period", columns="dim_3", values="value")
    tt.index = tt.index.astype(int)
    tt = tt.rename(columns={"1": "new", "2": "existing", "T": "total"}) if "1" in tt.columns else tt
    labels = dict(zip(t["dim_3"], t["dim_3_label"]))
    tt = tt.rename(columns={k: {"Novos": "new", "Existentes": "existing", "Total": "total"}.get(v, k) for k, v in labels.items()})
    for y in (2009, 2013, 2019, 2024, 2025):
        rep(f"{y}: transactions {tt.loc[y, 'total']:,.0f} (existing {tt.loc[y, 'existing']:,.0f}, new {tt.loc[y, 'new']:,.0f})")
    for y in (2009, 2013, 2019, last):
        rep(f"{y}: agency and finder workers per 1,000 transactions {s.loc[y] / tt.loc[y, 'total'] * 1000:.0f}")
    rep(f"2009->{last}: transactions x{tt.loc[last, 'total'] / tt.loc[2009, 'total']:.2f}; agency and finder workers x{s.loc[last] / s.loc[2009]:.2f}; "
        f"existing-dwelling sales x{tt.loc[last, 'existing'] / tt.loc[2009, 'existing']:.2f}; new-dwelling sales x{tt.loc[last, 'new'] / tt.loc[2009, 'new']:.2f}")
    tt.to_csv(RESULTS / "transactions.csv")

    layers = pd.DataFrame({
        "developers_41100": persons["41100"],
        "building_contractors_41200": persons["41200"],
        "specialised_trades_43": persons["43"],
        "agencies_finders_68311_68312": s,
        "buying_selling_own_68100": persons["68100"],
        "builders_per_seller": ratio,
        "agencies_finders_production_over_developers": pr,
    })
    layers.to_csv(RESULTS / "layers_persons_employed.csv")
    firms.to_csv(RESULTS / "scie_enterprises.csv")
    persons.to_csv(RESULTS / "scie_persons_employed.csv")
    production.to_csv(RESULTS / "scie_production.csv")
    out["layers"] = layers
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rep = Report()
    ratios: list[dict] = []
    rep("Portugal's housebuilding sector, measured layer by layer. Every number in the article comes from this file.")
    rep("Sources: Eurostat (reuse under Commission Decision 2011/833/EU) and INE, Statistics Portugal (CC BY 4.0).")

    a = section_a(rep, ratios)
    firms, persons, production = scie("firms"), scie("persons"), scie("production")
    rep.h("SCIE splice check (NUTS 2013 vs NUTS 2024 indicators, overlapping years, national totals)")
    for kind in ("firms", "persons", "production"):
        ov = scie_overlap(kind)
        if len(ov):
            diff = (ov.loc["new"] - ov.loc["old"]).abs().max().max()
            rep(f"{kind}: overlapping years {sorted(set(ov.index.get_level_values(1)))}; max absolute difference {diff:,.0f}")
        else:
            rep(f"{kind}: no overlapping year")
    rep(f"enterprise counts cover {int(firms.index.min())}-{int(firms.index.max())}; persons and production {int(persons.index.min())}-{int(persons.index.max())}")
    rep.h("M. Three counts of the same workforce (thousand persons in construction)")
    na = a["na"]["EMP_DC"]
    lfs = a["age"]["construction_total"]
    scie_f = persons["F"] / 1000
    tri = pd.DataFrame({"national_accounts": na, "labour_force_survey": lfs, "business_statistics": scie_f}).loc[2008:]
    for y in (2008, 2011, 2014, 2019, 2021, 2024):
        r = tri.loc[y]
        rep(f"{y}: national accounts {r.national_accounts:.1f}; LFS {r.labour_force_survey:.1f}; business statistics (SCIE) {r.business_statistics:.1f}; "
            f"SCIE minus LFS {r.business_statistics - r.labour_force_survey:+.1f}")
    for c in tri.columns:
        rep(f"{c}: 2024 as a share of 2008 {tri.loc[2024, c] / tri.loc[2008, c]:.3f}")
    tri.to_csv(RESULTS / "workforce_three_counts.csv")
    section_b(rep, ratios, firms, persons)
    section_c(rep, ratios, firms, persons, production, a["cost_a"])
    section_d(rep, ratios, firms, persons, production)

    rep.h("E. Recovered share of the loss, every layer on one scale (R = (last - trough) / (ref - trough))")
    rt = pd.DataFrame(ratios)
    rt = rt.sort_values(["layer", "ref_year", "R"]).reset_index(drop=True)
    for _, r in rt.iterrows():
        rep(f"{r.layer} | {r.series} | ref {r.ref_year} | trough {r.trough_year} | last {r.last_year} | last/ref {r.last_over_ref:.2f} | R {fmt(r.R, 2)}")
    rt.to_csv(RESULTS / "recovered_share.csv", index=False)
    rep.write(RESULTS / "summary.txt")
    print((RESULTS / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
