"""Every number in the article, printed in the order the article uses them.

    python services_numbers.py > output.txt

Reads the seven Eurostat responses in data/raw/ (downloaded from the data registry and verified by fetch.py)
and writes the tables behind the figures to results/. Neither folder is committed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import services_core as sc
from jsonstat import read_jsonstat

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "raw"
RESULTS = HERE / "results"

SERVICES = ["G-I", "J", "K", "L", "M_N", "O-Q", "R-U"]
SERVICES_EX_L = ["G-I", "J", "K", "M_N", "O-Q", "R-U"]
INDUSTRY_LABELS = {
    "TOTAL": "All industries",
    "A": "Agriculture, forestry and fishing",
    "B-E": "Industry except construction",
    "C": "Manufacturing",
    "F": "Construction",
    "G-I": "Trade, transport, accommodation and food",
    "J": "Information and communication",
    "K": "Finance and insurance",
    "L": "Real estate",
    "M_N": "Professional, scientific, administrative",
    "O-Q": "Public administration, education, health",
    "R-U": "Arts, recreation and other services",
}
EU27 = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL",
        "PL", "PT", "RO", "SE", "SI", "SK"]  # fmt: skip
FIRST, LAST = 1995, 2024
C_FIRST, C_LAST = 2000, 2023  # country comparison; 2023 is the last year with complete energy balances for all members


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def load():
    gva = read_jsonstat(CACHE / "nama_10_a10.json")
    hours = read_jsonstat(CACHE / "nama_10_a10_e.json")
    energy = read_jsonstat(CACHE / "nrg_bal_c.json")
    tus = read_jsonstat(CACHE / "tus_00age.json")
    pc = read_jsonstat(CACHE / "nama_10_pc.json")
    pop = read_jsonstat(CACHE / "nama_10_pe.json")
    hicp = read_jsonstat(CACHE / "prc_hicp_aind.json")
    return gva, hours, energy, tus, pc, pop, hicp


def panel(df, geo, unit=None, col="nace_r2", **filters):
    x = df[df.geo == geo]
    if unit is not None:
        x = x[x.unit == unit]
    for k, v in filters.items():
        x = x[x[k] == v]
    return x.pivot_table(index="time", columns=col, values="value")


def sector_series(gva, hours, energy, geo, first, last):
    """Real output, hours and final energy for services (with and without real estate) and industry."""
    clv = panel(gva, geo, "CLV20_MEUR").loc[first:last]
    cp = panel(gva, geo, "CP_MEUR").loc[first:last]
    hw = panel(hours, geo, "THS_HW").loc[first:last]
    en = panel(energy, geo, None, col="nrg_bal", siec="TOTAL").loc[first:last]
    out = pd.DataFrame(index=clv.index)
    out["services_output"] = sc.tornqvist_index(clv[SERVICES], cp[SERVICES])
    out["services_ex_l_output"] = sc.tornqvist_index(clv[SERVICES_EX_L], cp[SERVICES_EX_L])
    out["industry_output"] = clv["B-E"] / clv["B-E"].iloc[0]
    out["total_output"] = clv["TOTAL"] / clv["TOTAL"].iloc[0]
    out["services_hours"] = hw[SERVICES].sum(axis=1, min_count=len(SERVICES))
    out["services_ex_l_hours"] = hw[SERVICES_EX_L].sum(axis=1, min_count=len(SERVICES_EX_L))
    out["industry_hours"] = hw["B-E"]
    out["total_hours"] = hw["TOTAL"]
    out["services_energy"] = en.get("FC_OTH_CP_E")
    out["industry_energy"] = en.get("FC_IND_E")
    out["final_energy"] = en.get("FC_E")
    out["services_nominal_share"] = cp[SERVICES].sum(axis=1) / cp["TOTAL"]
    return out


# ================================================================================================
def eu_shares(gva, hours, energy):
    section("1. Services in the EU: shares of value added, hours and final energy (EU27, 2020 composition)")
    cp = panel(gva, "EU27_2020", "CP_MEUR")
    hw = panel(hours, "EU27_2020", "THS_HW")
    en = panel(energy, "EU27_2020", None, col="nrg_bal", siec="TOTAL")
    rows = []
    for y in range(FIRST, LAST + 1):
        rows.append(
            {
                "year": y,
                "services_gva_share": cp.loc[y, SERVICES].sum() / cp.loc[y, "TOTAL"],
                "services_hours_share": hw.loc[y, SERVICES].sum() / hw.loc[y, "TOTAL"],
                "services_energy_share": en.loc[y, "FC_OTH_CP_E"] / en.loc[y, "FC_E"],
                "industry_gva_share": cp.loc[y, "B-E"] / cp.loc[y, "TOTAL"],
                "industry_hours_share": hw.loc[y, "B-E"] / hw.loc[y, "TOTAL"],
                "industry_energy_share": en.loc[y, "FC_IND_E"] / en.loc[y, "FC_E"],
                "transport_energy_share": en.loc[y, "FC_TRA_E"] / en.loc[y, "FC_E"],
                "households_energy_share": en.loc[y, "FC_OTH_HH_E"] / en.loc[y, "FC_E"],
            }
        )
    s = pd.DataFrame(rows)
    s.to_csv(RESULTS / "eu_shares.csv", index=False, float_format="%.6g")
    for y in (FIRST, 2008, 2019, LAST):
        r = s[s.year == y].iloc[0]
        print(
            f"{y}: services {pct(r.services_gva_share)} of value added (current prices), {pct(r.services_hours_share)} of hours worked, "
            f"{pct(r.services_energy_share)} of final energy; industry (B-E) {pct(r.industry_gva_share)}, {pct(r.industry_hours_share)}, "
            f"{pct(r.industry_energy_share)}"
        )
    r = s[s.year == LAST].iloc[0]
    print(
        f"{LAST} final energy by user: industry {pct(r.industry_energy_share)}, transport {pct(r.transport_energy_share)}, "
        f"households {pct(r.households_energy_share)}, commercial and public services {pct(r.services_energy_share)}"
    )
    print(
        f"Hours worked, EU27: services {hw.loc[FIRST, SERVICES].sum() / 1e6:,.1f} billion in {FIRST}, "
        f"{hw.loc[LAST, SERVICES].sum() / 1e6:,.1f} billion in {LAST}; "
        f"industry {hw.loc[FIRST, 'B-E'] / 1e6:,.1f} -> {hw.loc[LAST, 'B-E'] / 1e6:,.1f} billion"
    )
    print(
        f"Final energy of commercial and public services, EU27 (ktoe): {FIRST} {en.loc[FIRST, 'FC_OTH_CP_E']:,.0f}, "
        f"peak {en['FC_OTH_CP_E'].max():,.0f} in {en['FC_OTH_CP_E'].idxmax()}, {LAST} {en.loc[LAST, 'FC_OTH_CP_E']:,.0f}"
    )
    print(
        f"    change {FIRST}-{LAST}: {pct(en.loc[LAST, 'FC_OTH_CP_E'] / en.loc[FIRST, 'FC_OTH_CP_E'] - 1)}; "
        f"total final energy {pct(en.loc[LAST, 'FC_E'] / en.loc[FIRST, 'FC_E'] - 1)}; "
        f"industry {pct(en.loc[LAST, 'FC_IND_E'] / en.loc[FIRST, 'FC_IND_E'] - 1)}"
    )
    el = panel(energy, "EU27_2020", None, col="nrg_bal", siec="E7000")
    for y in (FIRST, LAST):
        print(f"    electricity share of services' final energy in {y}: {pct(el.loc[y, 'FC_OTH_CP_E'] / en.loc[y, 'FC_OTH_CP_E'])}")
    print(f"    services' electricity use {FIRST}-{LAST}: {pct(el.loc[LAST, 'FC_OTH_CP_E'] / el.loc[FIRST, 'FC_OTH_CP_E'] - 1)}")


def eu_decomposition(gva, hours, energy):
    section("2. Services and industry: output, hours and energy (EU27)")
    s = sector_series(gva, hours, energy, "EU27_2020", FIRST, LAST)
    idx = s[[c for c in s.columns if not c.endswith("share")]].copy()
    for c in idx.columns:
        idx[c] = idx[c] / idx[c].iloc[0]
    idx.reset_index().rename(columns={"time": "year"}).to_csv(RESULTS / "eu_sector_index.csv", index=False, float_format="%.6g")
    print(f"Multiples {FIRST}-{LAST}:")
    for c in (
        "services_output",
        "services_ex_l_output",
        "services_hours",
        "services_ex_l_hours",
        "services_energy",
        "industry_output",
        "industry_hours",
        "industry_energy",
        "total_output",
        "total_hours",
    ):
        print(f"    {c:22s} x{s[c].loc[LAST] / s[c].loc[FIRST]:.3f}")
    rows = []
    periods = [(FIRST, 2008), (2008, 2019), (2019, LAST), (FIRST, LAST)]
    for sector, out, inputs in (
        ("services", "services_output", ("services_hours", "services_energy")),
        ("services excl. real estate", "services_ex_l_output", ("services_ex_l_hours", "services_energy")),
        ("industry (B-E)", "industry_output", ("industry_hours", "industry_energy")),
    ):
        for inp in inputs:
            for y0, y1 in periods:
                o, i, p = sc.decompose(s[out].loc[y0], s[out].loc[y1], s[inp].loc[y0], s[inp].loc[y1], y1 - y0)
                rows.append({"sector": sector, "input": inp.split("_")[-1], "from": y0, "to": y1, "output": o, "input_growth": i, "per_input": p})
                share = i / o if o != 0 else np.nan
                print(
                    f"{sector:27s} {inp.split('_')[-1]:6s} {y0}-{y1}: output {o:+.2f}%/yr = {inp.split('_')[-1]} {i:+.2f} + output per unit {p:+.2f}"
                    f"{'' if not np.isfinite(share) else f'  (input share of growth {100 * share:.0f}%)'}"
                )
    pd.DataFrame(rows).to_csv(RESULTS / "eu_decomposition.csv", index=False, float_format="%.6g")
    return s


def productivity_by_industry(gva, hours):
    section("3. Real value added per hour worked by industry (EU27)")
    clv = panel(gva, "EU27_2020", "CLV20_MEUR")
    hw = panel(hours, "EU27_2020", "THS_HW")
    rows = []
    for code, label in INDUSTRY_LABELS.items():
        p0 = clv.loc[FIRST, code] / hw.loc[FIRST, code]
        p1 = clv.loc[LAST, code] / hw.loc[LAST, code]
        g = sc.cagr(p0, p1, LAST - FIRST)
        h = sc.cagr(hw.loc[FIRST, code], hw.loc[LAST, code], LAST - FIRST)
        rows.append(
            {
                "code": code,
                "label": label,
                "output_per_hour_growth": g,
                "hours_growth": h,
                "hours_share_last": hw.loc[LAST, code] / hw.loc[LAST, "TOTAL"],
            }
        )
        print(
            f"{code:5s} {label:45s} output per hour {pct(g, 2)}/yr; hours {pct(h, 2)}/yr; "
            f"{pct(hw.loc[LAST, code] / hw.loc[LAST, 'TOTAL'])} of hours in {LAST}"
        )
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "productivity_by_industry.csv", index=False, float_format="%.6g")
    serv = d[d.code.isin(SERVICES_EX_L)]
    print(
        "Hours-weighted mean output-per-hour growth, services excluding real estate: "
        f"{pct((serv.output_per_hour_growth * serv.hours_share_last).sum() / serv.hours_share_last.sum(), 2)}"
    )


def required_rates(s):
    section("4. What flat inputs would require")
    o = sc.cagr(s.services_ex_l_output.loc[FIRST], s.services_ex_l_output.loc[LAST], LAST - FIRST)
    ph = sc.required_productivity_growth(o, sc.cagr(s.services_ex_l_hours.loc[FIRST], s.services_ex_l_hours.loc[LAST], LAST - FIRST))
    pe = sc.required_productivity_growth(o, sc.cagr(s.services_energy.loc[FIRST], s.services_energy.loc[LAST], LAST - FIRST))
    print(
        f"Services excluding real estate, {FIRST}-{LAST}: real output {pct(o, 2)}/yr; output per hour {pct(ph, 2)}/yr; "
        f"output per unit of energy {pct(pe, 2)}/yr"
    )
    print(f"    with hours flat, output per hour would have to rise {pct(o, 2)}/yr, {o / ph:.1f} times the rate achieved")
    print(f"    at the achieved rate, output per hour rises tenfold in {sc.years_to_factor(10, ph):.0f} years")
    print(f"    with energy flat, output per unit of energy would have to rise {pct(o, 2)}/yr; achieved {pct(pe, 2)}/yr")


def time_per_person(hours, pop):
    section("5. Hours worked per person and per worker (EU27)")
    hw = panel(hours, "EU27_2020", "THS_HW")
    persons = panel(hours, "EU27_2020", "THS_PER")
    p = panel(pop, "EU27_2020", "THS_PER", col="na_item")["POP_NC"]
    rows = []
    for y in range(FIRST, LAST + 1):
        rows.append(
            {
                "year": y,
                "hours_per_person": hw.loc[y, "TOTAL"] / p.loc[y],
                "services_hours_per_person": hw.loc[y, SERVICES].sum() / p.loc[y],
                "industry_hours_per_person": hw.loc[y, "B-E"] / p.loc[y],
                "hours_per_worker": hw.loc[y, "TOTAL"] / persons.loc[y, "TOTAL"],
            }
        )
    d = pd.DataFrame(rows).set_index("year")
    d.to_csv(RESULTS / "hours_per_person.csv", float_format="%.6g")
    for y in (FIRST, 2008, 2019, LAST):
        r = d.loc[y]
        print(
            f"{y}: hours worked per resident {r.hours_per_person:,.0f}; in services {r.services_hours_per_person:,.0f}; "
            f"in industry {r.industry_hours_per_person:,.0f}; "
            f"hours per person employed {r.hours_per_worker:,.0f}"
        )
    print(
        f"Hours in a year: {sc.HOURS_PER_YEAR:,.0f}; "
        f"hours worked per resident in {LAST} = {pct(d.loc[LAST, 'hours_per_person'] / sc.HOURS_PER_YEAR)} of them"
    )
    print(f"EU27 population {LAST}: {p.loc[LAST] / 1e3:,.1f} million; hours lived per year {p.loc[LAST] * 1e3 * sc.HOURS_PER_YEAR:.3g}")


def countries(gva, hours, energy):
    section(f"6. Member states, {C_FIRST} to {C_LAST}: services output, hours and energy")
    rows = []
    for geo in EU27:
        try:
            s = sector_series(gva, hours, energy, geo, C_FIRST, C_LAST)
        except KeyError:
            continue
        need = ["services_ex_l_output", "services_ex_l_hours", "services_energy"]
        if s[need].loc[[C_FIRST, C_LAST]].isna().any().any() or len(s) != C_LAST - C_FIRST + 1:
            continue
        rows.append(
            {
                "geo": geo,
                "output_ratio": s.services_ex_l_output.loc[C_LAST] / s.services_ex_l_output.loc[C_FIRST],
                "hours_ratio": s.services_ex_l_hours.loc[C_LAST] / s.services_ex_l_hours.loc[C_FIRST],
                "energy_ratio": s.services_energy.loc[C_LAST] / s.services_energy.loc[C_FIRST],
            }
        )
    d = pd.DataFrame(rows)
    d["hours_class"] = [sc.classify_decoupling(o, h) for o, h in zip(d.output_ratio, d.hours_ratio, strict=True)]
    d["energy_class"] = [sc.classify_decoupling(o, e) for o, e in zip(d.output_ratio, d.energy_ratio, strict=True)]
    d.to_csv(RESULTS / "country_services.csv", index=False, float_format="%.6g")
    print(f"Member states with complete data: {len(d)} (services excluding real estate; energy of commercial and public services)")
    for k in ("hours", "energy"):
        c = d[f"{k}_class"].value_counts()
        print(f"    {k:6s}: absolute {c.get('absolute', 0)}, relative {c.get('relative', 0)}, none {c.get('none', 0)}")
    print(
        f"    services output grew in {int((d.output_ratio > 1).sum())}; hours grew in {int((d.hours_ratio > 1).sum())}; "
        f"energy grew in {int((d.energy_ratio > 1).sum())}"
    )
    print(f"    both hours and energy fell while output grew: {int(((d.hours_ratio < 1) & (d.energy_ratio < 1) & (d.output_ratio > 1)).sum())}")
    print(f"    median ratios: output x{d.output_ratio.median():.2f}, hours x{d.hours_ratio.median():.2f}, energy x{d.energy_ratio.median():.2f}")
    for k in ("hours", "energy"):
        b, se, r2 = sc.ols_slope(np.log(d.output_ratio), np.log(d[f"{k}_ratio"]))
        print(f"    elasticity of {k} to services output across member states: {b:.2f} (HC1 se {se:.2f}), R2 {r2:.2f}")
    for k in ("hours", "energy"):
        for cls in ("absolute", "none"):
            names = ", ".join(d[d[f"{k}_class"] == cls].geo)
            print(f"    {k} {cls}: {names}")


def time_use(tus, pc):
    section("7. Time use: free time against consumption (HETUS 2000 and 2010 waves)")
    t = tus.pivot_table(index=["geo", "time"], columns="acl00", values="value")
    cons_pps = pc[(pc.na_item == "P41") & (pc.unit == "CP_PPS_EU27_2020_HAB")].pivot_table(index=["geo", "time"], values="value")["value"]
    cons_real = pc[(pc.na_item == "P41") & (pc.unit == "CLV20_EUR_HAB")].pivot_table(index=["geo", "time"], values="value")["value"]
    d = pd.DataFrame(
        {
            "leisure_min": t["AC4-8"],
            "leisure_travel_min": t.get("AC9D"),
            "paid_work_min": t["AC1_TR"],
            "personal_care_min": t["AC0"],
            "sleep_min": t["AC01"],
            "household_min": t["AC3"],
            "shopping_services_min": t["AC36"],
            "tv_video_min": t["AC82"],
            "entertainment_culture_min": t["AC52"],
            "sports_min": t["AC6A"],
            "study_min": t["AC2"],
        }
    )
    d["consumption_pps"] = cons_pps.reindex(d.index)
    d["consumption_real"] = cons_real.reindex(d.index)
    d = d.reset_index()
    d.to_csv(RESULTS / "time_use.csv", index=False, float_format="%.6g")
    full = d.dropna(subset=["leisure_min", "consumption_pps"])
    print(f"Country-waves with time use and consumption: {len(full)} ({', '.join(f'{g} {y}' for g, y in zip(full.geo, full.time, strict=True))})")
    print(
        f"Free time (leisure, social and associative life), minutes per day: median {full.leisure_min.median():.0f}, "
        f"range {full.leisure_min.min():.0f} ({full.loc[full.leisure_min.idxmin(), 'geo']} {full.loc[full.leisure_min.idxmin(), 'time']}) to "
        f"{full.leisure_min.max():.0f} ({full.loc[full.leisure_min.idxmax(), 'geo']} {full.loc[full.leisure_min.idxmax(), 'time']}); "
        f"max/min {sc.ratio_range(full.leisure_min):.2f}"
    )
    print(
        f"Actual individual consumption per person (PPS): range {full.consumption_pps.min():,.0f} to {full.consumption_pps.max():,.0f}; "
        f"max/min {sc.ratio_range(full.consumption_pps):.2f}"
    )
    print(
        f"Coefficient of variation: free time {sc.coefficient_of_variation(full.leisure_min):.3f}; "
        f"consumption {sc.coefficient_of_variation(full.consumption_pps):.3f}"
    )
    lo, hi = full.consumption_pps.idxmin(), full.consumption_pps.idxmax()
    print(f"    lowest {full.loc[lo, 'geo']} {full.loc[lo, 'time']}, highest {full.loc[hi, 'geo']} {full.loc[hi, 'time']}")
    per_hour = full.consumption_pps / (full.leisure_min * 365.25 / 60)
    print(f"Consumption per hour of free time (PPS): range {per_hour.min():.1f} to {per_hour.max():.1f}; max/min {sc.ratio_range(per_hour):.2f}")
    for col, label in (
        ("leisure_min", "free time"),
        ("paid_work_min", "paid work"),
        ("personal_care_min", "personal care"),
        ("household_min", "household and family care"),
    ):
        b, se, r2 = sc.ols_slope(np.log(full.consumption_pps), np.log(full[col]))
        print(f"    elasticity of {label:26s} to consumption per person across country-waves: {b:+.2f} (HC1 se {se:.2f}), R2 {r2:.2f}")
    b, se, r2 = sc.ols_slope(np.log(full.consumption_pps), np.log(per_hour))
    print(f"    elasticity of consumption per free hour to consumption per person: {b:.2f} (se {se:.2f})")
    for wave in (2000, 2010):
        w = full[full.time == wave]
        print(f"    wave {wave}: {len(w)} countries; free time median {w.leisure_min.median():.0f} min/day")
    both = d.dropna(subset=["leisure_min", "consumption_real"]).pivot_table(
        index="geo", columns="time", values=["leisure_min", "consumption_real", "tv_video_min", "entertainment_culture_min"]
    )
    both = both.dropna()
    print(f"Countries surveyed in both waves: {len(both)} ({', '.join(both.index)})")
    dc = both["consumption_real"][2010] / both["consumption_real"][2000] - 1
    dl = both["leisure_min"][2010] - both["leisure_min"][2000]
    print(f"    real consumption per person 2000-2010: median {pct(dc.median())}, range {pct(dc.min())} to {pct(dc.max())}")
    print(f"    free time 2000-2010: median change {dl.median():+.0f} min/day, range {dl.min():+.0f} to {dl.max():+.0f}")
    per_hour_change = (both["consumption_real"][2010] / both["leisure_min"][2010]) / (both["consumption_real"][2000] / both["leisure_min"][2000]) - 1
    print(f"    real consumption per minute of free time 2000-2010: median {pct(per_hour_change.median())}")
    print(f"    countries where free time rose: {int((dl > 0).sum())} of {len(dl)}")
    tvs = full.tv_video_min / full.leisure_min
    print(f"Television and video as a share of free time: median {pct(tvs.median())}, range {pct(tvs.min())} to {pct(tvs.max())}")
    b, se, _ = sc.ols_slope(np.log(full.consumption_pps), tvs)
    print(f"    slope of the TV share on log consumption: {b:+.3f} (se {se:.3f})")
    ec = full.entertainment_culture_min
    print(
        f"Entertainment and culture: median {ec.median():.0f} min/day; shopping and services median {full.shopping_services_min.median():.0f} min/day"
    )
    print(
        f"Sleep median {full.sleep_min.median():.0f} min/day; "
        f"personal care (incl. sleep and eating) median {full.personal_care_min.median():.0f} min/day"
    )


def prices(hicp):
    section("8. Prices: services against all items (HICP)")
    for geo in ("EA", "EU27_2020"):
        x = hicp[hicp.geo == geo].pivot_table(index="time", columns="coicop", values="value").dropna()
        y0, y1 = int(x.index.min()), int(x.index.max())
        s = sc.cagr(x.loc[y0, "SERV"], x.loc[y1, "SERV"], y1 - y0)
        a = sc.cagr(x.loc[y0, "CP00"], x.loc[y1, "CP00"], y1 - y0)
        print(
            f"{geo} {y0}-{y1}: services prices {pct(s, 2)}/yr, all items {pct(a, 2)}/yr; "
            f"relative price of services {pct((1 + s) / (1 + a) - 1, 2)}/yr, "
            f"x{(x.loc[y1, 'SERV'] / x.loc[y0, 'SERV']) / (x.loc[y1, 'CP00'] / x.loc[y0, 'CP00']):.3f} over the period"
        )


def main():
    RESULTS.mkdir(exist_ok=True)
    gva, hours, energy, tus, pc, pop, hicp = load()
    eu_shares(gva, hours, energy)
    s = eu_decomposition(gva, hours, energy)
    productivity_by_industry(gva, hours)
    required_rates(s)
    time_per_person(hours, pop)
    countries(gva, hours, energy)
    time_use(tus, pc)
    prices(hicp)


if __name__ == "__main__":
    main()
