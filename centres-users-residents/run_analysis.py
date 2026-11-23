"""Every number in the article, printed in order, and the tables in results/.

python panel.py
python run_analysis.py > output.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core import demean, fixed_effects, log_growth, percentile_of, ring, spearman
from panel import EUROPE, LISBOA, METRO, NATIONAL, PORTO

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
# Annual growth beyond these limits signals a boundary change rather than growth.
MAX_POP_GROWTH = 3.5
MAX_DWELLING_GROWTH = 3.5
MIN_DWELLING_GROWTH = -0.5  # a dwelling stock does not shrink by half a per cent a year without a boundary change
MAX_JOB_GROWTH = 8.0
MIN_RING = 1.02  # the FUA must exceed the city by at least 2% in each series


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def pct(a: float, b: float) -> float:
    return 100.0 * (a / b - 1.0)


# ---------------------------------------------------------------------------- Portugal
def units(m: pd.DataFrame, eu: pd.DataFrame) -> None:
    section("1. WHAT COUNTS AS THE CENTRE")
    for core, metro in [(LISBOA, "Lisbon"), (PORTO, "Porto")]:
        d = m[m.metro == metro]
        print(
            f"{d.loc[core, 'name']} municipality: {d.loc[core, 'pop_census_2021']:,.0f} residents in the 2021 census, "
            f"{100 * d.loc[core, 'pop_census_2021'] / d.pop_census_2021.sum():.1f}% of its metro area ({len(d)} municipalities, "
            f"{d.pop_census_2021.sum():,.0f}); {100 * d.loc[core, 'pop_est_2025'] / d.pop_est_2025.sum():.1f}% at the end of 2025"
        )
    for stem in ("PT001", "PT002"):
        r = eu.loc[stem]
        print(
            f"Eurostat {r.city}: {r.P1:,.0f} residents in {int(r.y1)}, {100 * r.P1 / r.PF1:.1f}% of its functional urban area "
            f"({r.PF1:,.0f}); {100 * r.J1 / r.JF1:.1f}% of the area's jobs"
        )


def portugal(m: pd.DataFrame) -> pd.DataFrame:
    section("2. LISBON AND PORTO: USERS AND RESIDENTS (municipalities of the two metro areas)")
    m = m.copy()
    m["users_per_resident"] = 1 + (m.commute_in_share - m.commute_out_share) / 100
    m["day_population"] = m.pop_census_2021 * m.users_per_resident
    m["employees_per_1000_2021"] = 1000 * m.employees_2021 / m.pop_est_2021
    m["employees_per_1000_2024"] = 1000 * m.employees_2024 / m.pop_est_2024
    m["vacant_market_pct"] = 100 * m.dw_vacant_market_2021 / m.dw_total_2021
    m["secondary_pct"] = 100 * m.dw_secondary_2021 / m.dw_total_2021
    m["vacant_other_pct"] = 100 * m.dw_vacant_other_2021 / m.dw_total_2021
    m["price_change_1925"] = 100 * (m.sale_price_2025 / m.sale_price_2019 - 1)
    m["group"] = np.where(m.core, m.name, m.metro + " ring")
    print("Census 2021: people entering and leaving each municipality to work or study, % of its residents")
    for code in (LISBOA, PORTO):
        r = m.loc[code]
        print(
            f"  {r['name']}: in {r.commute_in_share:.2f}%, out {r.commute_out_share:.2f}%; users per resident {r.users_per_resident:.3f}; "
            f"daytime population about {r.day_population:,.0f} for {r.pop_census_2021:,.0f} residents"
        )
    rest = m[~m.core]
    lo, hi = rest.users_per_resident.idxmin(), rest.users_per_resident.idxmax()
    print(
        f"  Ring municipalities: users per resident from {rest.users_per_resident.min():.3f} ({rest.loc[lo, 'name']}) "
        f"to {rest.users_per_resident.max():.3f} ({rest.loc[hi, 'name']}); "
        f"{(rest.users_per_resident < 1).sum()} of {len(rest)} below 1"
    )
    for metro in ("Lisbon", "Porto"):
        rr = rest[rest.metro == metro]
        print(
            f"  {metro} ring, population-weighted: in {np.average(rr.commute_in_share, weights=rr.pop_census_2021):.2f}%, "
            f"out {np.average(rr.commute_out_share, weights=rr.pop_census_2021):.2f}%, users per resident "
            f"{np.average(rr.users_per_resident, weights=rr.pop_census_2021):.3f}"
        )

    print()
    print("Full-time employees on full pay by place of work (Quadros de Pessoal, October) against residents (revised INE estimates, 31 December)")
    print(
        "  Withheld size bands in the 35 municipalities: 2021 "
        + str(int(m.employees_withheld_bands_2021.sum()))
        + f" (in {', '.join(m.name[m.employees_withheld_bands_2021 > 0])}); 2024 {int(m.employees_withheld_bands_2024.sum())}"
    )
    g = m.groupby("group")
    t = pd.DataFrame(
        {
            "residents_2021": g.pop_est_2021.sum(),
            "residents_2024": g.pop_est_2024.sum(),
            "residents_2025": g.pop_est_2025.sum(),
            "employees_2021": g.employees_2021.sum(),
            "employees_2024": g.employees_2024.sum(),
        }
    ).reindex(["Lisboa", "Lisbon ring", "Porto", "Porto ring"])
    t["employees_per_1000_2021"] = 1000 * t.employees_2021 / t.residents_2021
    t["employees_per_1000_2024"] = 1000 * t.employees_2024 / t.residents_2024
    t["employees_change_pct"] = 100 * (t.employees_2024 / t.employees_2021 - 1)
    t["residents_change_pct"] = 100 * (t.residents_2024 / t.residents_2021 - 1)
    t["employees_added"] = t.employees_2024 - t.employees_2021
    t["residents_added"] = t.residents_2024 - t.residents_2021
    t["employees_per_resident_added"] = t.employees_added / t.residents_added
    print(t.round(2).to_string())
    for metro in ("Lisbon", "Porto"):
        d = m[m.metro == metro]
        core = d[d.core].iloc[0]
        print(
            f"  {core['name']}: {100 * core.employees_2021 / d.employees_2021.sum():.1f}% of the metro area's employees in 2021, "
            f"{100 * core.employees_2024 / d.employees_2024.sum():.1f}% in 2024; "
            f"{100 * (core.employees_2024 - core.employees_2021) / (d.employees_2024.sum() - d.employees_2021.sum()):.1f}% of the employees added; "
            f"{100 * core.pop_est_2021 / d.pop_est_2021.sum():.1f}% of residents in 2021, "
            f"{100 * (core.pop_est_2024 - core.pop_est_2021) / (d.pop_est_2024.sum() - d.pop_est_2021.sum()):.1f}% of residents added to 2024"
        )
    t.to_csv(RESULTS / "portugal_cores_rings.csv")
    nat = pd.read_csv(NATIONAL).set_index(["series", "geo", "year"])["value"]
    q21, q24 = nat["employees_quadros_de_pessoal", "PT", 2021], nat["employees_quadros_de_pessoal", "PT", 2024]
    n21, n24 = nat["employment_national_accounts", "PT", 2021], nat["employment_national_accounts", "PT", 2024]
    print(
        f"  Portugal: full-time employees (Quadros de Pessoal) {q21:,.0f} in 2021, {q24:,.0f} in 2024 ({pct(q24, q21):+.2f}%); "
        f"employment in the national accounts {n21:,.1f} thousand to {n24:,.1f} thousand ({pct(n24, n21):+.2f}%)"
    )

    print()
    print("Dwellings completed in new construction for family housing, 2022-2025 (rehabilitation and conversions are not included)")
    m["completions_2224"] = m[["completions_2022", "completions_2023", "completions_2024"]].sum(axis=1)
    m["completions_2225"] = m.completions_2224 + m.completions_2025
    years = [f"completions_{y}" for y in (2022, 2023, 2024, 2025)]
    c = m.groupby("group")[[*years, "completions_2224", "completions_2225"]].sum()
    c = c.reindex(["Lisboa", "Lisbon ring", "Porto", "Porto ring"])
    c["per_1000_residents_a_year_2225"] = 1000 * c.completions_2225 / 4 / t.residents_2021
    c["per_100_employees_added_2224"] = 100 * c.completions_2224 / t.employees_added
    c["per_100_residents_added_2224"] = 100 * c.completions_2224 / t.residents_added
    c["per_1000_dwellings_2021_a_year_2225"] = 1000 * c.completions_2225 / 4 / m.groupby("group").dw_total_2021.sum().reindex(c.index)
    print(c.round(2).to_string())
    for metro in ("Lisbon", "Porto"):
        d = m[m.metro == metro]
        core = d[d.core].iloc[0]
        print(
            f"  {core['name']}: {100 * core.completions_2225 / d.completions_2225.sum():.1f}% of the metro area's completions 2022-2025, "
            f"against {100 * core.pop_est_2021 / d.pop_est_2021.sum():.1f}% of its residents and "
            f"{100 * (core.employees_2024 - core.employees_2021) / (d.employees_2024.sum() - d.employees_2021.sum()):.1f}% of the employees added"
        )
    c.to_csv(RESULTS / "portugal_completions.csv")

    print()
    print("Visitors: bed capacity in tourist accommodation establishments (hotels, local accommodation, rural tourism)")
    for metro in ("Lisbon", "Porto"):
        d = m[m.metro == metro]
        core = d[d.core].iloc[0]
        rr = d[~d.core]
        b23, b25 = core.tourist_beds_2023, core.tourist_beds_2025
        built = core.completions_2023 + core.completions_2024 + core.completions_2025
        print(
            f"  {core['name']}: {b23:,.0f} beds in 2023, {b25:,.0f} in 2025 ({pct(b25, b23):+.1f}%); "
            f"{1000 * b25 / core.pop_est_2025:.0f} per 1,000 residents (ring {1000 * rr.tourist_beds_2025.sum() / rr.pop_est_2025.sum():.0f}); "
            f"{100 * b25 / d.tourist_beds_2025.sum():.1f}% of the metro area's beds; "
            f"beds added 2023-2025 {b25 - b23:,.0f} against {built:,.0f} dwellings completed 2023-2025"
        )

    print()
    print("Services used by the metro area")
    for metro in ("Lisbon", "Porto"):
        d = m[m.metro == metro]
        core = d[d.core].iloc[0]
        print(
            f"  {core['name']}: {100 * core.sns_beds_2024 / d.sns_beds_2024.sum():.1f}% of the metro area's SNS/PPP hospital beds (2024) "
            f"with {100 * core.pop_est_2024 / d.pop_est_2024.sum():.1f}% of its residents; pre-school enrolment rate {core.school_pre_2025:.1f}, "
            f"basic {core.school_basic_2025:.1f} (2024/25)"
        )

    print()
    print("Housing in the cores and the rings (census 2021; prices are INE municipal medians, EUR/m2)")
    rows = {}
    for grp in ["Lisboa", "Lisbon ring", "Porto", "Porto ring"]:
        d = m[m.group == grp]
        rows[grp] = {
            "households_change_1121": np.average(d.households_change_pct, weights=d.households_2021),
            "dwellings_change_1121": np.average(d.dwellings_change_pct, weights=d.dw_total_2021),
            "vacant_market_pct": 100 * d.dw_vacant_market_2021.sum() / d.dw_total_2021.sum(),
            "vacant_other_pct": 100 * d.dw_vacant_other_2021.sum() / d.dw_total_2021.sum(),
            "secondary_pct": 100 * d.dw_secondary_2021.sum() / d.dw_total_2021.sum(),
            "crowded_pct": np.average(d.crowded_share_2021, weights=d.households_2021),
            "rented_pct": 100 * d.rented_2021.sum() / d.households_2021.sum(),
            "price_2019": np.average(d.sale_price_2019, weights=d.pop_census_2021),
            "price_2025": np.average(d.sale_price_2025, weights=d.pop_census_2021),
            "rent_2021": np.average(d.new_rent_2021, weights=d.pop_census_2021),
            "rent_2025": np.average(d.new_rent_2025, weights=d.pop_census_2021),
        }
    h = pd.DataFrame(rows).T
    h["price_change_1925"] = 100 * (h.price_2025 / h.price_2019 - 1)
    h["rent_change_2125"] = 100 * (h.rent_2025 / h.rent_2021 - 1)
    h["not_main_residence_pct"] = h.vacant_market_pct + h.vacant_other_pct + h.secondary_pct
    print(h.round(2).to_string())
    print("  (sale prices EUR/m2; rents of new leases EUR/m2 a month; ring values are population-weighted means of the municipal medians)")
    for core, grp in [("Lisboa", "Lisbon ring"), ("Porto", "Porto ring")]:
        print(
            f"  {core} rent relative to its ring: {h.loc[core, 'rent_2021'] / h.loc[grp, 'rent_2021']:.2f} in 2021, "
            f"{h.loc[core, 'rent_2025'] / h.loc[grp, 'rent_2025']:.2f} in 2025"
        )
    for core, grp in [("Lisboa", "Lisbon ring"), ("Porto", "Porto ring")]:
        print(
            f"  {core} price relative to its ring: {h.loc[core, 'price_2019'] / h.loc[grp, 'price_2019']:.2f} in 2019, "
            f"{h.loc[core, 'price_2025'] / h.loc[grp, 'price_2025']:.2f} in 2025"
        )
    for code in (LISBOA, PORTO):
        r = m.loc[code]
        print(
            f"  {r['name']}: {r.dw_total_2021:,.0f} classic dwellings in 2021; vacant for sale or rent {r.dw_vacant_market_2021:,.0f}, "
            f"vacant for other reasons {r.dw_vacant_other_2021:,.0f}, secondary or seasonal use {r.dw_secondary_2021:,.0f}"
        )
    h.to_csv(RESULTS / "portugal_housing.csv")

    print()
    print("The cores' share of their metro area (%)")
    rows = {}
    for metro in ("Lisbon", "Porto"):
        d = m[m.metro == metro]
        core = d[d.core].iloc[0]

        def share(col: str, core=core, d=d) -> float:
            return 100 * core[col] / d[col].sum()

        rows[core["name"]] = {
            "Dwellings, 2021 census": share("dw_total_2021"),
            "Residents, end 2021": share("pop_est_2021"),
            "Full-time employees, 2021": share("employees_2021"),
            "Full-time employees added, 2021-2024": 100
            * (core.employees_2024 - core.employees_2021)
            / (d.employees_2024.sum() - d.employees_2021.sum()),
            "Public hospital beds, 2024": share("sns_beds_2024"),
            "Tourist beds, 2025": share("tourist_beds_2025"),
            "Dwellings completed, 2022-2025": share("completions_2225"),
        }
    shares = pd.DataFrame(rows)
    print(shares.round(1).to_string())
    shares.to_csv(RESULTS / "core_shares.csv")

    print()
    print("Residents of the cores over time")
    for metro in ("Lisbon", "Porto"):
        d = m[m.metro == metro]
        core = d[d.core].iloc[0]
        rr = d[~d.core]
        print(
            f"  {core['name']}: {pct(core.pop_census_2021, core.pop_est_2011):+.2f}% from Dec 2011 to the 2021 census "
            f"(ring {pct(rr.pop_census_2021.sum(), rr.pop_est_2011.sum()):+.2f}%); "
            f"{pct(core.pop_est_2025, core.pop_est_2021):+.2f}% Dec 2021 to Dec 2025 "
            f"(ring {pct(rr.pop_est_2025.sum(), rr.pop_est_2021.sum()):+.2f}%); "
            f"share of metro residents {100 * core.pop_est_2011 / d.pop_est_2011.sum():.1f}% (end 2011) -> "
            f"{100 * core.pop_census_2021 / d.pop_census_2021.sum():.1f}% (2021 census); "
            f"{100 * core.pop_est_2021 / d.pop_est_2021.sum():.1f}% (end 2021, revised) -> "
            f"{100 * core.pop_est_2025 / d.pop_est_2025.sum():.1f}% (end 2025); "
            f"2024 to 2025 {pct(core.pop_est_2025, core.pop_est_2024):+.2f}% (ring {pct(rr.pop_est_2025.sum(), rr.pop_est_2024.sum()):+.2f}%); "
            f"revised end-2021 estimate {pct(core.pop_est_2021, core.pop_census_2021):+.1f}% above the census "
            f"(ring {pct(rr.pop_est_2021.sum(), rr.pop_census_2021.sum()):+.1f}%)"
        )

    print()
    print("Across the 35 municipalities (Spearman)")
    print(f"  users per resident vs change in dwellings 2011-2021: {spearman(m.users_per_resident, m.dwellings_change_pct):+.3f}")
    print(f"  users per resident vs change in households 2011-2021: {spearman(m.users_per_resident, m.households_change_pct):+.3f}")
    print(f"  users per resident vs sale price 2025: {spearman(m.users_per_resident, m.sale_price_2025):+.3f}")
    print(f"  users per resident vs price change 2019-2025: {spearman(m.users_per_resident, m.price_change_1925):+.3f}")
    m["completions_per_1000"] = 1000 * m.completions_2225 / m.pop_est_2021
    print(f"  users per resident vs dwellings completed 2022-2025 per 1,000 residents: {spearman(m.users_per_resident, m.completions_per_1000):+.3f}")
    print(f"  users per resident vs residents' growth 2021-2025: {spearman(m.users_per_resident, m.pop_est_2025 / m.pop_est_2021):+.3f}")
    growth = m.pop_est_2025 / m.pop_est_2021
    print(f"  dwellings completed per 1,000 residents vs residents' growth 2021-2025: {spearman(m.completions_per_1000, growth):+.3f}")
    m.to_csv(RESULTS / "metro_municipalities.csv")
    return m


# ---------------------------------------------------------------------------- Europe
def europe(eu: pd.DataFrame) -> pd.DataFrame:
    section("3. EUROPE: CENTRES AND RINGS IN THE FUNCTIONAL URBAN AREAS (Eurostat city statistics)")
    e = eu.copy()
    print(f"FUAs with population, jobs and dwellings for city and FUA in a start year 2014-2016 and an end year 2020-2022: {len(e)}")
    has_ring = pd.Series(True, index=e.index)
    for k in ("P", "J", "D"):
        has_ring &= (e[f"{k}F0"] > MIN_RING * e[f"{k}0"]) & (e[f"{k}F1"] > MIN_RING * e[f"{k}1"])
    print(f"  dropped because the city is (almost) the whole FUA in some series: {(~has_ring).sum()}")
    e = e[has_ring].copy()
    yrs = e.y1 - e.y0
    for k in ("P", "J", "D"):
        e[f"g{k}"] = log_growth(e[f"{k}0"], e[f"{k}1"], yrs)
        e[f"g{k}r"] = log_growth(ring(e[f"{k}F0"], e[f"{k}0"]), ring(e[f"{k}F1"], e[f"{k}1"]), yrs)
    sane = (
        (e[["gP", "gPr"]].abs() <= MAX_POP_GROWTH).all(axis=1)
        & e[["gD", "gDr"]].ge(MIN_DWELLING_GROWTH).all(axis=1)
        & e[["gD", "gDr"]].le(MAX_DWELLING_GROWTH).all(axis=1)
        & (e[["gJ", "gJr"]].abs() <= MAX_JOB_GROWTH).all(axis=1)
    )
    print(
        f"  dropped as signs of a boundary change (people moving more than {MAX_POP_GROWTH}% a year, jobs more than {MAX_JOB_GROWTH}%, "
        f"dwellings outside {MIN_DWELLING_GROWTH}% to {MAX_DWELLING_GROWTH}%): {(~sane).sum()}, "
        f"of which German: {(~sane & (e.country == 'DE')).sum()}"
    )
    e = e[sane].copy()
    print(f"  kept: {len(e)} FUAs in {e.country.nunique()} countries: " + ", ".join(f"{c} {n}" for c, n in e.country.value_counts().items()))
    print(f"  years: start {e.y0.value_counts().to_dict()}, end {e.y1.value_counts().to_dict()}")
    e["res_centre_minus_ring"] = e.gP - e.gPr
    e["jobs_centre_minus_ring"] = e.gJ - e.gJr
    e["homes_centre_minus_ring"] = e.gD - e.gDr
    e["jobs_minus_homes_centre"] = e.gJ - e.gD
    e["jobs_minus_homes_ring"] = e.gJr - e.gDr
    e["res_share_change"] = 100 * (e.P1 / e.PF1 - e.P0 / e.PF0) / (e.y1 - e.y0)
    e["jobs_share_change"] = 100 * (e.J1 / e.JF1 - e.J0 / e.JF0) / (e.y1 - e.y0)
    cols = [
        "gP",
        "gPr",
        "gJ",
        "gJr",
        "gD",
        "gDr",
        "res_centre_minus_ring",
        "jobs_centre_minus_ring",
        "homes_centre_minus_ring",
        "jobs_minus_homes_centre",
        "jobs_minus_homes_ring",
    ]

    print()
    print("Median annual growth, % (centre = Eurostat city; ring = rest of the FUA)")
    print(e[cols].median().round(3).to_string())
    two = (e.jobs_share_change > 0) & (e.res_share_change < 0)
    print(
        f"Centres whose share of FUA residents fell: {(e.res_share_change < 0).sum()} of {len(e)} ({100 * (e.res_share_change < 0).mean():.1f}%); "
        f"whose share of FUA jobs fell: {(e.jobs_share_change < 0).sum()} ({100 * (e.jobs_share_change < 0).mean():.1f}%)"
    )
    print(f"Centres that gained a share of jobs and lost a share of residents (both sides at once): {two.sum()} ({100 * two.mean():.1f}%)")
    slow = e.homes_centre_minus_ring < 0
    print(f"Centres where dwellings grew more slowly than in their ring: {slow.sum()} ({100 * slow.mean():.1f}%)")

    print()
    print("Lisbon and Porto (Eurostat greater cities) and their percentile among the kept FUAs")
    for stem in ("PT001", "PT002"):
        r = e.loc[stem]
        print(f"  {r.city} ({int(r.y0)}-{int(r.y1)}):")
        for c in cols + ["res_share_change", "jobs_share_change"]:
            print(f"     {c:<26} {r[c]:+7.3f}   percentile {percentile_of(e[c], r[c]):5.1f}")

    print()
    print("By country (countries with at least 5 FUAs): medians, % a year")
    by = e.groupby("country")
    ct = pd.DataFrame(
        {
            "n": by.size(),
            "jobs_centre": by.gJ.median(),
            "homes_centre": by.gD.median(),
            "residents_centre": by.gP.median(),
            "jobs_minus_homes_centre": by.jobs_minus_homes_centre.median(),
            "res_centre_minus_ring": by.res_centre_minus_ring.median(),
            "homes_centre_minus_ring": by.homes_centre_minus_ring.median(),
        }
    )
    ct = ct[ct.n >= 5].sort_values("jobs_minus_homes_centre", ascending=False)
    print(ct.round(2).to_string())
    pt = e[e.country == "PT"]
    print(
        f"Portuguese centres: jobs grew faster than dwellings in {(pt.jobs_minus_homes_centre > 0).sum()} of {len(pt)}; "
        f"median gap {pt.jobs_minus_homes_centre.median():.2f} points a year "
        f"against {e[e.country != 'PT'].jobs_minus_homes_centre.median():.2f} elsewhere"
    )

    print()
    print("Is the jobs series comparable across countries? FUA jobs (Urban Audit) against employment in the national accounts")
    na = pd.read_csv(NATIONAL)
    na = na[na.series == "employment_national_accounts"].set_index(["geo", "year"])["value"]
    e["gJF"] = log_growth(e.JF0, e.JF1, e.y1 - e.y0)
    e["g_national"] = [100 * np.log(na[c, int(y1)] / na[c, int(y0)]) / (y1 - y0) for c, y0, y1 in zip(e.country, e.y0, e.y1, strict=True)]
    by = e.groupby("country")
    bench = pd.DataFrame(
        {
            "n": by.size(),
            "fua_jobs": by.gJF.median(),
            "national_employment": by.g_national.median(),
            "centre_jobs": by.gJ.median(),
            "centre_dwellings": by.gD.median(),
        }
    )
    bench["fua_jobs_minus_national"] = bench.fua_jobs - bench.national_employment
    bench["national_employment_minus_centre_dwellings"] = bench.national_employment - bench.centre_dwellings
    bench = bench.sort_values("national_employment_minus_centre_dwellings", ascending=False)
    print("  medians across each country's FUAs, % a year (national growth over each FUA's own years)")
    print(bench.round(2).to_string())
    bench.to_csv(RESULTS / "country_benchmark.csv")
    big = bench[bench.n >= 5]
    print(
        f"  Countries with at least 5 FUAs: FUA jobs minus national employment from {big.fua_jobs_minus_national.min():+.2f} "
        f"({big.fua_jobs_minus_national.idxmin()}) to {big.fua_jobs_minus_national.max():+.2f} ({big.fua_jobs_minus_national.idxmax()}); "
        f"national employment minus centre dwellings highest in {', '.join(big.index[:2])}"
    )
    for stem in ("PT001", "PT002"):
        r_ = e.loc[stem]
        print(
            f"  {r_.city}: FUA jobs {r_.gJF:+.2f}% a year against national employment {r_.g_national:+.2f}% ({int(r_.y0)}-{int(r_.y1)}); "
            f"national employment minus centre dwellings {r_.g_national - r_.gD:+.2f} points"
        )

    print()
    print("What goes with a centre keeping its residents? Within-country relationships across FUAs")
    within = {c: demean(e[c], e.country) for c in ("res_centre_minus_ring", "jobs_centre_minus_ring", "homes_centre_minus_ring")}
    print(
        f"  Spearman, centre-minus-ring growth of dwellings vs of residents (within country): "
        f"{spearman(within['homes_centre_minus_ring'], within['res_centre_minus_ring']):+.3f}"
    )
    print(
        f"  Spearman, centre-minus-ring growth of jobs vs of residents (within country): "
        f"{spearman(within['jobs_centre_minus_ring'], within['res_centre_minus_ring']):+.3f}"
    )
    rows = []
    strict = (e[["gP", "gPr", "gD", "gDr"]].abs() <= 2).all(axis=1) & (e[["gJ", "gJr"]].abs() <= 5).all(axis=1)
    for name, sub in [
        ("all", e),
        ("without Portugal", e[e.country != "PT"]),
        ("FUAs over 500,000", e[e.PF1 >= 500_000]),
        ("without Germany", e[e.country != "DE"]),
        ("all series within 2% a year (jobs 5%)", e[strict]),
    ]:
        b, se, r2, g = fixed_effects(sub.res_centre_minus_ring, sub[["jobs_centre_minus_ring", "homes_centre_minus_ring"]], sub.country)
        print(
            f"  Country fixed effects, residents (centre minus ring) on jobs and dwellings (centre minus ring), {name}: n {len(sub)}, "
            f"countries {g}; jobs {b[0]:+.3f} ({se[0]:.3f}), dwellings {b[1]:+.3f} ({se[1]:.3f}); within R2 {r2:.3f}"
        )
        rows.append({"sample": name, "n": len(sub), "countries": g, "jobs": b[0], "jobs_se": se[0], "homes": b[1], "homes_se": se[1], "r2": r2})
    loo = []
    for c in sorted(e.country.unique()):
        sub = e[e.country != c]
        b, *_ = fixed_effects(sub.res_centre_minus_ring, sub[["jobs_centre_minus_ring", "homes_centre_minus_ring"]], sub.country)
        loo.append((c, b[0], b[1]))
    lo_ = min(loo, key=lambda t: t[2])
    hi_ = max(loo, key=lambda t: t[2])
    print(
        f"  Leaving out one country at a time: dwellings coefficient from {lo_[2]:+.3f} (without {lo_[0]}) to {hi_[2]:+.3f} (without {hi_[0]}); "
        f"jobs coefficient from {min(t[1] for t in loo):+.3f} to {max(t[1] for t in loo):+.3f}"
    )
    pd.DataFrame(rows).to_csv(RESULTS / "europe_fixed_effects.csv", index=False)

    print()
    print("Accounting: residents = dwellings x residents per dwelling, so the centre-minus-ring gap in residents' growth is the")
    print("dwellings gap plus the gap in residents per dwelling. Share of the cross-FUA variance carried by each (within country):")
    e["ppd_centre_minus_ring"] = e.res_centre_minus_ring - e.homes_centre_minus_ring
    rd = demean(e.res_centre_minus_ring, e.country)
    for name, col in [("dwellings", "homes_centre_minus_ring"), ("residents per dwelling", "ppd_centre_minus_ring")]:
        print(f"  {name}: {np.cov(demean(e[col], e.country), rd)[0, 1] / rd.var():.3f}")
    parts = [("dwellings", "homes_centre_minus_ring"), ("residents per dwelling", "ppd_centre_minus_ring"), ("residents", "res_centre_minus_ring")]
    for name, col in parts:
        b, se, r2, g = fixed_effects(e[col], e[["jobs_centre_minus_ring"]], e.country)
        print(f"  Country fixed effects, {name} (centre minus ring) on jobs (centre minus ring): {b[0]:+.3f} ({se[0]:.3f}), within R2 {r2:.3f}")
    for stem in ("PT001", "PT002"):
        r_ = e.loc[stem]
        print(
            f"  {r_.city}: residents gap {r_.res_centre_minus_ring:+.3f} = dwellings gap {r_.homes_centre_minus_ring:+.3f} "
            f"+ residents-per-dwelling gap {r_.ppd_centre_minus_ring:+.3f}"
        )
    q = pd.qcut(e.homes_centre_minus_ring, 4, labels=False)
    print(
        "  Residents, centre minus ring (% a year), by quarter of the dwellings gap (slowest to fastest centre building):  "
        + ", ".join(f"{v:+.2f}" for v in e.res_centre_minus_ring.groupby(q).median())
    )

    print()
    print("Capitals and the largest FUAs in the sample (FUA population at the end year over 1.5 million)")
    big = e[e.PF1 >= 1_500_000].sort_values("PF1", ascending=False)
    print(big[["city", "y0", "y1", "PF1", "gP", "gPr", "gJ", "gJr", "gD", "gDr", "jobs_minus_homes_centre"]].round(2).to_string())
    e.to_csv(RESULTS / "europe_fuas.csv")
    return e


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 30)
    eu = pd.read_csv(EUROPE).set_index("stem")
    m = pd.read_csv(METRO, dtype={"dico": str, "nuts3": str}).set_index("dico")
    print("Centres that attract users and lose residents: Lisbon and Porto against Europe. Every number in the article.")
    print("Sources: Eurostat city statistics (Urban Audit); INE, Statistics Portugal (see sources.py)")
    units(m, eu)
    portugal(m)
    europe(eu)


if __name__ == "__main__":
    main()
