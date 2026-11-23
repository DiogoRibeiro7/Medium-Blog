"""Every number in the article, printed in order, and the tables in results/.

python panel.py
python run_analysis.py > output.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core import (
    composite,
    dimension_scores,
    gabaix_ibragimov,
    leave_group_out_r2,
    ols_hc1,
    rank_under_random_weights,
    spearman,
    weighted_median,
)
from panel import GROUPS, PANEL

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEED = 2026
DRAWS = 5000
DWELLING_M2 = 90  # dwelling size used in the price-to-income ratio

DIMENSIONS = {
    "housing": ["crowded_share_2021", "neg_vacancy_market", "log_price_to_income_2024"],
    "schools": ["neg_school_pre_2025", "neg_school_basic_2025"],
    "transport": ["commute_min_collective", "commute_out_share"],
    "health": ["neg_sns_beds_per_1000_nuts3"],
}


def pct(a: float, b: float) -> float:
    return 100.0 * (a / b - 1.0)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def load() -> pd.DataFrame:
    p = pd.read_csv(PANEL, dtype={"dico": str, "nuts3": str}).set_index("dico")
    p["growth_1121"] = 100 * np.log(p["pop_census_2021"] / p["pop_est_2011"]) / 9.3  # 31 Dec 2011 to 19 Apr 2021
    p["growth_2125"] = 100 * np.log(p["pop_est_2025"] / p["pop_est_2021"]) / 4.0
    p["vacancy_market"] = 100 * p["dw_vacant_market_2021"] / p["dw_total_2021"]
    p["vacancy_other"] = 100 * p["dw_vacant_other_2021"] / p["dw_total_2021"]
    p["rented_share"] = 100 * p["rented_2021"] / p["households_2021"]
    p["price_to_income_2024"] = p["sale_price_2024"] * DWELLING_M2 / p["income_2024"]
    p["links_per_1000"] = 1000 * (p["pairs_lisboa"] + p["pairs_porto"]) / p["pop_census_2021"]
    beds = p.groupby("nuts3")["sns_beds_2024"].transform("sum")
    pop = p.groupby("nuts3")["pop_est_2024"].transform("sum")
    p["sns_beds_per_1000_nuts3"] = 1000 * beds / pop
    p["has_hospital"] = (p["hospitals_2024"] > 0).astype(float)
    return p


def population(p: pd.DataFrame) -> pd.DataFrame:
    section("1. WHERE THE PEOPLE WENT")
    tot = p[["pop_est_2011", "pop_census_2021", "pop_est_2021", "pop_est_2025"]].sum()
    print(f"Residents, 31 Dec 2011 estimate:   {tot.pop_est_2011:>12,.0f}")
    print(f"Residents, census 19 Apr 2021:     {tot.pop_census_2021:>12,.0f}  ({pct(tot.pop_census_2021, tot.pop_est_2011):+.2f}%)")
    print(f"Residents, 31 Dec 2021 estimate:   {tot.pop_est_2021:>12,.0f}  (revised series)")
    print(f"Residents, 31 Dec 2025 estimate:   {tot.pop_est_2025:>12,.0f}  ({pct(tot.pop_est_2025, tot.pop_est_2021):+.2f}% since Dec 2021)")
    gap = tot.pop_est_2021 - tot.pop_census_2021
    print(f"Revised Dec 2021 estimate minus 2021 census: {gap:,.0f} ({100 * gap / tot.pop_census_2021:.2f}% of the census)")

    g = p.groupby("group")
    t = pd.DataFrame(
        {
            "municipalities": g.size(),
            "pop_2011": g["pop_est_2011"].sum(),
            "census_2021": g["pop_census_2021"].sum(),
            "est_2021": g["pop_est_2021"].sum(),
            "est_2025": g["pop_est_2025"].sum(),
        }
    ).reindex(GROUPS)
    t["change_1121_pct"] = 100 * (t.census_2021 / t.pop_2011 - 1)
    t["change_2125_pct"] = 100 * (t.est_2025 / t.est_2021 - 1)
    t["change_2125"] = t.est_2025 - t.est_2021
    t["share_of_growth_2125"] = 100 * t.change_2125 / t.change_2125.sum()
    t["share_of_pop_2021"] = 100 * t.est_2021 / t.est_2021.sum()
    t["revision"] = t.est_2021 - t.census_2021
    t["revision_pct_of_census"] = 100 * t.revision / t.census_2021
    t["share_of_revision"] = 100 * t.revision / t.revision.sum()
    print()
    print("By group (Lisboa and Porto are the two core municipalities; rings are the rest of each metro area)")
    cols = [
        "municipalities",
        "census_2021",
        "change_1121_pct",
        "est_2025",
        "change_2125_pct",
        "share_of_pop_2021",
        "share_of_growth_2125",
        "revision_pct_of_census",
        "share_of_revision",
    ]
    print(t[cols].round(2).to_string())

    metro_l = t.loc[["Lisboa", "Lisbon metro ring"]].sum()
    metro_p = t.loc[["Porto", "Porto metro ring"]].sum()
    for name, m in [("Lisbon metro area", metro_l), ("Porto metro area", metro_p)]:
        print(
            f"{name}: {pct(m.census_2021, m.pop_2011):+.2f}% 2011-2021; {pct(m.est_2025, m.est_2021):+.2f}% 2021-2025 "
            f"({m.change_2125:,.0f} residents, {100 * m.change_2125 / t.change_2125.sum():.1f}% of national growth); "
            f"{100 * m.est_2021 / t.est_2021.sum():.1f}% of residents in 2021"
        )
    both = metro_l + metro_p
    print(
        f"Both metro areas: {100 * both.change_2125 / t.change_2125.sum():.1f}% of national growth 2021-2025, "
        f"{100 * both.est_2021 / t.est_2021.sum():.1f}% of residents in 2021, "
        f"{100 * both.revision / t.revision.sum():.1f}% of the census-to-estimate revision"
    )
    nat = p.groupby("group").apply(lambda d: (d.births_2225 - d.deaths_2225).sum()).reindex(GROUPS)
    print()
    print("Change Dec 2021-Dec 2025 split into natural change (births minus deaths, 2022-2025) and the rest (net migration)")
    for gname in GROUPS:
        ch = t.loc[gname, "change_2125"]
        print(f"  {gname:<18} change {ch:>9,.0f}; natural {nat[gname]:>9,.0f}; net migration {ch - nat[gname]:>9,.0f}")
    print(
        f"  {'Portugal':<18} change {t.change_2125.sum():>9,.0f}; natural {nat.sum():>9,.0f}; net migration {t.change_2125.sum() - nat.sum():>9,.0f}"
    )
    up = (p["pop_est_2025"] > p["pop_est_2021"]).sum()
    up_dec = (p["pop_census_2021"] > p["pop_est_2011"]).sum()
    print(f"Municipalities that grew: {up_dec} of 308 between 2011 and the 2021 census; {up} of 308 between Dec 2021 and Dec 2025")

    big = p.nlargest(5, "pop_est_2025")[["name", "pop_census_2021", "pop_est_2021", "pop_est_2025"]]
    big["revision_pct"] = 100 * (big.pop_est_2021 / big.pop_census_2021 - 1)
    big["change_2125_pct"] = 100 * (big.pop_est_2025 / big.pop_est_2021 - 1)
    print()
    print("Five largest municipalities in 2025")
    print(big.round(2).to_string())

    print()
    print("Rank-size (Zipf) exponent zeta, Gabaix-Ibragimov estimator, standard error in brackets")
    rows = []
    for col, lab in [("pop_est_2011", "2011 estimate"), ("pop_census_2021", "2021 census"), ("pop_est_2025", "2025 estimate")]:
        line = []
        for n in (50, 100, 308):
            z, se = gabaix_ibragimov(p[col].to_numpy(), n)
            rows.append({"population": lab, "n": n, "zeta": z, "se": se})
            line.append(f"top {n}: {z:.3f} ({se:.3f})")
        print(f"  {lab:<14} " + "; ".join(line))
    pd.DataFrame(rows).to_csv(RESULTS / "zipf.csv", index=False)

    print()
    print("Gibrat test on mainland municipalities: growth (% a year) on log initial size, HC1 errors")
    m = p[p["mainland"]]
    out = []
    for gcol, scol, lab in [("growth_1121", "pop_est_2011", "2011-2021"), ("growth_2125", "pop_est_2021", "2021-2025")]:
        b, se, r2 = ols_hc1(m[gcol].to_numpy(), np.log(m[scol].to_numpy()))
        dec = pd.qcut(np.log(m[scol]), 10, labels=False)
        means = m[gcol].groupby(dec.to_numpy()).mean()
        print(f"  {lab}: slope {b[1]:.3f} (s.e. {se[1]:.3f}) points a year per log point of size, R2 {r2:.3f}")
        print("     mean growth by size decile, smallest to largest: " + ", ".join(f"{v:.2f}" for v in means))
        for d, v in means.items():
            out.append({"period": lab, "decile": int(d) + 1, "mean_growth_pct_year": v})
    pd.DataFrame(out).to_csv(RESULTS / "gibrat_deciles.csv", index=False)
    print(f"  Correlation of 2011-2021 and 2021-2025 growth across mainland municipalities: {np.corrcoef(m.growth_1121, m.growth_2125)[0, 1]:.3f}")
    t.to_csv(RESULTS / "population_by_group.csv")
    return t


def infrastructure(p: pd.DataFrame) -> None:
    section("2. DID THE INFRASTRUCTURE FOLLOW?")
    rows = {}
    for gname in GROUPS:
        d = p[p["group"] == gname]
        w_pop, w_kids = d["pop_census_2021"], d["children_0_14_2021"]
        rows[gname] = {
            "dwellings_change_pct": np.average(d.dwellings_change_pct, weights=d.dw_total_2021),
            "households_change_pct": np.average(d.households_change_pct, weights=d.households_2021),
            "vacant_market_pct": 100 * d.dw_vacant_market_2021.sum() / d.dw_total_2021.sum(),
            "vacant_other_pct": 100 * d.dw_vacant_other_2021.sum() / d.dw_total_2021.sum(),
            "crowded_pct": np.average(d.crowded_share_2021, weights=d.households_2021),
            "rented_pct": 100 * d.rented_2021.sum() / d.households_2021.sum(),
            "price_2019": weighted_median(d.sale_price_2019, w_pop),
            "price_2025": weighted_median(d.sale_price_2025, w_pop),
            "price_to_income_2024": weighted_median(d.price_to_income_2024, w_pop),
            "pre_2012": np.average(d.school_pre_2012, weights=w_kids),
            "pre_2025": np.average(d.school_pre_2025, weights=w_kids),
            "basic_2012": np.average(d.school_basic_2012, weights=w_kids),
            "basic_2025": np.average(d.school_basic_2025, weights=w_kids),
            "commute_collective_min": np.average(d.commute_min_collective, weights=w_pop),
            "commute_individual_min": np.average(d.commute_min_individual, weights=w_pop),
            "commute_out_pct": np.average(d.commute_out_share, weights=w_pop),
            "links_per_1000": 1000 * (d.pairs_lisboa + d.pairs_porto).sum() / w_pop.sum(),
            "sns_beds_per_1000": 1000 * d.sns_beds_2024.sum() / d.pop_est_2024.sum(),
            "pop_without_hospital_pct": 100 * d.pop_est_2024[d.hospitals_2024 == 0].sum() / d.pop_est_2024.sum(),
            "growth_2125_pct": 100 * (d.pop_est_2025.sum() / d.pop_est_2021.sum() - 1),
        }
    t = pd.DataFrame(rows).T
    t["price_change_1925_pct"] = 100 * (t.price_2025 / t.price_2019 - 1)
    t.to_csv(RESULTS / "infrastructure_by_group.csv")
    print("Housing (census 2021; prices are population-weighted medians of municipal medians, EUR/m2)")
    print(
        t[
            [
                "households_change_pct",
                "dwellings_change_pct",
                "vacant_market_pct",
                "vacant_other_pct",
                "crowded_pct",
                "rented_pct",
                "price_2019",
                "price_2025",
                "price_change_1925_pct",
                "price_to_income_2024",
            ]
        ]
        .round(2)
        .to_string()
    )
    print(
        f"(price-to-income: median sale price of a {DWELLING_M2} m2 dwelling in 2024 over the median declared income net of IRS of a tax household)"
    )
    print()
    print("Schools: gross enrolment rate, pupils enrolled in the municipality per 100 resident children of that age (child-weighted)")
    print(t[["pre_2012", "pre_2025", "basic_2012", "basic_2025"]].round(1).to_string())
    ring = p[p["group"].isin(["Lisbon metro ring", "Porto metro ring"])]
    print(
        f"Ring municipalities with fewer pre-school places than resident children (rate < 100): "
        f"{(ring.school_pre_2012 < 100).sum()} of {len(ring)} in 2011/12, {(ring.school_pre_2025 < 100).sum()} in 2024/25"
    )
    lr = p[p["group"] == "Lisbon metro ring"]
    print(
        f"Lisbon ring municipalities with rate < 100 in 2024/25: {(lr.school_pre_2025 < 100).sum()} of {len(lr)}; "
        f"lowest: " + ", ".join(f"{r.name_} {r.v:.1f}" for r in lr.assign(name_=lr["name"], v=lr.school_pre_2025).nsmallest(4, "v").itertuples())
    )
    print()
    print("Transport (census 2021, population-weighted) and health (SNS and PPP hospital beds, 2024)")
    print(
        t[
            [
                "commute_collective_min",
                "commute_individual_min",
                "commute_out_pct",
                "links_per_1000",
                "sns_beds_per_1000",
                "pop_without_hospital_pct",
            ]
        ]
        .round(2)
        .to_string()
    )
    worst = p[p["mainland"]].nlargest(5, "commute_min_collective")[["name", "group", "commute_min_collective", "commute_out_share"]]
    print("Longest mean public-transport commutes:")
    print(worst.round(1).to_string())
    n3 = p[p["mainland"]].groupby("nuts3")["sns_beds_per_1000_nuts3"].first()
    print(
        f"SNS beds per 1000 residents across mainland NUTS III regions: min {n3.min():.2f}, median {n3.median():.2f}, max {n3.max():.2f}; "
        f"Grande Lisboa {n3['1A0']:.2f}, Peninsula de Setubal {n3['1B0']:.2f}, Area Metropolitana do Porto {n3['11A']:.2f}"
    )
    print("Beds are counted where INE locates each hospital entity; a centre with several sites can appear under one municipality.")
    print(
        "Municipalities with a hospital (public or private) but no SNS/PPP beds recorded there: "
        + ", ".join(p[(p.hospitals_2024 > 0) & (p.sns_beds_2024 == 0) & p.mainland & (p.group != "Rest of mainland")]["name"])
        + f" (and {((p.hospitals_2024 > 0) & (p.sns_beds_2024 == 0) & (p.group == 'Rest of mainland')).sum()} in the rest of the mainland)"
    )
    for name, groups in [
        ("Lisbon metro area", ["Lisboa", "Lisbon metro ring"]),
        ("Porto metro area", ["Porto", "Porto metro ring"]),
        ("Medium cities", ["Medium cities"]),
        ("Rest of mainland", ["Rest of mainland"]),
    ]:
        d = p[p.group.isin(groups)]
        print(
            f"  SNS/PPP beds per 1000 residents (2024), {name}: {1000 * d.sns_beds_2024.sum() / d.pop_est_2024.sum():.2f}; "
            f"residents per bed {d.pop_est_2024.sum() / d.sns_beds_2024.sum():.0f}"
        )


def saturation(p: pd.DataFrame) -> pd.DataFrame:
    section("3. A SATURATION INDEX AND HOW MUCH IT DEPENDS ON ITS WEIGHTS")
    m = p[p["mainland"]].copy()
    ind = pd.DataFrame(
        {
            "crowded_share_2021": m.crowded_share_2021,
            "neg_vacancy_market": -m.vacancy_market,
            "log_price_to_income_2024": np.log(m.price_to_income_2024),
            "neg_school_pre_2025": -m.school_pre_2025,
            "neg_school_basic_2025": -m.school_basic_2025,
            "commute_min_collective": m.commute_min_collective,
            "commute_out_share": m.commute_out_share,
            "neg_sns_beds_per_1000_nuts3": -m.sns_beds_per_1000_nuts3,
        }
    )
    print(f"Mainland municipalities: {len(m)}; indicators missing: " + ", ".join(f"{c} {n}" for c, n in ind.isna().sum().items() if n))
    scores = dimension_scores(ind, DIMENSIONS)
    m["saturation"] = composite(scores)
    m = m.join(scores)
    m["rank"] = m["saturation"].rank(ascending=False).astype(int)
    m["binding"] = scores.idxmax(axis=1)
    ru = rank_under_random_weights(scores, DRAWS, SEED)
    m = m.join(ru)

    print()
    print("Mean dimension scores and composite by group (z units; higher = more pressure)")
    gm = m.groupby("group")[["housing", "schools", "transport", "health", "saturation"]].mean().reindex(GROUPS[:-1])
    print(gm.round(2).to_string())
    print()
    print(f"Thirty most saturated municipalities (equal weights), with the 5-95% rank range over {DRAWS} random weightings")
    top = m.nsmallest(30, "rank")[
        ["name", "group", "saturation", "housing", "schools", "transport", "health", "rank_p05", "rank_median", "rank_p95", "p_top30", "binding"]
    ]
    print(top.round(2).to_string())
    print("Groups among the 30 most saturated:", top["group"].value_counts().to_dict())
    robust = m[m["p_top30"] >= 0.9]
    print(f"Municipalities in the top 30 under at least 90% of weightings: {len(robust)}: " + ", ".join(robust.sort_values("rank")["name"]))
    print(
        f"Lisboa: rank {m.loc['1106', 'rank']} (range {m.loc['1106', 'rank_p05']:.0f}-{m.loc['1106', 'rank_p95']:.0f}); "
        f"Porto: rank {m.loc['1312', 'rank']} (range {m.loc['1312', 'rank_p05']:.0f}-{m.loc['1312', 'rank_p95']:.0f})"
    )

    print()
    print("Robustness of the ranking")
    pr = ind.rank(pct=True)
    pr_scores = pd.DataFrame({d: pr[c].mean(axis=1) for d, c in DIMENSIONS.items()})
    print(f"  Spearman, z-score index vs percentile-rank index: {spearman(m.saturation, composite(pr_scores)):.3f}")
    for d in DIMENSIONS:
        rest = scores.drop(columns=d)
        print(f"  Spearman with the index that drops {d:<9}: {spearman(m.saturation, composite(rest)):.3f}")
    print(
        "  Dimension correlations (Spearman): "
        + "; ".join(f"{a}-{b} {spearman(scores[a], scores[b]):.2f}" for i, a in enumerate(DIMENSIONS) for b in list(DIMENSIONS)[i + 1 :])
    )

    print()
    print("Saturation and growth (mainland)")
    print(f"  Spearman, saturation index vs growth 2021-2025: {spearman(m.saturation, m.growth_2125):.3f}")
    print(f"  Spearman, saturation index vs growth 2011-2021: {spearman(m.saturation, m.growth_1121):.3f}")
    q = pd.qcut(m.saturation, 5, labels=False)
    gq = m.groupby(q).apply(lambda d: 100 * (d.pop_est_2025.sum() / d.pop_est_2021.sum() - 1))
    shr = m.groupby(q).apply(lambda d: d.pop_est_2025.sum() - d.pop_est_2021.sum())
    print("  Growth 2021-2025 (%) by saturation quintile, least to most saturated: " + ", ".join(f"{v:.2f}" for v in gq))
    print("  Share of mainland growth 2021-2025 by quintile (%): " + ", ".join(f"{100 * v / shr.sum():.1f}" for v in shr))

    print()
    print("Beyond the official boundary: municipalities outside both metro areas with 50+ census work/study links")
    print("to Lisboa or Porto per 1000 residents")
    belt = m[(~m["metro"]) & (m["links_per_1000"] >= 50)].sort_values("links_per_1000", ascending=False)
    print(belt[["name", "group", "links_per_1000", "growth_2125", "saturation", "rank", "binding"]].round(2).to_string())
    print(
        f"  {len(belt)} municipalities; mean saturation {belt.saturation.mean():.2f} against {m.saturation.mean():.2f} for the mainland; "
        f"growth Dec 2021-Dec 2025 {pct(belt.pop_est_2025.sum(), belt.pop_est_2021.sum()):.2f}%"
    )
    m.to_csv(RESULTS / "saturation.csv")
    return m


def model(m: pd.DataFrame) -> None:
    section("4. WHAT PREDICTS WHERE PEOPLE WENT AFTER 2021")
    d = m.copy()
    d["log_pop"] = np.log(d.pop_census_2021)
    d["log_price"] = np.log(d.sale_price_2021)
    d["log_income"] = np.log(d.income_2021)
    d["lisbon_metro"] = d.group.isin(["Lisboa", "Lisbon metro ring"]).astype(float)
    d["porto_metro"] = d.group.isin(["Porto", "Porto metro ring"]).astype(float)
    d["medium_city"] = (d.group == "Medium cities").astype(float)
    cont = ["log_pop", "housing", "schools", "transport", "health", "log_price", "log_income"]
    d = d.dropna(subset=cont + ["growth_2125"])
    print(f"Outcome: growth of residents, Dec 2021 to Dec 2025 (% a year). Mainland municipalities with prices and incomes: {len(d)}")
    print("Continuous predictors are standardised; coefficients are % a year per standard deviation. HC1 standard errors.")
    print("Out-of-sample R2: each NUTS III region is predicted from a model fitted on the other 22 (no region dummies in those models).")
    pressure = ["housing", "schools", "transport", "health"]
    specs = {
        "A size": (["log_pop"], True),
        "B size + pressure in 2021": (["log_pop"] + pressure, True),
        "C B + prices and incomes": (["log_pop"] + pressure + ["log_price", "log_income"], True),
        "D C + metro and medium-city dummies": (
            ["log_pop"] + pressure + ["log_price", "log_income", "lisbon_metro", "porto_metro", "medium_city"],
            False,
        ),
    }
    rows = []
    y = d["growth_2125"].to_numpy()
    groups = d["nuts3"].to_numpy()
    for name, (cols, do_cv) in specs.items():
        X = d[cols].copy()
        for c in cols:
            if c in cont:
                X[c] = (X[c] - X[c].mean()) / X[c].std(ddof=0)
        b, se, r2 = ols_hc1(y, X.to_numpy())
        cv = leave_group_out_r2(y, X.to_numpy(), groups) if do_cv else float("nan")
        cv_txt = f"; leave-one-NUTS-III-out R2 {cv:.3f}" if do_cv else " (in-sample only)"
        print(f"\n  [{name}]  R2 {r2:.3f}{cv_txt}")
        for c, bb, ss in zip(cols, b[1:], se[1:], strict=False):
            print(f"     {c:<14} {bb:+.3f} ({ss:.3f})")
            rows.append({"model": name, "term": c, "coef": bb, "se": ss, "r2": r2, "cv_r2": cv})
    pd.DataFrame(rows).to_csv(RESULTS / "growth_model.csv", index=False)
    print()
    print("Simple correlations with growth 2021-2025 (Pearson, same municipalities):")
    for c in [
        "log_pop",
        "links_per_1000",
        "vacancy_market",
        "crowded_share_2021",
        "log_price",
        "price_to_income_2024",
        "commute_min_collective",
        "school_pre_2025",
        "saturation",
    ]:
        print(f"     {c:<24} {np.corrcoef(d[c], d.growth_2125)[0, 1]:+.3f}")


def candidates(p: pd.DataFrame, m: pd.DataFrame) -> None:
    section("5. CANDIDATE NODES FOR A POLYCENTRIC NETWORK")
    hh_size = p.pop_census_2021.sum() / p.households_2021.sum()
    metro = p[p["metro"]]
    metro_growth = metro.pop_est_2025.sum() - metro.pop_est_2021.sum()
    med = m[m["group"] == "Medium cities"]
    vac_med = med.dw_vacant_market_2021.sum()
    vac_med_other = med.dw_vacant_other_2021.sum()
    print(f"Average household size, census 2021: {hh_size:.3f}")
    print(
        f"Growth of the two metro areas, Dec 2021-Dec 2025: {metro_growth:,.0f} residents, "
        f"about {metro_growth / hh_size:,.0f} households at that size"
    )
    print(
        f"Medium cities, census 2021: {vac_med:,.0f} dwellings vacant for sale or rent, {vac_med_other:,.0f} vacant for other reasons; "
        f"{med.dw_total_2021.sum():,.0f} classic dwellings in all"
    )
    print(f"Vacant-for-market dwellings in medium cities per metro household added 2021-2025: {vac_med / (metro_growth / hh_size):.2f}")
    print(
        f"Medium cities: {100 * med.pop_est_2021.sum() / p.pop_est_2021.sum():.1f}% of residents in 2021, growth 2021-2025 "
        f"{pct(med.pop_est_2025.sum(), med.pop_est_2021.sum()):.2f}% against {pct(p.pop_est_2025.sum(), p.pop_est_2021.sum()):.2f}% nationally"
    )

    cols = [
        "name",
        "pop_est_2025",
        "growth_2125",
        "saturation",
        "rank",
        "rank_p05",
        "rank_p95",
        "housing",
        "schools",
        "transport",
        "health",
        "binding",
        "dw_vacant_market_2021",
        "links_per_1000",
        "has_hospital",
    ]
    print()
    print("Medium cities (outside the metro areas, 50,000+ residents in 2021), least saturated first; rank 1 = most saturated of 278")
    print(med.sort_values("saturation")[cols].round(2).to_string())
    print("Binding constraint (dimension with the highest score) among medium cities:", med["binding"].value_counts().to_dict())
    ring = m[m["group"].isin(["Lisbon metro ring", "Porto metro ring"])]
    print()
    print("Metro ring municipalities, most saturated first")
    print(ring.sort_values("saturation", ascending=False)[["group"] + cols].round(2).to_string())
    print("Binding constraint among ring municipalities:", ring["binding"].value_counts().to_dict())
    pd.concat([med.assign(role="medium city"), ring.assign(role="metro ring")])[["role"] + cols].to_csv(RESULTS / "candidate_nodes.csv")

    print()
    print("Headroom with a link: medium cities below the mainland median saturation")
    mid = m["saturation"].median()
    low = med[med.saturation < mid].sort_values("saturation")
    print(f"  {len(low)} of {len(med)}: " + ", ".join(f"{r.name} (rank {r.rank}, binding {r.binding})" for r in low.itertuples()))
    print(
        f"  They held {low.pop_est_2021.sum():,.0f} residents in 2021, grew {pct(low.pop_est_2025.sum(), low.pop_est_2021.sum()):.2f}% to 2025, "
        f"and had {low.dw_vacant_market_2021.sum():,.0f} dwellings vacant for sale or rent"
    )


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    p = load()
    print("Portugal's saturated metro areas and the polycentric alternative: every number in the article")
    print("Source: INE, Statistics Portugal (see sources.py); municipalities in NUTS 2024 geography")
    population(p)
    infrastructure(p)
    m = saturation(p)
    model(m)
    candidates(p, m)


if __name__ == "__main__":
    main()
