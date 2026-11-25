"""Every number in the article, printed in the order the article uses them, and the tables in results/.

python legacy_numbers.py > output.txt

Reads data/raw/ (Eurostat snapshot from the data registry) and data/external/ (pinned public files); run fetch.py first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rdata

import legacy_core as lc
from jsonstat import read_jsonstat

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
EXT = HERE / "data" / "external"
RESULTS = HERE / "results"

REGIME = (1926, 1974)  # military coup of 28 May 1926 to the revolution of 25 April 1974
LONG_RUN_COUNTRIES = ["PRT", "ESP", "GRC", "ITA", "IRL", "FIN", "POL", "HUN", "CZE", "SVN", "LTU", "BGR", "ROU"]
LONG_RUN_YEARS = [1900, 1913, 1925, 1929, 1938, 1950, 1960, 1973, 1985, 1990, 2000, 2007, 2015, 2022]
PERIODS = [(1900, 1925), (1925, 1950), (1950, 1960), (1960, 1973), (1973, 1985), (1985, 2000), (2000, 2022)]
# Long-run schooling (Barro and Lee 2015, Education Matters), European countries; Luxembourg is left out because its series implies 0.3 years in 1925.
EUROPE_LONG = ["AUT", "BEL", "BGR", "DNK", "FIN", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "NLD", "NOR", "POL", "PRT", "ESP", "SWE", "CHE", "GBR", "ALB", "TUR", "RUS"]  # fmt: skip
COMMUNIST_RULED = ["BGR", "HUN", "POL", "RUS", "ALB"]  # ruled by communist parties for most of 1945-1975
GAIN_WINDOW = (1925, 1975)
ACCOUNTING_YEARS = (1970, 1990, 2019)
ACCOUNTING_COUNTRIES = ["PRT", "ESP", "GRC", "ITA", "IRL", "CZE", "SVK", "SVN", "POL", "HUN", "EST", "LVA", "LTU", "ROU", "BGR", "HRV"]  # fmt: skip
EXCLUDED_FROM_FIT = ["LU", "IE"]  # GDP per head distorted by cross-border workers and multinational accounting
BOOT_DRAWS, BOOT_SEED = 5000, 20261125


def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def fmt(x, d=1):
    return "n/a" if not np.isfinite(x) else f"{x:.{d}f}"


def eurostat(name):
    return read_jsonstat(RAW / f"{name}.json")


def main():
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)

    # ------------------------------------------------------------------------------------------------ long-run income
    section("1. GDP per head relative to twelve Western European countries (Maddison Project Database 2023, via OWID), percent")
    owid = pd.read_csv(EXT / "owid-co2-data.csv", usecols=["iso_code", "year", "gdp", "population"])
    owid = owid[owid["iso_code"].notna() & ~owid["iso_code"].astype(str).str.startswith("OWID")]
    gdp = owid.pivot_table(index="year", columns="iso_code", values="gdp")
    pop = owid.pivot_table(index="year", columns="iso_code", values="population")
    rel = lc.relative_to_group(gdp, pop, lc.W12)
    rel.to_csv(RESULTS / "relative_gdp_longrun.csv")
    print("reference: aggregate GDP per head of " + ", ".join(lc.W12) + "; years with all twelve: "
          f"{int(rel.index[rel['PRT'].notna() & rel['FRA'].notna()].min())} onward")  # fmt: skip
    table = rel.loc[[y for y in LONG_RUN_YEARS if y in rel.index], LONG_RUN_COUNTRIES]
    print(table.round(0).to_string())
    pt = rel["PRT"].dropna()
    print(f"\nPortugal: {fmt(pt.get(1925), 0)}% in 1925, {fmt(pt.get(1950), 0)}% in 1950, {fmt(pt.get(1960), 0)}% in 1960, "
          f"{fmt(pt.get(1973), 0)}% in 1973, {fmt(pt.get(2000), 0)}% in 2000, {fmt(pt.get(2022), 0)}% in 2022")  # fmt: skip
    print(
        f"Portugal's lowest relative level 1926-1959: {fmt(pt.loc[1926:1959].min(), 0)}% ({int(pt.loc[1926:1959].idxmin())}); highest after 1974: {fmt(pt.loc[1974:].max(), 0)}% ({int(pt.loc[1974:].idxmax())})"
    )
    print("average annual change of the relative level, percent a year:")
    for a, b in PERIODS:
        cells = "  ".join(f"{c} {fmt(lc.annual_change(rel[c].dropna(), a, b), 2)}" for c in ["PRT", "ESP", "GRC", "POL", "HUN"])
        print(f"   {a}-{b}: {cells}")
    per_capita = (gdp / pop)["PRT"]
    print(
        f"Portugal's real GDP per head grew {fmt(lc.annual_change(per_capita, 1960, 1973), 2)}% a year in 1960-1973 and {fmt(lc.annual_change(per_capita, 1926, 1950), 2)}% in 1926-1950"
    )

    # ------------------------------------------------------------------------------------------------ long-run schooling
    section("2. Adult years of schooling, ages 25-64 (Barro and Lee 2015 long-run series, 1870-2010)")
    oup = pd.read_csv(EXT / "OUP_long_MF2564_v1.csv")
    sch = oup[oup["WBcode"].isin(EUROPE_LONG)].pivot_table(index="WBcode", columns="year", values="yr_sch")
    sch.to_csv(RESULTS / "schooling_longrun.csv")
    show = [1900, 1925, 1950, 1970, 1975, 1990, 2010]
    print(
        sch.loc[[c for c in ["PRT", "ESP", "GRC", "ITA", "FIN", "IRL", "BGR", "POL", "HUN", "FRA", "DEU", "SWE"] if c in sch.index], show]
        .round(1)
        .to_string()
    )
    for y in (1900, 1925, 1975, 2010):
        r = sch[y].rank(ascending=True)
        print(
            f"{y}: Portugal {fmt(sch.loc['PRT', y], 2)} years, rank {int(r['PRT'])} from the bottom of {sch[y].notna().sum()} (lowest: {sch[y].idxmin()} {fmt(sch[y].min(), 2)})"
        )
    a, b = GAIN_WINDOW
    cu = lc.catch_up_residuals(sch[a], sch[b])
    cu.to_csv(RESULTS / "schooling_catch_up.csv")
    print(f"\ngain {a}-{b} regressed on the {a} level, {len(cu)} countries: gain = {cu.attrs['intercept']:.3f} {cu.attrs['slope']:+.3f} x level")
    print(cu.sort_values("residual").round(2).to_string())
    p = cu.loc["PRT"]
    print(f"Portugal: {fmt(p.start, 2)} -> {fmt(p.end, 2)} years, gain {fmt(p.gain, 2)}, predicted {fmt(p.predicted_gain, 2)}, residual {fmt(p.residual, 2)}; "
          f"rank {int(cu['residual'].rank()['PRT'])} of {len(cu)} from the bottom")  # fmt: skip
    print(
        f"Spain residual {fmt(cu.loc['ESP', 'residual'], 2)}; Greece {fmt(cu.loc['GRC', 'residual'], 2)}; Italy {fmt(cu.loc['ITA', 'residual'], 2)}"
    )
    com = cu.loc[[c for c in COMMUNIST_RULED if c in cu.index], "residual"]
    other = cu.drop(index=list(com.index) + ["PRT", "ESP"])["residual"]
    print(
        f"mean residual: communist-ruled ({', '.join(com.index)}) {fmt(com.mean(), 2)}; other countries excluding Portugal and Spain {fmt(other.mean(), 2)}"
    )
    hc_effect = lc.human_capital_ratio(p.end - p.residual, p.end)
    print(f"human capital per worker with the predicted gain instead of the actual one, {b}: x{float(hc_effect):.3f} (PWT returns)")
    peers = cu[(cu["start"] >= 2.0) & (cu["start"] <= 3.5)]
    print(
        f"countries starting between 2.0 and 3.5 years in {a}: {', '.join(peers.index)}; their mean {b} level {fmt(peers['end'].mean(), 2)}, Portugal {fmt(p.end, 2)}"
    )
    for label, gain in (
        ("its predicted gain", p.predicted_gain),
        ("Russia's gain", cu.loc["RUS", "gain"]),
        ("Albania's gain", cu.loc["ALB", "gain"]),
    ):
        print(
            f"Portugal in {b} with {label} ({fmt(gain, 2)} years): {fmt(p.start + gain, 2)} years; human capital x{float(lc.human_capital_ratio(p.start + gain, p.end)):.3f}"
        )
    gap_a, gap_b = peers["start"].mean() - p.start, peers["end"].mean() - p.end
    print(
        f"their mean {a} level {fmt(peers['start'].mean(), 2)}; gap to Portugal {fmt(gap_a, 2)} years in {a}, {fmt(gap_b, 2)} in {b} (x{gap_b / gap_a:.2f})"
    )

    # ------------------------------------------------------------------------------------------------ Barro-Lee 1950-2015
    section("3. Adult years of schooling, ages 25-64, EU members (Barro and Lee v3, 1950-2015)")
    bl = pd.read_csv(EXT / "BL_v3_MF2564.csv")
    eu3 = [lc.ISO2_TO_ISO3[g] for g in lc.EU27_ISO2]
    blw = bl[bl["WBcode"].isin(eu3)].pivot_table(index="WBcode", columns="year", values="yr_sch")
    blw.to_csv(RESULTS / "schooling_eu_1950_2015.csv")
    for y in (1950, 1970, 1975, 1990, 2015):
        col = blw[y].dropna()
        print(
            f"{y}: Portugal {fmt(col['PRT'], 2)}; EU members' median {fmt(col.median(), 2)}; lowest {col.idxmin()}; next lowest {col.drop('PRT').idxmin()} {fmt(col.drop('PRT').min(), 2)}; Portugal rank {int(col.rank()['PRT'])} of {len(col)}"
        )
    lowest_every = all(blw[y].dropna().idxmin() == "PRT" for y in blw.columns)
    print(f"Portugal lowest of the EU members in every year {int(blw.columns.min())}-{int(blw.columns.max())}: {lowest_every}")
    pair = blw[[1975, 2015]].dropna()
    print(
        f"rank correlation of members' levels, 1975 against 2015: {pair[1975].rank().corr(pair[2015].rank()):.2f} (n = {len(pair)}); "
        f"1950 against 2015: {blw[1950].rank().corr(blw[2015].rank()):.2f}"
    )
    level = blw.loc["PRT", 2015]
    reach = {
        c: lc.first_year_reaching(blw.loc[c], level)
        for c in ["CZE", "SVK", "EST", "HUN", "POL", "SVN", "LVA", "LTU", "ROU", "BGR", "HRV", "GRC", "ESP", "ITA"]
    }
    print(
        f"year each country first reached Portugal's 2015 level ({fmt(level, 2)}): "
        + ", ".join(f"{c} {int(v) if np.isfinite(v) else 'not yet'}" for c, v in sorted(reach.items(), key=lambda kv: kv[1]))
    )

    # ------------------------------------------------------------------------------------------------ Eurostat GDP per head
    section("4. GDP per head in purchasing power standards, EU27 = 100 (Eurostat nama_10_pc)")
    na = eurostat("nama_10_pc")
    gpc = na[(na["unit"] == "PC_EU27_2020_HAB_MPPS_CP") & (na["na_item"] == "B1GQ")].pivot_table(index="geo", columns="time", values="value")
    aic = na[(na["unit"] == "PC_EU27_2020_HAB_MPPS_CP") & (na["na_item"] == "P41")].pivot_table(index="geo", columns="time", values="value")
    gpc.to_csv(RESULTS / "gdp_per_head_pps.csv")
    members = [g for g in lc.EU27_ISO2 if g in gpc.index]
    last = int(gpc.columns.max())
    for y in (1995, 2000, 2008, 2015, last):
        col = gpc.loc[members, y].dropna()
        print(
            f"{y}: Portugal {fmt(col['PT'], 1)}; rank {int(col.rank(ascending=False)['PT'])} of {len(col)}; members below Portugal: {(col < col['PT']).sum()}"
        )
    first = 1995
    below_then = gpc.loc[members, first] < gpc.loc["PT", first]
    above_now = gpc.loc[members, last] > gpc.loc["PT", last]
    passed = [g for g in members if below_then[g] and above_now[g]]
    print(
        f"members below Portugal in {first} and above it in {last}: "
        + ", ".join(f"{g} {fmt(gpc.loc[g, first], 0)}->{fmt(gpc.loc[g, last], 1)}" for g in passed)
    )
    for g in ("PL", "EE", "RO", "HR", "HU", "SK", "ES", "EL"):
        print(f"   {g}: {fmt(gpc.loc[g, first], 0)} -> {fmt(gpc.loc[g, last], 1)}")
    col = aic.loc[members, last].dropna()
    print(f"actual individual consumption per head {last}: Portugal {fmt(col['PT'], 1)}, rank {int(col.rank(ascending=False)['PT'])} of {len(col)}")
    lp = eurostat("nama_10_lp_ulc").pivot_table(index=["na_item", "geo"], columns="time", values="value")
    for item, label in (("NLPR_PER", "per person employed"), ("NLPR_HW", "per hour worked")):
        tab = lp.loc[item]
        y = int(tab.columns[tab.loc[members].notna().sum() >= 25].max())
        col = tab.loc[members, y].dropna()
        print(
            f"labour productivity {label}, PPS, EU27 = 100, {y}: Portugal {fmt(col['PT'], 1)}; rank {int(col.rank(ascending=False)['PT'])} of {len(col)}; "
            f"lowest {col.idxmin()} {fmt(col.min(), 1)}; below Portugal: {', '.join(f'{g} {fmt(v, 1)}' for g, v in col[col < col['PT']].sort_values().items())}"
        )

    # ------------------------------------------------------------------------------------------------ attainment by age
    section("5. Share of people with at most lower secondary education (ISCED 0-2), by age (Eurostat edat_lfse_03)")
    ed = eurostat("edat_lfse_03")
    low = ed[ed["isced11"] == "ED0-2"].pivot_table(index=["geo", "age"], columns="time", values="value")
    low.to_csv(RESULTS / "low_education_by_age.csv")
    last_ed = int(low.columns.max())
    ages = ["Y25-34", "Y35-44", "Y45-54", "Y55-64", "Y25-64"]
    for g in ("PT", "EU27_2020", "ES", "IT", "EL", "MT", "CZ", "PL", "SI", "LT"):
        print(f"{g:10s} {last_ed}: " + "  ".join(f"{a[1:]} {fmt(low.loc[(g, a), last_ed], 1)}" for a in ages if (g, a) in low.index))
    for a in ages:
        col = low.xs(a, level="age")[last_ed].reindex(members).dropna()
        print(
            f"   {a}: Portugal rank {int(col.rank(ascending=False)['PT'])} of {len(col)} (1 = highest share); EU27 {fmt(low.loc[('EU27_2020', a), last_ed], 1)}"
        )
    for y in (2000, 2010, last_ed):
        print(
            f"Portugal 25-64: {y} {fmt(low.loc[('PT', 'Y25-64'), y], 1)}%; 25-34: {fmt(low.loc[('PT', 'Y25-34'), y], 1)}%; 55-64: {fmt(low.loc[('PT', 'Y55-64'), y], 1)}%"
        )
    print(
        f"Portugal 25-34 in 2000 (born 1966-1975): {fmt(low.loc[('PT', 'Y25-34'), 2000], 1)}%; the same cohort aged 45-54 in 2020: {fmt(low.loc[('PT', 'Y45-54'), 2020], 1)}%"
    )

    section("6. Managers (ISCO-08 group 1) with at most lower secondary education, ages 20-64 (Eurostat lfsa_egised)")
    eg = eurostat("lfsa_egised")
    m = eg[eg["isco08"] == "OC1"].pivot_table(index=["geo", "isced11"], columns="time", values="value")
    share = (m.xs("ED0-2", level="isced11") / m.xs("TOTAL", level="isced11") * 100).dropna(how="all")
    allw = eg[eg["isco08"] == "TOTAL"].pivot_table(index=["geo", "isced11"], columns="time", values="value")
    share_all = allw.xs("ED0-2", level="isced11") / allw.xs("TOTAL", level="isced11") * 100
    share.to_csv(RESULTS / "managers_low_education.csv")
    last_m = int(share.columns.max())
    for g in ("PT", "EU27_2020", "ES", "IT", "EL", "MT", "DE", "FR", "CZ", "PL", "SI"):
        if g in share.index:
            print(
                f"{g:10s} managers: 2011 {fmt(share.loc[g].get(2011, np.nan), 1)}%  {last_m} {fmt(share.loc[g].get(last_m, np.nan), 1)}%   all employed: 2011 {fmt(share_all.loc[g].get(2011, np.nan), 1)}%  {last_m} {fmt(share_all.loc[g].get(last_m, np.nan), 1)}%"
            )
    col = share[last_m].reindex(members).dropna()
    print(f"{last_m}: Portugal rank {int(col.rank(ascending=False)['PT'])} of {len(col)} members with published data (1 = highest share)")

    # ------------------------------------------------------------------------------------------------ development accounting
    section("7. Development accounting, output per worker relative to the twelve Western European countries (Penn World Table 10.01)")
    pwt = rdata.read_rda(str(EXT / "pwt10.01.rda"))["pwt10.01"]
    pwt["isocode"] = pwt["isocode"].astype(str)
    pwt = pwt[["isocode", "year", "cgdpo", "emp", "cn", "hc"]].dropna()
    pwt["y"] = pwt["cgdpo"] / pwt["emp"]
    pwt["ky"] = pwt["cn"] / pwt["cgdpo"]
    rows = []
    for year in ACCOUNTING_YEARS:
        t = pwt[pwt["year"] == year].set_index("isocode")
        ref = t.loc[[c for c in lc.W12 if c in t.index]]
        y_ref, ky_ref, h_ref = (lc.geometric_mean(ref[v]) for v in ("y", "ky", "hc"))
        for c in ACCOUNTING_COUNTRIES:
            if c in t.index:
                r = lc.accounting(t.loc[c, "y"], t.loc[c, "ky"], t.loc[c, "hc"], y_ref, ky_ref, h_ref)
                rows.append({"year": year, "country": c, "hc": t.loc[c, "hc"], "hc_ref": h_ref, **r})
    acc = pd.DataFrame(rows)
    acc.to_csv(RESULTS / "development_accounting.csv", index=False)
    for year in ACCOUNTING_YEARS:
        r = acc[(acc["year"] == year) & (acc["country"] == "PRT")].iloc[0]
        print(f"{year}: Portugal output per worker {100 * r.relative_output:.1f}% of the reference; log gap {r.total:+.3f} = capital {r.capital:+.3f} + human capital {r.human_capital:+.3f} + productivity {r.productivity:+.3f}; "
              f"hc {r.hc:.3f} against {r.hc_ref:.3f}; with the reference's human capital {100 * r.relative_output_with_ref_h:.1f}%")  # fmt: skip
    a19 = acc[acc["year"] == 2019].set_index("country")
    print("\n2019, all comparison countries (percent of the reference; human-capital term in logs):")
    print(
        (a19[["relative_output", "relative_output_with_ref_h"]] * 100)
        .round(1)
        .join(a19[["hc", "human_capital", "productivity"]].round(3))
        .sort_values("hc")
        .to_string()
    )
    print(
        f"Portugal's hc is the lowest of the {len(a19)} comparison countries in 2019: {a19['hc'].idxmin() == 'PRT'}; next lowest {a19['hc'].drop('PRT').idxmin()} {a19['hc'].drop('PRT').min():.3f}"
    )
    ratio_cz = a19.loc["CZE", "hc"] / a19.loc["PRT", "hc"]
    ratio_es = a19.loc["ESP", "hc"] / a19.loc["PRT", "hc"]
    print(
        f"Portugal with Czechia's human capital: output per worker x{ratio_cz:.3f} -> {100 * a19.loc['PRT', 'relative_output'] * ratio_cz:.1f}% of the reference (Czechia {100 * a19.loc['CZE', 'relative_output']:.1f}%)"
    )
    print(
        f"Portugal with Spain's human capital: x{ratio_es:.3f} -> {100 * a19.loc['PRT', 'relative_output'] * ratio_es:.1f}% (Spain {100 * a19.loc['ESP', 'relative_output']:.1f}%)"
    )
    share_h = a19.loc["PRT", "human_capital"] / a19.loc["PRT", "total"]
    print(
        f"human-capital term as a share of Portugal's 2019 log gap: {100 * share_h:.0f}% (capital {100 * a19.loc['PRT', 'capital'] / a19.loc['PRT', 'total']:.0f}%, productivity {100 * a19.loc['PRT', 'productivity'] / a19.loc['PRT', 'total']:.0f}%)"
    )

    for k in (0.5, 0.75, 1.0):
        hterm = k * a19.loc["PRT", "human_capital"]
        print(
            f"returns to schooling x{k}: human-capital term {hterm:+.3f}, {100 * hterm / a19.loc['PRT', 'total']:.0f}% of the log gap; "
            f"with the reference's human capital {100 * a19.loc['PRT', 'relative_output'] * np.exp(-hterm):.1f}%"
        )
    a90 = acc[acc["year"] == 1990].set_index("country")
    print(f"change in Portugal's human-capital term 1990-2019: {a90.loc['PRT', 'human_capital']:+.3f} -> {a19.loc['PRT', 'human_capital']:+.3f}")

    # ------------------------------------------------------------------------------------------------ convergence since 1995
    section("8. Convergence 1995-2025: growth of relative GDP per head on its 1995 level and 1995 adult schooling")
    s95 = pd.Series({g: blw.loc[lc.ISO2_TO_ISO3[g], 1995] for g in members if lc.ISO2_TO_ISO3[g] in blw.index})
    use = [g for g in s95.index if g not in EXCLUDED_FROM_FIT]
    fit = lc.convergence_fit(gpc.loc[use, 1995], gpc.loc[use, last], s95[use])
    b = fit["beta"]
    boot = lc.bootstrap_convergence(fit["data"], BOOT_DRAWS, BOOT_SEED)
    lo, hi = np.percentile(boot[:, 2], [5, 95])
    print(f"n = {len(fit['data'])} (excluding {', '.join(EXCLUDED_FROM_FIT)}); b(ln initial) = {b[1]:.3f}; b(schooling) = {b[2]:.4f} per year of schooling over {last - 1995} years, 90% bootstrap interval {lo:.4f} to {hi:.4f}; "
          f"share of draws above zero {100 * (boot[:, 2] > 0).mean():.1f}%")  # fmt: skip
    resid = fit["residual"]
    r2 = 1 - resid.var() / np.log(fit["data"]["y1"] / fit["data"]["y0"]).var()
    print(f"R2 {r2:.3f}")
    y0, s_pt = gpc.loc["PT", 1995], s95["PT"]
    med = float(s95[use].median())
    print(
        f"Portugal: 1995 level {fmt(y0, 1)}, schooling {s_pt:.2f} (lowest of {len(use)}: {s95[use].idxmin() == 'PT'}); fitted {last} level {lc.predict_level(b, y0, s_pt):.1f}; actual {fmt(gpc.loc['PT', last], 1)}"
    )
    for label, s in (("median schooling", med), ("Spain's", s95["ES"]), ("Czechia's", s95["CZ"])):
        draws = np.array([lc.predict_level(bb, y0, s) - lc.predict_level(bb, y0, s_pt) for bb in boot])
        print(
            f"with {label} ({s:.2f} years): {lc.predict_level(b, y0, s):.1f}, i.e. {lc.predict_level(b, y0, s) - lc.predict_level(b, y0, s_pt):+.1f} points; 90% interval of the difference {np.percentile(draws, 5):+.1f} to {np.percentile(draws, 95):+.1f}"
        )
    out = fit["data"].assign(fitted=fit["fitted_level"], residual=resid)
    out.to_csv(RESULTS / "convergence_fit.csv")


if __name__ == "__main__":
    main()
