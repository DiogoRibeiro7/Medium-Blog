"""Every number in the article, printed in order, and the tables behind the figures in results/.

python materials_numbers.py > output.txt

Material quantities are in billion tonnes (Gt) for the EU and million tonnes (Mt) for Portugal.
Changes are log points (100 times the natural log of a ratio) unless a percentage is stated.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jsonstat import read_jsonstat
from materials_core import (
    breakeven_growth,
    circular_rate,
    contributions,
    intensity_split,
    log_change,
    material_use_allowed,
    max_window_increase,
    ols_slope,
    pct,
    per_year,
    recycled_from_rate,
    recycled_needed,
    within_slope,
)

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
RESULTS = HERE / "results"
EU = "EU27_2020"
MS = ["BE", "BG", "CZ", "DK", "DE", "EE", "IE", "EL", "ES", "FR", "HR", "IT", "CY", "LV", "LT", "LU", "HU",
      "MT", "NL", "AT", "PL", "PT", "RO", "SI", "SK", "FI", "SE"]  # fmt: skip
GROUPS = {"MF1": "Biomass", "MF2": "Metal ores", "MF3": "Non-metallic minerals", "MF4": "Fossil energy materials"}
Y0, Y1 = 2000, 2024
TARGETS = {"Circular economy action plan (double 2020)": None, "Clean Industrial Deal": 0.24}


def lp(x):
    return f"{100 * x:+.1f}"


def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def load():
    t = {name: read_jsonstat(RAW / f"{name}.json") for name in
         ["env_ac_mfa", "env_ac_rme", "env_ac_rmefd", "env_ac_cur", "env_ac_curm", "env_ac_sd", "env_wastrt",
          "nama_10_gdp", "nama_10_a10", "nama_10_pe"]}  # fmt: skip
    return t


def series(df, **kw):
    x = df
    for k, v in kw.items():
        x = x[x[k] == v]
    return x.set_index("time")["value"].sort_index()


def flags(df, **kw):
    x = df
    for k, v in kw.items():
        x = x[x[k] == v]
    return x.set_index("time")["flag"].sort_index()


def dmc(t, geo, material="TOTAL", unit="THS_T"):
    return series(t["env_ac_mfa"], geo=geo, indic_env="DMC", material=material, unit=unit)


def rmc(t, geo, material="TOTAL", unit="THS_T"):
    return series(t["env_ac_rme"], geo=geo, indic_env="RMC", material=material, unit=unit)


def gdp(t, geo):
    return series(t["nama_10_gdp"], geo=geo, unit="CLV20_MEUR")


def construction(t, geo):
    return series(t["nama_10_a10"], geo=geo, nace_r2="F")


# --------------------------------------------------------------------------------------------- report
def report_headline(t):
    section(f"1. EU material use and GDP, {Y0}-{Y1}")
    d = dmc(t, EU) / 1e6
    y = gdp(t, EU)
    pc = dmc(t, EU, unit="T_HAB")
    m, g, i = intensity_split(d[Y0], d[Y1], y[Y0], y[Y1])
    print(
        f"DMC {d[Y0]:.2f} Gt in {Y0}, {d[Y1]:.2f} Gt in {Y1} ({lp(m)} log points, {pct(m):+.1f}%); "
        f"peak {d.loc[Y0:Y1].idxmax()} at {d.loc[Y0:Y1].max():.2f} Gt"
    )
    print(f"GDP in volume {lp(g)} log points ({pct(g):+.1f}%); material per unit of GDP {lp(i)} ({per_year(i, Y1 - Y0):+.2f}% a year)")
    print(f"DMC per person {pc[Y0]:.1f} t in {Y0}, {pc[Y1]:.1f} t in {Y1}")
    f25 = flags(t["env_ac_mfa"], geo=EU, indic_env="DMC", material="TOTAL", unit="THS_T")[2025]
    print(f"2025 early estimate: {d[2025]:.2f} Gt ({lp(log_change(d[Y1], d[2025]))} on {Y1}), flag '{f25}' (value imputed by Eurostat)")
    print("By period (log points):")
    f = construction(t, EU)
    for a, b in ((2000, 2007), (2007, 2013), (2013, 2024)):
        m, g, i = intensity_split(d[a], d[b], y[a], y[b])
        print(
            f"  {a}-{b}: DMC {lp(m)}  GDP {lp(g)}  intensity {lp(i)} ({per_year(i, b - a):+.2f}% a year)  "
            f"construction value added {lp(log_change(f[a], f[b]))}"
        )

    section("2. Which materials moved")
    rows = []
    for code, name in GROUPS.items():
        s = dmc(t, EU, code) / 1e6
        rows.append({"code": code, "name": name, **{str(yr): s[yr] for yr in (2000, 2007, 2013, 2024)}})
        print(f"  {name:25s} {s[2000]:.2f} Gt (2000)  {s[2007]:.2f} (2007)  {s[2013]:.2f} (2013)  {s[Y1]:.2f} ({Y1})  "
              f"share {Y1} {100 * s[Y1] / d[Y1]:.1f}%  change {Y0}-{Y1} {lp(log_change(s[Y0], s[Y1]))}")  # fmt: skip
    other0 = d[Y0] - sum(r["2000"] for r in rows)
    other1 = d[Y1] - sum(r["2024"] for r in rows)
    print(f"  {'Other products and waste':25s} {other0:.2f} Gt (2000)  {other1:.2f} ({Y1})")
    c = contributions({r["code"]: r["2000"] for r in rows} | {"other": other0}, {r["code"]: r["2024"] for r in rows} | {"other": other1})
    print(
        f"Contribution to the {Y0}-{Y1} change of {d[Y1] - d[Y0]:+.2f} Gt: " + ", ".join(f"{GROUPS.get(k, k)} {100 * v:.0f}%" for k, v in c.items())
    )
    nf = d - dmc(t, EU, "MF4") / 1e6
    print(f"DMC without fossil energy materials: {nf[Y0]:.2f} Gt ({Y0}), {nf[2007]:.2f} (2007), {nf[2013]:.2f} (2013), {nf[Y1]:.2f} ({Y1}); "
          f"change {lp(log_change(nf[Y0], nf[Y1]))}")  # fmt: skip
    mins = dmc(t, EU, "MF3") / 1e6
    print(f"Non-metallic minerals 2000-2007 {lp(log_change(mins[2000], mins[2007]))} ({pct(log_change(mins[2000], mins[2007])):+.0f}%), "
          f"2007-2013 {lp(log_change(mins[2007], mins[2013]))}, 2013-{Y1} {lp(log_change(mins[2013], mins[Y1]))}")  # fmt: skip
    fos = dmc(t, EU, "MF4") / 1e6
    print(f"2019-{Y1}: DMC {d[2019]:.2f} -> {d[Y1]:.2f} Gt ({d[Y1] - d[2019]:+.2f}); fossil energy materials {fos[2019]:.2f} -> {fos[Y1]:.2f} "
          f"({fos[Y1] - fos[2019]:+.2f}), {100 * (fos[Y1] - fos[2019]) / (d[Y1] - d[2019]):.0f}% of the change")  # fmt: skip
    idx = pd.DataFrame(
        {
            "year": range(Y0, Y1 + 1),
            "gdp": [100 * gdp(t, EU)[yy] / gdp(t, EU)[Y0] for yy in range(Y0, Y1 + 1)],
            "dmc": [100 * d[yy] / d[Y0] for yy in range(Y0, Y1 + 1)],
            "dmc_nonfossil": [100 * nf[yy] / nf[Y0] for yy in range(Y0, Y1 + 1)],
            "dmc_fossil": [100 * dmc(t, EU, "MF4")[yy] / dmc(t, EU, "MF4")[Y0] for yy in range(Y0, Y1 + 1)],
            "rmc": [100 * rmc(t, EU)[yy] / rmc(t, EU)[Y0] for yy in range(Y0, Y1 + 1)],
        }
    )
    idx.to_csv(RESULTS / "eu_index.csv", index=False)


def report_drift(t):
    section("3. Drift and growth: yearly changes in material use against yearly changes in GDP")
    d = dmc(t, EU).loc[Y0:Y1]
    y = gdp(t, EU).loc[Y0:Y1]
    nf = (dmc(t, EU) - dmc(t, EU, "MF4")).loc[Y0:Y1]
    fo = dmc(t, EU, "MF4").loc[Y0:Y1]
    dy = np.log(y).diff().dropna()
    print(f"EU27, {Y0 + 1}-{Y1}, {len(dy)} yearly changes; average GDP growth {100 * dy.mean():.2f}% a year")
    eu_rows = {}
    for name, s in (("All materials", d), ("Without fossil energy materials", nf), ("Fossil energy materials", fo)):
        dm = np.log(s).diff().dropna()
        b, se, r2, a = ols_slope(dy, dm)
        eu_rows[name] = (b, a)
        print(f"  {name:32s} response {b:.2f} (s.e. {se:.2f}), R2 {r2:.2f}, drift at zero growth {100 * a:+.2f}% a year, "
              f"flat at {100 * breakeven_growth(a, b):.2f}% growth")  # fmt: skip
    rows = []
    for g in MS:
        df = pd.DataFrame({"d": dmc(t, g), "y": gdp(t, g), "f": construction(t, g), "m3": dmc(t, g, "MF3"), "m4": dmc(t, g, "MF4")}).loc[Y0:Y1]
        ch = np.log(df.where(df > 0)).diff().iloc[1:]  # a non-positive tonnage (Malta's fossil DMC in 2008) has no log change
        ch["geo"] = g
        ch["year"] = ch.index
        rows.append(ch)
    p = pd.concat(rows, ignore_index=True)
    p.to_csv(RESULTS / "panel_changes.csv", index=False)

    def fit(col, on="y"):
        q = p[[col, on, "geo"]].dropna()
        return (*within_slope(q[col], q[on], q.geo), len(q), q.geo.nunique())

    b, se, a, n, k = fit("d")
    print(f"Member states: {k} countries, {n} country-years; country intercepts, errors clustered by country")
    for name, col in (("All materials", "d"), ("Non-metallic minerals", "m3"), ("Fossil energy materials", "m4")):
        b, se, a, n, k = fit(col)
        print(f"  {name:25s} response {b:.2f} (s.e. {se:.2f}), drift at zero growth: mean {100 * a.mean():+.2f}%, "
              f"median {100 * a.median():+.2f}% a year, "
              f"flat at {100 * breakeven_growth(a.mean(), b):.2f}% growth ({n} country-years)")  # fmt: skip
    b, se, a, n, k = fit("m3", on="f")
    print(f"  Non-metallic minerals on construction value added: response {b:.2f} (s.e. {se:.2f})")
    b, se, a, n, k = fit("f")
    print(f"  Construction value added on GDP: response {b:.2f} (s.e. {se:.2f})")
    b_all, a_all = eu_rows["All materials"]
    print("EU material use implied by the EU fit, change per year:")
    for gr in (0.0, 1.0, 1.38, 2.0, 3.0):
        print(f"  GDP growth {gr:.2f}%: {100 * a_all + b_all * gr:+.2f}% a year")


def report_footprint(t):
    section("4. The footprint: raw material consumption (imports counted in the raw materials behind them)")
    d = dmc(t, EU) / 1e6
    r = rmc(t, EU) / 1e6
    print(
        f"EU27 RMC {r[Y0]:.2f} Gt ({Y0}), {r[2008]:.2f} (2008), {r[Y1]:.2f} ({Y1}); "
        f"change {Y0}-{Y1} {lp(log_change(r[Y0], r[Y1]))} against DMC {lp(log_change(d[Y0], d[Y1]))}"
    )
    print(
        f"RMC / DMC: {r[Y0] / d[Y0]:.3f} ({Y0}), {r[Y1] / d[Y1]:.3f} ({Y1}); "
        f"per person RMC {rmc(t, EU, unit='T_HAB')[Y1]:.1f} t, DMC {dmc(t, EU, unit='T_HAB')[Y1]:.1f} t ({Y1})"
    )
    fl = flags(t["env_ac_rme"], geo=EU, indic_env="RMC", material="TOTAL", unit="THS_T")
    print(f"EU27 footprint years flagged 'i' (value imputed by Eurostat): {(fl == 'i').sum()} of {len(fl)}; 2025 early estimate {r[2025]:.2f} Gt")
    for code, name in GROUPS.items():
        dd = dmc(t, EU, code) / 1e6
        rr = rmc(t, EU, code) / 1e6
        print(
            f"  {name:25s} RMC {rr[Y0]:.2f} -> {rr[Y1]:.2f} Gt ({lp(log_change(rr[Y0], rr[Y1]))}); "
            f"DMC {lp(log_change(dd[Y0], dd[Y1]))}; RMC/DMC {Y1} {rr[Y1] / dd[Y1]:.2f}"
        )
    imp = series(t["env_ac_rme"], geo=EU, indic_env="IMP_RME", material="TOTAL", unit="THS_T") / 1e6
    impd = series(t["env_ac_mfa"], geo=EU, indic_env="IMP", material="TOTAL", unit="THS_T") / 1e6
    yy = imp.dropna().index.max()
    print(f"Imports {yy}: {impd[yy]:.2f} Gt as traded, {imp[yy]:.2f} Gt in raw material equivalents (x{imp[yy] / impd[yy]:.2f})")
    rows = []
    for g in MS:
        dg, rg = dmc(t, g), rmc(t, g)
        rows.append({"geo": g, "dmc_change": log_change(dg[2008], dg[Y1]), "rmc_change": log_change(rg[2008], rg[Y1]),
                     "dmc_pc": dmc(t, g, unit="T_HAB")[Y1], "rmc_pc": rmc(t, g, unit="T_HAB")[Y1]})  # fmt: skip
    c = pd.DataFrame(rows).set_index("geo")
    c.to_csv(RESULTS / "countries_footprint.csv")
    less = c[c.rmc_change > c.dmc_change].index.tolist()
    print(f"Member states, 2008-{Y1}: footprint fell less than (or rose more than) DMC in {len(less)} of {len(c)}: {', '.join(less)}")
    print(
        f"  mean change DMC {lp(c.dmc_change.mean())}, RMC {lp(c.rmc_change.mean())}; correlation {np.corrcoef(c.dmc_change, c.rmc_change)[0, 1]:.2f}"
    )
    print(
        f"  Portugal: DMC {lp(c.loc['PT', 'dmc_change'])}, RMC {lp(c.loc['PT', 'rmc_change'])}; "
        f"per person {Y1}: DMC {c.loc['PT', 'dmc_pc']:.1f} t, RMC {c.loc['PT', 'rmc_pc']:.1f} t"
    )
    fd = t["env_ac_rmefd"]
    yf = 2023
    tot = series(fd, geo=EU, indic_env="RMC", material="TOTAL", cpa08="TOTAL")[yf]
    for code, name in (("RMC_P3", "final consumption"), ("RMC_P3_S14", "  of which households"), ("RMC_P3_S13", "  of which government"),
                       ("RMC_P5", "gross capital formation"), ("RMC_P51G", "  of which gross fixed capital formation")):  # fmt: skip
        v = series(fd, geo=EU, indic_env=code, material="TOTAL", cpa08="TOTAL")[yf]
        print(f"  {yf} footprint by final use, {name:40s} {v / 1e6:.2f} Gt ({100 * v / tot:.1f}%)")
    cf = series(fd, geo=EU, indic_env="RMC", material="TOTAL", cpa08="CPA_F")[yf]
    print(f"  {yf} footprint of construction products: {cf / 1e6:.2f} Gt ({100 * cf / tot:.1f}%) of {tot / 1e6:.2f} Gt")


def report_loop(t):
    section("5. The loop: where processed material goes, and how much comes back")
    sd = t["env_ac_sd"]
    for yy in (2024,):
        pm = series(sd, geo=EU, indic_env="DMI_RCV_R_B", material="TOTAL")[yy]
        mu = series(sd, geo=EU, indic_env="MAT_USE", material="TOTAL")[yy]
        ac = series(sd, geo=EU, indic_env="MAT_ACCUM", material="TOTAL")[yy]
        em = series(sd, geo=EU, indic_env="EMIS", material="TOTAL")[yy]
        df_ = series(sd, geo=EU, indic_env="DFLOW", material="TOTAL")[yy]
        ex = series(sd, geo=EU, indic_env="EXP_X_RCV_R", material="TOTAL")[yy] + series(sd, geo=EU, indic_env="EXP_RCV_R", material="TOTAL")[yy]
        print(f"EU27 {yy}: processed material {pm / 1e6:.2f} Gt; material use {mu / 1e6:.2f} ({100 * mu / pm:.1f}%); "
              f"not material use (exported or used for energy) {(pm - mu) / 1e6:.2f} ({100 * (pm - mu) / pm:.1f}%), "
              f"of which exports {ex / 1e6:.2f} Gt")  # fmt: skip
        print(f"  material accumulation {ac / 1e6:.2f} Gt ({100 * ac / pm:.1f}% of processed, {100 * ac / mu:.1f}% of material use); "
              f"emissions {em / 1e6:.2f} Gt; dissipative flows {df_ / 1e6:.2f} Gt")  # fmt: skip
        print(f"  material use not accumulated: {(mu - ac) / 1e6:.2f} Gt ({100 * (mu - ac) / pm:.1f}% of processed)")
    rows = []
    yy = 2023
    for code in ["TOTAL", *GROUPS]:
        pm = series(sd, geo=EU, indic_env="DMI_RCV_R_B", material=code)[yy]
        mu = series(sd, geo=EU, indic_env="MAT_USE", material=code)[yy]
        ac = series(sd, geo=EU, indic_env="MAT_ACCUM", material=code)[yy]
        rows.append(
            {"code": code, "not_material_use": (pm - mu) / pm, "accumulated": ac / pm, "other_material_use": (mu - ac) / pm, "processed_gt": pm / 1e6}
        )
        print(f"  {yy} {GROUPS.get(code, 'All materials'):25s} processed {pm / 1e6:.2f} Gt: not material use {100 * (pm - mu) / pm:.1f}%, "
              f"accumulated {100 * ac / pm:.1f}%, other material use {100 * (mu - ac) / pm:.1f}%")  # fmt: skip
    pd.DataFrame(rows).to_csv(RESULTS / "processed_fates.csv", index=False)
    cur = series(t["env_ac_cur"], geo=EU)
    print(f"Circular material use rate, EU27: {cur[2004]:.1f}% (2004), {cur[2010]:.1f}% (2010), {cur[2020]:.1f}% (2020), {cur[2024]:.1f}% (2024)")
    for code, name in GROUPS.items():
        cm = series(t["env_ac_curm"], geo=EU, material=code)
        print(f"  {name:25s} {cm[2010]:.1f}% (2010), {cm[2024]:.1f}% (2024)")
    d = dmc(t, EU) / 1e6
    u = recycled_from_rate(cur / 100, d)
    print("Recycled tonnage implied by the rate and DMC (U = rate x DMC / (1 - rate)):")
    for a, b in ((2010, 2024), (2010, 2019), (2019, 2024)):
        only_u = circular_rate(u[b], d[a])
        only_d = circular_rate(u[a], d[b])
        print(f"  {a}-{b}: U {u[a]:.3f} -> {u[b]:.3f} Gt ({lp(log_change(u[a], u[b]))}), DMC {d[a]:.2f} -> {d[b]:.2f} Gt; "
              f"rate {cur[a]:.1f}% -> {cur[b]:.1f}%; "
              f"with only U changing {100 * only_u:.2f}%, with only DMC changing {100 * only_d:.2f}%")  # fmt: skip


def report_targets(t):
    section("6. What a target needs: recycling, less material, or both")
    cur = series(t["env_ac_cur"], geo=EU)
    d = dmc(t, EU) / 1e6
    u = recycled_from_rate(cur / 100, d)
    m = d + u
    yy = Y1
    double = 2 * cur[2020] / 100
    print(f"Material use (DMC + recycled) {yy}: {m[yy]:.2f} Gt; recycled {u[yy]:.3f} Gt; rate {cur[yy]:.1f}%")
    print(f"Targets for 2030: double the 2020 rate = {100 * double:.1f}%; Clean Industrial Deal = 24.0%")
    for name, tg in (("double 2020", double), ("24%", 0.24)):
        need = recycled_needed(tg, m[yy])
        allowed = material_use_allowed(tg, u[yy])
        print(f"  {name}: recycled {need:.2f} Gt at {yy} material use (x{need / u[yy]:.2f}, +{need - u[yy]:.2f} Gt); "
              f"or material use {allowed:.2f} Gt at {yy} recycling ({lp(log_change(m[yy], allowed))}, {pct(log_change(m[yy], allowed)):+.0f}%); "
              f"{100 * (tg - cur[yy] / 100) / (2030 - yy):.2f} points a year from {yy}")  # fmt: skip
    per = (cur[2024] - cur[2010]) / 14
    print(f"Observed change 2010-2024: {cur[2024] - cur[2010]:+.1f} points, {per:+.2f} a year; at that pace 2030 gives {cur[2024] + 6 * per:.1f}%")
    best = max_window_increase(cur, 6)
    print(f"EU27's largest increase over any six years: {best[0]:+.1f} points ({best[1]}-{best[1] + 6})")
    rows = []
    for g in MS:
        c = series(t["env_ac_cur"], geo=g)
        b = max_window_increase(c, 6)
        rows.append((g, c.get(2024, np.nan), b))
    rows.sort(key=lambda r: -r[2][0])
    print("Largest six-year increase by member state (top five): " + ", ".join(f"{g} {b[0]:+.1f} ({b[1]}-{b[1] + 6})" for g, _, b in rows[:5]))
    need = 100 * double - cur[2024]
    print(f"Member states that ever added {need:.1f} points or more in six years: {sum(1 for _, _, b in rows if b[0] >= need)} of {len(rows)}")
    lev = sorted(((g, v) for g, v, _ in rows), key=lambda r: r[1])
    print(f"Rates in 2024: highest {lev[-1][0]} {lev[-1][1]:.1f}%, lowest {lev[0][0]} {lev[0][1]:.1f}%, Portugal {dict(lev)['PT']:.1f}%")
    rank = [g for g, _ in lev].index("PT") + 1
    print(f"  Portugal ranks {rank} from the bottom of {len(lev)}; below it: {', '.join(f'{g} {v:.1f}%' for g, v in lev[: rank - 1])}")

    w = t["env_wastrt"]

    def wt(geo, waste, oper):
        return series(w, geo=geo, waste=waste, wst_oper=oper).get(yy, np.nan) / 1e6

    print(f"Waste treatment, EU27 {yy} (Mt): total {wt(EU, 'TOTAL', 'TRT'):.0f}; recycling {wt(EU, 'TOTAL', 'RCV_R'):.0f}, "
          f"backfilling {wt(EU, 'TOTAL', 'RCV_B'):.0f}, "
          f"energy recovery {wt(EU, 'TOTAL', 'RCV_E'):.0f}, landfill {wt(EU, 'TOTAL', 'DSP_L'):.0f}, "
          f"incineration {wt(EU, 'TOTAL', 'DSP_I'):.0f}, other disposal {wt(EU, 'TOTAL', 'DSP_OTH'):.0f}")  # fmt: skip
    cats = {"TOT_X_MIN": "Waste excluding major mineral wastes", "W121": "Construction and demolition mineral waste",
            "W12B": "Other mineral waste (mostly mining)",
            "W126": "Soils", "W127": "Dredging spoils", "W124": "Combustion wastes", "W128_13": "Mineral waste from waste treatment"}  # fmt: skip
    for code, name in cats.items():
        tr, rc, lf = wt(EU, code, "TRT"), wt(EU, code, "RCV_R"), wt(EU, code, "DSP_L")
        print(f"  {name:45s} treated {tr:6.0f}  recycled {rc:5.0f} ({100 * rc / tr:.0f}%)  landfilled {lf:5.0f} ({100 * lf / tr:.0f}%)")
    base = u[yy]
    steps = [
        ("Every tonne of non-mineral waste recycled", wt(EU, "TOT_X_MIN", "TRT") - wt(EU, "TOT_X_MIN", "RCV_R")),
        ("plus all construction and demolition mineral waste", wt(EU, "W121", "TRT") - wt(EU, "W121", "RCV_R")),
        (
            "plus combustion and treatment mineral wastes",
            wt(EU, "W124", "TRT") - wt(EU, "W124", "RCV_R") + wt(EU, "W128_13", "TRT") - wt(EU, "W128_13", "RCV_R"),
        ),
        ("plus soils and dredging spoils", wt(EU, "W126", "TRT") - wt(EU, "W126", "RCV_R") + wt(EU, "W127", "TRT") - wt(EU, "W127", "RCV_R")),
        ("plus mining and other mineral waste: all treated waste", wt(EU, "W12B", "TRT") - wt(EU, "W12B", "RCV_R")),
    ]
    print(
        f"Ceilings, recycled material substituting one for one so that material use stays {m[yy]:.2f} Gt "
        f"(rate now {100 * base / m[yy]:.1f}% on this basis):"
    )
    acc = base
    rows = [{"scenario": "Now", "recycled_gt": base, "rate": base / m[yy]}]
    for name, add in steps:
        acc += add / 1000
        rows.append({"scenario": name, "recycled_gt": acc, "rate": acc / m[yy]})
        print(f"  {name:55s} +{add:5.0f} Mt -> recycled {acc:.2f} Gt, rate {100 * acc / m[yy]:.1f}%")
    sc = pd.DataFrame(rows)
    sc.to_csv(RESULTS / "ceilings.csv", index=False)
    path = pd.DataFrame(
        {"year": range(2010, yy + 1), "material_use": [m[k] for k in range(2010, yy + 1)], "recycled": [u[k] for k in range(2010, yy + 1)]}
    )
    path.to_csv(RESULTS / "eu_path.csv", index=False)
    (RESULTS / "targets.json").write_text(json.dumps({"double_2020": double, "cid": 0.24}))


def report_portugal(t):
    section("7. Portugal")
    d = dmc(t, "PT") / 1e3
    y = gdp(t, "PT")
    m, g, i = intensity_split(d[Y0], d[Y1], y[Y0], y[Y1])
    print(f"DMC {d[Y0]:.0f} Mt ({Y0}), {d[2008]:.0f} (2008), {d[2013]:.0f} (2013), {d[Y1]:.0f} ({Y1}); peak {d.loc[Y0:Y1].idxmax()}")
    print(f"  {Y0}-{Y1}: DMC {lp(m)}, GDP {lp(g)}, intensity {lp(i)}; 2008-2013 DMC {lp(log_change(d[2008], d[2013]))}, construction value added "
          f"{lp(log_change(construction(t, 'PT')[2008], construction(t, 'PT')[2013]))}")  # fmt: skip
    pc = dmc(t, "PT", unit="T_HAB")
    print(f"  DMC per person {pc[Y0]:.1f} t ({Y0}), {pc[2008]:.1f} (2008), {pc[Y1]:.1f} ({Y1}); EU27 {dmc(t, EU, unit='T_HAB')[Y1]:.1f} t")
    for code, name in GROUPS.items():
        s = dmc(t, "PT", code) / 1e3
        print(f"  {name:25s} {s[Y1]:.0f} Mt in {Y1} ({100 * s[Y1] / d[Y1]:.1f}%)")
    r = rmc(t, "PT") / 1e3
    print(f"  RMC {r[2008]:.0f} Mt (2008), {r[Y1]:.0f} ({Y1}); RMC/DMC {r[Y1] / d[Y1]:.2f}")
    cur = series(t["env_ac_cur"], geo="PT")
    print(f"  Circular material use rate {cur.index.min()} {cur.iloc[0]:.1f}%, 2024 {cur[2024]:.1f}%")
    w = t["env_wastrt"]
    for code, name in (("TOTAL", "all waste"), ("TOT_X_MIN", "non-mineral waste"), ("W121", "construction and demolition mineral waste")):
        tr = series(w, geo="PT", waste=code, wst_oper="TRT").get(2024, np.nan) / 1e6
        rc = series(w, geo="PT", waste=code, wst_oper="RCV_R").get(2024, np.nan) / 1e6
        ld = series(w, geo="PT", waste=code, wst_oper="DSP_L").get(2024, np.nan) / 1e6
        print(f"  Waste treated 2024, {name}: {tr:.1f} Mt, recycled {rc:.1f} ({100 * rc / tr:.0f}%), landfilled {ld:.1f} ({100 * ld / tr:.0f}%)")


def main():
    RESULTS.mkdir(exist_ok=True)
    print("eu-material-use: decoupling, footprint and the loop, EU27 and member states")
    print("Data: Eurostat snapshot eurostat-material-use in DiogoRibeiro7/data (see ../data_sources).")
    t = load()
    report_headline(t)
    report_drift(t)
    report_footprint(t)
    report_loop(t)
    report_targets(t)
    report_portugal(t)


if __name__ == "__main__":
    main()
