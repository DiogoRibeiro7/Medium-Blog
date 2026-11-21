"""Every number in the article, printed in the order the article uses them.

    python growth_numbers.py > output.txt

Reads only data/ (built by build_data.py) and writes the tables behind the figures to results/.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import growth_core as gc

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
MIN_POP = 1_000_000  # countries below this population are left out of the cross-country work
CAPE_NOW = 40.0  # autumn 2026, reported by CFA Institute (Enterprising Investor, 18 November 2026); not in the pinned data


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# ================================================================================================
def firm_that_does_not_grow():
    section("1. A company that does not grow")
    cf, r = 3.0, 0.08
    v_flat = gc.perpetuity_value(cf, r)
    print(f"Flat $3B a year for ever at r = 8%: value = ${v_flat:.1f}B")
    price = 75.0
    g = gc.implied_growth(price, cf, r)
    print(f"Price $75B, cash flow $3B, r = 8%: implied perpetual growth = {pct(g)}")
    print(f"Cash yield at $75B: {pct(cf / price)}")
    print(f"If growth expectations fall to zero, price moves from $75B to ${v_flat:.1f}B: {pct(v_flat / price - 1, 1)}")
    for infl in (0.02, 0.03):
        print(f"Inflation {pct(infl, 0)}: real value of a flat nominal $3B halves in {gc.halving_time(infl):.1f} years")
    print(f"Real value of flat nominal $3B after 20 years at 2% inflation: ${cf / 1.02**20:.2f}B (in today's dollars)")
    print(
        f"Nominal growth needed just to stand still in real terms at 2% inflation: {pct(0.02, 0)}; "
        f"flat-real value at r = 8% nominal and g = 2%: ${gc.perpetuity_value(cf, r, 0.02):.1f}B"
    )


def where_the_value_sits():
    section("2. Where the value of a perpetuity sits in time")
    rows = []
    cases = [("flat, r=8%", 0.08, 0.0), ("4% growth, r=8%", 0.08, 0.04), ("6% growth, r=8%", 0.08, 0.06)]
    for label, r, g in cases:
        half = gc.years_to_share(0.5, r, g)
        tail10 = gc.years_to_share(0.10, r, g)
        print(f"{label:18s} half of the value arrives after year {half:5.1f}; the last 10% after year {tail10:6.1f}")
        for n in (10, 30, 50, 100, 200):
            s = gc.share_beyond(n, r, g)
            print(f"    share of value beyond year {n:3d}: {pct(s, 1)}")
        for n in range(0, 201):
            rows.append({"case": label, "r": r, "g": g, "year": n, "share_beyond": gc.share_beyond(n, r, g)})
    pd.DataFrame(rows).to_csv(RESULTS / "value_by_horizon.csv", index=False, float_format="%.6g")


def firm_inside_its_market():
    section("3. A firm inside its market")
    for s0, gf, gm in [(0.05, 0.10, 0.03), (0.01, 0.15, 0.03), (0.20, 0.08, 0.04)]:
        y = gc.years_to_saturate(s0, gf, gm)
        print(f"share {pct(s0, 0)}, firm {pct(gf, 0)}/yr, market {pct(gm, 0)}/yr: 100% of the market after {y:.1f} years")
    s0, gf, ge = 0.01, 0.07, 0.04
    y = gc.years_to_saturate(s0, gf, ge)
    print(f"firm at 1% of the economy growing 7%/yr in an economy growing 4%/yr (nominal): equals the economy in {y:.0f} years")

    s0, k, gm = 0.05, 0.12, 0.03
    t = np.arange(0, 81)
    share = gc.logistic_share(t, s0, k)
    growth = gc.firm_growth_path(t, s0, k, gm)
    print(f"Logistic share from {pct(s0, 0)} at rate k = {k}, market growth {pct(gm, 0)}:")
    for yr in (1, 10, 20, 30, 40, 60, 80):
        print(f"    year {yr:2d}: share {pct(share[yr], 1):>6s}, firm growth {pct(growth[yr - 1], 2)}")
    years_above = int(np.sum(growth > gm + 0.01))
    print(f"    years with firm growth more than 1 point above the market: {years_above}")
    pd.DataFrame({"year": t[1:], "share": share[1:], "firm_growth": growth, "market_growth": gm}).to_csv(
        RESULTS / "logistic_path.csv", index=False, float_format="%.6g"
    )


def annual_sp500():
    s = pd.read_csv(DATA / "sp500_monthly.csv")
    s["year"] = s.date.str[:4].astype(int)
    a = s.groupby("year").agg(price=("price", "mean"), E=("earnings", "mean"), D=("dividend", "mean"), cpi=("cpi", "mean"))
    a["realE"] = a.E / a.cpi
    a["realD"] = a.D / a.cpi
    return s, a


def eps_against_the_economy():
    section("4. Earnings per share against the economy (United States)")
    s, a = annual_sp500()
    u = pd.read_csv(DATA / "us_gdp.csv").set_index("year")
    u["gdp_pc"] = u.gdp / u.population
    print(f"S&P monthly data: {s.date.iloc[0]} to {s.date.iloc[-1]}; complete calendar years 1871-2022 used here")

    def decade_cagr(series, first_end, last_end):
        v0 = series.loc[first_end - 9 : first_end].mean()
        v1 = series.loc[last_end - 9 : last_end].mean()
        return gc.cagr(v0, v1, last_end - first_end)

    out = []
    for first_end, last_end in [(1880, 2022), (1880, 1950), (1950, 2022), (1990, 2022)]:
        r = {
            "window": f"{first_end - 9}-{first_end} to {last_end - 9}-{last_end}",
            "real_eps": decade_cagr(a.realE, first_end, last_end),
            "real_dps": decade_cagr(a.realD, first_end, last_end),
            "real_gdp": decade_cagr(u.gdp, first_end, last_end),
            "real_gdp_pc": decade_cagr(u.gdp_pc, first_end, last_end),
        }
        out.append(r)
        print(
            f"{r['window']}: real EPS {pct(r['real_eps'])}, real DPS {pct(r['real_dps'])}, "
            f"real GDP {pct(r['real_gdp'])}, real GDP per person {pct(r['real_gdp_pc'])}; "
            f"EPS minus GDP {100 * (r['real_eps'] - r['real_gdp']):+.2f} points"
        )
    pd.DataFrame(out).to_csv(RESULTS / "eps_vs_gdp.csv", index=False, float_format="%.6g")

    years = range(1871, 2023)
    idx = pd.DataFrame(
        {
            "year": list(years),
            "real_eps": [a.realE.loc[y] for y in years],
            "real_gdp": [u.gdp.loc[y] for y in years],
            "real_gdp_pc": [u.gdp_pc.loc[y] for y in years],
        }
    )
    base = idx[idx.year.between(1871, 1880)].mean()
    for c in ("real_eps", "real_gdp", "real_gdp_pc"):
        idx[c] = idx[c] / base[c]
    idx.to_csv(RESULTS / "eps_gdp_index.csv", index=False, float_format="%.6g")
    end = idx[idx.year.between(2013, 2022)].mean()
    print(
        f"Index (1871-1880 average = 1) in 2013-2022: real EPS {end.real_eps:.1f}, real GDP {end.real_gdp:.1f}, "
        f"real GDP per person {end.real_gdp_pc:.1f}"
    )
    lr = np.log(idx.real_eps / idx.real_gdp_pc)
    slope, se, _ = gc.ols_slope(idx.year, lr)
    print(f"Trend in log(real EPS / real GDP per person), 1871-2022: {100 * slope:+.2f}% per year (HC1 se {100 * se:.2f})")
    lr2 = np.log(idx.real_eps / idx.real_gdp)
    slope2, se2, _ = gc.ols_slope(idx.year, lr2)
    print(f"Trend in log(real EPS / real GDP), 1871-2022: {100 * slope2:+.2f}% per year (HC1 se {100 * se2:.2f})")


def what_prices_imply():
    section("5. What the index price implies")
    s, a = annual_sp500()
    tr = gc.real_total_return_index(s.price, s.dividend, s.cpi)
    months = len(s) - 1
    r_hist = tr.iloc[-1] ** (12 / months) - 1
    print(f"Real total return of the S&P Composite, {s.date.iloc[0]} to {s.date.iloc[-1]}: {pct(r_hist)} per year")
    for start in ("1950-01", "1990-01"):
        i = s.index[s.date == start][0]
        rr = (tr.iloc[-1] / tr.iloc[i]) ** (12 / (months - i)) - 1
        print(f"    from {start}: {pct(rr)} per year")

    c = s.dropna(subset=["cape"]).copy()
    c["year"] = c.date.str[:4].astype(int)
    print(f"CAPE: median {c.cape.median():.1f} (1881-{c.year.max()}), last value {c.cape.iloc[-1]:.1f} in {c.date.iloc[-1]}")
    print(f"    maximum {c.cape.max():.1f} in {c.date.loc[c.cape.idxmax()]}")
    last_cape = c.cape.iloc[-1]
    med_cape = c.cape.median()
    for label, cape in (("median", med_cape), ("last", last_cape)):
        ey = 1 / cape
        g_min = r_hist - ey
        print(f"{label} CAPE {cape:.1f}: earnings yield {pct(ey)}; minimum implied perpetual real growth at r = {pct(r_hist)}: {pct(g_min)}")
    print("Minimum implied real growth at the last CAPE for other required real returns:")
    for r in (0.04, 0.05, 0.06, 0.07):
        print(f"    r = {pct(r, 0)}: g >= {pct(r - 1 / last_cape)}")

    print(
        f"External figure, not in the pinned data: CAPE {CAPE_NOW:.0f} in autumn 2026 (Shiller data as reported by "
        f"CFA Institute, Enterprising Investor, 18 November 2026)"
    )
    g_now = r_hist - 1 / CAPE_NOW
    print(f"    earnings yield {pct(1 / CAPE_NOW)}; minimum implied perpetual real growth at r = {pct(r_hist)}: {pct(g_now)}")
    for r in (0.05, 0.06, 0.07):
        print(f"    r = {pct(r, 0)}: g >= {pct(r - 1 / CAPE_NOW)}")
    print(
        f"    share of value beyond 50 years {pct(gc.share_beyond(50, r_hist, g_now), 1)}, beyond 100 years "
        f"{pct(gc.share_beyond(100, r_hist, g_now), 1)}, half the value after year {gc.years_to_share(0.5, r_hist, g_now):.0f}"
    )

    # the CAPE divides by a ten-year average of real earnings, centred about five years back; when earnings grow,
    # current earnings are higher than that average and the bound above is too high by roughly this much
    for g_e, label in ((0.0183, "1871-2022 real EPS growth"), (0.0351, "1990-2022 real EPS growth")):
        lag = (1 + g_e) ** 5
        print(
            f"    lag-adjusted (earnings grow {pct(g_e)} a year, {label}): earnings yield {pct(lag / CAPE_NOW)}, "
            f"minimum implied growth {pct(r_hist - lag / CAPE_NOW)}"
        )
    last = s[s.earnings > 0].iloc[-1]
    ey_trail = last.earnings / last.price
    print(f"Trailing twelve-month earnings yield at {last.date}: {pct(ey_trail)} (price {last.price:,.0f})")
    print(f"    minimum implied perpetual real growth on trailing earnings at r = {pct(r_hist)}: {pct(r_hist - ey_trail)}")

    g_last = r_hist - 1 / last_cape
    g_med = r_hist - 1 / med_cape
    for label, g in (("median CAPE", g_med), ("last CAPE", g_last)):
        print(
            f"{label}: share of value beyond 50 years {pct(gc.share_beyond(50, r_hist, g), 1)}, "
            f"beyond 100 years {pct(gc.share_beyond(100, r_hist, g), 1)}, half the value after year "
            f"{gc.years_to_share(0.5, r_hist, g):.0f}"
        )

    c["g_min"] = r_hist - 1 / c.cape
    yearly = c.groupby("year").agg(cape=("cape", "mean"), g_min=("g_min", "mean")).reset_index()
    # realised real EPS growth over the following 20 years, decade averages at both ends
    realised = []
    for y in yearly.year:
        if y - 9 >= 1871 and y + 20 <= 2022:
            v0 = a.realE.loc[y - 9 : y].mean()
            v1 = a.realE.loc[y + 11 : y + 20].mean()
            realised.append(gc.cagr(v0, v1, 20))
        else:
            realised.append(np.nan)
    yearly["realised_eps_growth_next20"] = realised
    yearly.to_csv(RESULTS / "implied_growth_series.csv", index=False, float_format="%.6g")
    ok = yearly.dropna()
    gap = ok.g_min - ok.realised_eps_growth_next20
    print(
        f"Years 1890-2002 (n = {len(ok)}): implied minimum growth exceeded the next 20 years' realised real EPS growth "
        f"in {int((gap > 0).sum())} years; mean gap {100 * gap.mean():+.2f} points"
    )
    top = ok[ok.cape >= ok.cape.quantile(0.8)]
    gap_top = top.g_min - top.realised_eps_growth_next20
    print(
        f"    in the fifth of years with the highest CAPE (CAPE >= {ok.cape.quantile(0.8):.1f}, n = {len(top)}): "
        f"exceeded in {int((gap_top > 0).sum())}, mean gap {100 * gap_top.mean():+.2f} points"
    )
    hi = ok[ok.cape >= 25]
    print(
        f"    years with CAPE >= 25 and 20 years of later data: {', '.join(str(y) for y in hi.year)}; "
        f"implied {pct(hi.g_min.mean())} vs realised {pct(hi.realised_eps_growth_next20.mean())} on average"
    )
    slope, se, r2 = gc.ols_slope(ok.cape, ok.realised_eps_growth_next20)
    print(f"    slope of realised 20-year real EPS growth on CAPE: {100 * slope:+.3f} points per CAPE unit (se {100 * se:.3f})")
    return r_hist


def world_energy():
    section("6. The world economy and its energy")
    w = pd.read_csv(DATA / "world.csv").set_index("year")
    print(f"World primary energy: {w.energy_twh.loc[1965]:,.0f} TWh in 1965, {w.energy_twh.loc[2024]:,.0f} TWh in 2024")
    ge = gc.cagr(w.energy_twh.loc[1965], w.energy_twh.loc[2024], 59)
    print(f"    growth 1965-2024: {pct(ge)} per year; doubling time {gc.doubling_time(ge):.0f} years")
    gdp = w.gdp.dropna()
    print(f"World GDP (Maddison) benchmark years available: {', '.join(str(y) for y in gdp.index)}")
    g70 = gc.cagr(gdp.loc[1970], gdp.loc[2022], 52)
    e70 = gc.cagr(w.energy_twh.loc[1970], w.energy_twh.loc[2022], 52)
    i70 = gc.required_productivity_growth(g70, e70)
    print(f"1970-2022: GDP {pct(g70)}/yr, energy {pct(e70)}/yr, GDP per unit of energy {pct(i70)}/yr")
    print(f"    GDP multiplied by {gdp.loc[2022] / gdp.loc[1970]:.2f}, energy by {w.energy_twh.loc[2022] / w.energy_twh.loc[1970]:.2f}")
    rows = []
    for y0, y1 in [(1970, 1980), (1980, 1990), (1990, 2000), (2000, 2010), (2010, 2019), (2019, 2022), (1970, 2022)]:
        g, e, p = gc.decompose_log_growth(gdp.loc[y0], gdp.loc[y1], w.energy_twh.loc[y0], w.energy_twh.loc[y1], y1 - y0)
        rows.append({"from": y0, "to": y1, "gdp": g, "energy": e, "productivity": p, "energy_share": e / g})
        print(f"    {y0}-{y1}: GDP {g:.2f}%/yr = energy {e:.2f} + energy productivity {p:.2f}; energy accounts for {100 * e / g:.0f}% of growth")
    pd.DataFrame(rows).to_csv(RESULTS / "world_decomposition.csv", index=False, float_format="%.6g")

    d = w.energy_twh.pct_change().dropna()
    falls = d[d < 0]
    print(f"Years 1966-2024 in which world primary energy fell: {len(falls)} ({', '.join(str(y) for y in falls.index)})")
    print(f"    largest fall {pct(falls.min())} in {falls.idxmin()}")

    pc = w.energy_twh / w.population
    gpc = gc.cagr(pc.loc[1965], pc.loc[2024], 59)
    print(f"World energy per person: {pc.loc[1965] * 1e6:.1f} MWh in 1965, {pc.loc[2024] * 1e6:.1f} MWh in 2024")
    print(f"    per-person energy growth 1965-2024: {pct(gpc)} per year")

    best = max(r["productivity"] for r in rows[:-1])
    print(f"Fastest decade-scale gain in GDP per unit of energy: {best:.2f}% per year")
    for g in (0.02, 0.03):
        need = gc.required_productivity_growth(g, 0.0)
        print(
            f"GDP growth {pct(g, 0)} with energy flat needs energy productivity +{pct(need)} per year for ever; "
            f"energy per unit of GDP falls tenfold in {gc.years_to_factor(10, need):.0f} years, "
            f"a hundredfold in {gc.years_to_factor(100, need):.0f}"
        )
    print(f"At the 1970-2022 energy growth rate ({pct(e70)}): energy x{(1 + e70) ** 50:.1f} in 50 years, x{(1 + e70) ** 100:.1f} in 100")
    print(f"At the 1970-2022 productivity rate ({pct(i70)}), energy per unit of GDP falls tenfold in {gc.years_to_factor(10, i70):.0f} years")
    print(f"World CO2 (fossil and industry): {w.co2.loc[1970]:,.0f} Mt in 1970, {w.co2.loc[2024]:,.0f} Mt in 2024")


def countries():
    section("7. Countries: growth with less energy, and where the energy went")
    c = pd.read_csv(DATA / "countries.csv")
    y0, y1 = 2000, 2022
    a = c[c.year == y0].set_index("iso_code")
    b = c[c.year == y1].set_index("iso_code")
    keep = b.population >= MIN_POP
    df = pd.DataFrame(
        {
            "country": b.country,
            "pop": b.population,
            "gdp0": a.gdp,
            "gdp1": b.gdp,
            "e0": a.energy_twh,
            "e1": b.energy_twh,
            "co2_0": a.co2,
            "co2_1": b.co2,
            "cco2_0": a.consumption_co2,
            "cco2_1": b.consumption_co2,
            "fossil_prod_1": b.fossil_production_twh,
        }
    )[keep]
    df = df.dropna(subset=["gdp0", "gdp1", "e0", "e1"])
    df["gdp_ratio"] = df.gdp1 / df.gdp0
    df["e_ratio"] = df.e1 / df.e0
    df["class"] = [gc.classify_decoupling(g, e) for g, e in zip(df.gdp_ratio, df.e_ratio, strict=True)]
    df["gdp_pc_2022"] = df.gdp1 / df["pop"]
    df["e_pc_2022_mwh"] = df.e1 * 1e6 / df["pop"]
    print(f"Countries with population >= 1 million and GDP and energy in {y0} and {y1}: {len(df)}")
    counts = df["class"].value_counts()
    for k in ("absolute", "relative", "none"):
        print(f"    {k:9s}: {counts.get(k, 0)}")
    tot_gdp = df.gdp1.sum()
    abs_ = df[df["class"] == "absolute"]
    print(f"    share of the sample's 2022 GDP in absolute decouplers: {pct(abs_.gdp1.sum() / tot_gdp, 1)}")
    print(f"    share of the sample's 2022 population in absolute decouplers: {pct(abs_['pop'].sum() / df['pop'].sum(), 1)}")
    print(f"    absolute decouplers' energy change {y0}-{y1}: {abs_.e1.sum() - abs_.e0.sum():+,.0f} TWh")
    rest = df[df["class"] != "absolute"]
    print(f"    everyone else's energy change: {rest.e1.sum() - rest.e0.sum():+,.0f} TWh")
    med_pc_abs = abs_.gdp_pc_2022.median()
    med_pc_rest = rest.gdp_pc_2022.median()
    print(f"    median 2022 GDP per person: absolute decouplers ${med_pc_abs:,.0f}, others ${med_pc_rest:,.0f}")
    print(
        f"    median 2022 energy per person: absolute decouplers {abs_.e_pc_2022_mwh.median():.1f} MWh, others {rest.e_pc_2022_mwh.median():.1f} MWh"
    )

    both = abs_.dropna(subset=["co2_0", "co2_1", "cco2_0", "cco2_1"]).copy()
    both["terr"] = both.co2_1 / both.co2_0 - 1
    both["cons"] = both.cco2_1 / both.cco2_0 - 1
    print(f"Absolute decouplers with consumption-based CO2: {len(both)}")
    print(f"    median change {y0}-{y1}: territorial CO2 {pct(both.terr.median(), 1)}, consumption CO2 {pct(both.cons.median(), 1)}")
    print(f"    consumption CO2 fell less than territorial (or rose) in {int((both.cons > both.terr).sum())} of {len(both)}")
    print(f"    consumption CO2 rose in {int((both.cons > 0).sum())} of {len(both)}")
    tc = both.co2_1.sum() / both.co2_0.sum() - 1
    cc = both.cco2_1.sum() / both.cco2_0.sum() - 1
    print(f"    summed over them: territorial {pct(tc, 1)}, consumption {pct(cc, 1)}")
    imp1 = (both.cco2_1 - both.co2_1).sum()
    imp0 = (both.cco2_0 - both.co2_0).sum()
    print(f"    their net CO2 imported in goods: {imp0:,.0f} Mt in {y0}, {imp1:,.0f} Mt in {y1}")

    # cross-section: does energy use per person fall as countries get richer and more service-based?
    x = df.dropna(subset=["gdp_pc_2022", "e_pc_2022_mwh"])
    x = x[(x.gdp_pc_2022 > 0) & (x.e_pc_2022_mwh > 0)]
    slope, se, r2 = gc.ols_slope(np.log(x.gdp_pc_2022), np.log(x.e_pc_2022_mwh))
    print(f"Cross-section 2022 (n = {len(x)}): elasticity of energy per person to GDP per person {slope:.2f} (se {se:.2f}), R2 {r2:.2f}")
    q = x.gdp_pc_2022.quantile([0.2, 0.8])
    poor = x[x.gdp_pc_2022 <= q.iloc[0]]
    rich = x[x.gdp_pc_2022 >= q.iloc[1]]
    print(
        f"    median energy per person: richest fifth {rich.e_pc_2022_mwh.median():.1f} MWh, poorest fifth "
        f"{poor.e_pc_2022_mwh.median():.1f} MWh, ratio {rich.e_pc_2022_mwh.median() / poor.e_pc_2022_mwh.median():.0f}x"
    )
    w = pd.read_csv(DATA / "world.csv").set_index("year")
    world_pc = w.energy_twh.loc[2022] * 1e6 / w.population.loc[2022]
    print(f"    world energy per person 2022: {world_pc:.1f} MWh; richest fifth median / world: {rich.e_pc_2022_mwh.median() / world_pc:.1f}x")
    mid = x[x.gdp_pc_2022 >= 15000]
    s_m, e_m, r2_m = gc.ols_slope(np.log(mid.gdp_pc_2022), np.log(mid.e_pc_2022_mwh))
    print(f"    countries with GDP per person >= $15,000 (n = {len(mid)}): elasticity {s_m:.2f} (se {e_m:.2f}), R2 {r2_m:.2f}")
    exporter = x.fossil_prod_1.fillna(0) > 1.2 * x.e1
    print(f"    fossil-fuel net exporters (production > 1.2 x consumption) in 2022: {int(exporter.sum())}")
    y = x[~exporter]
    s_y, e_y, r2_y = gc.ols_slope(np.log(y.gdp_pc_2022), np.log(y.e_pc_2022_mwh))
    print(f"    without them (n = {len(y)}): elasticity {s_y:.2f} (se {e_y:.2f}), R2 {r2_y:.2f}")
    ym = y[y.gdp_pc_2022 >= 15000]
    s_ym, e_ym, r2_ym = gc.ols_slope(np.log(ym.gdp_pc_2022), np.log(ym.e_pc_2022_mwh))
    print(f"    without them, GDP per person >= $15,000 (n = {len(ym)}): elasticity {s_ym:.2f} (se {e_ym:.2f}), R2 {r2_ym:.2f}")
    # long differences within countries
    ld = df[(df.gdp_ratio > 0) & (df.e_ratio > 0)]
    slope, se, r2 = gc.ols_slope(np.log(ld.gdp_ratio), np.log(ld.e_ratio))
    print(f"Within-country long difference {y0}-{y1} (n = {len(ld)}): energy elasticity to GDP {slope:.2f} (se {se:.2f})")
    for label, sub in (
        ("richest half", ld[ld.gdp_pc_2022 >= ld.gdp_pc_2022.median()]),
        ("poorest half", ld[ld.gdp_pc_2022 < ld.gdp_pc_2022.median()]),
    ):
        s_, e_, _ = gc.ols_slope(np.log(sub.gdp_ratio), np.log(sub.e_ratio))
        print(f"    {label}: {s_:.2f} (se {e_:.2f}), n = {len(sub)}")

    # robustness: 2005-2019
    a2 = c[c.year == 2005].set_index("iso_code")
    b2 = c[c.year == 2019].set_index("iso_code")
    r2df = pd.DataFrame({"pop": b2.population, "g": b2.gdp / a2.gdp, "e": b2.energy_twh / a2.energy_twh}).dropna()
    r2df = r2df[r2df["pop"] >= MIN_POP]
    cl = pd.Series([gc.classify_decoupling(g, e) for g, e in zip(r2df.g, r2df.e, strict=True)]).value_counts()
    print(f"Robustness 2005-2019 (n = {len(r2df)}): absolute {cl.get('absolute', 0)}, relative {cl.get('relative', 0)}, none {cl.get('none', 0)}")

    df.reset_index().rename(columns={"index": "iso_code"}).to_csv(RESULTS / "country_decoupling.csv", index=False, float_format="%.6g")
    names = ", ".join(abs_.sort_values("gdp1", ascending=False).country.head(12))
    print(f"Largest absolute decouplers by 2022 GDP: {names}")


def time_budget():
    section("8. Time")
    w = pd.read_csv(DATA / "world.csv").set_index("year")
    pop = w.population.loc[2024]
    hours = pop * 24 * 365.25
    print(f"World population 2024: {pop / 1e9:.2f} billion; hours lived per year: {hours:.3g}")
    for g in (0.02, 0.03):
        print(f"With hours fixed, output per hour lived must grow {pct(g, 0)}/yr; it doubles every {gc.doubling_time(g):.0f} years")
    print(f"Over 100 years at 2%: x{1.02**100:.1f}; at 3%: x{1.03**100:.1f}")


def main():
    RESULTS.mkdir(exist_ok=True)
    firm_that_does_not_grow()
    where_the_value_sits()
    firm_inside_its_market()
    eps_against_the_economy()
    what_prices_imply()
    world_energy()
    countries()
    time_budget()


if __name__ == "__main__":
    main()
