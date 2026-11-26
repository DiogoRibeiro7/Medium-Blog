"""Every number in the article, printed in order, and the tables the figures read (results/).

python run_analysis.py > output.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import core
import transcribed as tr

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
SYSTEMS = {"lisboa": "Lisboa", "porto": "Porto", "mst": "Almada"}
NAMES = {"lisboa": "Metropolitano de Lisboa", "porto": "Metro do Porto", "mst": "Metro Sul do Tejo"}
GRID = np.round(np.arange(0.20, 0.9001, 0.02), 2)
# Estimation sample for the day weights: normal years, no pandemic, no recent decline.
FIT_YEARS = list(range(2015, 2020)) + [2023, 2024]
WIDE_YEARS = list(range(2005, 2012)) + list(range(2015, 2020)) + [2023, 2024]
ALL_MONTHS = tuple(range(1, 13))
JAN_AUG = tuple(range(1, 9))
QUARTERS = {"Q1": (1, 2, 3), "Q2": (4, 5, 6), "Q3": (7, 8, 9), "Q4": (10, 11, 12)}


def section(title: str) -> None:
    print(f"\n== {title} ==")


def pct(x: float, nd: int = 2) -> str:
    return f"{x:+.{nd}f}%"


def load() -> pd.DataFrame:
    m = pd.read_csv(DATA / "monthly.csv", parse_dates=["month"], index_col="month")
    m.index.freq = "MS"
    return m


def closures(table: dict) -> dict[str, float]:
    return {d: w for d, (w, _) in table.items()}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    m = load()
    pax = {k: m[f"pax_{k}"] for k in SYSTEMS}
    place = {k: m[f"placekm_{k}"] for k in SYSTEMS}
    pkm = {k: m[f"pkm_{k}"] for k in SYSTEMS}

    # ------------------------------------------------------------------ 1. The long series
    section("1. Annual passengers, INE metro survey (millions)")
    annual = pd.DataFrame({k: pax[k].groupby(pax[k].index.year).sum() / 1000 for k in SYSTEMS})
    annual_place = pd.DataFrame({k: place[k].groupby(place[k].index.year).sum() / 1e6 for k in SYSTEMS})
    for y in [2004, 2005, 2010, 2011, 2012, 2013, 2014, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
        print(f"{y}: Lisboa {annual.loc[y, 'lisboa']:.3f}  Porto {annual.loc[y, 'porto']:.3f}  MST {annual.loc[y, 'mst']:.3f}")
    lis = annual["lisboa"].loc[:2025]
    print(f"Lisboa peak {lis.idxmax()} {lis.max():.3f}; low after 2011 {lis.loc[2012:2019].idxmin()} {lis.loc[2012:2019].min():.3f}")
    print(f"Lisboa 2011->2013 {pct(core.change(lis[2013], lis[2011]), 1)}; 2014->2019 {pct(core.change(lis[2019], lis[2014]), 1)}")
    print(f"Lisboa 2018->2019 {pct(core.change(lis[2019], lis[2018]), 1)} (Navegante passes from April 2019)")
    for k in SYSTEMS:
        a = annual[k]
        print(
            f"{NAMES[k]}: 2024->2025 {pct(core.change(a[2025], a[2024]))}; 2019->2025 {pct(core.change(a[2025], a[2019]), 1)}; "
            f"place-km 2019->2025 {pct(core.change(annual_place.loc[2025, k], annual_place.loc[2019, k]), 1)} "
            f"({annual_place.loc[2019, k]:.3f} -> {annual_place.loc[2025, k]:.3f} billion)"
        )
    print(f"Lisboa 2025 against 2019: {annual.loc[2025, 'lisboa'] - annual.loc[2019, 'lisboa']:.3f} million")
    for k in SYSTEMS:
        r = [annual.loc[y, k] / annual_place.loc[y, k] for y in (2019, 2024, 2025)]
        print(f"{NAMES[k]} passengers per 1,000 place-km: 2019 {r[0]:.1f}, 2024 {r[1]:.1f}, 2025 {r[2]:.1f}")
    out = annual.round(4)
    out.to_csv(RESULTS / "annual_passengers.csv")

    # ------------------------------------------------------------------ 2. The headline numbers
    section("2. The headline numbers")
    a24, a25 = annual.loc[2024, "lisboa"], annual.loc[2025, "lisboa"]
    print(f"INE Lisboa 2024 {a24:.3f} m, 2025 {a25:.3f} m, change {pct(core.change(a25, a24))}, {a25 - a24:.3f} m")
    v24, ev24 = tr.ML_COUNTS["2024_valid_ticket"][0], tr.ML_COUNTS["2024_fare_evasion_estimate"][0]
    print(f"Company 2024: valid tickets {v24:,} + estimated evasion {ev24:,} = {v24 + ev24:,} (INE {a24 * 1e6:,.0f})")
    print(f"Estimated evasion share of the 2024 total: {ev24 / (v24 + ev24) * 100:.2f}%")
    v25 = tr.ML_COUNTS["2025_valid_ticket_m"][0]
    print(f"Company 2025 valid tickets {v25} m: {pct(core.change(v25 * 1e6, v24))} on 2024; INE minus valid = {a25 - v25:.2f} m")
    jan_aug_25 = core.period_total(pax["lisboa"], 2025, JAN_AUG) / 1000
    jan_aug_26 = core.period_total(pax["lisboa"], 2026, JAN_AUG) / 1000
    print(
        f"INE Lisboa Jan-Aug: 2025 {jan_aug_25:.3f} m, 2026 {jan_aug_26:.3f} m, {pct(core.change(jan_aug_26, jan_aug_25))}, {jan_aug_26 - jan_aug_25:.3f} m"
    )
    print(f"Company Jan-Sep 2026: {tr.ML_COUNTS['jan_sep_2026_m'][0]} m, {tr.ML_COUNTS['jan_sep_2026_change'][0]}%")
    for y in (2025, 2026):
        for q, ms in QUARTERS.items():
            if y == 2026 and q == "Q4":
                continue
            ms2 = ms if not (y == 2026 and q == "Q3") else (7, 8)
            print(f"Lisboa {y} {q}{'' if ms2 == ms else ' (Jul-Aug)'}: {pct(core.period_change(pax['lisboa'], y, ms2), 1)}")

    # ------------------------------------------------------------------ 3. Calendar and closures
    section("3. Calendar: day weights estimated from the monthly series")
    counts = {k: core.day_counts("2004-01", "2026-08", SYSTEMS[k]) for k in SYSTEMS}
    weights = {}
    for k in SYSTEMS:
        s = pax[k]
        fit_mask = s.index.year.isin(FIT_YEARS) & s.notna() & (s > 0)
        sat, sun, sse = core.fit_day_weights(s[fit_mask], counts[k], GRID)
        weights[k] = (sat, sun)
        print(f"{NAMES[k]}: Saturday {sat:.2f}, Sunday/holiday {sun:.2f} of a weekday (fit on {fit_mask.sum()} months, SSE {sse:.4f})")
    s = pax["lisboa"]
    wide = s.index.year.isin(WIDE_YEARS)
    wsat, wsun, _ = core.fit_day_weights(s[wide], counts["lisboa"], GRID)
    print(f"Lisboa, wider sample {WIDE_YEARS[0]}-2011, 2015-2019, 2023-2024: Saturday {wsat:.2f}, Sunday/holiday {wsun:.2f}")

    clos = {"lisboa": closures(tr.CLOSURES_LISBOA), "porto": closures(tr.CLOSURES_PORTO), "mst": closures(tr.CLOSURES_MST)}
    eq_cal = {k: core.weekday_equivalents(counts[k], *weights[k]) for k in SYSTEMS}
    eq_adj = {k: core.remove_closures(eq_cal[k], clos[k]) for k in SYSTEMS}

    section("3b. Weekday-equivalent days, Lisboa")
    for y, ms, lab in [(2024, ALL_MONTHS, "2024"), (2025, ALL_MONTHS, "2025"), (2025, JAN_AUG, "Jan-Aug 2025"), (2026, JAN_AUG, "Jan-Aug 2026")]:
        print(f"{lab}: calendar {core.period_total(eq_cal['lisboa'], y, ms):.2f}, after closures {core.period_total(eq_adj['lisboa'], y, ms):.2f}")
    print("Closures (share of a day lost):")
    for d, (w, why) in tr.CLOSURES_LISBOA.items():
        print(f"  {d} {w:.2f} {why}")

    section("3c. Decomposition of the change, Lisboa")
    decomp = []
    for lab, y, ms in [("2025 vs 2024", 2025, ALL_MONTHS), ("Jan-Aug 2026 vs 2025", 2026, JAN_AUG)]:
        raw = core.period_change(pax["lisboa"], y, ms)
        cal = core.period_change(pax["lisboa"], y, ms, eq_cal["lisboa"])
        adj = core.period_change(pax["lisboa"], y, ms, eq_adj["lisboa"])
        print(f"{lab}: raw {pct(raw)}; per calendar day {pct(cal)}; per day the network was open {pct(adj)}")
        print(f"   calendar accounts for {raw - cal:+.2f} points, closures for {cal - adj:+.2f} points, remainder {adj:+.2f}")
        decomp.append({"period": lab, "raw": raw, "calendar": raw - cal, "closures": cal - adj, "remainder": adj})
    pd.DataFrame(decomp).round(4).to_csv(RESULTS / "decomposition.csv", index=False)

    section("3d. Sensitivity of the remainder")
    alts = {
        "fitted weights": weights["lisboa"],
        "wider-sample weights": (wsat, wsun),
        "Saturday 0.50, Sunday 0.35": (0.50, 0.35),
        "Saturday 0.70, Sunday 0.50": (0.70, 0.50),
    }
    for name, (a_, b_) in alts.items():
        e = core.remove_closures(core.weekday_equivalents(counts["lisboa"], a_, b_), clos["lisboa"])
        r25 = core.period_change(pax["lisboa"], 2025, ALL_MONTHS, e)
        r26 = core.period_change(pax["lisboa"], 2026, JAN_AUG, e)
        print(f"{name}: 2025 {pct(r25)}, Jan-Aug 2026 {pct(r26)}")
    for partial, blackout in ((0.25, 0.4), (0.45, 0.8)):
        c2 = {d: (partial if abs(w - 0.35) < 1e-9 else blackout if d == "2025-04-28" else w) for d, w in clos["lisboa"].items()}
        e = core.remove_closures(eq_cal["lisboa"], c2)
        print(
            f"partial strikes lose {partial:.2f} of a day and the blackout {blackout:.1f}: 2025 {pct(core.period_change(pax['lisboa'], 2025, ALL_MONTHS, e))}, "
            f"Jan-Aug 2026 {pct(core.period_change(pax['lisboa'], 2026, JAN_AUG, e))}"
        )
    print("(2024 and 2025 each have two partial strikes, so their weight nearly cancels in the 2025 comparison)")
    no_aug = tuple(x for x in JAN_AUG if x != 8)
    print(f"Jan-Jul 2026 (August works left out): {pct(core.period_change(pax['lisboa'], 2026, no_aug, eq_adj['lisboa']))}")
    no_may_aug = tuple(x for x in JAN_AUG if x not in (5, 8))
    print(f"Jan-Aug 2026 without May and August (months with works): {pct(core.period_change(pax['lisboa'], 2026, no_may_aug, eq_adj['lisboa']))}")

    # ------------------------------------------------------------------ 4. Three systems, quarter by quarter
    section("4. Change on the same quarter a year earlier, per day the network was open")
    rows = []
    for y in (2024, 2025, 2026):
        for q, ms in QUARTERS.items():
            if y == 2026 and q == "Q4":
                continue
            ms2 = (7, 8) if (y == 2026 and q == "Q3") else ms
            row = {"quarter": f"{y} {q}" + (" (Jul-Aug)" if ms2 != ms else "")}
            for k in SYSTEMS:
                row[k] = core.period_change(pax[k], y, ms2, eq_adj[k])
                row[f"{k}_raw"] = core.period_change(pax[k], y, ms2)
                row[f"{k}_place"] = core.period_change(place[k], y, ms2)
            rows.append(row)
    qt = pd.DataFrame(rows)
    for _, r in qt.iterrows():
        print(
            f"{r['quarter']}: Lisboa {r['lisboa']:+.1f} (raw {r['lisboa_raw']:+.1f}, place-km {r['lisboa_place']:+.1f}) | "
            f"Porto {r['porto']:+.1f} (place-km {r['porto_place']:+.1f}) | MST {r['mst']:+.1f} (place-km {r['mst_place']:+.1f}) | "
            f"Lisboa minus MST {r['lisboa'] - r['mst']:+.1f}, minus Porto {r['lisboa'] - r['porto']:+.1f}"
        )
    qt.round(4).to_csv(RESULTS / "quarterly_changes.csv", index=False)
    for lab, y, ms in [("2025", 2025, ALL_MONTHS), ("Jan-Aug 2026", 2026, JAN_AUG)]:
        vals = {k: core.period_change(pax[k], y, ms, eq_adj[k]) for k in SYSTEMS}
        raws = {k: core.period_change(pax[k], y, ms) for k in SYSTEMS}
        print(f"{lab}: " + "; ".join(f"{NAMES[k]} {pct(vals[k], 1)} (raw {pct(raws[k], 1)})" for k in SYSTEMS))
    g24 = qt[qt.quarter.str.startswith("2024")]
    g25 = qt[qt.quarter.str.startswith("2025")]
    g26 = qt[qt.quarter.str.startswith("2026")]
    for lab, g in (("2024", g24), ("2025", g25), ("2026", g26)):
        print(f"mean quarterly gap {lab}: Lisboa minus Porto {(g.lisboa - g.porto).mean():+.1f}, Lisboa minus MST {(g.lisboa - g.mst).mean():+.1f}")

    section("4b. Monthly changes, per day open (for the figure)")
    mon = pd.DataFrame({k: (pax[k] / eq_adj[k]).pct_change(12) * 100 for k in SYSTEMS}).loc["2024-01-01":]
    mon["lisboa_raw"] = pax["lisboa"].pct_change(12).loc["2024-01-01":] * 100
    mon.round(4).to_csv(RESULTS / "monthly_changes.csv", index_label="month")
    for d, r in mon.loc["2025-01-01":].iterrows():
        print(f"{d:%Y-%m}: Lisboa {r['lisboa']:+.1f} (raw {r['lisboa_raw']:+.1f}), Porto {r['porto']:+.1f}, MST {r['mst']:+.1f}")
    print("Works months (not adjusted): " + "; ".join(w for _, w in tr.WORKS_LISBOA))

    # ------------------------------------------------------------------ 5. Supply
    section("5. Supply: what the metro ran")
    ml = tr.ML_ANNUAL
    tk19, tk24, tk25 = ml[2019]["train_km_thousand"], ml[2024]["train_km_thousand"], ml[2025]["train_km_thousand"]
    pl19, pl24, pl25 = ml[2019]["place_km_million"], ml[2024]["place_km_million"], ml[2025]["place_km_million"]
    p24, p25 = ml[2024]["passengers"], ml[2025]["passengers"]
    print(f"Train-km (thousand): 2019 ~{tk19:.0f} (rounded label), 2024 {tk24:.2f}, 2025 {tk25:.2f}: 2025 vs 2024 {pct(core.change(tk25, tk24))}")
    print(
        f"Circulations: 2024 {ml[2024]['circulations']:,}, 2025 {ml[2025]['circulations']:,}: {pct(core.change(ml[2025]['circulations'], ml[2024]['circulations']))}"
    )
    print(f"Place-km (million): 2019 {pl19:.0f}, 2024 {pl24:.2f}, 2025 {pl25:.2f}: 2025 vs 2024 {pct(core.change(pl25, pl24))}")
    print(f"Places per train-km: 2019 {pl19 * 1000 / tk19:.0f}, 2024 {pl24 * 1000 / tk24:.1f}, 2025 {pl25 * 1000 / tk25:.1f}")
    print(f"Passengers (AMT): 2024 {p24:,}, 2025 {p25:,}: {pct(core.change(p25, p24))}")
    print(
        f"Passengers per train-km: 2019 {ml[2019]['passengers_m'] * 1e6 / (tk19 * 1000):.2f}, 2024 {p24 / (tk24 * 1000):.2f}, 2025 {p25 / (tk25 * 1000):.2f}"
    )
    print(f"Regularity index (AMT definition): 2024 {ml[2024]['regularity_index']:.2f}, 2025 {ml[2025]['regularity_index']:.2f}")
    print(f"Average occupancy: 2024 {ml[2024]['occupancy_rate']:.2f}%, 2025 {ml[2025]['occupancy_rate']:.2f}%")
    print(
        f"Fleet: {ml[2025]['cars']} cars, average age {ml[2024]['fleet_age']} (2024) -> {ml[2025]['fleet_age']} (2025); network {ml[2025]['network_km']} km, unchanged since 2016"
    )
    lp = place["lisboa"]
    for lab, y, ms in [("2025", 2025, ALL_MONTHS), ("Jan-Aug 2026", 2026, JAN_AUG), ("Oct 2025-May 2026", None, None)]:
        if y is not None:
            print(f"INE place-km Lisboa {lab}: {pct(core.period_change(lp, y, ms))}")
    sub = lp.loc["2025-10-01":"2026-05-01"].sum() / lp.loc["2024-10-01":"2025-05-01"].sum()
    print(f"INE place-km Lisboa Oct 2025-May 2026 vs a year earlier: {pct((sub - 1) * 100)}")
    print(f"INE place-km Lisboa Jul-Aug 2026 vs 2025: {pct(core.period_change(lp, 2026, (7, 8)))}")

    section("5b. Concession-contract indicators (fleet availability DMC, regularity REG, circulations run FSC)")
    ref = tr.KPI_REFERENCE
    k25, k26 = tr.KPI_2025_QUARTER, tr.KPI_2026_MONTH
    kq = []
    for i, q in enumerate(QUARTERS):
        kq.append({"quarter": f"2025 {q}", "DMC": k25["DMC"][i], "FSC": k25["FSC"][i], "REG": k25["REG"][i]})
    for i, q in enumerate(["Q1", "Q2"]):
        sl = slice(3 * i, 3 * i + 3)
        kq.append({"quarter": f"2026 {q}", **{x: float(np.mean(k26[x][sl])) for x in ("DMC", "FSC", "REG")}})
    kq = pd.DataFrame(kq)
    lis_q = qt.set_index("quarter")["lisboa"]
    mst_q = qt.set_index("quarter")["mst"]
    kq["lisboa_change"] = [lis_q[q] for q in kq.quarter]
    kq["lisboa_minus_mst"] = [lis_q[q] - mst_q[q] for q in kq.quarter]
    for _, r in kq.iterrows():
        print(
            f"{r['quarter']}: DMC {r['DMC']:.2f} (ref {ref['DMC']}), FSC {r['FSC']:.2f} (ref {ref['FSC']}), REG {r['REG']:.2f} (ref {ref['REG']}) | "
            f"Lisboa {r['lisboa_change']:+.1f}, minus MST {r['lisboa_minus_mst']:+.1f}"
        )
    print(f"2026 monthly DMC: {k26['DMC']}; min {min(k26['DMC'])}, max {max(k26['DMC'])}")
    print(f"2026 monthly REG: {k26['REG']}")
    print(
        f"Correlation over the six quarters, DMC with Lisboa's change: {core.pearson(kq.DMC, kq.lisboa_change):.2f}; "
        f"with Lisboa minus MST: {core.pearson(kq.DMC, kq.lisboa_minus_mst):.2f} (six points; descriptive only)"
    )
    print(f"Correlation, REG with Lisboa's change: {core.pearson(kq.REG, kq.lisboa_change):.2f}")
    kq.round(4).to_csv(RESULTS / "kpi_quarters.csv", index=False)
    print("Causes given by the operator: " + tr.KPI_CAUSES)

    # ------------------------------------------------------------------ 6. Who stopped riding
    section("6. Ticket types and visitors")
    c = tr.ML_COUNTS
    o24, o25 = c["occasional_2024_m"][0], c["occasional_2025_m"][0]
    pp24 = c["customer_paid_passes_2024_m"][0]
    pp25lo, pp25hi = c["customer_paid_passes_2025_m_low"][0], c["customer_paid_passes_2025_m_high"][0]
    v24m = v24 / 1e6
    other24 = v24m - o24 - pp24
    other25_lo, other25_hi = v25 - o25 - pp25hi, v25 - o25 - pp25lo
    print(f"Occasional tickets {o24} -> {o25} m: {pct(core.change(o25, o24), 1)}, {o25 - o24:+.1f} m")
    print(f"Customer-paid passes {pp24} -> {pp25lo}-{pp25hi} m: {pct(core.change(pp25hi, pp24), 1)} to {pct(core.change(pp25lo, pp24), 1)}")
    print(
        f"Everything else (free passes, other) by subtraction: {other24:.1f} -> {other25_lo:.1f}-{other25_hi:.1f} m, "
        f"{pct(core.change(other25_lo, other24), 0)} to {pct(core.change(other25_hi, other24), 0)}"
    )
    print(
        f"Shares of valid-ticket trips: occasional {o24 / v24m * 100:.1f}% -> {o25 / v25 * 100:.1f}%; "
        f"other {other24 / v24m * 100:.1f}% -> {other25_lo / v25 * 100:.1f}-{other25_hi / v25 * 100:.1f}%"
    )
    r24, r25 = ml[2024]["tariff_revenue_keur"], ml[2025]["tariff_revenue_keur"]
    print(f"Tariff revenue {r24:,} -> {r25:,} thousand EUR: {pct(core.change(r25, r24), 1)}")
    for key, lab in [
        ("revenue_single_ticket_keur", "Carris/Metro single ticket"),
        ("revenue_zapping_keur", "Zapping"),
        ("revenue_navegante_metropolitano_keur", "Navegante Metropolitano pass"),
        ("revenue_navegante_lisboa_keur", "Navegante Lisboa pass"),
    ]:
        print(f"  {lab}: {ml[2024][key]:,} -> {ml[2025][key]:,} ({pct(core.change(ml[2025][key], ml[2024][key]), 1)})")
    air = m["airport_lisbon"]
    print(
        f"Lisbon airport passengers: 2024 {air.loc['2024'].sum() / 1e6:.2f} m, 2025 {air.loc['2025'].sum() / 1e6:.2f} m ({pct(core.period_change(air, 2025, ALL_MONTHS), 1)})"
    )
    print(
        f"Lisbon airport Jan-Aug 2026: {pct(core.period_change(air, 2026, JAN_AUG), 1)}; Jun-Aug 2026: {pct(core.period_change(air, 2026, (6, 7, 8)), 1)}"
    )
    air_q = []
    for _, r in qt.iterrows():
        y = int(r["quarter"][:4])
        q = r["quarter"][5:7]
        ms = QUARTERS[q] if "Jul-Aug" not in r["quarter"] else (7, 8)
        air_q.append(core.period_change(air, y, ms))
    qt["airport"] = air_q
    print("Airport change by quarter: " + "; ".join(f"{q} {a:+.1f}" for q, a in zip(qt.quarter, qt.airport, strict=True)))
    print(
        f"Correlation over {len(qt)} quarters, airport with Lisboa: {core.pearson(qt.airport, qt.lisboa):.2f}; with Lisboa minus MST: {core.pearson(qt.airport, qt.lisboa - qt.mst):.2f}"
    )
    mm = pd.DataFrame({"metro": (pax["lisboa"] / eq_adj["lisboa"]).pct_change(12) * 100, "air": air.pct_change(12) * 100}).loc["2024-01-01":].dropna()
    print(f"Correlation over {len(mm)} months (2024-01 to 2026-08), monthly changes: {core.pearson(mm.air, mm.metro):.2f}")
    trip24 = core.period_total(pkm["lisboa"], 2025, JAN_AUG) / core.period_total(pax["lisboa"], 2025, JAN_AUG)
    trip26 = core.period_total(pkm["lisboa"], 2026, JAN_AUG) / core.period_total(pax["lisboa"], 2026, JAN_AUG)
    print(f"Average trip length (INE pkm / passengers), Jan-Aug: 2025 {trip24:.2f} km, 2026 {trip26:.2f} km")

    # ------------------------------------------------------------------ 7. The rest of the system
    section("7. The rest of public transport in the Lisbon metro area")
    bus = pd.read_csv(DATA / "bus.csv", index_col="year")
    gl, ps = bus["Grande Lisboa"], bus["Península de Setúbal"]
    tot = gl + ps
    print(
        f"Regular bus passengers (INE, thousands): Grande Lisboa {gl[2023]:,.1f} / {gl[2024]:,.1f} / {gl[2025]:,.1f}; "
        f"Península de Setúbal {ps[2023]:,.1f} / {ps[2024]:,.1f} / {ps[2025]:,.1f}"
    )
    print(
        f"2025 vs 2024: Grande Lisboa {pct(core.change(gl[2025], gl[2024]), 1)} ({(gl[2025] - gl[2024]) / 1000:+.1f} m); "
        f"Setúbal {pct(core.change(ps[2025], ps[2024]), 1)} ({(ps[2025] - ps[2024]) / 1000:+.1f} m); "
        f"metro area {pct(core.change(tot[2025], tot[2024]), 1)} ({(tot[2025] - tot[2024]) / 1000:+.1f} m)"
    )
    print(f"2024 vs 2023: metro area buses {pct(core.change(tot[2024], tot[2023]), 1)}")
    river = pd.read_csv(DATA / "river.csv", index_col="year")
    tejo = river["Rio Tejo"]
    print(
        f"Tagus river lines: 2019 {tejo[2019] / 1e6:.2f} m, 2023 {tejo[2023] / 1e6:.2f}, 2024 {tejo[2024] / 1e6:.2f}, 2025 {tejo[2025] / 1e6:.2f} "
        f"({pct(core.change(tejo[2025], tejo[2024]), 1)}, {(tejo[2025] - tejo[2024]) / 1e6:+.2f} m)"
    )
    mst_d = annual.loc[2025, "mst"] - annual.loc[2024, "mst"]
    lis_d = annual.loc[2025, "lisboa"] - annual.loc[2024, "lisboa"]
    oo = tr.OTHER_OPERATORS
    cp25 = oo["cp_lisbon_suburban_2025_validations"][0] / 1e6
    cp24 = cp25 / (1 + oo["cp_lisbon_suburban_2025_change"][0] / 100)
    print(f"CP Lisbon suburban (press, from CP): 2025 {cp25:.2f} m, {oo['cp_lisbon_suburban_2025_change'][0]}% -> about {cp25 - cp24:+.2f} m")
    changes = pd.DataFrame(
        [
            ("Buses, Grande Lisboa", (gl[2025] - gl[2024]) / 1000),
            ("Buses, Península de Setúbal", (ps[2025] - ps[2024]) / 1000),
            ("Metropolitano de Lisboa", lis_d),
            ("Metro Sul do Tejo", mst_d),
            ("Tagus ferries", (tejo[2025] - tejo[2024]) / 1e6),
            ("CP Lisbon suburban trains", cp25 - cp24),
        ],
        columns=["mode", "change_m"],
    )
    changes.round(4).to_csv(RESULTS / "aml_changes_2025.csv", index=False)
    print(f"Sum of the six changes: {changes.change_m.sum():+.2f} m (different counting units; indicative)")
    cm24, cm25 = oo["carris_metropolitana_2024_m"][0], oo["carris_metropolitana_2025_m"][0]
    print(
        f"Carris Metropolitana (press): {cm24} -> {cm25} m, {pct(core.change(cm25, cm24), 1)}; into Lisbon 2025 {oo['carris_metropolitana_2025_into_lisbon_m'][0]} m; "
        f"Jan-May 2026 {oo['carris_metropolitana_jan_may_2026_change'][0]:+.1f}%"
    )
    print(f"Carris (city buses, trams, lifts): 2025 {oo['carris_2025_m'][0]} m, {oo['carris_2025_change'][0]}%; {oo['carris_2025_change'][1]}")
    print(f"Navegante: {oo['navegante_share_of_trips'][1]}")

    # ------------------------------------------------------------------ 8. People and jobs
    section("8. Residents and jobs")
    pop = pd.read_csv(DATA / "population.csv", index_col="municipality")
    from panel import METRO_MUNICIPALITIES

    served = pop.loc[METRO_MUNICIPALITIES].sum()
    rest = pop.drop(index=METRO_MUNICIPALITIES).sum()
    print(
        f"Lisboa municipality: {pop.loc['Lisboa', '2024']:,.0f} -> {pop.loc['Lisboa', '2025']:,.0f} ({pct(core.change(pop.loc['Lisboa', '2025'], pop.loc['Lisboa', '2024']))})"
    )
    print(
        f"Four municipalities with metro stations: {served['2024']:,.0f} -> {served['2025']:,.0f} ({pct(core.change(served['2025'], served['2024']))})"
    )
    print(f"Other 14 municipalities of the metro area: {rest['2024']:,.0f} -> {rest['2025']:,.0f} ({pct(core.change(rest['2025'], rest['2024']))})")
    emp = pd.read_csv(DATA / "employment.csv", index_col="year")
    for g, lab in [("PT1A", "Grande Lisboa"), ("PT1B", "Península de Setúbal"), ("PT", "Portugal")]:
        print(
            f"Employed 15-74, {lab}: 2019 {emp.loc[2019, g]:.1f}, 2024 {emp.loc[2024, g]:.1f}, 2025 {emp.loc[2025, g]:.1f} thousand "
            f"(2025 {pct(core.change(emp.loc[2025, g], emp.loc[2024, g]), 1)}; vs 2019 {pct(core.change(emp.loc[2025, g], emp.loc[2019, g]), 1)})"
        )

    # ------------------------------------------------------------------ 9. Cars
    section("9. Roads")
    mt = tr.MOTORWAY_TMDM
    for road in ("A1", "A2", "A5", "A12"):
        a = np.mean(mt[road]["2025"])
        b = np.mean(mt[road]["2026"])
        print(f"{road}: Apr-Jun 2025 {a:,.0f}, Apr-Jun 2026 {b:,.0f} vehicles a day: {pct(core.change(b, a))}")
    rv = tr.ROAD_VKM_H1
    print(
        f"Main roads, first half: {rv['2025_bn']} -> {rv['2026_bn']} bn vehicle-km ({pct(core.change(rv['2026_bn'], rv['2025_bn']), 1)}); {rv['source']}"
    )
    q2_25 = core.period_total(pax["lisboa"], 2025, (4, 5, 6))
    q2_26 = core.period_total(pax["lisboa"], 2026, (4, 5, 6))
    days_q2 = 91
    lost_per_day = (q2_25 - q2_26) * 1000 / days_q2
    four = sum(np.mean(mt[r]["2025"]) for r in ("A1", "A2", "A5", "A12"))
    four26 = sum(np.mean(mt[r]["2026"]) for r in ("A1", "A2", "A5", "A12"))
    print(
        f"Metro trips lost, Apr-Jun 2026 vs 2025: {(q2_25 - q2_26) / 1000:.2f} m, {lost_per_day:,.0f} a day on average "
        f"({(q2_25 - q2_26) / q2_25 * 100:.1f}% of the metro's trips)"
    )
    print(
        f"Sum of the four motorways' average daily traffic: {four:,.0f} -> {four26:,.0f} ({pct(core.change(four26, four))}, {four26 - four:+,.0f} vehicles a day)"
    )
    for occ in (1.0, 1.3):
        cars = lost_per_day / occ
        print(f"  if every lost trip became a car trip with {occ} people: {cars:,.0f} car trips a day")

    # ------------------------------------------------------------------ 10. Elsewhere
    section("10. Other cities")
    md = tr.MADRID_METRO
    print(f"Madrid metro: {md['2024']:,} -> {md['2025']:,} ({pct(core.change(md['2025'], md['2024']), 1)}); {md['source']}")


if __name__ == "__main__":
    main()
