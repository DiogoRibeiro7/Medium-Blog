"""Every number in the article, printed in order, and the tables behind the figures in results/.

python prices_numbers.py > output.txt

Log changes are reported in log points (100 times the natural log of a ratio), because they add up:
the price gap of two groups is exactly the wage gap plus the productivity contribution plus the
margins contribution. Percentages are given where a reader needs them.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from jsonstat import read_jsonstat
from prices_core import block_identity, covariance_shares, gap, ols_slope, pct, per_year, ratio_range, rmse, substitute_productivity, within

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
RESULTS = HERE / "results"

GOODS = ["A", "B", "C", "D", "E", "F"]
SERVICES = ["G", "H", "I", "J", "K", "M", "N", "O", "P", "Q", "R", "S", "T"]
MARKET = ["G", "H", "I", "J", "K", "M", "N", "R", "S"]
NONMARKET = ["O", "P", "Q"]
ECONOMY = GOODS + SERVICES  # every section except L (real estate, mostly imputed rent) and U
PANEL_SECTIONS = [s for s in ECONOMY if s != "T"]
MS = ["BE", "BG", "CZ", "DK", "DE", "EE", "IE", "EL", "ES", "FR", "HR", "IT", "CY", "LV", "LT", "LU", "HU",
      "MT", "NL", "AT", "PL", "PT", "RO", "SI", "SK", "FI", "SE"]  # fmt: skip
NAMES = {
    "A": "Agriculture", "B": "Mining", "C": "Manufacturing", "D": "Electricity and gas", "E": "Water and waste",
    "F": "Construction", "G": "Trade", "H": "Transport", "I": "Hotels and restaurants", "J": "Information and communication",
    "K": "Finance and insurance", "M": "Professional services", "N": "Administrative services", "O": "Public administration",
    "P": "Education", "Q": "Health and social work", "R": "Arts and recreation", "S": "Other services",
    "T": "Households as employers", "J61": "Telecommunications", "Q86": "Health care", "Q87_Q88": "Residential care and social work",
}  # fmt: skip
Y0, Y1 = 1995, 2024


def lp(x):
    """Log points, one decimal."""
    return f"{100 * x:+.1f}"


def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# --------------------------------------------------------------------------------------- loading
def national_accounts():
    va = read_jsonstat(RAW / "nama_10_a64.json")
    hw = read_jsonstat(RAW / "nama_10_a64_e.json")

    def piv(df):
        return df.pivot_table(index="time", columns="nace_r2", values="value", aggfunc="first").sort_index()

    out = {}
    for geo in sorted(set(va.geo)):
        v = va[va.geo == geo]
        h = hw[hw.geo == geo]
        out[geo] = {
            "cp": piv(v[(v.na_item == "B1G") & (v.unit == "CP_MEUR")]),
            "clv": piv(v[(v.na_item == "B1G") & (v.unit == "CLV20_MEUR")]),
            "d1": piv(v[(v.na_item == "D1") & (v.unit == "CP_MEUR")]),
            "os": piv(v[(v.na_item == "B2A3N") & (v.unit == "CP_MEUR")]),
            "cfc": piv(v[(v.na_item == "P51C") & (v.unit == "CP_MEUR")]),
            "tax": piv(v[(v.na_item == "D29X39") & (v.unit == "CP_MEUR")]),
            "hours": piv(h[h.na_item == "EMP_DC"]),
            "emp_hours": piv(h[h.na_item == "SAL_DC"]),
        }
    return out


def complete(t, cols, y0, y1):
    for k in ("cp", "clv", "d1", "hours", "emp_hours"):
        df = t[k]
        if any(c not in df.columns for c in cols):
            return False
        sub = df.loc[y0:y1, cols]
        if len(sub) != y1 - y0 + 1 or sub.isna().any().any():
            return False
    return True


def hicp():
    h = read_jsonstat(RAW / "prc_hicp_aind.json")
    w = read_jsonstat(RAW / "prc_hicp_inw.json")
    idx = {g: d.pivot_table(index="time", columns="coicop", values="value", aggfunc="first") for g, d in h.groupby("geo")}
    wts = {g: d.pivot_table(index="time", columns="coicop", values="value", aggfunc="first") for g, d in w.groupby("geo")}
    return idx, wts


def hicp_labels():
    import json

    d = json.loads((RAW / "prc_hicp_aind.json").read_text(encoding="utf-8"))
    return d["dimension"]["coicop"]["category"]["label"]


def price_levels():
    p = read_jsonstat(RAW / "prc_ppp_ind.json")
    return p


# ------------------------------------------------------------------------------------------- report
def report_consumer_prices(idx, wts, labels):
    section("1. What households pay: HICP, EU27, 2001-2025")
    eu = idx["EU27_2020"]
    a, b = 2001, 2025
    groups = [
        ("CP00", "All items"), ("SERV", "Services"), ("GD", "Goods"), ("IGD_NNRG", "Non-energy industrial goods"),
        ("NRG", "Energy"), ("FOOD", "Food, alcohol and tobacco"), ("SERV_HOUS", "Services related to housing"),
        ("SERV_TRA", "Services related to transport"), ("SERV_REC", "Recreation, repairs and personal care"),
        ("SERV_MSC", "Miscellaneous services"), ("SERV_COM", "Communication services"),
        ("EDUC_HLTH_SPR", "Education, health and social protection"),
    ]  # fmt: skip
    for code, name in groups:
        s = eu[code]
        x = math.log(s[b] / s[a])
        print(f"{name:45s} {lp(x):>7s} log points  {pct(x):+6.1f}%  {per_year(x, b - a):+.2f} lp a year")
    serv = math.log(eu["SERV"][b] / eu["SERV"][a])
    neig = math.log(eu["IGD_NNRG"][b] / eu["IGD_NNRG"][a])
    gd = math.log(eu["GD"][b] / eu["GD"][a])
    print(f"Services minus non-energy industrial goods: {lp(serv - neig)} log points; relative price x{math.exp(serv - neig):.2f}")
    print(f"Services minus all goods: {lp(serv - gd)} log points")
    w = wts["EU27_2020"]
    print(f"Weight of services in the HICP: {w.loc[a, 'SERV'] / 10:.1f}% in {a}, {w.loc[b, 'SERV'] / 10:.1f}% in {b}")
    print(f"Weight of non-energy industrial goods: {w.loc[a, 'IGD_NNRG'] / 10:.1f}% in {a}, {w.loc[b, 'IGD_NNRG'] / 10:.1f}% in {b}")

    print()
    print(f"Items with an EU27 index from {a} (log change {a}-{b}, percent change, lp a year):")
    rows = []
    for c in eu.columns:
        if not c.startswith("CP") or c == "CP00":
            continue
        s = eu[c].dropna()
        if s.index.min() <= a and b in s.index:
            x = math.log(s[b] / s[a])
            rows.append((c, labels[c], x))
    rows.sort(key=lambda r: -r[2])
    goods_like = {"CP0913", "CP0911", "CP0531_0532", "CP0931", "CP03", "CP0511", "CP0721", "CP0611", "CP01"}
    item_table = []
    for c, name, x in rows:
        kind = "goods" if c in goods_like else "services"
        item_table.append({"code": c, "item": name, "kind": kind, "log_change": x, "change_pct": pct(x)})
        print(f"  {c:12s} {kind:8s} {lp(x):>8s}  {pct(x):+7.1f}%  {per_year(x, b - a):+6.2f}  {name}")
    pd.DataFrame(item_table).to_csv(RESULTS / "hicp_items.csv", index=False)

    print()
    print("Items with an EU27 index only from 2017 (log change 2017-2025):")
    late = []
    for c in eu.columns:
        if not c.startswith("CP"):
            continue
        s = eu[c].dropna()
        if s.index.min() == 2017 and b in s.index:
            late.append((c, labels[c], math.log(s[b] / s[2017])))
    late.sort(key=lambda r: -r[2])
    x_all = math.log(eu["CP00"][b] / eu["CP00"][2017])
    print(f"  {'CP00':12s} {lp(x_all):>8s}  {pct(x_all):+7.1f}%  All items")
    for c, name, x in late:
        print(f"  {c:12s} {lp(x):>8s}  {pct(x):+7.1f}%  {name}")

    print()
    print("Administered prices, EU27, 2002-2025 (first year with all five indices):")
    for code, name in (("AP", "Administered prices"), ("AP_NNRG", "Administered prices, non-energy"),
                       ("TOT_X_AP", "All items excluding administered prices"), ("SERV", "Services"),
                       ("IGD_NNRG", "Non-energy industrial goods")):  # fmt: skip
        x = math.log(eu[code][2025] / eu[code][2002])
        print(f"  {name:45s} {lp(x):>7s}  {per_year(x, 23):+.2f} lp a year")
    for g in ("EU27_2020", "PT"):
        if "AP_NNRG" in wts.get(g, pd.DataFrame()).columns:
            print(f"  Weight of non-energy administered prices, {g}, 2025: {wts[g].loc[2025, 'AP_NNRG'] / 10:.1f}%")

    print()
    print("Portugal, 2001-2025:")
    pt = idx["PT"]
    for code, name in groups[:6] + groups[6:11]:
        x = math.log(pt[code][b] / pt[code][a])
        print(f"  {name:45s} {lp(x):>7s}  {pct(x):+6.1f}%")
    x = math.log(pt["SERV"][b] / pt["SERV"][a]) - math.log(pt["IGD_NNRG"][b] / pt["IGD_NNRG"][a])
    print(f"  Services minus non-energy industrial goods: {lp(x)} log points; relative price x{math.exp(x):.2f}")
    for code in ("CP1111", "CP11201", "CP1211", "CP0432", "CP0622", "CP10"):
        s = pt[code].dropna()
        y = max(a, int(s.index.min()))
        x = math.log(s[b] / s[y])
        print(f"  {code:10s} {y}-{b} {lp(x):>7s}  {pct(x):+6.1f}%  {labels[code]}")
    wp = wts["PT"]
    print(f"  Weight of services in Portugal's HICP: {wp.loc[a, 'SERV'] / 10:.1f}% in {a}, {wp.loc[b, 'SERV'] / 10:.1f}% in {b}")
    print(f"  Weight of restaurants and hotels (CP1111 + CP11201), 2025: {(wp.loc[b, 'CP1111'] + wp.loc[b, 'CP11201']) / 10:.1f}%")
    print(f"  Same weight in the EU27, 2025: {(w.loc[b, 'CP1111'] + w.loc[b, 'CP11201']) / 10:.1f}%")
    return serv - neig


def report_blocks(na):
    section(f"2. What producers charge: the identity for services, EU27, {Y0}-{Y1}")
    t = na["EU27_2020"]
    serv = block_identity(t, SERVICES, Y0, Y1)
    manuf = block_identity(t, ["C"], Y0, Y1)
    goods = block_identity(t, GOODS, Y0, Y1)
    econ = block_identity(t, ECONOMY, Y0, Y1)
    print("Services are sections G to T without real estate (L); goods are A to F; the economy is A to T without L.")
    print(f"Cumulative log change {Y0}-{Y1}:            price    wage    prod  lshare")
    for name, b in (("Services", serv), ("Manufacturing", manuf), ("Goods", goods), ("Economy", econ)):
        r = b.loc[Y1]
        print(f"  {name:38s} {lp(r.price):>7s} {lp(r.wage):>7s} {lp(r['prod']):>7s} {lp(r.lshare):>7s}")
    print(f"Real value added per hour, lp a year: services {per_year(serv.loc[Y1, 'prod'], Y1 - Y0):.2f}, "
          f"manufacturing {per_year(manuf.loc[Y1, 'prod'], Y1 - Y0):.2f}, goods {per_year(goods.loc[Y1, 'prod'], Y1 - Y0):.2f}, "
          f"economy {per_year(econ.loc[Y1, 'prod'], Y1 - Y0):.2f}")  # fmt: skip
    print(f"Adjusted labour share of the economy: {lp(econ.loc[Y1, 'lshare'])} log points over the period")

    print()
    print("Price gap = wage gap + productivity contribution + margins contribution (log points):")
    out = {}
    for name, ref in (("Services vs manufacturing", manuf), ("Services vs goods", goods), ("Services vs economy", econ)):
        g = gap(serv, ref)
        out[name] = g
        r = g.loc[Y1]
        check = r.price - (r.wage + r.productivity + r.margins)
        print(f"  {name:28s} price {lp(r.price):>6s} (= {pct(r.price):+.1f}%)  wage {lp(r.wage):>6s}  productivity {lp(r.productivity):>6s}  "
              f"margins {lp(r.margins):>6s}  check {check:.1e}")  # fmt: skip
    g = out["Services vs manufacturing"]
    g.assign(year=g.index).to_csv(RESULTS / "eu_services_vs_manufacturing.csv", index=False)
    r = g.loc[Y1]
    print(f"  Share of the productivity contribution offset by wages and margins: {100 * -(r.wage + r.margins) / r.productivity:.0f}%")
    print(f"  Wages alone offset {100 * -r.wage / r.productivity:.0f}%, margins alone {100 * -r.margins / r.productivity:.0f}%")

    print()
    print("Services vs manufacturing by year (price gap, log points):")
    print("  " + "  ".join(f"{y}:{lp(g.loc[y, 'price'])}" for y in (2000, 2005, 2008, 2010, 2015, 2019, 2020, 2021, 2022, 2023, 2024)))
    peak = g["price"].idxmax()
    print(f"  Peak {peak} at {lp(g.loc[peak, 'price'])}")

    print()
    print("Subperiods, services vs manufacturing:")
    for a, b in ((1995, 2010), (2010, 2024), (2010, 2019)):
        s2 = block_identity(t, SERVICES, a, b)
        m2 = block_identity(t, ["C"], a, b)
        r2 = gap(s2, m2).loc[b]
        pm, ps = per_year(m2.loc[b, "prod"], b - a), per_year(s2.loc[b, "prod"], b - a)
        print(f"  {a}-{b}: price {lp(r2.price)}  wage {lp(r2.wage)}  productivity {lp(r2.productivity)}  margins {lp(r2.margins)}  "
              f"| productivity lp a year: manufacturing {pm:.2f}, services {ps:.2f}")  # fmt: skip

    print()
    print("Pre-pandemic window, 1995-2019, services vs manufacturing:")
    r3 = gap(block_identity(t, SERVICES, 1995, 2019), block_identity(t, ["C"], 1995, 2019)).loc[2019]
    print(f"  price {lp(r3.price)}  wage {lp(r3.wage)}  productivity {lp(r3.productivity)}  margins {lp(r3.margins)}")

    print()
    print("Composition of value added at current prices (percent):")
    for name, cols in (("Services", SERVICES), ("Manufacturing", ["C"]), ("Goods", GOODS), ("Economy", ECONOMY)):
        parts = []
        for y in (Y0, Y1):
            va = t["cp"].loc[y, cols].sum()
            sh = {k: 100 * t[k].loc[y, cols].sum() / va for k in ("d1", "os", "cfc", "tax")}
            parts.append(
                f"{y}: compensation {sh['d1']:.1f}, net operating surplus and mixed income {sh['os']:.1f}, "
                f"depreciation {sh['cfc']:.1f}, taxes less subsidies {sh['tax']:.1f}"
            )
        print(f"  {name:14s} " + " | ".join(parts))
    for name, cols in (("Services", SERVICES), ("Manufacturing", ["C"]), ("Goods", GOODS), ("Economy", ECONOMY)):
        hs = t["emp_hours"].loc[[Y0, Y1], cols].sum(axis=1) / t["hours"].loc[[Y0, Y1], cols].sum(axis=1)
        print(f"  Employees' share of hours, {name}: {100 * hs[Y0]:.1f}% -> {100 * hs[Y1]:.1f}%")
    return serv, manuf, goods, econ, out


def industry_table(t, y0, y1, sections, reference=ECONOMY):
    """Each section's price, wage, productivity and margins terms against the reference aggregate."""
    econ = block_identity(t, reference, y0, y1).loc[y1]
    rows = []
    shares = t["cp"].loc[y0:y1, sections].mean()
    shares = shares / shares.sum()
    for s in sections:
        r = block_identity(t, [s], y0, y1).loc[y1] - econ
        rows.append({"code": s, "price": r.price, "wage": r.wage, "productivity": -r["prod"], "margins": -r.lshare, "weight": shares.get(s, np.nan)})
    return pd.DataFrame(rows).set_index("code")


def report_industries(na):
    section(f"3. Industry by industry against the economy, EU27, {Y0}-{Y1} (log points)")
    t = na["EU27_2020"]
    tab = industry_table(t, Y0, Y1, ECONOMY)
    extra = industry_table(t, Y0, Y1, ["J61", "Q86", "Q87_Q88"])
    extra["weight"] = np.nan
    full = pd.concat([tab, extra])
    full.assign(name=[NAMES[c] for c in full.index]).to_csv(RESULTS / "eu_industries.csv")
    print(f"{'':10s}{'industry':34s}{'price':>8s}{'wage':>8s}{'prod.':>8s}{'margins':>8s}{'VA share':>9s}")
    for c, r in full.sort_values("price").iterrows():
        w = "" if np.isnan(r.weight) else f"{100 * r.weight:8.1f}%"
        print(f"  {c:8s}{NAMES[c]:34s}{lp(r.price):>8s}{lp(r.wage):>8s}{lp(r.productivity):>8s}{lp(r.margins):>8s}{w:>9s}")
    parts = {"wage": tab.wage, "productivity": tab.productivity, "margins": tab.margins}
    un = covariance_shares(tab.price, parts)
    wt = covariance_shares(tab.price, parts, tab.weight)
    print(f"Covariance shares across the {len(tab)} sections (sum to 1):")
    print("  unweighted: " + ", ".join(f"{k} {v:.2f}" for k, v in un.items()))
    print("  weighted by value added: " + ", ".join(f"{k} {v:.2f}" for k, v in wt.items()))
    b, se, r2 = ols_slope(tab.productivity, tab.price)
    print(f"  Slope of the price on the productivity contribution: {b:.2f} (HC1 s.e. {se:.2f}), R2 {r2:.2f}")
    print(f"  Standard deviations across sections: price {100 * tab.price.std(ddof=0):.1f}, wage {100 * tab.wage.std(ddof=0):.1f}, "
          f"productivity {100 * tab.productivity.std(ddof=0):.1f}, margins {100 * tab.margins.std(ddof=0):.1f}")  # fmt: skip
    rising = tab[tab.price > 0].index.tolist()
    print(f"  Sections whose price rose relative to the economy: {', '.join(rising)} ({len(rising)} of {len(tab)})")
    goods_up = [c for c in rising if c in GOODS]
    serv_down = [c for c in tab[tab.price < 0].index if c in SERVICES]
    print(f"  Goods sections that got dearer: {', '.join(goods_up)}; services sections that got cheaper: {', '.join(serv_down)}")
    return tab


def report_panel(na):
    section(f"4. The same identity in every member state, {Y0}-{Y1}: {len(PANEL_SECTIONS)} sections (A to S without L)")
    rows = []
    kept, dropped = [], []
    for g in MS:
        t = na[g]
        if not complete(t, PANEL_SECTIONS, Y0, Y1):
            dropped.append(g)
            continue
        kept.append(g)
        tab = industry_table(t, Y0, Y1, PANEL_SECTIONS, PANEL_SECTIONS)
        tab["geo"] = g
        rows.append(tab.reset_index())
    p = pd.concat(rows, ignore_index=True)
    p.to_csv(RESULTS / "panel.csv", index=False)
    print(f"Countries with every series for every year: {len(kept)}; left out: {', '.join(dropped)} (no hours or value added by section for {Y0})")
    print(f"Country-industry cells: {len(p)}")
    parts = {"wage": p.wage, "productivity": p.productivity, "margins": p.margins}
    un = covariance_shares(p.price, parts)
    wt = covariance_shares(p.price, parts, p.weight)
    print("Covariance shares (sum to 1):")
    print("  unweighted: " + ", ".join(f"{k} {v:.2f}" for k, v in un.items()))
    print("  weighted by the section's share of its country's value added: " + ", ".join(f"{k} {v:.2f}" for k, v in wt.items()))
    b, se, r2 = ols_slope(p.productivity, p.price)
    print(f"Slope of the price on the productivity contribution: {b:.2f} (HC1 s.e. {se:.2f}), R2 {r2:.2f}")
    bw, sew, r2w = ols_slope(within(p.productivity, p.geo), within(p.price, p.geo))
    print(f"  with country means removed: {bw:.2f} (s.e. {sew:.2f}), R2 {r2w:.2f}")
    same = (np.sign(p.price) == np.sign(p.productivity)).mean()
    print(f"Cells where the price moved in the direction of the productivity contribution: {100 * same:.0f}%")
    print(f"Standard deviations: price {100 * p.price.std(ddof=0):.1f}, wage {100 * p.wage.std(ddof=0):.1f}, "
          f"productivity {100 * p.productivity.std(ddof=0):.1f}, margins {100 * p.margins.std(ddof=0):.1f}")  # fmt: skip
    print(f"Correlations with the productivity contribution: wage {np.corrcoef(p.productivity, p.wage)[0, 1]:.2f}, "
          f"margins {np.corrcoef(p.productivity, p.margins)[0, 1]:.2f}")  # fmt: skip
    by_country = []
    for g, d in p.groupby("geo"):
        sh = covariance_shares(d.price, {"productivity": d.productivity}, d.weight)["productivity"]
        by_country.append((g, sh))
    by_country.sort(key=lambda r: r[1])
    vals = [s for _, s in by_country]
    print(f"Productivity's covariance share country by country (weighted): median {np.median(vals):.2f}, "
          f"lowest {by_country[0][0]} {by_country[0][1]:.2f}, highest {by_country[-1][0]} {by_country[-1][1]:.2f}, "
          f"above 0.5 in {sum(v > 0.5 for v in vals)} of {len(vals)}")  # fmt: skip
    pt = dict(by_country).get("PT")
    print(f"  Portugal: {pt:.2f}")
    return p


def report_wages(tab, p):
    section("5. Baumol's wage assumption: do wages grow alike across industries?")
    print("EU27 sections, wage gap against the economy (log points, 1995-2024), lowest and highest:")
    w = tab.wage.sort_values()
    for c in list(w.index[:5]) + list(w.index[-5:]):
        print(f"  {c:4s}{NAMES[c]:34s}{lp(w[c]):>8s}")
    print(f"Across the 450 country-industry cells, the wage gap's standard deviation is "
          f"{100 * p.wage.std(ddof=0):.1f} log points against {100 * p.productivity.std(ddof=0):.1f} for the productivity contribution "
          f"(ratio {p.wage.std(ddof=0) / p.productivity.std(ddof=0):.2f})")  # fmt: skip
    b, se, r2 = ols_slope(p.productivity, p.wage)
    print(f"Slope of the wage gap on the productivity contribution: {b:.2f} (s.e. {se:.2f}), R2 {r2:.2f}")
    low = tab.loc[["G", "H", "I", "N", "S", "T"], "wage"]
    print(f"Wage gaps in trade, transport, hotels and restaurants, administrative, other and household services: "
          f"{', '.join(f'{c} {lp(v)}' for c, v in low.items())}")  # fmt: skip
    hi = tab.loc[["O", "Q", "D", "B", "A"], "wage"]
    print(f"Wage gaps in public administration, health, energy, mining and agriculture: {', '.join(f'{c} {lp(v)}' for c, v in hi.items())}")


def report_margins(tab, na):
    section("6. Margins: the labour-share term")
    m = tab.margins.sort_values()
    print("EU27 sections, margins contribution (log points, 1995-2024), lowest and highest:")
    for c in list(m.index[:5]) + list(m.index[-6:]):
        print(f"  {c:4s}{NAMES[c]:34s}{lp(m[c]):>8s}")
    t = na["EU27_2020"]
    for name, cols in (("Services", SERVICES), ("Manufacturing", ["C"]), ("Goods", GOODS)):
        s = block_identity(t, cols, Y0, Y1)
        print(f"  {name}: adjusted labour share {lp(s.loc[Y1, 'lshare'])} log points")
    for c in ("D", "K", "H", "B", "F", "C"):
        va0, va1 = t["cp"].loc[Y0, c], t["cp"].loc[Y1, c]
        os0, os1 = t["os"].loc[Y0, c] / va0, t["os"].loc[Y1, c] / va1
        print(f"  {c} {NAMES[c]}: net operating surplus and mixed income {100 * os0:.1f}% -> {100 * os1:.1f}% of value added")


def report_measurement(na):
    section("7. Measurement: market and non-market services against manufacturing, EU27, 1995-2024")
    t = na["EU27_2020"]
    manuf = block_identity(t, ["C"], Y0, Y1)
    serv = block_identity(t, SERVICES, Y0, Y1)
    market = block_identity(t, MARKET, Y0, Y1)
    va24 = t["cp"].loc[Y1, SERVICES].sum()
    for name, cols in (("Market services (G-N, R, S)", MARKET), ("Non-market services (O, P, Q)", NONMARKET), ("Public administration (O)", ["O"]),
                       ("Education (P)", ["P"]), ("Health and social work (Q)", ["Q"]), ("Households as employers (T)", ["T"])):  # fmt: skip
        b = block_identity(t, cols, Y0, Y1)
        r = gap(b, manuf).loc[Y1]
        own = per_year(b.loc[Y1, "prod"], Y1 - Y0)
        share = 100 * t["cp"].loc[Y1, cols].sum() / va24
        print(f"  {name:34s} price {lp(r.price):>6s} wage {lp(r.wage):>6s} productivity {lp(r.productivity):>6s} margins {lp(r.margins):>6s} "
              f"| own productivity {own:+.2f} lp a year | {share:.1f}% of services value added in {Y1}")  # fmt: skip
    print()
    base = gap(serv, manuf).loc[Y1, "price"]
    path = market["prod"]
    for name, cols in (("education and health (P, Q)", ["P", "Q"]), ("public administration, education and health (O, P, Q)", NONMARKET)):
        t2 = substitute_productivity(t, cols, path)
        alt = gap(block_identity(t2, SERVICES, Y0, Y1), manuf).loc[Y1, "price"]
        print(f"If the output of {name} per hour had grown like market services' ({per_year(path.loc[Y1], Y1 - Y0):.2f} lp a year):")
        print(f"  services vs manufacturing price gap {lp(alt)} instead of {lp(base)}; the difference is {100 * (base - alt):.1f} log points, "
              f"{100 * (base - alt) / base:.0f}% of the gap")  # fmt: skip
    t3 = substitute_productivity(t, NONMARKET, block_identity(t, ["C"], Y0, Y1)["prod"])
    alt = gap(block_identity(t3, SERVICES, Y0, Y1), manuf).loc[Y1, "price"]
    print(f"If it had grown like manufacturing's ({per_year(manuf.loc[Y1, 'prod'], Y1 - Y0):.2f} lp a year): gap {lp(alt)}")
    print()
    tel = gap(block_identity(t, ["J61"], Y0, Y1), block_identity(t, ECONOMY, Y0, Y1)).loc[Y1]
    own = per_year(block_identity(t, ["J61"], Y0, Y1).loc[Y1, "prod"], Y1 - Y0)
    print(f"Telecommunications (J61) against the economy: price {lp(tel.price)} ({pct(tel.price):+.0f}%), productivity {lp(tel.productivity)}, "
          f"wage {lp(tel.wage)}, margins {lp(tel.margins)}; own productivity {own:.2f} lp a year")  # fmt: skip


def report_price_levels(p, na):
    section("8. Across countries: price levels of goods and services (EU27 = 100)")
    p = p[p.geo.isin(MS)]
    res = None
    for y in (2003, 2024):
        q = p[p.time == y]
        pli = q[q.na_item == "PLI_EU27_2020"].pivot_table(index="geo", columns="ppp_cat", values="value", aggfunc="first")
        gdp = q[(q.na_item == "VI_PPS_EU27_2020_HAB") & (q.ppp_cat == "GDP")].set_index("geo")["value"]
        x = np.log(gdp.reindex(pli.index))
        print(f"{y}: {len(pli)} member states")
        for code, name in (("P01", "Goods"), ("P0101", "Consumer goods"), ("P02", "Services"), ("P0201", "Consumer services"),
                           ("P0202", "Government services"), ("A0111", "Restaurants and hotels"), ("A0106", "Health"),
                           ("A0110", "Education"), ("A0103", "Clothing and footwear"), ("A010901", "Audio-visual and IT equipment")):  # fmt: skip
            v = pli[code]
            b, se, _ = ols_slope(x, np.log(v))
            keep = ~v.index.isin(["LU", "IE"])
            b2, _, _ = ols_slope(x[keep], np.log(v[keep]))
            print(f"  {name:32s} lowest {v.min():6.1f} ({v.idxmin()})  highest {v.max():6.1f} ({v.idxmax()})  ratio {ratio_range(v):4.2f}  "
                  f"elasticity to GDP per head {b:.2f} (s.e. {se:.2f}; {b2:.2f} without LU and IE)  PT {v['PT']:.1f}")  # fmt: skip
        print(f"  GDP per head in PPS, Portugal: {gdp['PT']:.0f}")
        if y == 2024:
            res = pd.DataFrame({"gdp_pc": gdp, "goods": pli["P01"], "services": pli["P02"], "consumer_services": pli["P0201"]})
    # services price level and hourly pay; households as employers (T) left out, as several countries lack it
    pay = {}
    cols = [s for s in SERVICES if s != "T"]
    for g in MS:
        t = na[g]
        pay[g] = t["d1"].loc[2024, cols].sum(min_count=len(cols)) / t["emp_hours"].loc[2024, cols].sum(min_count=len(cols)) * 1000
    pay = pd.Series(pay).dropna()
    res["services_pay_per_hour"] = pay
    res.index.name = "geo"
    res.to_csv(RESULTS / "price_levels_2024.csv")
    r = res.dropna()
    b, se, r2 = ols_slope(np.log(r.services_pay_per_hour), np.log(r.services))
    hp = r.services_pay_per_hour
    print(f"2024, {len(r)} member states: compensation per employee hour in services (G to S without L), euro: lowest {hp.min():.1f} "
          f"({hp.idxmin()}), highest {hp.max():.1f} ({hp.idxmax()}), PT {hp['PT']:.1f}")  # fmt: skip
    print(f"  Elasticity of the services price level to services hourly pay: {b:.2f} (s.e. {se:.2f}), R2 {r2:.2f}")
    a0 = np.log(r.services).mean() - b * np.log(r.services_pay_per_hour).mean()
    fit_pt = math.exp(a0 + b * math.log(hp["PT"]))
    above = 100 * (r.services["PT"] / fit_pt - 1)
    print(f"  Portugal: services price level {r.services['PT']:.1f}, fitted from its hourly pay {fit_pt:.1f} ({above:+.0f}%)")
    b, se, r2 = ols_slope(np.log(r.services_pay_per_hour), np.log(r.goods))
    print(f"  Elasticity of the goods price level to services hourly pay: {b:.2f} (s.e. {se:.2f}), R2 {r2:.2f}")
    print(f"  Ratio of highest to lowest hourly pay: {ratio_range(r.services_pay_per_hour):.1f}")


def report_portugal(na):
    section(f"9. Portugal, {Y0}-{Y1}")
    t = na["PT"]
    serv = block_identity(t, SERVICES, Y0, Y1)
    for name, ref in (("manufacturing", block_identity(t, ["C"], Y0, Y1)), ("goods", block_identity(t, GOODS, Y0, Y1)),
                      ("the economy", block_identity(t, ECONOMY, Y0, Y1))):  # fmt: skip
        r = gap(serv, ref).loc[Y1]
        print(f"  Services vs {name:14s} price {lp(r.price):>6s}  wage {lp(r.wage):>6s}  "
              f"productivity {lp(r.productivity):>6s}  margins {lp(r.margins):>6s}")  # fmt: skip
    print(f"  Real value added per hour, lp a year: services {per_year(serv.loc[Y1, 'prod'], Y1 - Y0):.2f}, "
          f"manufacturing {per_year(block_identity(t, ['C'], Y0, Y1).loc[Y1, 'prod'], Y1 - Y0):.2f}")  # fmt: skip
    tab = industry_table(t, Y0, Y1, PANEL_SECTIONS, PANEL_SECTIONS)
    print(
        "  Hotels and restaurants (I) against the economy: "
        + ", ".join(f"{k} {lp(v)}" for k, v in tab.loc["I", ["price", "wage", "productivity", "margins"]].items())
    )
    print(
        "  Education (P) against the economy: "
        + ", ".join(f"{k} {lp(v)}" for k, v in tab.loc["P", ["price", "wage", "productivity", "margins"]].items())
    )
    for g in ("DE", "FR", "ES", "IT", "PL"):
        tg = na[g]
        r = gap(block_identity(tg, SERVICES, Y0, Y1), block_identity(tg, ["C"], Y0, Y1)).loc[Y1]
        print(f"  {g}, services vs manufacturing: price {lp(r.price)}  wage {lp(r.wage)}  productivity {lp(r.productivity)}  margins {lp(r.margins)}")


def report_forecast(na):
    section("10. Can past drift forecast future drift? Country-industry cells, 1995-2010 against 2010-2024")
    a_rows, b_rows = [], []
    for g in MS:
        t = na[g]
        if not complete(t, PANEL_SECTIONS, 1995, 2024):
            continue
        a = industry_table(t, 1995, 2010, PANEL_SECTIONS, PANEL_SECTIONS)
        b = industry_table(t, 2010, 2024, PANEL_SECTIONS, PANEL_SECTIONS)
        a["geo"] = g
        b["geo"] = g
        a_rows.append(a.reset_index())
        b_rows.append(b.reset_index())
    a = pd.concat(a_rows, ignore_index=True)
    b = pd.concat(b_rows, ignore_index=True)
    first = 100 * a.price / 15
    second = 100 * b.price / 14
    pd.DataFrame({"geo": a.geo, "code": a.code, "drift_1995_2010": first, "drift_2010_2024": second, "weight": b.weight}).to_csv(
        RESULTS / "persistence.csv", index=False
    )
    print(f"Cells: {len(a)}. Drift = relative price change against the country's economy, lp a year.")
    print(f"Correlation of the two periods' drift: {np.corrcoef(first, second)[0, 1]:.2f}")
    s, se, r2 = ols_slope(first, second)
    print(f"Slope of later drift on earlier drift: {s:.2f} (s.e. {se:.2f}), R2 {r2:.2f}")
    print(f"Mean absolute drift: {np.abs(first).mean():.2f} (1995-2010), {np.abs(second).mean():.2f} (2010-2024) lp a year")
    print("Forecast error (RMSE, lp a year) for 2010-2024 drift:")
    print(f"  no drift (zero):                 {rmse(second):.2f}")
    for k in (0.25, 0.5, 1.0):
        print(f"  {k:.2f} x the 1995-2010 drift:       {rmse(second - k * first):.2f}")
    print(f"Correlation of the productivity contribution across the two periods: {np.corrcoef(a.productivity, b.productivity)[0, 1]:.2f}")
    cw = np.cov(first, second, aweights=b.weight)
    print(f"Correlation of the two periods' drift, weighted by value added: {cw[0, 1] / math.sqrt(cw[0, 0] * cw[1, 1]):.2f}")
    t = na["EU27_2020"]
    ea = industry_table(t, 1995, 2010, ECONOMY)
    eb = industry_table(t, 2010, 2024, ECONOMY)
    f1, f2 = 100 * ea.price / 15, 100 * eb.price / 14
    print(f"EU27 sections ({len(ea)}): correlation {np.corrcoef(f1, f2)[0, 1]:.2f}; RMSE zero {rmse(f2):.2f}, persistence {rmse(f2 - f1):.2f}, "
          f"half {rmse(f2 - 0.5 * f1):.2f}")  # fmt: skip
    same = (np.sign(f1) == np.sign(f2)).mean()
    print(f"  sections whose drift kept its sign: {100 * same:.0f}%")
    pe = covariance_shares(eb.price, {"productivity": eb.productivity}, eb.weight)["productivity"]
    print(f"  but within 2010-2024 the productivity term still carries {pe:.2f} of the cross-section variance (weighted)")


def main():
    RESULTS.mkdir(exist_ok=True)
    print("services-relative-prices: why services get dearer, EU27 and member states")
    print("Data: Eurostat snapshot eurostat-services-relative-prices in DiogoRibeiro7/data (see ../data_sources).")
    idx, wts = hicp()
    labels = hicp_labels()
    na = national_accounts()
    consumer_gap = report_consumer_prices(idx, wts, labels)
    serv, manuf, goods, econ, gaps = report_blocks(na)
    producer = gap(block_identity(na["EU27_2020"], SERVICES, 2001, 2024), block_identity(na["EU27_2020"], ["C"], 2001, 2024)).loc[2024, "price"]
    eu = idx["EU27_2020"]
    consumer_24 = math.log(eu["SERV"][2024] / eu["SERV"][2001]) - math.log(eu["IGD_NNRG"][2024] / eu["IGD_NNRG"][2001])
    print()
    print("Consumer against producer prices, 2001-2024:")
    print(f"  HICP services minus non-energy industrial goods {lp(consumer_24)}; value added prices, services minus manufacturing {lp(producer)}")
    print(f"  (HICP gap 2001-2025: {lp(consumer_gap)})")
    tab = report_industries(na)
    p = report_panel(na)
    report_wages(tab, p)
    report_margins(tab, na)
    report_measurement(na)
    report_price_levels(price_levels(), na)
    report_portugal(na)
    report_forecast(na)


if __name__ == "__main__":
    main()
