"""Every number in the article.

python pension_numbers.py > output.txt

Also writes the tables behind the figures to results/.
"""

import csv
import math
from pathlib import Path

import pension_core as pc

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
HORIZON = 26  # years 0..25 from 65: real value at 75, 85 and 90
AGES = {75: 10, 85: 20, 90: 25}


def head(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def dist(values, digits=1):
    q = [pc.quantile(values, p) for p in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)]
    return "min {} | p10 {} | p25 {} | median {} | p75 {} | p90 {} | max {}".format(*(pct(x, digits) for x in q))


def write(name, header, rows):
    with open(RES / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([f"{x:.6f}" if isinstance(x, float) else x for x in r])


def solve(f, lo=-0.05, hi=0.2):
    """Root of an increasing function of a rate by bisection."""
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def name(c):
    return pc.NAMES[c]


def main():
    RES.mkdir(exist_ok=True)
    d = pc.load()
    life = d.qx[(pc.LIFE_GEO, pc.LIFE_YEAR, "T")]
    S = pc.survival(life)

    # ------------------------------------------------------------------------------------------------ 1
    head("1. Data and conventions")
    print(f"countries: {len(pc.PANEL)} (OECD members by 1973, without Turkey): {', '.join(pc.PANEL)}")
    first = {c: min(y for y in d.cpi[c] if y >= pc.FIRST_START) for c in pc.PANEL}
    last = {c: max(d.cpi[c]) for c in pc.PANEL}
    late = sorted((c, y) for c, y in first.items() if y > pc.FIRST_START)
    print(f"consumer prices from {pc.FIRST_START} for all but: {', '.join(f'{name(c)} {y}' for c, y in late)}")
    print(f"last year of prices: 2025 for all but: {', '.join(f'{name(c)} {y}' for c, y in last.items() if y < 2025)}")
    w26 = pc.windows(d.cpi, pc.PANEL, HORIZON)
    w30 = pc.windows(d.cpi, pc.PANEL, pc.SPAN)
    print(f"pension starts with 25 years of prices after them (real value at 90): {len(w26)} country-years")
    print(f"pension starts with 29 years after them (ages 65 to 94, for lifetime measures): {len(w30)} country-years")
    print(f"start years in the lifetime set: {min(s for _, s in w30)} to {max(s for _, s in w30)}")
    print(f"life table: Eurostat {pc.LIFE_GEO} {pc.LIFE_YEAR}, both sexes; payments counted at ages 65 to 94")
    total, s95 = pc.curtate_check(life)
    e65 = d.e65[(pc.LIFE_GEO, pc.LIFE_YEAR, "T")]
    print(
        f"sum of survival weights, ages 65-94: {math.fsum(S):.2f} years; Eurostat life expectancy at 65: {e65:.1f};"
        f" years lived 65-94 by the table {total:.2f}, alive at 95 {pct(s95)}"
    )
    print(
        "alive at 75, 85, 90 and 94, given alive at 65: " + ", ".join(f"{a} {pct(S[a - 65])}" for a in (75, 85, 90, 94))
    )
    for sex in ("F", "M"):
        s = pc.survival(d.qx[(pc.LIFE_GEO, pc.LIFE_YEAR, sex)])
        print(
            f"  {sex}: alive at 85 {pct(s[20])}, at 90 {pct(s[25])}; life expectancy at 65"
            f" {d.e65[(pc.LIFE_GEO, pc.LIFE_YEAR, sex)]:.1f}"
        )

    # ------------------------------------------------------------------------------------------------ 2
    head("2. Steady inflation: what a fixed payment is worth later")
    print(
        f"a fixed 1,000 a month at 2% inflation: worth {1000 / 1.02:,.2f} after one year, {1000 / 1.02**20:,.2f}"
        f" after 20"
    )
    for infl in (0.02, 0.03, 0.05, 0.10):
        vals = ", ".join(f"at {a} {pct((1 + infl) ** -k)}" for a, k in AGES.items())
        life_v = pc.weighted_real([(1 + infl) ** -k for k in range(pc.SPAN)], S)
        half = math.log(2) / math.log(1 + infl)
        quarter = math.log(1 / 0.75) / math.log(1 + infl)
        print(
            f"inflation {pct(infl, 0)} a year: {vals}; lifetime real income {pct(life_v)};"
            f" loses a quarter after {quarter:.1f} years, half after {half:.1f}"
        )

    # ------------------------------------------------------------------------------------------------ 3
    head("3. A fixed pension replayed on each country's prices")
    paths = {(c, s): pc.real_path(d.cpi[c], s, HORIZON) for c, s in w26}
    for a, k in AGES.items():
        v = [p[k] for p in paths.values()]
        print(f"real value at {a}: {dist(v)}")
        print(
            f"   below half at {a}: {pct(pc.share(v, lambda x: x < 0.5))}; below a third:"
            f" {pct(pc.share(v, lambda x: x < 1 / 3))}"
        )
    v85 = {key: p[20] for key, p in paths.items()}
    worst = sorted(v85.items(), key=lambda x: x[1])
    print("lowest real value at 85, outside Iceland:")
    for (c, s), x in [w for w in worst if w[0][0] != "ISL"][:6]:
        print(f"   {name(c)} {s}: {pct(x)} (average inflation {pct(pc.average_inflation(d.cpi[c], s, 20))} a year)")
    print("lowest in Iceland: " + ", ".join(f"{s} {pct(x, 2)}" for (c, s), x in worst[:3]))
    print("share of starts before 1980 that lost more than 90% by 85:")
    for c in ("ISL", "PRT", "GRC", "ESP", "ITA"):
        p = d.cpi[c]
        starts = [s for s in p if pc.FIRST_START <= s < 1980 and s + 20 in p]
        print(f"   {name(c)}: {pct(pc.share(starts, lambda s, p=p: p[s] / p[s + 20] < 0.1))} of {len(starts)}")
    best = sorted(v85.items(), key=lambda x: -x[1])[:4]
    print("highest real value at 85: " + "; ".join(f"{name(c)} {s} {pct(x)}" for (c, s), x in best))
    print("median real value at 85 by start decade, with the lowest and highest:")
    for dec in range(1950, 2000, 10):
        vv = [x for (c, s), x in v85.items() if dec <= s < dec + 10]
        if vv:
            print(
                f"   {dec}s: {len(vv):3d} starts, median {pct(pc.quantile(vv, 0.5))}, range {pct(min(vv))} to"
                f" {pct(max(vv))}"
            )
    fan = []
    for k in range(HORIZON):
        col = [p[k] for p in paths.values()]
        fan.append((k, *(pc.quantile(col, q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)), 1.02**-k))
    write("fan", ["k", "p10", "p25", "p50", "p75", "p90", "steady2"], fan)
    write(
        "real_paths",
        ["country", "start", *[f"k{k}" for k in range(HORIZON)]],
        [(c, s, *p) for (c, s), p in sorted(paths.items())],
    )
    heat = []
    for c in pc.PANEL:
        for s in sorted(d.cpi[c]):
            if s >= pc.FIRST_START and pc.complete(d.cpi[c], s, 21):
                heat.append((c, s, d.cpi[c][s] / d.cpi[c][s + 20]))
    write("real_at_85", ["country", "start", "real85"], heat)

    euro_era = []
    starters = [c for c in pc.PANEL if 1999 in d.cpi[c] and 2025 in d.cpi[c]]
    for k in range(2025 - 1999 + 1):
        euro_era.append((k, pc.quantile([d.cpi[c][1999] / d.cpi[c][1999 + k] for c in starters], 0.5)))
    write("euro_era_1999", ["k", "median"], euro_era)
    print("pensions started in the euro era, real value by 2025 (prices of each country):")
    for s in (1999, 2005, 2010, 2015, 2020):
        v = [d.cpi[c][s] / d.cpi[c][2025] for c in pc.PANEL if s in d.cpi[c] and 2025 in d.cpi[c]]
        print(
            f"   started {s}: median {pct(pc.quantile(v, 0.5))}, range {pct(min(v))} to {pct(max(v))} ({len(v)}"
            f" countries)"
        )

    # ------------------------------------------------------------------------------------------------ 4
    head("4. Counting only the years a pensioner is alive to be paid")
    life_fixed = {key: pc.weighted_real(pc.real_path(d.cpi[key[0]], key[1], pc.SPAN), S) for key in w30}
    v = list(life_fixed.values())
    print(f"share of lifetime real income kept by a fixed pension (survival-weighted, ages 65-94): {dist(v)}")
    print(
        f"   so the median pensioner loses {pct(1 - pc.quantile(v, 0.5))}, the worst tenth at least"
        f" {pct(1 - pc.quantile(v, 0.1))}"
    )
    v85_30 = [pc.real_path(d.cpi[c], s, pc.SPAN)[20] for c, s in w30]
    print(f"same starts, real value at 85: median {pct(pc.quantile(v85_30, 0.5))}")
    weights_after_80 = math.fsum(S[15:]) / math.fsum(S)
    print(f"share of the survival-weighted payment years that fall after 80: {pct(weights_after_80)}")
    lost = [
        (
            math.fsum(w * (1 - x) for w, x in zip(S[15:], pc.real_path(d.cpi[c], s, pc.SPAN)[15:], strict=True))
            / math.fsum(w * (1 - x) for w, x in zip(S, pc.real_path(d.cpi[c], s, pc.SPAN), strict=True))
        )
        for c, s in w30
    ]
    print(f"share of the lifetime loss that falls after 80: median {pct(pc.quantile(lost, 0.5))}")
    print("by start decade (median, worst):")
    for dec in range(1950, 2000, 10):
        vv = [x for (c, s), x in life_fixed.items() if dec <= s < dec + 10]
        print(f"   {dec}s: {len(vv):3d} starts, median {pct(pc.quantile(vv, 0.5))}, lowest {pct(min(vv))}")
    print("by country (median over start years, lowest):")
    by_country = []
    for c in pc.PANEL:
        vv = [x for (cc, s), x in life_fixed.items() if cc == c]
        lo_s = min((x, s) for (cc, s), x in life_fixed.items() if cc == c)
        by_country.append((pc.quantile(vv, 0.5), c, lo_s))
    for med, c, (lo, s) in sorted(by_country):
        print(f"   {name(c):15s} median {pct(med)}, lowest {pct(lo)} ({s})")
    for sex in ("F", "M"):
        s_sex = pc.survival(d.qx[(pc.LIFE_GEO, pc.LIFE_YEAR, sex)])
        vv = [pc.weighted_real(pc.real_path(d.cpi[c], s, pc.SPAN), s_sex) for c, s in w30]
        print(
            f"   with the {'women' if sex == 'F' else 'men'}'s table: median kept {pct(pc.quantile(vv, 0.5))}, p10"
            f" {pct(pc.quantile(vv, 0.1))}"
        )

    # ------------------------------------------------------------------------------------------------ 5
    head("5. Partial indexation rules, replayed on the same histories")
    rule_rows, rule_summary = [], []
    cut50 = pc.quantile([1 - x for x in life_fixed.values()], 0.5)
    cut90 = pc.quantile([1 - x for x in life_fixed.values()], 0.9)
    calm = [key for key in w30 if 1 - life_fixed[key] < cut50]
    rough = [key for key in w30 if 1 - life_fixed[key] >= cut90]
    print(f"calmer half of histories: {len(calm)} starts; worst tenth (largest loss without indexation): {len(rough)}")
    rules = ["fixed", "half", "minus05", "cap25", "cap3", "cap5", "step2", "step3", "full"]
    fixed_loss = {key: 1 - x for key, x in life_fixed.items()}
    for rule in rules:
        vals = {key: pc.weighted_real(pc.indexed_path(d.cpi[key[0]], key[1], pc.SPAN, rule), S) for key in w30}
        v = list(vals.values())
        at85 = [pc.indexed_path(d.cpi[c], s, pc.SPAN, rule)[20] for c, s in w30]
        print(f"{pc.RULE_LABELS[rule]:26s} lifetime kept: {dist(v)}")
        print(f"{'':26s} real value at 85: median {pct(pc.quantile(at85, 0.5))}, p10 {pct(pc.quantile(at85, 0.1))}")
        m_calm = pc.quantile([vals[k] for k in calm], 0.5)
        m_rough = pc.quantile([vals[k] for k in rough], 0.5)
        print(f"{'':26s} lifetime kept: median {pct(m_calm)} in the calmer half, {pct(m_rough)} in the worst tenth")
        if rule not in ("fixed", "full"):
            cover = lambda keys: pc.quantile([(vals[k] - life_fixed[k]) / fixed_loss[k] for k in keys], 0.5)  # noqa: E731, B023
            print(
                f"{'':26s} share of the fixed pension's loss it removes: median {pct(cover(calm))} in the calmer half"
                f" of histories, {pct(cover(rough))} in the worst tenth"
            )
        rule_summary.append((rule, pc.quantile(v, 0.1), pc.quantile(v, 0.5), pc.quantile(v, 0.9), m_calm, m_rough))
        for (c, s), x in vals.items():
            rule_rows.append((rule, c, s, x))
    write("rules_lifetime", ["rule", "country", "start", "kept"], rule_rows)
    write("rules_summary", ["rule", "p10", "median", "p90", "calm_median", "worst_tenth_median"], rule_summary)
    # where a capped rule loses: inflation years above the cap
    above = []
    for c, s in w30:
        p = d.cpi[c]
        infl = [p[s + k] / p[s + k - 1] - 1 for k in range(1, pc.SPAN)]
        if any(x > 0.025 for x in infl):
            above.append(sum(1 for x in infl if x > 0.025) / len(infl))
    print(f"share of years with inflation above 2.5%, across histories: median {pct(pc.quantile(above, 0.5))}")
    ea = d.hicp["EA20"]
    years = [(y, ea[y] / ea[y - 1] - 1) for y in range(2011, 2026)]
    print("euro-area HICP inflation, 2011-2025: " + ", ".join(f"{y} {pct(x)}" for y, x in years))
    print(f"   above 2.5%: {', '.join(str(y) for y, x in years if x > 0.025)}")
    capped = math.prod(1 + min(max(x, 0), 0.025) for y, x in years if y >= 2021) / math.prod(
        1 + x for y, x in years if y >= 2021
    )
    print(f"a pension indexed to euro-area prices with a 2.5% cap, 2021-2025: real value at the end {pct(capped)}")

    # ------------------------------------------------------------------------------------------------ 6
    head("6. What indexation costs at the start: first payment per 100,000 of premium at 65")
    premium = 100_000
    for nominal, real in ((0.03, 0.01), (0.035, 0.015), (0.04, 0.02)):
        level = pc.first_payment(premium, S, nominal)
        indexed = pc.first_payment(premium, S, real)
        priced = pc.breakeven(nominal, real)
        step3 = pc.first_payment(premium, S, nominal, 0.03)
        step2 = pc.first_payment(premium, S, nominal, 0.02)
        print(
            f"nominal yield {pct(nominal)}, real {pct(real)} (inflation priced {pct(priced, 2)}): level {level:,.0f},"
            f" indexed {indexed:,.0f} ({pct(indexed / level - 1)}), rising 2% {step2:,.0f}"
            f" ({pct(step2 / level - 1)}), rising 3% {step3:,.0f} ({pct(step3 / level - 1)}) a year"
        )
        # crossing points if inflation turns out equal to the priced inflation
        k_pay = math.log(level / indexed) / math.log(1 + priced)
        cum_l = cum_i = 0.0
        k_cum = None
        for k in range(60):
            cum_l += level
            cum_i += indexed * (1 + priced) ** k
            if cum_i >= cum_l and k_cum is None:
                k_cum = k
        s_pay = S[round(k_pay)]
        print(
            f"   at that inflation the indexed payment overtakes the level one after {k_pay:.1f} years"
            f" (age {65 + round(k_pay)}, alive {pct(s_pay)}); the payments received add up to more from age"
            f" {65 + k_cum}"
            f" (alive {pct(S[k_cum])})"
        )

    level_q, rpi_q = 7483, 5186  # Legal & General, single life, age 65, 100,000 pounds, rates at 16 November 2026
    print(
        f"market quote (UK, Legal & General, 16 November 2026): level {level_q:,}, RPI-linked {rpi_q:,}"
        f" ({pct(rpi_q / level_q - 1)})"
    )
    implied_i = solve(lambda r: pc.first_payment(premium, S, r) - level_q)
    implied_r = solve(lambda r: pc.first_payment(premium, S, r) - rpi_q)
    print(
        f"   on the EU table these quotes imply yields of {pct(implied_i, 2)} and {pct(implied_r, 2)}:"
        f" inflation priced at {pct(pc.breakeven(implied_i, implied_r), 2)} a year"
    )

    # ------------------------------------------------------------------------------------------------ 7
    head("7. Level against indexed annuities bought in each year, with that year's long-term yield")
    ex = []
    for c in pc.PANEL:
        for s in sorted(d.rates.get(c, {})):
            if s >= pc.FIRST_START and s in d.cpi[c] and pc.complete(d.cpi[c], s, pc.SPAN):
                i = d.rates[c][s] / 100
                vals = {r: pc.level_value(d.cpi[c], s, i, S, r) for r in (0.01, 0.02, 0.03)}
                realised = pc.average_inflation(d.cpi[c], s, 20)
                ex.append((c, s, i, pc.breakeven(i), realised, vals[0.01], vals[0.02], vals[0.03]))
    write(
        "level_vs_indexed",
        [
            "country",
            "start",
            "nominal",
            "priced_inflation",
            "realised_inflation_20y",
            "value_r1",
            "value_r2",
            "value_r3",
        ],
        ex,
    )
    print("10-year government bond yields in the purchase years, by decade (p10, median, p90):")
    for dec in range(1950, 2000, 10):
        ys = [e[2] for e in ex if dec <= e[1] < dec + 10]
        print(f"   {dec}s: {pct(pc.quantile(ys, 0.1))}, {pct(pc.quantile(ys, 0.5))}, {pct(pc.quantile(ys, 0.9))}")
    countries = sorted({c for c, *_ in ex})
    print(
        f"purchases: {len(ex)} country-years, {len(countries)} countries,"
        f" {min(e[1] for e in ex)}-{max(e[1] for e in ex)}"
    )
    print(f"   countries: {', '.join(countries)}")
    for col, r in ((6, 0.02), (5, 0.01), (7, 0.03)):
        v = [e[col] for e in ex]
        print(
            f"real rate {pct(r, 0)}: level annuity worth {dist(v)} of the indexed one;"
            f" level better in {pct(pc.share(v, lambda x: x > 1))}"
        )
    print("by start decade, real rate 2% (median value of the level annuity, share where it beat the indexed one):")
    for dec in range(1950, 2000, 10):
        vv = [e[6] for e in ex if dec <= e[1] < dec + 10]
        if vv:
            print(
                f"   {dec}s: {len(vv):3d} purchases, median {pct(pc.quantile(vv, 0.5))}, level better in"
                f" {pct(pc.share(vv, lambda x: x > 1))}"
            )
    lo = min(ex, key=lambda e: e[6])
    hi = max(ex, key=lambda e: e[6])
    for tag, e in (("worst", lo), ("best", hi)):
        print(
            f"{tag} for the level buyer: {name(e[0])} {e[1]}, yield {pct(e[2])}, priced inflation {pct(e[3])},"
            f" realised over 20 years {pct(e[4])}, value {pct(e[6])}"
        )
    gap = [e[4] - e[3] for e in ex]
    print(f"realised inflation (20 years) minus priced inflation: {dist(gap, 2)}")

    # ------------------------------------------------------------------------------------------------ 8
    head("8. Price indexation keeps purchasing power, not the place next to wages")
    rel = []
    for c in pc.PANEL:
        w = d.wages.get(c, {})
        end = 2025 if 2025 in w else max(w)
        if 2000 in w:
            rel.append((w[2000] / w[end], c, end))
    for x, c, end in sorted(rel):
        print(
            f"   {name(c):15s} average real wage 2000-{end}: {pct(1 / x - 1):>7s}; price-indexed pension relative"
            f" to it: {pct(x)}"
        )
    print(
        f"median: price-indexed pension started 2000 is worth {pct(pc.quantile([r[0] for r in rel], 0.5))} of its"
        f" starting ratio to the average wage"
    )

    # ------------------------------------------------------------------------------------------------ 9
    head("9. The recent shock: a fixed income started in 2020, valued in 2025 (Eurostat HICP)")
    rows = []
    for g in sorted(d.hicp):
        h = d.hicp[g]
        if all(y in h for y in (2015, 2020, 2025)):
            rows.append((h[2020] / h[2025], h[2015] / h[2020], g))
    for now, before, g in sorted(rows):
        print(f"   {g:10s} 2020 to 2025: {pct(now - 1):>7s}; 2015 to 2020: {pct(before - 1):>6s}")
    eu = [r for r in rows if r[2] not in ("EA20", "EU27_2020")]
    print(
        f"27 member states: median loss {pct(1 - pc.quantile([r[0] for r in eu], 0.5))} in 2020-2025,"
        f" {pct(1 - pc.quantile([r[1] for r in eu], 0.5))} in 2015-2020"
    )

    # ------------------------------------------------------------------------------------------------ 10
    head("10. Portugal: one general-regime pension rebuilt from the annual updates, 2008-2026")
    pt = pc.pt_pension(d.pt_updates, 1000.0)
    cpi = d.cpi["PRT"]
    low = pc.cumulative(pc.pt_first_bracket(d.pt_updates), 2008)
    out = []
    for y in range(2008, 2027):
        jan, jul, avg = pt[y]
        real = avg / 1000.0 * cpi[2008] / cpi[y] if y in cpi else math.nan
        low_real = low[y] * cpi[2008] / cpi[y] if y in cpi else math.nan
        out.append((y, jan, jul, avg, real, low[y], low_real))
        rr = f"real {pct(real)}" if y in cpi else "real n/a"
        print(
            f"   {y}: January {jan:8.2f}, from July {jul:8.2f}, year average {avg:8.2f}; {rr};"
            f" lowest bracket's percentages {pct(low[y] - 1):>6s}" + (f", real {pct(low_real)}" if y in cpi else "")
        )
    write("portugal_pension", ["year", "january", "july", "average", "real", "lowest_index", "lowest_real"], out)
    print("years where the ordinance's minimum rise in euros was larger than the percentage rise:")
    for y in range(2009, 2027):
        b = pc.pt_bracket(pc.pt_update_rows(d.pt_updates, y, "jan"), pt[y - 1][1])
        by_rate = float(b["rate"]) / 100 * pt[y - 1][1]
        if b["minimum"] and float(b["minimum"]) > by_rate:
            print(f"   {y}: {b['rate']}% of {pt[y - 1][1]:.2f} = {by_rate:.2f} euros; minimum {b['minimum']} euros")
    reals = [(r[4], r[0]) for r in out if not math.isnan(r[4])]
    lo = min(reals)
    print(f"lowest real value: {pct(lo[0])} in {lo[1]}; in 2025: {pct(reals[-1][0])}")
    print(
        f"prices 2008-2025: {pct(cpi[2025] / cpi[2008] - 1)}; fully indexed the pension would be"
        f" {1000 * cpi[2025] / cpi[2008]:,.2f} in 2025"
    )
    print(f"nominal rise 2008-2025: {pct(pt[2025][2] / 1000 - 1)}; 2010-2017 unchanged at {pt[2010][0]:.2f}")
    freeze = cpi[2017] / cpi[2010] - 1
    print(f"prices while the amount was unchanged (2010 to 2017): {pct(freeze)}")
    lowmin = min((r[6], r[0]) for r in out if not math.isnan(r[6]))
    print(f"lowest bracket's percentages alone: lowest real {pct(lowmin[0])} in {lowmin[1]}, 2025 {pct(out[17][6])}")
    half = 0.5 / 14
    print(f"half a monthly pension paid in October 2022, as a share of a year's 14 payments: {pct(half, 2)}")
    print(f"2022 prices: {pct(cpi[2022] / cpi[2021] - 1)}; 2023: {pct(cpi[2023] / cpi[2022] - 1)}")
    coef = d.pt_coefficients
    factors = pc.pt_first_bracket(d.pt_updates, regular=True)
    diffs = [abs(coef[y] / coef[y + 1] - factors[y + 2]) for y in range(2007, 2024)]
    print(
        f"check: lowest bracket's regular rates against the official coefficients (Portaria 372-B/2024/1),"
        f" 2009-2025: largest gap {max(diffs):.5f}"
    )

    # ------------------------------------------------------------------------------------------------ 11
    head("11. Checks")
    base = pc.quantile(list(life_fixed.values()), 0.5)
    print(f"median lifetime real income kept, base case: {pct(base)}")
    for label, table in (
        ("EU 2024 table", d.qx[(pc.LIFE_GEO, 2024, "T")]),
        ("Portugal 2019 table", d.qx[("PT", 2019, "T")]),
    ):
        s_alt = pc.survival(table)
        vv = [pc.weighted_real(pc.real_path(d.cpi[c], s, pc.SPAN), s_alt) for c, s in w30]
        print(f"   {label}: {pct(pc.quantile(vv, 0.5))}")
    w25 = pc.windows(d.cpi, pc.PANEL, 25)
    vv = [pc.weighted_real(pc.real_path(d.cpi[c], s, 25), S[:25]) for c, s in w25]
    print(f"   payments to 89 only (more recent starts, {len(w25)}): {pct(pc.quantile(vv, 0.5))}")
    vv = [x for (c, s), x in life_fixed.items() if c != "ISL"]
    print(f"   without Iceland: median {pct(pc.quantile(vv, 0.5))}, p10 {pct(pc.quantile(vv, 0.1))}")
    vv = [x for (c, s), x in life_fixed.items() if s >= 1960]
    print(f"   starts from 1960: {pct(pc.quantile(vv, 0.5))}")
    eq = []
    for c in pc.PANEL:
        vv = [x for (cc, s), x in life_fixed.items() if cc == c]
        eq.append(pc.quantile(vv, 0.5))
    print(f"   median of country medians (each country counted once): {pct(pc.quantile(eq, 0.5))}")
    print(f"   Turkey's average inflation, 1980-1999: {pct(pc.average_inflation(d.cpi['TUR'], 1980, 19))} a year")
    tur = pc.windows(d.cpi, ["TUR"], pc.SPAN)
    vv = [pc.weighted_real(pc.real_path(d.cpi["TUR"], s, pc.SPAN), S) for c, s in tur]
    print(
        f"   Turkey, left out of the panel ({len(tur)} starts): median kept {pct(pc.quantile(vv, 0.5))}, highest"
        f" {pct(max(vv))}"
    )
    hp = d.hicp["PT"]
    print(
        f"   Portugal 2020-2025, OECD CPI {pct(cpi[2020] / cpi[2025] - 1)} against Eurostat HICP"
        f" {pct(hp[2020] / hp[2025] - 1)}"
    )


if __name__ == "__main__":
    main()
