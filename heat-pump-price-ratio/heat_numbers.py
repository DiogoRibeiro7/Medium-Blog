"""Print every number the article uses, and write the tables behind the figures to results/.

python heat_numbers.py > output.txt
"""

import csv
import math
from pathlib import Path

import heat_core as hc

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SPFS = (2.5, 2.8, 3.0, 3.4, 4.0)
NEAR = 0.3  # distance from the break-even ratio counted as close


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def write(name, header, rows):
    RES.mkdir(exist_ok=True)
    with open(RES / name, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def series(pts):
    out = {}
    for p in pts:
        out.setdefault(p.country, {})[p.half] = p
    return out


def latest_year(by_country):
    """The last two half-years with both prices, as one year: prices are the mean of the two halves."""
    out = {}
    for c, s in by_country.items():
        last = max(s)
        if last - 1 not in s:
            continue
        a, b = s[last - 1], s[last]
        out[c] = (last, hc.Point(c, last, (a.p_e + b.p_e) / 2, (a.p_g + b.p_g) / 2))
    return out


def year_label(last):
    return f"{hc.label(last - 1)}+{hc.label(last)}"


def corr(xs, ys):
    mx, my = hc.mean(xs), hc.mean(ys)
    sxy = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    syy = math.fsum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy)


def main():
    e = hc.read_prices("electricity")
    g = hc.read_prices("gas")
    hdd = hc.read_hdd()
    pts = hc.points(e, g)
    by_country = series(pts)
    latest = latest_year(by_country)

    print("1. Data")
    print(f"   {len(by_country)} countries, {len(pts)} country half-years, {hc.FIRST} to {hc.LAST}")
    print(f"   electricity band {hc.ELECTRICITY_BAND} (DD), gas band {hc.GAS_BAND} (D2), all taxes and levies, euros")
    for c in hc.PANEL:
        s = by_country[c]
        print(f"   {c} {hc.NAMES[c]:<15} {hc.label(min(s))} to {hc.label(max(s))}, {len(s)} half-years")
    flagged = sum(
        1
        for p in pts
        if e[(p.country, hc.ELECTRICITY_BAND, "I_TAX")][p.half][1] or g[(p.country, hc.GAS_BAND, "I_TAX")][p.half][1]
    )
    print(f"   half-years with a flagged price (estimated, provisional, break or definition): {flagged}")

    print()
    print("2. The ratio of the electricity price to the gas price, per kWh")
    r = [p.ratio for p in pts]
    lo = min(pts, key=lambda p: p.ratio)
    hi = max(pts, key=lambda p: p.ratio)
    print(
        f"   all country half-years: median {hc.quantile(r, 0.5):.2f}, p10 {hc.quantile(r, 0.1):.2f}, "
        f"p90 {hc.quantile(r, 0.9):.2f}"
    )
    print(
        f"   lowest {lo.ratio:.2f} ({lo.country} {hc.label(lo.half)}), "
        f"highest {hi.ratio:.2f} ({hi.country} {hc.label(hi.half)})"
    )
    print(f"   below 1 (electricity cheaper than gas per kWh): {sum(x < 1 for x in r)} country half-years")
    print("   by country: lowest, median, highest, latest year")
    rows = []
    for c in hc.PANEL:
        s = by_country[c]
        v = [p.ratio for p in s.values()]
        lmin = min(s.values(), key=lambda p: p.ratio)
        lmax = max(s.values(), key=lambda p: p.ratio)
        last, lp = latest[c]
        print(
            f"   {c} {min(v):.2f} ({hc.label(lmin.half)})  {hc.quantile(v, 0.5):.2f}  {max(v):.2f} "
            f"({hc.label(lmax.half)})  {lp.ratio:.2f} ({year_label(last)})"
        )
        rows.append(
            (
                c,
                hc.NAMES[c],
                f"{min(v):.4f}",
                f"{hc.quantile(v, 0.5):.4f}",
                f"{max(v):.4f}",
                f"{lp.ratio:.4f}",
                hc.label(last),
            )
        )
    write(
        "ratio_by_country.csv",
        ["country", "name", "lowest", "median", "highest", "latest", "latest_half"],
        rows,
    )
    current = {c: lp for c, (last, lp) in latest.items() if hc.label(last) == hc.LAST}
    cr = sorted(current.values(), key=lambda p: p.ratio)
    print(
        f"   latest year ({year_label(hc.half(hc.LAST))}), {len(cr)} countries: "
        f"median {hc.quantile([p.ratio for p in cr], 0.5):.2f}; " + ", ".join(f"{p.country} {p.ratio:.2f}" for p in cr)
    )
    eu = hc.points(e, g, countries=("EU27_2020",))
    eus = {p.half: p for p in eu}
    print(
        "   EU (27) average prices: "
        + ", ".join(
            f"{hc.label(h)} {eus[h].ratio:.2f}" for h in sorted(eus) if hc.label(h).endswith("S1") and h % 4 == 0
        )
    )
    eumin = min(eu, key=lambda p: p.ratio)
    eumax = max(eu, key=lambda p: p.ratio)
    print(
        f"   EU (27): first {eus[min(eus)].ratio:.2f} ({hc.label(min(eus))}), "
        f"highest {eumax.ratio:.2f} ({hc.label(eumax.half)}), "
        f"lowest {eumin.ratio:.2f} ({hc.label(eumin.half)}), last {eus[max(eus)].ratio:.2f} ({hc.label(max(eus))})"
    )

    print()
    print(f"3. How good a heat pump has to be (boiler efficiency {hc.ETA:.2f}, gross calorific value)")
    be = [hc.breakeven_spf(x) for x in r]
    print(
        f"   break-even seasonal performance factor, all country half-years: median {hc.quantile(be, 0.5):.2f}, "
        f"p10 {hc.quantile(be, 0.1):.2f}, p90 {hc.quantile(be, 0.9):.2f}"
    )
    print("   share of country half-years in which the heat pump was cheaper to run")
    rows = []
    for eta in (0.90, 0.85, 0.80):
        line = []
        for spf in SPFS:
            a = hc.share(p.cheaper(spf, eta) for p in pts)
            line.append(f"SPF {spf:.1f}: {pct(a)}")
        print(f"   boiler {eta:.2f}: " + ", ".join(line))
    for i in range(201, 501):
        spf = i / 100
        rows.append(
            (
                f"{spf:.2f}",
                f"{hc.share(p.cheaper(spf) for p in pts):.4f}",
                f"{hc.share(p.cheaper(spf) for p in cr):.4f}",
                f"{hc.share(p.cheaper(spf, 0.85) for p in pts):.4f}",
            )
        )
    write(
        "share_by_spf.csv",
        ["spf", "all_half_years", "latest_year", "all_half_years_boiler_085"],
        rows,
    )
    print(
        f"   latest year, {len(cr)} countries: cheaper at SPF 2.8 in {sum(p.cheaper(2.8) for p in cr)}, "
        f"at 3.4 in {sum(p.cheaper(3.4) for p in cr)}, at 4.0 in {sum(p.cheaper(4.0) for p in cr)}"
    )
    print("   latest year, break-even SPF: " + ", ".join(f"{p.country} {hc.breakeven_spf(p.ratio):.2f}" for p in cr))
    print("   countries in which the heat pump was cheaper, by half-year (SPF 2.8 / 3.4, of countries with data)")
    rows = []
    halves = sorted({p.half for p in pts})
    for h in halves:
        here = [p for p in pts if p.half == h]
        n28 = sum(p.cheaper(2.8) for p in here)
        n34 = sum(p.cheaper(3.4) for p in here)
        rows.append(
            (
                hc.label(h),
                len(here),
                n28,
                n34,
                f"{eus[h].ratio:.4f}" if h in eus else "",
            )
        )
        print(f"   {hc.label(h)}: {n28} / {n34} of {len(here)}")
    write(
        "by_half.csv",
        ["half", "countries", "cheaper_spf_28", "cheaper_spf_34", "eu_ratio"],
        rows,
    )
    before = [p for p in pts if p.half <= hc.half("2021-S2")]
    after = [p for p in pts if p.half >= hc.half("2022-S1")]
    print(
        f"   share cheaper at SPF 2.8: to 2021-S2 {pct(hc.share(p.cheaper(2.8) for p in before))}, "
        f"from 2022-S1 {pct(hc.share(p.cheaper(2.8) for p in after))}; at SPF 3.4: "
        f"{pct(hc.share(p.cheaper(3.4) for p in before))} and {pct(hc.share(p.cheaper(3.4) for p in after))}"
    )
    write(
        "panel.csv",
        ["country", "half", "p_e", "p_g", "ratio", "breakeven_spf"],
        [
            (
                p.country,
                hc.label(p.half),
                f"{p.p_e:.4f}",
                f"{p.p_g:.4f}",
                f"{p.ratio:.4f}",
                f"{hc.breakeven_spf(p.ratio):.4f}",
            )
            for p in pts
        ],
    )

    print()
    print(f"4. What it saves: a home that needs {hc.HEAT:,.0f} kWh of heat a year, prices of the latest year")
    eu_hdd = hc.mean(hdd["EU27_2020"].values())
    rows = []
    for c in sorted(latest, key=lambda c: latest[c][1].ratio):
        last, p = latest[c]
        boiler = hc.HEAT * p.p_g / hc.ETA
        s28, s34 = p.saving(2.8), p.saving(3.4)
        h_c = hc.mean(hdd[c].values())
        heat_c = hc.HEAT * h_c / eu_hdd
        s34c = p.saving(3.4, heat=heat_c)
        print(
            f"   {c} ({year_label(last)}): electricity {p.p_e:.3f}, gas {p.p_g:.3f}, ratio {p.ratio:.2f}; boiler costs "
            f"{boiler:,.0f}; saving at SPF 2.8 {s28:,.0f}, at 3.4 {s34:,.0f}; degree days {h_c:,.0f}, "
            f"heat {heat_c:,.0f} kWh, saving at 3.4 {s34c:,.0f}"
        )
        rows.append(
            (c, hc.NAMES[c], hc.label(last), f"{p.p_e:.4f}", f"{p.p_g:.4f}", f"{p.ratio:.4f}", f"{boiler:.1f}",
             f"{s28:.1f}", f"{s34:.1f}", f"{h_c:.1f}", f"{heat_c:.1f}", f"{s34c:.1f}")
        )  # fmt: skip
    write(
        "latest.csv",
        ["country", "name", "last_half", "p_e", "p_g", "ratio", "boiler_cost", "saving_28", "saving_34", "hdd",
         "heat_scaled", "saving_34_scaled"],
        rows,
    )  # fmt: skip
    print(f"   EU (27) degree days, mean 2015-2024: {eu_hdd:,.0f}")
    xs = [latest[c][1] for c in current]
    print(
        f"   latest year, {len(xs)} countries: correlation of the saving at SPF 3.4 with the ratio "
        f"{corr([p.ratio for p in xs], [p.saving(3.4) for p in xs]):.2f}, with the gas price "
        f"{corr([p.p_g for p in xs], [p.saving(3.4) for p in xs]):.2f}, with the electricity price "
        f"{corr([p.p_e for p in xs], [p.saving(3.4) for p in xs]):.2f}"
    )
    gp = [p.p_g for p in pts]
    print(
        f"   gas price across country half-years: p10 {hc.quantile(gp, 0.1):.3f}, median {hc.quantile(gp, 0.5):.3f}, "
        f"p90 {hc.quantile(gp, 0.9):.3f} euros per kWh"
    )
    sav = [p.saving(3.4) for p in pts]
    print(
        f"   saving at SPF 3.4 across all country half-years: p10 {hc.quantile(sav, 0.1):,.0f}, "
        f"median {hc.quantile(sav, 0.5):,.0f}, p90 {hc.quantile(sav, 0.9):,.0f}"
    )

    print()
    print("5. Taxes: the ratio with all taxes, without VAT, and without any taxes or levies (latest year)")
    rows = []
    diffs = []
    for c in sorted(latest, key=lambda c: latest[c][1].ratio):
        last = latest[c][0]
        vals = []
        for tax in ("I_TAX", "X_VAT", "X_TAX"):
            s = {p.half: p for p in hc.points(e, g, tax=tax, countries=(c,))}
            a, b = s[last - 1], s[last]
            vals.append((a.p_e + b.p_e) / (a.p_g + b.p_g))
        diffs.append((c, vals[0] - vals[2]))
        print(f"   {c}: all taxes {vals[0]:.2f}, without VAT {vals[1]:.2f}, without taxes {vals[2]:.2f}")
        rows.append(
            (
                c,
                hc.NAMES[c],
                hc.label(last),
                f"{vals[0]:.4f}",
                f"{vals[1]:.4f}",
                f"{vals[2]:.4f}",
            )
        )
    write(
        "taxes.csv",
        [
            "country",
            "name",
            "last_half",
            "ratio_all_taxes",
            "ratio_without_vat",
            "ratio_without_taxes",
        ],
        rows,
    )
    up = [c for c, d in diffs if d > 0]
    print(
        f"   taxes raise the ratio in {len(up)} of {len(diffs)} countries; median change "
        f"{hc.quantile([d for _, d in diffs], 0.5):+.2f}; largest rise "
        + ", ".join(f"{c} {d:+.2f}" for c, d in sorted(diffs, key=lambda x: -x[1])[:4])
        + "; largest fall "
        + ", ".join(f"{c} {d:+.2f}" for c, d in sorted(diffs, key=lambda x: x[1])[:3])
    )
    for c in ("DE", "DK", "NL", "SE"):
        s_all = {p.half: p for p in hc.points(e, g, tax="I_TAX", countries=(c,))}
        s_none = {p.half: p for p in hc.points(e, g, tax="X_TAX", countries=(c,))}
        line = ", ".join(
            f"{x} {s_all[hc.half(x)].ratio:.2f}/{s_none[hc.half(x)].ratio:.2f}"
            for x in ("2008-S1", "2013-S1", "2018-S1", "2021-S2", "2023-S1", "2026-S1")
            if hc.half(x) in s_all and hc.half(x) in s_none
        )
        print(f"   {c} with/without taxes: {line}")

    print()
    print("6. Deciding on today's ratio: does it hold for the next five and ten years?")
    rows = []
    for years in (5, 10):
        n_h = 2 * years
        for spf in (2.8, 3.4):
            agree = total = yes_no = no_yes = 0
            changes = []
            near = {True: [0, 0], False: [0, 0]}
            for c, s in by_country.items():
                for start in sorted(s):
                    out = hc.lifetime(s, start, n_h, spf)
                    if out is None:
                        continue
                    decide = s[start].cheaper(spf)
                    realised = out > 0
                    total += 1
                    agree += decide == realised
                    yes_no += decide and not realised
                    no_yes += realised and not decide
                    close = abs(s[start].ratio - spf / hc.ETA) < NEAR
                    near[close][0] += decide == realised
                    near[close][1] += 1
                    if spf == 2.8:
                        changes.append(abs(hc.mean(s[h].ratio for h in range(start, start + n_h)) - s[start].ratio))
                    rows.append((years, spf, c, hc.label(start), int(decide), f"{out:.1f}"))
            line = (
                f"   {years} years, SPF {spf:.1f}: {total} starts; the ratio at the start gave the right answer in "
                f"{pct(agree / total)}; said cheaper but was not {yes_no}, said dearer but was not {no_yes}"
            )
            if changes:
                gap = hc.quantile(changes, 0.5)
                line += f"; median gap between the starting ratio and the mean ratio over the period {gap:.2f}"
            print(line)
            print(
                f"      starting ratio within {NEAR} of the break-even ratio: right in "
                f"{pct(near[True][0] / near[True][1])} of {near[True][1]}; further away: right in "
                f"{pct(near[False][0] / near[False][1])} of {near[False][1]}"
            )
    write(
        "persistence.csv",
        ["years", "spf", "country", "start", "cheaper_at_start", "saving_over_period"],
        rows,
    )

    print()
    print("7. Sensitivity")
    small = hc.points(e, g, e_band=hc.ELECTRICITY_SMALL)
    print(
        "   electricity band DC (2,500-4,999 kWh) instead of DD: "
        f"median ratio {hc.quantile([p.ratio for p in small], 0.5):.2f}; "
        f"cheaper at SPF 2.8 {pct(hc.share(p.cheaper(2.8) for p in small))}, "
        f"at 3.4 {pct(hc.share(p.cheaper(3.4) for p in small))}"
    )
    clean = [
        p
        for p in pts
        if not e[(p.country, hc.ELECTRICITY_BAND, "I_TAX")][p.half][1]
        and not g[(p.country, hc.GAS_BAND, "I_TAX")][p.half][1]
    ]
    print(
        f"   without flagged prices ({len(clean)} half-years): "
        f"cheaper at SPF 2.8 {pct(hc.share(p.cheaper(2.8) for p in clean))}, "
        f"at 3.4 {pct(hc.share(p.cheaper(3.4) for p in clean))}"
    )
    no_se = [p for p in pts if p.country != "SE"]
    print(
        f"   without Sweden: cheaper at SPF 2.8 {pct(hc.share(p.cheaper(2.8) for p in no_se))}, "
        f"at 3.4 {pct(hc.share(p.cheaper(3.4) for p in no_se))}"
    )
    common = [p for p in pts if p.country not in ("EL", "PL", "UK")]
    print(
        f"   only the 22 countries with every half-year from 2008-S1: cheaper at SPF 2.8 "
        f"{pct(hc.share(p.cheaper(2.8) for p in common if p.half >= hc.half('2008-S1')))}"
    )
    eq = {}
    for p in pts:
        eq.setdefault(p.country, []).append(p.cheaper(2.8))
    print(
        f"   each country counted once: median of the country shares cheaper at SPF 2.8 "
        f"{pct(hc.quantile([hc.share(v) for v in eq.values()], 0.5))}"
    )


if __name__ == "__main__":
    main()
