"""Every number in the article.

python lump_numbers.py > output.txt

Also writes the tables behind the figures to results/.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import fetch_data
import lump_core as lc

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
LENGTHS = (3, 6, 12, 24, 36)
YIELDS = (0.02, 0.03, 0.04)
RATE_BANDS = ((-5.0, 2.0), (2.0, 5.0), (5.0, 8.0), (8.0, 12.0), (12.0, 100.0))
EPISODES = ("2000-03", "2007-10", "2008-09", "2009-03", "2020-02", "2022-01")
EPISODE_COUNTRIES = ("USA", "DEU", "GBR", "JPN", "PRT")


def head(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def dist(values, digits=1):
    q = [lc.quantile(values, p) for p in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)]
    return "min {} | p5 {} | p25 {} | median {} | p75 {} | p95 {} | max {}".format(*(pct(x, digits) for x in q))


def wins(outs):
    return lc.share([o.advantage for o in outs], lambda a: a > 0)


def med(outs):
    return lc.quantile([o.advantage for o in outs], 0.5)


def write(name, header, rows):
    with open(RES / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([f"{x:.6f}" if isinstance(x, float) else x for x in r])


def loss_aversion_breakeven(outs):
    """Weight on losses at which the two strategies have the same expected gain-minus-weighted-loss.

    The value of an outcome x (windfall = 1) is x - 1 if it is a gain and lambda * (x - 1) if it is a loss. The
    lump sum has the larger expected gain and the larger expected loss; the lambda that equalises them is returned.
    """
    gain = lambda v: lc.mean(max(x - 1.0, 0.0) for x in v)  # noqa: E731
    loss = lambda v: lc.mean(max(1.0 - x, 0.0) for x in v)  # noqa: E731
    lump = [o.lump for o in outs]
    spread = [o.spread for o in outs]
    return (gain(lump) - gain(spread)) / (loss(lump) - loss(spread))


def band_label(lo, hi):
    if lo < 0:
        return f"below {hi:.0f}%"
    if hi > 50:
        return f"{lo:.0f}% or more"
    return f"{lo:.0f}% to {hi:.0f}%"


def us_shiller(d, n=lc.SPREAD):
    rows = lc.shiller(fetch_data.SHILLER)
    price, total = lc.shiller_indices(rows)
    cash = d.cash["USA"]
    out = {}
    for name, idx in (("price", price), ("total", total)):
        res = []
        for s in sorted(idx):
            if lc.can_start(idx, cash, s, n):
                lump, spread = lc.replay(idx, cash, s, n)
                res.append((s, lump, spread, cash[s]))
        out[name] = res
    return out


def main():
    RES.mkdir(exist_ok=True)
    d = lc.load()
    base = lc.outcomes(d)

    # ------------------------------------------------------------------------------------------------ 1
    head("1. Data and conventions")
    print(f"countries: {len(lc.PANEL)} (OECD members by 1973, without Turkey): {', '.join(lc.PANEL)}")
    first = {c: min(d.prices[c]) for c in lc.PANEL}
    print("first month of share prices: " + ", ".join(f"{lc.NAMES[c]} {lc.label(m)}" for c, m in first.items()))
    print(f"last month of share prices: {lc.label(max(max(d.prices[c]) for c in lc.PANEL))}")
    print(f"start months with {lc.SPREAD} months of prices and cash rates after them: {len(base)}")
    print(f"first and last start month: {lc.label(min(o.start for o in base))}, {lc.label(max(o.start for o in base))}")
    src = [d.cash_source[o.country][o.start] for o in base]
    print(
        f"cash rate in the start month from the overnight rate (3-month rate missing):"
        f" {pct(lc.share(src, lambda x: x == 'overnight'))}"
    )
    neg = lc.share([o.cash_rate for o in base], lambda r: r < 0)
    print(f"start months with a negative cash rate: {pct(neg)}")
    per = defaultdict(list)
    for o in base:
        per[o.country].append(o)
    print("start months by country: " + ", ".join(f"{c} {len(v)}" for c, v in sorted(per.items())))

    # ------------------------------------------------------------------------------------------------ 2
    head("2. Twelve monthly instalments against a lump sum, share prices without dividends")
    adv = [o.advantage for o in base]
    print(f"lump sum ahead after 12 months in {pct(wins(base))} of {len(base)} start months")
    print(f"advantage of the lump sum (per cent of the spread value): {dist(adv, 2)}")
    print(f"mean advantage: {pct(lc.mean(adv), 2)}")
    up = [a for a in adv if a > 0]
    down = [a for a in adv if a <= 0]
    print(
        f"when the lump sum wins: median {pct(lc.quantile(up, 0.5), 2)}; when it loses: median"
        f" {pct(lc.quantile(down, 0.5), 2)}"
    )
    print(
        f"lump sum ahead by more than 10%: {pct(lc.share(adv, lambda a: a > 0.10))}; behind by more than 10%:"
        f" {pct(lc.share(adv, lambda a: a < -0.10))}"
    )
    worst = min(base, key=lambda o: o.advantage)
    best = max(base, key=lambda o: o.advantage)
    for tag, o in (("worst for the lump sum", worst), ("best for the lump sum", best)):
        print(
            f"{tag}: {lc.NAMES[o.country]} {lc.label(o.start)}: lump sum {o.lump:.3f}, spread {o.spread:.3f}"
            f" (advantage {pct(o.advantage)})"
        )
    rows = []
    for c, v in sorted(per.items(), key=lambda kv: wins(kv[1])):
        rows.append((c, len(v), wins(v), med(v)))
        print(
            f"   {lc.NAMES[c]:15s} {len(v):4d} starts: lump sum ahead {pct(wins(v))}, median advantage {pct(med(v), 2)}"
        )
    print(
        f"countries where the lump sum was ahead in more than half of start months:"
        f" {sum(1 for r in rows if r[2] > 0.5)} of {len(rows)}"
    )
    print(f"median of the 23 country shares: {pct(lc.quantile([r[2] for r in rows], 0.5))}")
    div3 = lc.outcomes(d, dividend_yield=0.03)
    per3 = defaultdict(list)
    for o in div3:
        per3[o.country].append(o)
    write(
        "by_country",
        ["country", "starts", "lump_wins", "median_advantage", "lump_wins_div3", "median_advantage_div3"],
        [(c, n, w, m, wins(per3[c]), med(per3[c])) for c, n, w, m in rows],
    )
    write(
        "advantage",
        ["country", "start", "lump", "spread", "advantage"],
        [(o.country, lc.label(o.start), o.lump, o.spread, o.advantage) for o in base],
    )
    stocks_beat_cash = []
    for o in base:
        cash = d.cash[o.country]
        grow = math.prod(1 + cash[o.start + k] / 1200 for k in range(lc.SPREAD) if o.start + k in cash)
        stocks_beat_cash.append(o.lump > grow)
    print(
        f"share of start months where the index (no dividends) beat cash over the 12 months:"
        f" {pct(lc.share(stocks_beat_cash, bool))}"
    )

    # ------------------------------------------------------------------------------------------------ 3
    head("3. What spreading buys: the value of the windfall after one year")
    lump = [o.lump for o in base]
    spread = [o.spread for o in base]
    half = [(o.lump + o.spread) / 2 for o in base]
    for q in (0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
        print(
            f"   p{round(100 * q):02d}: lump sum {lc.quantile(lump, q):.3f}, spread {lc.quantile(spread, q):.3f},"
            f" half and half {lc.quantile(half, q):.3f}"
        )
    for cut in (1.0, 0.9, 0.8, 0.7):
        print(
            f"   worth less than {cut:.1f} of the windfall: lump sum {pct(lc.share(lump, lambda v, c=cut: v < c))},"
            f" spread {pct(lc.share(spread, lambda v, c=cut: v < c))}, half and half"
            f" {pct(lc.share(half, lambda v, c=cut: v < c))}"
        )
    hh = [h / s - 1 for h, s in zip(half, spread, strict=True)]
    print(
        f"half now, half over 12 months: ahead of spreading in {pct(lc.share(hh, lambda a: a > 0))}, median"
        f" {pct(lc.quantile(hh, 0.5), 2)}"
    )
    write(
        "outcome_cdf",
        ["q", "lump", "spread", "half"],
        [
            (q / 200, lc.quantile(lump, q / 200), lc.quantile(spread, q / 200), lc.quantile(half, q / 200))
            for q in range(1, 200)
        ],
    )
    for name, outs in (("without dividends", base), ("with 3% dividends", lc.outcomes(d, dividend_yield=0.03))):
        lam = loss_aversion_breakeven(outs)
        print(
            f"loss aversion at which spreading becomes the better bet ({name}): losses weighted {lam:.2f} times gains"
        )
    regret_lump = [-o.advantage for o in base if o.advantage < 0]
    regret_spread = [o.advantage for o in base if o.advantage > 0]
    print(
        f"regret of the lump-sum investor (spreading would have done better), median"
        f" {pct(lc.quantile(regret_lump, 0.5), 2)},"
        f" p95 {pct(lc.quantile(regret_lump, 0.95), 2)}"
    )
    print(
        f"regret of the spreading investor (the lump sum would have done better), median"
        f" {pct(lc.quantile(regret_spread, 0.5), 2)},"
        f" p95 {pct(lc.quantile(regret_spread, 0.95), 2)}"
    )

    # ------------------------------------------------------------------------------------------------ 4
    head("4. Dividends: the OECD indices leave them out")
    for y in YIELDS:
        o = lc.outcomes(d, dividend_yield=y)
        print(
            f"with an assumed dividend yield of {pct(y, 0)} a year: lump sum ahead in {pct(wins(o))}, median"
            f" advantage {pct(med(o), 2)}"
        )
    us = us_shiller(d)
    for name, res in us.items():
        a = [lump / spread - 1 for _, lump, spread, _ in res]
        print(
            f"United States, Shiller's S&P composite, {name} {'return' if name == 'total' else 'only'},"
            f" starts {lc.label(res[0][0])} to {lc.label(res[-1][0])} ({len(res)}): lump sum ahead in"
            f" {pct(lc.share(a, lambda x: x > 0))}, median advantage {pct(lc.quantile(a, 0.5), 2)}"
        )
    for name, res in us.items():
        outs = [lc.Outcome("USA", st, lump, spread, r, False, False) for st, lump, spread, r in res]
        print(
            f"United States, Shiller {name}: loss aversion at which spreading becomes the better bet"
            f" {loss_aversion_breakeven(outs):.2f}"
        )
    oecd_us = [o for o in base if o.country == "USA" and o.start <= us["total"][-1][0]]
    print(
        f"United States, OECD index without dividends, same start months ({len(oecd_us)}): lump sum ahead in"
        f" {pct(wins(oecd_us))}"
    )
    rows = lc.shiller(fetch_data.SHILLER)
    dy = [rows[m][1] / rows[m][0] for m in rows if m >= lc.month("1954-07")]
    print(
        f"Shiller dividend yield, 1954-07 to 2023-06: median {pct(lc.quantile(dy, 0.5), 2)}, range"
        f" {pct(min(dy), 2)} to {pct(max(dy), 2)}"
    )
    write(
        "us_total",
        ["start", "lump", "spread", "advantage", "cash_rate"],
        [(lc.label(s), lump, spread, lump / spread - 1, r) for s, lump, spread, r in us["total"]],
    )
    print("United States with dividends, by cash rate in the start month:")
    for lo, hi in RATE_BANDS:
        a = [lump / spread - 1 for _, lump, spread, r in us["total"] if lo <= r < hi]
        if a:
            print(
                f"   {band_label(lo, hi):12s} {len(a):4d} starts: lump sum ahead in"
                f" {pct(lc.share(a, lambda x: x > 0))}, median {pct(lc.quantile(a, 0.5), 2)}"
            )

    # ------------------------------------------------------------------------------------------------ 5
    head("5. How long to spread: 3 to 36 monthly instalments")
    rows = []
    for n in LENGTHS:
        o = lc.outcomes(d, n)
        lp = [x.lump for x in o]
        sp = [x.spread for x in o]
        rows.append(
            (
                n,
                len(o),
                wins(o),
                med(o),
                lc.quantile(lp, 0.05),
                lc.quantile(sp, 0.05),
                lc.quantile(lp, 0.5),
                lc.quantile(sp, 0.5),
            )
        )
        print(
            f"   {n:2d} months ({len(o)} starts): lump sum ahead in {pct(wins(o))}, median advantage {pct(med(o), 2)};"
            f" 5th percentile of value at the end: lump sum {lc.quantile(lp, 0.05):.3f}, spread"
            f" {lc.quantile(sp, 0.05):.3f}"
        )
    write(
        "lengths",
        ["months", "starts", "lump_wins", "median_advantage", "p5_lump", "p5_spread", "p50_lump", "p50_spread"],
        rows,
    )
    for n in (12, 24, 36):
        print(f"   average share of the windfall in cash over the {n} months of spreading: {pct((n - 1) / (2 * n))}")

    # ------------------------------------------------------------------------------------------------ 6
    head("6. Conditions at the start")
    groups = (
        ("at an all-time high", [o for o in base if o.at_high]),
        ("20% or more below the high", [o for o in base if o.after_fall]),
        ("anywhere else", [o for o in base if not o.at_high and not o.after_fall]),
    )
    cond_rows = []
    for name, g in groups:
        g3 = [o for o in div3 if (o.country, o.start) in {(x.country, x.start) for x in g}]
        print(
            f"   {name:28s} {len(g):5d} starts: lump sum ahead in {pct(wins(g))} (with 3% dividends"
            f" {pct(wins(g3))}), median advantage {pct(med(g), 2)}"
        )
        cond_rows.append(("condition", name, len(g), wins(g), wins(g3), med(g)))
    print(
        f"   at an all-time high: index lower a year later in"
        f" {pct(lc.share([o.lump for o in groups[0][1]], lambda v: v < 1))} of starts"
    )
    print("by the cash rate in the start month:")
    for lo, hi in RATE_BANDS:
        g = [o for o in base if lo <= o.cash_rate < hi]
        g3 = [o for o in div3 if lo <= o.cash_rate < hi]
        print(
            f"   {band_label(lo, hi):12s} {len(g):5d} starts: lump sum ahead in {pct(wins(g))} (with 3% dividends"
            f" {pct(wins(g3))}), median advantage {pct(med(g), 2)}"
        )
        cond_rows.append(("cash", band_label(lo, hi), len(g), wins(g), wins(g3), med(g)))
    write("conditions", ["kind", "group", "starts", "lump_wins", "lump_wins_div3", "median_advantage"], cond_rows)

    # ------------------------------------------------------------------------------------------------ 7
    head("7. By decade of the start month")
    for dec in range(1950, 2030, 10):
        g = [o for o in base if dec <= o.start // 12 < dec + 10]
        g3 = [o for o in div3 if dec <= o.start // 12 < dec + 10]
        rates = [o.cash_rate for o in g]
        print(
            f"   {dec}s: {len(g):5d} starts, lump sum ahead in {pct(wins(g))} (with 3% dividends {pct(wins(g3))}),"
            f" median advantage {pct(med(g), 2)}, median cash rate {lc.quantile(rates, 0.5):.1f}%"
        )

    # ------------------------------------------------------------------------------------------------ 8
    head("8. Episodes: a windfall of 100,000 in a given month, one year after the last instalment")
    index = {(o.country, lc.label(o.start)): o for o in base}
    for e in EPISODES:
        parts = []
        for c in EPISODE_COUNTRIES:
            o = index.get((c, e))
            if o:
                parts.append(f"{c} lump {100_000 * o.lump:,.0f} / spread {100_000 * o.spread:,.0f}")
        print(f"   {e}: " + "; ".join(parts))

    # ------------------------------------------------------------------------------------------------ 9
    head("9. Checks")
    eq = [wins(v) for v in per.values()]
    print(f"each country counted once (median of country shares): {pct(lc.quantile(eq, 0.5))}; mean {pct(lc.mean(eq))}")
    g = [o for o in base if d.cash_source[o.country][o.start] == "3m"]
    print(f"only start months with a 3-month rate: {pct(wins(g))} of {len(g)}")
    g = [o for o in base if o.start >= lc.month("1970-01")]
    print(
        f"starts from 1970: {pct(wins(g))}; from 1990: {pct(wins([o for o in base if o.start >= lc.month('1990-01')]))}"
    )
    g = [o for o in base if o.country not in ("ISL", "GRC")]
    print(f"without Iceland and Greece: {pct(wins(g))}")
    starts = sorted({o.start for o in base})
    non_overlap = [o for o in base if (o.start - starts[0]) % 13 == 0]
    print(f"non-overlapping windows only (one start every 13 months): {pct(wins(non_overlap))} of {len(non_overlap)}")
    zero_cash = []
    for c in lc.PANEL:
        idx = lc.total_return(d.prices[c])
        zero = dict.fromkeys(d.cash[c], 0.0)
        for s in sorted(d.prices[c]):
            if lc.can_start(idx, zero, s, lc.SPREAD):
                lump, spread = lc.replay(idx, zero, s, lc.SPREAD)
                zero_cash.append(lump / spread - 1)
    print(
        f"cash earning nothing (same start months as the base case): lump sum ahead in"
        f" {pct(lc.share(zero_cash, lambda a: a > 0))}"
    )


if __name__ == "__main__":
    main()
