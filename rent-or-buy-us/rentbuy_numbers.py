"""Every number in the rent-or-buy-us article.

    python fetch.py                          # inputs into data/, checked against the registry
    python rentbuy_numbers.py > output.txt   # the numbers; also writes results/*.csv for figures.py

The question: does buying a home beat renting mainly because the mortgage forces the owner to save,
so that a renter who invests the difference does as well or better? Each household buys or rents in
the middle of a year from 1987 (later where a series starts later) and leaves after 5, 10 or 20
years, in 16 metropolitan areas and the United States as a whole. See rentbuy_core.py for the rules.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import rentbuy_core as C

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("RENTBUY_DATA", HERE / "data"))
RES = HERE / "results"
HORIZONS = (5, 10, 20)
CATS = [
    "renting wins either way",
    "saving decides",
    "owning wins either way",
    "saving favours the owner",
]
BASE = C.Assumptions()


def run(a: C.Assumptions, market: C.Market, horizons=HORIZONS) -> pd.DataFrame:
    frames = []
    for name in C.AREAS:
        area = C.load_area(name, DATA, a)
        for h in horizons:
            frames.append(C.table(C.cohorts(area, market, h, a)))
    return pd.concat(frames, ignore_index=True)


def shares(d: pd.DataFrame) -> str:
    c = d.category.value_counts(normalize=True)
    return ", ".join(f"{k} {100 * c.get(k, 0):.0f}%" for k in CATS)


def pct(x) -> str:
    return f"{100 * x:.0f}%"


def section_data(market: C.Market) -> None:
    print("== Data")
    print("areas: 16 metropolitan areas with both a Case-Shiller index and a BLS rent index, plus the United States")
    print(
        "anchor year 2023 (ACS 1-year): median value, median contract rent, rooms owner/renter, comparable rent at e=0.5, "
        "price-to-rent, property tax rate; first and last purchase year for a 10-year hold"
    )
    for name in C.AREAS:
        ar = C.load_area(name, DATA, BASE)
        cr0 = C.load_area(name, DATA, C.with_(BASE, rent_elasticity=0.0)).rent.loc[2023] / 12
        first = [o.year for o in C.cohorts(ar, market, 10, BASE)]
        print(
            f"  {name}: value {ar.price.loc[2023]:,.0f}; contract rent {cr0:,.0f}/month; rooms ratio {ar.rooms_ratio:.2f}; "
            f"comparable rent {ar.rent.loc[2023] / 12:,.0f}/month; price-to-rent {ar.price.loc[2023] / ar.rent.loc[2023]:.1f}; "
            f"tax {100 * ar.tax_rate:.2f}%{' (Prop 13 rule used)' if name in C.CA else ''}; "
            f"10-year cohorts {first[0]}-{first[-1]} ({len(first)})"
        )
    r = market.rate
    print(
        f"30-year fixed rate (yearly average of weekly PMMS): 1987 {100 * r[1987]:.2f}%, 1995 {100 * r[1995]:.2f}%, 2006 {100 * r[2006]:.2f}%, "
        f"2012 {100 * r[2012]:.2f}%, 2021 {100 * r[2021]:.2f}%, 2023 {100 * r[2023]:.2f}%"
    )
    x = market.returns.loc[1987:2025]
    gm = {k: 100 * (np.prod(1 + x[k]) ** (1 / len(x)) - 1) for k in ("stocks", "bonds", "bills")}
    print(
        f"annual total returns 1987-2025, geometric mean: S&P 500 {gm['stocks']:.2f}%, "
        f"10-year Treasury {gm['bonds']:.2f}%, 3-month bills {gm['bills']:.2f}%"
    )
    print()


def section_main(d: pd.DataFrame) -> None:
    print("== 1. Who ends richer: central case (comparable rent e=0.5, S&P 500, refinancing, 6% selling cost)")
    for h in HORIZONS:
        for label, sub in [
            ("16 metros", d[(d.horizon == h) & (d.area != "United States")]),
            ("United States", d[(d.horizon == h) & (d.area == "United States")]),
        ]:
            print(
                f"  {h:2d} years, {label}: n={len(sub)}; {shares(sub)}; owner beats the disciplined renter "
                f"{pct((sub.owner_disc > sub.renter_disc).mean())}, "
                f"owner beats the renter who spends the difference {pct((sub.owner_lazy > sub.renter_lazy).mean())}"
            )
    print()


def section_bracket(market: C.Market) -> None:
    print("== 2. The comparable rent: median renter's contract rent scaled by rooms ** e (16 metros)")
    for e in (0.0, 0.5, 1.0):
        d = run(C.with_(BASE, rent_elasticity=e), market, (10, 20))
        d = d[d.area != "United States"]
        for h in (10, 20):
            sub = d[d.horizon == h]
            print(f"  e={e:.1f}, {h} years: {shares(sub)}; owner beats the disciplined renter {pct((sub.owner_disc > sub.renter_disc).mean())}")
    print()


def section_timing(d: pd.DataFrame) -> None:
    print("== 3. Purchase year against city: share of the variance of log(owner / disciplined renter) explained (16 metros)")
    for h in HORIZONS:
        s = d[(d.horizon == h) & (d.area != "United States") & (d.owner_disc > 0) & (d.renter_disc > 0)].copy()
        s["y"] = np.log(s.owner_disc / s.renter_disc)
        tot = ((s.y - s.y.mean()) ** 2).sum()
        by_year = s.groupby("year").y.transform("mean")
        by_area = s.groupby("area").y.transform("mean")
        both = by_year + by_area - s.y.mean()
        print(
            f"  {h:2d} years: purchase year {1 - ((s.y - by_year) ** 2).sum() / tot:.2f}, city {1 - ((s.y - by_area) ** 2).sum() / tot:.2f}, "
            f"both {1 - ((s.y - both) ** 2).sum() / tot:.2f}; correlation with log price-to-rent at purchase "
            f"{np.corrcoef(np.log(s.price_to_rent), s.y)[0, 1]:+.2f}; "
            f"saving decides in {pct((s.category == 'saving decides').mean())}"
        )
    s = d[(d.horizon == 10) & (d.area != "United States")]
    w = s.groupby("year").apply(lambda g: (g.owner_disc > g.renter_disc).mean(), include_groups=False)
    print(
        "  10 years, share of the 16 metros where owning beat the disciplined renter, by purchase year: "
        + " ".join(f"{y} {100 * v:.0f}" for y, v in w.items())
    )
    print()


def section_forced(d: pd.DataFrame, market: C.Market) -> None:
    print("== 4. What the owner's wealth is made of (central case)")
    cpi = market.cpi
    for h in (10, 20):
        s = d[(d.horizon == h) & (d.area != "United States")]
        eq = s.owner_lazy
        print(
            f"  {h} years, 16 metros, medians of shares of the owner's home equity at sale: down payment {pct((s.down_payment / eq).median())}, "
            f"principal repaid {pct((s.principal_paid / eq).median())}, price gain {pct((s.price_gain / eq).median())}, "
            f"selling cost {pct((-s.selling_cost / eq).median())}; median interest paid / principal repaid "
            f"{(s.interest_paid / s.principal_paid).median():.2f}"
        )
        sd = s[s.category == "saving decides"]
        print(
            f"     saving decides in {len(sd)} of {len(s)}: median share of the difference the renter must invest to tie "
            f"{pct(sd.breakeven_s.median())} "
            f"(quartiles {pct(sd.breakeven_s.quantile(0.25))}-{pct(sd.breakeven_s.quantile(0.75))})"
        )
    us = d[(d.area == "United States") & (d.horizon == 10)].set_index("year")
    for y in (1990, 1995, 2000, 2006, 2012):
        if y not in us.index:
            continue
        r = us.loc[y]
        k = cpi[2025] / cpi[y + 10]
        print(
            f"  United States, bought {y}, sold {y + 10} (2025 dollars): owner {k * r.owner_lazy:,.0f} "
            f"(+{k * r.owner_side:,.0f} if he invests his savings); "
            f"renter {k * r.renter_lazy:,.0f} (+{k * r.renter_side:,.0f} if he invests the difference); category: {r.category}; "
            f"refinances {int(r.refinances)}; price-to-rent at purchase {r.price_to_rent:.1f}"
        )
    print()


def section_sensitivity(market: C.Market) -> None:
    print("== 5. Sensitivity (16 metros): category shares at 10 and 20 years")
    variants = [
        ("60% stocks, 40% Treasuries", dict(portfolio="6040")),
        ("3-month Treasury bills", dict(portfolio="bills")),
        ("no refinancing", dict(refinance=False)),
        ("selling cost 3%", dict(sell_cost=0.03)),
        ("maintenance 1.5%", dict(maintenance=0.015)),
        ("insurance 0.6%", dict(insurance=0.006)),
        ("tax drag 0.75 points on the invested account", dict(tax_drag=0.0075)),
        ("anchor year 2022", dict(anchor_year=2022)),
    ]
    for label, kw in variants:
        d = run(C.with_(BASE, **kw), market, (10, 20))
        d = d[d.area != "United States"]
        parts = []
        for h in (10, 20):
            sub = d[d.horizon == h]
            parts.append(f"{h}y: {shares(sub)}; owner beats disciplined renter {pct((sub.owner_disc > sub.renter_disc).mean())}")
        print(f"  {label}: " + " | ".join(parts))
    print()


def section_areas(d: pd.DataFrame) -> None:
    print("== 6. By area, 10 years: share of purchase years where owning beat the disciplined renter / the renter who spends the difference")
    s = d[d.horizon == 10]
    for name, g in s.groupby("area", sort=False):
        print(
            f"  {name}: {len(g)} years; {pct((g.owner_disc > g.renter_disc).mean())} / {pct((g.owner_lazy > g.renter_lazy).mean())}; "
            f"saving decides {pct((g.category == 'saving decides').mean())}; median price-to-rent at purchase {g.price_to_rent.median():.1f}"
        )
    print()


def main() -> None:
    market = C.load_market(DATA)
    d = run(BASE, market)
    RES.mkdir(exist_ok=True)
    d.to_csv(RES / "cohorts_central.csv", index=False, float_format="%.2f")
    section_data(market)
    section_main(d)
    section_bracket(market)
    section_timing(d)
    section_forced(d, market)
    section_sensitivity(market)
    section_areas(d)


if __name__ == "__main__":
    main()
