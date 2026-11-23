"""Every number in the crypto-survivorship article.

    python build_snapshots.py <folder with the saved CoinMarketCap pages>   # -> data/snapshots/*.csv
    python survivorship_numbers.py > output.txt

The question: how much do rankings that only show today's surviving coins flatter crypto returns? Each
January from 2014, the top 10, 20, 50 and 100 coins by market capitalisation are bought and held for 1, 3
and 5 years, dead coins included, and compared with portfolios of the coins that are in the top ranks today.
See crypto_core.py for the rules.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import crypto_core as C

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("CRYPTO_DATA", HERE / "data" / "snapshots"))
RES = HERE / "results"
SIZES = (10, 20, 50, 100)
HOLDS = (1, 3, 5)
SCHEMES = ("cap", "equal")


def pct(x: float, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{100 * x:.{digits}f}%"


def num(x: float, digits: int = 2) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{x:.{digits}f}"


def load(folder: Path) -> list[C.Snapshot]:
    snaps = []
    for path in sorted(folder.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            coins = [
                C.Coin(
                    key=r["key"],
                    name=r["name"],
                    symbol=r["symbol"],
                    rank=int(r["rank"]),
                    market_cap=float(r["market_cap"]),
                    price=float(r["price"]),
                    supply=float(r["supply"]),
                )
                for r in csv.DictReader(f)
            ]
        snaps.append(C.Snapshot(path.stem, coins))
    return snaps


def geometric_mean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(math.fsum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main() -> None:
    RES.mkdir(exist_ok=True)
    snaps = load(DATA)
    latest = snaps[-1]
    yearly = [s for s in snaps if s.date[5:7] == "01"]
    print("SNAPSHOTS")
    for s in snaps:
        print(
            f"  {s.date}: {len(s.coins)} coins; last listed market cap {s.cutoff_cap:,.0f}; top coin {s.coins[0].name}"
        )
    btc = next(c.key for c in yearly[0].coins if c.symbol == "BTC")

    rows = []
    print("\nPOINT IN TIME AGAINST TODAY'S SURVIVORS (multiples; point in time as high / low)")
    for n in SIZES:
        for scheme in SCHEMES:
            for h in HOLDS:
                pits, survs, btcs = [], [], []
                print(f"\n  top {n}, {scheme}-weighted, held {h} year(s)")
                for i, start in enumerate(yearly):
                    if i + h >= len(yearly):
                        break
                    end = yearly[i + h]
                    years = C.years_between(start.date, end.date)
                    p = C.point_in_time(start, end, n, scheme)
                    s = C.survivors(start, end, latest, n, scheme)
                    b = C.single(start, end, btc)
                    pits.append(p)
                    survs.append(s)
                    btcs.append(b)
                    rows.append(
                        {
                            "n": n,
                            "scheme": scheme,
                            "hold": h,
                            "start": start.date,
                            "end": end.date,
                            "pit_high": p.high,
                            "pit_low": p.low,
                            "dropped": p.dropped,
                            "dropped_weight": p.dropped_weight,
                            "survivor": s.high,
                            "survivor_members": s.members,
                            "btc": b,
                        }
                    )
                    print(
                        f"    {start.date[:4]}: point in time {num(p.high)} / {num(p.low)}"
                        f" ({p.dropped} of {p.members} dropped, {pct(p.dropped_weight)} of the weight);"
                        f" survivors {num(s.high)} ({s.members} coins); bitcoin {num(b)};"
                        f" gap at least {pct(C.gap_per_year(s.high, p.high, years))} a year"
                    )
                if pits:
                    gh = geometric_mean([p.high for p in pits])
                    gl = geometric_mean([p.low for p in pits]) if all(p.low > 0 for p in pits) else float("nan")
                    gs = geometric_mean([s.high for s in survs])
                    gb = geometric_mean(btcs)
                    ahead = sum(s.high > p.high for s, p in zip(survs, pits, strict=True))
                    print(
                        f"    geometric mean over {len(pits)} starts: point in time {num(gh)} / {num(gl)},"
                        f" survivors {num(gs)}, bitcoin {num(gb)}; survivors ahead in {ahead} of {len(pits)}"
                    )

    with (RES / "portfolios.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\nTURNOVER: share of a January top n still in the top n later")
    for n in (10, 20, 100):
        by_gap = {}
        for a, b, share in C.turnover(yearly, n):
            gap = int(b[:4]) - int(a[:4])
            by_gap.setdefault(gap, []).append(share)
        parts = [f"{g} y {pct(math.fsum(v) / len(v), 0)}" for g, v in sorted(by_gap.items()) if g in (1, 3, 5, 10)]
        print(f"  top {n}: " + ", ".join(parts))

    print("\nSURVIVAL IN THE TOP n: Kaplan-Meier from each coin's first January in the top n")
    for n in (20, 100):
        d = C.entry_durations(yearly, n)
        km = C.kaplan_meier([(k, cens) for _, k, cens in d.values()])
        parts = [f"{step} y {pct(s, 0)} (at risk {r})" for step, s, r, _ in km if step in (1, 2, 3, 5, 8, 10)]
        print(f"  top {n}: {len(d)} coins; still in after " + ", ".join(parts))

    print("\nTODAY'S LEADERS IN THE PAST")
    for n in (10, 20):
        keys = [c.key for c in latest.top(n)]
        for s in yearly:
            listed = sum(k in s.by_key for k in keys)
            in_top = sum(k in {c.key for c in s.top(n)} for k in keys)
            print(
                f"  today's top {n} in {s.date[:4]}: {listed} listed in the top {len(s.coins)}, {in_top} in the top {n}"
            )

    print("\nGONE FROM THE LIST: share of a January top 100 missing from the whole list later")
    for i, s in enumerate(yearly):
        keys = [c.key for c in s.top(100)]
        parts = []
        for later in yearly[i + 1 :]:
            gone = sum(k not in later.by_key for k in keys) / len(keys)
            parts.append(f"{later.date[:4]} {pct(gone, 0)}")
        if parts:
            print(f"  {s.date[:4]}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
