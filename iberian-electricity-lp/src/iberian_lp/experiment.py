"""Every programme of sections 4-7, for every zone-period of the derived tables.

For each zone-period, ``LP_R``; for the Spanish or joint zone also ``LP_X``, its two partial
relaxations (section 7), ``LP_R`` with one more sell step at 0 EUR/MWh (500, 1,000 and
2,000 MW), and ``LP_R`` with the France flows at zero. Each programme's exact dual
interval is written to ``results/programmes.csv``, and every solve is checked against
HiGHS (``programme.solve``).

Three programmes were added after the critical review (docs/REVIEW.md, finding M2;
METHODS.md, change log), because an exclusive group's blocks are alternatives, of which
the anonymous curves cannot say which belong together: the blocks relaxed without the
exclusive blocks, and the blocks, and every order, relaxed with the exclusive blocks
deduplicated (of the steps with the same quantity in a zone-period, only the cheapest).
"""

from __future__ import annotations

import csv
import gzip
import importlib.metadata
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from iberian_lp.contract import E3_MW, E3_SENSITIVITY_MW, NON_CONVEX, SIMPLE
from iberian_lp.programme import Exact, Steps, solve

EXTRA_MW = tuple(sorted((E3_MW, *E3_SENSITIVITY_MW)))
PROGRAMMES = ("R", "X", "X_blocks", "X_sco", *(f"M{q}" for q in EXTRA_MW), "F0")
POST_HOC = ("X_c01", "X_blocks_dedup", "X_dedup")     # added after the review
SPANISH = ("MI", "ES")                         # the zone of Q2-Q4 (section 5)
FIELDS = ("date", "period", "hour", "quarter", "zone", "price", "matched_buy",
          "R_buy_min", "R_buy_max", "imports_fr", "exports_fr",
          *(f"{p}_{end}" for p in (*PROGRAMMES, *POST_HOC) for end in ("lo", "hi")))
PACKAGES = ("numpy", "scipy")

Bids = dict[tuple[str, str], tuple[list[int], list[int]]]


def fixed(text: str, places: int) -> int:
    """'21756.1' with places=1 -> 217561: the tables' numbers, as exact integers."""
    whole, _, frac = text.partition(".")
    if len(frac) != places:
        raise ValueError(f"Expected {places} decimals: {text!r}.")
    return int(whole + frac)


def _end(cents: float) -> str:
    return "inf" if cents == np.inf else "-inf" if cents == -np.inf else f"{cents / 100:.2f}"


def read_steps(path: Path) -> dict[tuple[int, str], Bids]:
    """One day's summed offered steps: (period, zone) -> (side, type) -> (prices, tenths)."""
    out: dict[tuple[int, str], Bids] = defaultdict(lambda: defaultdict(lambda: ([], [])))
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for s in csv.DictReader(stream):
            prices, qty = out[(int(s["period"]), s["zone"])][(s["side"], s["type"])]
            prices.append(fixed(s["price"], 2))
            qty.append(fixed(s["mw"], 1))
    return out


def _stack(bids: Bids, keys: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    prices = [p for k in keys for p in bids.get(k, ([], []))[0]]
    qty = [q for k in keys for q in bids.get(k, ([], []))[1]]
    return np.array(prices, dtype=np.int64), np.array(qty, dtype=np.int64)


def deduplicated(bids: Bids, kind: str = "C04") -> Bids:
    """The bids with, of the steps of ``kind`` with the same quantity, only the cheapest."""
    cheapest: dict[int, int] = {}
    for price, qty in sorted(zip(*bids.get(("V", kind), ([], [])), strict=True)):
        cheapest.setdefault(qty, price)
    out = dict(bids)
    out[("V", kind)] = ([p for p in cheapest.values()], [q for q in cheapest])
    return out


def programmes(bids: Bids, row: dict[str, str]) -> dict[str, Steps]:
    """The zone-period's programmes (sections 4 and 7, and those added after the review).
    R is fixed demand minus fixed supply: exports, minus imports and the non-convex volume
    not relaxed."""
    n = {t: fixed(row[f"matched_sell_{t}"], 1) for t in NON_CONVEX}
    imp = fixed(row["imports_fr"], 1) + fixed(row["imports_ib"], 1)
    exp = fixed(row["exports_fr"], 1) + fixed(row["exports_ib"], 1)
    buy = _stack(bids, [("C", SIMPLE)])
    unique = deduplicated(bids)

    def lp(relaxed: tuple[str, ...], extra: int = 0, france: bool = True,
           dedup: bool = False) -> Steps:
        sell_p, sell_q = _stack(unique if dedup else bids,
                                [("V", t) for t in (SIMPLE, *relaxed)])
        if extra:
            sell_p, sell_q = np.append(sell_p, 0), np.append(sell_q, 10 * extra)
        r = exp - imp - sum(n[t] for t in NON_CONVEX if t not in relaxed)
        if not france:
            r += fixed(row["imports_fr"], 1) - fixed(row["exports_fr"], 1)
        return Steps(sell_p, sell_q, buy[0], buy[1], r)

    out = {"R": lp(())}
    if row["zone"] in SPANISH:
        out |= {"X": lp(NON_CONVEX), "X_blocks": lp(("C01", "C04")), "X_sco": lp(("C02",)),
                **{f"M{q}": lp((), extra=q) for q in EXTRA_MW}, "F0": lp((), france=False),
                "X_c01": lp(("C01",)), "X_blocks_dedup": lp(("C01", "C04"), dedup=True),
                "X_dedup": lp(NON_CONVEX, dedup=True)}
    return out


def solve_day(derived: Path, day: str, rows: list[dict[str, str]], checks: dict[str, Any],
              solver: Callable[[Steps], tuple[Exact, float]] = solve
              ) -> Iterator[dict[str, str]]:
    steps = read_steps(derived / "steps" / f"{day.replace('-', '')}.csv.gz")
    for row in rows:
        price = row["price_pt"] if row["zone"] == "PT" else row["price_es"]
        out = {"date": day, **{k: row[k] for k in ("period", "hour", "quarter", "zone")},
               "price": price, "matched_buy": row["matched_buy"],
               "imports_fr": row["imports_fr"], "exports_fr": row["exports_fr"],
               **{f"{p}_{end}": "" for p in (*PROGRAMMES, *POST_HOC) for end in ("lo", "hi")}}
        for name, lp in programmes(steps[(int(row["period"]), row["zone"])], row).items():
            e, gap = solver(lp)
            checks["solves"] += 1
            checks["objective_gap_max"] = max(checks["objective_gap_max"], gap)
            out[f"{name}_lo"], out[f"{name}_hi"] = _end(e.lo), _end(e.hi)
            if name == "R":
                out["R_buy_min"] = f"{e.buy_min / 10:.1f}"
                out["R_buy_max"] = f"{e.buy_max / 10:.1f}"
        yield out


def run(derived: Path, out: Path, progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Solve every programme of every drawn day; write ``out`` (programmes.csv)."""
    with (derived / "periods.csv").open(encoding="utf-8") as stream:
        periods = list(csv.DictReader(stream))
    by_day: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in periods:
        by_day[row["date"]].append(row)
    checks: dict[str, Any] = {"solves": 0, "objective_gap_max": 0.0,
                              "versions": {p: importlib.metadata.version(p) for p in PACKAGES}}
    rows: list[dict[str, str]] = []
    for day in sorted(by_day):
        rows += solve_day(derived, day, by_day[day], checks)
        progress(f"{day}: {len(by_day[day])} zone-periods, {checks['solves']:,} solves")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return checks
