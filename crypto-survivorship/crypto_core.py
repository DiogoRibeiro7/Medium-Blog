"""Point-in-time crypto portfolios against portfolios of today's survivors, from yearly ranking snapshots.

The rules, in one place:

- Snapshots. A snapshot is CoinMarketCap's ranking on one date: rank, a coin identifier, name, symbol,
  market capitalisation, price and circulating supply, for the top coins listed that day (200 here).
- Point-in-time portfolio. On a start date, take the top N coins of that day's ranking, weighted by market
  capitalisation or equally, and hold them without trading for H years.
- Coins that leave the list. A coin missing from the end-date snapshot is valued in two ways:
  - high: at the most it could have been worth while ranking below the list, the market capitalisation of the
    last listed coin divided by the coin's own circulating supply at the start (supply rarely shrinks, so this
    is generous);
  - low: at zero.
  Every point-in-time result is reported with both, so the truth lies between them.
- Survivor portfolio. On the same start date, take only the coins that are in the top N of the latest snapshot
  and were already listed on the start date, with the same weighting. This is the history a chart or an index
  built from today's leaders shows.
- Survivorship gap. The survivor portfolio's value at the end divided by the point-in-time portfolio's,
  per year of holding. With the high valuation of dropped coins it is a lower bound on the true gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Coin:
    key: str  # stable identifier (CoinMarketCap slug where available)
    name: str
    symbol: str
    rank: int
    market_cap: float
    price: float
    supply: float


@dataclass
class Snapshot:
    date: str  # YYYY-MM-DD
    coins: list[Coin] = field(default_factory=list)

    def __post_init__(self):
        self.coins = sorted(self.coins, key=lambda c: c.rank)
        self.by_key = {c.key: c for c in self.coins}

    def top(self, n: int) -> list[Coin]:
        return self.coins[:n]

    @property
    def cutoff_cap(self) -> float:
        """Market capitalisation of the last listed coin: anything missing was worth less than this."""
        return self.coins[-1].market_cap


def years_between(a: str, b: str) -> float:
    ya, ma, da = (int(x) for x in a.split("-"))
    yb, mb, db = (int(x) for x in b.split("-"))
    return (yb - ya) + (mb - ma) / 12 + (db - da) / 365.25


# ----------------------------------------------------------------------------------------------- holding


@dataclass(frozen=True)
class Holding:
    coin: Coin
    multiple_high: float  # value at the end per unit at the start, dropped coins at the most they could be worth
    multiple_low: float  # dropped coins at zero
    listed_at_end: bool


def hold(coin: Coin, end: Snapshot) -> Holding:
    """Value of one unit of `coin` bought at its start-date price, on the end date.

    A coin missing from the end snapshot had a market capitalisation below the last listed coin's. Its price
    was therefore below cutoff / supply at the end, and, as long as its supply did not shrink, below
    cutoff / supply at the start. Its multiple is at most cutoff / (market capitalisation at the start).
    """
    later = end.by_key.get(coin.key)
    if later is not None:
        m = later.price / coin.price
        return Holding(coin, m, m, True)
    ceiling = end.cutoff_cap / coin.market_cap if coin.market_cap > 0 else 0.0
    return Holding(coin, ceiling, 0.0, False)


def weights(coins: list[Coin], scheme: str) -> list[float]:
    if scheme == "cap":
        total = math.fsum(c.market_cap for c in coins)
        return [c.market_cap / total for c in coins]
    if scheme == "equal":
        return [1.0 / len(coins)] * len(coins)
    raise ValueError(scheme)


@dataclass(frozen=True)
class Portfolio:
    start: str
    end: str
    n: int
    scheme: str
    members: int
    high: float  # portfolio multiple with dropped coins at their ceiling
    low: float  # with dropped coins at zero
    dropped: int  # members missing from the end snapshot
    dropped_weight: float  # their combined starting weight


def portfolio(coins: list[Coin], end: Snapshot, scheme: str, start: str, n: int) -> Portfolio:
    w = weights(coins, scheme)
    hs = [hold(c, end) for c in coins]
    return Portfolio(
        start=start,
        end=end.date,
        n=n,
        scheme=scheme,
        members=len(coins),
        high=math.fsum(wi * h.multiple_high for wi, h in zip(w, hs, strict=True)),
        low=math.fsum(wi * h.multiple_low for wi, h in zip(w, hs, strict=True)),
        dropped=sum(not h.listed_at_end for h in hs),
        dropped_weight=math.fsum(wi for wi, h in zip(w, hs, strict=True) if not h.listed_at_end),
    )


def point_in_time(start: Snapshot, end: Snapshot, n: int, scheme: str) -> Portfolio:
    return portfolio(start.top(n), end, scheme, start.date, n)


def survivors(start: Snapshot, end: Snapshot, latest: Snapshot, n: int, scheme: str) -> Portfolio:
    """Today's top n, as far as they were already listed on the start date."""
    keys = {c.key for c in latest.top(n)}
    coins = [c for c in start.coins if c.key in keys]
    return portfolio(coins, end, scheme, start.date, n)


def single(start: Snapshot, end: Snapshot, key: str) -> float:
    return hold(start.by_key[key], end).multiple_high


def annualised(multiple: float, years: float) -> float:
    return multiple ** (1.0 / years) - 1.0 if multiple > 0 else -1.0


def gap_per_year(survivor: float, pit: float, years: float) -> float:
    """How much faster per year the survivor portfolio grew than the point-in-time one."""
    if pit <= 0:
        return float("inf")
    return (survivor / pit) ** (1.0 / years) - 1.0


# ----------------------------------------------------------------------------------------------- survival


def entry_durations(snapshots: list[Snapshot], n: int) -> dict[str, tuple[str, int, bool]]:
    """For every coin that ever ranks in the top n: when it first did, how many later snapshots in a row it
    stayed in the top n, and whether it was still there at the last snapshot (censored).

    {key: (first snapshot date, later snapshots survived, censored)}. Re-entries after leaving are ignored.
    """
    tops = [{c.key for c in s.top(n)} for s in snapshots]
    out = {}
    for i, s in enumerate(snapshots):
        for c in s.top(n):
            if c.key in out:
                continue
            k = 0
            while i + k + 1 < len(snapshots) and c.key in tops[i + k + 1]:
                k += 1
            out[c.key] = (s.date, k, i + k + 1 == len(snapshots))
    return out


def kaplan_meier(durations: list[tuple[int, bool]]) -> list[tuple[int, float, int, int]]:
    """Discrete-time Kaplan-Meier from (later snapshots survived, censored) pairs.

    A coin with (k, False) left at step k + 1. A coin with (k, True) was last seen at step k and is not known
    after it. Returns [(step, share still in after the step, at risk, left)].
    """
    out = []
    surv = 1.0
    step = 1
    while True:
        at_risk = sum(1 for k, cens in durations if k >= step or (k == step - 1 and not cens))
        if at_risk == 0:
            break
        left = sum(1 for k, cens in durations if k == step - 1 and not cens)
        surv *= 1.0 - left / at_risk
        out.append((step, surv, at_risk, left))
        step += 1
    return out


def turnover(snapshots: list[Snapshot], n: int) -> list[tuple[str, str, float]]:
    """Share of each snapshot's top n that is still in the top n of every later snapshot."""
    out = []
    for i, a in enumerate(snapshots):
        keys = {c.key for c in a.top(n)}
        for b in snapshots[i + 1 :]:
            out.append((a.date, b.date, len(keys & {c.key for c in b.top(n)}) / len(keys)))
    return out


# ----------------------------------------------------------------------------------------------- readers


def number(text: str) -> float:
    """CoinMarketCap's displayed numbers: '$1,234.56', '18,123,456 BTC', '$0.0₄123' style subscripts, '--'."""
    t = text.strip().replace("$", "").replace(",", "").replace(" ", "").replace("\xa0", " ")
    t = t.split(" ")[0] if " " in t else t
    if t in ("", "--", "?", "N/A"):
        return float("nan")
    sub = {"₀": 0, "₁": 1, "₂": 2, "₃": 3, "₄": 4, "₅": 5, "₆": 6, "₇": 7, "₈": 8, "₉": 9}
    if any(ch in sub for ch in t):  # 0.0₄123 means 0.0000123
        head, _, rest = t.partition("0.0")
        zeros = "".join(str(sub[ch]) for ch in rest if ch in sub)
        digits = "".join(ch for ch in rest if ch not in sub)
        return float("0." + "0" * int(zeros) + digits)
    mult = 1.0
    for suffix, m in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if t.endswith(suffix):
            t, mult = t[:-1], m
            break
    return float(t) * mult
