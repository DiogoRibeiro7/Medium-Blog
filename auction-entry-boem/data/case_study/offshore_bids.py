"""Game and Auction Theory for Data Scientists - case study.

Bids in thirteen US offshore oil and gas lease sales in the Gulf, August 2017
to August 2026, from the files published by the Bureau of Ocean Energy
Management (BOEM). Run build_dataset.py first; it writes data/bids.csv and
checks it against BOEM's published totals.

Everything the article reports is printed by ``python offshore_bids.py``.
The figures are produced by ``python figures.py``.

Requires numpy and pandas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def load_bids() -> pd.DataFrame:
    bids = pd.read_csv(HERE / "data" / "bids.csv", parse_dates=["sale_date"],
                       dtype={"sale": str, "block": str, "lead_company": str})
    bids["ratio"] = bids.amount / bids.minimum_bid        # bid relative to the minimum
    bids["block_id"] = bids.protraction + "-" + bids.block
    return bids


def high_bids(bids: pd.DataFrame) -> pd.DataFrame:
    """One row per tract offered in a sale that received bids: the high bid."""
    return bids[bids["rank"] == 1].copy()


# ---------------------------------------------------------------------------
# Competition and the minimum bid
# ---------------------------------------------------------------------------

def share_within(ratio: pd.Series, limits=(1.05, 1.10, 1.25, 1.50, 2.0)) -> dict:
    return {x: float((ratio <= x + 1e-12).mean()) for x in limits}


def fixed_effect_ratios(tracts: pd.DataFrame) -> dict:
    """Least squares of log(high bid / minimum) on dummies for two and for
    three or more bids, with a fixed effect for every sale and depth band.
    Returns the multiplicative differences against single-bid tracts."""
    y = np.log(tracts.ratio.to_numpy())
    cells = pd.get_dummies(tracts.sale + "|" + tracts.depth_band, drop_first=False).astype(float)
    n = tracts.n_bids.clip(upper=3)
    X = np.column_stack([cells.to_numpy(), (n == 2).astype(float), (n == 3).astype(float)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return {"two bids": float(np.exp(beta[-2])), "three or more": float(np.exp(beta[-1]))}


def standard_bids(bids: pd.DataFrame, min_repeats: int = 3) -> pd.Series:
    """True for a bid whose company submitted the same amount on at least
    ``min_repeats`` tracts in the same sale."""
    counts = bids.groupby(["sale", "lead_company", "amount"]).amount.transform("size")
    return counts >= min_repeats


# ---------------------------------------------------------------------------
# Tracts with several bids
# ---------------------------------------------------------------------------

def first_and_second(bids: pd.DataFrame) -> pd.DataFrame:
    multi = bids[bids.n_bids >= 2]
    wide = multi.pivot_table(index=["sale", "lease"], columns="rank", values="amount")
    out = pd.DataFrame({"first": wide[1], "second": wide[2]})
    out["n_bids"] = multi.groupby(["sale", "lease"]).n_bids.first()
    return out


def log_spread(bids: pd.DataFrame) -> dict:
    """Spread of log bids on the same tract. Under the model of Part 5, with
    bids proportional to noisy estimates of a common value, this estimates
    the standard deviation of the estimation noise (in log units)."""
    multi = bids[bids.n_bids >= 2].assign(lb=lambda d: np.log(d.amount))
    var = multi.groupby(["sale", "lease"]).lb.var(ddof=1)
    pooled = float(np.sqrt(var.mean()))
    # Robust version from the gap between the two highest bids: for two draws
    # from N(0, s^2), the median absolute difference is 0.6745 * sqrt(2) * s.
    fs = first_and_second(bids)
    gap = np.log(fs["first"] / fs["second"])
    robust = float(gap.median() / (0.6745 * np.sqrt(2)))
    return {"tracts": int(var.size), "pooled": pooled, "robust": robust}


# ---------------------------------------------------------------------------
# BOEM's review of high bids
# ---------------------------------------------------------------------------

def rejected_then_rebid(tracts: pd.DataFrame) -> pd.DataFrame:
    """For every rejected high bid, the next sale in which the same block
    received bids, with the high bid there and its outcome. The column
    'drift' is the ratio of the median high bid relative to the minimum in
    the later sale to the same median in the earlier sale, a control for
    changes in bidding across sales."""
    median_ratio = tracts.groupby("sale").ratio.median()
    rows = []
    for _, r in tracts[tracts.tract_rejected].sort_values("sale_date").iterrows():
        later = tracts[(tracts.block_id == r.block_id) & (tracts.sale_date > r.sale_date)]
        if later.empty:
            rows.append({"sale": r.sale, "block_id": r.block_id, "area": r.area,
                         "block": r.block, "amount": r.amount, "rebid": False})
            continue
        nxt = later.sort_values("sale_date").iloc[0]
        rows.append({"sale": r.sale, "block_id": r.block_id, "area": r.area,
                     "block": r.block, "amount": r.amount, "rebid": True,
                     "next_sale": nxt.sale, "next_amount": nxt.amount,
                     "next_rejected": bool(nxt.tract_rejected), "next_reviewed": bool(nxt.review_known),
                     "same_company": nxt.lead_company == r.lead_company,
                     "next_n_bids": nxt.n_bids,
                     "drift": median_ratio[nxt.sale] / median_ratio[r.sale]})
    return pd.DataFrame(rows)


def block_history(tracts: pd.DataFrame, block_id: str) -> pd.DataFrame:
    h = tracts[tracts.block_id == block_id].sort_values("sale_date")
    return h[["sale", "sale_date", "area", "block", "amount", "n_bids", "tract_rejected"]]


# ---------------------------------------------------------------------------
# A higher minimum bid: what is at stake, and the break-even
# ---------------------------------------------------------------------------

def minimum_bid_breakeven(tracts: pd.DataFrame, new_per_acre: float) -> dict:
    """Raise the minimum to ``new_per_acre`` for the tracts in ``tracts``.

    Tracts whose high bid is already at or above the new minimum are assumed
    to pay the same as before (a conservative assumption: theory says bids
    above a reserve rise when it is raised). For the tracts below it, the
    data cannot say whether the bidder would pay the new minimum or walk
    away. If a share p of them pay it, the change in bonus is

        p * sum(new minimum) - sum(current bids),

    which is zero at the break-even share p* = sum(current bids) / sum(new minimum).
    """
    acres = np.ceil(tracts.acres - 1e-9)
    new_min = new_per_acre * acres
    below = tracts.amount < new_min - 1e-6
    current = tracts.amount[below].sum()
    if_all_pay = new_min[below].sum()
    return {
        "new_per_acre": new_per_acre,
        "share_tracts_below": float(below.mean()),
        "tracts_below": int(below.sum()),
        "share_bonus_below": float(current / tracts.amount.sum()),
        "gain_if_all_pay": float((if_all_pay - current) / tracts.amount.sum()),
        "loss_if_none_pay": float(-current / tracts.amount.sum()),
        "breakeven_share": float(current / if_all_pay) if if_all_pay > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def money(x: float) -> str:
    return f"${x / 1e6:,.1f} million"


def main() -> None:
    bids = load_bids()
    tracts = high_bids(bids)

    print("=" * 72)
    print("1. THE DATA")
    print("=" * 72)
    first, last = bids.sale_date.min(), bids.sale_date.max()
    print(f"Sales {bids.sale.nunique()}, from {first:%d %B %Y} to {last:%d %B %Y}")
    print(f"Bids {len(bids):,}; tracts receiving bids {len(tracts):,}; "
          f"distinct blocks {tracts.block_id.nunique():,}")
    companies = {c for s in bids.companies for c in s.split(";")}
    print(f"Companies bidding, alone or in a group: {len(companies)}; "
          f"joint bids {pct((bids.n_partners > 1).mean())}")
    print(f"Sum of all bids {money(bids.amount.sum())}; sum of high bids {money(tracts.amount.sum())}")
    print(f"Tracts in water shallower than 400 m: {(tracts.shallow).sum()}, "
          f"400 m or deeper: {(~tracts.shallow).sum()}")

    print()
    print("=" * 72)
    print("2. COMPETITION")
    print("=" * 72)
    counts = tracts.n_bids.value_counts().sort_index()
    print("Tracts by number of bids: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    print(f"Single-bid tracts: {pct((tracts.n_bids == 1).mean())} "
          f"(shallow {pct((tracts[tracts.shallow].n_bids == 1).mean())}, "
          f"deep {pct((tracts[~tracts.shallow].n_bids == 1).mean())}); "
          f"average bids per tract {tracts.n_bids.mean():.2f}")
    by_sale = tracts.groupby("sale").n_bids.apply(lambda s: (s == 1).mean())
    print(f"Single-bid share by sale: from {pct(by_sale.min())} to {pct(by_sale.max())}")
    print(f"Share of the sum of high bids from single-bid tracts: "
          f"{pct(tracts.amount[tracts.n_bids == 1].sum() / tracts.amount.sum())}")

    print()
    print("=" * 72)
    print("3. HIGH BIDS AGAINST THE MINIMUM BID")
    print("=" * 72)
    print("Minimum bid: $25 per acre below 400 m of water, $100 per acre at 400 m or deeper")
    low = bids.groupby("shallow").per_acre.min()
    print(f"Lowest bid per acre: shallow ${low[True]:.2f}, deep ${low[False]:.2f}; "
          f"bids exactly at the minimum: {int(np.isclose(bids.amount, bids.minimum_bid).sum())}")
    q = tracts.ratio.quantile([0.25, 0.5, 0.75, 0.9])
    print("High bid / minimum bid, quartiles and 90th percentile: "
          + ", ".join(f"{v:.2f}" for v in q))
    print("Share of tracts won with a bid at most x times the minimum: "
          + ", ".join(f"{x:.2f}: {pct(s)}" for x, s in share_within(tracts.ratio).items()))
    for label, sub in (("shallow", tracts[tracts.shallow]), ("deep", tracts[~tracts.shallow])):
        s = share_within(sub.ratio)
        print(f"  {label:7s}: median {sub.ratio.median():.2f}; within 5% {pct(s[1.05])}, "
              f"within 10% {pct(s[1.10])}")
    med = tracts.groupby(tracts.n_bids.clip(upper=3)).ratio.median()
    print(f"Median high bid / minimum by number of bids: one {med[1]:.2f}, two {med[2]:.2f}, "
          f"three or more {med[3]:.2f}; two or more {tracts[tracts.n_bids > 1].ratio.median():.2f}")
    fe = fixed_effect_ratios(tracts)
    print(f"Same comparison within sale and depth band (fixed effects): two bids "
          f"{fe['two bids']:.2f} times, three or more {fe['three or more']:.2f} times "
          f"the single-bid level")
    std = standard_bids(bids)
    print(f"Standard bids (same company, same amount, 3 or more tracts in a sale): "
          f"{pct(std.mean())} of all bids, {pct(std[bids['rank'] == 1].mean())} of high bids")
    top = bids[std].groupby(["sale", "lead_company", "amount"]).size().sort_values().iloc[-1:]
    (sale, _, amount), n = top.index[0], int(top.iloc[0])
    acres = bids[(bids.sale == sale) & (bids.amount == amount)].acres.max()
    print(f"  largest: {n} tracts in sale {sale} at ${amount:,.0f} each "
          f"(${amount / acres:.2f} per acre on a {acres:,.0f}-acre block)")
    same = bids[(bids.sale == sale) & (bids.amount == amount)]
    print(f"  of these, single-bid tracts {int((same.n_bids == 1).sum())}, in water shallower "
          f"than 400 m {int(same.shallow.sum())}")
    print(f"Median high bid / minimum by sale: from {tracts.groupby('sale').ratio.median().min():.2f} "
          f"to {tracts.groupby('sale').ratio.median().max():.2f}")

    print()
    print("=" * 72)
    print("4. TRACTS WITH SEVERAL BIDS")
    print("=" * 72)
    fs = first_and_second(bids)
    r12 = fs["first"] / fs["second"]
    print(f"Tracts with two or more bids: {len(fs)}")
    print(f"High bid / second bid: median {r12.median():.2f}, quartiles "
          f"{r12.quantile(0.25):.2f} and {r12.quantile(0.75):.2f}; more than twice: {pct((r12 > 2).mean())}")
    print(f"Second bid as a share of the high bid: median {pct((fs['second'] / fs['first']).median())}")
    second_ratio = bids[(bids.n_bids >= 2) & (bids["rank"] == 2)].ratio
    print(f"Second bids within 10% of the minimum bid: {pct((second_ratio <= 1.10 + 1e-12).mean())}")
    left = fs["first"] - fs["second"]
    print(f"Gap between high and second bid: median {pct((left / fs['first']).median())} of the high bid "
          f"({money(left.median())})")
    print(f"Money left on the table (high minus second bid): {money(left.sum())}, "
          f"{pct(left.sum() / fs['first'].sum())} of the high bids on these tracts; "
          f"the largest 10 tracts account for {pct(left.nlargest(10).sum() / left.sum())} of it")
    spread = log_spread(bids)
    print(f"Standard deviation of log bids on the same tract: pooled {spread['pooled']:.2f}, "
          f"robust (from the median gap) {spread['robust']:.2f}")
    for s in (spread["robust"], spread["pooled"]):
        over = s / np.sqrt(np.pi)     # E[max of two N(0, s^2)]
        print(f"  if all of it were estimation noise (s = {s:.2f}), the higher of two "
              f"estimates would exceed the truth by {over:.2f} log units "
              f"(a factor of {np.exp(over):.2f}) on average")

    print()
    print("=" * 72)
    print("5. BOEM'S REVIEW OF HIGH BIDS")
    print("=" * 72)
    rev = tracts[tracts.review_known]
    rej = rev[rev.tract_rejected]
    print(f"High bids reviewed (all sales except BBG3, whose final recap is not "
          f"published): {len(rev):,}; rejected {len(rej)} ({pct(len(rej) / len(rev))}), "
          f"{money(rej.amount.sum())}, {pct(rej.amount.sum() / rev.amount.sum())} of the reviewed high bids")
    for label, sub in (("shallow", rev[rev.shallow]), ("deep", rev[~rev.shallow])):
        print(f"  {label:7s}: {int(sub.tract_rejected.sum())} of {len(sub)} ({pct(sub.tract_rejected.mean())})")
    for label, sub in (("one bid", rev[rev.n_bids == 1]), ("two or more", rev[rev.n_bids > 1])):
        print(f"  {label:11s}: {int(sub.tract_rejected.sum())} of {len(sub)} ({pct(sub.tract_rejected.mean())})")
    deep = rev[~rev.shallow]
    print(f"Deep water, median high bid / minimum: rejected {deep[deep.tract_rejected].ratio.median():.2f}, "
          f"accepted {deep[~deep.tract_rejected].ratio.median():.2f}")
    print("Rejected by sale: " + ", ".join(f"{s} {int(v)}" for s, v in
                                          rev.groupby('sale').tract_rejected.sum().items()))

    rb = rejected_then_rebid(tracts)
    again = rb[rb.rebid]
    print(f"Rejected blocks that received bids again in a later sale: {len(again)} of {len(rb)}")
    higher = again.next_amount > again.amount
    higher_accepted = higher & again.next_reviewed.astype(bool) & ~again.next_rejected.astype(bool)
    print(f"  of all {len(rb)} rejected blocks: a higher next bid {int(higher.sum())} "
          f"({pct(higher.sum() / len(rb))}), a higher next bid that BOEM accepted "
          f"{int(higher_accepted.sum())} ({pct(higher_accepted.sum() / len(rb))})")
    print(f"  next high bid higher than the rejected one: {pct((again.next_amount > again.amount).mean())}; "
          f"median ratio next / rejected {(again.next_amount / again.amount).median():.2f}")
    print(f"  median change in the typical bid between the same two sales "
          f"(median high bid / minimum): {again.drift.median():.2f}")
    print(f"  next high bid from the same company: {pct(again.same_company.mean())}; "
          f"with two or more bids: {pct((again.next_n_bids > 1).mean())}")
    known = again[again.next_reviewed.astype(bool)].astype({"next_rejected": bool})
    print(f"  next high bid accepted: {int((~known.next_rejected).sum())} of {len(known)} "
          f"({pct((~known.next_rejected).mean())}); rejected again: {int(known.next_rejected.sum())}")
    hist = block_history(tracts, "NH16-10-40")
    print(f"Block {hist.area.iloc[0]} {hist.block.iloc[0]}:")
    for _, h in hist.iterrows():
        print(f"  sale {h.sale} ({h.sale_date:%b %Y}): high bid {money(h.amount)}, "
              f"{h.n_bids} bid{'s' if h.n_bids > 1 else ''}, "
              f"{'rejected' if h.tract_rejected else 'accepted'}")

    print()
    print("=" * 72)
    print("6. A HIGHER MINIMUM BID: WHAT IS AT STAKE")
    print("=" * 72)
    print("Bonus only; tracts already above the new minimum assumed to pay the same.")
    for label, sub, grid in (("deep (now $100 per acre)", tracts[~tracts.shallow], (125, 150, 200, 300)),
                             ("shallow (now $25 per acre)", tracts[tracts.shallow], (30, 40, 50, 75))):
        print(f"  {label}: {len(sub)} tracts, high bids {money(sub.amount.sum())}")
        for r in grid:
            b = minimum_bid_breakeven(sub, r)
            print(f"    ${r:>3d} per acre: {pct(b['share_tracts_below'])} of tracts below it "
                  f"({b['tracts_below']}), {pct(b['share_bonus_below'])} of the bonus; "
                  f"change if all pay {'+' if b['gain_if_all_pay'] >= 0 else ''}{pct(b['gain_if_all_pay'])}, "
                  f"if none pay {pct(b['loss_if_none_pay'])}; break-even share {pct(b['breakeven_share'], 0)}")


if __name__ == "__main__":
    main()
