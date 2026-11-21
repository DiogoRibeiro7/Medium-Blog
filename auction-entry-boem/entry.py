"""What one more bidder is worth in BOEM's Gulf lease sales, 2013-2026.

    python build_dataset.py     # data/bids.csv (run once; checked against BOEM's totals)
    python entry.py             # prints every number in the article, writes results/

The question is Bulow and Klemperer's (1996): is a seller better off with one more bidder
or with a better reserve price? In these sales nine tracts in ten get one bid. The script
measures how much of the revenue gap between contested and uncontested tracts comes from
competition and how much from which tracts attract competition, and then compares the value
of a second bidder on today's single-bid tracts with what a higher minimum bid could add.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
RES = HERE / "results"
SEED = 20261121
BOOT = 1000


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    b = pd.read_csv(DATA / "bids.csv", dtype={"sale": str, "lead_company": str, "companies": str},
                    parse_dates=["sale_date"])
    b["cell"] = b.sale + " | " + b.depth_band
    t = (b[b["rank"] == 1][["sale", "sale_date", "lease", "block_id", "depth_band", "shallow", "acres",
                             "n_bids", "amount", "minimum_bid", "lead_company", "tract_rejected",
                             "review_known", "cell"]]
         .rename(columns={"amount": "high", "n_bids": "n"}).copy())
    second = b[b["rank"] == 2].set_index(["sale", "lease"]).amount
    t["second"] = [second.get((s, l), np.nan) for s, l in zip(t.sale, t.lease)]
    t["ratio"] = t.high / t.minimum_bid
    t["lr"] = np.log(t.ratio)
    t["ratio2"] = t.second / t.minimum_bid
    t["ncat"] = pd.cut(t.n, [0, 1, 2, 99], labels=["1", "2", "3+"]).astype(str)
    # standard bids: the same company bidding the same amount on three or more tracts in a sale
    std = b.groupby(["sale", "lead_company", "amount"]).lease.transform("size") >= 3
    b["standard_bid"] = std
    t = t.merge(b[b["rank"] == 1][["sale", "lease", "standard_bid"]], on=["sale", "lease"])
    return b, t.reset_index(drop=True)


def demean(v: np.ndarray, groups: list[np.ndarray], tol: float = 1e-12, max_iter: int = 10_000) -> np.ndarray:
    """Remove any number of fixed effects by alternating projections."""
    v = v.astype(float).copy()
    for _ in range(max_iter):
        before = v.copy()
        for g in groups:
            means = np.bincount(g, weights=v) / np.bincount(g)
            v -= means[g]
        if np.max(np.abs(v - before)) < tol:
            break
    return v


def fe_regression(df: pd.DataFrame, y: str, xs: list[str], fe: list[str], cluster: str = "block_id") -> pd.DataFrame:
    """OLS of y on xs with the fixed effects in fe absorbed; standard errors clustered by block."""
    groups = [pd.factorize(df[f])[0] for f in fe]
    Y = demean(df[y].to_numpy(), groups)
    X = np.column_stack([demean(df[x].to_numpy(), groups) for x in xs])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    e = Y - X @ beta
    cl = pd.factorize(df[cluster])[0]
    G = cl.max() + 1
    scores = np.zeros((G, X.shape[1]))
    np.add.at(scores, cl, X * e[:, None])
    V = XtX_inv @ (scores.T @ scores) @ XtX_inv * G / (G - 1)
    se = np.sqrt(np.diag(V))
    return pd.DataFrame({"coef": beta, "se": se, "n": len(df), "clusters": G}, index=xs)


def fe_multiplier(df: pd.DataFrame, y: str, xs: list[str], fe: list[str], term: str) -> tuple[float, float, float, int]:
    r = fe_regression(df, y, xs, fe).loc[term]
    return (float(np.exp(r.coef)), float(np.exp(r.coef - 1.96 * r.se)), float(np.exp(r.coef + 1.96 * r.se)), int(r.n))


# ---------------------------------------------------------------------------
# A second bidder as a second draw
# ---------------------------------------------------------------------------

def second_draw_uplift(t: pd.DataFrame, rng: np.random.Generator | None = None) -> float:
    """Dollar uplift on single-bid tracts if each received one more bid, drawn independently
    from the high/minimum ratios of the single-bid tracts in the same sale and depth band:
    sum of minimum x E[max(own ratio, draw)] over sum of high bids."""
    num = den = 0.0
    for _, g in t[t.n == 1].groupby("cell"):
        if len(g) < 10:
            continue
        if rng is not None:                               # bootstrap: resample tracts within the cell
            g = g.iloc[rng.integers(0, len(g), len(g))]
        r = g.ratio.to_numpy()
        mins = g.minimum_bid.to_numpy()
        s = np.sort(r)
        csum = np.concatenate([[0.0], np.cumsum(s)])
        k = len(s)
        pos = np.searchsorted(s, r, side="right")        # draws <= own ratio
        emax = (pos * r + (csum[-1] - csum[pos])) / k
        num += (mins * emax).sum()
        den += (mins * r).sum()
    return num / den


def pair_uplift(t: pd.DataFrame, idx: np.ndarray | None = None) -> float:
    """On two-bid tracts, high bid over the mean of the two bids: the expected gain from a second
    bid when the two bids on a tract are as far apart as they are on contested tracts."""
    two = t[t.n == 2]
    if idx is not None:
        two = two.iloc[idx]
    return float(two.high.sum() / ((two.high + two.second) / 2).sum())


def order_stat_cells(t: pd.DataFrame, n: int) -> tuple[float, float]:
    """E[max of n draws] / E[one draw] from the single-bid ratios of each cell, weighted by the
    number of tracts with n bids in the cell; and the observed mean ratio of n-bid tracts over
    the mean ratio of single-bid tracts, with the same weights."""
    pred, obs, w = [], [], []
    for _, g in t.groupby("cell"):
        s = np.sort(g[g.n == 1].ratio.to_numpy()); m = g[g.ncat == ("2" if n == 2 else "3+")]
        if len(s) < 10 or len(m) == 0:
            continue
        k = len(s); cdf = np.arange(1, k + 1) / k
        emax = np.sum(s * (cdf ** n - (cdf - 1 / k) ** n))
        pred.append(emax / s.mean()); obs.append(m.ratio.mean() / s.mean()); w.append(len(m))
    w = np.array(w)
    return float(np.average(pred, weights=w)), float(np.average(obs, weights=w))


# ---------------------------------------------------------------------------
# A higher minimum bid on the same tracts
# ---------------------------------------------------------------------------

def reserve_change(t: pd.DataFrame, shallow: bool, new_per_acre: float) -> dict:
    g = t[(t.n == 1) & (t.shallow == shallow)]
    old = 25.0 if shallow else 100.0
    new_min = g.minimum_bid * new_per_acre / old
    below = g.high < new_min - 0.5
    base = g.high.sum()
    all_pay = (np.where(below, new_min, g.high)).sum()
    none_pay = g.high[~below].sum()
    gain, loss = all_pay - base, base - none_pay
    return dict(depth="shallow" if shallow else "deep", new_per_acre=new_per_acre, tracts=len(g),
                below=int(below.sum()), below_share=below.mean(), base=base,
                all_pay=all_pay / base - 1, none_pay=none_pay / base - 1,
                break_even_share=loss / (loss + gain) if gain + loss > 0 else np.nan)


def main() -> None:
    RES.mkdir(exist_ok=True)
    b, t = load()
    rng = np.random.default_rng(SEED)
    out = io.StringIO()
    with redirect_stdout(out):
        print("=" * 78); print("1. THE DATA"); print("=" * 78)
        invalid = t[t.ratio < 0.999]
        print(f"Sales {t.sale.nunique()}, from {t.sale_date.min():%d %B %Y} to {t.sale_date.max():%d %B %Y}; "
              f"bids {len(b):,}; tracts receiving bids {len(t):,}; companies bidding alone or in a group "
              f"{b.companies.str.split(';').explode().nunique()}")
        print(f"Tracts whose high bid is below the minimum bid (BOEM's 'invalid bid'), left out below: {len(invalid)}")
        t = t[t.ratio >= 0.999].reset_index(drop=True)
        counts = t.n.value_counts().sort_index()
        print("Tracts by number of bids: " + ", ".join(f"{k}: {v:,}" for k, v in counts.items()))
        print(f"Single-bid tracts: {100 * (t.n == 1).mean():.1f}%; shallow {100 * (t[t.shallow].n == 1).mean():.1f}%, "
              f"deep {100 * (t[~t.shallow].n == 1).mean():.1f}%")
        old, new = t[t.sale_date < "2017-08-01"], t[t.sale_date >= "2017-08-01"]
        print(f"  August 2013 to March 2017 (8 sales): {100 * (old.n == 1).mean():.1f}% of {len(old):,} tracts; "
              f"August 2017 to August 2026 (13 sales): {100 * (new.n == 1).mean():.1f}% of {len(new):,}")
        bysale = t.groupby("sale").agg(date=("sale_date", "first"), single=("n", lambda s: (s == 1).mean())).sort_values("date")
        print(f"Single-bid share by sale: from {100 * bysale.single.min():.1f}% to {100 * bysale.single.max():.1f}%")
        comp = b.assign(c=b.companies.str.split(";")).explode("c").groupby("sale").c.nunique()
        print(f"Companies per sale: median {comp.median():.0f}, range {comp.min()} to {comp.max()}")
        print(f"Sum of high bids ${t.high.sum() / 1e6:,.1f} million; on single-bid tracts ${t[t.n == 1].high.sum() / 1e6:,.1f} million "
              f"({100 * t[t.n == 1].high.sum() / t.high.sum():.1f}%)")
        bysale.to_csv(RES / "single_bid_share_by_sale.csv")

        print(); print("=" * 78); print("2. CONTESTED AND UNCONTESTED TRACTS SIDE BY SIDE"); print("=" * 78)
        for k in ("1", "2", "3+"):
            g = t[t.ncat == k]
            print(f"{k} bid(s): {len(g):,} tracts, median high bid / minimum {g.ratio.median():.2f}, mean {g.ratio.mean():.2f}")
        rows = []
        t["two"] = (t.ncat == "2").astype(int); t["three"] = (t.ncat == "3+").astype(int)
        t["multi"] = (t.n >= 2).astype(int)
        for name, xs, term in (("two bids vs one, same sale and depth band", ["two", "three"], "two"),
                               ("three or more vs one, same sale and depth band", ["two", "three"], "three"),
                               ("two or more vs one, same sale and depth band", ["multi"], "multi")):
            m, lo, hi, n = fe_multiplier(t, "lr", xs, ["cell"], term)
            rows.append(dict(comparison=name, multiplier=m, lo=lo, hi=hi, n=n))
            print(f"{name}: high bid x{m:.2f} [95% {lo:.2f}, {hi:.2f}], tracts {n:,}")

        print(); print("=" * 78); print("3. THE SAME BLOCK IN DIFFERENT SALES"); print("=" * 78)
        rep = t.groupby("block_id").sale.transform("size") >= 2
        tb = t[rep].copy()
        nvar = tb.groupby("block_id").n.agg(lambda s: s.min() == 1 and s.max() >= 2)
        print(f"Blocks bid on in two or more sales: {tb.block_id.nunique():,} ({len(tb):,} tract-sales); "
              f"with one bid in one sale and two or more in another: {int(nvar.sum())}")
        m, lo, hi, n = fe_multiplier(tb, "lr", ["multi"], ["cell"], "multi")
        print(f"same tracts, cross-section comparison (sale x depth band): x{m:.2f} [{lo:.2f}, {hi:.2f}]")
        rows.append(dict(comparison="repeat blocks, sale x depth band only", multiplier=m, lo=lo, hi=hi, n=n))
        m, lo, hi, n = fe_multiplier(tb, "lr", ["multi"], ["block_id", "cell"], "multi")
        print(f"same block, two or more bids vs one (block and sale x depth band effects): x{m:.2f} [{lo:.2f}, {hi:.2f}], tract-sales {n:,}")
        rows.append(dict(comparison="same block, different sales", multiplier=m, lo=lo, hi=hi, n=n))
        # leave out appearances that follow a rejected high bid on the same block
        tb = tb.sort_values(["block_id", "sale_date"])
        tb["after_rejection"] = tb.groupby("block_id").tract_rejected.shift(1).fillna(False).astype(bool)
        nr = tb[~tb.after_rejection]
        nr = nr[nr.groupby("block_id").sale.transform("size") >= 2]
        m, lo, hi, n = fe_multiplier(nr, "lr", ["multi"], ["block_id", "cell"], "multi")
        print(f"  without appearances that follow a rejected high bid: x{m:.2f} [{lo:.2f}, {hi:.2f}], tract-sales {n:,}")
        rows.append(dict(comparison="same block, without re-offers after a rejection", multiplier=m, lo=lo, hi=hi, n=n))
        gap = tb.groupby("block_id").sale_date.diff().dt.days.dropna() / 365.25
        print(f"  years between consecutive appearances of a block: median {gap.median():.1f}")

        print(); print("=" * 78); print("4. THE SAME COMPANY IN THE SAME SALE"); print("=" * 78)
        bb = b.merge(t[["sale", "lease", "n"]], on=["sale", "lease"])
        bb = bb[bb.amount >= bb.minimum_bid - 0.5].copy()
        bb["lrb"] = np.log(bb.amount / bb.minimum_bid); bb["contested"] = (bb.n >= 2).astype(int)
        bb["cs"] = bb.lead_company + "_" + bb.sale
        keep = bb.groupby("cs").contested.transform(lambda s: s.nunique() == 2)
        bk = bb[keep]
        m, lo, hi, n = fe_multiplier(bk, "lrb", ["contested"], ["cell", "cs"], "contested")
        print(f"bids by companies that bid on both contested and uncontested tracts in a sale: {len(bk):,} bids, "
              f"{bk.cs.nunique()} company-sales")
        print(f"same company, same sale: bid on a contested tract / bid on an uncontested tract x{m:.2f} [{lo:.2f}, {hi:.2f}]")
        rows.append(dict(comparison="same company, same sale, own bids", multiplier=m, lo=lo, hi=hi, n=n))
        lose = bb[(bb["rank"] > 1)]
        print(f"losing bids: {len(lose):,}")

        print(); print("=" * 78); print("5. THE LOSING BID ON A CONTESTED TRACT"); print("=" * 78)
        med1 = t[t.n == 1].groupby("cell").lr.median()
        mm = t[t.n >= 2].copy(); mm["m1"] = mm.cell.map(med1)
        mm = mm.dropna(subset=["m1"])
        above = (np.log(mm.ratio2) > mm.m1).mean()
        print(f"contested tracts whose second-highest bid exceeds the median winning bid of single-bid tracts "
              f"in the same sale and depth band: {100 * above:.1f}% of {len(mm)}; median ratio "
              f"x{np.exp((np.log(mm.ratio2) - mm.m1).median()):.2f}")

        print(); print("=" * 78); print("6. A SECOND BIDDER AS A SECOND DRAW"); print("=" * 78)
        p2, o2 = order_stat_cells(t, 2); p3, o3 = order_stat_cells(t, 3)
        print(f"two bids: E[higher of two draws] / E[one draw] from single-bid tracts in the cell x{p2:.2f}; observed mean ratio "
              f"of two-bid tracts over single-bid tracts x{o2:.2f}")
        print(f"three or more: E[highest of three draws] / E[one draw] x{p3:.2f}; observed x{o3:.2f}")
        u_cell = second_draw_uplift(t)
        boot_cell = [second_draw_uplift(t, np.random.default_rng(SEED + i)) for i in range(BOOT)]
        u_pair = pair_uplift(t)
        n2 = (t.n == 2).sum()
        boot_pair = [pair_uplift(t, rng.integers(0, n2, n2)) for _ in range(BOOT)]
        print(f"one more bid on every single-bid tract, drawn from the single-bid tracts of the same sale and depth band: "
              f"+{100 * (u_cell - 1):.1f}% of their high bids [bootstrap 95% {100 * (np.quantile(boot_cell, .025) - 1):.1f}%, "
              f"{100 * (np.quantile(boot_cell, .975) - 1):.1f}%]")
        print(f"one more bid as far from the first as the two bids on a two-bid tract: +{100 * (u_pair - 1):.1f}% "
              f"[{100 * (np.quantile(boot_pair, .025) - 1):.1f}%, {100 * (np.quantile(boot_pair, .975) - 1):.1f}%] "
              f"({n2} two-bid tracts)")
        two = t[t.n == 2]
        print(f"  two-bid tracts: median high / mean of the two bids {(two.high / ((two.high + two.second) / 2)).median():.2f}; "
              f"median high / second {(two.high / two.second).median():.2f}")
        share_comp = np.log(p2) / np.log(o2)
        print(f"share of the observed log gap between two-bid and single-bid tracts that a second draw explains: {100 * share_comp:.0f}%")
        rows.append(dict(comparison="second draw from the cell", multiplier=u_cell, lo=np.quantile(boot_cell, .025),
                         hi=np.quantile(boot_cell, .975), n=int((t.n == 1).sum())))
        rows.append(dict(comparison="second draw at contested-tract dispersion", multiplier=u_pair, lo=np.quantile(boot_pair, .025),
                         hi=np.quantile(boot_pair, .975), n=int(n2)))
        pd.DataFrame(rows).to_csv(RES / "multipliers.csv", index=False)

        print(); print("=" * 78); print("7. DO BIDDERS KNOW THEY ARE ALONE?"); print("=" * 78)
        s1 = t[t.n == 1]
        print(f"single-bid tracts won at exactly the minimum bid: {(s1.ratio < 1.0005).sum()} of {len(s1):,} "
              f"({100 * (s1.ratio < 1.0005).mean():.1f}%); within 5% of it {100 * (s1.ratio <= 1.05).mean():.1f}%; "
              f"median ratio {s1.ratio.median():.2f}")
        ns = s1[~s1.standard_bid]
        print(f"  leaving out standard bids: {len(ns):,} tracts, exactly at the minimum {100 * (ns.ratio < 1.0005).mean():.1f}%, "
              f"median ratio {ns.ratio.median():.2f}")
        extra = (s1.high - s1.minimum_bid).sum()
        print(f"amount bid above the minimum on single-bid tracts: ${extra / 1e6:,.1f} million "
              f"({100 * extra / s1.high.sum():.1f}% of their high bids)")

        print(); print("=" * 78); print("8. ONE MORE BIDDER OR A HIGHER MINIMUM BID (single-bid tracts)"); print("=" * 78)
        base = s1.high.sum()
        print(f"single-bid tracts: {len(s1):,}, high bids ${base / 1e6:,.1f} million")
        rr = []
        for shallow, levels in ((False, (125, 150, 200, 300)), (True, (30, 40, 50))):
            for lev in levels:
                r = reserve_change(t, shallow, lev); rr.append(r)
                print(f"  {r['depth']} minimum ${lev}/acre: {100 * r['below_share']:.1f}% of {r['tracts']:,} tracts bid below it; "
                      f"if all raise to it {100 * r['all_pay']:+.1f}%, if none do {100 * r['none_pay']:+.1f}%, "
                      f"break-even share {100 * r['break_even_share']:.0f}%")
        rr = pd.DataFrame(rr); rr.to_csv(RES / "reserve_changes.csv", index=False)
        ups = {}
        for depth, sub in (("deep", t[~t.shallow]), ("shallow", t[t.shallow])):
            ups[depth] = {"second draw from the cell": second_draw_uplift(sub),
                          "second draw at contested-tract dispersion": pair_uplift(sub)}
            print(f"{depth}: one more bid on every single-bid tract, drawn from the cell +{100 * (ups[depth]['second draw from the cell'] - 1):.1f}%; "
                  f"at contested-tract dispersion +{100 * (ups[depth]['second draw at contested-tract dispersion'] - 1):.1f}% "
                  f"({int((sub.n == 2).sum())} two-bid tracts)")
        print("share of single-bid tracts that would need one more bidder to match a minimum's best case:")
        need_rows = []
        for r in rr.itertuples():
            u1, u2 = (ups[r.depth][k] - 1 for k in ("second draw from the cell", "second draw at contested-tract dispersion"))
            print(f"    {r.depth} ${r.new_per_acre:.0f}/acre (best case {100 * r.all_pay:+.1f}%, worst {100 * r.none_pay:+.1f}%): "
                  f"{100 * r.all_pay / u1:.0f}% (drawn from the cell), {100 * r.all_pay / u2:.0f}% (contested-tract dispersion)")
            need_rows.append(dict(depth=r.depth, new_per_acre=r.new_per_acre, best=r.all_pay, worst=r.none_pay,
                                  uplift_cell=u1, uplift_pair=u2))
        pd.DataFrame(need_rows).to_csv(RES / "bidder_or_reserve.csv", index=False)
        per_tract = (u_cell - 1) * base / len(s1)
        print(f"value of one more bid on an average single-bid tract: ${per_tract / 1e6:.2f} million (second draw from the cell); "
              f"${(u_pair - 1) * base / len(s1) / 1e6:.2f} million (contested-tract dispersion)")

        print(); print("=" * 78); print("9. ROBUSTNESS"); print("=" * 78)
        for name, sub in (("deep water", t[~t.shallow]), ("shallow water", t[t.shallow]),
                          ("August 2013 to March 2017", t[t.sale_date < "2017-08-01"]),
                          ("August 2017 to August 2026", t[t.sale_date >= "2017-08-01"]),
                          ("without standard bids", t[~t.standard_bid])):
            m, lo, hi, n = fe_multiplier(sub, "lr", ["multi"], ["cell"], "multi")
            p, o = order_stat_cells(sub, 2)
            print(f"{name}: cross-section x{m:.2f} [{lo:.2f}, {hi:.2f}]; second draw x{p:.2f} vs observed x{o:.2f}; "
                  f"two-bid tracts {int((sub.n == 2).sum())}")
        t.to_csv(RES / "tracts.csv", index=False)

    text = out.getvalue()
    (RES / "summary.txt").write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
