"""Every number in the long-horizon-stock-risk article.

    python fetch.py                            # inputs into data/, checked against the registry
    python horizon_numbers.py > output.txt     # the numbers; also writes results/*.csv for figures.py

The question: does a long holding period make stocks safe? A long horizon lowers how often stocks lose
after inflation, and how often they lose to bonds. The test is whether it also lowers how much they
lose when they do. Real stock and bond wealth over 1 to 30 years, every start year, in 23 OECD markets
from the 1950s to 2025 (share prices plus an assumed dividend yield) and in the United States from 1871
to 2022 (Shiller, with actual dividends). See horizon_core.py for the rules.
"""

from __future__ import annotations

import csv
import io
import math
import os
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

import horizon_core as C

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("HORIZON_DATA", HERE / "data"))
RES = HERE / "results"
HORIZONS = (1, 2, 3, 5, 10, 15, 20, 25, 30)
KEY = (1, 10, 20, 30)
YIELDS = (0.0, 0.02, 0.03, 0.04)
BASE = 0.03
SEED = 20261123
DRAWS = 2000
COMMON_FIRST, COMMON_LAST = 1955, 1995  # start years for which a 30-year window can end by 2025


def pct(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f}%"


def num(x: float, digits: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def worst(w) -> str:
    return "n/a" if w is None else f"{num(w.multiple)} ({C.NAMES[w.area]}, from {w.start})"


def load():
    raw = DATA / "oecd-share-prices-bond-yields-cpi" / "raw"
    share = C.read_sdmx((raw / "finmark_share.csv").read_text(encoding="utf-8"))
    irlt = C.read_sdmx((raw / "finmark_irlt.csv").read_text(encoding="utf-8"))
    cpi = C.read_sdmx((raw / "prices_cpi.csv").read_text(encoding="utf-8"))
    rows = C.read_shiller((DATA / "shiller-sp-composite-monthly" / "raw" / "data.csv").read_text(encoding="utf-8"))
    return share, irlt, cpi, rows


def panel(share, irlt, cpi, d):
    out = {}
    for a in C.PANEL:
        m = C.oecd_market(a, share, irlt, cpi, d)
        if m is not None:
            out[a] = m
    return out


def stock_windows(markets, h, starts=None):
    ws = []
    for m in markets.values():
        ws += C.windows(m.stocks, m.area, h, starts)
    return ws


def bond_windows(markets, h, starts=None):
    ws = []
    for m in markets.values():
        ws += C.relative_windows(m, h, starts)
    return ws


def line(s: C.Summary) -> str:
    return (
        f"  {s.horizon:>2} y  windows {s.n:>5}  markets {s.markets:>2}  loss {pct(s.p_loss):>6}"
        f" (equal-weight {pct(s.p_loss_equal):>6})  median loss when losing {pct(s.median_loss):>6}"
        f"  mean {pct(s.mean_loss):>6}  average shortfall {pct(s.expected_shortfall, 2):>6}"
        f"  p5 {num(s.p05)}  median {num(s.median)}  worst {worst(s.worst)}"
        f"  episodes {s.episodes:>3}  independent {s.independent:>4}"
    )


# ----------------------------------------------------------------------------------------------- shuffles


def shuffle_test(series: dict[str, list[float]], horizons, rng, draws=DRAWS):
    """History against the same years in random order, per market, pooled over markets.

    series: {area: one-year log returns in time order}. Returns {h: dict of history and shuffle stats}.
    """
    out = {}
    hist_blocks = {h: [] for h in horizons}
    perm_blocks = {h: [] for h in horizons}
    vr_hist = {h: [] for h in horizons}
    vr_perm = {h: [] for h in horizons}
    for _area, r in sorted(series.items()):
        r = np.asarray(r, dtype=float)
        n = len(r)
        lv = np.concatenate([[0.0], np.cumsum(r)])
        perm = rng.permuted(np.tile(r, (draws, 1)), axis=1)
        plv = np.concatenate([np.zeros((draws, 1)), np.cumsum(perm, axis=1)], axis=1)
        for h in horizons:
            if n < h:
                continue
            hist_blocks[h].append(lv[h:] - lv[:-h])
            perm_blocks[h].append(plv[:, h:] - plv[:, :-h])
            if n >= 3 * h:
                vr_hist[h].append(C.variance_ratio(list(r), h))
                vr_perm[h].append(np.array([C.variance_ratio(list(row), h) for row in perm[:400]]))
    for h in horizons:
        if not hist_blocks[h]:
            continue
        hist = np.concatenate(hist_blocks[h])
        perm = np.concatenate(perm_blocks[h], axis=1)
        h_loss = float(np.mean(hist < 0))
        h_size = float(np.median(1 - np.exp(hist[hist < 0]))) if np.any(hist < 0) else float("nan")
        p_loss = np.mean(perm < 0, axis=1)
        sizes = np.where(perm < 0, 1 - np.exp(perm), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p_size = np.nanmedian(sizes, axis=1)
        p_size = p_size[~np.isnan(p_size)]
        rec = {
            "h": h,
            "windows": hist.size,
            "hist_loss": h_loss,
            "perm_loss_median": float(np.median(p_loss)),
            "perm_loss_p05": float(np.quantile(p_loss, 0.05)),
            "perm_loss_p95": float(np.quantile(p_loss, 0.95)),
            "share_perm_at_or_below": float(np.mean(p_loss <= h_loss)),
            "hist_size": h_size,
            "perm_size_median": float(np.median(p_size)) if p_size.size else float("nan"),
            "perm_size_p05": float(np.quantile(p_size, 0.05)) if p_size.size else float("nan"),
            "perm_size_p95": float(np.quantile(p_size, 0.95)) if p_size.size else float("nan"),
        }
        if vr_hist[h]:
            rec["vr_markets"] = len(vr_hist[h])
            rec["vr_hist_mean"] = float(np.mean(vr_hist[h]))
            pm = np.mean(np.vstack(vr_perm[h]), axis=0)
            rec["vr_perm_median"] = float(np.median(pm))
            rec["vr_perm_p05"] = float(np.quantile(pm, 0.05))
            rec["vr_perm_p95"] = float(np.quantile(pm, 0.95))
            rec["vr_share_perm_at_or_below"] = float(np.mean(pm <= rec["vr_hist_mean"]))
        out[h] = rec
    return out


def shuffle_lines(res, label):
    print(f"\n{label}")
    for h, r in res.items():
        s = (
            f"  {h:>2} y  windows {r['windows']:>5}  loss: history {pct(r['hist_loss'])}, shuffled median"
            f" {pct(r['perm_loss_median'])} [5-95%: {pct(r['perm_loss_p05'])} to {pct(r['perm_loss_p95'])}],"
            f" shuffles at or below history {pct(r['share_perm_at_or_below'])}"
            f" | median loss when losing: history {pct(r['hist_size'])}, shuffled median"
            f" {pct(r['perm_size_median'])} [5-95%: {pct(r['perm_size_p05'])} to {pct(r['perm_size_p95'])}]"
        )
        if "vr_hist_mean" in r:
            s += (
                f" | variance ratio over {r['vr_markets']} markets: history {num(r['vr_hist_mean'], 2)},"
                f" shuffled median {num(r['vr_perm_median'], 2)} [5-95%: {num(r['vr_perm_p05'], 2)} to"
                f" {num(r['vr_perm_p95'], 2)}], shuffles at or below history {pct(r['vr_share_perm_at_or_below'])}"
            )
        print(s)


# ----------------------------------------------------------------------------------------------- main


def main() -> None:
    RES.mkdir(exist_ok=True)
    share, irlt, cpi, rows = load()
    us = C.shiller_market(rows)
    base = panel(share, irlt, cpi, BASE)
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------ data
    print("DATA")
    months = [r["month"] for r in rows]
    gaps = sum(
        1
        for a, b in zip(months, months[1:], strict=False)
        if (int(b[:4]) * 12 + int(b[5:])) - (int(a[:4]) * 12 + int(a[5:])) != 1
    )
    print(f"  Shiller: {months[0]} to {months[-1]}, {len(months)} months, {gaps} gaps;")
    print(f"    year ends {min(us.stocks)} to {max(us.stocks)}; bonds {min(us.bonds)} to {max(us.bonds)}")
    print(f"  OECD: {len(base)} markets with a share price index and consumer prices")
    for a, m in base.items():
        b = f"{min(m.bonds)}-{max(m.bonds)}" if m.bonds else "none"
        print(f"    {a} {C.NAMES[a]:<15} stocks {min(m.stocks)}-{max(m.stocks)}  bonds {b:<9}  CPI {m.cpi_source}")
    n30 = [a for a, m in base.items() if min(m.stocks) + 30 <= max(m.stocks)]
    print(f"  markets with at least one 30-year window: {len(n30)}")

    # ------------------------------------------------------------------ dividends
    print("\nDIVIDENDS (the United States, where Shiller's actual dividends can be compared)")
    for lo, hi in ((1871, 2022), (1957, 2022), (1957, 1989), (1990, 2022)):
        dy = C.shiller_dividend_yield(rows, lo, hi)
        print(f"  Shiller's average December dividend yield, {lo}-{hi}: {pct(dy, 2)}")
    for d in YIELDS:
        usd = panel(share, irlt, cpi, d)["USA"]
        for h in KEY:
            a = {w.start: w.multiple for w in C.windows(usd.stocks, "USA", h)}
            b = {w.start: w.multiple for w in C.windows(us.stocks, "USA", h)}
            common = sorted(set(a) & set(b))
            diff = [C.annualised(a[s], h) - C.annualised(b[s], h) for s in common]
            la = sum(a[s] < 1 for s in common) / len(common)
            lb = sum(b[s] < 1 for s in common) / len(common)
            print(
                f"  OECD US index + {pct(d, 0)} against Shiller with dividends, {h:>2} y, {len(common)} common starts:"
                f" median gap {pct(C.quantile(diff, 0.5), 2)} a year (OECD minus Shiller);"
                f" real loss {pct(la)} against {pct(lb)}"
            )

    # ------------------------------------------------------------------ the textbook case
    logs = [x for m in base.values() for x in C.log_returns(m.stocks)]
    mu = sum(logs) / len(logs)
    sd = (sum((x - mu) ** 2 for x in logs) / (len(logs) - 1)) ** 0.5
    print(f"\nTEXTBOOK CASE: independent yearly log returns with the panel's mean {pct(mu, 2)} and sd {pct(sd, 2)}")
    print(f"  ({len(logs)} market-years, real, share prices plus {pct(BASE, 0)})")
    for h in KEY + (5,):
        z = NormalDist(mu * h, sd * math.sqrt(h))
        p_loss = z.cdf(0.0)
        median_log = z.inv_cdf(p_loss / 2)
        print(
            f"  {h:>2} y: real loss {pct(p_loss)}, median loss when losing {pct(1 - math.exp(median_log))},"
            f" average shortfall {pct(p_loss * (1 - math.exp(median_log)), 2)} (approximate)"
        )

    # ------------------------------------------------------------------ United States, 1871-2022
    print("\nUNITED STATES, SHILLER, 1871-2022: real stock wealth")
    us_rows = []
    for h in HORIZONS:
        s = C.summarise(C.windows(us.stocks, "USA", h), h)
        print(line(s))
        us_rows.append({"horizon": h, "p_loss": s.p_loss, "median_loss": s.median_loss, "p05": s.p05})
    print("UNITED STATES, SHILLER: stocks against bonds")
    for h in HORIZONS:
        s = C.summarise(C.relative_windows(us, h), h)
        print(line(s))
        us_rows[HORIZONS.index(h)] |= {"p_lag": s.p_loss, "median_lag": s.median_loss}
    pd.DataFrame(us_rows).to_csv(RES / "us_horizons.csv", index=False)
    print("UNITED STATES, SHILLER: the same rule on two halves of the record (start years)")
    for lo, hi in ((1871, 1945), (1946, 2022)):
        starts = set(range(lo, hi + 1))
        parts = []
        for h in KEY:
            s = C.summarise(C.windows(us.stocks, "USA", h, starts), h)
            parts.append(f"{h} y {pct(s.p_loss)} (n {s.n}, worst {worst(s.worst)})")
        print(f"  starts {lo}-{hi}: real loss " + "; ".join(parts))

    # ------------------------------------------------------------------ OECD panel
    print(f"\nOECD, 23 MARKETS, DIVIDEND YIELD {pct(BASE, 0)}: real stock wealth")
    panel_rows = []
    for h in HORIZONS:
        s = C.summarise(stock_windows(base, h), h)
        print(line(s))
        panel_rows.append(
            {
                "horizon": h,
                "p_loss": s.p_loss,
                "p_loss_equal": s.p_loss_equal,
                "median_loss": s.median_loss,
                "mean_loss": s.mean_loss,
                "expected_shortfall": s.expected_shortfall,
                "p05": s.p05,
                "median": s.median,
                "episodes": s.episodes,
                "independent": s.independent,
            }
        )
    print(f"OECD, DIVIDEND YIELD {pct(BASE, 0)}: stocks against bonds")
    for h in HORIZONS:
        s = C.summarise(bond_windows(base, h), h)
        print(line(s))
        panel_rows[HORIZONS.index(h)] |= {"p_lag": s.p_loss, "median_lag": s.median_loss, "lag_episodes": s.episodes}
    pd.DataFrame(panel_rows).to_csv(RES / "panel_horizons.csv", index=False)

    print("\nOECD: THE DIVIDEND ASSUMPTION (real loss; median loss when losing; 5th percentile)")
    sens = []
    for d in YIELDS:
        mk = panel(share, irlt, cpi, d)
        parts = []
        for h in HORIZONS:
            s = C.summarise(stock_windows(mk, h), h)
            r = C.summarise(bond_windows(mk, h), h)
            sens.append({"yield": d, "horizon": h, "p_loss": s.p_loss, "median_loss": s.median_loss, "p_lag": r.p_loss})
            if h not in KEY:
                continue
            parts.append(
                f"{h} y {pct(s.p_loss)}; {pct(s.median_loss)}; p5 {num(s.p05)};"
                f" vs bonds {pct(r.p_loss)}, {pct(r.median_loss)}"
            )
        print(f"  {pct(d, 0)}: " + " | ".join(parts))
    pd.DataFrame(sens).to_csv(RES / "dividend_sensitivity.csv", index=False)

    print(f"\nOECD, {pct(BASE, 0)}: THE SAME START YEARS FOR EVERY HORIZON ({COMMON_FIRST}-{COMMON_LAST})")
    common = set(range(COMMON_FIRST, COMMON_LAST + 1))
    mk30 = {a: m for a, m in base.items() if any(s in common and s + 30 in m.stocks for s in m.stocks)}
    starts_by = {a: {s for s in m.stocks if s in common and s + 30 in m.stocks} for a, m in mk30.items()}
    same_rows = []
    for h in HORIZONS:
        ws = []
        rs = []
        for a, m in mk30.items():
            ws += C.windows(m.stocks, a, h, starts_by[a])
            rs += C.relative_windows(m, h, starts_by[a])
        s = C.summarise(ws, h)
        r = C.summarise(rs, h) if rs else None
        print(line(s))
        same_rows.append({"horizon": h, "p_loss": s.p_loss, "median_loss": s.median_loss, "n": s.n})
        if r is not None and h in KEY:
            print(
                f"     against bonds: windows {r.n}, lag {pct(r.p_loss)}, median lag when lagging {pct(r.median_loss)}"
            )
    pd.DataFrame(same_rows).to_csv(RES / "same_starts.csv", index=False)

    # ------------------------------------------------------------------ funnel
    print(f"\nOECD, {pct(BASE, 0)}: ANNUAL RATE AGAINST WEALTH (5th, 50th and 95th percentiles)")
    funnel = []
    for h in HORIZONS:
        ws = stock_windows(base, h)
        rates = [C.annualised(w.multiple, h) for w in ws]
        mult = [w.multiple for w in ws]
        q = [C.quantile(rates, p) for p in (0.05, 0.5, 0.95)]
        m = [C.quantile(mult, p) for p in (0.05, 0.5, 0.95)]
        print(
            f"  {h:>2} y  annual rate {pct(q[0])} / {pct(q[1])} / {pct(q[2])} (spread {pct(q[2] - q[0])})"
            f"  wealth {num(m[0])} / {num(m[1])} / {num(m[2])} (95th over 5th {num(m[2] / m[0], 2)}x)"
        )
        funnel.append({"horizon": h, "r05": q[0], "r50": q[1], "r95": q[2], "w05": m[0], "w50": m[1], "w95": m[2]})
    pd.DataFrame(funnel).to_csv(RES / "funnel.csv", index=False)

    # ------------------------------------------------------------------ shuffles
    print(f"\nSHUFFLE TEST: the same years in random order ({DRAWS} shuffles per market, seed {SEED})")
    stock_logs = {a: C.log_returns(m.stocks) for a, m in base.items()}
    sh_panel = shuffle_test(stock_logs, HORIZONS, rng)
    shuffle_lines(sh_panel, f"OECD, {pct(BASE, 0)}: real stock wealth")
    rel_logs = {}
    for a, m in base.items():
        ys = C.longest_run(set(m.stocks) & set(m.bonds))
        if len(ys) > 10:
            rel_logs[a] = [math.log((m.stocks[y] / m.stocks[y - 1]) / (m.bonds[y] / m.bonds[y - 1])) for y in ys[1:]]
    sh_rel = shuffle_test(rel_logs, HORIZONS, rng)
    shuffle_lines(sh_rel, f"OECD, {pct(BASE, 0)}: stocks against bonds")
    sh_us = shuffle_test({"USA": C.log_returns(us.stocks)}, HORIZONS, rng)
    shuffle_lines(sh_us, "UNITED STATES, SHILLER: real stock wealth")
    us_rel_years = C.longest_run(set(us.stocks) & set(us.bonds))
    us_rel = [math.log((us.stocks[y] / us.stocks[y - 1]) / (us.bonds[y] / us.bonds[y - 1])) for y in us_rel_years[1:]]
    sh_us_rel = shuffle_test({"USA": us_rel}, HORIZONS, rng)
    shuffle_lines(sh_us_rel, "UNITED STATES, SHILLER: stocks against bonds")
    rows_sh = []
    for label, res in (("panel", sh_panel), ("panel_bonds", sh_rel), ("us", sh_us), ("us_bonds", sh_us_rel)):
        for r in res.values():
            rows_sh.append({"series": label} | r)
    pd.DataFrame(rows_sh).to_csv(RES / "shuffles.csv", index=False)

    # ------------------------------------------------------------------ mix
    print(f"\nEQUAL-WEIGHT MIX OF THE MARKETS, {pct(BASE, 0)}: local real returns, rebalanced yearly, no currencies")
    mix = C.equal_weight_mix([m.stocks for m in base.values()])
    print(f"  year ends {min(mix)}-{max(mix)}")
    mix_rows = []
    for h in HORIZONS:
        s = C.summarise(C.windows(mix, "MIX", h), h)
        single = C.summarise(stock_windows(base, h, set(range(min(mix), max(mix) - h + 1))), h)
        print(
            f"  {h:>2} y  windows {s.n:>3}  loss {pct(s.p_loss)}  median loss when losing {pct(s.median_loss)}"
            f"  p5 {num(s.p05)}  worst {num(s.worst.multiple)} from {s.worst.start}"
            f" | single markets, same start years: loss {pct(single.p_loss)}, median loss when losing"
            f" {pct(single.median_loss)}, p5 {num(single.p05)}"
        )
        mix_rows.append({"horizon": h, "p_loss": s.p_loss, "median_loss": s.median_loss, "p05": s.p05})
    pd.DataFrame(mix_rows).to_csv(RES / "mix.csv", index=False)
    w20 = [(w.area, w.start, w.multiple) for w in stock_windows(base, 20)]
    w20 += [("MIX", w.start, w.multiple) for w in C.windows(mix, "MIX", 20)]
    w20 += [("USA_SHILLER", w.start, w.multiple) for w in C.windows(us.stocks, "USA", 20)]
    pd.DataFrame(w20, columns=["area", "start", "multiple"]).to_csv(RES / "windows20.csv", index=False)

    # ------------------------------------------------------------------ markets
    print(f"\nBY MARKET, {pct(BASE, 0)}: 20- and 30-year real stock wealth (worst at 0% and 4% in brackets)")
    zero = panel(share, irlt, cpi, 0.0)
    four = panel(share, irlt, cpi, 0.04)
    by_market = []
    for a, m in base.items():
        parts = []
        rec = {"area": a, "name": C.NAMES[a]}
        for h in (20, 30):
            ws = C.windows(m.stocks, a, h)
            if not ws:
                parts.append(f"{h} y none")
                continue
            s = C.summarise(ws, h)
            w0 = C.summarise(C.windows(zero[a].stocks, a, h), h).worst
            w4 = C.summarise(C.windows(four[a].stocks, a, h), h).worst
            parts.append(
                f"{h} y n {s.n:>2}, loss {pct(s.p_loss, 0):>4}, worst {num(s.worst.multiple, 2)} from {s.worst.start}"
                f" [{num(w0.multiple, 2)}, {num(w4.multiple, 2)}], median {num(s.median, 2)}"
            )
            rec |= {
                f"n{h}": s.n,
                f"loss{h}": s.p_loss,
                f"worst{h}": s.worst.multiple,
                f"worst{h}_start": s.worst.start,
                f"worst{h}_0": w0.multiple,
                f"worst{h}_4": w4.multiple,
                f"min{h}": min(w.multiple for w in ws),
                f"p25_{h}": C.quantile([w.multiple for w in ws], 0.25),
                f"median{h}": s.median,
                f"p75_{h}": C.quantile([w.multiple for w in ws], 0.75),
                f"max{h}": max(w.multiple for w in ws),
            }
        print(f"  {C.NAMES[a]:<15} " + " | ".join(parts))
        by_market.append(rec)
    us20 = C.summarise(C.windows(us.stocks, "USA", 20), 20)
    us30 = C.summarise(C.windows(us.stocks, "USA", 30), 30)
    print(
        f"  US (Shiller, 1871-2022): 20 y worst {num(us20.worst.multiple, 2)} from {us20.worst.start},"
        f" 30 y worst {num(us30.worst.multiple, 2)} from {us30.worst.start}"
    )
    pd.DataFrame(by_market).to_csv(RES / "by_market.csv", index=False)
    for h in (10, 20, 30):
        lost = sorted({w.area for w in stock_windows(base, h) if w.multiple < 1})
        print(f"  markets with at least one real loss over {h} years: {len(lost)} ({', '.join(lost)})")

    # ------------------------------------------------------------------ episodes
    print(f"\nLOSS EPISODES, {pct(BASE, 0)}: runs of consecutive start years with a real loss")
    ep_rows = []
    for h in (10, 20, 30):
        ws = stock_windows(base, h)
        eps = C.episodes(ws)
        by = {(w.area, w.start): w.multiple for w in ws}
        print(f"  {h} years: {len(eps)} episodes, {sum(e[2] - e[1] + 1 for e in eps)} start years")
        for a, lo, hi in eps:
            low = min(by[(a, s)] for s in range(lo, hi + 1))
            print(f"    {C.NAMES[a]:<15} {lo}-{hi}  worst {num(low, 2)}")
            ep_rows.append({"horizon": h, "area": a, "first": lo, "last": hi, "worst": low})
        rel = bond_windows(base, h)
        reps = C.episodes(rel)
        print(f"  {h} years against bonds: {len(reps)} episodes, {sum(e[2] - e[1] + 1 for e in reps)} start years")
    pd.DataFrame(ep_rows).to_csv(RES / "episodes.csv", index=False)
    loss_starts = defaultdict(int)
    for w in stock_windows(base, 20):
        if w.multiple < 1:
            loss_starts[(w.start // 10) * 10] += 1
    print("  20-year losses by decade of the start: " + ", ".join(f"{k}s {v}" for k, v in sorted(loss_starts.items())))

    # ------------------------------------------------------------------ start conditions
    print("\nSTART CONDITIONS: real share price against its average over the ten year-ends to the start")
    price_only = zero
    recs = []
    for a, m in base.items():
        p = price_only[a].stocks
        for h in (10, 20):
            for w in C.windows(m.stocks, a, h):
                past = [p.get(w.start - k) for k in range(10)]
                if any(v is None for v in past):
                    continue
                ratio = p[w.start] / (sum(past) / 10)
                recs.append({"area": a, "start": w.start, "h": h, "ratio": ratio, "mult": w.multiple})
    st = pd.DataFrame(recs)
    st.to_csv(RES / "start_conditions.csv", index=False)
    for h in (10, 20):
        x = st[st.h == h].copy()
        x["q"] = pd.qcut(x.ratio, 5, labels=False)
        rho = x[["ratio", "mult"]].rank().corr().iloc[0, 1]
        print(f"  {h} y: {len(x)} windows in {x.area.nunique()} markets, Spearman correlation {num(rho, 2)}")
        for q, g in x.groupby("q"):
            print(
                f"    quintile {q + 1}: ratio {num(g.ratio.min(), 2)}-{num(g.ratio.max(), 2)}, real loss"
                f" {pct((g.mult < 1).mean())}, median wealth {num(g.mult.median(), 2)}, worst {num(g.mult.min(), 2)}"
            )
    us_cape = {}
    text = (DATA / "shiller-sp-composite-monthly" / "raw" / "data.csv").read_text(encoding="utf-8")
    for r in csv.DictReader(io.StringIO(text)):
        if r["Date"][5:7] == "12" and float(r["PE10"]) > 0:
            us_cape[int(r["Date"][:4])] = float(r["PE10"])
    xs = [(us_cape[s], us.stocks[s + 20] / us.stocks[s]) for s in sorted(us_cape) if s + 20 in us.stocks]
    df = pd.DataFrame(xs, columns=["cape", "mult"])
    rho = df.rank().corr().iloc[0, 1]
    df["q"] = pd.qcut(df.cape, 5, labels=False)
    print(f"  US, Shiller's CAPE in December and the next 20 years: {len(df)} starts, Spearman {num(rho, 2)}")
    for q, g in df.groupby("q"):
        med = g.mult.median()
        print(
            f"    quintile {q + 1}: CAPE {num(g.cape.min(), 1)}-{num(g.cape.max(), 1)}, median wealth"
            f" {num(med, 2)} ({pct(C.annualised(med, 20))} a year), worst {num(g.mult.min(), 2)}"
        )


if __name__ == "__main__":
    main()
