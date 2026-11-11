"""Every number quoted in Part 3, printed in the order the article uses them.

Run: python realdata_numbers.py > output.txt   (after the fetch_*.py scripts have written data/)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import realdata as r


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def q3(s: pd.Series) -> str:
    return f"median {pct(s.median())} (10th percentile {pct(s.quantile(0.1))}, 90th {pct(s.quantile(0.9))})"


# ---------------------------------------------------------------------------
section("R0. The RKI case files")
melde = r.load_vintages("melde")
onset = r.load_vintages("onset")
snap = r.load_snapshots()
final = r.final_melde()
pubs = melde.V.index
daily = pubs[(pubs >= r.RKI_FIRST_DAILY) & (pubs <= r.RKI_LAST_DAILY)]
expected = pd.date_range(r.RKI_FIRST_DAILY, r.RKI_LAST_DAILY)
print(f"daily publications read: {len(daily)} of {len(expected)} days from {daily.min().date()} to {daily.max().date()}; "
      f"missing: {[d.date().isoformat() for d in expected.difference(daily)]}")
print(f"monthly snapshots after that: {int((snap.columns > r.RKI_LAST_DAILY).sum())}, the last on {snap.columns.max().date()}")
log = [l.split() for l in (r.DATA / "rki_fetch_log.txt").read_text().splitlines()[1:]]
sizes = np.array([int(x[-1]) for x in log])
print(f"files read: {len(log)}, {sizes.sum() / 1e9:.1f} GB compressed; smallest {sizes.min() / 1e6:.1f} MB, largest {sizes.max() / 1e6:.1f} MB")
newp = melde.new.loc[daily].sum(axis=1)
zero = newp[newp == 0]
print(f"daily publications with no new case at all: {len(zero)}, the first on {zero.index.min().date()}; "
      f"by weekday: " + ", ".join(f"{['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][k]} {v}" for k, v in
                                  zero.groupby(zero.index.dayofweek).size().items()))
after = newp.loc["2022-07-01":"2023-12-31"]
print(f"  from July 2022: Sunday publications with no new case {int(((after == 0) & (after.index.dayofweek == 6)).sum())} of "
      f"{int((after.index.dayofweek == 6).sum())}, Monday {int(((after == 0) & (after.index.dayofweek == 0)).sum())} of "
      f"{int((after.index.dayofweek == 0).sum())}")
win = final.loc["2021-04-01":"2023-12-30"]
print(f"cases with Meldedatum 1 April 2021 to 30 December 2023 in the last file: {int(win.sum()):,}")
print(f"largest day: {int(win.max()):,} on {win.idxmax().date()}")
onset_share = onset.at_age(r.MAX_AGE).loc["2021-04-12":"2023-08-31"].sum()
print(f"cases with a known onset date (onset counts at 120 days / Meldedatum counts at 120 days, "
      f"dates 12 Apr 2021 - 31 Aug 2023): {pct(onset_share / melde.at_age(r.MAX_AGE).loc['2021-04-12':'2023-08-31'].sum())}")

section("R1. How long the case counts took to arrive")
FIRST, LAST = "2021-04-12", "2023-08-31"
ages = [0, 1, 2, 3, 4, 7, 14, 28, 120]
ct = r.completeness_table(melde, final, ages, FIRST, LAST)
print(f"By reporting date (Meldedatum), share of the count in the last file, dates {FIRST} to {LAST}:")
for a in ages:
    s = ct[a].dropna()
    if len(s):
        print(f"  published {a:3d} days after: {q3(s)}; n = {len(s)}")
late = 1 - ct[120].dropna()
print(f"  arriving more than 120 days after the reporting date: median {pct(late.median(), 2)}, "
      f"mean {pct(late.mean(), 2)}; dates with more than 1%: {int((late > 0.01).sum())}, with less than -1%: {int((late < -0.01).sum())}")

d1 = ct[1].dropna()
by_wd = d1.groupby(d1.index.dayofweek).median()
names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
print("  share published the next day, by weekday of the reporting date: "
      + ", ".join(f"{names[k]} {pct(v)}" for k, v in by_wd.items()))
by_year = d1.groupby(d1.index.year).median()
print("  share published the next day, by year: " + ", ".join(f"{y} {pct(v)}" for y, v in by_year.items()))
d2 = ct[2].dropna()
print("  share published within two days, by year: "
      + ", ".join(f"{y} {pct(v)}" for y, v in d2.groupby(d2.index.year).median().items()))

print("\nBy onset date (cases with a known onset), share of the count published 120 days after:")
onset_final = onset.at_age(r.MAX_AGE)
oc = r.completeness_table(onset, onset_final, [1, 2, 3, 5, 7, 10, 14, 21, 28], FIRST, LAST)
for a in oc.columns:
    print(f"  published {a:3d} days after onset: {q3(oc[a].dropna())}")

# ---------------------------------------------------------------------------
section("R2. What the latest periods were missing: the 7-day count")
sd = r.seven_day_sums(melde, final)
sd = sd.loc[FIRST:LAST]
ratio = sd.published / sd.final
print(f"7-day count as published each day / final, publications {FIRST} to {LAST}: {q3(ratio)}")
print(f"  lowest {pct(ratio.min())} on {ratio.idxmin().date()}; highest {pct(ratio.max())} on {ratio.idxmax().date()}")
print(f"  days on which the published count was more than 5% short: {int((ratio < 0.95).sum())} of {len(ratio)}")
wk = pd.DataFrame({
    "published": sd.published / sd.published.shift(7) - 1,
    "final": sd.final / sd.final.shift(7) - 1,
}).dropna()
same = int((np.sign(wk.published) == np.sign(wk.final)).sum())
con_rki = r.rki_connection()
sql7 = con_rki.execute(r.SEVEN_DAY_SQL, {"last": r.RKI_LAST.date()}).df().set_index("publication").loc[FIRST:LAST]
print(f"  the same 7-day totals by SQL (calendar of dates, left joins): {len(sql7)} publications, equal to pandas: "
      f"{bool((sql7.published == sd.published).all() and (sql7.final == sd.final).all())}")
print(f"  week-over-week change as published (each week from its own day) against final: same sign on {same} of {len(wk)} days; "
      f"mean absolute difference {100 * (wk.published - wk.final).abs().mean():.1f} points")

live = []
for p in sd.index:
    this = pd.date_range(p - pd.Timedelta(days=7), p - pd.Timedelta(days=1))
    prev = this - pd.Timedelta(days=7)
    if prev[0] in melde.V.columns:
        live.append((p, melde.V.loc[p, this].fillna(0).sum() / melde.V.loc[p, prev].fillna(0).sum() - 1))
live = pd.Series(dict(live))
wk2 = pd.DataFrame({"live": live, "final": wk.final}).dropna()
print(f"  week-over-week change read from one day's file for both weeks: same sign as final on "
      f"{int((np.sign(wk2.live) == np.sign(wk2.final)).sum())} of {len(wk2)} days; mean error "
      f"{100 * (wk2.live - wk2.final).mean():+.1f} points; falling while the final rose: "
      f"{int(((wk2.live < 0) & (wk2.final > 0)).sum())} days")

print("\nThe 7-day count nowcast each day (dates p-7 .. p-1; publications 1 May 2021 - 31 Aug 2023), error against final:")
nc = r.nowcast_seven_day(melde, final, "2021-05-01", LAST)
for k, label in (("raw", "as published"), ("chain", "chain ladder, 28-day window"),
                 ("weekday", "same-weekday factors, last 4 weeks")):
    e = nc[k]
    print(f"  {label:36s} mean {e.mean() * 100:+.1f}%, mean absolute {pct(e.abs().mean())}, "
          f"days more than 5% off {int((e.abs() > 0.05).sum())} of {len(e)}, Monday publications mean absolute "
          f"{pct(e[e.index.dayofweek == 0].abs().mean())}")

# ---------------------------------------------------------------------------
section("R3. Restatements in the case files")
rem = melde.removed.loc[daily].sum().sum()
add = melde.new.loc[daily].sum().sum()
print(f"removed rows (NeuerFall = -1) summed over the daily publications, dates within 120 days: {int(-rem):,} cases; "
      f"new (NeuerFall = 1): {int(add):,}")
daily_rem = -melde.removed.loc[daily].sum(axis=1)
print(f"  largest one-day removal: {int(daily_rem.max()):,} cases in the publication of {daily_rem.idxmax().date()}")
print(f"  publications with more than 1,000 cases removed: {int((daily_rem > 1000).sum())}")
rec = r.flag_reconciliation(melde)
print(f"  consecutive daily files compared: {len(rec)}; pairs where, for at least one of the 100 dates before, the change "
      f"between the files differs from the flagged change: {int((rec.abs_mismatch > 0).sum())}")
print(f"  absolute difference between flagged and actual change: median {rec.abs_mismatch.median():.0f} cases a day, "
      f"total {rec.abs_mismatch.sum():,.0f} against a total absolute change of {rec.abs_change.sum():,.0f} "
      f"({pct(rec.abs_mismatch.sum() / rec.abs_change.sum(), 2)}); largest {rec.abs_mismatch.max():,.0f} on {rec.abs_mismatch.idxmax().date()}")
d0, p1, p2 = pd.Timestamp("2021-03-16"), pd.Timestamp("2021-04-03"), pd.Timestamp("2021-04-04")
print(f"  example: reporting date {d0.date()}, count in the file of {p1.date()} {int(melde.V.at[p1, d0]):,}, in the file of "
      f"{p2.date()} {int(melde.V.at[p2, d0]):,}; flagged in the second file: new {int(melde.new.at[p2, d0])}, "
      f"removed {int(-melde.removed.at[p2, d0])}")
a120 = melde.at_age(r.MAX_AGE).loc[FIRST:LAST]
chg = final.reindex(a120.index) / a120 - 1
print(f"  change between 120 days after the reporting date and the last file: total "
      f"{(final.reindex(a120.index).sum() / a120.sum() - 1) * 100:+.3f}%; dates changed by more than 1%: "
      f"{int((chg.abs() > 0.01).sum())} of {len(chg)}")
cols = [c for c in snap.columns if c >= pd.Timestamp("2024-01-01")]
base = snap[cols[0]].loc["2021-04-01":"2023-11-30"].sum()
print(f"  cases with Meldedatum April 2021 - November 2023 in the file of {cols[0].date()}: {int(base):,}; "
      f"in the file of {cols[-1].date()}: {int(snap[cols[-1]].loc['2021-04-01':'2023-11-30'].sum()):,}")

# ---------------------------------------------------------------------------
section("R4. States")
st = pd.read_csv(r.DATA / "rki_state.csv.gz", parse_dates=["publication", "date"])
st["age"] = (st.publication - st.date).dt.days
pivot = st.pivot_table(index=["state", "date"], columns="age", values="cases", aggfunc="sum")
share1 = (pivot[1] / pivot[28]).dropna()
share1 = share1[(share1.index.get_level_values("date") >= FIRST) & (share1.index.get_level_values("date") <= LAST)]
by_state = share1.groupby(level="state").median().sort_values()
print("Share of the 28-day count published the next day, median over dates, by state:")
for s, v in by_state.items():
    print(f"  {r.STATES[s]:24s} {pct(v)}")

# ---------------------------------------------------------------------------
section("R5. A nowcast on the onset-date counts, delays as they really changed")
dates = pd.date_range("2021-04-12", "2023-08-31")
C = r.cumulative_triangle(onset, dates, 60)
target = C[:, 60]
res = {w: [] for w in (7, 28)}
raw = []
for T in range(90, len(dates)):
    Cm = C[:T + 1].copy()
    for j in range(max(0, T - 60), T + 1):
        Cm[j, T - j + 1:] = np.nan
    est = {w: r.chain_ladder_daily(Cm, T, w, 60) for w in res}
    for a in (3, 7):
        t = T - a
        if np.isnan(target[t]) or target[t] <= 0:
            continue
        for w in res:
            res[w].append((dates[T], a, est[w][t] / target[t] - 1))
        raw.append((dates[T], a, C[t, a] / target[t] - 1))
raw = pd.DataFrame(raw, columns=["as_of", "age", "err"])
comp7 = pd.Series(C[:, 7] / target, index=dates).loc[:"2023-07-31"]
monthly = comp7.groupby(comp7.index.to_period("M")).median()
print(f"  onset counts 7 days after onset, monthly median share of the 60-day count: from {pct(monthly.min())} "
      f"({monthly.idxmin()}) to {pct(monthly.max())} ({monthly.idxmax()})")
for a in (3, 7):
    rr = raw[raw.age == a].err
    line = f"  onset {a} days before: raw count error mean {pct(rr.mean())}, mean absolute {pct(rr.abs().mean())}"
    for w, v in res.items():
        e = pd.DataFrame(v, columns=["as_of", "age", "err"])
        e = e[e.age == a].err
        line += f"; chain ladder ({w}-day window) mean {pct(e.mean())}, mean absolute {pct(e.abs().mean())}, " \
                f"worst {pct(e.abs().max())}"
    print(line)
e28 = pd.DataFrame(res[28], columns=["as_of", "age", "err"])
e28 = e28[e28.age == 7].set_index("as_of").err
worst = e28.abs().groupby(e28.index.to_period("M")).mean().sort_values(ascending=False).head(3)
print("  months with the largest mean absolute error (28-day window, 7 days after onset): "
      + ", ".join(f"{m} {pct(v)}" for m, v in worst.items()))

# ---------------------------------------------------------------------------
section("H1. The hospitalisation triangle")
hub_log = (r.DATA / "hub_fetch_log.txt").read_text().splitlines()
print(f"nowcast files read: {sum(1 for l in hub_log if l.startswith('data-processed/'))}")
tri = r.load_hub_triangle()
i0, i1 = tri.index(r.HUB_FIRST - pd.Timedelta(days=28)), tri.index(r.HUB_LAST)
sel = slice(i0, i1 + 1)
print(f"dates {tri.dates[0].date()} to {tri.dates[-1].date()}; evaluation dates {tri.dates[i0].date()} to {tri.dates[i1].date()}")
for k in (0, 1, 3, 7, 14, 28, 56, 80):
    c = tri.cum[sel, k] / tri.final[sel]
    print(f"  share of the latest count known {k:2d} days after the reporting date: median {pct(np.median(c))} "
          f"(10th {pct(np.quantile(c, 0.1))}, 90th {pct(np.quantile(c, 0.9))})")
late = 1 - tri.at80[sel] / tri.final[sel]
print(f"  share arriving more than 80 days after: median {pct(np.median(late))}")
neg = (tri.inc[:, 1:] < 0)
print(f"  negative daily changes (corrections) in the national triangle: {int(neg.sum())} cells, "
      f"{int(-tri.inc[:, 1:][neg].sum()):,} cases")

seven = []
for t in range(i0, i1 + 1):
    seven.append((tri.dates[t], tri.known7(t, t) / tri.final7(t), tri.known7(t, t + 7) / tri.final7(t)))
seven = pd.DataFrame(seven, columns=["date", "published", "week_later"]).set_index("date")
print(f"7-day hospitalisation count as first published / latest, {len(seven)} days: {q3(seven.published)}; a week later: {q3(seven.week_later)}")
hub_days = seven.loc[r.HUB_FIRST:r.HUB_LAST]
print(f"  the {len(hub_days)} days of the hub's evaluation period: first published {q3(hub_days.published)}; "
      f"a week later {q3(hub_days.week_later)}; highest first-published share {pct(hub_days.published.max())}")
wow = []
for t in range(i0 + 7, i1 + 1):
    pub = tri.known7(t, t) / tri.known7(t - 7, t - 7) - 1
    live = tri.known7(t, t) / tri.known7(t - 7, t) - 1
    fin = tri.final7(t) / tri.final7(t - 7) - 1
    wow.append((pub, live, fin))
wow = pd.DataFrame(wow, columns=["published", "live", "final"])
print(f"  week-over-week change, {len(wow)} days: as published (each week on its own day) same sign as final on "
      f"{int((np.sign(wow.published) == np.sign(wow.final)).sum())}; both weeks from one day's data: "
      f"{int((np.sign(wow.live) == np.sign(wow.final)).sum())}, mean error {100 * (wow.live - wow.final).mean():+.1f} points, "
      f"falling while the final rose {int(((wow.live < 0) & (wow.final > 0)).sum())}")

section("H2. Ten nowcasts against the truth, 22 Nov 2021 - 29 Apr 2022")
q = r.evaluate_hub(tri, r.load_hub_nowcasts())
cl = pd.concat([r.evaluate_hub(tri, r.chain_ladder_hub(tri, w)) for w in (28, 56)])
allm = pd.concat([q, cl], ignore_index=True)
known = q.drop_duplicates(["forecast_date", "target_end_date"])[["forecast_date", "target_end_date", "h", "known", "truth", "truth80"]].copy()
known["point"] = known["known"]
known["model"] = "value known on the day"
allm = pd.concat([allm, known], ignore_index=True)
for H in (0, -3, -7, -14, -28):
    print(f"horizon {H} days:")
    rows = []
    for m, g in allm[allm.h == H].groupby("model"):
        s, s80 = r.score(g, "truth"), r.score(g, "truth80")
        rows.append((m, s, s80))
    for m, s, s80 in sorted(rows, key=lambda x: x[1]["mape"]):
        cov = f", 95% interval covered the latest value {pct(s['cov95'], 0)}" if "cov95" in s and not np.isnan(s["cov95"]) else ""
        print(f"  {m:32s} n={int(s['n']):3d} against latest: bias {s['bias'] * 100:+.1f}%, MAE {pct(s['mape'])}{cov}; "
              f"against 80 days: bias {s80['bias'] * 100:+.1f}%, MAE {pct(s80['mape'])}")

# ---------------------------------------------------------------------------
section("G. GDP vintages, quarter-on-quarter growth (100 x log difference)")
print(f"quarters {r.GDP_FIRST_Q.date()} to {r.GDP_LAST_Q.date()}, vintages 2002 Q4 to 2024 Q4")
con = r.gdp_connection()
for cid in r.GDP_IDS:
    s = r.revision_stats(cid)
    print(f"{r.GDP_NAMES[cid]} ({s['quarters']} quarters):")
    print(f"  mean absolute revision, first release to latest: {s['mean_abs_revision']:.2f} points (mean {s['mean_revision']:+.2f}); "
          f"annualised about {4 * s['mean_abs_revision']:.1f} points")
    print("  mean absolute revision after 1, 4, 8, 12 vintages: " + ", ".join(f"{v:.2f}" for v in s["after"].values())
          + "; still to come after 4 and 12: " + ", ".join(f"{s['remaining'][k]:.2f}" for k in (4, 12)))
    print(f"  sign of growth changed: {s['sign_changes']} quarters; negative quarters first release {s['negative_first']}, "
          f"latest {s['negative_latest']}")
    t, v = s["largest"]
    rel = r.releases(r.gdp_growth(cid))[t]
    print(f"  largest revision: {t.year} Q{(t.month - 1) // 3 + 1}, first {rel[0]:+.2f}, latest {rel[-1]:+.2f} ({v:.2f} points)")
    for tq in ("2008-10-01", "2020-04-01"):
        rel = r.releases(r.gdp_growth(cid))[pd.Timestamp(tq)]
        print(f"  {tq[:4]} Q{(int(tq[5:7]) - 1) // 3 + 1}: first {rel[0]:+.2f}, latest {rel[-1]:+.2f}")
    rf, rl = r.recession_quarters(s["first"]), r.recession_quarters(s["latest"])
    print(f"  quarters in two-quarter declines: first release {len(rf)}, latest {len(rl)}; only in first {len(rf - rl)}, only in latest {len(rl - rf)}")
    jumps = ", ".join(f"{d.year} Q{(d.month - 1) // 3 + 1} {100 * x:+.1f}%" for d, x in s["level_jumps"])
    a, b = s["level_2000q1_first_last"]
    print(f"  level of 2000 Q1 changed by more than 2% between consecutive vintages at: {jumps}; "
          f"first vintage {a:,.0f}, latest {b:,.0f} ({(b / a - 1) * 100:+.0f}%)")
us = con.execute(r.AS_OF_SQL, {"id": "US", "as_of": "2009-01-01"}).df().set_index("time")["value"]
print(f"US real GDP for 2008 Q3 as known on 1 Jan 2009 (as-of SQL): {us.loc[pd.Timestamp('2008-07-01')]:,.0f}")
fr = con.execute(r.FIRST_RELEASE_SQL, {"id": "US"}).df().set_index("time")
first_py = {t: r.gdp_levels("US").loc[t].dropna().iloc[0] for t in fr.index}
print(f"first-release SQL against the pandas version, US levels: {len(fr)} quarters, equal: "
      f"{bool(np.allclose(fr.first_value.to_numpy(), np.array([first_py[t] for t in fr.index])))}")
