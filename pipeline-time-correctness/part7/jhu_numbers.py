"""Every number in Part 7.

Run: python jhu_numbers.py > output.txt   (a few seconds; reads data/ and the RKI extracts of Parts 3 and 6)
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import jhu as J

RESULTS = J.HERE / "results"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def rate(s: pd.Series) -> str:
    return f"{pct(s.mean())} ({int(s.sum())} of {len(s)})"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    hol = J.holidays()
    us = J.us_states()
    c = J.countries()

    section("P0. The data")
    log = (J.DATA / "jhu_fetch_log.txt").read_text().splitlines()
    print(log[0])
    files = us.drop_duplicates("date")
    landed = pd.to_datetime(files.landed_utc).dt.tz_localize(None)
    hours = (landed - files.date).dt.total_seconds() / 3600
    print(f"US daily reports: {len(files)} files, {files.date.min().date()} to {files.date.max().date()}; "
          f"{us.state.nunique()} states and DC; {len(us):,} state rows")
    print(f"  landing time (commit), hours after the start of the file's date in UTC: 5th percentile "
          f"{hours.quantile(.05):.1f}, median {hours.median():.1f}, 95th {hours.quantile(.95):.1f}, max {hours.max():.1f}")
    print(f"  versions per file: median {files.n_versions.median():.0f}, max {files.n_versions.max()}; "
          f"files never changed after landing: {(files.n_versions == 1).sum()}")
    changed = us["first"].ne(us.final)
    print(f"  state rows whose count in the final version differs from the first: {rate(changed)}")
    print(f"global time series: {c.date.nunique()} dates, {c.date.min().date()} to {c.date.max().date()}, "
          f"{c.country.nunique()} countries; {len(c):,} rows")
    print(log[3])

    section("P1. The RKI files after days off for the institute only (file of the next day)")
    inc = J.rki_increments()
    days = [("2022-03-08", "International Women's Day, Berlin"), ("2023-03-08", "International Women's Day, Berlin"),
            ("2024-03-08", "International Women's Day, Berlin"), ("2025-05-08", "one-off holiday, Berlin"),
            ("2024-12-24", "Christmas Eve"), ("2025-12-24", "Christmas Eve"),
            ("2024-12-31", "New Year's Eve"), ("2025-12-31", "New Year's Eve")]
    for day, what in days:
        d = pd.Timestamp(day)
        nxt = d + pd.Timedelta(days=1)
        same = [nxt + pd.Timedelta(days=7 * k) for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
        ref = inc.reindex(same).dropna()
        print(f"  {d.strftime('%a %d %b %Y')} ({what}): file of {nxt.strftime('%a %d %b')} brought "
              f"{int(inc.get(nxt, np.nan)):,} new cases; same weekday, four weeks either side: median "
              f"{ref.median():,.0f}, smallest {ref.min():,.0f}")
    print("  weekday Berlin-only holidays in 2021-2023 daily files: 8 March 2022 (Tue) and 8 March 2023 (Wed);"
          " Christmas Eve and New Year's Eve fell on a weekend in 2022 and 2023")

    section("P2. US states: unchanged files")
    u = us[us.flat.notna()]
    print(f"state rows with a file of the day before at landing: {len(u):,}; unchanged {rate(u.flat)}")
    print("  by year: " + ", ".join(f"{y} {pct(g.flat.mean())}" for y, g in u.groupby(u.date.dt.year)))
    print("  by weekday: " + ", ".join(f"{n} {pct(u[u.wd == i].flat.mean())}"
                                       for i, n in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])))
    ud = J.us_day_types(us, hol)
    dr = J.us_daily_regime(ud)
    wk = ud[(ud.wd < 5) & ud.flat.notna()]
    print(f"weekday state rows: {len(wk):,}; in the daily regime (at most {pct(J.THRESHOLD, 0)} of ordinary weekdays "
          f"unchanged within {J.HALF_WINDOW} days): {len(dr):,} ({pct(len(dr) / len(wk))}), {dr.state.nunique()} states")
    print("  daily-regime state rows by year: " + ", ".join(f"{y} {n:,}" for y, n in dr.groupby(dr.date.dt.year).size().items()))

    section("P3. US states, daily regime: unchanged files by kind of weekday")
    order = ["ordinary", "federal and state holiday", "state holiday only", "federal holiday, state open",
             "Christmas Eve or New Year's Eve", "day after a holiday"]
    r = J.rates(dr, "day").reindex(order)
    for k, row in r.iterrows():
        print(f"  {k:34s} {pct(row.rate)} ({int(row['sum'])} of {int(row['size'])})")
    r.to_csv(RESULTS / "us_day_types.csv")
    so = dr[dr.day == "state holiday only"]
    print("  state holidays only, by name (at least 5 state-days): " + "; ".join(
        f"{n} {int(g.flat.sum())}/{len(g)}" for n, g in sorted(so.groupby("hol_name"), key=lambda kv: -len(kv[1]))
        if len(g) >= 5))
    fo = dr[dr.day == "federal holiday, state open"]
    print("  federal holidays the state did not observe, by name: " + "; ".join(
        f"{n} {int(g.flat.sum())}/{len(g)}" for n, g in fo.groupby("hol_name")))
    fs = dr[dr.day == "federal and state holiday"].copy()
    fs["fed_name"] = [hol["US"].get(x).replace(" (observed)", "") for x in fs.date.dt.date]
    print("  federal and state holidays, by name: " + "; ".join(
        f"{n} {pct(g.flat.mean(), 0)}" for n, g in fs.groupby("fed_name")))

    section("P4. The same date, states that observed it against states that did not (daily regime)")
    w = J.within_date(dr)
    a = w.groupby("holiday")[["obs_flat", "obs_n", "non_flat", "non_n"]].sum()
    a = a[(a.obs_n >= 10) & (a.non_n >= 10)].sort_values("obs_n", ascending=False)
    for k, row in a.iterrows():
        print(f"  {k:28s} observed: {pct(row.obs_flat / row.obs_n)} ({row.obs_flat} of {row.obs_n}); "
              f"not observed: {pct(row.non_flat / row.non_n)} ({row.non_flat} of {row.non_n})")
    s = a.sum()
    print(f"  pooled: observed {pct(s.obs_flat / s.obs_n)} ({s.obs_flat} of {s.obs_n}); "
          f"not observed {pct(s.non_flat / s.non_n)} ({s.non_flat} of {s.non_n})")
    a.to_csv(RESULTS / "us_within_date.csv")
    eve = dr[dr.day == "Christmas Eve or New Year's Eve"]
    print("  Christmas Eve and New Year's Eve on a weekday where not a legal holiday: " + "; ".join(
        f"{d.date()} {int(g.flat.sum())}/{len(g)}" for d, g in eve.groupby("date")))

    section("P5. What a federal holiday calendar would miss, and what it would flag for nothing")
    hol_days = dr[dr.day.isin(["federal and state holiday", "state holiday only", "federal holiday, state open"])]
    gaps = hol_days[hol_days.flat == 1]
    missed = (gaps.day == "state holiday only").sum()
    print(f"unchanged weekday files on a federal or state holiday: {len(gaps)}; on state holidays that are not "
          f"federal: {missed} ({pct(missed / len(gaps))})")
    fo = dr[dr.day == "federal holiday, state open"]
    print(f"federal holidays a state did not observe: {len(fo)} state-days, {int((fo.flat == 0).sum())} of them "
          f"with new data ({pct((fo.flat == 0).mean())})")
    flagged = dr[dr.fed]
    print(f"all state-days a federal calendar flags: {len(flagged)}, unchanged on {rate(flagged.flat)}")
    statecal = dr[dr.state_hol]
    print(f"all state-days the state's own calendar flags: {len(statecal)}, unchanged on {rate(statecal.flat)}")

    section("P6. Robustness of P3 to the definition of the daily regime")
    for thr in (0.05, 0.10, 0.20):
        for hw in (14, 28, 42):
            x = J.us_daily_regime(ud, thr, hw)
            rr = J.rates(x, "day").rate
            print(f"  at most {pct(thr, 0):>3s} within {hw:2d} days: n {len(x):6,}; ordinary {pct(rr['ordinary'])}, "
                  f"federal and state {pct(rr['federal and state holiday'])}, state only "
                  f"{pct(rr['state holiday only'])}, federal only {pct(rr['federal holiday, state open'])}")

    section("P7. The aggregator: countries outside the US")
    cd = J.country_day_types(c, hol)
    cw = cd[(cd.wd < 5) & cd.flat.notna()]
    cr = cw[cw.base <= J.THRESHOLD]
    print(f"countries with a holiday calendar: {cd.country.nunique()}; weekday country rows {len(cw):,}; "
          f"in the daily regime {len(cr):,}, {cr.country.nunique()} countries")
    order = ["ordinary", "own public holiday", "day after own holiday", "US federal holiday only", "24-31 Dec or 2 Jan"]
    rc = J.rates(cr, "day").reindex(order)
    for k, row in rc.iterrows():
        print(f"  {k:26s} {pct(row.rate)} ({int(row['sum'])} of {int(row['size'])})")
    rc.to_csv(RESULTS / "country_day_types.csv")
    usd = cr[cr.day == "US federal holiday only"].copy()
    usd["us_name"] = usd.us_name.str.replace(" (observed)", "", regex=False)
    print("  US federal holidays, by name: " + "; ".join(
        f"{n} {pct(g.flat.mean())} of {len(g)}" for n, g in usd.groupby("us_name")))
    de = cd[(cd.country == "Germany") & (cd.flat == 1) & (cd.wd < 5)]
    print("  Germany, weekday dates with an unchanged count: " + ", ".join(
        f"{d.strftime('%a %d %b %Y')}" + (" (day after a holiday)" if p else "")
        for d, p in zip(de.date, de.own_prev)))

    section("P8. Timestamps")
    lu = us[(us.flat == 1) & us.last_update.notna() & us.last_update_prev.notna()]
    adv = lu.last_update.ne(lu.last_update_prev)
    print(f"state rows unchanged from the day before: {len(lu):,}; with a new Last_Update all the same: {rate(adv)}")
    ch = us[(us.flat == 0) & us.last_update.notna() & us.last_update_prev.notna()]
    print(f"state rows that changed: {len(ch):,}; with a new Last_Update: {rate(ch.last_update.ne(ch.last_update_prev))}")
    late = hours > 36
    print(f"files landing more than 36 hours after the start of their date: {int(late.sum())} of {len(files)}")
    f2 = us.merge(files.assign(hours=hours.values)[["date", "hours"]], on="date")
    print(f"unchanged state rows whose file landed within 36 hours: {rate(f2[f2.flat == 1].hours <= 36)}")

    section("P9. Unchanged files and the largest errors: US states against their final version")
    e = J.us_errors(ud)
    print(f"state-days with a final 7-day count of at least {J.MIN_WEEKLY}: {len(e):,}; unchanged {pct(e.flat.mean())}")
    print(f"  absolute relative error of the 7-day count as first published: median {pct(e.rel.abs().median(), 2)}, "
          f"90th percentile {pct(e.rel.abs().quantile(.9))}, 99th {pct(e.rel.abs().quantile(.99))}; "
          f"revised by more than 0.1%: {pct((e.rel.abs() > 0.001).mean())}")
    for q in (0.99, 0.95):
        thr, ww = J.worst(e, q)
        print(f"  worst {pct(1 - q, 0)} (at least {pct(thr)} off): {len(ww)}; unchanged {pct(ww.flat.mean())}; "
              f"too high {pct((ww.rel > 0).mean(), 0)}")
    fl = e[e.flat == 1]
    print(f"  unchanged files still unchanged in the final version: {rate(fl.still_flat)}")
    thr1, w1 = J.worst(e, 0.99)
    thr5, w5 = J.worst(e, 0.95)
    for label, part in (("filled in later", fl[~fl.still_flat]), ("still unchanged", fl[fl.still_flat])):
        print(f"  unchanged files {label}: {len(part):,}; among the worst 1% {pct(part.index.isin(w1.index).mean())}, "
              f"among the worst 5% {pct(part.index.isin(w5.index).mean())}; mean absolute error "
              f"{pct(part.rel.abs().mean(), 2)}")
    filled = fl[~fl.still_flat]
    print(f"  of those filled in later, published a 7-day count of zero: {rate(filled.pub7 == 0)}; "
          f"published too low: {rate(filled.rel < 0)}")
    other = e[e.flat == 0]
    print(f"  files that changed: {len(other):,}; among the worst 1% {pct(other.index.isin(w1.index).mean())}, "
          f"among the worst 5% {pct(other.index.isin(w5.index).mean())}; mean absolute error {pct(other.rel.abs().mean(), 2)}")
    e["base"] = J.regime_base(e, "state", e.day.eq("ordinary"))
    ed = e[e.base <= J.THRESHOLD]
    print(f"daily regime (all days of the week): {len(ed):,} state-days; unchanged {pct(ed.flat.mean())}")
    for q in (0.99, 0.95):
        thr, ww = J.worst(ed, q)
        print(f"  worst {pct(1 - q, 0)} (at least {pct(thr)} off): {len(ww)}; unchanged {pct(ww.flat.mean())}")
    print(f"  mean absolute error: unchanged mornings {pct(ed[ed.flat == 1].rel.abs().mean(), 2)}, "
          f"others {pct(ed[ed.flat == 0].rel.abs().mean(), 2)}")
    print(f"  unchanged files still unchanged in the final version: {rate(ed[ed.flat == 1].still_flat)}")

    section("P10. Germany as Johns Hopkins published it, against the RKI's count by reporting date")
    inc = J.rki_increments()
    de1 = J.germany(c, lag=1)
    jinc = (de1["first"] - de1.prev_in_first).astype(float)
    for k in (0, 1, 2):
        x = pd.concat([jinc, inc.shift(-k)], axis=1, sort=True).dropna()
        x = x[(x.index >= "2021-04-02") & (x.index <= "2023-03-09")]
        print(f"  daily count for date d against the RKI file of d+{k}: correlation {x.corr().iloc[0, 1]:.3f}, "
              f"identical on {pct((x.iloc[:, 0] == x.iloc[:, 1]).mean())} of {len(x)} days")
    span = lambda d: d[(d.index >= "2021-04-10") & (d.index <= "2023-03-09")].dropna(subset=["rel"])
    for lag in (1, 0):
        g = span(J.germany(c, lag=lag))
        print(f"RKI reporting dates ending on d-{lag}: {len(g)} mornings, {g.index.min().date()} to "
              f"{g.index.max().date()}; unchanged {rate(g.flat)}")
        print(f"  absolute relative error of the 7-day count as first published: mean {pct(g.rel.abs().mean(), 2)}, "
              f"median {pct(g.rel.abs().median(), 2)}; unchanged mornings {pct(g[g.flat].rel.abs().mean(), 2)}, "
              f"others {pct(g[~g.flat].rel.abs().mean(), 2)}")
        for q in (0.95, 0.90):
            thr = g.rel.abs().quantile(q)
            ww = g[g.rel.abs() >= thr]
            print(f"  worst {pct(1 - q, 0)} (at least {pct(thr)} off): {len(ww)}; unchanged {pct(ww.flat.mean())}")
        if lag == 1:
            print("  by weekday: " + ", ".join(
                f"{n} {pct(g[g.index.dayofweek == i].rel.abs().mean())}"
                for i, n in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])))
            top = g.reindex(g.rel.abs().sort_values(ascending=False).index[:10])
            print("  the ten largest errors: " + "; ".join(
                f"{d.strftime('%a %d %b %Y')} {r.rel:+.1%}{' unchanged' if r.flat else ''}" for d, r in top.iterrows()))
            g[["first", "pub7", "truth7", "rel", "flat"]].to_csv(RESULTS / "germany.csv")
    g = span(de1)
    differ = g.final.ne(g["first"])
    print(f"Johns Hopkins' final version of the German series: differs from the first published on {rate(differ)}")
    lvl = (g.final / g["first"] - 1)
    print(f"  final cumulative count against the first published: from {pct(lvl.min(), 2)} to {pct(lvl.max(), 2)}")
    sh = g[g.shifted]
    print(f"  equal to the count first published for the day before: {len(sh)} dates, {sh.index.min().date()} to "
          f"{sh.index.max().date()}")
    print(f"  unchanged dates: first published {int(g.flat.sum())}, final version {int(g.final_flat.sum())}; by quarter "
          + ", ".join(f"{str(q)} {int(a)}/{int(b)}" for q, (a, b) in
                      g.groupby(g.index.to_period("Q"))[["flat", "final_flat"]].sum().iterrows()))

if __name__ == "__main__":
    main()
