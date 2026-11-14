"""Every number in Part 9.

Run: python calendar_numbers.py > output.txt   (about a minute; reads data/ and Part 7's Johns Hopkins extracts)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import learned as L

RESULTS = L.HERE / "results"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def f3(x: float) -> str:
    return f"{x:.3f}"


def ci(lo_hi: tuple[float, float]) -> str:
    return f"[{lo_hi[0]:+.3f}, {lo_hi[1]:+.3f}]"


def scores(x: pd.DataFrame, models=L.MODELS, metric=L.logloss) -> str:
    return ", ".join(f"{m} {f3(metric(x.flat, x[m]).mean())}" for m in models)


def diff_ci(x: pd.DataFrame, a: str, b: str) -> str:
    d = pd.Series(L.logloss(x.flat, x[a]) - L.logloss(x.flat, x[b]), index=x.index)
    return f"{d.mean():+.3f} {ci(L.cluster_boot(d, x.producer))}"


def persistence(e: pd.DataFrame) -> pd.DataFrame:
    """Pairs of the same named holiday, on a working day, for the same producer in consecutive years."""
    h = e[e.kind == "holiday"].copy()
    h["y"] = h.date.dt.year
    pairs = h.merge(h, on=["producer", "hol_name"], suffixes=("_a", "_b"))
    return pairs[pairs.y_b == pairs.y_a + 1]


def ramp_scores(n_series: int = 200, length: int = 150, seed: int = L.SEED) -> dict:
    """Mean log loss of the sliding window and of change-point detection on simulated series of one day of
    the week with two moves between a low and a high rate, as steps or as straight ramps."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    out = {}
    for ramp in (0, 4, 8, 13, 26):
        res = {"window 8 weeks": [], "change points": []}
        for _ in range(n_series):
            cps = np.sort(rng.choice(np.arange(20, length - 20), 2, replace=False))
            levels = (0.03, 0.95, 0.03)
            p = np.full(length, levels[0])
            for i, c in enumerate(cps):
                move = np.clip((t - c) / ramp, 0, 1) if ramp else (t >= c).astype(float)
                p = np.where(t >= c, levels[i] + (levels[i + 1] - levels[i]) * move, p)
            y = (rng.random(length) < p).astype(float)
            df = pd.DataFrame({"producer": "x", "group": "countries", "wd": 0, "kind": "working",
                               "eligible": True, "flat": y,
                               "date": pd.Timestamp("2020-06-01") + pd.to_timedelta(t * 7, unit="D")})
            pooled = pd.Series(0.3, index=df.index)
            cy = np.concatenate([[0], np.cumsum(y)])
            lo = np.maximum(t - 8, 0)
            win = (cy[t] - cy[lo] + L.PRIOR * 0.3) / (t - lo + L.PRIOR)
            cp = L.bocpd_rates(df, 1 / 52, pooled).sort_index().to_numpy()
            res["window 8 weeks"].append(L.logloss(y, win).mean())
            res["change points"].append(L.logloss(y, cp).mean())
        out[ramp] = {k: float(np.mean(v)) for k, v in res.items()}
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = L.frame()
    e = df[df.eligible]
    test = lambda x: x[x.date >= L.FIRST_TEST]  # noqa: E731

    section("P0. The producers and their files")
    print((L.DATA / "fetch_log.txt").read_text().strip())
    for g in L.GROUPS:
        x, xe = df[df.group == g], e[e.group == g]
        print(f"{g:10s} producers {x.producer.nunique():3d}; producer-days {len(x):7,}; counted {len(xe):7,} "
              f"({xe.producer.nunique()} producers); unchanged {pct(xe.flat.mean())}")
    print(f"all       producers {df.producer.nunique():3d}; producer-days {len(df):7,}; counted {len(e):7,}")
    print("counted producer-days are those with a file and a count of the day before, on or before the producer's "
          f"last change, with at least {L.MIN_COUNT} new cases in the 28 days before (files as first published)")
    print("unchanged share by year (April to March):")
    for g in L.GROUPS:
        x = e[e.group == g]
        print(f"  {g:10s} " + ", ".join(f"{y} {pct(v)}" for y, v in x.groupby('year').flat.mean().items()))
    wk = pd.read_csv(L.DATA / "weekends.csv", dtype={"weekend": str})
    cals = df.drop_duplicates("producer")[["producer", "cal"]]
    w21 = wk[wk.year == 2021].set_index("calendar").weekend
    other = cals[cals.cal.map(w21) != "56"]
    print(f"producers whose legal weekend in 2021 is not Saturday and Sunday: {len(other)} "
          f"({', '.join(sorted(other.producer))})")
    print("unchanged share by kind of day (counted days, all years):")
    for g in L.GROUPS:
        x = e[e.group == g]
        print(f"  {g:10s} " + ", ".join(f"{k} {pct(v)} (n={int(n):,})" for k, (v, n) in
                                        x.groupby('kind').flat.agg(['mean', 'size']).loc[list(L.KINDS)].iterrows()))

    section("P1. The published calendar as a rule (test period, 1 April 2021 to 9 March 2023)")
    t = test(e)
    off = t.weekend | t.kind.eq("holiday")
    u = t.flat == 1
    print(f"counted producer-days {len(t):,}; unchanged {int(u.sum()):,} ({pct(u.mean())})")
    print(f"unchanged files on a legal weekend day or holiday: {pct((off & u).sum() / u.sum())}; "
          f"on a legal working day: {pct((~off & u).sum() / u.sum())}")
    print(f"legal weekend days and holidays with an unchanged file: {pct(t.flat[off].mean())}; "
          f"legal working days with one: {pct(t.flat[~off].mean())}")
    for g in L.GROUPS:
        x = t[t.group == g]
        o, uu = x.weekend | x.kind.eq("holiday"), x.flat == 1
        print(f"  {g:10s} unchanged on legal off-days {pct((o & uu).sum() / uu.sum())}; "
              f"legal off-days unchanged {pct(x.flat[o].mean())}; legal working days unchanged {pct(x.flat[~o].mean())}")

    section("P2. Four models, each fitted only on days before the one it predicts (test period)")
    p = L.predictions(df)
    _, pooled, _ = L.weekday_rates(df)
    cp = L.bocpd_rates(df, 1 / 52, pooled)
    pb = L.predictions(df, base=cp)
    p["weekdays, change points by day"] = pb["weekdays"]
    cj = L.bocpd_joint(df, 1 / 365, pooled)
    pj = L.predictions(df, base=cj)
    p["learned, change points"] = pj["learned"]
    p["weekdays, change points"] = pj["weekdays"]
    tp = test(p)
    models = (*L.MODELS, "learned, change points")
    print(f"weekday window {L.WINDOW} days, holiday window {L.HOLIDAY_WINDOW} days, prior {L.PRIOR} pseudo-days, "
          f"holiday-effect prior sd {L.TAU}")
    print("mean log loss (lower is better; predicting the group's overall rate everywhere would score about "
          "0.5 to 0.6):")
    for g in L.GROUPS:
        x = tp[tp.group == g]
        base = x.flat.mean()
        naive = -(base * np.log(base) + (1 - base) * np.log(1 - base))
        print(f"  {g:10s} n={len(x):6,}  constant rate {f3(naive)}; " + scores(x, models))
    x = tp
    print(f"  {'all':10s} n={len(x):6,}  " + scores(x, models))
    print("Brier score:")
    for g in L.GROUPS:
        x = tp[tp.group == g]
        print(f"  {g:10s} " + scores(x, models, L.brier))
    print("difference in log loss, published minus learned, with a 95% interval resampling producers:")
    for g in L.GROUPS:
        x = tp[tp.group == g]
        print(f"  {g:10s} {diff_ci(x, 'published', 'learned')}")
    print("difference, learned minus learned with change points:")
    for g in L.GROUPS:
        x = tp[tp.group == g]
        print(f"  {g:10s} {diff_ci(x, 'learned', 'learned, change points')}")
    print("what the learned calendar (change points) said before each unchanged file:")
    for g in L.GROUPS:
        uu = tp[(tp.group == g) & (tp.flat == 1)]["learned, change points"]
        print(f"  {g:10s} unchanged files {len(uu):6,}: expected (p >= 0.5) {pct((uu >= 0.5).mean())}, "
              f"in between {pct(((uu >= 0.05) & (uu < 0.5)).mean())}, unexpected (p < 0.05) {pct((uu < 0.05).mean())}")
    ex = tp["learned, change points"] >= 0.5
    print(f"  days the learned calendar expected an unchanged file: {int(ex.sum()):,}, of which unchanged "
          f"{pct(tp.flat[ex].mean())}; days it gave below 0.05: {int((tp['learned, change points'] < 0.05).sum()):,}, "
          f"of which unchanged {pct(tp.flat[tp['learned, change points'] < 0.05].mean())}")

    section("P3. Holidays and the days after them (test period)")
    for g in L.GROUPS:
        for k in ("holiday", "after"):
            x = tp[(tp.group == g) & (tp.kind == k)]
            print(f"  {g:10s} {k:7s} n={len(x):5,} unchanged {pct(x.flat.mean())}: " + scores(x))
    print("learned minus weekdays + published holidays, on holidays (95% interval resampling producers):")
    for g in L.GROUPS:
        x = tp[(tp.group == g) & (tp.kind == "holiday")]
        print(f"  {g:10s} {diff_ci(x, 'learned', 'weekdays + published holidays')}; "
              f"weekdays + published holidays minus weekdays {diff_ci(x, 'weekdays + published holidays', 'weekdays')}")
    print("holiday effects on the odds of an unchanged file (test holidays on working days):")
    for g in L.GROUPS:
        x = tp[(tp.group == g) & (tp.kind == "holiday")]
        last = x.sort_values("date").groupby("producer").tail(1)
        q = np.exp(last.dp).quantile([0.1, 0.5, 0.9])
        print(f"  {g:10s} group effect, median over the test holidays: odds x{np.exp(x.d0).median():.1f}; "
              f"producers' own effects at their last test holiday (n={len(last)}): 10th pct x{q[0.1]:.1f}, "
              f"median x{q[0.5]:.1f}, 90th pct x{q[0.9]:.1f}")
    for tau in (0.5, 2.0):
        pt = test(L.predictions(df, tau=tau))
        print(f"  prior sd {tau}: " + "; ".join(
            f"{g} holidays learned {f3(L.logloss(x.flat, x.learned).mean())}"
            for g in L.GROUPS for x in [pt[(pt.group == g) & (pt.kind == 'holiday')]]))

    section("P4. The same holiday a year later")
    pr = persistence(p)
    print("pairs: the same named legal holiday on a working day, the same producer, two consecutive years, both "
          "days counted")
    for g in L.GROUPS:
        x = pr[pr.group_a == g]
        print(f"  {g:10s} pairs {len(x):4d}; unchanged in the second year: after an unchanged file the year before "
              f"{pct(x[x.flat_a == 1].flat_b.mean())} (n={int((x.flat_a == 1).sum())}), after a changed one "
              f"{pct(x[x.flat_a == 0].flat_b.mean())} (n={int((x.flat_a == 0).sum())})")
        d = x[(x.weekdays_a < 0.1) & (x.weekdays_b < 0.1)]
        print(f"  {'':10s} both years in a daily regime (weekday rate below 10%): pairs {len(d)}; after unchanged "
              f"{pct(d[d.flat_a == 1].flat_b.mean())} (n={int((d.flat_a == 1).sum())}), after changed "
              f"{pct(d[d.flat_a == 0].flat_b.mean())} (n={int((d.flat_a == 0).sum())})")

    section("P5. The conjecture as worded: a calendar fitted to one year, used for the next")
    seg, tr, ch = L.changes(df)
    for a, b in (("2020-21", "2021-22"), ("2021-22", "2022-23")):
        fz = L.predictions(df, fixed=L.YEARS[a])
        x = fz[fz.year == b]
        print(f"fitted on {a}, scored on {b}:")
        for g in L.GROUPS:
            xx = x[x.group == g]
            print(f"  {g:10s} n={len(xx):6,} " + scores(xx))
        start, end = pd.Timestamp(L.YEARS[a][0]), pd.Timestamp(L.YEARS[b][1])
        moved = tr[(tr.first_ref >= start) & (tr.first_ref <= end)][["producer", "wd"]].drop_duplicates()
        key = pd.MultiIndex.from_frame(x[["producer", "wd"]])
        stable = ~key.isin(pd.MultiIndex.from_frame(moved))
        xs = x[stable]
        print(f"  days of the week with no move in either year ({pct(stable.mean())} of the scored days): "
              + scores(xs))
        print(f"  the others: " + scores(x[~stable]))

    section("P6. How the calendars changed")
    print(f"segments (one series per producer and day of the week, holidays left out): {len(seg):,} in "
          f"{seg.groupby(['producer', 'wd']).ngroups:,} series")
    print(f"transitions between a rate of at most 20% and one of at least 80%: {len(tr)} "
          f"(up {int((tr.direction == 'up').sum())}, down {int((tr.direction == 'down').sum())}); "
          f"observed week by week (no gap over 14 days): {int((tr.gap <= 14).sum())}")
    o = tr[tr.gap <= 14]
    print(f"  of these, directly from one segment to the next: {pct((o.n_between == 0).mean())}")
    print(f"changes of process (transitions of one producer, same direction, within 6 days): {len(ch)}; "
          "by number of days of the week involved: " +
          ", ".join(f"{k}: {v}" for k, v in ch.n_days.value_counts().sort_index().items()))
    multi = ch[ch.n_days >= 2]
    share = multi.n_days.sum() / ch.n_days.sum()
    print(f"  transitions that belong to a change involving two or more days: {pct(share)}")
    print(f"  producers with at least one change involving two or more days: {multi.producer.nunique()} "
          f"(US states {multi[multi.group == 'US states'].producer.nunique()}, provinces "
          f"{multi[multi.group == 'provinces'].producer.nunique()}, countries "
          f"{multi[multi.group == 'countries'].producer.nunique()})")
    for g in L.GROUPS:
        x = multi[multi.group == g]
        print(f"  {g:10s} changes {len(x)} (up {int((x.direction == 'up').sum())}, down {int((x.direction == 'down').sum())}); "
              f"by half-year: " + ", ".join(f"{k} {v}" for k, v in
                                            x.groupby(x.date.dt.year.astype(str) + np.where(x.date.dt.month > 6, ' H2', ' H1')).size().items()))
    up = multi[multi.direction == "up"]
    prof = L.event_profile(df, up)
    np.savetxt(RESULTS / "profile_observed.csv", prof, delimiter=",", fmt="%.6g")
    m = np.nanmean(prof, 0)
    print(f"share of unchanged files on the days of the week that changed, by week relative to the change "
          f"({len(up)} upward changes involving two or more days):")
    print("  weeks -12..+12: " + " ".join(f"{v:.2f}" for v in m))
    print(f"  weeks -4 to -2 {pct(np.nanmean(prof[:, 8:11]))}, weeks +1 to +3 {pct(np.nanmean(prof[:, 13:16]))}")
    for rw in (0, 4, 8, 13):
        sp, direct = L.simulated_profile(rw)
        np.savetxt(RESULTS / f"profile_ramp{rw}.csv", sp, delimiter=",", fmt="%.6g")
        print(f"  simulated, {'step' if rw == 0 else f'{rw}-week ramp':12s} ({len(sp)} series): weeks -4 to -2 "
              f"{pct(np.nanmean(sp[:, 8:11]))}, weeks +1 to +3 {pct(np.nanmean(sp[:, 13:16]))}; "
              f"fitted as a direct step from one week to the next {pct(direct)}")

    section("P7. Changes that were announced")
    doc = L.documented(ch)
    print("sampling frame: the ten most populous US states and the ten most populous EU member states by the "
          "Johns Hopkins lookup table, and the United Kingdom; each announced change matched to the producer's "
          "detected change, involving two or more days of the week, nearest to the announced first day")
    print(f"frame producers with such a change: {len(doc)}; without one: California, Texas, New York, Germany, "
          "Poland, Czechia")
    for r in doc.itertuples():
        days = "".join("MTWTFSS"[d] for d in r.detected_days)
        print(f"  {r.producer:15s} announced first day {r.first_day.date()}  detected {r.detected.date()} "
              f"({days})  offset {r.offset:+d}; changes of two or more days detected for this producer: {r.n_multi}")
    print(f"offsets: {int((doc.offset == 0).sum())} at zero, all within {doc.offset.min():+d} to {doc.offset.max():+d} days")
    ro = ch[(ch.producer == "Romania") & (ch.direction == "down") & (ch.n_days >= 2)]
    print(f"Romania's return to daily reporting (announced from 12 July 2022): detected {ro.date.iloc[0].date()}")

    section("P8. Keeping the weekday rate current: windows, forgetting, change points")
    train = tp[tp.kind.isin(["working", "weekend"])]
    rows = []
    for w in (7, 14, 28, 56, 91, 182, 365, 730):
        r, _, _ = L.weekday_rates(df, w)
        rows.append(("window", w, L.logloss(train.flat, r[train.index]).mean()))
    for h in (7, 14, 28, 56, 112, 224):
        r = L.ewma_rates(df, h, pooled).reindex(train.index)
        rows.append(("half-life", h, L.logloss(train.flat, r).mean()))
    for runs in (8, 13, 26, 52, 104, 208):
        r = L.bocpd_rates(df, 1 / runs, pooled).reindex(train.index)
        rows.append(("change points by day of the week", runs * 7, L.logloss(train.flat, r).mean()))
    for days in (91, 182, 365, 730):
        r = L.bocpd_joint(df, 1 / days, pooled).reindex(train.index)
        rows.append(("change points shared by the producer's days", days, L.logloss(train.flat, r).mean()))
    est = pd.DataFrame(rows, columns=["family", "days", "logloss"])
    est.to_csv(RESULTS / "estimators.csv", index=False)
    print(f"working and weekend days of the test period, n={len(train):,}; mean log loss of the weekday rate")
    for fam, g in est.groupby("family", sort=False):
        print(f"  {fam}: " + ", ".join(f"{int(d)}d {f3(v)}" for d, v in zip(g.days, g.logloss)))
    print("  (window: days in the window; half-life: days; change points: expected days between changes)")
    for g in L.GROUPS:
        x = train[train.group == g]
        r56 = L.logloss(x.flat, x.weekdays).mean()
        rcd = L.logloss(x.flat, x["weekdays, change points by day"]).mean()
        rcp = L.logloss(x.flat, x["weekdays, change points"]).mean()
        print(f"  {g:10s} window 56 days {f3(r56)}, change points by day (a year apart) {f3(rcd)}, "
              f"shared change points (a year apart) {f3(rcp)}")
    d = pd.Series(L.logloss(train.flat, train["weekdays, change points by day"]) -
                  L.logloss(train.flat, train["weekdays, change points"]), index=train.index)
    print(f"  by day minus shared: {d.mean():+.3f} {ci(L.cluster_boot(d, train.producer))}")
    d = pd.Series(L.logloss(train.flat, train.weekdays) - L.logloss(train.flat, train["weekdays, change points"]),
                  index=train.index)
    print(f"  window 56 days minus shared: {d.mean():+.3f} {ci(L.cluster_boot(d, train.producer))}")
    sim = ramp_scores()
    print("simulated series (one day of the week, two moves between 3% and 95%): mean log loss")
    for ramp, v in sim.items():
        print(f"  {'steps' if ramp == 0 else f'{ramp}-week ramps':14s} window 8 weeks {f3(v['window 8 weeks'])}, "
              f"change points {f3(v['change points'])}")

    section("P9. Robustness")
    for mw in (400, 2000):
        old = L.MIN_COUNT
        L.MIN_COUNT = mw
        d2 = L.frame()
        L.MIN_COUNT = old
        pt = test(L.predictions(d2))
        print(f"28-day count of at least {mw}: counted {int(d2.eligible.sum()):,}; " + "; ".join(
            f"{g} published {f3(L.logloss(x.flat, x.published).mean())} learned {f3(L.logloss(x.flat, x.learned).mean())}"
            for g in L.GROUPS for x in [pt[pt.group == g]]))
    for prior in (0.5, 8.0):
        pt = test(L.predictions(df, prior=prior))
        print(f"prior {prior}: " + "; ".join(f"{g} learned {f3(L.logloss(x.flat, x.learned).mean())}"
                                             for g in L.GROUPS for x in [pt[pt.group == g]]))
    for w in (28, 91):
        pt = test(L.predictions(df, window=w))
        print(f"weekday window {w}: " + "; ".join(
            f"{g} published {f3(L.logloss(x.flat, x.published).mean())} learned {f3(L.logloss(x.flat, x.learned).mean())}"
            for g in L.GROUPS for x in [pt[pt.group == g]]))

    section("P10. Worked example: Florida, June 2021")
    fl = p[(p.producer == "Florida") & (p.date >= "2021-05-24") & (p.date <= "2021-07-04")]
    for _, r in fl.iterrows():
        print(f"  {r.date.strftime('%a %d %b')} {r.kind:8s} unchanged {int(r.flat)}  published {r.published:.2f}  "
              f"window {r.weekdays:.2f}  by day {r['weekdays, change points by day']:.2f}  "
              f"shared {r['weekdays, change points']:.2f}")
    after = p[(p.producer == "Florida") & (p.date >= "2021-06-07") & (p.wd < 4) & p.kind.isin(["working"])]
    for col, label in (("weekdays", "window"), ("weekdays, change points by day", "change points by day"),
                       ("weekdays, change points", "shared change points")):
        first = after[after[col] >= 0.5].date.min()
        print(f"  first Monday to Thursday with p >= 0.5, {label}: {first.date()}")


if __name__ == "__main__":
    main()
