"""Every number in Part 8.

Run: python hub_numbers.py > output.txt   (about a minute; reads data/ and Part 7's Johns Hopkins extract)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import hub as H

RESULTS = H.HERE / "results"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def pts(x: float, digits: int = 1) -> str:
    return f"{100 * x:+.{digits}f}"


def ci(lo_hi: tuple[float, float], digits: int = 1) -> str:
    return f"[{100 * lo_hi[0]:+.{digits}f}, {100 * lo_hi[1]:+.{digits}f}]"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    hol = H.holidays()
    d = H.daily()
    w = H.week_types(H.weekly(d), hol)
    f = H.forecasts()
    s = H.score(f, w)

    section("P0. The data")
    print((H.DATA / "hub_fetch_log.txt").read_text().strip())
    print(f"forecasts: {len(f):,} (model, forecast date, location, horizon), {f.model.nunique()} models; "
          f"for Germany {int((f.location == 'GM').sum()):,}, for the states {int((f.location != 'GM').sum()):,}")
    print(f"  forecast dates {f.forecast_date.min().date()} to {f.forecast_date.max().date()}; "
          f"state forecasts {f[f.location != 'GM'].forecast_date.min().date()} to "
          f"{f[f.location != 'GM'].forecast_date.max().date()}")
    print(f"scored (target week {H.FIRST_WEEK.date()} to {H.LAST_WEEK.date()}, all three counts positive): "
          f"{len(s):,}; Germany {int((s.location == 'GM').sum()):,}, states {int((s.location != 'GM').sum()):,}")
    r = pd.read_csv(H.DATA / "rki_dates.csv.gz", parse_dates=["reporting_date"])
    p = r[(r.reporting_date >= "2020-05-01") & (r.reporting_date <= "2022-11-12")]
    print(f"RKI final file: {r.cases.sum():,.0f} cases; reported May 2020 to November 2022: {p.cases.sum():,.0f}, "
          f"onset known for {pct(p[p.onset_known == 1].cases.sum() / p.cases.sum())}")

    section("P1. Where the holiday gaps are: the day of a weekday national holiday against the same weekday a week "
            "before and after (Germany)")
    days = sorted(pd.Timestamp(x) for x in hol["DE"] if pd.Timestamp(x).dayofweek < 5
                  and pd.Timestamp("2020-05-01") <= pd.Timestamp(x) <= pd.Timestamp("2022-12-31"))
    hr = H.holiday_ratios(d, "GM", days)
    print("publication is read on the day after the holiday, because it lags by a day")
    for _, row in hr.iterrows():
        print(f"  {row.holiday.strftime('%a %d %b %Y')} {hol['DE'][row.holiday.date()]:26s} publication "
              f"{pct(row.publication, 0):>5s}, reporting {pct(row.reporting, 0):>5s}, reference "
              f"{pct(row.reference, 0):>5s}, onset {pct(row.onset, 0):>5s}")
    print("  median over the holidays: " + ", ".join(f"{c} {pct(hr[c].median(), 0)}" for c in
                                                     ("publication", "reporting", "reference", "onset")))
    hr.to_csv(RESULTS / "holiday_ratios.csv", index=False)
    # regional holidays: observing states against the rest on the same day
    rows = []
    for loc, (_, sub, name) in H.STATES.items():
        cal = H.calendar(hol, loc)
        for day, nm in cal.items():
            t = pd.Timestamp(day)
            if day in hol["DE"] or t.dayofweek >= 5 or not (pd.Timestamp("2020-05-01") <= t <= pd.Timestamp("2022-12-31")):
                continue
            x = H.holiday_ratios(d, loc, [t]).iloc[0]
            others = [l2 for l2 in H.STATES if day not in H.calendar(hol, l2)]
            o = pd.concat([H.holiday_ratios(d, l2, [t]) for l2 in others])
            rows.append({"holiday": t, "name": nm, "state": sub, **{c: x[c] for c in ("publication", "reporting", "reference", "onset")},
                         **{f"others_{c}": o[c].median() for c in ("publication", "reporting", "reference", "onset")}})
    rg = pd.DataFrame(rows)
    print(f"regional weekday holidays (state-days): {len(rg)}; median ratio in the observing state against the median of "
          "the other states on the same day:")
    for c in ("publication", "reporting", "reference", "onset"):
        print(f"  {c:12s} observing {pct(rg[c].median(), 0)}, others {pct(rg['others_' + c].median(), 0)}")
    rg.to_csv(RESULTS / "regional_ratios.csv", index=False)

    section("P2. The daily pipeline: Johns Hopkins' morning 7-day count for Germany, scored three ways")
    for lag in (1, 0):
        m = H.jhu_mornings(d, hol, lag)
        hm = m[m.holiday]
        print(f"RKI window ending on d-{lag}: {len(m)} mornings, {m.index.min().date()} to {m.index.max().date()}; "
              f"holiday mornings (a national holiday on the day or the day before) {len(hm)}, from "
              f"{hm.which.nunique()} holidays")
        groups = pd.Series(np.where(m.holiday, m.which.astype(str), m.index.to_period("W").astype(str)), index=m.index)
        for t, label in (("final", "Johns Hopkins' final version"), ("reporting", "RKI by reporting date"),
                         ("reference", "RKI by reference date")):
            e = m[f"e_{t}"]
            diff = e[m.holiday].mean() - e[~m.holiday].mean()
            lo_hi = H.cluster_boot(pd.concat([e, m.holiday], axis=1), groups,
                                   lambda z: z.iloc[:, 0][z.iloc[:, 1]].mean() - z.iloc[:, 0][~z.iloc[:, 1]].mean())
            print(f"  against {label:28s} holiday mornings {pct(e[m.holiday].mean(), 2)}, others "
                  f"{pct(e[~m.holiday].mean(), 2)}, difference {pts(diff, 2)} points {ci(lo_hi, 2)}")
        if lag == 1:
            m.to_csv(RESULTS / "jhu_mornings.csv")

    section("P3. Weekly counts: publication against reference date in holiday weeks")
    w["lr_pub"] = np.log(w.publication / w.reference)
    w["lr_rep"] = np.log(w.reporting / w.reference)
    g = w[w.location == "GM"].set_index("week")
    nat_weeks = g[g.national].index
    rows = []
    for wk in nat_weeks:
        nb = [x for x in H.neighbours(wk, set(nat_weeks)) if x is not None and x in g.index]
        if len(nb) == 2:
            rows.append((wk, g.loc[wk, "lr_pub"] - g.loc[nb, "lr_pub"].mean(), g.loc[wk, "lr_rep"] - g.loc[nb, "lr_rep"].mean(),
                         ", ".join(sorted(set(H.weekday_holidays(hol["DE"], wk))))))
    nw = pd.DataFrame(rows, columns=["week", "pub", "rep", "names"])
    print(f"Germany, weeks with a weekday national holiday: {len(nw)}; publication against reference, holiday week "
          "minus the mean of the nearest weeks without one, before and after (log points):")
    for _, row in nw.iterrows():
        print(f"  week ending {row.week.strftime('%d %b %Y')} ({row.names}): publication {pts(row.pub)}, "
              f"reporting {pts(row.rep)}")
    print(f"  mean: publication {pts(nw.pub.mean())}, reporting {pts(nw.rep.mean())}")
    st = w[w.location != "GM"]
    reg_weeks = sorted(st[st.regional & ~st.national].week.unique())
    x = st[st.week.isin(reg_weeks)]
    nxt = st[st.week.isin([pd.Timestamp(a) + pd.Timedelta(days=7) for a in reg_weeks])].copy()
    prev_reg = set(zip(x[x.regional].location, x[x.regional].week + pd.Timedelta(days=7)))
    nxt["prev"] = [(l, k) in prev_reg for l, k in zip(nxt.location, nxt.week)]
    print(f"states, weeks with a regional weekday holiday and no national one: {len(reg_weeks)}: "
          + ", ".join(f"{pd.Timestamp(k).strftime('%d %b %Y')} ({', '.join(sorted(set(x[(x.week == k) & x.regional].regional_name)))})"
                      for k in reg_weeks))
    a, b = x[x.regional], x[~x.regional]
    print(f"  holiday week, log(publication/reference): observing states {pts(a.lr_pub.mean())} ({len(a)}), "
          f"other states {pts(b.lr_pub.mean())} ({len(b)}); log(reporting/reference): {pts(a.lr_rep.mean())} against "
          f"{pts(b.lr_rep.mean())}")
    print(f"  the week after, log(publication/reference): observing states {pts(nxt[nxt.prev].lr_pub.mean())}, "
          f"other states {pts(nxt[~nxt.prev].lr_pub.mean())}")

    section("P4. The same forecasts scored against the three counts")
    for label, part in (("Germany", s[s.location == "GM"]), ("states", s[s.location != "GM"])):
        print(f"{label}: {len(part):,} forecasts; mean absolute log error of the median: " + ", ".join(
            f"{t} {pct(part[f'e_{t}'].mean())}" for t in H.TRUTHS))
        cov = part.dropna(subset=["c80_publication"])
        print(f"  80% interval coverage ({len(cov):,} forecasts with quantiles): " + ", ".join(
            f"{t} {pct(cov[f'c80_{t}'].mean())}" for t in H.TRUTHS))
        print("  by horizon (publication / reference): " + "; ".join(
            f"{int(h)} wk {pct(g2.e_publication.mean())} / {pct(g2.e_reference.mean())}" for h, g2 in part.groupby("horizon")))
    for label, part, minn in (("Germany", s[s.location == "GM"], 50), ("states", s[s.location != "GM"], 500)):
        m = part.groupby("model").agg(n=("e_publication", "size"), **{t: (f"e_{t}", "mean") for t in H.TRUTHS})
        m = m[m.n >= minn].sort_values("publication")
        print(f"{label}, models with at least {minn} forecasts: {len(m)}; rank correlation of their mean errors: "
              f"publication-reference {m.publication.rank().corr(m.reference.rank()):.3f}, publication-reporting "
              f"{m.publication.rank().corr(m.reporting.rank()):.3f}")
        moved = (m.publication.rank() - m.reference.rank()).abs()
        print(f"  largest change of rank: {int(moved.max())}; models whose rank changes: {int((moved > 0).sum())}; "
              f"first under each count: " + ", ".join(f"{t} {m[t].idxmin()}" for t in H.TRUTHS))
        m.to_csv(RESULTS / f"models_{label.lower()}.csv")

    section("P5. Regional holiday weeks: observing states against the others, same week, same forecasts' errors")
    sx = s[(s.location != "GM") & s.week.isin(reg_weeks)].copy()
    sx = sx.merge(w[["location", "week"]].assign(_k=1), on=["location", "week"])
    obs, non = sx[sx.regional], sx[~sx.regional]
    print(f"forecasts: observing {len(obs):,}, others {len(non):,}, in {sx.week.nunique()} weeks")
    for t in H.TRUTHS:
        print(f"  against {t:11s} observing {pct(obs[f'e_{t}'].mean())}, others {pct(non[f'e_{t}'].mean())}, "
              f"difference {pts(obs[f'e_{t}'].mean() - non[f'e_{t}'].mean())}")
    sx["gap"] = sx.e_reference - sx.e_publication
    diff = sx[sx.regional].gap.mean() - sx[~sx.regional].gap.mean()
    lo_hi = H.cluster_boot(sx[["gap", "regional"]], sx.week,
                           lambda z: z.gap[z.regional].mean() - z.gap[~z.regional].mean())
    print(f"  error against reference minus error against publication: observing {pts(sx[sx.regional].gap.mean())}, "
          f"others {pts(sx[~sx.regional].gap.mean())}; difference {pts(diff)} points, 95% interval over weeks {ci(lo_hi)}")
    print(f"  medians above the publication count: observing {pct((obs['median'] > obs.publication).mean())}, "
          f"others {pct((non['median'] > non.publication).mean())}")

    section("P6. National holiday weeks against the nearest weeks without one, same model, location and horizon "
            "(Germany and states)")
    s["gap"] = s.e_reference - s.e_publication
    key = ["model", "location", "horizon"]
    idx = s.set_index(key + ["week"])
    nat = sorted(w[(w.location == "GM") & w.national].week)
    rows = []
    for wk in nat:
        nb = list(H.neighbours(wk, set(nat)))
        if None in nb:
            continue
        try:
            h = idx.xs(wk, level="week")
            p_, n_ = idx.xs(nb[0], level="week"), idx.xs(nb[1], level="week")
        except KeyError:
            continue
        j = h[["gap", "e_publication", "e_reference"]].join(p_[["gap", "e_publication", "e_reference"]], rsuffix="_p", how="inner") \
            .join(n_[["gap", "e_publication", "e_reference"]], rsuffix="_n", how="inner")
        j["week"] = wk
        rows.append(j.reset_index())
    J = pd.concat(rows)
    for c in ("gap", "e_publication", "e_reference"):
        J[f"d_{c}"] = J[c] - (J[f"{c}_p"] + J[f"{c}_n"]) / 2
    used = sorted(J.week.unique())
    print(f"holiday weeks with forecasts for them and for both neighbours: {len(used)}: " +
          ", ".join(f"{pd.Timestamp(k).strftime('%d %b %Y')} ({', '.join(sorted(set(H.weekday_holidays(hol['DE'], pd.Timestamp(k)))))})" for k in used))
    print(f"forecast triples: {len(J):,}")
    print(f"  error against publication: holiday week minus neighbours {pts(J.d_e_publication.mean())}; "
          f"against reference {pts(J.d_e_reference.mean())}")
    lo_hi = H.cluster_boot(J.d_gap.reset_index(drop=True), J.week.reset_index(drop=True), lambda z: z.mean())
    print(f"  (error against reference minus against publication), holiday week minus neighbours: {pts(J.d_gap.mean())} "
          f"points, 95% interval over weeks {ci(lo_hi)}")
    for hz, g2 in J.groupby("horizon"):
        lo_hi = H.cluster_boot(g2.d_gap.reset_index(drop=True), g2.week.reset_index(drop=True), lambda z: z.mean())
        print(f"  horizon {int(hz)}: {pts(g2.d_gap.mean())} points {ci(lo_hi)} ({len(g2):,} triples)")
    print("  by week: " + "; ".join(f"{pd.Timestamp(k).strftime('%d %b %Y')} {pts(g2.d_gap.mean())}" for k, g2 in J.groupby("week")))
    J.to_csv(RESULTS / "national_weeks.csv", index=False)


if __name__ == "__main__":
    main()
