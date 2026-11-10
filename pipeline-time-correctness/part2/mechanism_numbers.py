"""Every number quoted in Part 2, printed in the order the article uses them.

Run: python mechanism_numbers.py > output.txt
"""
from __future__ import annotations

import math

import numpy as np

import mechanism as m


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
section("Delay distributions (whole days from event date to report date)")
for d in m.SCENARIOS + (m.BACKLOG,):
    F = d.F()
    print(f"{d.name:8s} log-normal median {d.median} days, log-sd {d.sigma}")
    print(f"  reported by day 0, 1, 3, 7, 14: " + ", ".join(pct(F[k]) for k in (0, 1, 3, 7, 14)))
    print(f"  days until 50%, 90%, 99% reported: {d.quantile_days(0.5)}, {d.quantile_days(0.9)}, "
          f"{d.quantile_days(0.99)}")
    print(f"  mean delay (days, capped at {m.D_MAX}): {float(np.arange(m.D_MAX + 1) @ d.pmf()):.2f}")

# ---------------------------------------------------------------------------
section("M1. Right truncation")
print("Share still missing at the end of the last day, constant incidence:")
for d in m.SCENARIOS:
    print(f"  {d.name:6s} last day {pct(m.window_shortfall(d, 1))}, last 7 days {pct(m.window_shortfall(d, 7))}, "
          f"last 28 days {pct(m.window_shortfall(d, 28))}")

print("\nLast-7-day total, 1000 events a day, 2000 Monte Carlo runs:")
for d in m.SCENARIOS:
    r = m.truncation_bias_against_noise(d)
    print(f"  {d.name:6s} final {r['mean_final']:.0f}, known {r['mean_known']:.0f}, shortfall "
          f"{r['mean_shortfall']:.0f}, sd of the final total {r['sd_final']:.0f}, "
          f"shortfall / sd = {r['mean_shortfall'] / r['sd_final']:.1f}")

print("\nWeek-over-week change of the last two weeks, as known on the last day:")
for label, g in (("flat", 0.0), ("doubling every 14 days", math.log(2) / 14), ("doubling every 7 days", math.log(2) / 7)):
    true = math.exp(7 * g) - 1
    seen = ", ".join(f"{d.name} {m.apparent_weekly_change(d, g) * 100:+.1f}%" for d in m.SCENARIOS)
    print(f"  {label:24s} true {true * 100:+.1f}%; seen: {seen}")

print("\nGrowth at which the last week, as known, looks the same as the week before:")
for d in m.SCENARIOS:
    g = m.growth_that_looks_flat(d)
    print(f"  {d.name:6s} {100 * g:.2f}% a day = {100 * (math.exp(7 * g) - 1):+.0f}% a week, "
          f"doubling every {math.log(2) / g:.1f} days")

print("\nFigure 1 series: 56 days, doubling every 14 days, 1000 a day on the last day.")
for d in m.SCENARIOS:
    final, known = m.simulate_counts(d, growth=math.log(2) / 14)
    peak = int(np.argmax(known))
    print(f"  {d.name:6s} last 7 days final {final[-7:].sum()}, known {known[-7:].sum()} "
          f"({pct(known[-7:].sum() / final[-7:].sum())}); highest known day is day {peak} of 55")

# ---------------------------------------------------------------------------
section("M2. Waiting against estimating")
print("Watermark with fixed slack, windows of one day: share of a window in its on-time pane")
for d in m.SCENARIOS:
    vals = ", ".join(f"{s}d {pct(m.on_time_completeness(d, s))}" for s in (0, 0.5, 1, 2, 3, 7, 14))
    print(f"  {d.name:6s} {vals}")
print("Slack needed for 50%, 90%, 99% in the on-time pane (days):")
for d in m.SCENARIOS:
    print(f"  {d.name:6s} " + ", ".join(f"{m.slack_for(d, q):.1f}" for q in (0.5, 0.9, 0.99)))
print("Share dropped for good, medium delays, slack 1 day, allowed lateness 0, 3, 7, 14 days:")
print("  " + ", ".join(f"L={L}d {pct(m.dropped_share(m.MEDIUM, 1, L), 2)}" for L in (0, 3, 7, 14)))
print("Share dropped for good, slow delays, slack 1 day, allowed lateness 0, 3, 7, 14 days:")
print("  " + ", ".join(f"L={L}d {pct(m.dropped_share(m.SLOW, 1, L), 2)}" for L in (0, 3, 7, 14)))

waits = np.arange(15)
stable = m.simulate_reporting()
shifted = m.simulate_reporting(after=m.BACKLOG, shift_day=240)
r_stable = m.compare_policies(stable, range(100, 400), waits)
r_shift = m.compare_policies(shifted, range(240, 270), waits)
r_after = m.compare_policies(shifted, range(270, 330), waits)
print("\nPublishing day t after waiting w days: mean absolute percentage error against the final count")
print("420 simulated days, about 1000 events a day, medium delays; chain-ladder factors from the latest 28 days")
print(" w   raw (stable)   nowcast (stable)   raw (30 days after shift)   nowcast (30 days after shift)   nowcast (days 31-90 after)")
for w in waits:
    print(f"{w:2d}   {pct(r_stable.raw[w]):>12s}   {pct(r_stable.nowcast[w]):>16s}   {pct(r_shift.raw[w]):>25s}   "
          f"{pct(r_shift.nowcast[w]):>29s}   {pct(r_after.nowcast[w]):>26s}")
def wait_text(w):
    return "more than 14 days" if w is None else f"{w} days"


for level in (0.05, 0.02):
    print(f"First wait with error below {pct(level, 0)}: raw {wait_text(m.first_wait_below(r_stable.raw, waits, level))}, "
          f"nowcast {wait_text(m.first_wait_below(r_stable.nowcast, waits, level))} (stable delays)")

print("\nAfter the shift (30 days), nowcast error at waits 0, 1, 3, 7 by estimation window:")
for win in (7, 14, 28, 56):
    r = m.compare_policies(shifted, range(240, 270), waits, window=win)
    s = m.compare_policies(stable, range(100, 400), waits, window=win)
    print(f"  window {win:2d} days: after shift " + ", ".join(pct(r.nowcast[w]) for w in (0, 1, 3, 7))
          + "; stable " + ", ".join(pct(s.nowcast[w]) for w in (0, 1, 3, 7)))

low = m.simulate_reporting(base=50.0)
r_low = m.compare_policies(low, range(100, 400), waits)
print("\nAbout 50 events a day, stable medium delays, waits 0, 1, 3, 7:")
print("  raw " + ", ".join(pct(r_low.raw[w]) for w in (0, 1, 3, 7))
      + "; nowcast " + ", ".join(pct(r_low.nowcast[w]) for w in (0, 1, 3, 7)))

# ---------------------------------------------------------------------------
section("The simulated shop (M3 and M5)")
shop = m.simulate_shop()
o, v = shop.orders, shop.versions
lag_h = (o.arrival_ts - o.event_ts) / m.HOUR
print(f"orders {len(o)} over {m.SHOP_DAYS} days; offline {pct(o.offline.mean())}; versions {len(v)}")
print(f"lag from order to arrival: median {60 * np.quantile(lag_h, 0.5):.1f} min, 90% {60 * np.quantile(lag_h, 0.9):.0f} min, "
      f"95% {np.quantile(lag_h, 0.95):.1f} h, 99% {np.quantile(lag_h, 0.99):.1f} h, 99.9% {np.quantile(lag_h, 0.999):.0f} h")
off = lag_h[o.offline]
print(f"offline orders: median lag {np.quantile(off, 0.5):.1f} h, 90% {np.quantile(off, 0.9):.1f} h")
corrected = v[v.version == 2]
print(f"corrected orders {len(corrected)} ({pct(len(corrected) / len(o))}), of which cancelled "
      f"{int((corrected.amount == 0).sum())}; median correction "
      f"{np.median((corrected.visible_at - corrected.event_ts) / m.DAY):.1f} days after the order")
print(f"versions not yet visible at the last run (day {m.LAST_RUN_DAY}): {int((v.visible_at > m.run_ts(m.LAST_RUN_DAY)).sum())}")

section("M3a. Incremental loads with a high-water mark")
ev, arr = o.event_ts.to_numpy(), o.arrival_ts.to_numpy()
for interval, label in ((300, "5 minutes"), (900, "15 minutes"), (m.HOUR, "1 hour"), (6 * m.HOUR, "6 hours"), (m.DAY, "1 day (06:00)")):
    start = m.RUN_HOUR * m.HOUR if interval == m.DAY else 0
    lost = ~m.hwm_loaded(ev, arr, interval, start)
    print(f"  every {label:14s} rows lost {pct(lost.mean(), 2)}, revenue lost {pct(o.amount[lost].sum() / o.amount.sum(), 2)}; "
          f"offline orders lost {pct(lost[o.offline.to_numpy()].mean())}, online {pct(lost[~o.offline.to_numpy()].mean(), 2)}")
sql_rows = m.hwm_daily_sql(o)
np_rows = int(m.hwm_loaded(ev, arr, m.DAY, m.RUN_HOUR * m.HOUR).sum())
print(f"  daily load as SQL in DuckDB: {sql_rows} rows loaded; vectorized: {np_rows}; equal: {sql_rows == np_rows}")
print("  a high-water mark on arrival time loses no rows here, by construction (each row is read by the first run after it arrives)")

section("M3b. Corrections under three load patterns")
run = m.run_shop(shop)
tot = run.totals.set_index("pattern")
truth_rev = tot.loc["truth", "revenue"]
for p in ("insert_only", "upsert", "version_log"):
    print(f"  {p:12s} orders {tot.loc[p, 'orders']}, revenue {tot.loc[p, 'revenue'] / 100:,.2f} "
          f"({(tot.loc[p, 'revenue'] / truth_rev - 1) * 100:+.2f}% against the final truth)")
print(f"  truth        orders {tot.loc['truth', 'orders']}, revenue {truth_rev / 100:,.2f}")

sent = run.sent.pivot(index="day", columns="pattern", values=["orders", "revenue"])
regen = run.regenerated.pivot(index="day", columns="pattern", values=["orders", "revenue"])
truth = run.truth.set_index("day")
print("\nThe 06:00 report for yesterday, rerun on day 150 from each table (90 reports):")
for p in ("insert_only", "upsert", "version_log"):
    same = int(((regen["revenue"][p] == sent["revenue"][p]) & (regen["orders"][p] == sent["orders"][p])).sum())
    print(f"  {p:12s} reproduced exactly {same} of 90")

print("\nWhat the 06:00 report said against the final value of the same day (upsert table):")
for col in ("orders", "revenue"):
    rel = (sent[col]["upsert"] - truth[col]) / truth[col]
    print(f"  {col:8s} mean {rel.mean() * 100:+.2f}%, range {rel.min() * 100:+.2f}% to {rel.max() * 100:+.2f}%, "
          f"days off by more than 1%: {int((rel.abs() > 0.01).sum())} of 90, more than 2%: {int((rel.abs() > 0.02).sum())}")

# ---------------------------------------------------------------------------
section("M5. Backfills")
B = m.BACKFILL_DAY
rng_days = list(range(0, B - 1))
frozen = run.frozen.set_index("event_day")
daily = run.daily.set_index("event_day")
appended = run.daily_appended.set_index("event_day")
print(f"Backfill on day {B} of days 0..{B - 2} ({len(rng_days)} days)")
for col in ("orders", "revenue"):
    before, after, app = frozen[col].loc[rng_days], daily[col].loc[rng_days], appended[col].loc[rng_days]
    rel = (after - before) / before
    print(f"  {col:8s} appended backfill: total {app.sum() / before.sum():.2f} times what was sent; "
          f"replaced: {(after.sum() / before.sum() - 1) * 100:+.2f}%")
    print(f"           restated days changed by more than 1%: {int((rel.abs() > 0.01).sum())}, more than 2%: "
          f"{int((rel.abs() > 0.02).sum())}, range {rel.min() * 100:+.2f}% to {rel.max() * 100:+.2f}%")
    ratio = daily[col] / truth[col]
    print(f"           daily table / final truth: days {B - 8}..{B - 2} {ratio.loc[B - 8:B - 2].mean():.4f}, "
          f"days {B - 1}..{B + 5} {ratio.loc[B - 1:B + 5].mean():.4f}")

print(f"\nWeek-over-week rule: alert when a day is more than {pct(m.ALERT_DROP, 0)} below the same weekday a week earlier.")
print(f"Real falls in demand on days {sorted(m.OUTAGES)} (factor {m.OUTAGES[25]})")
for col in ("orders", "revenue"):
    c = m.comparisons(run, col)
    mixed = c["mixed"]
    print(f"  {col}:")
    for k in ("truth", "frozen", "stored", "live"):
        alerts = list(c.index[c[k + "_alert"]])
        bias = (c[k] / c["truth"] - 1).mean()
        print(f"    {k:7s} alerts on days {alerts}; mean relative error of the ratio {bias * 100:+.2f}%")
    print(f"    stored, the {int(mixed.sum())} days (days {c.index[mixed].min()}-{c.index[mixed].max()}) compared with "
          f"a restated week: mean relative error {(c.loc[mixed, 'stored'] / c.loc[mixed, 'truth'] - 1).mean() * 100:+.2f}%; "
          f"other days {(c.loc[~mixed, 'stored'] / c.loc[~mixed, 'truth'] - 1).mean() * 100:+.2f}%")

completeness = float((sent["orders"]["upsert"] / truth["orders"]).mean())
print(f"\nMean completeness of the orders count in the 06:00 report: {completeness:.4f}")
print("Chance a week-over-week rule fires on noise alone (Poisson; 1,000,000 draws):")
for mean in (2000, 1600):
    same = m.false_alert_rate(mean, 1.0)
    mixed = m.false_alert_rate(mean, completeness)
    print(f"  {mean} orders a day: both days complete {pct(same, 3)} ({same * 365:.1f} a year); latest at "
          f"{completeness:.3f} {pct(mixed, 3)} ({mixed * 365:.1f} a year); ratio {mixed / same:.0f}")

# ---------------------------------------------------------------------------
section("M4. Features")
print("(a) Next week's event count from this week's count, medium delays, 10,000 customers")
train, serve = m.count_features(seed=5), m.count_features(seed=7)
print(f"  this week's count as known on the day / final: {train.as_of.mean() / train.final.mean():.3f}")
for d in m.SCENARIOS:
    c = m.calibration(m.count_features(d, seed=5), m.count_features(d, seed=7))
    print(f"  {d.name:6s} trained on final values: mean prediction / mean outcome {c['trained_on_final']:.3f}, "
          f"MAE {c['trained_on_final_mae']:.1f}, slope {c['trained_on_final_slope']:.2f}; trained as of the day: "
          f"{c['trained_as_of']:.3f}, MAE {c['trained_as_of_mae']:.1f}, slope {c['trained_as_of_slope']:.2f}")

print("\n(b) Churn in the next 90 days from usage and plan, 20,000 customers per cohort")
r = m.leakage_experiment()
print(f"  churn rate: training cohort {pct(r['churn_rate_training'])}, production cohort {pct(r['churn_rate_production'])}")
print(f"  training rows whose plan differs between the two joins: {r['rows_changed']} of {r['rows_training']} "
      f"({pct(r['rows_changed'] / r['rows_training'])})")
print(f"  basic plan share: Type 1 join {pct(r['basic_share_type1'])}, as of the snapshot {pct(r['basic_share_as_of'])}")
print(f"  churn rate, basic against pro: Type 1 join {pct(r['churn_if_basic_type1'])} against {pct(r['churn_if_pro_type1'])}; "
      f"as of the snapshot {pct(r['churn_if_basic_as_of'])} against {pct(r['churn_if_pro_as_of'])}")
for name, label in (("type1", "Type 1 join"), ("as_of", "point-in-time join")):
    print(f"  {label:18s} offline AUC {r[name + '_offline_auc']:.3f}, production AUC {r[name + '_production_auc']:.3f}, "
          f"mean predicted churn in production {pct(r[name + '_production_mean_pred'])}, "
          f"coefficient on basic {r[name + '_coef_basic']:.2f}, on usage {r[name + '_coef_usage']:.2f}")
