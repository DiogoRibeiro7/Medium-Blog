"""Print every number quoted in Part 1. The committed output.txt is this script's output.

Run: python history_numbers.py > output.txt
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

import history as h


def section(title: str) -> None:
    print(f"\n== {title}")


def insurance() -> None:
    section("A. Insurance: Taylor-Ashe run-off triangle, chain ladder in SQL, Mack (1993) standard errors")
    tri = h.taylor_ashe()
    factors, ult = h.chain_ladder_sql(tri)
    print(f"  cells observed {len(tri)} of {tri['origin'].max() ** 2}; accident years {tri['origin'].min()}-{tri['origin'].max()}")
    print("  development factors f_k (k = 1..9): " + ", ".join(f"{f:.4f}" for f in factors["f"]))
    se, se_total = h.mack_standard_errors(h.triangle_matrix(tri))
    for r, s in zip(ult.itertuples(), se):
        pct = f"{s / r.reserve:5.1%}" if r.reserve > 0 else "    -"
        print(f"  accident year {r.origin:2d}: latest {r.latest:12,.0f} at development year {r.latest_dev:2d}; "
              f"ultimate {r.ultimate:12,.0f}; reserve {r.reserve:11,.0f}; Mack s.e. {s:10,.0f} ({pct})")
    total_res, total_ult, total_latest = ult["reserve"].sum(), ult["ultimate"].sum(), ult["latest"].sum()
    print(f"  total: latest {total_latest:,.0f}; ultimate {total_ult:,.0f}; reserve {total_res:,.0f} "
          f"({total_res / total_ult:.1%} of the ultimate still to come)")
    print(f"  total reserve Mack s.e. {se_total:,.0f} ({se_total / total_res:.1%} of the reserve)")
    last = ult.iloc[-1]
    print(f"  latest accident year: known {last.latest:,.0f} is {last.latest / last.ultimate:.1%} of its projected ultimate")
    print("  Mack (1993), Tables 2-3: total reserve 18,681 (in 1000s), s.e. 13%; by year 95, 470, 710, 985, 1,419, "
          "2,178, 3,920, 4,279, 4,626 with s.e. 80%, 26%, 19%, 27%, 29%, 26%, 22%, 23%, 29%")


def epidemiology() -> None:
    section("B. Epidemiology: HUS cases of the STEC O104:H4 outbreak, Germany 2011")
    cases = h.hus_cases()
    q = cases["delay"].quantile([0.25, 0.5, 0.75])
    print(f"  cases {len(cases)}; hospitalized {cases['hospitalized'].min().date()} to {cases['hospitalized'].max().date()}; "
          f"reported {cases['reported'].min().date()} to {cases['reported'].max().date()}")
    print(f"  delay hospitalization -> report (days): mean {cases['delay'].mean():.1f}; quartiles {q[0.25]:.0f}, "
          f"{q[0.5]:.0f}, {q[0.75]:.0f}; at the 15-day cap {int((cases['delay'] == h.MAX_DELAY).sum())} "
          f"({(cases['delay'] == h.MAX_DELAY).mean():.1%}); within 3 days {(cases['delay'] <= 3).mean():.1%}")
    weeks = cases.assign(week=cases["hospitalized"].dt.to_period("W-SUN"))
    for wk, g in weeks.groupby("week"):
        if len(g) >= 15:
            print(f"  hospitalized in week {wk.start_time.date()} to {wk.end_time.date()}: {len(g):3d} cases, "
                  f"median delay {g['delay'].median():.1f} days")
    final = h.final_counts(cases)
    print(f"  final epidemic curve: peak {final.idxmax().date()} with {final.max()} cases")
    for d in ("2011-05-24", "2011-05-28", "2011-05-31", "2011-06-04"):
        D = pd.Timestamp(d)
        known = h.known_by(cases, D)
        last7 = pd.date_range(D - pd.Timedelta(days=6), D)
        k7, f7 = int(known[last7].sum()), int(final.reindex(last7, fill_value=0).sum())
        print(f"  as known on {d}: highest day {known.idxmax().date()} with {known.max()}; "
              f"last 7 days {k7} known against {f7} eventually ({k7 / f7:.0%})")
    print("  chain-ladder nowcast (time-homogeneous delays), split at estimated completeness 50%:")
    for d in ("2011-05-28", "2011-05-31", "2011-06-02", "2011-06-05", "2011-06-08"):
        D = pd.Timestamp(d)
        nc = h.nowcast_chain_ladder(cases, D)
        fin = final.reindex(nc.index, fill_value=0)
        mature = nc[nc["completeness"] >= 0.5].index[-7:]
        young = nc[nc["completeness"] < 0.5].index
        print(f"    {d}: 7 latest days at >= 50%: known {nc.loc[mature, 'known'].sum():.0f}, nowcast "
              f"{nc.loc[mature, 'nowcast'].sum():.0f}, final {fin[mature].sum()}; "
              f"{len(young)} newest days below 50%: known {nc.loc[young, 'known'].sum():.0f}, nowcast "
              f"{nc.loc[young, 'nowcast'].sum():.0f}, final {fin[young].sum()}")


def temporal_database() -> None:
    section("C. Temporal databases: one bitemporal price table")
    con = h.bitemporal_connection()
    print(f"  rows {len(h.BITEMPORAL_ROWS)}; orders {len(h.ORDERS)}")
    for valid, known, label in ((date(2026, 2, 20), date(2026, 2, 20), "price on 20 Feb, as known on 20 Feb"),
                                (date(2026, 2, 20), date(2026, 4, 1), "price on 20 Feb, as known on 1 Apr"),
                                (date(2026, 3, 5), date(2026, 3, 5), "price on 5 Mar, as known on 5 Mar"),
                                (date(2026, 3, 5), date(2026, 4, 1), "price on 5 Mar, as known on 1 Apr")):
        print(f"  {label}: {h.price_as_of(con, valid, known):.2f}")
    inv = h.invoices(con, today=date(2026, 4, 1))
    for r in inv.itertuples():
        print(f"  order {r.order_id} ({pd.Timestamp(r.order_date).date()}, {r.quantity} units): billed {float(r.billed_price):.2f}, "
              f"due {float(r.due_price):.2f}, restatement {float(r.restatement):+.2f}")
    billed = float((inv["quantity"] * inv["billed_price"].astype(float)).sum())
    due = float((inv["quantity"] * inv["due_price"].astype(float)).sum())
    print(f"  total billed {billed:,.2f}; total due on 1 Apr knowledge {due:,.2f}; difference {due - billed:+,.2f}")


def warehouse() -> None:
    section("D. Data warehouses: Type 1 against Type 2 dimensions, 2024 revenue by region")
    wh = h.simulate_warehouse()
    moved = wh.customers.groupby("customer_id").size()
    print(f"  customers {wh.customers['customer_id'].nunique()}; dimension rows {len(wh.customers)}; "
          f"customers who moved at least once {(moved > 1).sum()} ({(moved > 1).mean():.1%}); sales {len(wh.sales)}")
    rev = h.regional_revenue_2024(wh)
    for region, r in rev.iterrows():
        print(f"  {region:5s}: Type 2 {r.type2:11,.2f}; Type 1 run 2025-01-05 {r.type1_run_2025_01_05:11,.2f} "
              f"({r.type1_run_2025_01_05 / r.type2 - 1:+.1%}); Type 1 rerun 2026-01-05 {r.type1_run_2026_01_05:11,.2f} "
              f"({r.type1_run_2026_01_05 / r.type2 - 1:+.1%}); change between the two Type 1 runs "
              f"{r.type1_run_2026_01_05 - r.type1_run_2025_01_05:+,.2f}")
    print(f"  total 2024 revenue {rev['type2'].sum():,.2f} under all three (Type 1 moves revenue between regions, not in or out)")
    moved_rev = (rev["type1_run_2026_01_05"] - rev["type2"]).abs().sum() / 2
    print(f"  revenue attributed to a different region by the 2026 Type 1 rerun: {moved_rev:,.2f} "
          f"({moved_rev / rev['type2'].sum():.1%} of the total)")


def streaming() -> None:
    section("E. Stream processing: event time, processing time, watermark, triggers")
    print(f"  {len(h.EVENTS)} elements, total {sum(v for v, _, _ in h.EVENTS)}; two-minute event-time windows; "
          f"watermark = largest event time seen - {h.WATERMARK_SLACK} s; allowed lateness {h.ALLOWED_LATENESS} s")
    late = [(v, et, at) for v, et, at in h.EVENTS if at - et > 60]
    print("  out of order by more than a minute: " + ", ".join(f"{v} (event {et} s, arrival {at} s)" for v, et, at in late))
    print("  true sums by event-time window: " + str(h.true_window_sums()))
    print("  sums by processing-time window (arrival): " + str(h.processing_time_sums()))
    panes, dropped = h.stream()
    for p in panes.itertuples():
        when = f"at {p.time:.0f} s" if p.time == p.time else "at end of input"
        retract = f"retract {p.retraction:+.0f} then " if p.retraction == p.retraction else ""
        print(f"  pane window {p.window} {p.kind:7s} {when}: discarding {p.discarding}; accumulating {p.accumulating}; "
              f"accumulating & retracting {retract}{p.accumulating}")
    print("  dropped as too late: " + ", ".join(f"{v} (event {et} s, arrival {at} s)" for v, et, at in dropped))
    path = dict(h.watermark_path())
    for v, et, at in dropped:
        before = max(w for a, w in path.items() if a < at)
        end = (h.window_of(et) + 1) * h.WINDOW
        print(f"  watermark when the {v} arrived: {before} s; its window ends at {end} s and stops accepting "
              f"late data at {end + h.ALLOWED_LATENESS} s")
    on_time0 = panes[(panes["window"] == 0) & (panes["kind"] == "on time")]["accumulating"].iloc[0]
    accepted0 = panes[panes["window"] == 0]["discarding"].sum()
    print(f"  window 0 on-time pane {on_time0}: {on_time0 / h.true_window_sums()[0]:.0%} of the window's true sum, "
          f"{on_time0 / accepted0:.0%} of what is eventually accepted")
    sums = {}
    for mode in ("discarding", "accumulating"):
        sums[mode] = panes.groupby("window")[mode].sum().to_dict()
    retr = panes.groupby("window").apply(lambda g: g["accumulating"].sum() + g["retraction"].fillna(0).sum(),
                                         include_groups=False).to_dict()
    print(f"  downstream sum of panes per window: discarding {sums['discarding']}; accumulating {sums['accumulating']}; "
          f"accumulating & retracting {retr}")
    on_time = panes[panes["kind"] != "late"].groupby("window")["accumulating"].first().to_dict()
    print(f"  speed layer (on-time panes only) {on_time}; batch layer recomputed from the full log {h.true_window_sums()}")


def main() -> None:
    print("Part 1 numbers: a history of late data in four fields")
    insurance()
    epidemiology()
    temporal_database()
    warehouse()
    streaming()


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    np.set_printoptions(linewidth=200)
    main()
