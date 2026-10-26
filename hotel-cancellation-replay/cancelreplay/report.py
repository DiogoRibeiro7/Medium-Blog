"""Run everything and write every number the article quotes."""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd

from . import data
from . import experiments as E
from .paths import FIGURES, RESULTS
from .replay import Replayer


def _fmt(x, digits=3):
    if isinstance(x, (float, np.floating)):
        return f"{x:.{digits}f}"
    return str(x)


def run(make_figures: bool = True) -> dict:
    warnings.filterwarnings("ignore", category=UserWarning)
    started = time.time()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    df, contract = data.load()
    rp = Replayer(df)

    table, logs = E.run_variants(rp)
    hotels = E.by_hotel(logs)
    walk = E.walk_cost_sensitivity(rp)
    timing = E.cancellation_timing(df)
    leaks = E.leak_smoke_test(df)
    aucs = E.random_split_auc(df, rp)
    blocks = E.block_structure(rp)
    drift = E.drift_by_quarter(rp)
    non_refund = E.non_refund_composition(rp)
    tuned = {f"{k[0]} | {k[1]} | h={k[3]}": v for k, v in rp._tuned.items()}

    results = dict(
        contract=contract.checks | {"rows_read": contract.rows_read, "rows_kept": contract.rows_kept},
        variants=table.reset_index().to_dict(orient="records"),
        by_hotel=hotels.to_dict(orient="records"),
        walk_cost_sensitivity=walk.to_dict(orient="records"),
        timing=timing,
        leak_smoke_test=leaks,
        auc_by_question=aucs,
        blocks=blocks,
        drift_by_quarter=drift,
        non_refund=non_refund,
        tuned_parameters=tuned,
        settings=dict(walk_multiple=E.WALK_MULTIPLE, bootstrap_draws=E.BOOTSTRAP_DRAWS,
                      live_start=str(E.LIVE_START.date()), live_end=str(E.LIVE_END.date())),
    )
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (RESULTS / "summary.txt").write_text(summary_text(results))

    if make_figures:
        from . import figures
        figures.draw_all(results, logs, rp)

    print(summary_text(results))
    print(f"\nfinished in {time.time() - started:.0f}s")
    return results


def summary_text(r: dict) -> str:
    L = []
    L.append("HOTEL CANCELLATION REPLAY - every number in the article")
    L.append("=" * 72)
    L.append("\nData contract")
    for k, v in r["contract"].items():
        L.append(f"  {k}: {v}")

    t = r["timing"]
    L.append("\nCancellation timing")
    L.append(f"  bookings kept: {t['bookings']}, released (cancelled or no-show): {t['cancelled_or_no_show']} "
             f"({t['ever_cancel_rate']:.3f}), of which no-shows: {t['no_shows']}")
    L.append(f"  cancellations: median {t['median_days_before_arrival']:.0f} days before arrival, "
             f"quartiles {t['quartiles_days_before'][0]:.0f} and {t['quartiles_days_before'][1]:.0f}")
    for h, v in t["by_horizon"].items():
        L.append(f"  h={h:>3}: visible {v['visible_share_of_bookings']:.3f} of bookings, "
                 f"{v['visible_per_night']:.1f} per hotel-night; late-release rate {v['late_release_rate']:.3f}; "
                 f"share of all releases still ahead {v['share_of_all_releases_still_ahead']:.3f}")

    L.append("\nLeak smoke test (column values with >= 500 rows and a 0% or 100% cancel rate)")
    for k, v in r["leak_smoke_test"].items():
        L.append(f"  {k}: {v}")

    nr = r["non_refund"]
    L.append(f"  non-refund bookings: {nr['rows']}, in blocks of 2+: {nr['in_blocks_of_two_or_more']:.3f}, "
             f"of 10+: {nr['in_blocks_of_ten_or_more']:.3f}, segments: "
             + ", ".join(f"{k} {v:.3f}" for k, v in nr["segment_shares"].items()))

    L.append("\nLate-release rate among bookings visible 30 days out, by arrival quarter")
    for hotel, quarters in r["drift_by_quarter"].items():
        L.append(f"  {hotel}: " + ", ".join(f"{q} {v['rate']:.3f}" for q, v in quarters.items()))

    L.append("\nAUC of boosting by question (random 5-fold CV unless stated)")
    for k, v in r["auc_by_question"].items():
        L.append(f"  {k}: " + ", ".join(f"{h} {x:.3f}" for h, x in v.items()))

    b = r["blocks"]
    L.append("\nBlocks (same hotel, arrival, booking date, agent, company, segment)")
    for k, v in b.items():
        L.append(f"  {k}: {_fmt(v)}")

    L.append(f"\nReplay: live arrivals {r['settings']['live_start']} to {r['settings']['live_end']}, "
             f"walk cost = {r['settings']['walk_multiple']:.0f} x nightly rate, "
             f"{r['settings']['bootstrap_draws']} weekly-block bootstrap draws")
    L.append("  key | share of oracle | diff vs reference [95% CI] | resold/night | walked/night | "
             "nights with walks | profit/night EUR | AUC | log loss | predicted/realised")
    for row in r["variants"]:
        L.append(
            f"  {row['key']:<15} {row['share_of_oracle']:7.3f}  {row['diff_vs_reference']:+.3f} "
            f"[{row['diff_lo']:+.3f}, {row['diff_hi']:+.3f}]  {row['resold_per_night']:5.2f}  "
            f"{row['walked_per_night']:5.2f}  {row['nights_with_walks']:.3f}  {row['profit_per_night']:8.1f}  "
            f"{row['auc']:.3f}  {row['log_loss']:.4f}  {row['predicted_over_realised']:.3f}   ({row['label']})"
        )
    ref = next(x for x in r["variants"] if x["key"] == "reference")
    L.append(f"  reference: oracle per night EUR {ref['oracle_per_night']:.1f}, "
             f"released per night {ref['released_per_night']:.2f}, nights {ref['nights']}, rows {ref['rows']}")
    for row in r["variants"]:
        if row["key"].startswith("horizon"):
            L.append(f"  {row['key']}: oracle per night EUR {row['oracle_per_night']:.1f}, "
                     f"released per night {row['released_per_night']:.2f}, profit per night EUR {row['profit_per_night']:.1f}")

    L.append("\nBy hotel: share of oracle | resold/night | walked/night | profit/night")
    for row in r["by_hotel"]:
        L.append(f"  {row['key']:<15} {row['hotel']:<13} {row['share_of_oracle']:7.3f} "
                 f"{row['resold_per_night']:5.2f} {row['walked_per_night']:5.2f} {row['profit_per_night']:8.1f}")

    L.append("\nWalk-cost sensitivity: share of oracle, difference from the reference [95% CI], walked/night")
    for row in r["walk_cost_sensitivity"]:
        L.append(f"  x{row['walk_multiple']:.0f} {row['key']:<15} {row['share_of_oracle']:7.3f}  "
                 f"{row['diff_vs_reference']:+.3f} [{row['diff_lo']:+.3f}, {row['diff_hi']:+.3f}]  "
                 f"{row['walked_per_night']:.2f}")

    L.append("\nTuned boosting parameters")
    for k, v in r["tuned_parameters"].items():
        L.append(f"  {k}: {v}")
    return "\n".join(L) + "\n"
