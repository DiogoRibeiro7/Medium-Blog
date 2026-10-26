"""R2. Redials measured directly from customer IDs.

For queued calls whose ID places at most 50 calls a year: the share whose
ID calls again within w minutes, after an abandonment and after service,
and the share with a call in the w minutes before. The excess after
abandonment over after service is the extra calling that hanging up
causes. Also by the ID's annual volume, and the delay between hanging up
and calling again (saved as quantiles for the simulator).

Only calls that waited in the queue carry their ID, so a callback is seen
only if it queued. The shares are also given with each visible callback
weighted by one over its half hour's queue share, and the queue share that
a callback twenty minutes later would meet is compared between abandoned
and served calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import parse, raw_source, save
from anonbank.data import load_calls
from anonbank.ids import (callback_curves, excess_by_volume, queue_share,
                          redial_delays, with_neighbours)

WINDOWS = np.array([1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240])


def visibility(calls, ids, share) -> dict:
    """How much of the calling the IDs can see."""
    off = calls[calls["queued"] | calls["served"]]
    base = ids[ids["queued"] & (ids["abandoned"] | ids["served"])
               & (ids["volume"] <= 50)]
    midnight = base["t_entry"] - base["vru_entry_s"]
    later = ((base["t_end"] + 1200.0 - midnight) // 1800).astype(int)
    q20 = share.reindex(pd.MultiIndex.from_arrays(
        [base["date"], later])).to_numpy()
    return {"queued_share_of_offered": float(off["queued"].mean()),
            "id_share_queued": float(off.loc[off["queued"], "has_id"]
                                     .mean()),
            "id_share_not_queued": float(off.loc[~off["queued"], "has_id"]
                                         .mean()),
            "queue_share_quartiles": share.quantile([0.25, 0.5, 0.75])
            .tolist(),
            "queue_share_20min_after_abandoned":
                float(np.nanmean(q20[base["abandoned"].to_numpy()])),
            "queue_share_20min_after_served":
                float(np.nanmean(q20[base["served"].to_numpy()]))}


def main() -> None:
    parse(__doc__)
    src = raw_source()
    if src is None:
        raise SystemExit("raw data not found: run python fetch.py")
    calls, _ = load_calls(src)
    ids = with_neighbours(calls)
    share = queue_share(calls)
    curves = {str(v): callback_curves(ids, WINDOWS, v, share)
              .to_dict("records") for v in (10, 50, None)}
    detection = visibility(calls, ids, share)
    by_volume = {str(w): excess_by_volume(ids, w).to_dict("records")
                 for w in (10, 60, 180)}
    delays = redial_delays(ids, 50, 180.0)
    q = np.linspace(0, 1, 201)
    payload = {"windows_min": WINDOWS, "curves": curves,
               "detection": detection,
               "by_volume": by_volume,
               "delay_quantiles_min": np.quantile(delays, q),
               "delay_summary": {
                   "n": int(delays.size),
                   "median": float(np.median(delays)),
                   "mean": float(delays.mean()),
                   "share_within_10": float(np.mean(delays <= 10)),
                   "share_within_30": float(np.mean(delays <= 30)),
                   "share_within_60": float(np.mean(delays <= 60))}}
    for v, rows in curves.items():
        print("max volume", v)
        for r in rows:
            if r["window_min"] in (10, 60, 180):
                print("  ", r)
    for w, rows in by_volume.items():
        print("window", w)
        for r in rows:
            print("  ", {k: (round(x, 3) if isinstance(x, float) else x)
                         for k, x in r.items()})
    print(payload["delay_summary"])
    print(detection)
    save("r2_ids", payload)


if __name__ == "__main__":
    main()
