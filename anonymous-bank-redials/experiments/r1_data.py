"""R1. Read the calls, build the half-hourly table, describe the centre.

Needs the raw archive (run ``python fetch.py`` first, or set
``ANONBANK_DATA`` to the archive or to a folder of extracted files). Writes
``data/halfhour.csv.gz``, the aggregate table every later step uses.
"""

from __future__ import annotations

import numpy as np

from common import DATA, TABLE, parse, raw_source, save
from anonbank.data import load_calls
from anonbank.ids import flag_identified_redials, with_neighbours
from anonbank.table import halfhour_table


def main() -> None:
    parse(__doc__)
    src = raw_source()
    if src is None:
        raise SystemExit("raw data not found: run python fetch.py")
    calls, info = load_calls(src)
    flag = flag_identified_redials(calls, 60.0)
    tab = halfhour_table(calls, flag)
    DATA.mkdir(exist_ok=True)
    tab.to_csv(TABLE, index=False, float_format="%.6g")
    ids = with_neighbours(calls)
    vol = ids.groupby("customer_id").size()
    queued = calls[calls["queued"]]
    offered = tab["offered"].sum()
    by_interval = tab.groupby("interval")[["offered", "abandoned", "agents",
                                           "answered_20s"]].mean()
    payload = {
        **info,
        "outcomes": calls["outcome"].value_counts().to_dict(),
        "share_with_id": float(calls["has_id"].mean()),
        "share_with_id_queued": float(queued["has_id"].mean()),
        "share_with_id_abandoned": float(calls.loc[calls["abandoned"],
                                                   "has_id"].mean()),
        "distinct_ids": int(vol.size),
        "id_calls_quantiles": vol.quantile([0.5, 0.9, 0.99]).to_dict(),
        "ids_over_200_calls": int((vol > 200).sum()),
        "share_of_id_calls_from_ids_over_200": float(
            vol[vol > 200].sum() / vol.sum()),
        "working_days_kept": int(tab["day"].nunique()),
        "offered_per_day": float(offered / tab["day"].nunique()),
        "abandon_share": float(tab["abandoned"].sum() / offered),
        "service_level_20s": float(tab["answered_20s"].sum() / offered),
        "mean_agents": float(tab["agents"].mean()),
        "mean_offered_per_interval": float(tab["offered"].mean()),
        "recorded_handle_time": float(tab["service_total"].sum()
                                      / tab["answered"].sum()),
        "mean_wait_per_offered": float(tab["wait_total"].sum() / offered),
        "identified_redial_share_of_offered": float(
            tab["id_redials"].sum() / offered),
        "by_interval": by_interval.round(3).to_dict(orient="list"),
    }
    print({k: v for k, v in payload.items() if k != "by_interval"})
    save("r1_data", payload)


if __name__ == "__main__":
    main()
