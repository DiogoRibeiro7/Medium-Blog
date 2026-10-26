"""What the customer IDs say about callers who hang up and call again.

Only calls that carry a customer ID can be followed. Some IDs place
hundreds or thousands of calls a year, several at a time, and behave like
shared or business lines rather than people; results are therefore reported
by the ID's annual number of calls.

For a call that waited in the queue, "calling again within w" means that
the next call from the same ID entered the system between the end of this
call and w seconds later. The end of an abandoned call is the moment the
caller hung up; the end of a served call is the end of service. The share
calling again after a served call is the baseline: people call back for
reasons that have nothing to do with hanging up. A third reference is the
share whose previous call from the same ID began within w seconds before
this call's entry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VOLUME_BINS = (0, 10, 50, 200, np.inf)
VOLUME_LABELS = ("1-10", "11-50", "51-200", "over 200")


def with_neighbours(calls: pd.DataFrame) -> pd.DataFrame:
    """Calls with an ID, each with the previous and next call of its ID."""
    ids = calls[calls["has_id"]].sort_values(["customer_id", "t_entry"]) \
        .copy()
    by = ids.groupby("customer_id")
    ids["volume"] = by["t_entry"].transform("size")
    ids["volume_class"] = pd.cut(ids["volume"], VOLUME_BINS,
                                 labels=VOLUME_LABELS)
    ids["t_end"] = (ids["t_entry"] + ids["vru_time"] + ids["q_time"]
                    + np.where(ids["served"], ids["ser_time"], 0))
    ids["next_entry"] = by["t_entry"].shift(-1)
    ids["prev_entry"] = by["t_entry"].shift(1)
    ids["prev_end"] = by["t_end"].shift(1)
    ids["prev_abandoned"] = by["abandoned"].shift(1).astype("boolean") \
        .fillna(False).astype(bool)
    ids["after"] = ids["next_entry"] - ids["t_end"]
    ids["before"] = ids["t_entry"] - ids["prev_entry"]
    return ids


def callback_curves(ids: pd.DataFrame, windows_min: np.ndarray,
                    max_volume: int | None = 50,
                    share: pd.Series | None = None) -> pd.DataFrame:
    """Share calling again within each window, after abandonment and after
    service, and the share with a previous call in the same window.

    With ``share`` (the queue share by date and half hour, see
    ``queue_share``), also the same shares with each visible next or
    previous call weighted by one over its half hour's queue share. The
    weighting counts only the first visible call, so it overstates the
    share once callers place several calls inside the window; it is meant
    for windows of an hour or less.
    """
    q = ids[ids["queued"] & (ids["abandoned"] | ids["served"])]
    if max_volume is not None:
        q = q[q["volume"] <= max_volume]
    if share is not None:
        wt = pd.Series(detection_weights(ids, share), index=ids.index)
        by = wt.groupby(ids["customer_id"])
        w_next = by.shift(-1).reindex(q.index).fillna(0.0)
        w_prev = by.shift(1).reindex(q.index).fillna(0.0)
    rows = []
    for w in windows_min:
        sec = 60.0 * w
        again = (q["after"] >= 0) & (q["after"] <= sec)
        prior = (q["before"] >= 0) & (q["before"] <= sec)
        for name, mask in (("abandoned", q["abandoned"]),
                           ("served", q["served"])):
            row = {"window_min": w, "group": name,
                   "again": float(again[mask].mean()),
                   "prior": float(prior[mask].mean()),
                   "n": int(mask.sum())}
            if share is not None:
                row["again_weighted"] = float((again * w_next)[mask].mean())
                row["prior_weighted"] = float((prior * w_prev)[mask].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def excess_by_volume(ids: pd.DataFrame, window_min: float) -> pd.DataFrame:
    """Calling again within the window, abandoned minus served, by volume."""
    q = ids[ids["queued"] & (ids["abandoned"] | ids["served"])].copy()
    sec = 60.0 * window_min
    q["again"] = (q["after"] >= 0) & (q["after"] <= sec)
    q["overlap"] = q["after"] < 0
    g = q.groupby(["volume_class", "abandoned"], observed=True)
    out = g.agg(again=("again", "mean"), overlap=("overlap", "mean"),
                n=("again", "size")).unstack("abandoned")
    out.columns = [f"{a}_{'abandoned' if b else 'served'}"
                   for a, b in out.columns]
    out["excess"] = out["again_abandoned"] - out["again_served"]
    return out.reset_index()


def redial_delays(ids: pd.DataFrame, max_volume: int | None = 50,
                  horizon_min: float = 180.0) -> np.ndarray:
    """Minutes from hanging up to the next call, for abandoners who call
    again within the horizon."""
    q = ids[ids["abandoned"]]
    if max_volume is not None:
        q = q[q["volume"] <= max_volume]
    gap = q["after"]
    return (gap[(gap >= 0) & (gap <= 60.0 * horizon_min)] / 60.0).to_numpy()


def flag_identified_redials(calls: pd.DataFrame,
                            window_min: float = 60.0) -> pd.Series:
    """True for calls whose previous call from the same ID was abandoned
    and ended at most ``window_min`` minutes before this one entered."""
    ids = with_neighbours(calls)
    gap = ids["t_entry"] - ids["prev_end"]
    flag = ids["prev_abandoned"] & (gap >= 0) & (gap <= 60.0 * window_min)
    out = pd.Series(False, index=calls.index)
    out.loc[flag.index] = flag.to_numpy()
    return out


# --------------------------------------------------------------------------
# Correcting for calls that cannot be seen
#
# The recording system stored the customer ID only for calls that waited in
# the queue (the data documentation calls it a system bug). A call from an
# identified customer that was answered at once carries ID zero, so a
# callback is visible only if it queued. The chance of queueing changes
# with the load, and the load is higher after an abandonment than after a
# service, so raw callback shares see more of the callbacks after an
# abandonment. Weighting each visible further call by one over the share
# of offered calls that queued in its half hour undoes this, under the
# assumption that within a half hour identified customers queue as often
# as everyone else.


def queue_share(calls: pd.DataFrame) -> pd.Series:
    """Share of offered calls that waited in the queue, by date and half
    hour of the moment they were offered."""
    off = calls[calls["queued"] | calls["served"]]
    half = (off["vru_exit_s"] // 1800).astype(int).rename("half")
    return off.groupby([off["date"], half])["queued"].mean()


def detection_weights(ids: pd.DataFrame, share: pd.Series) -> np.ndarray:
    """One over the queue share of each call's half hour."""
    key = pd.MultiIndex.from_arrays(
        [ids["date"], (ids["vru_exit_s"] // 1800).astype(int)])
    q = share.reindex(key).to_numpy()
    q = np.where(np.isfinite(q) & (q > 0), q, float(share.mean()))
    return 1.0 / q
