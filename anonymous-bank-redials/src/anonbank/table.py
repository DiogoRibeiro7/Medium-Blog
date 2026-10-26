"""The half-hourly table a contact-centre platform would export.

Working days only (Sunday to Thursday in Israel), from 07:00 to 24:00, in
34 half-hour intervals. A call is counted in the interval in which it left
the voice-response unit and was offered to the agents' queue. For each day
and interval:

* ``offered``: calls that waited in the queue or reached an agent;
* ``abandoned``: calls that hung up while queued;
* ``answered_20s``: calls answered with a queue time of at most 20 s;
* ``wait_total``: seconds spent queuing by the offered calls;
* ``service_total`` and ``answered``: for the mean handle time;
* ``agents``: distinct agents with a service overlapping the interval, a
  lower bound on the agents on shift;
* ``id_redials``: offered calls whose previous call from the same customer
  ID was abandoned less than an hour earlier.

Days with fewer than half the median day's offered calls (holidays, partial
days) are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WORKING_DOWS = (6, 0, 1, 2, 3)      # Sunday .. Thursday (Monday = 0)
FIRST_HALF, LAST_HALF = 14, 47      # 07:00 .. 23:30
N_INTERVALS = LAST_HALF - FIRST_HALF + 1
SL_THRESHOLD = 20.0


def halfhour_table(calls: pd.DataFrame,
                   id_redial: pd.Series | None = None) -> pd.DataFrame:
    wk = calls[calls["dow"].isin(WORKING_DOWS)].copy()
    wk["half"] = (wk["vru_exit_s"] // 1800).astype(int)
    if id_redial is not None:
        wk["id_redial"] = id_redial.loc[wk.index].to_numpy()
    else:
        wk["id_redial"] = False
    offered = wk[(wk["queued"] | wk["served"])
                 & wk["half"].between(FIRST_HALF, LAST_HALF)]
    offered = offered.assign(
        answered_20s=offered["served"] & (offered["q_time"] <= SL_THRESHOLD),
        service=np.where(offered["served"], offered["ser_time"], 0.0))
    g = offered.groupby(["date", "half"])
    tab = pd.DataFrame({
        "offered": g.size(),
        "abandoned": g["abandoned"].sum(),
        "answered": g["served"].sum(),
        "answered_20s": g["answered_20s"].sum(),
        "wait_total": g["q_time"].sum(),
        "service_total": g["service"].sum(),
        "id_redials": g["id_redial"].sum(),
    })
    days = sorted(offered["date"].unique())
    grid = pd.MultiIndex.from_product(
        [days, range(FIRST_HALF, LAST_HALF + 1)], names=["date", "half"])
    tab = tab.reindex(grid).fillna(0.0)
    tab["agents"] = _agents(wk, grid)
    tab = tab.reset_index()
    tab["dow"] = tab["date"].dt.dayofweek
    tab["interval"] = tab["half"] - FIRST_HALF
    totals = tab.groupby("date")["offered"].sum()
    keep = totals[totals >= 0.5 * totals.median()].index
    tab = tab[tab["date"].isin(keep)].reset_index(drop=True)
    tab["day"] = tab["date"].rank(method="dense").astype(int) - 1
    for c in ("offered", "abandoned", "answered", "answered_20s",
              "id_redials", "agents"):
        tab[c] = tab[c].astype(int)
    return tab


def _agents(wk: pd.DataFrame, grid: pd.MultiIndex) -> np.ndarray:
    sv = wk[wk["served"] & (wk["ser_exit_s"] >= wk["ser_start_s"])]
    h0 = (sv["ser_start_s"] // 1800).astype(int).to_numpy()
    h1 = (sv["ser_exit_s"] // 1800).astype(int).to_numpy()
    span = h1 - h0 + 1
    rows = pd.DataFrame({
        "date": np.repeat(sv["date"].to_numpy(), span),
        "server": np.repeat(sv["server"].to_numpy(), span),
        "half": np.concatenate([np.arange(a, b + 1) for a, b in zip(h0, h1)])
        if len(sv) else np.array([], int)})
    counts = rows.drop_duplicates().groupby(["date", "half"]).size()
    return counts.reindex(grid).fillna(0).to_numpy().astype(int)


def day_matrix(tab: pd.DataFrame, column: str) -> np.ndarray:
    """(days, intervals) array of one column."""
    return tab.pivot(index="day", columns="interval",
                     values=column).to_numpy(float)
