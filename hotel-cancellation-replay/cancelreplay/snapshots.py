"""Point-in-time views of the booking history.

A decision for arrival date d is taken at the start of day t = d - h. At that
moment the hotel can see a booking only if it was entered before t and has not
already been released (cancelled) before t. The label is whether the booking
releases its room on or after t: a later cancellation or a no-show.

Each booking therefore has exactly one snapshot for a given horizon h, at its
own d - h. Training rows and serving rows are built by the same function, so
the population the model learns from is the population it is asked about.
"""

from __future__ import annotations

import pandas as pd


def decision_time(arrival: pd.Series, horizon: int) -> pd.Series:
    return arrival - pd.Timedelta(days=horizon)


def visible_at_horizon(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Bookings the hotel can see at d - h: entered before t, not yet released."""
    t = decision_time(df["arrival"], horizon)
    entered = df["booked"] < t
    released_before = df["release_date"].notna() & (df["release_date"] < t)
    return entered & ~released_before


def snapshot_rows(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """One row per booking visible at its own decision time, labelled."""
    rows = df.loc[visible_at_horizon(df, horizon)].copy()
    rows["label"] = rows["is_canceled"].astype(int)
    rows["decision_time"] = decision_time(rows["arrival"], horizon)
    return rows


# --- Training sets -----------------------------------------------------------
# All three take the retraining time T and return labelled rows whose features
# are read from the booking record. They differ in which bookings they include
# and what the label means.


def train_point_in_time(df: pd.DataFrame, horizon: int, T: pd.Timestamp) -> pd.DataFrame:
    """Snapshot rows whose arrival has passed by T, so the label is final."""
    rows = snapshot_rows(df, horizon)
    return rows.loc[rows["arrival"] < T]


def train_all_bookings(df: pd.DataFrame, horizon: int, T: pd.Timestamp) -> pd.DataFrame:
    """Every booking that arrived before T, labelled 'ever cancelled'.

    This is the usual framing of the problem. It is not conditioned on the
    booking still being on the books at d - h, so it estimates P(cancel) and
    not P(cancel after t | still active at t).
    """
    rows = df.loc[df["arrival"] < T].copy()
    rows["label"] = rows["is_canceled"].astype(int)
    return rows


def train_known_status(df: pd.DataFrame, horizon: int, T: pd.Timestamp) -> pd.DataFrame:
    """'Everything whose status is known at T'.

    Adds bookings for future arrivals that have already cancelled (label 1)
    and drops future bookings whose outcome is still open, the construction
    the data authors report having used and then abandoned in production.
    """
    past = train_all_bookings(df, horizon, T)
    future = df.loc[(df["arrival"] >= T) & df["release_date"].notna() & (df["release_date"] < T)].copy()
    future["label"] = 1
    return pd.concat([past, future])


TRAINING_SETS = {
    "point_in_time": train_point_in_time,
    "all_bookings": train_all_bookings,
    "known_status": train_known_status,
}
