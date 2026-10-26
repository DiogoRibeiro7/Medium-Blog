import numpy as np
import pandas as pd
import pytest

from cancelreplay import snapshots as S


def bookings():
    """Five bookings for one arrival night, each probing one edge of visibility."""
    arrival = pd.Timestamp("2017-03-31")
    rows = [
        # booked 40 days out, never cancelled
        dict(booked=arrival - pd.Timedelta(days=40), release_date=pd.NaT, is_canceled=0),
        # booked 40 days out, cancelled 35 days out: gone before a 30-day batch
        dict(booked=arrival - pd.Timedelta(days=40), release_date=arrival - pd.Timedelta(days=35), is_canceled=1),
        # booked 40 days out, cancelled 10 days out: visible, label 1
        dict(booked=arrival - pd.Timedelta(days=40), release_date=arrival - pd.Timedelta(days=10), is_canceled=1),
        # booked on the batch day itself: not yet in the system
        dict(booked=arrival - pd.Timedelta(days=30), release_date=pd.NaT, is_canceled=0),
        # cancelled on the batch day: still visible at the start of the day
        dict(booked=arrival - pd.Timedelta(days=50), release_date=arrival - pd.Timedelta(days=30), is_canceled=1),
    ]
    df = pd.DataFrame(rows)
    df["arrival"] = arrival
    return df


def test_visibility_edges():
    rows = S.snapshot_rows(bookings(), 30)
    assert list(rows.index) == [0, 2, 4]
    assert list(rows["label"]) == [0, 1, 1]


def test_nothing_visible_was_entered_or_released_before_its_batch():
    rng = np.random.default_rng(0)
    n = 5000
    arrival = pd.Timestamp("2017-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    lead = rng.integers(0, 300, n)
    booked = arrival - pd.to_timedelta(lead, unit="D")
    cancel = rng.random(n) < 0.4
    cancel_lead = (rng.random(n) * lead).astype(int)
    release = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    release[cancel] = (arrival - pd.to_timedelta(cancel_lead, unit="D"))[cancel]
    df = pd.DataFrame(dict(arrival=arrival, booked=booked, release_date=release, is_canceled=cancel.astype(int)))
    for h in (1, 7, 30, 60):
        rows = S.snapshot_rows(df, h)
        t = rows["arrival"] - pd.Timedelta(days=h)
        assert (rows["booked"] < t).all()
        released = rows["release_date"].notna()
        assert (rows.loc[released, "release_date"] >= t[released]).all()
        assert (rows.loc[rows["label"] == 1, "release_date"].notna()).all()


def test_training_rows_have_final_labels_and_match_serving_rows():
    df = bookings()
    T = pd.Timestamp("2017-04-01")
    train = S.train_point_in_time(df, 30, T)
    serve = S.snapshot_rows(df, 30)
    assert (train["arrival"] < T).all()
    pd.testing.assert_frame_equal(train, serve)


def test_known_status_adds_only_cancelled_future_bookings():
    df = bookings()
    T = pd.Timestamp("2017-03-20")  # before the arrival, after two cancellations
    rows = S.train_known_status(df, 30, T)
    assert len(rows) == 2 and (rows["label"] == 1).all()


@pytest.mark.parametrize("name", list(S.TRAINING_SETS))
def test_training_sets_never_use_unfinished_arrivals_as_negatives(name):
    df = bookings()
    T = pd.Timestamp("2017-03-20")
    rows = S.TRAINING_SETS[name](df, 30, T)
    open_ = rows["arrival"] >= T
    assert (rows.loc[open_, "label"] == 1).all()
