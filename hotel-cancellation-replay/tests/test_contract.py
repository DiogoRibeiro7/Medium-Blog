"""Checks on the real data and on the feature registry."""

import pytest

from cancelreplay import data
from cancelreplay import features as F


@pytest.fixture(scope="module")
def bookings():
    df, report = data.load()
    return df, report


def test_checksum_and_shape(bookings):
    df, report = bookings
    assert report.rows_read == 119390
    assert report.checks["city hotel soft-opening rows dropped"] > 0
    assert len(df) == report.rows_kept


def test_every_registered_feature_exists(bookings):
    df, _ = bookings
    missing = [f.name for f in F.REGISTRY if f.name not in df.columns]
    assert not missing


def test_point_in_time_set_reads_only_booking_time_columns():
    F.assert_point_in_time(F.POINT_IN_TIME)
    with pytest.raises(ValueError):
        F.assert_point_in_time(F.AS_RECORDED)


def test_parking_column_cannot_be_a_booking_time_value(bookings):
    """Thousands of bookings with a parking space and not one cancellation."""
    df, _ = bookings
    with_parking = df.loc[df["required_car_parking_spaces"] > 0]
    assert len(with_parking) > 5000
    assert with_parking["is_canceled"].sum() == 0
    assert next(f for f in F.REGISTRY if f.name == "required_car_parking_spaces").known == F.LATER
