"""Load the hotel booking data and enforce the contract the rest of the code relies on.

The data are the two Portuguese hotel datasets published by Antonio, de Almeida
and Nunes (2019), "Hotel booking demand datasets", Data in Brief 22, 41-49,
doi:10.1016/j.dib.2018.11.126, as redistributed by the TidyTuesday project.
"""

from __future__ import annotations

import gzip
import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .paths import DATA_FILE

SOURCE_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/"
    "data/2020/2020-02-11/hotels.csv"
)
# SHA-256 of the uncompressed CSV as downloaded on 2026-10-26.
SOURCE_SHA256 = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"

MONTHS = {
    name: number
    for number, name in enumerate(
        [
            "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}

EXPECTED_COLUMNS = [
    "hotel", "is_canceled", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies", "meal", "country",
    "market_segment", "distribution_channel", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled", "reserved_room_type",
    "assigned_room_type", "booking_changes", "deposit_type", "agent", "company",
    "days_in_waiting_list", "customer_type", "adr", "required_car_parking_spaces",
    "total_of_special_requests", "reservation_status", "reservation_status_date",
]

# The city hotel's soft opening (July-August 2015) is excluded, as the data
# authors did when they built their production system (Antonio et al., 2019,
# Data Science Journal 18, 32).
CITY_FIRST_ARRIVAL = pd.Timestamp("2015-09-01")


@dataclass
class ContractReport:
    """What the loader checked and what it found."""

    rows_read: int = 0
    rows_kept: int = 0
    checks: dict[str, int] = field(default_factory=dict)

    def lines(self) -> list[str]:
        out = [f"rows read: {self.rows_read}", f"rows kept: {self.rows_kept}"]
        out += [f"{name}: {value}" for name, value in self.checks.items()]
        return out


def sha256_of_gzip(path=DATA_FILE) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path=DATA_FILE, verify_checksum: bool = True) -> tuple[pd.DataFrame, ContractReport]:
    """Read the bookings, derive event dates and validate them.

    Returns the cleaned frame and a report of every check. Violations that
    would make a point-in-time reconstruction impossible raise; soft issues
    are counted and reported.
    """
    if verify_checksum:
        found = sha256_of_gzip(path)
        if found != SOURCE_SHA256:
            raise ValueError(f"data checksum mismatch: {found}")

    raw = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
    report = ContractReport(rows_read=len(raw))
    if list(raw.columns) != EXPECTED_COLUMNS:
        raise ValueError("unexpected columns in the bookings file")

    df = raw.copy()
    df["arrival"] = pd.to_datetime(
        dict(
            year=df["arrival_date_year"],
            month=df["arrival_date_month"].map(MONTHS),
            day=df["arrival_date_day_of_month"],
        )
    )
    df["booked"] = df["arrival"] - pd.to_timedelta(df["lead_time"], unit="D")
    df["status_date"] = pd.to_datetime(df["reservation_status_date"])
    df["children"] = pd.to_numeric(df["children"], errors="coerce").fillna(0).astype(int)
    df["agent"] = df["agent"].astype(str)
    df["company"] = df["company"].astype(str)

    status = df["reservation_status"]
    if set(status.unique()) != {"Check-Out", "Canceled", "No-Show"}:
        raise ValueError("unexpected reservation status values")
    if not (df["is_canceled"] == status.isin(["Canceled", "No-Show"]).astype(int)).all():
        raise ValueError("is_canceled disagrees with reservation_status")

    cancelled = status == "Canceled"
    # The timestamp of a cancellation must lie between booking and arrival,
    # or the event history cannot be replayed.
    report.checks["cancellations dated before booking"] = int(
        (cancelled & (df["status_date"] < df["booked"])).sum()
    )
    report.checks["cancellations dated after arrival"] = int(
        (cancelled & (df["status_date"] > df["arrival"])).sum()
    )
    if report.checks["cancellations dated before booking"] or report.checks[
        "cancellations dated after arrival"
    ]:
        raise ValueError("cancellation dates outside the booking window")

    report.checks["no-shows with status date before arrival"] = int(
        ((status == "No-Show") & (df["status_date"] < df["arrival"])).sum()
    )
    report.checks["rows with negative ADR"] = int((df["adr"] < 0).sum())
    report.checks["rows with ADR above 1000"] = int((df["adr"] > 1000).sum())
    report.checks["rows with zero guests"] = int(
        ((df["adults"] + df["children"] + df["babies"]) == 0).sum()
    )

    soft_opening = (df["hotel"] == "City Hotel") & (df["arrival"] < CITY_FIRST_ARRIVAL)
    report.checks["city hotel soft-opening rows dropped"] = int(soft_opening.sum())
    df = df.loc[~soft_opening].reset_index(drop=True)

    # The event that matters for the room: the booking stops holding it.
    # Cancellations happen on their status date; no-shows on the arrival date.
    status = df["reservation_status"]
    release = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    release[status == "Canceled"] = df.loc[status == "Canceled", "status_date"]
    release[status == "No-Show"] = df.loc[status == "No-Show", "arrival"]
    df["release_date"] = release
    df["arrival_dow"] = df["arrival"].dt.dayofweek
    df["arrival_week"] = df["arrival"].dt.isocalendar().week.astype(int)
    df["nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

    report.rows_kept = len(df)
    return df, report
