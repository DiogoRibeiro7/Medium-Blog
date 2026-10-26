"""A feature registry that records when each value becomes known.

Every column the models may read is listed with the moment at which the hotel
system can know its value. A model that serves at a decision time may only
read columns known at booking. The reason given for each entry is the evidence
for its timing: the data documentation, the authors' deployment paper, or a
pattern in the data that no booking-time value could produce.
"""

from __future__ import annotations

from dataclasses import dataclass

BOOKING = "booking"
UNCLEAR = "unclear"
LATER = "later"
OUTCOME = "outcome"


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str  # "numeric" or "categorical"
    known: str
    reason: str


REGISTRY: tuple[Feature, ...] = (
    Feature("lead_time", "numeric", BOOKING, "days from entry in the PMS to arrival"),
    Feature("arrival_week", "numeric", BOOKING, "derived from the arrival date"),
    Feature("arrival_dow", "categorical", BOOKING, "derived from the arrival date"),
    Feature("stays_in_weekend_nights", "numeric", BOOKING, "requested stay"),
    Feature("stays_in_week_nights", "numeric", BOOKING, "requested stay"),
    Feature("adults", "numeric", BOOKING, "party size"),
    Feature("children", "numeric", BOOKING, "party size"),
    Feature("babies", "numeric", BOOKING, "party size"),
    Feature("meal", "categorical", BOOKING, "board booked"),
    Feature("market_segment", "categorical", BOOKING, "sales segment"),
    Feature("distribution_channel", "categorical", BOOKING, "sales channel"),
    Feature("is_repeated_guest", "numeric", BOOKING, "guest history"),
    Feature("previous_cancellations", "numeric", BOOKING, "guest history"),
    Feature("previous_bookings_not_canceled", "numeric", BOOKING, "guest history"),
    Feature("reserved_room_type", "categorical", BOOKING, "room booked"),
    Feature("agent", "categorical", BOOKING, "booking agency"),
    Feature("company", "categorical", BOOKING, "paying company"),
    Feature("customer_type", "categorical", BOOKING, "contract type"),
    Feature("adr", "numeric", BOOKING, "agreed average daily rate"),
    Feature("days_in_waiting_list", "numeric", BOOKING, "days before confirmation"),
    Feature(
        "deposit_type", "categorical", UNCLEAR,
        "computed from payments recorded before arrival or cancellation",
    ),
    Feature(
        "country", "categorical", LATER,
        "filled with Portugal by default and corrected at check-in",
    ),
    Feature(
        "assigned_room_type", "categorical", LATER,
        "room actually assigned, set by hotel operations around arrival",
    ),
    Feature(
        "booking_changes", "numeric", LATER,
        "amendments counted until check-in or cancellation",
    ),
    Feature(
        "required_car_parking_spaces", "numeric", LATER,
        "no booking with a parking space ever cancelled, so it is recorded on arrival",
    ),
    Feature(
        "total_of_special_requests", "numeric", LATER,
        "requests accumulated until check-in or cancellation",
    ),
)

TARGET_LEAKS = ("reservation_status", "reservation_status_date")


def columns(*allowed: str) -> list[Feature]:
    """Features whose timing is in ``allowed``."""
    return [f for f in REGISTRY if f.known in allowed]


POINT_IN_TIME = columns(BOOKING)
POINT_IN_TIME_PLUS_DEPOSIT = columns(BOOKING, UNCLEAR)
AS_RECORDED = columns(BOOKING, UNCLEAR, LATER)


def assert_point_in_time(features: list[Feature]) -> None:
    """Refuse a feature set that reads anything not known at booking."""
    late = [f.name for f in features if f.known != BOOKING]
    if late:
        raise ValueError(f"features not known at decision time: {late}")
