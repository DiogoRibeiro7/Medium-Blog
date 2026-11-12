"""Rules written from the TLC data dictionaries of 18 March 2025, one contract per feed.

Each field records what a careful engineer would take from its description in the dictionary:

- ``kind``: the type its description implies (integer for counts, codes and zones; float for amounts and
  distances; timestamp; string for base numbers and Y/N flags);
- ``required``: every trip must have it (the dictionary describes it as a property of every trip);
- ``nonneg``: an amount, a distance or a duration, so it cannot be negative;
- ``codes``: the values the dictionary lists for it (nulls are judged by ``required``, not here);
- ``zone``: a TLC Taxi Zone, so a number from the zone lookup, 1 to 265;
- ``since``: the first month the dictionary says the field exists (only ``cbd_congestion_fee`` has one).

``ORDER`` holds timestamps whose descriptions imply an order within a trip, and ``PICKUP`` the timestamp that
places a trip in a month. ``TOTAL`` (yellow only) is the total and the charges it is the sum of.

The quoted descriptions are kept next to each field so a rule can be traced to the sentence it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Field:
    name: str
    kind: str
    required: bool = True
    nonneg: bool = False
    codes: tuple = ()
    zone: bool = False
    since: str | None = None
    says: str = ""


@dataclass(frozen=True)
class Contract:
    feed: str
    fields: tuple[Field, ...]
    pickup: str
    order: tuple[tuple[str, str], ...]
    total: tuple[str, tuple[str, ...]] | None = None
    first_month: str = "2019-01"
    report_classes: tuple[str, ...] = field(default=())

    def by_lower(self) -> dict[str, Field]:
        return {f.name.lower(): f for f in self.fields}


YELLOW = Contract(
    feed="yellow",
    fields=(
        Field("VendorID", "integer", codes=(1, 2, 6, 7), says="A code indicating the TPEP provider"),
        Field("tpep_pickup_datetime", "timestamp", says="The date and time when the meter was engaged."),
        Field("tpep_dropoff_datetime", "timestamp", says="The date and time when the meter was disengaged."),
        Field("passenger_count", "integer", nonneg=True, says="The number of passengers in the vehicle."),
        Field("trip_distance", "float", nonneg=True, says="The elapsed trip distance in miles"),
        Field("RatecodeID", "integer", codes=(1, 2, 3, 4, 5, 6, 99), says="The final rate code in effect"),
        Field("store_and_fwd_flag", "string", codes=("Y", "N"), says="Y = store and forward trip; N = not"),
        Field("PULocationID", "integer", zone=True, says="TLC Taxi Zone in which the taximeter was engaged."),
        Field("DOLocationID", "integer", zone=True, says="TLC Taxi Zone in which the taximeter was disengaged."),
        Field("payment_type", "integer", codes=(0, 1, 2, 3, 4, 5, 6), says="A numeric code signifying how"),
        Field("fare_amount", "float", nonneg=True, says="The time-and-distance fare calculated by the meter."),
        Field("extra", "float", nonneg=True, says="Miscellaneous extras and surcharges."),
        Field("mta_tax", "float", nonneg=True, says="Tax that is automatically triggered"),
        Field("tip_amount", "float", nonneg=True, says="Tip amount"),
        Field("tolls_amount", "float", nonneg=True, says="Total amount of all tolls paid in trip."),
        Field("improvement_surcharge", "float", nonneg=True, says="Improvement surcharge assessed trips"),
        Field("total_amount", "float", nonneg=True, says="The total amount charged to passengers."),
        Field("congestion_surcharge", "float", nonneg=True, says="Total amount collected in trip for NYS"),
        Field("airport_fee", "float", nonneg=True, says="For pick up only at LaGuardia and JFK."),
        Field("cbd_congestion_fee", "float", nonneg=True, since="2025-01", says="starting Jan. 5, 2025."),
    ),
    pickup="tpep_pickup_datetime",
    order=(("tpep_pickup_datetime", "tpep_dropoff_datetime"),),
    total=(
        "total_amount",
        (
            "fare_amount",
            "extra",
            "mta_tax",
            "tip_amount",
            "tolls_amount",
            "improvement_surcharge",
            "congestion_surcharge",
            "airport_fee",
            "cbd_congestion_fee",
        ),
    ),
    report_classes=("Yellow",),
)

FHV = Contract(
    feed="fhv",
    fields=(
        Field("dispatching_base_num", "string", says="The TLC Base License Number of the base that dispatched"),
        Field("pickup_datetime", "timestamp", says="The date and time the trip was picked up."),
        Field("dropOff_datetime", "timestamp", says="The date and time the trip ended."),
        Field("PUlocationID", "integer", zone=True, says="The TLC Taxi Zone where the trip began."),
        Field("DOlocationID", "integer", zone=True, says="The TLC Taxi Zone where the trip ended."),
        Field("SR_Flag", "integer", required=False, codes=(1,), says="Shared trips are coded 1, non-shared null."),
        Field("Affiliated_base_number", "string", says="This must be provided even if it is the same"),
    ),
    pickup="pickup_datetime",
    order=(("pickup_datetime", "dropOff_datetime"),),
    report_classes=("FHV - Black Car", "FHV - Livery", "FHV - Lux Limo"),
)

YN = ("Y", "N")
FHVHV = Contract(
    feed="fhvhv",
    fields=(
        Field(
            "hvfhs_license_num",
            "string",
            codes=("HV0002", "HV0003", "HV0004", "HV0005"),
            says="The TLC license number of the HVFHS base or business.",
        ),
        Field("dispatching_base_num", "string", says="The TLC Base License Number of the base that dispatched"),
        Field("originating_base_num", "string", says="Base number of the base that received the original request."),
        Field("request_datetime", "timestamp", says="Date/time when passenger requested to be picked up."),
        Field("on_scene_datetime", "timestamp", required=False, says="(Accessible Vehicles only)."),
        Field("pickup_datetime", "timestamp", says="The date and time of the trip pick-up."),
        Field("dropoff_datetime", "timestamp", says="The date and time of the trip drop-off."),
        Field("PULocationID", "integer", zone=True, says="TLC Taxi Zone in which the trip began."),
        Field("DOLocationID", "integer", zone=True, says="TLC Taxi Zone in which the trip ended."),
        Field("trip_miles", "float", nonneg=True, says="Total miles for passenger trip."),
        Field("trip_time", "integer", nonneg=True, says="Total time in seconds for passenger trip."),
        Field("base_passenger_fare", "float", nonneg=True, says="Base passenger fare before tolls, tips, taxes"),
        Field("tolls", "float", nonneg=True, says="Total amount of all tolls paid in trip."),
        Field("bcf", "float", nonneg=True, says="Total amount collected in trip for Black Car Fund."),
        Field("sales_tax", "float", nonneg=True, says="Total amount collected in trip for NYS sales tax."),
        Field("congestion_surcharge", "float", nonneg=True, says="Total amount collected for NYS congestion"),
        Field(
            "airport_fee",
            "float",
            nonneg=True,
            codes=(0.0, 2.5, 5.0),
            says="$2.50 for both drop off and pick up at LaGuardia, Newark, and John F. Kennedy airports.",
        ),
        Field("tips", "float", nonneg=True, says="Total amount of tips received from passenger."),
        Field("driver_pay", "float", nonneg=True, says="Total driver pay (not including tolls or tips"),
        Field("shared_request_flag", "string", codes=YN, says="(Y/N)"),
        Field("shared_match_flag", "string", codes=YN, says="(Y/N)"),
        Field("access_a_ride_flag", "string", codes=YN, says="(Y/N)"),
        Field("wav_request_flag", "string", codes=YN, says="(Y/N)"),
        Field("wav_match_flag", "string", codes=YN, says="(Y/N)"),
        Field("cbd_congestion_fee", "float", nonneg=True, since="2025-01", says="starting Jan. 5, 2025."),
    ),
    pickup="pickup_datetime",
    order=(
        ("request_datetime", "pickup_datetime"),
        ("on_scene_datetime", "pickup_datetime"),
        ("pickup_datetime", "dropoff_datetime"),
    ),
    first_month="2019-02",
    report_classes=("FHV - High Volume",),
)

GREEN_REPORT = ("Green",)  # green enters the outside-count test only; its dictionary was not available here

CONTRACTS = {c.feed: c for c in (YELLOW, FHV, FHVHV)}
FEEDS = ("yellow", "green", "fhv", "fhvhv")  # green is profiled for its row count and schema only
FIRST_MONTH = {"yellow": "2019-01", "green": "2019-01", "fhv": "2019-01", "fhvhv": "2019-02"}
ZONES = (1, 265)
TOLERANCE = 0.01
DICTIONARY_DATE = "2025-03"
