"""EASA flight time limitations used in the model.

Source: Commission Regulation (EU) No 965/2012, Annex III (Part-ORO), Subpart FTL, as
amended by Commission Regulation (EU) No 83/2014.

* ``ORO.FTL.205(b)(1)``, Table 2: maximum daily flight duty period (FDP) without
  extensions for acclimatised crew members, by start of FDP at reference time and
  number of sectors.
* ``ORO.FTL.235(a)(1)``: minimum rest before an FDP starting at home base is at
  least as long as the preceding duty period, or 12 hours, whichever is greater.
* ``ORO.FTL.235(b)``: minimum rest before an FDP starting away from home base is at
  least as long as the preceding duty period, or 10 hours, whichever is greater.
* ``ORO.FTL.205(f)``: commander's discretion, an extension of up to two hours in
  unforeseen circumstances arising after reporting.

Reporting time before the first departure and post-flight duty after the last
arrival are set by the operator (``ORO.FTL.205(a)(1)``); the values here are
assumptions of this model, not Lufthansa or EASA numbers.

Simplifications: every crew member is acclimatised to the home base, the reference
time is the local time at the reporting point (valid because the network spans at
most two hours of time difference), and FDP extensions, split duty, in-flight rest
and reduced rest are not used in planning.
"""

from __future__ import annotations

from dataclasses import dataclass

from airline_ops.errors import ConfigError

# Table 2 rows: (first minute of the band, last minute of the band) at reference time,
# and the maximum FDP in minutes for 1-2, 3, 4, ..., 10 sectors.
_TABLE_2: tuple[tuple[int, int, tuple[int, ...]], ...] = (
    (6 * 60, 13 * 60 + 29, (780, 750, 720, 690, 660, 630, 600, 570, 540)),
    (13 * 60 + 30, 13 * 60 + 59, (765, 735, 705, 675, 645, 615, 585, 555, 540)),
    (14 * 60, 14 * 60 + 29, (750, 720, 690, 660, 630, 600, 570, 540, 540)),
    (14 * 60 + 30, 14 * 60 + 59, (735, 705, 675, 645, 615, 585, 555, 540, 540)),
    (15 * 60, 15 * 60 + 29, (720, 690, 660, 630, 600, 570, 540, 540, 540)),
    (15 * 60 + 30, 15 * 60 + 59, (705, 675, 645, 615, 585, 555, 540, 540, 540)),
    (16 * 60, 16 * 60 + 29, (690, 660, 630, 600, 570, 540, 540, 540, 540)),
    (16 * 60 + 30, 16 * 60 + 59, (675, 645, 615, 585, 555, 540, 540, 540, 540)),
    (17 * 60, 23 * 60 + 59, (660, 630, 600, 570, 540, 540, 540, 540, 540)),
    (0, 4 * 60 + 59, (660, 630, 600, 570, 540, 540, 540, 540, 540)),
    (5 * 60, 5 * 60 + 14, (720, 690, 660, 630, 600, 570, 540, 540, 540)),
    (5 * 60 + 15, 5 * 60 + 29, (735, 705, 675, 645, 615, 585, 555, 540, 540)),
    (5 * 60 + 30, 5 * 60 + 44, (750, 720, 690, 660, 630, 600, 570, 540, 540)),
    (5 * 60 + 45, 5 * 60 + 59, (765, 735, 705, 675, 645, 615, 585, 555, 540)),
)

MAX_SECTORS_TABLE_2 = 10
MIN_REST_HOME_MIN = 12 * 60
MIN_REST_AWAY_MIN = 10 * 60
COMMANDER_DISCRETION_MAX_MIN = 2 * 60


def max_fdp_minutes(report_local_minute: int, sectors: int) -> int:
    """Maximum daily FDP (minutes) from EASA ORO.FTL.205 Table 2.

    Args:
        report_local_minute: Start of the FDP at reference time, in minutes after local
            midnight. Values outside ``[0, 1440)`` are wrapped.
        sectors: Number of sectors flown in the FDP, 1 to 10.

    Raises:
        ValueError: When ``sectors`` is outside 1-10.

    """
    if not 1 <= sectors <= MAX_SECTORS_TABLE_2:
        raise ValueError(f"Table 2 covers 1-10 sectors, got {sectors}")
    minute = report_local_minute % 1440
    column = max(0, sectors - 2)
    for start, end, limits in _TABLE_2:
        if start <= minute <= end:
            return limits[column]
    raise AssertionError(f"Table 2 bands do not cover minute {minute}")  # pragma: no cover


def min_rest_minutes(preceding_duty_min: int, *, at_home_base: bool) -> int:
    """Minimum rest before the next FDP (ORO.FTL.235(a)(1) and (b)).

    Raises:
        ValueError: When the preceding duty is negative.

    """
    if preceding_duty_min < 0:
        raise ValueError("preceding duty period cannot be negative")
    floor = MIN_REST_HOME_MIN if at_home_base else MIN_REST_AWAY_MIN
    return max(preceding_duty_min, floor)


@dataclass(frozen=True, slots=True)
class OperatorTimes:
    """Operator-defined reporting and post-flight times (assumptions of this model)."""

    report_before_min: int = 60
    post_flight_min: int = 30

    def __post_init__(self) -> None:
        if self.report_before_min <= 0 or self.post_flight_min < 0:
            raise ConfigError("reporting time must be positive and post-flight time non-negative")


def fdp_is_legal(
    report_local_minute: int,
    fdp_min: int,
    sectors: int,
    *,
    discretion_min: int = 0,
) -> bool:
    """Whether an FDP of ``fdp_min`` minutes is within Table 2, plus optional discretion.

    Args:
        report_local_minute: Report time at reference time (local minutes).
        fdp_min: Length of the FDP from report to on-blocks of the last sector.
        sectors: Sectors in the FDP.
        discretion_min: Commander's discretion applied, at most two hours.

    Raises:
        ValueError: When the discretion exceeds the two hours allowed by ORO.FTL.205(f).

    """
    if not 0 <= discretion_min <= COMMANDER_DISCRETION_MAX_MIN:
        raise ValueError("commander's discretion is limited to 0-120 minutes")
    if sectors > MAX_SECTORS_TABLE_2:
        return False
    return fdp_min <= max_fdp_minutes(report_local_minute, sectors) + discretion_min
