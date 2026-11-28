"""Data model of the daily schedule.

Times are integer minutes in UTC. A flight's scheduled departure ``dep`` is a minute of
the day in ``[0, 1440)``; its arrival ``dep + block`` may pass midnight. The schedule
repeats every day, so a plan is a daily pattern (as in the classical daily aircraft
routing and crew pairing problems) and the simulator unrolls it over several days.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from airline_ops.errors import DataError

DAY = 1440
COUNT_LINE_UTC = 2 * 60  # aircraft and crews are counted at 02:00 UTC, when no flight is scheduled


@dataclass(frozen=True, slots=True)
class Airport:
    """An airport of the network."""

    code: str
    name: str
    latitude: float
    longitude: float
    utc_offset_min: int
    is_hub: bool
    curfew_utc: tuple[int, int] | None = None

    def local_minute(self, utc_minute: int) -> int:
        """Local clock minute (0-1439) of an absolute UTC minute."""
        return (utc_minute + self.utc_offset_min) % DAY

    def in_curfew(self, utc_minute: int) -> bool:
        """Whether an absolute UTC minute falls inside the airport's curfew."""
        if self.curfew_utc is None:
            return False
        start, end = self.curfew_utc
        t = utc_minute % DAY
        return start <= t < end if start < end else (t >= start or t < end)


@dataclass(frozen=True, slots=True)
class Flight:
    """One daily scheduled flight."""

    idx: int
    number: str
    origin: str
    destination: str
    dep: int
    block: int
    distance_km: float
    seats: int

    def __post_init__(self) -> None:
        if not 0 <= self.dep < DAY:
            raise DataError(f"{self.number}: departure minute {self.dep} outside [0, 1440)")
        if self.block <= 0:
            raise DataError(f"{self.number}: block time must be positive")
        if self.origin == self.destination:
            raise DataError(f"{self.number}: origin equals destination")

    @property
    def arr(self) -> int:
        """Scheduled arrival in minutes after the departure day's midnight (may exceed 1440)."""
        return self.dep + self.block


@dataclass(frozen=True, slots=True)
class PassengerGroup:
    """Passengers who share the same itinerary (one or two legs, connecting at the hub)."""

    idx: int
    legs: tuple[int, ...]
    count: int
    origin: str
    destination: str
    distance_km: float

    def __post_init__(self) -> None:
        if not 1 <= len(self.legs) <= 2:
            raise DataError("an itinerary has one or two legs")
        if self.count <= 0:
            raise DataError("a passenger group needs at least one passenger")


def ground_gap(arrival: int, departure_minute: int) -> int:
    """Minutes from an arrival (absolute, any day) to the next occurrence of a daily departure minute."""
    return (departure_minute - arrival) % DAY


def crosses_count_line(start: int, length: int) -> int:
    """How many times the interval ``[start, start + length)`` crosses the daily count line."""
    if length <= 0:
        return 0
    first = start - (start - COUNT_LINE_UTC) % DAY
    if first < start:
        first += DAY
    if first >= start + length:
        return 0
    return 1 + (start + length - 1 - first) // DAY


@dataclass(frozen=True)
class Network:
    """A daily schedule with its airports, fleet size and passengers."""

    airports: Mapping[str, Airport]
    flights: tuple[Flight, ...]
    hub: str
    fleet_size: int
    passengers: tuple[PassengerGroup, ...]
    turn_min: Mapping[str, int]
    seed: int
    by_number: Mapping[str, int] = field(init=False)

    def __post_init__(self) -> None:
        if self.hub not in self.airports or not self.airports[self.hub].is_hub:
            raise DataError("the hub must be an airport flagged as hub")
        for i, f in enumerate(self.flights):
            if f.idx != i:
                raise DataError("flight indices must be 0..n-1 in order")
            if f.origin not in self.airports or f.destination not in self.airports:
                raise DataError(f"{f.number}: unknown airport")
        numbers = [f.number for f in self.flights]
        if len(set(numbers)) != len(numbers):
            raise DataError("flight numbers must be unique")
        for code in self.airports:
            if code not in self.turn_min:
                raise DataError(f"no minimum turn time for {code}")
        object.__setattr__(self, "by_number", {f.number: f.idx for f in self.flights})

    @property
    def n_flights(self) -> int:
        """Number of daily flights."""
        return len(self.flights)

    def airborne_at_count_line(self) -> int:
        """Flights in the air at the daily count line (each one occupies an aircraft)."""
        return sum(crosses_count_line(f.dep, f.block) for f in self.flights)

    def minimum_fleet(self, extra_turn: int = 0) -> int:
        """Smallest fleet that can fly the daily schedule with the given minimum turn times.

        For a periodic schedule with a single fleet, the minimum number of aircraft is the
        number in the air at the count line plus, at each station, the largest deficit of
        ready arrivals against departures over the day measured from the count line. A
        first-in-first-out assignment attains it.
        """
        # Aircraft that are airborne or still inside their minimum turn at the count line ...
        total = sum(
            crosses_count_line(f.dep, f.block + self.turn_min[f.destination] + extra_turn) for f in self.flights
        )
        # ... plus aircraft that are ready but idle at each station at the count line.
        for code in self.airports:
            turn = self.turn_min[code] + extra_turn
            events: list[tuple[int, int]] = []
            for f in self.flights:
                if f.origin == code:
                    events.append(((f.dep - COUNT_LINE_UTC) % DAY, -1))
                if f.destination == code:
                    ready = (f.arr + turn - COUNT_LINE_UTC) % DAY
                    events.append((ready, +1))
            # Arrivals that become ready before a departure at the same minute can serve it.
            events.sort(key=lambda e: (e[0], -e[1]))
            level, worst = 0, 0
            for _, delta in events:
                level += delta
                worst = min(worst, level)
            total += -worst
        return total

    def pax_on_flight(self) -> list[int]:
        """Passengers booked on each flight."""
        load = [0] * self.n_flights
        for g in self.passengers:
            for leg in g.legs:
                load[leg] += g.count
        return load
