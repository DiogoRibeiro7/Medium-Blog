"""Disruption scenarios.

Every scenario includes the baseline stochastic primary delays; the random draws of the
baseline part depend only on the replication, so a scenario and the baseline share them
(common random numbers) and the footprint of a disruption can be measured flight by
flight. Scenario-specific events are drawn from a separate stream that does not depend
on the plan, so every planner faces exactly the same events: the same flights delayed,
the aircraft due to operate the same flight grounded, the crews due to operate the same
flights calling in sick.

All events happen on the first simulated day (day 0). Times are local to the hub.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from airline_ops.config import parse_clock
from airline_ops.network.model import DAY, Network


@dataclass(frozen=True, slots=True)
class Scenario:
    """A disruption scenario, defined independently of any plan."""

    name: str
    label: str
    onset_local: str = "00:00"
    inbound_delay_flights: int = 0
    inbound_delay_range: tuple[int, int] = (0, 0)
    aog_aircraft: int = 0
    aog_hours: float = 0.0
    sick_crews: int = 0
    capacity_window_local: tuple[str, str] | None = None
    capacity_factor: float = 1.0
    primary_scale: float = 1.0
    scale_window_local: tuple[str, str] | None = None


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("S0", "Baseline: stochastic delays only"),
    Scenario(
        "S1", "Delayed inbound aircraft", onset_local="06:00", inbound_delay_flights=3, inbound_delay_range=(90, 150)
    ),
    Scenario("S2", "Aircraft unavailable at an outstation (AOG)", onset_local="05:00", aog_aircraft=1, aog_hours=8.0),
    Scenario("S3", "Crew unavailable", onset_local="06:00", sick_crews=2),
    Scenario(
        "S4",
        "Hub capacity restriction",
        onset_local="07:00",
        capacity_window_local=("07:00", "11:00"),
        capacity_factor=0.5,
    ),
    Scenario(
        "S5",
        "Cascading delays (stressed day)",
        onset_local="05:00",
        primary_scale=2.0,
        scale_window_local=("05:00", "23:59"),
    ),
    Scenario(
        "S6",
        "Evening crunch: duty limits",
        onset_local="15:00",
        capacity_window_local=("16:00", "20:00"),
        capacity_factor=0.6,
        primary_scale=1.5,
        scale_window_local=("15:00", "23:59"),
    ),
)

SCENARIO_BY_NAME = {s.name: s for s in SCENARIOS}


@dataclass(frozen=True)
class ScenarioEvents:
    """Concrete, plan-independent events of one scenario replication (absolute UTC minutes)."""

    onset: int
    inbound_delays: dict[int, float] = field(default_factory=dict)  # day-0 flight index -> extra minutes
    aog_flights: tuple[int, ...] = ()  # the aircraft due to operate these day-0 flights is grounded
    aog_minutes: int = 0
    sick_flights: tuple[int, ...] = ()  # the crews due to operate these day-0 flights call in sick
    capacity: tuple[int, int, float] | None = None  # (start, end, factor) for hub arrivals
    scale_window: tuple[int, int] | None = None
    primary_scale: float = 1.0


def _utc(network: Network, local: str) -> int:
    hub = network.airports[network.hub]
    return (parse_clock(local) - hub.utc_offset_min) % DAY


def draw_events(scenario: Scenario, network: Network, rng: np.random.Generator) -> ScenarioEvents:
    """Draw the scenario's events for one replication."""
    onset = _utc(network, scenario.onset_local)
    flights = network.flights
    hub = network.hub
    inbound: dict[int, float] = {}
    if scenario.inbound_delay_flights:
        hub_off = network.airports[hub].utc_offset_min
        first_wave = [f.idx for f in flights if f.destination == hub and (f.dep + hub_off) % DAY < parse_clock("08:00")]
        k = min(scenario.inbound_delay_flights, len(first_wave))
        chosen = rng.choice(len(first_wave), size=k, replace=False)
        lo, hi = scenario.inbound_delay_range
        for c in chosen:
            inbound[first_wave[int(c)]] = float(rng.uniform(lo, hi))
    aog: tuple[int, ...] = ()
    if scenario.aog_aircraft:
        # A technical fault found before the first departure of the day at an outstation,
        # where no spare aircraft is based.
        hub_off = network.airports[hub].utc_offset_min
        first_wave = [f.idx for f in flights if f.destination == hub and (f.dep + hub_off) % DAY < parse_clock("08:00")]
        k = min(scenario.aog_aircraft, len(first_wave))
        aog = tuple(int(first_wave[int(c)]) for c in rng.choice(len(first_wave), size=k, replace=False))
    sick: tuple[int, ...] = ()
    if scenario.sick_crews:
        morning = [f.idx for f in flights if f.origin == hub and onset <= f.dep < onset + 240]
        k = min(scenario.sick_crews, len(morning))
        sick = tuple(int(morning[int(c)]) for c in rng.choice(len(morning), size=k, replace=False))
    capacity = None
    if scenario.capacity_window_local is not None:
        start = _utc(network, scenario.capacity_window_local[0])
        end = _utc(network, scenario.capacity_window_local[1])
        capacity = (start, end if end > start else end + DAY, scenario.capacity_factor)
    window = None
    if scenario.scale_window_local is not None:
        s0 = _utc(network, scenario.scale_window_local[0])
        s1 = _utc(network, scenario.scale_window_local[1])
        window = (s0, s1 if s1 > s0 else s1 + DAY)
    return ScenarioEvents(
        onset=onset,
        inbound_delays=inbound,
        aog_flights=aog,
        aog_minutes=int(scenario.aog_hours * 60),
        sick_flights=sick,
        capacity=capacity,
        scale_window=window,
        primary_scale=scenario.primary_scale,
    )
