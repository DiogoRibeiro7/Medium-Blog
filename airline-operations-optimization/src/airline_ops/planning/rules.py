"""The planning rules shared by every planner, the validator and the simulator.

A :class:`PlanningRules` object fixes the minimum connection times, the FTL limits in
force and the buffers a robust plan adds. Building every planner and the validator
from the same object is what makes the comparison between plans fair.
"""

from __future__ import annotations

from dataclasses import dataclass

from airline_ops.config import Config
from airline_ops.network.model import Network
from airline_ops.regulation.ftl import MIN_REST_AWAY_MIN, OperatorTimes, max_fdp_minutes, min_rest_minutes


@dataclass(frozen=True, slots=True)
class PlanningRules:
    """Connection, duty and pairing rules, with optional robustness buffers."""

    turn_min: dict[str, int]
    crew_change_min: int
    max_sit_min: int
    max_sectors: int
    max_duties: int
    max_rest_min: int
    operator: OperatorTimes
    base: str
    maintenance_min_ground: int
    maintenance_max_nights: int
    aircraft_buffer: int = 0
    crew_buffer: int = 0

    @classmethod
    def from_config(cls, config: Config, network: Network, *, buffer_min: int = 0) -> PlanningRules:
        """Rules for a network, with the same buffer on aircraft turns and crew connections."""
        return cls(
            turn_min=dict(network.turn_min),
            crew_change_min=config.crew.change_aircraft_min,
            max_sit_min=config.crew.max_sit_min,
            max_sectors=config.crew.max_sectors,
            max_duties=config.crew.max_duties,
            max_rest_min=config.crew.max_rest_min,
            operator=OperatorTimes(),
            base=network.hub,
            maintenance_min_ground=config.maintenance.min_ground_min,
            maintenance_max_nights=config.maintenance.max_nights_between,
            aircraft_buffer=buffer_min,
            crew_buffer=buffer_min,
        )

    def aircraft_turn(self, station: str) -> int:
        """Minimum planned aircraft turn at a station, including the robustness buffer.

        The buffer applies at the hub only. An outstation turn is fixed by the timetable
        (an out-and-back aircraft has no alternative), so a buffer there would make the
        schedule infeasible rather than more robust; adding slack at outstations means
        retiming flights, which is a schedule-design decision outside these planners.
        """
        return self.turn_min[station] + (self.aircraft_buffer if station == self.base else 0)

    def crew_same_aircraft_min(self, station: str) -> int:
        """Minimum crew connection when the crew stays with its aircraft (buffered at the hub)."""
        return self.turn_min[station] + (self.crew_buffer if station == self.base else 0)

    def crew_change_aircraft_min(self) -> int:
        """Minimum crew connection when the crew changes aircraft."""
        return self.crew_change_min + self.crew_buffer

    def max_fdp(self, report_local_minute: int, sectors: int) -> int:
        """EASA Table 2 limit; no extensions are planned."""
        return max_fdp_minutes(report_local_minute, sectors)

    def min_rest_away(self, duty_length: int) -> int:
        """Minimum rest away from base after a duty (ORO.FTL.235(b))."""
        return max(min_rest_minutes(duty_length, at_home_base=False), MIN_REST_AWAY_MIN)
