"""Experiment configuration, read from TOML and validated on load.

Every number that drives a result lives in a configuration file, not in the code.
Cost parameters are illustrative assumptions and are labelled as such in the article;
regulatory values (EASA flight time limitations) are in :mod:`airline_ops.regulation.ftl`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

from airline_ops.errors import ConfigError

T = TypeVar("T")


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")


def _non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ConfigError(f"{name} must be non-negative, got {value}")


def parse_clock(text: str) -> int:
    """Convert ``"HH:MM"`` to minutes after midnight.

    Raises:
        ConfigError: When the text is not a valid 24-hour clock time.

    """
    try:
        hours, minutes = text.split(":")
        h, m = int(hours), int(minutes)
    except ValueError as exc:
        raise ConfigError(f"invalid clock time {text!r}; expected HH:MM") from exc
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ConfigError(f"invalid clock time {text!r}")
    return h * 60 + m


@dataclass(frozen=True, slots=True)
class SpokeConfig:
    """One outstation and the number of daily round trips to it."""

    code: str
    frequency: int

    def __post_init__(self) -> None:
        if len(self.code) != 3 or not self.code.isalpha() or not self.code.isupper():
            raise ConfigError(f"spoke code must be a 3-letter IATA code, got {self.code!r}")
        if not 1 <= self.frequency <= 4:
            raise ConfigError(f"spoke {self.code}: frequency must be 1-4, got {self.frequency}")


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    """Shape of the synthetic hub-and-spoke schedule."""

    hub: str
    reference_date: str
    spokes: tuple[SpokeConfig, ...]
    banks_local: tuple[str, ...]
    nightstop_departure_local: str
    morning_departure_local: str
    bank_jitter_min: int
    seats: int
    load_factor: float
    connecting_share: float
    min_connection_pax_min: int
    max_connection_pax_min: int
    spare_aircraft: int
    hub_curfew_local: tuple[str, str]
    turn_min_hub: int
    turn_min_spoke: int
    spoke_turn_scheduled_min: int

    def __post_init__(self) -> None:
        if not self.spokes:
            raise ConfigError("network.spokes must not be empty")
        codes = [s.code for s in self.spokes]
        if len(set(codes)) != len(codes):
            raise ConfigError("network.spokes contains duplicate codes")
        if self.hub in codes:
            raise ConfigError("the hub cannot also be a spoke")
        for clock in (*self.banks_local, self.nightstop_departure_local, self.morning_departure_local):
            parse_clock(clock)
        for clock in self.hub_curfew_local:
            parse_clock(clock)
        _positive("network.seats", self.seats)
        if not 0 < self.load_factor <= 1:
            raise ConfigError("network.load_factor must be in (0, 1]")
        if not 0 <= self.connecting_share < 1:
            raise ConfigError("network.connecting_share must be in [0, 1)")
        if self.min_connection_pax_min >= self.max_connection_pax_min:
            raise ConfigError("network.min_connection_pax_min must be below max_connection_pax_min")
        _non_negative("network.spare_aircraft", self.spare_aircraft)
        _positive("network.turn_min_hub", self.turn_min_hub)
        _positive("network.turn_min_spoke", self.turn_min_spoke)
        if self.spoke_turn_scheduled_min < self.turn_min_spoke:
            raise ConfigError("network.spoke_turn_scheduled_min must be at least turn_min_spoke")


@dataclass(frozen=True, slots=True)
class CrewConfig:
    """Operator rules for crew connections and pairings (not regulatory limits)."""

    change_aircraft_min: int
    max_sit_min: int
    max_duties: int
    max_rest_min: int
    reserve_crews_per_day: int
    reserve_callout_min: int
    max_sectors: int

    def __post_init__(self) -> None:
        _positive("crew.change_aircraft_min", self.change_aircraft_min)
        if self.max_sit_min <= self.change_aircraft_min:
            raise ConfigError("crew.max_sit_min must exceed crew.change_aircraft_min")
        if self.max_duties not in (1, 2):
            raise ConfigError("crew.max_duties must be 1 or 2 in this implementation")
        _positive("crew.max_rest_min", self.max_rest_min)
        _non_negative("crew.reserve_crews_per_day", self.reserve_crews_per_day)
        _non_negative("crew.reserve_callout_min", self.reserve_callout_min)
        if not 1 <= self.max_sectors <= 10:
            raise ConfigError("crew.max_sectors must be 1-10 (EASA Table 2 covers up to 10 sectors)")


@dataclass(frozen=True, slots=True)
class MaintenanceConfig:
    """Overnight maintenance checks at the base."""

    base: str
    max_nights_between: int
    min_ground_min: int

    def __post_init__(self) -> None:
        if self.max_nights_between < 1:
            raise ConfigError("maintenance.max_nights_between must be at least 1")
        _positive("maintenance.min_ground_min", self.min_ground_min)


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Cost parameters in euros. Illustrative assumptions unless a source is given in the README."""

    aircraft_day_eur: float
    crew_hour_eur: float
    min_duty_paid_h: float
    hotel_layover_eur: float
    deadhead_eur: float
    reserve_callout_eur: float
    ferry_block_hour_eur: float
    delay_minute_eur: float
    eurocontrol_minute_eur: float
    eu261_short_eur: float
    eu261_medium_eur: float
    care_meal_eur: float
    hotel_pax_eur: float
    rebooking_eur: float
    overtime_factor: float
    utilization_tiebreak_eur_per_min: float

    def __post_init__(self) -> None:
        for f in fields(self):
            _non_negative(f"costs.{f.name}", getattr(self, f.name))


@dataclass(frozen=True, slots=True)
class DelayConfig:
    """Primary (non-reactionary) delay model.

    A flight receives a primary departure delay with probability ``p_delayed``; the
    delay is log-normal with parameters chosen so that the mean primary delay per
    flight equals ``mean_primary_min``. Block times vary multiplicatively.
    """

    mean_primary_min: float
    p_delayed: float
    sigma: float
    block_noise_sd: float
    block_bias: float

    def __post_init__(self) -> None:
        _positive("delays.mean_primary_min", self.mean_primary_min)
        if not 0 < self.p_delayed <= 1:
            raise ConfigError("delays.p_delayed must be in (0, 1]")
        _positive("delays.sigma", self.sigma)
        _non_negative("delays.block_noise_sd", self.block_noise_sd)
        if not -0.2 < self.block_bias < 0.2:
            raise ConfigError("delays.block_bias must be within (-0.2, 0.2)")


@dataclass(frozen=True, slots=True)
class RobustConfig:
    """Knobs of the robust planner."""

    buffer_min: int
    penalty_weight: float
    penalty_samples: int

    def __post_init__(self) -> None:
        _non_negative("robust.buffer_min", self.buffer_min)
        _non_negative("robust.penalty_weight", self.penalty_weight)
        _positive("robust.penalty_samples", self.penalty_samples)


@dataclass(frozen=True, slots=True)
class PlanningConfig:
    """Solver settings."""

    time_limit_s: float
    mip_gap: float
    cg_max_iterations: int
    cg_columns_per_iteration: int
    cg_tolerance: float
    maintenance_cut_rounds: int
    threads: int

    def __post_init__(self) -> None:
        _positive("planning.time_limit_s", self.time_limit_s)
        _non_negative("planning.mip_gap", self.mip_gap)
        _positive("planning.cg_max_iterations", self.cg_max_iterations)
        _positive("planning.cg_columns_per_iteration", self.cg_columns_per_iteration)
        _positive("planning.cg_tolerance", self.cg_tolerance)
        _positive("planning.maintenance_cut_rounds", self.maintenance_cut_rounds)
        _positive("planning.threads", self.threads)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Rule-based recovery used identically for every plan."""

    horizon_days: int
    max_delay_min: int
    swap_threshold_min: int
    allow_commander_discretion: bool
    rebooking_horizon_min: int

    def __post_init__(self) -> None:
        if not 1 <= self.horizon_days <= 4:
            raise ConfigError("simulation.horizon_days must be 1-4")
        _positive("simulation.max_delay_min", self.max_delay_min)
        _non_negative("simulation.swap_threshold_min", self.swap_threshold_min)
        _positive("simulation.rebooking_horizon_min", self.rebooking_horizon_min)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Replications, seeds and sweeps."""

    name: str
    seed: int
    replications: int
    network_seeds: tuple[int, ...]
    buffer_sweep: tuple[int, ...]
    penalty_sweep: tuple[float, ...]
    delay_climates: tuple[float, ...]
    calibration_replications: int
    history_replications: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("experiment.name must not be empty")
        _positive("experiment.replications", self.replications)
        if not self.network_seeds:
            raise ConfigError("experiment.network_seeds must not be empty")
        if any(b < 0 for b in self.buffer_sweep):
            raise ConfigError("experiment.buffer_sweep values must be non-negative")
        if any(w < 0 for w in self.penalty_sweep):
            raise ConfigError("experiment.penalty_sweep values must be non-negative")
        if any(m <= 0 for m in self.delay_climates):
            raise ConfigError("experiment.delay_climates values must be positive")
        _positive("experiment.calibration_replications", self.calibration_replications)
        _positive("experiment.history_replications", self.history_replications)


@dataclass(frozen=True, slots=True)
class Config:
    """The full configuration of one experiment run."""

    experiment: ExperimentConfig
    network: NetworkConfig
    crew: CrewConfig
    maintenance: MaintenanceConfig
    costs: CostConfig
    delays: DelayConfig
    robust: RobustConfig
    planning: PlanningConfig
    simulation: SimulationConfig
    source: str = field(default="<memory>")

    def __post_init__(self) -> None:
        if self.maintenance.base != self.network.hub:
            raise ConfigError("maintenance.base must be the hub in this single-base model")


def _build[T](cls: type[T], data: Any, path: str) -> T:
    """Recursively construct a frozen dataclass from parsed TOML, checking keys and types."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a table, got {type(data).__name__}")
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls) if f.init and f.name != "source"}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name == "source" or not f.init:
            continue
        if f.name not in data:
            raise ConfigError(f"{path}: missing key {f.name!r}")
        kwargs[f.name] = _coerce(hints[f.name], data[f.name], f"{path}.{f.name}")
    return cls(**kwargs)


def _coerce(hint: Any, value: Any, path: str) -> Any:
    origin = get_origin(hint)
    if is_dataclass(hint) and isinstance(hint, type):
        return _build(hint, value, path)
    if origin is tuple:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected an array")
        args = get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            raise ConfigError(f"{path}: expected {len(args)} items, got {len(value)}")
        return tuple(_coerce(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value, strict=True)))
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{path}: expected a number")
        return float(value)
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer")
        return value
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true or false")
        return value
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string")
        return value
    raise ConfigError(f"{path}: unsupported type {hint!r}")


def load_config(path: str | Path) -> Config:
    """Read and validate a TOML configuration file.

    Raises:
        ConfigError: When the file is missing, is not valid TOML, or fails validation.

    """
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"configuration file not found: {file}")
    try:
        data = tomllib.loads(file.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{file}: invalid TOML: {exc}") from exc
    config = _build(Config, data, file.name)
    return Config(
        experiment=config.experiment,
        network=config.network,
        crew=config.crew,
        maintenance=config.maintenance,
        costs=config.costs,
        delays=config.delays,
        robust=config.robust,
        planning=config.planning,
        simulation=config.simulation,
        source=str(file),
    )
