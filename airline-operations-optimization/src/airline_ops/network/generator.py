"""Generate the synthetic hub-and-spoke schedule and its passengers.

The structure uses two common outstation patterns: a daytime out-and-back from the
hub with a short turn at the outstation, and a "night-stop" pair in which the aircraft
leaves the hub in the evening, spends the night at the outstation and returns early the
next morning. Spokes with two or more daily frequencies get one night-stop pair; the
remaining frequencies are daytime out-and-backs departing the hub in banks.

Block times come from great-circle distance with direction-dependent speeds calibrated
to typical scheduled block times on the Munich–Porto sector (180 minutes westbound, 170
eastbound); the calibration is in :func:`block_minutes`. Flights and passengers are
synthetic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from airline_ops.config import NetworkConfig, parse_clock
from airline_ops.data.airports import AirportRecord, great_circle_km, utc_offset_minutes
from airline_ops.errors import DataError
from airline_ops.logging_setup import get_logger
from airline_ops.network.model import DAY, Airport, Flight, Network, PassengerGroup

LOG = get_logger("network.generator")

# Calibration of the block-time model (see module docstring).
TAXI_AND_GROUND_MIN = 30.0
CALIBRATION_DISTANCE_KM = 1792.0  # Munich-Porto great-circle distance, rounded
CALIBRATION_WEST_MIN = 180.0  # typical scheduled block, Munich to Porto
CALIBRATION_EAST_MIN = 170.0  # typical scheduled block, Porto to Munich
SPEED_WEST_KMH = CALIBRATION_DISTANCE_KM / ((CALIBRATION_WEST_MIN - TAXI_AND_GROUND_MIN) / 60.0)
SPEED_EAST_KMH = CALIBRATION_DISTANCE_KM / ((CALIBRATION_EAST_MIN - TAXI_AND_GROUND_MIN) / 60.0)


def block_minutes(distance_km: float, westbound: bool) -> int:
    """Scheduled block time, rounded up to five minutes."""
    speed = SPEED_WEST_KMH if westbound else SPEED_EAST_KMH
    minutes = TAXI_AND_GROUND_MIN + distance_km / speed * 60.0
    return int(5 * math.ceil(minutes / 5.0))


def _local_to_utc(local_minute: int, offset: int) -> int:
    return (local_minute - offset) % DAY


def build_airports(
    config: NetworkConfig, records: Mapping[str, AirportRecord]
) -> tuple[dict[str, Airport], dict[str, int]]:
    """Airports with UTC offsets on the reference date, and minimum turn times."""
    airports: dict[str, Airport] = {}
    turns: dict[str, int] = {}
    codes = [config.hub, *(s.code for s in config.spokes)]
    for code in codes:
        rec = records[code]
        offset = utc_offset_minutes(rec.timezone, config.reference_date)
        curfew = None
        if code == config.hub:
            start = _local_to_utc(parse_clock(config.hub_curfew_local[0]), offset)
            end = _local_to_utc(parse_clock(config.hub_curfew_local[1]), offset)
            curfew = (start, end)
        airports[code] = Airport(
            code=code,
            name=rec.name,
            latitude=rec.latitude,
            longitude=rec.longitude,
            utc_offset_min=offset,
            is_hub=code == config.hub,
            curfew_utc=curfew,
        )
        turns[code] = config.turn_min_hub if code == config.hub else config.turn_min_spoke
    return airports, turns


def generate_network(
    config: NetworkConfig,
    records: Mapping[str, AirportRecord],
    seed: int,
) -> Network:
    """Build the daily schedule, passengers and fleet size for one network seed.

    Raises:
        DataError: When the configuration cannot produce a schedule that respects the
            hub curfew.

    """
    rng = np.random.default_rng(seed)
    airports, turns = build_airports(config, records)
    hub = airports[config.hub]
    banks = sorted(parse_clock(b) for b in config.banks_local)
    raw: list[tuple[str, str, int, int, float]] = []  # origin, destination, dep_utc, block, distance

    for spoke_cfg in config.spokes:
        spoke = airports[spoke_cfg.code]
        dist = great_circle_km(hub.latitude, hub.longitude, spoke.latitude, spoke.longitude)
        out_west = spoke.longitude < hub.longitude
        block_out = block_minutes(dist, westbound=out_west)
        block_back = block_minutes(dist, westbound=not out_west)
        n_day = spoke_cfg.frequency - 1 if spoke_cfg.frequency >= 2 else 1
        if spoke_cfg.frequency >= 2:
            jitter = int(rng.integers(0, config.bank_jitter_min + 1))
            out_dep = _local_to_utc(parse_clock(config.nightstop_departure_local) + jitter, hub.utc_offset_min)
            back_jitter = int(rng.integers(0, config.bank_jitter_min + 1))
            back_dep = _local_to_utc(parse_clock(config.morning_departure_local) + back_jitter, spoke.utc_offset_min)
            raw.append((hub.code, spoke.code, out_dep, block_out, dist))
            raw.append((spoke.code, hub.code, back_dep, block_back, dist))
        chosen = sorted(rng.choice(len(banks), size=min(n_day, len(banks)), replace=False).tolist())
        for bank_idx in chosen:
            placed = False
            for candidate in [bank_idx, *range(bank_idx - 1, -1, -1)]:
                jitter = int(rng.integers(0, config.bank_jitter_min + 1))
                out_local = banks[candidate] + jitter
                out_dep = _local_to_utc(out_local, hub.utc_offset_min)
                turn = config.spoke_turn_scheduled_min + int(rng.integers(0, 11))
                back_dep = (out_dep + block_out + turn) % DAY
                back_arr = back_dep + block_back
                if hub.in_curfew(back_arr) or hub.in_curfew(back_arr - 30):
                    continue
                raw.append((hub.code, spoke.code, out_dep, block_out, dist))
                raw.append((spoke.code, hub.code, back_dep, block_back, dist))
                placed = True
                break
            if not placed:
                raise DataError(f"cannot place a daytime round trip to {spoke.code} outside the hub curfew")

    raw.sort(key=lambda r: (r[2], r[0], r[1]))
    flights = tuple(
        Flight(
            idx=i,
            number=f"S{100 + i}",
            origin=o,
            destination=d,
            dep=dep,
            block=blk,
            distance_km=round(dist, 1),
            seats=config.seats,
        )
        for i, (o, d, dep, blk, dist) in enumerate(raw)
    )
    passengers = _generate_passengers(config, flights, airports, rng)
    provisional = Network(
        airports=airports,
        flights=flights,
        hub=config.hub,
        fleet_size=1,
        passengers=passengers,
        turn_min=turns,
        seed=seed,
    )
    fleet = provisional.minimum_fleet() + config.spare_aircraft
    network = Network(
        airports=airports,
        flights=flights,
        hub=config.hub,
        fleet_size=fleet,
        passengers=passengers,
        turn_min=turns,
        seed=seed,
    )
    LOG.info(
        "generated network",
        extra={
            "fields": {
                "seed": seed,
                "flights": len(flights),
                "fleet": fleet,
                "passenger_groups": len(passengers),
                "passengers": sum(g.count for g in passengers),
            }
        },
    )
    return network


def _generate_passengers(
    config: NetworkConfig,
    flights: tuple[Flight, ...],
    airports: Mapping[str, Airport],
    rng: np.random.Generator,
) -> tuple[PassengerGroup, ...]:
    """Local passengers on every flight and connecting passengers through the hub."""
    booked = round(config.seats * config.load_factor)
    conn_target = round(booked * config.connecting_share)
    inbound = [f for f in flights if f.destination == config.hub]
    outbound = [f for f in flights if f.origin == config.hub]
    out_capacity = {f.idx: conn_target for f in outbound}
    conn_on = [0] * len(flights)
    groups: list[PassengerGroup] = []
    order = rng.permutation(len(inbound))
    for k in order:
        i = inbound[int(k)]
        cands = [
            j
            for j in outbound
            if j.destination != i.origin
            and config.min_connection_pax_min <= (j.dep - i.arr) % DAY <= config.max_connection_pax_min
            and out_capacity[j.idx] > 0
        ]
        if not cands:
            continue
        weights = np.array([out_capacity[j.idx] for j in cands], dtype=float)
        counts = rng.multinomial(conn_target, weights / weights.sum())
        for j, c in zip(cands, counts, strict=True):
            c = int(min(c, out_capacity[j.idx]))
            if c <= 0:
                continue
            out_capacity[j.idx] -= c
            conn_on[i.idx] += c
            conn_on[j.idx] += c
            o, d = airports[i.origin], airports[j.destination]
            groups.append(
                PassengerGroup(
                    idx=0,
                    legs=(i.idx, j.idx),
                    count=c,
                    origin=i.origin,
                    destination=j.destination,
                    distance_km=round(great_circle_km(o.latitude, o.longitude, d.latitude, d.longitude), 1),
                )
            )
    for f in flights:
        local = booked - conn_on[f.idx]
        if local > 0:
            groups.append(
                PassengerGroup(
                    idx=0,
                    legs=(f.idx,),
                    count=local,
                    origin=f.origin,
                    destination=f.destination,
                    distance_km=f.distance_km,
                )
            )
    return tuple(
        PassengerGroup(
            idx=n, legs=g.legs, count=g.count, origin=g.origin, destination=g.destination, distance_km=g.distance_km
        )
        for n, g in enumerate(groups)
    )
