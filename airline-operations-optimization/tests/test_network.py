from collections import Counter

import pytest

from airline_ops.config import Config
from airline_ops.data.airports import AirportRecord, great_circle_km, parse_airports, utc_offset_minutes
from airline_ops.errors import DataError
from airline_ops.network.generator import block_minutes
from airline_ops.network.model import Network, crosses_count_line
from airline_ops.planning.arcs import aircraft_count
from airline_ops.planning.greedy import fifo_routing
from airline_ops.planning.rules import PlanningRules

from .conftest import RECORDS


def test_block_model_reproduces_calibration_route() -> None:
    muc, opo = RECORDS["MUC"], RECORDS["OPO"]
    d = great_circle_km(muc.latitude, muc.longitude, opo.latitude, opo.longitude)
    assert 1780 < d < 1800
    assert block_minutes(d, westbound=True) == 180
    assert block_minutes(d, westbound=False) == 170


def test_offsets_on_reference_date() -> None:
    assert utc_offset_minutes("Europe/Berlin", "2026-11-26") == 60
    assert utc_offset_minutes("Europe/Lisbon", "2026-11-26") == 0
    with pytest.raises(DataError):
        utc_offset_minutes("Mars/Olympus", "2026-11-26")


def test_schedule_is_balanced(network: Network) -> None:
    out = Counter(f.origin for f in network.flights)
    inn = Counter(f.destination for f in network.flights)
    assert out == inn
    assert all(f.origin == network.hub or f.destination == network.hub for f in network.flights)


def test_minimum_fleet_matches_fifo(network: Network, smoke_config: Config) -> None:
    rules = PlanningRules.from_config(smoke_config, network)
    succ = fifo_routing(network, rules)
    assert aircraft_count(succ, network) == network.minimum_fleet()
    assert network.fleet_size == network.minimum_fleet() + smoke_config.network.spare_aircraft


def test_passengers_fit_in_seats(network: Network, smoke_config: Config) -> None:
    load = network.pax_on_flight()
    assert max(load) <= smoke_config.network.seats
    assert all(len(g.legs) in (1, 2) for g in network.passengers)


def test_count_line_crossings() -> None:
    assert crosses_count_line(100, 50) == 1  # 01:40 to 02:30 crosses 02:00
    assert crosses_count_line(200, 50) == 0
    assert crosses_count_line(1400, 200) == 1
    assert crosses_count_line(100, 1440 * 2) == 2


def test_parse_airports_rejects_missing_code() -> None:
    row = b'1,"X","X","X","AAA","XXXX",10.0,20.0,0,1,"E","Europe/Lisbon","airport","OurAirports"\n'
    assert parse_airports(row, {"AAA"})["AAA"] == AirportRecord("AAA", "X", 10.0, 20.0, "Europe/Lisbon")
    with pytest.raises(DataError, match="not found"):
        parse_airports(row, {"BBB"})
