from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from airline_ops.config import Config, load_config
from airline_ops.data.airports import AirportRecord
from airline_ops.network.generator import generate_network
from airline_ops.network.model import Network

ROOT = Path(__file__).resolve().parents[1]

# Coordinates and time zones of the smoke-test airports, copied from OpenFlights for an offline fixture.
RECORDS = {
    "MUC": AirportRecord("MUC", "Munich Airport", 48.353802, 11.7861, "Europe/Berlin"),
    "OPO": AirportRecord("OPO", "Francisco de Sá Carneiro Airport", 41.2481002808, -8.68138980865, "Europe/Lisbon"),
    "LIS": AirportRecord("LIS", "Humberto Delgado Airport", 38.7813, -9.13592, "Europe/Lisbon"),
    "MAD": AirportRecord("MAD", "Adolfo Suárez Madrid–Barajas Airport", 40.471926, -3.56264, "Europe/Madrid"),
    "CDG": AirportRecord("CDG", "Charles de Gaulle International Airport", 49.012798, 2.55, "Europe/Paris"),
    "HAM": AirportRecord("HAM", "Hamburg Airport", 53.630401611328, 9.9882297515869, "Europe/Berlin"),
    "VIE": AirportRecord("VIE", "Vienna International Airport", 48.110298156738, 16.569700241089, "Europe/Vienna"),
}


@pytest.fixture(scope="session")
def smoke_config() -> Config:
    return load_config(ROOT / "configs" / "smoke.toml")


@pytest.fixture(scope="session")
def network(smoke_config: Config) -> Network:
    return generate_network(smoke_config.network, RECORDS, 11)


@pytest.fixture(scope="session")
def quiet_config(smoke_config: Config) -> Config:
    """No primary delays and exact block times."""
    return replace(smoke_config, delays=replace(smoke_config.delays, block_noise_sd=0.0, block_bias=0.0))
