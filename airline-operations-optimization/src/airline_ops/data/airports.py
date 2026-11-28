"""Airport coordinates and time zones from the OpenFlights airport database.

The bytes are never committed. :func:`fetch_airports_file` downloads ``airports.dat``
at the commit pinned in ``data_sources/openflights-airports.json`` (a contract with the
``DiogoRibeiro7/data`` registry), checks its size and SHA-256, and caches it locally.
The database is published under the Open Database License (ODbL) 1.0; most airport
rows are sourced by OpenFlights from OurAirports.

Only four fields are used: IATA code, latitude, longitude and the IANA time zone.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from airline_ops.errors import DataError
from airline_ops.logging_setup import get_logger

LOG = get_logger("data.airports")
EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True, slots=True)
class AirportRecord:
    """The fields of one OpenFlights row that the model uses."""

    iata: str
    name: str
    latitude: float
    longitude: float
    timezone: str


@dataclass(frozen=True, slots=True)
class FileContract:
    """One file listed in a data-source contract."""

    path: str
    url: str
    size: int
    sha256: str


def read_contract(path: Path) -> FileContract:
    """Read the single-file contract for the airport database.

    Raises:
        DataError: When the contract is missing or malformed.

    """
    if not path.is_file():
        raise DataError(f"data contract not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entry = payload["files"][0]
        return FileContract(
            path=str(entry["path"]),
            url=str(entry["url"]),
            size=int(entry["bytes"]),
            sha256=str(entry["sha256"]),
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DataError(f"malformed data contract {path}: {exc}") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_airports_file(contract: FileContract, cache_dir: Path, *, timeout_s: float = 60.0) -> bytes:
    """Return the verified bytes of ``airports.dat``, downloading them if not cached.

    Raises:
        DataError: When the download fails or the bytes do not match the contract.

    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"openflights-airports-{contract.sha256[:12]}.dat"
    if cached.is_file():
        data = cached.read_bytes()
        if len(data) == contract.size and _sha256(data) == contract.sha256:
            return data
        LOG.warning(
            "cached airport file failed its checksum; downloading again", extra={"fields": {"path": str(cached)}}
        )
    if os.environ.get("AIRLINE_OPS_OFFLINE") == "1":
        raise DataError("AIRLINE_OPS_OFFLINE=1 and no verified cached copy of airports.dat")
    try:
        with urllib.request.urlopen(contract.url, timeout=timeout_s) as response:
            data = bytes(response.read())
    except OSError as exc:
        raise DataError(f"could not download {contract.url}: {exc}") from exc
    if len(data) != contract.size or _sha256(data) != contract.sha256:
        raise DataError(
            f"airports.dat does not match the contract: {len(data)} bytes, sha256 {_sha256(data)}; "
            f"expected {contract.size} bytes, sha256 {contract.sha256}"
        )
    cached.write_bytes(data)
    LOG.info("downloaded and verified airports.dat", extra={"fields": {"bytes": len(data)}})
    return data


def parse_airports(data: bytes, codes: set[str]) -> dict[str, AirportRecord]:
    """Extract the requested IATA codes from OpenFlights ``airports.dat`` bytes.

    Raises:
        DataError: When a requested code is missing, duplicated or has invalid fields.

    """
    found: dict[str, AirportRecord] = {}
    reader = csv.reader(io.StringIO(data.decode("utf-8")))
    for row in reader:
        if len(row) < 12:
            continue
        iata = row[4]
        if iata not in codes:
            continue
        if iata in found:
            raise DataError(f"IATA code {iata} appears more than once in airports.dat")
        try:
            lat, lon = float(row[6]), float(row[7])
        except ValueError as exc:
            raise DataError(f"{iata}: invalid coordinates") from exc
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise DataError(f"{iata}: coordinates out of range")
        found[iata] = AirportRecord(iata=iata, name=row[1], latitude=lat, longitude=lon, timezone=row[11])
    missing = codes - set(found)
    if missing:
        raise DataError(f"airport codes not found in airports.dat: {sorted(missing)}")
    return found


def utc_offset_minutes(timezone: str, reference_date: str) -> int:
    """UTC offset of an IANA time zone at noon on the reference date.

    Raises:
        DataError: When the time zone or date is invalid.

    """
    try:
        zone = ZoneInfo(timezone)
        moment = datetime.fromisoformat(f"{reference_date}T12:00:00").replace(tzinfo=zone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise DataError(f"invalid time zone {timezone!r} or date {reference_date!r}") from exc
    offset = moment.utcoffset()
    if offset is None:  # pragma: no cover - ZoneInfo always returns an offset
        raise DataError(f"no UTC offset for {timezone}")
    return int(offset.total_seconds() // 60)


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in kilometres on a sphere of mean Earth radius."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
