"""Downloading OMIE's public files and reading them (format: METHODS.md, section 2).

Quantities are read exactly as integer tenths of a MW, and prices as integer cents of a
euro per MWh, so that sums and comparisons carry no rounding.
"""

from __future__ import annotations

import hashlib
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from iberian_lp.contract import DOWNLOAD_URL, USER_AGENT

PERIOD = re.compile(r"^H(\d{1,2})Q([1-4])$")
PAUSE_S = 1.0                              # between requests to omie.es


def _get(kind: str, name: str, attempts: int = 3) -> bytes | None:
    """The file's bytes, or None if OMIE has no such file (404). A dropped connection is
    retried after 30 s, then 60 s."""
    request = urllib.request.Request(DOWNLOAD_URL.format(kind=kind, name=name),
                                     headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        time.sleep(PAUSE_S if attempt == 0 else 30 * attempt)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise ValueError(f"OMIE returned a web page for {name}, not a file.")
                body: bytes = response.read()
                return body
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if attempt == attempts - 1:
                raise
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == attempts - 1:
                raise
    raise AssertionError("unreachable")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(kind: str, day: date, raw: Path) -> dict[str, Any]:
    """The highest published version of one day's file (section 3): versions 1, 2, ... are
    tried until OMIE answers 404. A version already on disk is not fetched again."""
    stem = f"{kind}_{day:%Y%m%d}"
    raw.mkdir(parents=True, exist_ok=True)
    best: Path | None = None
    version = 1
    while True:
        path = raw / f"{stem}.{version}"
        if not path.exists():
            body = _get(kind, path.name)
            if body is None:
                break
            path.write_bytes(body)
        best = path
        version += 1
    if best is None:
        raise FileNotFoundError(f"OMIE has no {stem}.")
    for older in raw.glob(f"{stem}.*"):
        if older != best:
            older.unlink()
    return {"kind": kind, "day": day.isoformat(), "file": best.name,
            "bytes": best.stat().st_size, "sha256": sha256(best)}


def tenths(text: str) -> int:
    """'2.150,8' -> 21508 (tenths of a MW); '-4.373,0' -> -43730."""
    sign = -1 if text.startswith("-") else 1
    whole, _, frac = text.lstrip("-").replace(".", "").partition(",")
    if len(frac) > 1:
        raise ValueError(f"More than one decimal in a quantity: {text!r}.")
    return sign * (int(whole or "0") * 10 + int((frac + "0")[0]))


def cents(text: str, decimal: str = ",") -> int:
    """'1.500,00' -> 150000; with ``decimal="."`` (marginalpdbc), '219.86' -> 21986."""
    sign = -1 if text.startswith("-") else 1
    body = text.lstrip("-")
    if decimal == ",":
        body = body.replace(".", "")
    whole, _, frac = body.partition(decimal)
    if len(frac) > 2:
        raise ValueError(f"More than two decimals in a price: {text!r}.")
    return sign * (int(whole or "0") * 100 + int((frac + "00")[:2]))


def period_number(label: str) -> tuple[int, int, int]:
    """'H3Q2' -> (10, 3, 2): the period's number within the day, its hour and quarter."""
    m = PERIOD.match(label)
    if m is None:
        raise ValueError(f"Not a 15-minute period: {label!r}.")
    hour, quarter = int(m.group(1)), int(m.group(2))
    return 4 * (hour - 1) + quarter, hour, quarter


@dataclass(frozen=True)
class Step:
    period: int
    hour: int
    quarter: int
    zone: str          # MI, ES or PT
    side: str          # "V" sell, "C" buy
    tenths: int        # MW, in tenths
    cents: int         # EUR/MWh, in cents
    status: str        # "O" offered, "C" matched
    kind: str          # S, C01, C02, C04, or a flow label such as "Imp FR"


def read_curve(path: Path, day: date) -> Iterator[Step]:
    """Every step of a ``curva_pbc`` file. The first three lines are a title, a blank line
    and the column names; the file ends with a line of separators."""
    with path.open(encoding="latin-1") as stream:
        lines = stream.read().splitlines()
    if not lines[2].startswith("Periodo;Fecha;Pais;"):
        raise ValueError(f"{path.name}: unexpected header {lines[2]!r}.")
    stamp = f"{day:%d/%m/%Y}"
    for line in lines[3:]:
        f = line.split(";")
        if not f[0]:
            continue                                   # the closing line
        if f[1] != stamp:
            raise ValueError(f"{path.name}: a row dated {f[1]}.")
        number, hour, quarter = period_number(f[0])
        yield Step(number, hour, quarter, f[2], f[4], tenths(f[5]), cents(f[6]), f[7], f[8])


def read_prices(path: Path, day: date) -> dict[int, tuple[int, int]]:
    """``marginalpdbc``: period -> (Portuguese price, Spanish price), in cents."""
    out: dict[int, tuple[int, int]] = {}
    with path.open(encoding="latin-1") as stream:
        lines = stream.read().splitlines()
    if lines[0].strip() != "MARGINALPDBC;":
        raise ValueError(f"{path.name}: unexpected first line {lines[0]!r}.")
    for line in lines[1:]:
        f = line.split(";")
        if f[0] == "*" or not f[0]:
            continue
        if (int(f[0]), int(f[1]), int(f[2])) != (day.year, day.month, day.day):
            raise ValueError(f"{path.name}: a row dated {f[:3]}.")
        out[int(f[3])] = (cents(f[4], "."), cents(f[5], "."))
    return out


def read_capacity(path: Path, border: str) -> dict[int, dict[str, int]]:
    """``capacidad_inter_pbc``: period -> the border's capacities and occupations, in
    tenths of a MW (imports negative, as OMIE writes them)."""
    with path.open(encoding="latin-1") as stream:
        lines = stream.read().splitlines()
    names = ("import_capacity", "import_occupied", "import_free", "export_capacity",
             "export_occupied", "export_free")
    out: dict[int, dict[str, int]] = {}
    for line in lines[3:]:
        f = line.split(";")
        if f[0] and f[2] == border:
            number, _, _ = period_number(f[0])
            out[number] = {n: tenths(x) for n, x in zip(names, f[3:9], strict=True)}
    return out
