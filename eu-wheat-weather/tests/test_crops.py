"""Offline tests of the JSON-stat decoding, the wheat table and the crop snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eu_wheat_weather.contract import AREA, CROP, PRODUCTION, DataError
from eu_wheat_weather.crops import (
    RAW_NAME,
    TABLE_NAME,
    audit_snapshot,
    crops_url,
    is_eu_nuts2,
    save_snapshot,
    wheat_table,
)
from eu_wheat_weather.jsonstat import decode

ROOT = Path(__file__).resolve().parents[1]

GEOS = ["FRB0", "PT18", "DE4", "TRC1", "ESZZ"]
YEARS = ["1999", "2000", "2001"]


def payload(values: dict[str, float], status: dict[str, str] | None = None,
            measures: tuple[str, ...] = (AREA, PRODUCTION)) -> dict[str, Any]:
    """A JSON-stat 2.0 response shaped like ``apro_cpshr``."""
    return {
        "version": "2.0", "class": "dataset",
        "id": ["freq", "crops", "strucpro", "geo", "time"],
        "size": [1, 1, len(measures), len(GEOS), len(YEARS)],
        "dimension": {
            "freq": {"category": {"index": {"A": 0}}},
            "crops": {"category": {"index": {CROP: 0}}},
            "strucpro": {"category": {"index": {m: i for i, m in enumerate(measures)}}},
            "geo": {"category": {"index": {g: i for i, g in enumerate(GEOS)}}},
            "time": {"category": {"index": YEARS}},
        },
        "value": values, "status": status or {},
    }


def flat(measure: int, geo: int, year: int) -> str:
    return str((measure * len(GEOS) + geo) * len(YEARS) + year)


def test_decode_keeps_flags_and_gaps_and_never_invents_zero() -> None:
    p = payload({flat(0, 0, 1): 100.0}, {flat(1, 0, 1): ":c"})
    cells = {(c.codes["strucpro"], c.codes["geo"], c.codes["time"]): c
             for c in decode(p, {"freq": "A", "crops": CROP, "strucpro": (AREA, PRODUCTION)})}
    assert cells[(AREA, "FRB0", "2000")].value == 100.0
    flagged = cells[(PRODUCTION, "FRB0", "2000")]
    assert flagged.value is None and flagged.flag == ":c"
    assert len(cells) == 2


def test_decode_rejects_unrequested_codes_and_dimensions() -> None:
    p = payload({flat(0, 0, 1): 1.0})
    with pytest.raises(DataError, match="not requested"):
        decode(p, {"freq": "A", "crops": CROP, "strucpro": AREA})
    with pytest.raises(DataError, match="Unexpected dimensions"):
        decode(p, {"freq": "A", "strucpro": (AREA, PRODUCTION)})
    p["value"] = {"999": 1.0}
    with pytest.raises(DataError, match="out of range"):
        decode(p, {"freq": "A", "crops": CROP, "strucpro": (AREA, PRODUCTION)})


def test_wheat_table_computes_yield_and_keeps_only_eu_nuts2_in_the_period() -> None:
    p = payload({
        flat(0, 0, 1): 200.0, flat(1, 0, 1): 1500.0,      # FRB0 2000: 7.5 t/ha
        flat(0, 1, 2): 0.0, flat(1, 1, 2): 3.0,           # PT18 2001: zero area
        flat(0, 0, 0): 190.0, flat(1, 0, 0): 1400.0,      # FRB0 1999: outside the period
        flat(0, 2, 1): 5.0, flat(0, 3, 1): 5.0, flat(0, 4, 1): 5.0,  # NUTS-1, Turkey, extra-regio
    }, {flat(1, 0, 1): "p"})
    rows = {(r.geo, r.year): r for r in wheat_table(p)}
    assert set(rows) == {("FRB0", 2000), ("PT18", 2001)}
    fr = rows[("FRB0", 2000)]
    assert fr.yield_t_ha == pytest.approx(7.5) and fr.production_flag == "p"
    assert rows[("PT18", 2001)].yield_t_ha is None


def test_eu_nuts2_codes() -> None:
    assert is_eu_nuts2("FRB0") and is_eu_nuts2("EL52")
    assert not is_eu_nuts2("FR") and not is_eu_nuts2("DE4") and not is_eu_nuts2("TRC1")
    assert not is_eu_nuts2("ESZZ") and not is_eu_nuts2("UKC1")


def test_request_asks_for_the_registered_cells() -> None:
    url = crops_url()
    assert f"crops={CROP}" in url and "geoLevel=nuts2" in url
    assert f"strucpro={AREA}" in url and f"strucpro={PRODUCTION}" in url


def test_snapshot_round_trip_and_tampering(tmp_path: Path) -> None:
    raw = json.dumps(payload({flat(0, 0, 1): 200.0, flat(1, 0, 1): 1500.0})).encode()
    manifest = save_snapshot(tmp_path, raw, "https://example.invalid")
    assert manifest["summary"]["regions_with_yield"] == 1
    assert audit_snapshot(tmp_path) == manifest["summary"]
    table = tmp_path / TABLE_NAME
    table.write_text(table.read_text(encoding="utf-8").replace("7.5", "7.6"),
                     encoding="utf-8", newline="\n")
    with pytest.raises(DataError, match="SHA-256"):
        audit_snapshot(tmp_path)
    save_snapshot(tmp_path, raw, "https://example.invalid")
    (tmp_path / RAW_NAME).write_bytes(raw + b" ")
    with pytest.raises(DataError, match="SHA-256"):
        audit_snapshot(tmp_path)


def test_committed_snapshots_audit() -> None:
    snapshots = sorted((ROOT / "snapshots").glob("eurostat-crops-*"))
    if not snapshots:
        pytest.skip("No committed crop snapshot.")
    for directory in snapshots:
        summary = audit_snapshot(directory)
        assert summary["regions_with_yield"] > 0
