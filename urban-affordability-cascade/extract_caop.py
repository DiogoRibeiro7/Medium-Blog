"""Extract fixed AMP municipal polygon centroids from official DGT CAOP2025.

Uses only the standard library. The official CAOP2025 OGC API is optional at
runtime; reproducible analyses use a checked-in 17-row frozen derivative.
Projected EPSG:3763 polygon centroids represent *administrative area*, not
population centers, road distance, transport access or residential movement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "data" / "amp_municipal_panel_2019_2025.csv"
SOURCE_URL = (
    "https://ogcapi.dgterritorio.gov.pt/collections/municipios/items?"
    "f=json&nuts3_cod=11A&limit=100&"
    "crs=http%3A%2F%2Fwww.opengis.net%2Fdef%2Fcrs%2FEPSG%2F0%2F3763"
)
SNAPSHOT = ROOT / "data" / "amp_caop2025_municipal_centroids.csv"
PROVENANCE = ROOT / "data" / "amp_caop2025_municipal_centroids_provenance.json"
FIELDS = ("municipality", "municipality_code", "centroid_easting_epsg3763_m",
          "centroid_northing_epsg3763_m", "official_area_ha")


def panel_municipality_codes(panel_path: Path = PANEL) -> dict[str, str]:
    """Obtain the 17 fixed AMP DICO identities without scientific dependencies."""
    mapping: dict[str, str] = {}
    municipality_years: set[tuple[str, int]] = set()
    with panel_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"dico", "municipality", "nuts3", "year"}.issubset(reader.fieldnames or ()):
            raise ValueError("Expected fixed municipal DICO panel columns")
        for row in reader:
            code, name = row["dico"], row["municipality"]
            year = int(row["year"])
            if (not re.fullmatch(r"\d{4}", code) or not name or row["nuts3"] != "11A"
                    or year not in range(2019, 2026) or (code, year) in municipality_years
                    or (code in mapping and mapping[code] != name)):
                raise ValueError("Invalid pinned AMP municipality DICO, name, year or NUTS3")
            municipality_years.add((code, year))
            mapping[code] = name
    if len(mapping) != 17 or len(set(mapping.values())) != 17 or len(municipality_years) != 119:
        raise ValueError("Expected 17 complete AMP municipal time series")
    return mapping


def ring_area_centroid(points: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    """Return unsigned planar ring area and centroid using shoelace moments."""
    if len(points) < 4 or points[0] != points[-1]:
        raise ValueError("Projected polygon ring must be closed and nontrivial")
    xy: list[tuple[float, float]] = []
    for point in points:
        if len(point) < 2:
            raise ValueError("Malformed projected polygon vertex")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Nonfinite projected polygon vertex")
        xy.append((x, y))
    # Centering reduces floating-point cancellation in coordinate moments.
    anchor_x, anchor_y = xy[0]
    cross: list[float] = []
    moment_x: list[float] = []
    moment_y: list[float] = []
    for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
        ax, ay = x0 - anchor_x, y0 - anchor_y
        bx, by = x1 - anchor_x, y1 - anchor_y
        c = ax * by - bx * ay
        cross.append(c)
        moment_x.append((ax + bx) * c)
        moment_y.append((ay + by) * c)
    signed_twice_area = math.fsum(cross)
    if abs(signed_twice_area) <= 1e-8:
        raise ValueError("Degenerate municipal polygon ring")
    return (abs(signed_twice_area) / 2,
            anchor_x + math.fsum(moment_x) / (3 * signed_twice_area),
            anchor_y + math.fsum(moment_y) / (3 * signed_twice_area))


def multipolygon_area_centroid(coordinates: Any) -> tuple[float, float, float]:
    """Combine exterior rings minus holes, including disjoint islands."""
    weighted_x, weighted_y, net_area = [], [], []
    if not isinstance(coordinates, list) or not coordinates:
        raise ValueError("Expected a nonempty MultiPolygon")
    for polygon in coordinates:
        if not isinstance(polygon, list) or not polygon:
            raise ValueError("Expected polygon rings")
        for i, ring in enumerate(polygon):
            area, x, y = ring_area_centroid(ring)
            signed_area = area if i == 0 else -area
            net_area.append(signed_area)
            weighted_x.append(signed_area * x)
            weighted_y.append(signed_area * y)
    total = math.fsum(net_area)
    if total <= 0:
        raise ValueError("Non-positive multipolygon net area")
    return total, math.fsum(weighted_x) / total, math.fsum(weighted_y) / total


def parse_caop(source_bytes: bytes) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Strictly validate the official API response before forming a derivative."""
    try:
        source = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid CAOP API JSON") from exc
    if not isinstance(source, dict):
        raise ValueError("Expected a CAOP FeatureCollection object")
    features = source.get("features")
    if (source.get("type") != "FeatureCollection" or not isinstance(features, list)
            or len(features) != 17 or source.get("numberReturned") != 17
            or source.get("numberMatched") != 17):
        raise ValueError("Expected exactly 17 CAOP2025 AMP features")
    codes = panel_municipality_codes()
    expected_codes = {name: code for code, name in codes.items()}
    rows: list[dict[str, str]] = []
    for feature in features:
        properties = feature.get("properties", {})
        geom = feature.get("geometry", {})
        municipality = properties.get("municipio")
        code = str(properties.get("dtmn", ""))
        if (municipality not in expected_codes or code != expected_codes[municipality]
                or str(feature.get("id")) != code or properties.get("nuts3_cod") != "11A"
                or geom.get("type") != "MultiPolygon"):
            raise ValueError("CAOP municipality identity, NUTS3 or geometry mismatch")
        area_m2, east, north = multipolygon_area_centroid(geom.get("coordinates"))
        official_area_ha = float(properties["area_ha"])
        if not (0 < official_area_ha < 1e6 and
                abs(area_m2 / 10000 - official_area_ha) / official_area_ha < .025):
            raise ValueError("Official municipal area disagrees with EPSG:3763 geometry")
        rows.append({
            "municipality": municipality,
            "municipality_code": code,
            "centroid_easting_epsg3763_m": f"{east:.3f}",
            "centroid_northing_epsg3763_m": f"{north:.3f}",
            "official_area_ha": f"{official_area_ha:.2f}",
        })
    if len({r["municipality"] for r in rows}) != 17:
        raise ValueError("Duplicate or missing AMP municipality")
    rows.sort(key=lambda row: row["municipality"])
    report: dict[str, object] = {
        "source_name": "DGT Carta Administrativa Oficial de Portugal, CAOP2025 Municípios",
        "source_url": SOURCE_URL,
        "source_response_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_license": "CC-BY 4.0; attribution: Direção-Geral do Território",
        "asserted_version_year": 2025,
        "asserted_publication_date": "2026-02-18",
        "asserted_dates_basis": "DGT official CAOP2025 landing page; not attributes of the OGC feature response",
        "asserted_dates_source_url": "https://www.dgterritorio.gov.pt/atividades/cartografia/cartografia-tematica/caop",
        "raw_response_retention": "Response bytes not committed; the recorded response SHA-256 requires an independent re-download to verify",
        "source_epsg": 3763,
        "source_nuts3_code": "11A",
        "records": len(rows),
        "method": "Area-weighted planar centroids of official municipal MultiPolygon rings in EPSG:3763; holes subtracted",
        "not_a_measure_of": ["household location", "urban-centre distance", "road route", "travel time", "migration"],
        "temporal_caveat": "CAOP2025 boundaries are a geographic reference, not annual 2020-2025 boundary vintages",
    }
    return rows, report


def serialize_csv(rows: Sequence[dict[str, str]]) -> bytes:
    """Produce exact, stable derivative bytes for source pinning."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def extract(source_bytes: bytes) -> tuple[bytes, bytes]:
    """Return the CSV and its metadata without writing side effects."""
    rows, report = parse_caop(source_bytes)
    csv_bytes = serialize_csv(rows)
    report["derived_csv_sha256"] = hashlib.sha256(csv_bytes).hexdigest()
    return csv_bytes, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    """Refresh explicitly from the source, or parse supplied API bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--download", action="store_true", help="Fetch official API; requires network")
    group.add_argument("--stdin", action="store_true", help="Read official response JSON from stdin")
    parser.add_argument("--print-snapshot", action="store_true", help="Print both derived files as JSON")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args(argv)
    if args.download:
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "urban-affordability-cascade-research/1.0"})
        with urllib.request.urlopen(request, timeout=40) as response:
            raw = response.read(25_000_001)
        if len(raw) > 25_000_000:
            raise ValueError("CAOP response exceeds bounded 25 MB download")
    else:
        raw = sys.stdin.buffer.read(25_000_001)
        if len(raw) > 25_000_000:
            raise ValueError("CAOP stdin input exceeds bounded 25 MB")
    derived, metadata = extract(raw)
    if args.print_snapshot:
        print(json.dumps({"csv": derived.decode("utf-8"), "provenance": metadata.decode("utf-8")}, ensure_ascii=False))
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / SNAPSHOT.name).write_bytes(derived)
        (args.output / PROVENANCE.name).write_bytes(metadata)
        print(f"Derived {len(derived)} CSV bytes from 17 official DGT municipal polygons")


if __name__ == "__main__":
    main()
