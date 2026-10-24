"""Region codes harmonised to NUTS 2021 (METHODS.md, section 3.1, and its change log).

A crop series published under a code joins a NUTS-2021 NUTS-2 region when:

- the code is that region's code (``nuts2021``);
- it is a NUTS-1 code whose only NUTS-2 region in NUTS 2021 is that region
  (``nuts1_single``);
- Eurostat's official change tables record a chain of pure recodes from it to that
  region (``recode``).

Every other code is dropped. For every NUTS-2 code of another version, the symmetric
difference of its GISCO boundary with the closest NUTS-2021 region is reported as a
cross-check; it decides nothing (change log, 2026-10-23).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import shapely
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from eu_wheat_weather.contract import EU27, NUTS_VERSIONS, TARGET_NUTS, DataError, gisco_file
from eu_wheat_weather.crops import WheatRow
from eu_wheat_weather.nuts_changes import Change, recode_target

RULE_PRIORITY = {"nuts2021": 0, "recode": 1, "nuts1_single": 2}


def load_boundaries(path: Path) -> dict[str, BaseGeometry]:
    """NUTS-2 polygons of one GISCO file, by code (valid geometries)."""
    document = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, BaseGeometry] = {}
    for feature in document["features"]:
        properties = feature["properties"]
        code = properties.get("NUTS_ID")
        if not isinstance(code, str) or int(properties.get("LEVL_CODE", -1)) != 2:
            continue
        out[code] = shapely.make_valid(shape(feature["geometry"]))
    if not out:
        raise DataError(f"No NUTS-2 regions in {path}.")
    return out


def load_all(directory: Path) -> dict[int, dict[str, BaseGeometry]]:
    return {v: load_boundaries(directory / gisco_file(v)) for v in NUTS_VERSIONS}


def target_regions(boundaries: Mapping[int, Mapping[str, BaseGeometry]],
                   ) -> dict[str, BaseGeometry]:
    """EU-27 NUTS-2 regions of NUTS 2021, extra-regio codes excluded."""
    return {code: geometry for code, geometry in boundaries[TARGET_NUTS].items()
            if code[:2] in EU27 and not code.endswith("ZZ")}


@dataclass(frozen=True)
class CodeMatch:
    """Where a published code's series goes, and why."""

    code: str
    target: str | None
    rule: str             # nuts2021, nuts1_single, recode, no_match or no_boundary
    version: int | None   # the latest NUTS version the code belongs to
    share: float | None   # symmetric difference with the closest region / its area
    candidate: str        # the closest NUTS-2021 region, or why the chain stopped


def symmetric_share(a: BaseGeometry, b: BaseGeometry) -> float:
    """Area of the symmetric difference of ``a`` and ``b``, relative to ``b``."""
    return float(shapely.area(shapely.symmetric_difference(a, b)) / shapely.area(b))


def match_codes(codes: Iterable[str], boundaries: Mapping[int, Mapping[str, BaseGeometry]],
                changes: Sequence[Change]) -> list[CodeMatch]:
    targets = target_regions(boundaries)
    names = sorted(targets)
    tree = shapely.STRtree([targets[n] for n in names])
    out = []
    for code in sorted(set(codes)):
        if code in targets:
            out.append(CodeMatch(code, code, "nuts2021", TARGET_NUTS, 0.0, code))
            continue
        if len(code) == 3:
            children = [n for n in names if n.startswith(code)]
            if len(children) == 1:
                out.append(CodeMatch(code, children[0], "nuts1_single", TARGET_NUTS, None,
                                     children[0]))
            else:
                out.append(CodeMatch(code, None, "no_match", TARGET_NUTS, None,
                                     ",".join(children)))
            continue
        versions = [v for v in NUTS_VERSIONS if code in boundaries[v]]
        if not versions:
            out.append(CodeMatch(code, None, "no_boundary", None, None, ""))
            continue
        version = max(versions)
        geometry = boundaries[version][code]
        best, best_share = "", float("inf")
        for i in tree.query(geometry, predicate="intersects"):
            share = symmetric_share(geometry, targets[names[int(i)]])
            if share < best_share:
                best, best_share = names[int(i)], share
        share_or_none = None if not best else best_share
        target, stopped = recode_target(code, version, changes)
        if target is not None and target in targets:
            out.append(CodeMatch(code, target, "recode", version,
                                 symmetric_share(geometry, targets[target]), target))
        else:
            out.append(CodeMatch(code, None, "no_match", version, share_or_none,
                                 f"{best} ({stopped})" if stopped else best))
    return out


@dataclass(frozen=True)
class HarmonisedRow:
    """One NUTS-2021 region and year, from the highest-priority source with a yield."""

    geo: str
    year: int
    source: str
    rule: str
    area_kha: float | None
    production_kt: float | None
    yield_t_ha: float | None
    area_flag: str
    production_flag: str


@dataclass(frozen=True)
class Conflict:
    """Two sources with a yield for the same region and year that differ by over 1%."""

    geo: str
    year: int
    used: str
    other: str
    used_yield: float
    other_yield: float


def harmonise(rows: Sequence[WheatRow], matches: Sequence[CodeMatch],
              ) -> tuple[list[HarmonisedRow], list[Conflict]]:
    """Join every matched series to its NUTS-2021 region (section 3.1).

    For each region and year the source with a yield and the highest priority wins
    (the region's own code, then a coinciding code, then a NUTS-1 code); without any
    yield, the highest-priority row is kept for the report.
    """
    by_code = {m.code: m for m in matches}
    grouped: dict[tuple[str, int], list[tuple[int, WheatRow, CodeMatch]]] = {}
    for row in rows:
        match = by_code.get(row.geo)
        if match is None:
            raise DataError(f"Code {row.geo} has no match record.")
        if match.target is None:
            continue
        grouped.setdefault((match.target, row.year), []).append(
            (RULE_PRIORITY[match.rule], row, match))
    out, conflicts = [], []
    for (geo, year), candidates in sorted(grouped.items()):
        candidates.sort(key=lambda c: (c[1].yield_t_ha is None, c[0], c[1].geo))
        _, row, match = candidates[0]
        out.append(HarmonisedRow(geo, year, row.geo, match.rule, row.area_kha,
                                 row.production_kt, row.yield_t_ha, row.area_flag,
                                 row.production_flag))
        used = row.yield_t_ha
        for _, other, _ in candidates[1:]:
            if used is not None and other.yield_t_ha is not None and \
                    abs(other.yield_t_ha - used) > 0.01 * used:
                conflicts.append(Conflict(geo, year, row.geo, other.geo, used,
                                          other.yield_t_ha))
    return out, conflicts


def match_records(matches: Sequence[CodeMatch]) -> list[dict[str, Any]]:
    return [{"code": m.code, "target": m.target or "", "rule": m.rule,
             "version": "" if m.version is None else m.version,
             "share": "" if m.share is None else round(m.share, 6),
             "candidate": m.candidate} for m in matches]
