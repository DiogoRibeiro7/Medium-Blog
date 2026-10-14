"""Defensive JSON-stat 2.0 decoding; never infer a missing value as zero."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from eu_okun.contract import EU27, QUARTER_RE, EurostatDataError, Observation, SeriesSpec


def _positions(dimension: Mapping[str, Any]) -> dict[str, int]:
    category = dimension.get("category")
    if not isinstance(category, dict):
        raise EurostatDataError("JSON-stat dimension lacks categories.")
    index = category.get("index")
    if isinstance(index, dict):
        if not all(isinstance(k, str) and type(v) is int for k, v in index.items()):
            raise EurostatDataError("Invalid JSON-stat category index.")
        return dict(index)
    if isinstance(index, list) and all(isinstance(k, str) for k in index):
        return {key: i for i, key in enumerate(index)}
    raise EurostatDataError("Unsupported JSON-stat category index.")


def _cell(source: object, index: int) -> object:
    if isinstance(source, dict):
        return source.get(str(index))
    if isinstance(source, list) and index < len(source):
        return source[index]
    return None


def decode_series(
    payload: Mapping[str, Any], spec: SeriesSpec, countries: Sequence[str] = EU27,
) -> dict[tuple[str, str], Observation]:
    """Decode requested country/quarter cells, retaining quality flags and gaps.

    All dimensions other than country/time must match exactly the requested
    singleton filters: accidentally mixing units or adjustments is an error.
    """
    ids, sizes = payload.get("id"), payload.get("size")
    dims, values = payload.get("dimension"), payload.get("value")
    if (
        not isinstance(ids, list) or not all(isinstance(k, str) for k in ids)
        or not isinstance(sizes, list)
        or not all(type(n) is int and n > 0 for n in sizes)
        or len(ids) != len(sizes) or len(set(ids)) != len(ids)
        or not isinstance(dims, dict) or not isinstance(values, (dict, list))
        or "geo" not in ids or "time" not in ids
    ):
        raise EurostatDataError(f"Invalid JSON-stat structure in {spec.dataset}.")
    if set(ids) - {"geo", "time"} != set(spec.filters):
        raise EurostatDataError(f"Dataset {spec.dataset} changed its dimension set.")

    positions: dict[str, dict[str, int]] = {}
    for dim, size in zip(ids, sizes, strict=True):
        node = dims.get(dim)
        if not isinstance(node, dict):
            raise EurostatDataError(f"Missing {dim} dimension in {spec.dataset}.")
        index = _positions(node)
        if set(index.values()) != set(range(size)):
            raise EurostatDataError(f"Inconsistent {dim} positions in {spec.dataset}.")
        if dim not in {"geo", "time"} and (size != 1 or spec.filters[dim] not in index):
            raise EurostatDataError(f"Unexpected {dim} code in {spec.dataset}.")
        positions[dim] = index

    absent = sorted(set(countries) - set(positions["geo"]))
    if absent:
        raise EurostatDataError(f"Countries not in {spec.dataset}: {', '.join(absent)}.")
    if not all(QUARTER_RE.fullmatch(q) for q in positions["time"]):
        raise EurostatDataError(f"Non-quarterly observation in {spec.dataset}.")
    statuses = payload.get("status", {})
    if not isinstance(statuses, (dict, list)):
        raise EurostatDataError(f"Invalid status mapping in {spec.dataset}.")

    strides: dict[str, int] = {}
    multiplier = 1
    for dim, size in reversed(list(zip(ids, sizes, strict=True))):
        strides[dim] = multiplier
        multiplier *= size

    result: dict[tuple[str, str], Observation] = {}
    for country in countries:
        for quarter in positions["time"]:
            idx = sum(
                (positions[dim][country if dim == "geo" else quarter]
                 if dim in {"geo", "time"} else 0) * strides[dim]
                for dim in ids
            )
            raw, flag = _cell(values, idx), _cell(statuses, idx)
            if raw is None:
                value = None
            elif type(raw) not in (int, float) or not math.isfinite(float(raw)):
                raise EurostatDataError(f"Invalid numeric observation in {spec.dataset}.")
            else:
                value = float(raw)
            if flag is not None and not isinstance(flag, str):
                raise EurostatDataError(f"Invalid quality flag in {spec.dataset}.")
            result[(country, quarter)] = Observation(value, flag or "")
    if not any(ob.value is not None for ob in result.values()):
        raise EurostatDataError(f"No observations returned for {spec.dataset}.")
    return result
