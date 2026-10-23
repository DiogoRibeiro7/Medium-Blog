"""Strict JSON-stat 2.0 decoding. A missing cell stays missing; it is never zero."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from eu_wheat_weather.contract import DataError


@dataclass(frozen=True)
class Cell:
    """One published or flagged cell: its codes, its value (None if absent) and flag."""

    codes: dict[str, str]
    value: float | None
    flag: str


def _index(dimension: object, name: str) -> dict[str, int]:
    category = dimension.get("category") if isinstance(dimension, dict) else None
    index = category.get("index") if isinstance(category, dict) else None
    if isinstance(index, dict) and all(isinstance(k, str) and type(v) is int
                                       for k, v in index.items()):
        return dict(index)
    if isinstance(index, list) and all(isinstance(k, str) for k in index):
        return {code: i for i, code in enumerate(index)}
    raise DataError(f"Dimension {name} has no valid category index.")


def _lookup(source: object, position: int) -> object:
    if isinstance(source, dict):
        return source.get(str(position))
    if isinstance(source, list) and position < len(source):
        return source[position]
    return None


def decode(payload: Mapping[str, Any], filters: Mapping[str, str | tuple[str, ...]],
           ) -> list[Cell]:
    """Every cell that has a value or a flag.

    ``filters`` gives, for each dimension other than ``geo`` and ``time``, the codes that
    were requested. The response must contain exactly those dimensions and only those
    codes, so a changed dataset fails loudly instead of being misread.
    """
    ids, sizes = payload.get("id"), payload.get("size")
    dimensions, values = payload.get("dimension"), payload.get("value")
    if (not isinstance(ids, list) or not all(isinstance(i, str) for i in ids)
            or not isinstance(sizes, list) or not all(type(n) is int and n >= 0 for n in sizes)
            or len(ids) != len(sizes) or len(set(ids)) != len(ids)
            or not isinstance(dimensions, dict) or not isinstance(values, (dict, list))):
        raise DataError("Not a JSON-stat 2.0 dataset.")
    if not {"geo", "time"} <= set(ids) or set(ids) - {"geo", "time"} != set(filters):
        raise DataError(f"Unexpected dimensions: {ids}.")
    positions = {name: _index(dimensions.get(name), name) for name in ids}
    for name, size in zip(ids, sizes, strict=True):
        if sorted(positions[name].values()) != list(range(size)):
            raise DataError(f"Dimension {name} has inconsistent positions.")
        if name in filters:
            wanted = filters[name]
            wanted = (wanted,) if isinstance(wanted, str) else wanted
            if not set(positions[name]) <= set(wanted):
                raise DataError(f"Dimension {name} has codes that were not requested: "
                                f"{sorted(set(positions[name]) - set(wanted))}.")
    statuses = payload.get("status", {})
    if not isinstance(statuses, (dict, list)):
        raise DataError("Invalid status mapping.")
    strides = [1] * len(ids)
    for k in range(len(ids) - 2, -1, -1):
        strides[k] = strides[k + 1] * sizes[k + 1]
    inverse = {name: {i: code for code, i in positions[name].items()} for name in ids}
    total = 1
    for size in sizes:
        total *= size
    occupied: set[int] = set()
    for source in (values, statuses):
        keys = (source.keys() if isinstance(source, dict)
                else (str(i) for i, v in enumerate(source) if v is not None))
        for key in keys:
            if not key.isdigit() or int(key) >= total:
                raise DataError(f"Cell index {key} is out of range.")
            occupied.add(int(key))
    cells = []
    for flat in sorted(occupied):
        codes = {name: inverse[name][(flat // stride) % size]
                 for name, stride, size in zip(ids, strides, sizes, strict=True)}
        raw = _lookup(values, flat)
        if raw is not None and (isinstance(raw, bool) or not isinstance(raw, (int, float))):
            raise DataError(f"Non-numeric value at {codes}.")
        flag = _lookup(statuses, flat)
        if flag is not None and not isinstance(flag, str):
            raise DataError(f"Non-text flag at {codes}.")
        cells.append(Cell(codes, None if raw is None else float(raw), flag or ""))
    return cells
