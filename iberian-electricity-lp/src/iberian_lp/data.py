"""The data step (METHODS.md, sections 3-4): read the downloaded files, aggregate the
offered steps and write the derived tables. No programme is solved here.

Writes, under ``out``:
- ``steps/<yyyymmdd>.csv.gz``: the offered steps of each zone-period, summed by side, order
  type and price. Steps of the same zone, side, type and price are interchangeable in both
  programmes of section 4, so the sum loses nothing they use.
- ``periods.csv``: one row per zone-period, with the published prices, the offered and
  matched volumes by side and type, and the matched flows.
- ``days.csv`` and ``manifest.json`` (files, sizes, hashes, versions and seeds).
"""

from __future__ import annotations

import csv
import gzip
import importlib.metadata
import io
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from iberian_lp.contract import (
    DAYS_PER_MONTH,
    DAYS_SEED,
    FIRST_DAY,
    FRANCE_BORDER,
    KINDS,
    LAST_DAY,
    NON_CONVEX,
    SIMPLE,
    SOURCE,
    SPEC,
)
from iberian_lp.days import draw_days
from iberian_lp.omie import read_capacity, read_curve, read_prices, sha256

# What a step of each status may be: (side, type). Anything else stops the build.
OFFERED = {("C", SIMPLE), ("V", SIMPLE)} | {("V", k) for k in NON_CONVEX}
TYPES = (SIMPLE, *NON_CONVEX)
PERIOD_FIELDS = (
    "date", "period", "hour", "quarter", "zone", "price_pt", "price_es", "steps_offered",
    "steps_aggregated", "offered_buy", *(f"offered_sell_{t}" for t in TYPES), "matched_buy",
    *(f"matched_sell_{t}" for t in TYPES), "imports_fr", "exports_fr", "imports_ib",
    "exports_ib", "fr_import_occupied", "fr_export_occupied")
STEP_FIELDS = ("period", "zone", "side", "type", "price", "mw")
PACKAGES = ("numpy", "scipy")


def _mw(tenths: int) -> str:
    return f"{tenths / 10:.1f}"


def _eur(cents: int) -> str:
    return f"{cents / 100:.2f}"


def _file(raw: Path, kind: str, day: date) -> Path:
    found = sorted(raw.glob(f"{kind}_{day:%Y%m%d}.*"))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one {kind} file for {day} in {raw}, found "
                                f"{[p.name for p in found]}.")
    return found[0]


def _flow(side: str, kind: str) -> str:
    """A matched flow by its side: on the sell side it is an import (fixed supply), on the
    buy side an export (fixed demand). The label names the other market: FR, or PT/ES for
    the border between Spain and Portugal, whose labels differ between dates."""
    border = "fr" if kind.endswith("FR") else "ib"
    if not (kind.startswith(("Imp ", "Exp ")) and kind[4:] in ("FR", "PT", "ES")):
        raise ValueError(f"Unknown flow label {kind!r}.")
    return f"{'imports' if side == 'V' else 'exports'}_{border}"


def read_day(raw: Path, day: date) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One day's zone-period rows and aggregated offered steps."""
    offered: dict[tuple[int, str, str, str, int], int] = defaultdict(int)
    matched: dict[tuple[int, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    raw_steps: Counter[tuple[int, str]] = Counter()
    hours: dict[int, tuple[int, int]] = {}
    for s in read_curve(_file(raw, "curva_pbc", day), day):
        hours[s.period] = (s.hour, s.quarter)
        key = (s.period, s.zone)
        if s.status == "O":
            if (s.side, s.kind) not in OFFERED:
                raise ValueError(f"{day}: an offered step of side {s.side}, type {s.kind}.")
            offered[(s.period, s.zone, s.side, s.kind, s.cents)] += s.tenths
            raw_steps[key] += 1
        elif s.status == "C":
            if s.kind in TYPES:
                if (s.side, s.kind) not in OFFERED:
                    raise ValueError(f"{day}: a matched step of side {s.side}, type {s.kind}.")
                matched[key]["buy" if s.side == "C" else f"sell_{s.kind}"] += s.tenths
            else:
                matched[key][_flow(s.side, s.kind)] += s.tenths
        else:
            raise ValueError(f"{day}: status {s.status!r}.")
    prices = read_prices(_file(raw, "marginalpdbc", day), day)
    france = read_capacity(_file(raw, "capacidad_inter_pbc", day), FRANCE_BORDER)
    if sorted(hours) != sorted(prices) or sorted(prices) != list(range(1, len(prices) + 1)):
        raise ValueError(f"{day}: the curve and price files have different periods.")
    totals: dict[tuple[int, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    counts: Counter[tuple[int, str]] = Counter()
    for (period, zone, side, kind, _), tenths in offered.items():
        totals[(period, zone)]["buy" if side == "C" else f"sell_{kind}"] += tenths
        counts[(period, zone)] += 1
    rows = []
    for period, zone in sorted(set(raw_steps) | set(matched)):
        o, m = totals[(period, zone)], matched[(period, zone)]
        fr = france.get(period, {})
        rows.append({
            "date": day.isoformat(), "period": period, "hour": hours[period][0],
            "quarter": hours[period][1], "zone": zone,
            "price_pt": _eur(prices[period][0]), "price_es": _eur(prices[period][1]),
            "steps_offered": raw_steps[(period, zone)],
            "steps_aggregated": counts[(period, zone)],
            "offered_buy": _mw(o["buy"]), **{f"offered_sell_{t}": _mw(o[f"sell_{t}"])
                                            for t in TYPES},
            "matched_buy": _mw(m["buy"]), **{f"matched_sell_{t}": _mw(m[f"sell_{t}"])
                                            for t in TYPES},
            **{f: _mw(m[f]) for f in ("imports_fr", "exports_fr", "imports_ib", "exports_ib")},
            "fr_import_occupied": _mw(fr.get("import_occupied", 0)),
            "fr_export_occupied": _mw(fr.get("export_occupied", 0))})
    steps = [{"period": period, "zone": zone, "side": side, "type": kind, "price": _eur(c),
              "mw": _mw(t)} for (period, zone, side, kind, c), t in sorted(offered.items())]
    return rows, steps


def _write_gzip_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                               mtime=0) as stream:
        stream.write(text.getvalue().encode("utf-8"))


def build(raw: Path, out: Path) -> dict[str, Any]:
    days = draw_days()
    files = [{"kind": kind, "day": day.isoformat(), "file": p.name, "bytes": p.stat().st_size,
              "sha256": sha256(p)} for day in days for kind in KINDS
             for p in [_file(raw, kind, day)]]
    (out / "steps").mkdir(parents=True, exist_ok=True)
    periods: list[dict[str, Any]] = []
    for day in days:
        rows, steps = read_day(raw, day)
        periods += rows
        _write_gzip_csv(out / "steps" / f"{day:%Y%m%d}.csv.gz", steps, STEP_FIELDS)
    with (out / "periods.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PERIOD_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(periods)
    with (out / "days.csv").open("w", encoding="utf-8", newline="") as stream:
        stream.write("date\n" + "".join(f"{d.isoformat()}\n" for d in days))
    manifest = {"source": SOURCE, "specification": SPEC,
                "period": [FIRST_DAY.isoformat(), LAST_DAY.isoformat()],
                "days_per_month": DAYS_PER_MONTH, "days_seed": DAYS_SEED,
                "versions": {p: importlib.metadata.version(p) for p in PACKAGES},
                "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False)
                                       + "\n", encoding="utf-8", newline="\n")
    return manifest
