"""The registered runs (sections 5 and 7): every instance, plan and seed.

Each solve is appended to ``results/solves.partial.csv`` as it finishes, so an interrupted
run resumes where it stopped; at the end the rows are sorted into ``results/solves.csv``,
which is then the same whatever order the workers finished in.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from routing_roads.contract import (
    CAPACITY,
    ITERATIONS,
    PAIRS,
    SEEDS,
    SENSITIVITY_DAYS,
    SENSITIVITY_SEEDS,
)
from routing_roads.report import Derived, load
from routing_roads.routing import cost, encode, projected, solve

AREAS = ("city", "both")                       # the city alone; with the other municipality
PLANS = ("straight", "road")                    # P_S and P_R
FIELDS = ("variant", "pair", "area", "day", "plan", "seed", "clients", "routes",
          "cross_bank_routes", "cost_road", "cost_straight", "plan_routes")
KEY = FIELDS[:6]


@dataclass(frozen=True)
class Variant:
    key: str
    capacity: int
    iterations: int
    depot: str                                  # "depot" (section 4) or "depot_variant"
    days: int
    seeds: tuple[int, ...]


VARIANTS = (
    Variant("main", CAPACITY, ITERATIONS, "depot", 40, SEEDS),
    Variant("depot_all", CAPACITY, ITERATIONS, "depot_variant", SENSITIVITY_DAYS,
            SENSITIVITY_SEEDS),
    Variant("capacity_6", 6, ITERATIONS, "depot", SENSITIVITY_DAYS, SENSITIVITY_SEEDS),
    Variant("capacity_24", 24, ITERATIONS, "depot", SENSITIVITY_DAYS, SENSITIVITY_SEEDS),
    Variant("iterations_20000", CAPACITY, 20_000, "depot", SENSITIVITY_DAYS,
            SENSITIVITY_SEEDS),
)


def tasks() -> Iterator[tuple[str, str, str, int, str, int]]:
    for v in VARIANTS:
        for pair in PAIRS:
            for area in AREAS:
                for day in range(1, v.days + 1):
                    for plan in PLANS:
                        for seed in v.seeds:
                            yield v.key, pair.key, area, day, plan, seed


_D: Derived | None = None


def _init(directory: str) -> None:
    global _D
    _D = load(Path(directory))


def instance(d: Derived, pair_key: str, area: str, day: int, depot_role: str) -> list[int]:
    """The point indices of one instance: the depot first, then the day's orders in the
    area, by index."""
    pair = next(p for p in PAIRS if p.key == pair_key)
    rows = [r for r in d.points if r["pair"] == pair_key]
    depot = next(int(r["index"]) for r in rows if r["role"] == depot_role)
    town = {int(r["index"]): r["municipality"] for r in rows if r["role"] == "pharmacy"}
    orders = sorted(int(r["index"]) for r in d.days
                    if r["pair"] == pair_key and int(r["day"]) == day)
    keep = (pair.city,) if area == "city" else (pair.city, pair.other)
    return [depot] + [i for i in orders if town[i] in keep]


def run_one(task: tuple[str, str, str, int, str, int]) -> dict[str, Any]:
    assert _D is not None
    variant_key, pair_key, area, day, plan, seed = task
    v = next(x for x in VARIANTS if x.key == variant_key)
    points = instance(_D, pair_key, area, day, v.depot)
    rows = [r for r in _D.points if r["pair"] == pair_key]
    xy = projected([float(r["node_lon"]) for r in rows], [float(r["node_lat"]) for r in rows])
    road, straight = _D.road[pair_key], _D.straight[pair_key]
    routes = solve(straight if plan == "straight" else road, xy, points, v.capacity,
                   v.iterations, seed)
    town = {int(r["index"]): r["municipality"] for r in rows}
    return {"variant": variant_key, "pair": pair_key, "area": area, "day": day, "plan": plan,
            "seed": seed, "clients": len(points) - 1, "routes": len(routes),
            "cross_bank_routes": sum(1 for r in routes if len({town[i] for i in r}) > 1),
            "cost_road": cost(routes, road, points[0]),
            "cost_straight": cost(routes, straight, points[0]),
            "plan_routes": encode(routes)}


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    order = [v.key for v in VARIANTS]
    pairs = [p.key for p in PAIRS]
    return (order.index(row["variant"]), pairs.index(row["pair"]), AREAS.index(row["area"]),
            int(row["day"]), PLANS.index(row["plan"]), int(row["seed"]))


def run(derived: Path, results: Path, workers: int = 4) -> Path:
    results.mkdir(parents=True, exist_ok=True)
    partial, final = results / "solves.partial.csv", results / "solves.csv"
    done: set[tuple[str, ...]] = set()
    if partial.exists():
        with partial.open(encoding="utf-8") as stream:
            done = {tuple(r[k] for k in KEY) for r in csv.DictReader(stream)}
    todo = [t for t in tasks() if tuple(map(str, t)) not in done]
    start = time.time()
    with partial.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        if not done:
            writer.writeheader()
        with Pool(workers, initializer=_init, initargs=(str(derived),)) as pool:
            for n, row in enumerate(pool.imap_unordered(run_one, todo, chunksize=4), 1):
                writer.writerow(row)
                stream.flush()
                if n % 100 == 0 or n == len(todo):
                    print(f"{n}/{len(todo)} solves, {time.time() - start:.0f} s", flush=True)
    with partial.open(encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=_key)
    if len(rows) != sum(1 for _ in tasks()):
        raise ValueError(f"{len(rows)} rows, expected {sum(1 for _ in tasks())}.")
    with final.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    partial.unlink()
    return final


def timing(derived: Path, n: int = 4) -> list[float]:
    """Seconds per solve for the largest main instances (a check before the run)."""
    _init(str(derived))
    assert _D is not None
    out = []
    for pair in PAIRS:
        sizes = {day: len(instance(_D, pair.key, "both", day, "depot")) for day in range(1, 41)}
        day = max(sizes, key=lambda k: sizes[k])
        for plan in PLANS[: n // 2]:
            t = time.time()
            run_one(("main", pair.key, "both", day, plan, 1))
            out.append(time.time() - t)
    return [float(np.round(x, 2)) for x in out]
