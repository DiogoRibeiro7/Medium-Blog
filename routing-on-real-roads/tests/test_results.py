"""The committed solves: complete, valid, costed correctly, reproducible; the report."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from routing_roads.experiment import VARIANTS, instance, tasks
from routing_roads.report import load
from routing_roads.routing import cost, decode, projected, solve

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
SOLVES = ROOT / "results" / "solves.csv"
ESTIMATES = ROOT / "results" / "estimates.json"
REPORT = ROOT / "docs" / "RESULTS.md"

pytestmark = pytest.mark.skipif(not SOLVES.exists(), reason="No committed solves.")


def solves() -> list[dict[str, str]]:
    from routing_roads.estimates import load_solves

    return load_solves(SOLVES)


def test_every_registered_solve_is_there_once() -> None:
    keys = [tuple(r[k] for k in ("variant", "pair", "area", "day", "plan", "seed"))
            for r in solves()]
    assert len(keys) == len(set(keys)) == sum(1 for _ in tasks())
    assert set(keys) == {tuple(map(str, t)) for t in tasks()}


def test_plans_serve_each_order_once_and_are_costed_correctly() -> None:
    d = load(DERIVED)
    capacity = {v.key: v.capacity for v in VARIANTS}
    depot = {v.key: v.depot for v in VARIANTS}
    for r in solves():
        points = instance(d, r["pair"], r["area"], int(r["day"]), depot[r["variant"]])
        routes = decode(r["plan_routes"])
        assert sorted(i for route in routes for i in route) == points[1:]
        assert all(len(route) <= capacity[r["variant"]] for route in routes)
        assert int(r["cost_road"]) == cost(routes, d.road[r["pair"]], points[0])
        assert int(r["cost_straight"]) == cost(routes, d.straight[r["pair"]], points[0])


@pytest.mark.parametrize("which", [0, -1])
def test_a_solve_reproduces_on_this_machine(which: int) -> None:
    """The iteration budget and seed fix the routes; two main solves are re-run."""
    d = load(DERIVED)
    main = [r for r in solves() if r["variant"] == "main" and r["pair"] == "porto"
            and r["area"] == "city"]
    r = sorted(main, key=lambda x: int(x["clients"]))[which]
    v = next(x for x in VARIANTS if x.key == "main")
    points = instance(d, r["pair"], r["area"], int(r["day"]), v.depot)
    rows = [p for p in d.points if p["pair"] == r["pair"]]
    xy = projected([float(p["node_lon"]) for p in rows], [float(p["node_lat"]) for p in rows])
    matrix = d.straight[r["pair"]] if r["plan"] == "straight" else d.road[r["pair"]]
    assert solve(matrix, xy, points, v.capacity, v.iterations, int(r["seed"])) == \
        decode(r["plan_routes"])


def _same(a: object, b: object, path: str = "") -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for k in a:
            _same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert isinstance(b, float) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), path
    else:
        assert a == b, path


def test_estimates_and_report_regenerate() -> None:
    from routing_roads.estimates import _round, estimate, render

    e = _round(estimate(solves(), load(DERIVED)))
    _same(json.loads(ESTIMATES.read_text(encoding="utf-8")), json.loads(json.dumps(e)))
    text = REPORT.read_text(encoding="utf-8")
    command = re.search(r"```\n(routing-roads estimate [^\n]+)\n```", text)
    assert command is not None
    assert render(json.loads(ESTIMATES.read_text(encoding="utf-8")), command.group(1)) == text
