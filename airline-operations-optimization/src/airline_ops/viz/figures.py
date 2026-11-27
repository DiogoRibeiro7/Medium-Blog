"""Figures of the article, drawn from the results files.

Every empirical figure reads the JSON written by :mod:`airline_ops.experiments.runner`.
The opening-case figure also reads the EASA rules in :mod:`airline_ops.regulation.ftl`.
Each figure is written as SVG and as a 200 dpi PNG. Colours follow one fixed assignment:
a planning strategy keeps its colour in every figure.
"""

from __future__ import annotations

import json
import math
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator

from airline_ops.config import load_config
from airline_ops.errors import DataError
from airline_ops.logging_setup import get_logger
from airline_ops.network.model import COUNT_LINE_UTC
from airline_ops.regulation.ftl import OperatorTimes, max_fdp_minutes, min_rest_minutes

LOG = get_logger("viz.figures")
FOLDER = Path(__file__).resolve().parents[3]

TEXT = "#191919"
MUTED = "#666666"
AXIS = "#bdbdbd"
GRID = "#ebebeb"
NEUTRAL = "#d9d9d9"
PALE = "#cce3fd"
BLUE, ORANGE, AQUA, YELLOW = "#257de0", "#f47230", "#00b894", "#f2a600"
SEQUENTIAL_RAMP = ["#e6f1fe", "#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]

PLAN_ORDER = ("greedy", "sequential", "integrated", "robust")
PLAN_COLOR = {"sequential": BLUE, "integrated": ORANGE, "robust": AQUA, "greedy": YELLOW}
PLAN_LABEL = {"greedy": "Greedy", "sequential": "Sequential", "integrated": "Integrated", "robust": "Robust"}
DPI = 200
DAY = 1440


# --------------------------------------------------------------------------- helpers


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "DejaVu Sans"],
            "font.size": 9.5,
            "text.color": TEXT,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": MUTED,
            "ytick.labelcolor": MUTED,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "path",
            "svg.hashsalt": "airline-operations-optimization",
        }
    )


def _figure(width: float, height: float, title: str, subtitle: str) -> tuple[Figure, SubFigure]:
    """A figure with a header band (title and subtitle) above a body for the axes."""
    subtitle = textwrap.fill(subtitle, width=int(width * 12.2))
    header = 0.36 + 0.19 * (subtitle.count("\n") + 1)
    fig = plt.figure(figsize=(width, height + header), layout="constrained")
    subs = fig.subfigures(2, 1, height_ratios=[header, height], hspace=0.0)
    if isinstance(subs, SubFigure):  # pragma: no cover - two rows always give an array
        raise TypeError("expected two subfigures")
    top, body = subs[0], subs[1]
    top.text(0.0, 1.0, title, ha="left", va="top", fontsize=12.5, fontweight="bold", color=TEXT)
    top.text(0.0, 0.0, subtitle, ha="left", va="bottom", fontsize=9.5, color=MUTED, linespacing=1.35)
    return fig, body


def _save(fig: Figure, out: Path, stem: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    svg, png = out / f"{stem}.svg", out / f"{stem}.png"
    fig.savefig(svg, metadata={"Date": None, "Creator": None}, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(png, dpi=DPI, metadata={"Software": None}, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    LOG.info("figure written", extra={"fields": {"figure": stem}})
    return [svg, png]


def _load(results: Path, name: str) -> Any:
    path = results / name
    if not path.is_file():
        raise DataError(f"missing results file {path}; run the experiment first")
    return json.loads(path.read_text(encoding="utf-8"))


def _horizon(results: Path) -> int:
    run = _load(results, "run.json")
    return load_config(FOLDER / run["config"]).simulation.horizon_days


def _grid_y(ax: Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _grid_x(ax: Axes) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _euro_k(x: float, _pos: float | None = None) -> str:
    return f"€{x / 1000:,.0f}k" if abs(x) >= 10_000 else f"€{x / 1000:,.1f}k"


def _hm(minutes: float) -> str:
    m = round(minutes)
    return f"{m // 60} h {m % 60:02d} min"


def _clip_label(ax: Axes, x: float, y: float, text: str, color: str, side: str = "right") -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(5 if side == "right" else -5, 0),
        textcoords="offset points",
        ha="left" if side == "right" else "right",
        va="center",
        fontsize=8.5,
        color=TEXT,
    )
    del color


# --------------------------------------------------------------------------- figure 1


def _outstation_case(results: Path, plan: str, station: str) -> dict[str, Any]:
    """Flights, aircraft lines and crew pairings of one plan that touch one outstation."""
    net = _load(results, "network.json")
    rec = _load(results, "plans.json")[plan]
    flights: list[dict[str, Any]] = net["flights"]
    touch = {k for k, f in enumerate(flights) if station in (f["origin"], f["destination"])}
    successor = {int(k): int(v) for k, v in rec["successor"].items()}
    lines: list[tuple[int, int]] = []
    for k in sorted(touch, key=lambda i: flights[i]["dep_utc"]):
        nxt = successor.get(k)
        if flights[k]["destination"] == station and nxt in touch and flights[nxt]["origin"] == station:
            lines.append((k, nxt))
    pairings = [p for p in rec["pairing_duties"] if {f for d in p["duties"] for f in d} & touch]
    if not lines or not pairings:
        raise DataError(f"plan {plan!r} has no aircraft line or pairing through {station}")
    return {"flights": flights, "lines": lines, "pairings": pairings, "hub_offset": int(net["hub_utc_offset_min"])}


def _pairing_instance(
    pairing: dict[str, Any], flights: list[dict[str, Any]], day: int, times: OperatorTimes
) -> list[dict[str, Any]]:
    """Duties of one pairing started on ``day``: report, end and flight times in UTC minutes."""
    first = flights[pairing["duties"][0][0]]
    report0 = first["dep_utc"] + day * DAY - times.report_before_min
    duties: list[dict[str, Any]] = []
    for duty, start in zip(pairing["duties"], pairing["starts"], strict=True):
        report = report0 + int(start)
        shift = round((report + times.report_before_min - flights[duty[0]]["dep_utc"]) / DAY) * DAY
        legs: list[tuple[str, int, int]] = []
        for f in duty:
            dep = flights[f]["dep_utc"] + shift
            if legs and dep < legs[-1][2]:
                shift += DAY
                dep += DAY
            legs.append((flights[f]["number"], dep, dep + int(flights[f]["block"])))
        duties.append({"report": report, "end": legs[-1][2] + times.post_flight_min, "legs": legs})
    return duties


def fig01_rotation(results: Path, out: Path, plan: str = "integrated", station: str = "OPO") -> list[Path]:
    """The synthetic airline's Porto flights over two days, with the aircraft lines and crews of one plan."""
    case = _outstation_case(results, plan, station)
    flights: list[dict[str, Any]] = case["flights"]
    hub_offset = int(case["hub_offset"])
    times = OperatorTimes()
    x0 = 12 * 60 - hub_offset  # 12:00 on day 1, Munich time
    x1 = x0 + 2 * DAY

    def hx(utc: float) -> float:
        return (utc + hub_offset) / 60.0

    def fname(k: int) -> str:
        return str(flights[k]["number"])

    # Aircraft lanes: the daily line of flying out of the hub and back.
    aircraft: list[tuple[str, list[tuple[str, int, int]]]] = []
    for out_k, back_k in case["lines"]:
        o, b = flights[out_k], flights[back_k]
        o_arr = o["dep_utc"] + o["block"]
        overnight = b["dep_utc"] < o_arr
        legs: list[tuple[str, int, int]] = []
        for d in range(-1, 3):
            dep_o = o["dep_utc"] + d * DAY
            dep_b = b["dep_utc"] + (d + 1 if overnight else d) * DAY
            legs += [(fname(out_k), dep_o, dep_o + o["block"]), (fname(back_k), dep_b, dep_b + b["block"])]
        kind = "night-stop" if overnight else "day turn"
        aircraft.append((f"Aircraft · {kind}\n{fname(out_k)} out, {fname(back_k)} back", legs))
    aircraft.sort(key=lambda a: "night-stop" in a[0])
    # Crew lanes: every instance of each pairing that is visible in the window.
    day_crews: list[list[dict[str, Any]]] = []
    night_crews: list[list[dict[str, Any]]] = []
    for pairing in case["pairings"]:
        for d in range(-3, 3):
            inst = _pairing_instance(pairing, flights, d, times)
            if inst[-1]["end"] <= x0 or inst[0]["report"] >= x1:
                continue
            (night_crews if pairing["nights"] else day_crews).append(inst)
    night_crews.sort(key=lambda inst: inst[0]["report"])
    crews: list[tuple[str, list[list[dict[str, Any]]]]] = []
    if day_crews:
        crews.append(("Day crews\n(a new crew each day)", day_crews))
    letters = "ABCDEFGH"
    crews += [(f"Night crew {letters[i]}", [inst]) for i, inst in enumerate(night_crews)]

    lanes: list[str] = [lab for lab, _ in aircraft] + [lab for lab, _ in crews]
    plan_label = PLAN_LABEL.get(plan, plan).lower()
    fig, body = _figure(
        10.5,
        4.9,
        f"Porto in the synthetic airline: two aircraft lines and the crews the {plan_label} plan gives them",
        "Fictional flights between real airports, over two days of the daily schedule; all times are Munich "
        "time (Porto is one hour behind). An aircraft lane is the daily line of flying, which may be flown by a "
        "different aircraft each day. Every rest meets ORO.FTL.235(b).",
    )
    ax = body.subplots()
    height = 0.52
    ys = {i: len(lanes) - 1 - i for i in range(len(lanes))}

    def bar(start: float, stop: float, y: float, color: str, z: int) -> None:
        lo, hi = max(start, x0), min(stop, x1)
        if hi > lo:
            ax.add_patch(Rectangle((hx(lo), y - height / 2), (hi - lo) / 60, height, color=color, lw=0, zorder=z))

    def flight(name: str, dep: int, arr: int, y: float) -> None:
        bar(dep, arr, y, BLUE, 3)
        if x0 <= dep and arr <= x1:
            ax.text(hx((dep + arr) / 2), y, name, ha="center", va="center", color="white", fontsize=8, zorder=4)

    for i, (_lab, legs) in enumerate(aircraft):
        y = ys[i]
        for k, (name, dep, arr) in enumerate(legs):
            flight(name, dep, arr, y)
            if k % 2 == 0:
                nxt = legs[k + 1][1]
                bar(arr, nxt, y, NEUTRAL, 2)
                if not (x0 <= arr and nxt <= x1):
                    continue
                if nxt - arr < 120:
                    ax.text(hx((arr + nxt) / 2), y + 0.38, f"{nxt - arr} min turn", ha="center", va="bottom",
                            fontsize=7.5, color=MUTED)  # fmt: skip
                else:
                    ax.text(hx((arr + nxt) / 2), y, _hm(nxt - arr), ha="center", va="center", fontsize=8,
                            color=TEXT, zorder=4)  # fmt: skip
    for j, (_lab, instances) in enumerate(crews):
        y = ys[len(aircraft) + j]
        for duties, k, duty in ((ds, k, d) for ds in instances for k, d in enumerate(ds)):
            bar(duty["report"], duty["end"], y, PALE, 2)
            for name, dep, arr in duty["legs"]:
                flight(name, dep, arr, y)
            if k + 1 < len(duties):
                later = duties[k + 1]
                rest = later["report"] - duty["end"]
                need = min_rest_minutes(duty["end"] - duty["report"], at_home_base=False)
                lo, hi = max(duty["end"], x0), min(later["report"], x1)
                ax.plot([hx(lo), hx(hi)], [y, y], color=MUTED, lw=0.9, zorder=1)
                label = f"rest {rest // 60} h {rest % 60:02d} min in Porto (min {need // 60} h)"
                if later["report"] > x1:
                    day_no = (later["report"] + hub_offset) // DAY + 1
                    label = f"rest in Porto, then {later['legs'][0][0]} on day {day_no}"
                    ax.text(hx(lo) + 0.4, y + 0.1, label, ha="left", va="bottom", fontsize=8, color=TEXT)
                elif duty["end"] < x0:
                    ax.text(hx(x0) + 0.4, y + 0.1, label, ha="left", va="bottom", fontsize=8, color=TEXT)
                else:
                    ax.text(hx((lo + hi) / 2), y + 0.1, label, ha="center", va="bottom", fontsize=8, color=TEXT)
    # Why the crew that brings the night-stop aircraft in cannot take it out the next morning.
    night_line = next((legs for lab, legs in aircraft if "night-stop" in lab), None)
    first_night = next(
        (i for i, (lab, insts) in enumerate(crews) if lab.startswith("Night") and insts[0][0]["report"] >= x0), None
    )
    if night_line is not None and first_night is not None:
        duty = crews[first_night][1][0][0]
        arr_in = duty["legs"][-1][2]
        dep_back = next(dep for k, (_n, dep, _a) in enumerate(night_line) if k % 2 == 1 and dep > arr_in)
        back_name = next(n for k, (n, dep, _a) in enumerate(night_line) if k % 2 == 1 and dep == dep_back)
        report = dep_back - times.report_before_min
        gap = report - duty["end"]
        y_lane = ys[len(aircraft) + first_night]
        ax.plot([hx(report)] * 2, [y_lane - 0.34, y_lane + 0.02], color=ORANGE, lw=1.4, zorder=3)
        ax.text(hx(report) + 0.25, y_lane - 0.5,
                f"Reporting for {back_name} next morning: {gap // 60} h {gap % 60:02d} min of rest, not legal",
                ha="left", va="center", fontsize=8, color=MUTED)  # fmt: skip
    # FDP of the first visible day duty against its Table 2 limit.
    if day_crews:
        duty = next(d for inst in day_crews for d in inst if d["report"] >= x0)
        sectors = len(duty["legs"])
        limit = max_fdp_minutes((duty["report"] + hub_offset) % DAY, sectors)
        fdp = duty["legs"][-1][2] - duty["report"]
        ax.text(hx(duty["end"]) + 0.4, ys[len(aircraft)], f"FDP {_hm(fdp)} (limit {_hm(limit)})", ha="left",
                va="center", fontsize=8, color=MUTED)  # fmt: skip
    top_y = len(lanes) - 0.35
    ax.text(hx(x0) + 0.2, top_y, "Day 1", ha="left", va="bottom", fontsize=9, color=TEXT)
    for d in (1, 2):
        midnight = d * DAY - hub_offset
        ax.axvline(hx(midnight), color=AXIS, lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.text(hx(midnight) + 0.2, top_y, f"Day {d + 1}", ha="left", va="bottom", fontsize=9, color=TEXT)
    ax.axhline(len(crews) - 0.5, color=AXIS, lw=0.6)
    ax.set_yticks([ys[i] for i in range(len(lanes))], lanes, fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ticks = np.arange(12, 12 + 49, 6)
    ax.set_xticks(ticks, [f"{int(t) % 24:02d}:00" for t in ticks])
    ax.set_xlim(hx(x0), hx(x1))
    ax.set_ylim(-0.9, len(lanes) - 0.1)
    ax.set_xlabel("Munich time (CET)")
    ax.spines["left"].set_visible(False)
    handles = [
        Rectangle((0, 0), 1, 1, color=BLUE),
        Rectangle((0, 0), 1, 1, color=PALE),
        Rectangle((0, 0), 1, 1, color=NEUTRAL),
        Line2D([0], [0], color=MUTED, lw=0.9),
        Line2D([0], [0], color=ORANGE, lw=1.4),
    ]
    ax.legend(
        handles,
        [
            "Flight (block time)",
            "Crew duty: report to post-flight",
            "Aircraft on the ground",
            "Crew rest",
            "Report time that rest rules forbid",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=5,
        fontsize=8.5,
    )
    return _save(fig, out, "fig01_muc_opo_rotation")


# --------------------------------------------------------------------------- figure 2


def fig02_time_space(results: Path, out: Path, stations: Sequence[str] = ("MUC", "OPO", "LIS", "MAD")) -> list[Path]:
    """Time-expanded aircraft network for the hub and three spokes over one day."""
    net = _load(results, "network.json")
    config = load_config(FOLDER / _load(results, "run.json")["config"])
    hub = config.network.hub
    present = [s for s in stations if s in net["airports"]]
    flights = [(k, f) for k, f in enumerate(net["flights"]) if f["origin"] in present and f["destination"] in present]
    turn = {s: (config.network.turn_min_hub if s == hub else config.network.turn_min_spoke) for s in present}
    ypos = {s: len(present) - 1 - i for i, s in enumerate(present)}
    fig, body = _figure(
        10.5,
        4.6,
        "Time-expanded network for daily aircraft routing (Munich and three Iberian spokes)",
        "Departure nodes sit at the departure time, arrival nodes at "
        "arrival plus the minimum turn. Ground arcs let an aircraft "
        "wait, overnight arcs carry it into the next day, and the "
        "count line sizes the fleet. Synthetic schedule, times in UTC.",
    )
    ax = body.subplots()
    nodes: dict[str, list[float]] = {s: [] for s in present}
    arcs: list[tuple[float, float, float, float]] = []
    for _k, f in flights:
        dep = f["dep_utc"] / 60
        ready = (f["dep_utc"] + f["block"] + turn[f["destination"]]) / 60
        nodes[f["origin"]].append(dep)
        nodes[f["destination"]].append(ready)
        arcs.append((dep, ypos[f["origin"]], ready, ypos[f["destination"]]))
    count_t = COUNT_LINE_UTC / 60
    lo = min(min(min(v) for v in nodes.values() if v) - 1.0, count_t - 0.6)
    hi = max(max(v) for v in nodes.values() if v) + 1.5
    for s in present:
        y = ypos[s]
        pts = sorted(nodes[s])
        ax.plot([lo, pts[0]], [y, y], color=MUTED, lw=1.0, ls=(0, (2, 2)), zorder=1)
        ax.plot([pts[-1], hi], [y, y], color=MUTED, lw=1.0, ls=(0, (2, 2)), zorder=1)
        ax.plot([pts[0], pts[-1]], [y, y], color=MUTED, lw=1.6, zorder=1)
        ax.scatter(pts, [y] * len(pts), s=22, facecolor="white", edgecolor=TEXT, linewidth=0.9, zorder=4)
        if s != hub:
            # Aircraft needed overnight at a spoke: the largest running deficit of departures over ready arrivals.
            deps = [(a[0], -1) for a in arcs if a[1] == y]
            arrs = [(a[2], 1) for a in arcs if a[3] == y]
            events = sorted(deps + arrs, key=lambda e: (e[0], -e[1]))
            run, worst = 0, 0
            for _t, d in events:
                run += d
                worst = min(worst, run)
            ax.text(count_t + 0.15, y + 0.1, f"{-worst} aircraft overnight", ha="left", va="bottom", fontsize=8,
                    color=MUTED)  # fmt: skip
    for x_a, y_a, x_b, y_b in arcs:
        ax.add_patch(
            FancyArrowPatch(
                (x_a, y_a), (x_b, y_b), arrowstyle="-|>", mutation_scale=8, color=BLUE, lw=1.1, shrinkA=3,
                shrinkB=3, zorder=3, alpha=0.9,
            )
        )  # fmt: skip
    ax.axvline(count_t, color=ORANGE, lw=1.4, zorder=2)
    ax.text(count_t + 0.12, len(present) - 0.55, "count line (03:00 Munich time)", color=TEXT, fontsize=8.5,
            ha="left", va="bottom")  # fmt: skip
    ax.set_yticks([ypos[s] for s in present], [f"{s}" for s in present])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ticks = np.arange(math.ceil(lo), math.floor(hi) + 1, 2)
    ax.set_xticks(ticks, [f"{int(t) % 24:02d}:00" for t in ticks])
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, len(present) - 0.3)
    ax.set_xlabel("Time (UTC)")
    handles = [
        Line2D([0], [0], color=BLUE, lw=1.4, marker=">", markersize=5),
        Line2D([0], [0], color=MUTED, lw=1.6),
        Line2D([0], [0], color=MUTED, lw=1.0, ls=(0, (2, 2))),
        Line2D([0], [0], color=ORANGE, lw=1.4),
        Line2D([0], [0], color="white", marker="o", markeredgecolor=TEXT, markersize=6, lw=0),
    ]
    ax.legend(
        handles,
        ["Flight arc", "Ground arc", "Overnight arc (wraps to the next day)", "Count line", "Event node"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=5,
        fontsize=8.5,
    )
    return _save(fig, out, "fig02_time_expanded_network")


# --------------------------------------------------------------------------- figure 3


def _single_note(ex: dict[str, Any], route: Callable[[dict[str, Any]], str]) -> str:
    chosen = {tuple(p[0]) for p in ex["chosen"].get("integrated", []) if len(p) == 1}
    singles = [d for d in ex["single_duties_touching"] if tuple(d["flight_ids"]) in chosen]
    if not singles:
        return ""
    names = ", ".join(f"{route(d)} (FDP {_hm(d['fdp'])})" for d in singles)
    what = "an out-and-back duty" if len(singles) == 1 else f"{len(singles)} out-and-back duties"
    return f" The plan also flies {what} through {ex['station']}, not drawn: {names}."


def fig03_pairing_network(results: Path, out: Path, max_duties: int = 8) -> list[Path]:
    """Duties into and out of one outstation and the legal ways to chain them across nights."""
    ex = _load(results, "examples.json")
    net = _load(results, "network.json")
    config = load_config(FOLDER / _load(results, "run.json")["config"])
    station = ex["station"]
    flights = net["flights"]

    def route(d: dict[str, Any]) -> str:
        stops = [flights[d["flight_ids"][0]]["origin"]] + [flights[f]["destination"] for f in d["flight_ids"]]
        return "–".join(stops)

    chosen = ex["chosen"].get("integrated", [])
    chosen_pairs = {(tuple(p[0]), tuple(p[1])) for p in chosen if len(p) == 2}
    used = {a for a, _b in chosen_pairs} | {b for _a, b in chosen_pairs}
    outs = sorted(ex["out_duties"], key=lambda d: d["report_utc"])
    backs = sorted(ex["back_duties"], key=lambda d: d["report_utc"])

    def pick(duties: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(duties) <= max_duties:
            return duties
        keep = [d for d in duties if tuple(d["flight_ids"]) in used]
        rest = sorted((d for d in duties if tuple(d["flight_ids"]) not in used),
                      key=lambda d: (len(d["flight_ids"]), d["report_utc"]))  # fmt: skip
        return sorted(keep + rest[: max(0, max_duties - len(keep))], key=lambda d: d["report_utc"])

    outs_s, backs_s = pick(outs), pick(backs)
    by_idx = {d["idx"]: d for d in outs + backs}
    shown = {d["idx"] for d in outs_s + backs_s}
    links: list[tuple[dict[str, Any], dict[str, Any], int, int]] = []
    for p in ex["pairs"]:
        a, b = by_idx.get(p["first"]), by_idx.get(p["second"])
        if a is None or b is None or a["idx"] not in shown or b["idx"] not in shown:
            continue
        day = round((a["report_utc"] + a["length"] + p["rest"] - b["report_utc"]) / DAY)
        links.append((a, b, day, p["rest"]))
    n_days = max([day for *_x, day, _r in links] + [1]) + 1
    # Rows: a header, the duties into the station, a header, the duties out of it.
    labels: list[tuple[str, bool]] = [(f"Duties that end in {station}", True)]
    labels += [(f"{route(d)}  ·  FDP {_hm(d['fdp'])}", False) for d in outs_s]
    labels += [(f"Duties that start in {station}", True)]
    labels += [(f"{route(d)}  ·  FDP {_hm(d['fdp'])}", False) for d in backs_s]
    n_rows = len(labels)
    y_out = {d["idx"]: n_rows - 2 - i for i, d in enumerate(outs_s)}
    y_back = {d["idx"]: n_rows - 3 - len(outs_s) - i for i, d in enumerate(backs_s)}
    fig, body = _figure(
        10.5,
        0.34 * n_rows + 0.9,
        f"Crew pairing network through {station}: legal duties and the rests that can join them",
        f"Bars are legal duties (FDP within Table 2, at most {config.crew.max_sectors} "
        f"sectors). A line joins a duty into {station} to a later duty out "
        "of it when the rest in between meets ORO.FTL.235(b). Dark: the "
        "pairings the integrated plan chose. Synthetic network, UTC. "
        f"Shown: {len(outs_s)} of {len(outs)} duties into {station}, "
        f"{len(backs_s)} of {len(backs)} out of it. Whole network: "
        f"{ex['n_duties_network']:,} legal duties and {ex['n_pairings_network']:,} pairings of one or two duties."
        + _single_note(ex, route),
    )
    ax = body.subplots()
    height = 0.56

    def bar(d: dict[str, Any], y: float, day: int, strong: bool) -> tuple[float, float]:
        start = (d["report_utc"] + day * DAY) / 60
        end = start + d["length"] / 60
        ax.add_patch(Rectangle((start, y - height / 2), end - start, height, color=BLUE if strong else PALE, lw=0,
                               zorder=3))  # fmt: skip
        return start, end

    pos_out = {}
    for d in outs_s:
        _s, e = bar(d, y_out[d["idx"]], 0, tuple(d["flight_ids"]) in used)
        pos_out[d["idx"]] = (e, y_out[d["idx"]])
    copies = {(b["idx"], day) for _a, b, day, _r in links} | {(d["idx"], 1) for d in backs_s}
    pos_back = {}
    for idx, day in sorted(copies):
        d = by_idx[idx]
        strong = any(
            b["idx"] == idx and dd == day and (tuple(a["flight_ids"]), tuple(b["flight_ids"])) in chosen_pairs
            for a, b, dd, _r in links
        )
        s, _e = bar(d, y_back[idx], day, strong)
        pos_back[(idx, day)] = (s, y_back[idx])
    for a, b, day, rest in links:
        xa, ya = pos_out[a["idx"]]
        xb, yb = pos_back[(b["idx"], day)]
        strong = (tuple(a["flight_ids"]), tuple(b["flight_ids"])) in chosen_pairs
        ax.plot([xa, xb], [ya, yb], color=TEXT if strong else AXIS, lw=2.0 if strong else 0.7,
                zorder=2 if strong else 1)  # fmt: skip
        if strong:
            ax.text(xb - 0.3, yb + 0.33, f"rest {rest // 60} h {rest % 60:02d}", fontsize=7.8, color=TEXT,
                    ha="right", va="bottom", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
                    zorder=5)  # fmt: skip
    for k in range(1, n_days):
        ax.axvline(24 * k, color=AXIS, lw=0.8, ls=(0, (3, 3)), zorder=0)
    for k in range(n_days):
        ax.text(24 * k + 0.4, n_rows - 0.35, f"Day {k + 1}", fontsize=8.5, color=MUTED, va="bottom")
    ax.axhline(n_rows - 1.5 - len(outs_s), color=AXIS, lw=0.6)
    ax.set_yticks(range(n_rows - 1, -1, -1), [t for t, _h in labels], fontsize=8)
    for tick, (_t, header) in zip(ax.get_yticklabels(), labels, strict=True):
        if header:
            tick.set_fontweight("bold")
            tick.set_color(TEXT)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ticks = np.arange(0, 24 * n_days + 1, 6)
    ax.set_xticks(ticks, [f"{int(t) % 24:02d}:00" for t in ticks], fontsize=8)
    ax.set_xlim(0, 24 * n_days)
    ax.set_ylim(-0.7, n_rows - 0.1)
    ax.set_xlabel("Time (UTC)")
    return _save(fig, out, "fig03_pairing_network")


# --------------------------------------------------------------------------- figure 4


def fig04_column_generation(results: Path, out: Path, plan: str = "integrated") -> list[Path]:
    """Restricted master objective, Lagrangian bound and column pool by iteration."""
    rec = _load(results, "plans.json")[plan]
    hist = rec["history"]
    if not hist:
        raise DataError(f"plan {plan} has no column-generation history")
    it = np.array([h["iteration"] for h in hist])
    lp = np.array([h["lp_objective"] for h in hist])
    lag = np.maximum.accumulate(np.array([h["lagrangian_bound"] for h in hist]))
    cols = np.array([h["columns"] for h in hist])
    final_mip = float(rec["objective"])
    fig, body = _figure(
        10.5,
        3.4,
        f"Column generation on the crew pairing master problem ({PLAN_LABEL[plan].lower()} plan)",
        "Left: restricted master LP objective and the best Lagrangian "
        "lower bound from the pricing step. Middle: their relative gap. "
        f"Right: columns in the master. Final integer solution €{final_mip:,.0f} "
        "per day (dashed). Early iterations sit above the axis: "
        "artificial columns are still in the basis.",
    )
    ax_a, ax_b, ax_c = body.subplots(1, 3, sharex=True)
    ref = lp[-1]
    ax_a.plot(it, lp, color=BLUE, marker="o", markersize=3.5, label="Master LP objective")
    ax_a.plot(it, lag, color=ORANGE, marker="o", markersize=3.5, label="Lagrangian bound")
    ax_a.axhline(final_mip, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    span = max(abs(ref) * 0.06, abs(final_mip - ref) * 3, 1.0)
    ax_a.set_ylim(ref - span, ref + span)
    ax_a.yaxis.set_major_formatter(_euro_k)
    ax_a.set_title("Objective (€ per day)")
    _grid_y(ax_a)
    ax_a.legend(loc="lower right", fontsize=8)
    gap = np.where(lag > 0, (lp - lag) / np.abs(lp), np.nan)
    gap = np.where(gap > 1e-9, gap, np.nan)
    ax_b.plot(it, gap, color=TEXT, marker="o", markersize=3.5)
    if np.isfinite(gap).any():
        ax_b.set_yscale("log")
    closed = [int(i) for i, g, b in zip(it, gap, lag, strict=True) if b > 0 and not np.isfinite(g)]
    if closed:
        ax_b.annotate("gap closed", (closed[0], np.nanmin(gap) if np.isfinite(gap).any() else 1.0), fontsize=8,
                      color=MUTED, xytext=(0, 6), textcoords="offset points", ha="center")  # fmt: skip
    ax_b.set_title("Relative gap, master LP to bound")
    _grid_y(ax_b)
    ax_c.plot(it, cols, color=BLUE, marker="o", markersize=3.5)
    ax_c.set_title("Columns in the master")
    ax_c.yaxis.set_major_formatter(lambda x, _p: f"{x:,.0f}")
    _grid_y(ax_c)
    for a in (ax_a, ax_b, ax_c):
        a.set_xlabel("Iteration")
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    return _save(fig, out, "fig04_column_generation")


# --------------------------------------------------------------------------- figure 5


BIND_COLOR = {1: BLUE, 2: ORANGE, 3: AQUA}
_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _count_word(n: int) -> str:
    return _WORDS.get(n, str(n))


BIND_LABEL = {1: "Passed on by the aircraft", 2: "Passed on by the crew", 3: "Hub arrival slot"}


def fig05_propagation(results: Path, out: Path) -> list[Path]:
    """Flights changed by the delayed-inbound scenario in one replication, for two plans."""
    prop = _load(results, "propagation.json")
    names = [n for n in ("sequential", "robust") if n in prop["plans"]]
    fig, body = _figure(
        10.5,
        5.2,
        f"How {_count_word(len(prop['delayed_flights']))} late "
        "arrivals spread through the day, sequential and robust plans",
        "Delayed-inbound scenario, one replication with the same random "
        "draws for both plans. Each dot is a flight whose departure "
        "moved by more than a minute against the same day without the disruption; lines show what passed the delay on.",
    )
    axes = body.subplots(len(names), 1, sharex=True, sharey=True)
    axes_list: list[Axes] = list(np.atleast_1d(axes))
    initial = set(prop["delayed_flights"])
    all_x = [n["sched_dep"] / 60 for p in prop["plans"].values() for n in p["nodes"]]
    for ax, name in zip(axes_list, names, strict=True):
        nodes = prop["plans"][name]["nodes"]
        by_inst = {n["instance"]: n for n in nodes}
        for n in nodes:
            pred = by_inst.get(n["bind_pred"])
            if pred is None or n["bind_type"] not in BIND_COLOR:
                continue
            y0 = pred["extra_delay"] if pred["extra_delay"] is not None else -20.0
            y1 = n["extra_delay"] if n["extra_delay"] is not None else -20.0
            ax.plot([pred["sched_dep"] / 60, n["sched_dep"] / 60], [y0, y1], color=BIND_COLOR[n["bind_type"]],
                    lw=1.2, zorder=2)  # fmt: skip
        for n in nodes:
            x = n["sched_dep"] / 60
            if n["cancelled"]:
                ax.scatter([x], [-20.0], marker="X", s=40, color=TEXT, zorder=4, linewidths=0)
                continue
            y = n["extra_delay"]
            if n["number"] in initial and n["bind_type"] == 0:
                ax.scatter([x], [y], marker="D", s=40, color=TEXT, edgecolor="white", linewidth=1.0, zorder=5)
            else:
                ax.scatter([x], [y], s=30, color=BIND_COLOR.get(n["bind_type"], MUTED), edgecolor="white",
                           linewidth=1.0, zorder=4)  # fmt: skip
        total = prop["plans"][name]["extra_delay_total"]
        n_canc = sum(1 for n in nodes if n["cancelled"])
        canc = f", {n_canc} cancelled" if n_canc else ""
        ax.set_title(f"{PLAN_LABEL[name]} plan: {len(nodes)} flights changed{canc}, {total:,.0f} extra delay minutes")
        ax.axhline(0, color=AXIS, lw=0.8)
        _grid_y(ax)
        ax.set_ylabel("Extra departure delay (min)")
        ax.axhline(-20, color=GRID, lw=0.6)
    lo, hi = min(all_x) - 0.5, max(all_x) + 0.5
    ticks = np.arange(math.floor(lo), math.ceil(hi) + 1, 2)
    axes_list[-1].set_xticks(ticks, [f"D{int(t) // 24 + 1} {int(t) % 24:02d}:00" for t in ticks], fontsize=8)
    axes_list[-1].set_xlim(lo, hi)
    axes_list[-1].set_xlabel("Scheduled departure (UTC; D1 is the first simulated day)")
    yt = axes_list[0].get_yticks()
    axes_list[0].set_yticks([-20, *[t for t in yt if t >= 0]], ["cancelled", *[f"{t:.0f}" for t in yt if t >= 0]])
    handles = [
        Line2D([0], [0], marker="D", color="white", markerfacecolor=TEXT, markersize=7, lw=0),
        *[Line2D([0], [0], color=c, lw=1.4, marker="o", markersize=5) for c in BIND_COLOR.values()],
        Line2D([0], [0], marker="o", color="white", markerfacecolor=MUTED, markersize=6, lw=0),
        Line2D([0], [0], marker="X", color="white", markerfacecolor=TEXT, markersize=7, lw=0),
    ]
    axes_list[0].legend(
        handles,
        ["Late inbound (the disruption)", *BIND_LABEL.values(), "Changed, no single binding resource", "Cancelled"],
        loc="upper right",
        fontsize=8,
        ncol=1,
    )
    return _save(fig, out, "fig07_disruption_propagation")


# --------------------------------------------------------------------------- figure 6


def _per_day(summary: dict[str, float], horizon: int) -> tuple[float, float, float]:
    return summary["mean"] / horizon, summary["lo"] / horizon, summary["hi"] / horizon


def fig06_buffer_curves(results: Path, out: Path) -> list[Path]:
    """Planned, disruption and total cost against the hub turn buffer."""
    sweep = _load(results, "sweep.json")
    horizon = _horizon(results)
    rows = sorted((r for r in sweep["rows"] if r.get("feasible") and r["weight"] == 0.0), key=lambda r: r["buffer"])
    reps = int(rows[0]["S0"]["total_cost"]["n"])
    b = np.array([r["buffer"] for r in rows])
    fig, body = _figure(
        10.5,
        3.6,
        "Planned cost, disruption cost and total cost against the hub turn buffer",
        "Robust planner without the delay penalty. Means per day "
        f"over {reps} seeded replications, 95% intervals shaded; "
        "baseline delays (S0) and a stressed day (S5). The ring marks the lowest total cost in each scenario.",
    )
    ax_a, ax_b, ax_c = body.subplots(1, 3, sharex=True)
    ax_a.plot(b, [r["planned_cost_day"] for r in rows], color=TEXT, marker="o", markersize=4)
    ax_a.set_title("Planned cost")
    for sc, color, lab in (("S0", BLUE, "Baseline (S0)"), ("S5", ORANGE, "Stressed day (S5)")):
        op = np.array([_per_day(r[sc]["operational_cost"], horizon) for r in rows])
        tot = np.array([_per_day(r[sc]["total_cost"], horizon) for r in rows])
        ax_b.fill_between(b, op[:, 1], op[:, 2], color=color, alpha=0.15, lw=0)
        ax_b.plot(b, op[:, 0], color=color, marker="o", markersize=4, label=lab)
        ax_c.fill_between(b, tot[:, 1], tot[:, 2], color=color, alpha=0.15, lw=0)
        ax_c.plot(b, tot[:, 0], color=color, marker="o", markersize=4, label=lab)
        k = int(np.argmin(tot[:, 0]))
        ax_c.scatter([b[k]], [tot[k, 0]], s=130, facecolor="none", edgecolor=TEXT, linewidth=1.3, zorder=5)
        ax_b.annotate(lab, (b[-1], op[-1, 0]), xytext=(4, 0), textcoords="offset points", fontsize=8,
                      va="center", color=TEXT)  # fmt: skip
    ax_b.set_title("Expected disruption cost")
    ax_c.set_title("Expected total cost")
    for a in (ax_a, ax_b, ax_c):
        a.yaxis.set_major_formatter(_euro_k)
        a.set_xlabel("Hub turn buffer (min)")
        a.set_xticks(b)
        _grid_y(a)
    ax_a.set_ylabel("€ per day")
    handles = [Line2D([0], [0], color=c, marker="o", lw=1.6, markersize=5) for c in (TEXT, BLUE, ORANGE)]
    body.legend(handles, ["Planned (same in every scenario)", "Baseline (S0)", "Stressed day (S5)"],
                loc="outside lower center", ncol=3, fontsize=8.5)  # fmt: skip
    ax_b.set_xlim(b.min() - 2, b.max() + 12)
    return _save(fig, out, "fig08_buffer_cost_curves")


# --------------------------------------------------------------------------- figure 7


METRICS = (
    ("total_cost", "Total cost (€ per day)", 1.0),
    ("operational_cost", "Disruption cost (€ per day)", 1.0),
    ("pax_delay_min", "Passenger delay (min per day)", 1.0),
    ("missed_connection_pax", "Missed connections (pax per day)", 1.0),
)


def _nice(v: float) -> float:
    """A round number at or above v, for symmetric axis limits."""
    if v <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(v))
    for step in (1, 2, 2.5, 5, 10):
        if step * mag >= v:
            return float(step * mag)
    return float(10 * mag)


def _short(x: float) -> str:
    a = abs(x)
    if a >= 1000:
        return f"{x / 1000:+,.0f}k" if a >= 10_000 else f"{x / 1000:+,.1f}k"
    return f"{x:+,.0f}" if a >= 10 else f"{x:+.1f}"


def fig07_strategy_comparison(results: Path, out: Path) -> list[Path]:
    """Paired differences against the sequential plan, one panel per scenario and metric, each on its own scale."""
    ev = _load(results, "evaluation.json")
    horizon = _horizon(results)
    scen = list(ev["scenarios"])
    others = [p for p in PLAN_ORDER if p in ev["paired_vs_sequential"]]
    reps = ev["replications"]
    fig, body = _figure(
        11.0,
        0.62 * len(scen) + 0.9,
        "Each strategy against the sequential plan, by disruption scenario",
        f"Mean paired difference per day with 95% interval, {reps} replications with common random numbers. Left of "
        "zero is better than sequential. Each panel has its own scale, set by the integrated and robust plans; greedy "
        "values beyond it are printed at the edge.",
    )
    axes = body.subplots(len(scen), len(METRICS), squeeze=False)
    ypos = {p: k for k, p in enumerate(reversed(others))}
    for i, sc in enumerate(scen):
        for j, (key, title, _scale) in enumerate(METRICS):
            ax = axes[i][j]
            vals = {p: [ev["paired_vs_sequential"][p][sc][key][q] / horizon for q in ("mean", "lo", "hi")]
                    for p in others}  # fmt: skip
            core = [abs(v) for p in others if p != "greedy" for v in vals[p]]
            lim = _nice(max(core, default=1.0) * 1.1)
            for p in others:
                m, lo, hi = vals[p]
                y = ypos[p]
                ax.plot([max(lo, -lim), min(hi, lim)], [y, y], color=PLAN_COLOR[p], lw=1.6, solid_capstyle="butt")
                if -lim <= m <= lim:
                    ax.scatter([m], [y], s=22, color=PLAN_COLOR[p], edgecolor="white", linewidth=0.8, zorder=4)
                else:
                    edge = lim if m > 0 else -lim
                    ax.scatter([edge], [y], marker=">" if m > 0 else "<", s=26, color=PLAN_COLOR[p], zorder=4)
                    ax.annotate(_short(m), (edge, y), xytext=(-5 if m > 0 else 5, 0), textcoords="offset points",
                                ha="right" if m > 0 else "left", va="center", fontsize=7, color=TEXT)  # fmt: skip
            ax.axvline(0, color=TEXT, lw=0.8)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-0.7, len(others) - 0.3)
            ax.set_yticks([])
            ax.set_xticks([-lim, 0, lim], [_short(-lim), "0", _short(lim)], fontsize=7)
            ax.tick_params(axis="x", length=2, pad=1)
            ax.spines["left"].set_visible(False)
            _grid_x(ax)
            if i == 0:
                ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{sc}  {ev['scenarios'][sc]}", rotation=0, ha="right", va="center", fontsize=8.3,
                              color=TEXT, labelpad=6)  # fmt: skip
    handles = [Line2D([0], [0], color=PLAN_COLOR[p], marker="o", lw=1.6, markersize=5) for p in others]
    body.legend(handles, [f"{PLAN_LABEL[p]} − sequential" for p in others], loc="outside lower center",
                ncol=len(others), fontsize=8.5)  # fmt: skip
    return _save(fig, out, "fig05_strategy_comparison")


# --------------------------------------------------------------------------- figure 8


def _survival(h: dict[str, Any], xs: np.ndarray) -> np.ndarray:
    edges = np.array(h["edges"][:-1], dtype=float)
    counts = np.array(h["counts"], dtype=float)
    total = counts.sum() + h["over_300"]
    tail = np.array([counts[edges >= x].sum() for x in xs]) + h["over_300"]
    return np.asarray(tail / total)


def fig08_delay_tails(results: Path, out: Path) -> list[Path]:
    """Share of flights arriving more than x minutes late, by plan, in two scenarios."""
    ev = _load(results, "evaluation.json")
    hist = ev["histograms"]
    fig, body = _figure(
        10.5,
        3.9,
        "Arrival delay tails by planning strategy",
        "Share of operated flights arriving more than x minutes late, "
        f"all {ev['replications']} replications pooled, log scale. "
        "Cancelled flights are excluded here and counted in the scenario metrics.",
    )
    axes = body.subplots(1, 2, sharey=True)
    xs = np.arange(0, 241, 1)
    for ax, sc in zip(axes, ("S0", "S5"), strict=True):
        for p in PLAN_ORDER:
            if p not in hist[sc]:
                continue
            surv = _survival(hist[sc][p], xs)
            ax.plot(xs, np.where(surv > 0, surv, np.nan), color=PLAN_COLOR[p], lw=1.6)
        ax.axvline(15, color=AXIS, lw=0.8, ls=(0, (3, 3)))
        ax.text(16, 0.9, "15 min", fontsize=8, color=MUTED, va="top")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.0)
        ax.set_xlim(0, 240)
        ax.set_xticks(np.arange(0, 241, 30))
        ax.set_title(f"{sc}  {ev['scenarios'][sc]}")
        ax.set_xlabel("Arrival delay x (min)")
        _grid_y(ax)
        ax.yaxis.set_major_formatter(lambda y, _p: f"{y:.0%}" if y >= 0.01 else f"{y:.2%}")
    axes[0].set_ylabel("Share of flights later than x")
    handles = [Line2D([0], [0], color=PLAN_COLOR[p], lw=1.8) for p in PLAN_ORDER if p in hist["S0"]]
    body.legend(handles, [PLAN_LABEL[p] for p in PLAN_ORDER if p in hist["S0"]], loc="outside lower center",
                ncol=4, fontsize=8.5)  # fmt: skip
    return _save(fig, out, "fig06_delay_tails")


# --------------------------------------------------------------------------- figure 9


def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a != b


def fig09_pareto(results: Path, out: Path) -> list[Path]:
    """Planned cost against expected disruption cost over the buffer and penalty grid."""
    sweep = _load(results, "sweep.json")
    ev = _load(results, "evaluation.json")
    plans = _load(results, "plans.json")
    horizon = _horizon(results)
    rows = [r for r in sweep["rows"] if r.get("feasible")]
    pts = [(r["planned_cost_day"], r["S0"]["operational_cost"]["mean"] / horizon) for r in rows]
    front = [k for k, p in enumerate(pts) if not any(_dominates(q, p) for q in pts)]
    fig, body = _figure(
        9.0,
        4.8,
        f"Planned cost against expected disruption cost for {len(rows)} robust plans",
        "Each point is a robust plan with hub buffer b (minutes) and "
        "delay-penalty weight w, under baseline delays. The line joins "
        "plans that no other plan beats on both costs. Diamonds: the "
        "three reference plans. Means of the replications, 95% intervals.",
    )
    ax = body.subplots()
    for k, (r, (x, y)) in enumerate(zip(rows, pts, strict=True)):
        lo, hi = r["S0"]["operational_cost"]["lo"] / horizon, r["S0"]["operational_cost"]["hi"] / horizon
        on = k in front
        ax.plot([x, x], [lo, hi], color=TEXT if on else AXIS, lw=0.9, zorder=2)
        ax.scatter([x], [y], s=34 if on else 26, facecolor=TEXT if on else "white", edgecolor=TEXT if on else MUTED,
                   linewidth=1.0, zorder=4)  # fmt: skip
    order = sorted(front, key=lambda k: pts[k][0])
    ax.plot([pts[k][0] for k in order], [pts[k][1] for k in order], color=TEXT, lw=1.3, zorder=3)
    for k in order:
        r = rows[k]
        ax.annotate(f"b={r['buffer']}, w={r['weight']:g}", pts[k], xytext=(-6, -4), textcoords="offset points",
                    ha="right", va="top", fontsize=7.8, color=TEXT)  # fmt: skip
    refs: list[tuple[float, float, list[str]]] = []
    for p in ("sequential", "integrated", "robust"):
        x = plans[p]["planned_cost"]["total"]
        y = ev["plans"][p]["S0"]["operational_cost"]["mean"] / horizon
        ax.scatter([x], [y], marker="D", s=60, color=PLAN_COLOR[p], edgecolor="white", linewidth=1.2, zorder=6)
        for r in refs:
            if abs(r[0] - x) < 1.0 and abs(r[1] - y) < 1.0:
                r[2].append(PLAN_LABEL[p])
                break
        else:
            refs.append((x, y, [PLAN_LABEL[p]]))
    for x, y, names in refs:
        ax.annotate(" = ".join(names), (x, y), xytext=(8, 6), textcoords="offset points", ha="left",
                    fontsize=8.5, color=TEXT)  # fmt: skip
    ax.xaxis.set_major_locator(MultipleLocator(2000))
    ax.yaxis.set_major_locator(MultipleLocator(2000))
    ax.xaxis.set_major_formatter(_euro_k)
    ax.yaxis.set_major_formatter(_euro_k)
    ax.set_xlabel("Planned cost (€ per day)")
    ax.set_ylabel("Expected disruption cost (€ per day)")
    _grid_y(ax)
    _grid_x(ax)
    return _save(fig, out, "fig09_pareto_frontier")


# --------------------------------------------------------------------------- figure 10


def fig10_buffer_climate(results: Path, out: Path) -> list[Path]:
    """Excess total cost of each buffer against the best buffer, by delay climate."""
    clim = _load(results, "climates.json")
    horizon = _horizon(results)
    rows = [r for r in clim["rows"] if "buffer" in r]
    climates = sorted({r["climate"] for r in rows})
    buffers = sorted({r["buffer"] for r in rows})
    total = np.full((len(climates), len(buffers)), np.nan)
    for r in rows:
        total[climates.index(r["climate"]), buffers.index(r["buffer"])] = r["S0"]["total_cost"]["mean"] / horizon
    excess = total - np.nanmin(total, axis=1, keepdims=True)
    best = np.nanargmin(total, axis=1)
    fig, body = _figure(
        9.0,
        0.62 * len(climates) + 1.3,
        "Excess total cost of each hub buffer over the best buffer, by delay climate",
        "€ per day above the cheapest buffer in the same row (ringed), "
        "robust planner without the delay penalty, baseline scenario. "
        "Rows are the mean primary delay per flight fed to the simulator.",
    )
    ax = body.subplots()
    vmax = float(np.nanmax(excess)) or 1.0
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", SEQUENTIAL_RAMP)
    ax.imshow(excess, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(len(climates)):
        for j in range(len(buffers)):
            v = excess[i, j]
            if math.isnan(v):
                continue
            dark = v > 0.55 * vmax
            text = "best" if j == best[i] else f"+€{v:,.0f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color="white" if dark else TEXT)
            if j == best[i]:
                ax.add_patch(Rectangle((j - 0.46, i - 0.42), 0.92, 0.84, fill=False, edgecolor=TEXT, lw=1.4))
    labels = {6.8: "November 2025", 9.5: "2024 average", 19.0: "stressed (2×)"}
    ax.set_yticks(range(len(climates)),
                  [f"{c:g} min · {labels.get(c, '')}".rstrip(" ·") for c in climates])  # fmt: skip
    ax.set_xticks(range(len(buffers)), [f"{b} min" for b in buffers])
    ax.set_xlabel("Hub turn buffer")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return _save(fig, out, "fig10_buffer_climate_heatmap")


# --------------------------------------------------------------------------- entry point


def draw_all(results: Path, out: Path) -> list[Path]:
    """Draw every figure of the article.

    Raises:
        DataError: When a results file the figures need is missing.

    """
    _style()
    jobs: list[Callable[[], list[Path]]] = [
        lambda: fig01_rotation(results, out),
        lambda: fig02_time_space(results, out),
        lambda: fig03_pairing_network(results, out),
        lambda: fig04_column_generation(results, out),
        lambda: fig05_propagation(results, out),
        lambda: fig06_buffer_curves(results, out),
        lambda: fig07_strategy_comparison(results, out),
        lambda: fig08_delay_tails(results, out),
        lambda: fig09_pareto(results, out),
        lambda: fig10_buffer_climate(results, out),
    ]
    written: list[Path] = []
    for job in jobs:
        written.extend(job())
    return written
