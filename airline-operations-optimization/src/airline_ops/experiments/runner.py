"""The experiment pipeline behind the article.

Stages (each writes a JSON file in the results directory):

1. ``network``: generate the main synthetic network and record its shape.
2. ``calibration``: choose the shape of the primary delay distribution so that the
   sequential plan's simulated departure punctuality matches EUROCONTROL's 2024 value,
   with the mean primary delay fixed at the CODA figure.
3. ``plans``: build the four plans, validate them, record planning cost and solver data.
4. ``evaluation``: simulate every plan in every scenario with common random numbers.
5. ``sweep``: robust plans over a grid of buffers and penalty weights (Pareto frontier).
6. ``climates``: buffer-only plans under three delay climates (buffer sensitivity).
7. ``learning``: robust plans whose penalty comes from a predictor fitted to simulated
   operating history, as a point forecast or as a full distribution.
8. ``structural``: the four planners on further network seeds.
9. ``propagation``: one disruption traced flight by flight.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from airline_ops.config import Config
from airline_ops.data.airports import AirportRecord, fetch_airports_file, parse_airports, read_contract
from airline_ops.errors import AirlineOpsError, InfeasibleModelError
from airline_ops.experiments.stats import mean_ci, paired_difference, pareto_front, quantiles
from airline_ops.logging_setup import get_logger
from airline_ops.network.generator import SPEED_EAST_KMH, SPEED_WEST_KMH, generate_network
from airline_ops.network.model import DAY, Network
from airline_ops.planning.arcs import aircraft_count
from airline_ops.planning.plan import Plan, planned_cost, rotation_summary
from airline_ops.planning.planners import (
    RobustSettings,
    check_plan,
    plan_greedy,
    plan_integrated,
    plan_robust,
    plan_sequential,
)
from airline_ops.planning.rules import PlanningRules
from airline_ops.simulation.delays import DelayModel, EmpiricalPenalty, PropagationPenalty
from airline_ops.simulation.scenarios import SCENARIOS, Scenario, draw_events
from airline_ops.simulation.simulator import Outcome, Unrolled, simulate, unroll

LOG = get_logger("experiments.runner")
PLANNERS = ("greedy", "sequential", "integrated", "robust")
PLANNING_SEED_OFFSET = 7_000_003  # robust penalties are estimated on draws never used for evaluation
HISTORY_SEED_OFFSET = 9_000_011


def _write(out_dir: Path, name: str, payload: Any) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(payload, indent=1, sort_keys=True, default=_json_default) + "\n")


def _json_default(obj: object) -> object:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def load_records(config: Config, contract_path: Path, cache_dir: Path) -> dict[str, AirportRecord]:
    """Verified OpenFlights rows for the airports in the configuration."""
    contract = read_contract(contract_path)
    data = fetch_airports_file(contract, cache_dir)
    codes = {config.network.hub, *(s.code for s in config.network.spokes)}
    return parse_airports(data, codes)


def penalty_function(network: Network, model: DelayModel, samples: int, seed: int) -> Callable[[int, float], float]:
    """Expected excess delay of a connection out of flight ``i`` with ``slack`` minutes."""
    pen = PropagationPenalty(model, samples, seed)
    blocks = [f.block for f in network.flights]
    return lambda i, slack: pen.expected_excess(blocks[i], slack)


# ---------------------------------------------------------------------------------
# Per-replication metrics
# ---------------------------------------------------------------------------------
def rep_metrics(
    out: Outcome,
    base: Outcome | None,
    plan_cost: dict[str, float],
    onset: int,
    horizon: int,
    plan_aircraft: int,
) -> dict[str, float]:
    """Scalar metrics of one replication."""
    op = out.operated
    dep = np.where(op, out.dep_delay, np.nan)
    arr = np.where(op, out.arr_delay, np.nan)
    pos_dep = np.clip(np.nan_to_num(dep, nan=0.0), 0, None)
    total_dep = float(pos_dep.sum())
    react = float(out.reactionary[op].sum())
    c = out.counters
    planned_total = plan_cost["total"] * horizon
    footprint, recovery = 0.0, 0.0
    if base is not None:
        changed = (out.operated != base.operated) | (
            out.operated & base.operated & (np.abs(np.nan_to_num(out.dep_delay) - np.nan_to_num(base.dep_delay)) > 15.0)
        )
        changed &= out.sched_dep >= onset
        if changed.any():
            footprint = float(changed.sum())
            recovery = float(out.sched_dep[changed].max() - onset)
    return {
        "total_cost": planned_total + out.costs["operational_total"],
        "planned_cost": planned_total,
        "operational_cost": out.costs["operational_total"],
        "eurocontrol_reference": out.costs["eurocontrol_reference"],
        "eu261": out.costs["eu261"],
        "aircraft_utilization_h": c["block_min"] / 60.0 / max(1, plan_aircraft) / horizon,
        "crew_utilization": c["block_min"] / c["crew_paid_min"] if c["crew_paid_min"] else math.nan,
        "cancelled": c["cancelled"],
        "pax_delay_min": c["pax_delay_min"],
        "pax_delay_per_pax": c["pax_delay_min"] / c["passengers"] if c["passengers"] else math.nan,
        "missed_connection_pax": c["missed_connection_pax"],
        "crew_hotel": plan_cost["hotel"] * horizon,
        "recovery_min": recovery,
        "footprint": footprint,
        "d15": float(np.nanmean(dep <= 15.0)),
        "a15": float(np.nanmean(arr <= 15.0)),
        "mean_dep_delay": float(np.nanmean(np.clip(dep, 0, None))),
        "reactionary_share": react / total_dep if total_dep > 0 else 0.0,
        "discretion_uses": c["discretion_uses"],
        "standby_calls": c["standby_calls"],
        "swaps": c["swaps"],
        "ferries": c["ferries"],
        "repositionings": c["repositionings"],
        "overtime_min": c["overtime_min"],
    }


def evaluate_plan(
    network: Network,
    plan: Plan,
    config: Config,
    model: DelayModel,
    scenarios: tuple[Scenario, ...],
    reps: int,
    base_seed: int,
) -> dict[str, list[dict[str, float]]]:
    """Simulate a plan in each scenario; returns per-replication metrics by scenario."""
    rules = PlanningRules.from_config(config, network)
    unrolled = unroll(network, plan, config.simulation.horizon_days, rules)
    cost = planned_cost(plan, network, config.costs)
    aircraft = aircraft_count(plan.successor, network)
    out: dict[str, list[dict[str, float]]] = {}
    baseline: list[Outcome] = []
    s0 = next(s for s in SCENARIOS if s.name == "S0")
    for rep in range(reps):
        ev = draw_events(s0, network, np.random.default_rng([base_seed, rep, 0]))
        baseline.append(simulate(network, plan, unrolled, config, model, ev, base_seed, rep))
    for sc in scenarios:
        rows: list[dict[str, float]] = []
        for rep in range(reps):
            ev = draw_events(sc, network, np.random.default_rng([base_seed, rep, int(sc.name[1:])]))
            res = (
                baseline[rep]
                if sc.name == "S0"
                else simulate(network, plan, unrolled, config, model, ev, base_seed, rep)
            )
            rows.append(
                rep_metrics(
                    res,
                    baseline[rep] if sc.name != "S0" else None,
                    cost,
                    ev.onset,
                    config.simulation.horizon_days,
                    aircraft,
                )
            )
        out[sc.name] = rows
    return out


def summarise(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and 95% interval of every metric."""
    keys = rows[0].keys() if rows else []
    return {k: mean_ci([r[k] for r in rows]) for k in keys}


def arrival_histogram(
    network: Network, plan: Plan, config: Config, model: DelayModel, scenario: Scenario, reps: int, base_seed: int
) -> dict[str, Any]:
    """Histogram of arrival delays (1-minute bins from -30 to 300) and cancellations."""
    rules = PlanningRules.from_config(config, network)
    unrolled = unroll(network, plan, config.simulation.horizon_days, rules)
    edges = np.arange(-30, 301, 1)
    counts = np.zeros(edges.size - 1)
    over = 0
    cancelled = 0
    for rep in range(reps):
        ev = draw_events(scenario, network, np.random.default_rng([base_seed, rep, int(scenario.name[1:])]))
        res = simulate(network, plan, unrolled, config, model, ev, base_seed, rep)
        d = res.arr_delay[res.operated]
        counts += np.histogram(np.clip(d, -30, 299.5), bins=edges)[0]
        over += int((d >= 300).sum())
        cancelled += int((~res.operated).sum())
    return {"edges": edges.tolist(), "counts": counts.tolist(), "over_300": over, "cancelled": cancelled}


# ---------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------
def network_summary(network: Network, config: Config) -> dict[str, Any]:
    """Shape of the network, for the article and the figures."""
    flights = network.flights
    hub = network.airports[network.hub]
    return {
        "seed": network.seed,
        "flights_per_day": network.n_flights,
        "fleet": network.fleet_size,
        "minimum_fleet": network.minimum_fleet(),
        "spokes": len(network.airports) - 1,
        "passengers_per_day": sum(g.count for g in network.passengers),
        "connecting_passengers_per_day": sum(g.count for g in network.passengers if len(g.legs) == 2),
        "nightstop_spokes": sum(1 for s in config.network.spokes if s.frequency >= 2),
        "block_model": {"speed_west_kmh": SPEED_WEST_KMH, "speed_east_kmh": SPEED_EAST_KMH, "ground_min": 30.0},
        "mean_block_min": float(np.mean([f.block for f in flights])),
        "flights": [
            {
                "number": f.number,
                "origin": f.origin,
                "destination": f.destination,
                "dep_utc": f.dep,
                "block": f.block,
                "dep_local": network.airports[f.origin].local_minute(f.dep),
                "arr_local": network.airports[f.destination].local_minute(f.arr),
                "distance_km": f.distance_km,
            }
            for f in flights
        ],
        "hub_utc_offset_min": hub.utc_offset_min,
        "airports": {
            code: {"lat": a.latitude, "lon": a.longitude, "utc_offset_min": a.utc_offset_min, "name": a.name}
            for code, a in network.airports.items()
        },
    }


def calibrate(
    network: Network, plan: Plan, config: Config, reps: int, base_seed: int
) -> tuple[DelayModel, dict[str, Any]]:
    """Grid search over (p, sigma) at fixed mean primary delay to match the 2024 D15 punctuality."""
    target_d15 = 0.692  # EUROCONTROL CODA Digest, annual 2024: 69.2% of departures within 15 minutes
    target_react = 0.46  # CODA 2024: reactionary share of delay minutes (validation only)
    grid: list[dict[str, float]] = []
    s0 = SCENARIOS[0]
    rules = PlanningRules.from_config(config, network)
    unrolled = unroll(network, plan, config.simulation.horizon_days, rules)
    best: tuple[float, DelayModel] | None = None
    for p in (0.35, 0.45, 0.55, 0.65, 0.75):
        for sigma in (0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
            model = DelayModel(
                config.delays.mean_primary_min, p, sigma, config.delays.block_noise_sd, config.delays.block_bias
            )
            d15s, reacts, means = [], [], []
            for rep in range(reps):
                ev = draw_events(s0, network, np.random.default_rng([base_seed, rep, 0]))
                res = simulate(network, plan, unrolled, config, model, ev, base_seed, rep)
                dep = res.dep_delay[res.operated]
                d15s.append(float(np.mean(dep <= 15.0)))
                pos = float(np.clip(dep, 0, None).sum())
                reacts.append(float(res.reactionary[res.operated].sum()) / pos if pos else 0.0)
                means.append(float(np.clip(dep, 0, None).mean()))
            d15 = float(np.mean(d15s))
            grid.append(
                {
                    "p": p,
                    "sigma": sigma,
                    "d15": d15,
                    "reactionary_share": float(np.mean(reacts)),
                    "mean_dep_delay": float(np.mean(means)),
                }
            )
            score = abs(d15 - target_d15)
            if best is None or score < best[0]:
                best = (score, model)
    assert best is not None
    chosen = best[1]
    row = next(g for g in grid if g["p"] == chosen.p_delayed and g["sigma"] == chosen.sigma)
    return chosen, {
        "targets": {
            "d15": target_d15,
            "reactionary_share_validation": target_react,
            "mean_primary_min": config.delays.mean_primary_min,
        },
        "grid": grid,
        "chosen": {"p_delayed": chosen.p_delayed, "sigma": chosen.sigma, **row},
        "replications": reps,
    }


def plan_record(plan: Plan, network: Network, config: Config) -> dict[str, Any]:
    """Planning metrics of one plan."""
    cost = planned_cost(plan, network, config.costs)
    rot = rotation_summary(plan, network)
    durations = [p.span for p in plan.pairings]
    duties = [d for p in plan.pairings for d in p.duties]
    return {
        "name": plan.name,
        "planned_cost": cost,
        "rotations": rot,
        "pairings": len(plan.pairings),
        "two_duty_pairings": sum(1 for p in plan.pairings if len(p.duties) == 2),
        "duties": len(duties),
        "mean_duty_h": float(np.mean([d.length for d in duties]) / 60.0) if duties else 0.0,
        "mean_fdp_h": float(np.mean([d.fdp for d in duties]) / 60.0) if duties else 0.0,
        "mean_pairing_span_h": float(np.mean(durations) / 60.0) if durations else 0.0,
        "solve_seconds": plan.solve_seconds,
        "lower_bound": plan.lower_bound,
        "objective": plan.objective,
        "violations": plan.notes.get("violations", []),
        "successor": {str(i): j for i, j in sorted(plan.successor.items())},
        "pairing_duties": [
            {
                "duties": [list(d.flights) for d in p.duties],
                "starts": [d.start for d in p.duties],
                "nights": p.hotel_nights,
            }
            for p in plan.pairings
        ],
        "notes": {k: v for k, v in plan.notes.items() if k != "violations"},
        "history": plan.history,
    }


def fdp_margin(plan: Plan, network: Network, config: Config) -> list[float]:
    """Minutes between each planned FDP and its Table 2 limit."""
    rules = PlanningRules.from_config(config, network)
    out: list[float] = []
    for p in plan.pairings:
        for d in p.duties:
            first = network.flights[d.flights[0]]
            local = network.airports[first.origin].local_minute(d.report_utc)
            sectors = sum(1 for dh in d.deadhead if not dh) or 1
            out.append(float(rules.max_fdp(local, min(sectors, 10)) - d.fdp))
    return out


def connection_slack(plan: Plan, network: Network, config: Config) -> dict[str, list[float]]:
    """Slack above the legal minimum on aircraft turns and on crew aircraft changes, at the hub."""
    rules = PlanningRules.from_config(config, network)
    flights = network.flights
    ac: list[float] = []
    for i, j in plan.successor.items():
        fi, fj = flights[i], flights[j]
        gap = (fj.dep - fi.arr) % DAY
        if fi.destination == network.hub and gap < 600:
            ac.append(float(gap - rules.turn_min[fi.destination]))
    crew: list[float] = []
    for p in plan.pairings:
        for d in p.duties:
            for a, b in zip(d.flights, d.flights[1:], strict=False):
                if plan.successor.get(a) == b:
                    continue
                gap = (flights[b].dep - flights[a].arr) % DAY
                crew.append(float(gap - rules.crew_change_min))
    return {"aircraft_hub_turn_slack": ac, "crew_change_slack": crew}


def run_experiment(
    config: Config, out_dir: Path, contract: Path, cache_dir: Path, *, stages: set[str] | None = None
) -> None:
    """Run all stages and write their results.

    Raises:
        AirlineOpsError: When a stage fails; partial results already written are kept.

    """
    t_start = time.perf_counter()
    exp = config.experiment
    records = load_records(config, contract, cache_dir)
    main_seed = exp.network_seeds[0]
    network = generate_network(config.network, records, main_seed)
    _write(out_dir, "network.json", network_summary(network, config))
    reps = exp.replications
    base_seed = exp.seed

    def want(stage: str) -> bool:
        return stages is None or stage in stages

    # Plans that do not depend on the delay model come first; calibration uses the sequential plan.
    LOG.info("planning greedy, sequential and integrated", extra={"fields": {"seed": main_seed}})
    plans: dict[str, Plan] = {
        "greedy": plan_greedy(network, config),
        "sequential": plan_sequential(network, config),
        "integrated": plan_integrated(network, config),
    }
    model, calib = calibrate(network, plans["sequential"], config, exp.calibration_replications, base_seed + 1)
    _write(out_dir, "calibration.json", calib)
    LOG.info("calibrated delay model", extra={"fields": asdict(model)})
    excess = penalty_function(network, model, config.robust.penalty_samples, base_seed + PLANNING_SEED_OFFSET)
    plans["robust"] = plan_robust(
        network, config, RobustSettings(config.robust.buffer_min, config.robust.penalty_weight), excess
    )
    records_out: dict[str, Any] = {}
    for name, plan in plans.items():
        check_plan(plan, network, config)
        rec = plan_record(plan, network, config)
        rec["fdp_margin_min"] = fdp_margin(plan, network, config)
        rec["slack"] = connection_slack(plan, network, config)
        records_out[name] = rec
    _write(out_dir, "plans.json", records_out)

    if want("evaluation"):
        evaluation: dict[str, Any] = {
            "scenarios": {s.name: s.label for s in SCENARIOS},
            "plans": {},
            "paired_vs_sequential": {},
        }
        per_rep: dict[str, dict[str, list[dict[str, float]]]] = {}
        for name, plan in plans.items():
            LOG.info("evaluating plan", extra={"fields": {"plan": name, "replications": reps}})
            per_rep[name] = evaluate_plan(network, plan, config, model, SCENARIOS, reps, base_seed)
            evaluation["plans"][name] = {sc: summarise(rows) for sc, rows in per_rep[name].items()}
        for name in plans:
            if name == "sequential":
                continue
            evaluation["paired_vs_sequential"][name] = {
                sc: {
                    k: paired_difference([r[k] for r in per_rep[name][sc]], [r[k] for r in per_rep["sequential"][sc]])
                    for k in (
                        "total_cost",
                        "operational_cost",
                        "pax_delay_min",
                        "cancelled",
                        "missed_connection_pax",
                        "recovery_min",
                    )
                }
                for sc in per_rep[name]
            }
        # The article's question also needs the robust plan against the integrated one.
        evaluation["paired_robust_vs_integrated"] = {
            sc: {
                k: paired_difference([r[k] for r in per_rep["robust"][sc]], [r[k] for r in per_rep["integrated"][sc]])
                for k in ("total_cost", "operational_cost", "pax_delay_min", "cancelled", "missed_connection_pax")
            }
            for sc in per_rep["robust"]
        }
        evaluation["histograms"] = {
            sc: {
                name: arrival_histogram(network, plan, config, model, SCENARIOS[int(sc[1:])], reps, base_seed)
                for name, plan in plans.items()
            }
            for sc in ("S0", "S5")
        }
        evaluation["replications"] = reps
        _write(out_dir, "evaluation.json", evaluation)

    robust_cache: dict[tuple[int, float], Plan] = {}

    def robust_plan(b: int, w: float) -> Plan:
        if (b, w) not in robust_cache:
            robust_cache[(b, w)] = plan_robust(network, config, RobustSettings(b, w), excess, name=f"robust_b{b}_w{w}")
        return robust_cache[(b, w)]

    robust_cache[(config.robust.buffer_min, config.robust.penalty_weight)] = plans["robust"]

    if want("sweep"):
        sweep_rows: list[dict[str, Any]] = []
        for b in exp.buffer_sweep:
            for w in exp.penalty_sweep:
                try:
                    plan = robust_plan(b, w)
                except InfeasibleModelError as err:
                    sweep_rows.append({"buffer": b, "weight": w, "feasible": False, "reason": str(err)})
                    continue
                violations = check_plan(plan, network, config)
                ev = evaluate_plan(network, plan, config, model, (SCENARIOS[0], SCENARIOS[5]), reps, base_seed)
                cost = planned_cost(plan, network, config.costs)
                sweep_rows.append(
                    {
                        "buffer": b,
                        "weight": w,
                        "feasible": not violations,
                        "violations": violations,
                        "planned_cost_day": cost["total"],
                        "aircraft": cost["aircraft"],
                        "crew_pay_day": cost["crew_pay"],
                        "solve_seconds": plan.solve_seconds,
                        "gap": plan.notes.get("gap_to_lp_bound"),
                        "S0": summarise(ev["S0"]),
                        "S5": summarise(ev["S5"]),
                        "slack": connection_slack(plan, network, config),
                    }
                )
                LOG.info("sweep point", extra={"fields": {"buffer": b, "weight": w}})
        feasible = [r for r in sweep_rows if r.get("feasible")]
        pts = [
            (r["planned_cost_day"], r["S0"]["operational_cost"]["mean"] / config.simulation.horizon_days)
            for r in feasible
        ]
        front = [{"buffer": feasible[k]["buffer"], "weight": feasible[k]["weight"]} for k in pareto_front(pts)]
        _write(out_dir, "sweep.json", {"rows": sweep_rows, "pareto_front": front})

    if want("climates"):
        climate_rows: list[dict[str, Any]] = []
        buffer_plans: dict[int, Plan] = {}
        for b in exp.buffer_sweep:
            try:
                buffer_plans[b] = robust_plan(b, 0.0)
            except InfeasibleModelError:
                continue
        for climate in exp.delay_climates:
            m = replace(model, mean_primary_min=climate)
            for b, plan in buffer_plans.items():
                ev = evaluate_plan(network, plan, config, m, (SCENARIOS[0],), reps, base_seed)
                climate_rows.append(
                    {
                        "climate": climate,
                        "buffer": b,
                        "planned_cost_day": planned_cost(plan, network, config.costs)["total"],
                        "S0": summarise(ev["S0"]),
                    }
                )
            for name in ("sequential", "integrated", "robust"):
                ev = evaluate_plan(network, plans[name], config, m, (SCENARIOS[0],), reps, base_seed)
                climate_rows.append({"climate": climate, "plan": name, "S0": summarise(ev["S0"])})
        _write(out_dir, "climates.json", {"rows": climate_rows})

    if want("learning"):
        _write(out_dir, "learning.json", learning_stage(network, plans, config, model, excess, reps, base_seed))

    if want("structural"):
        rows_s: list[dict[str, Any]] = []
        for seed in exp.network_seeds[1:]:
            net = generate_network(config.network, records, seed)
            ex = penalty_function(net, model, config.robust.penalty_samples, base_seed + PLANNING_SEED_OFFSET)
            built: dict[str, Plan] = {}
            try:
                built = {
                    "greedy": plan_greedy(net, config),
                    "sequential": plan_sequential(net, config),
                    "integrated": plan_integrated(net, config),
                    "robust": plan_robust(
                        net, config, RobustSettings(config.robust.buffer_min, config.robust.penalty_weight), ex
                    ),
                }
            except AirlineOpsError as err:
                rows_s.append({"seed": seed, "error": str(err)})
                continue
            entry: dict[str, Any] = {"seed": seed, "flights": net.n_flights, "fleet": net.fleet_size, "plans": {}}
            for name, plan in built.items():
                violations = check_plan(plan, net, config)
                ev = evaluate_plan(
                    net, plan, config, model, (SCENARIOS[0], SCENARIOS[1], SCENARIOS[5]), reps, base_seed
                )
                entry["plans"][name] = {
                    "planned_cost_day": planned_cost(plan, net, config.costs)["total"],
                    "violations": len(violations),
                    "solve_seconds": plan.solve_seconds,
                    "gap": plan.notes.get("gap_to_lp_bound"),
                    **{sc: summarise(rows) for sc, rows in ev.items()},
                }
            rows_s.append(entry)
            LOG.info("structural replication done", extra={"fields": {"seed": seed}})
        _write(out_dir, "structural.json", {"rows": rows_s})

    if want("examples"):
        _write(out_dir, "examples.json", examples_stage(network, plans, config))

    if want("propagation"):
        _write(out_dir, "propagation.json", propagation_stage(network, plans, config, model, base_seed))

    _write(
        out_dir, "run.json", {"seconds": time.perf_counter() - t_start, "config": config.source, "model": asdict(model)}
    )


def learning_stage(
    network: Network,
    plans: dict[str, Plan],
    config: Config,
    model: DelayModel,
    excess: Callable[[int, float], float],
    reps: int,
    base_seed: int,
) -> dict[str, Any]:
    """Predict-then-optimise: penalties from a point forecast versus a predictive distribution.

    The training history is the sequential plan operated on replications drawn with a
    seed that evaluation never uses. For each daily flight, the predictor is the
    empirical distribution of its arrival delay in that history (a non-parametric
    conditional model); the point forecast keeps only its mean.
    """
    exp = config.experiment
    rules = PlanningRules.from_config(config, network)
    plan = plans["sequential"]
    unrolled: Unrolled = unroll(network, plan, config.simulation.horizon_days, rules)
    n = network.n_flights
    samples: list[list[float]] = [[] for _ in range(n)]
    s0 = SCENARIOS[0]
    hist_seed = base_seed + HISTORY_SEED_OFFSET
    for rep in range(exp.history_replications):
        ev = draw_events(s0, network, np.random.default_rng([hist_seed, rep, 0]))
        res = simulate(network, plan, unrolled, config, model, ev, hist_seed, rep)
        for i in np.nonzero(res.operated)[0]:
            samples[int(i) % n].append(float(res.arr_delay[i]))
    arrays = [np.array(s) for s in samples]
    point = EmpiricalPenalty(arrays, point=True)
    dist = EmpiricalPenalty(arrays, point=False)
    settings = RobustSettings(0, config.robust.penalty_weight)
    variants: dict[str, Plan] = {
        "no_penalty": plan_robust(network, config, RobustSettings(0, 0.0), excess, name="learn_none"),
        "primary_model": plan_robust(network, config, settings, excess, name="learn_primary"),
        "point_forecast": plan_robust(network, config, settings, point.expected_excess_flight, name="learn_point"),
        "distribution": plan_robust(network, config, settings, dist.expected_excess_flight, name="learn_dist"),
    }
    out: dict[str, Any] = {
        "history_replications": exp.history_replications,
        "mean_abs_forecast": float(np.mean([abs(a.mean()) for a in arrays if a.size])),
        "share_flights_mean_above_10": float(np.mean([a.mean() > 10 for a in arrays if a.size])),
        "variants": {},
    }
    for name, p in variants.items():
        violations = check_plan(p, network, config)
        evals = evaluate_plan(network, p, config, model, (SCENARIOS[0], SCENARIOS[5]), reps, base_seed)
        out["variants"][name] = {
            "planned_cost_day": planned_cost(p, network, config.costs)["total"],
            "violations": len(violations),
            "S0": summarise(evals["S0"]),
            "S5": summarise(evals["S5"]),
            "slack": connection_slack(p, network, config),
        }
    return out


def propagation_stage(
    network: Network, plans: dict[str, Plan], config: Config, model: DelayModel, base_seed: int
) -> dict[str, Any]:
    """Trace the delayed-inbound scenario flight by flight for one replication and two plans."""
    s1 = SCENARIOS[1]
    rules = PlanningRules.from_config(config, network)
    rep = 3
    ev = draw_events(s1, network, np.random.default_rng([base_seed, rep, 1]))
    s0 = draw_events(SCENARIOS[0], network, np.random.default_rng([base_seed, rep, 0]))
    out: dict[str, Any] = {
        "replication": rep,
        "delayed_flights": {network.flights[k].number: v for k, v in ev.inbound_delays.items()},
        "plans": {},
    }
    n = network.n_flights
    for name in ("sequential", "robust"):
        plan = plans[name]
        u = unroll(network, plan, config.simulation.horizon_days, rules)
        res = simulate(network, plan, u, config, model, ev, base_seed, rep)
        base = simulate(network, plan, u, config, model, s0, base_seed, rep)
        nodes = []
        for i in range(res.operated.size):
            extra = (
                (np.nan_to_num(res.dep_delay[i]) - np.nan_to_num(base.dep_delay[i]))
                if res.operated[i] and base.operated[i]
                else math.nan
            )
            changed = (res.operated[i] != base.operated[i]) or (not math.isnan(extra) and abs(extra) > 1.0)
            if not changed:
                continue
            nodes.append(
                {
                    "instance": i,
                    "number": network.flights[i % n].number,
                    "origin": network.flights[i % n].origin,
                    "destination": network.flights[i % n].destination,
                    "sched_dep": float(res.sched_dep[i]),
                    "extra_delay": None if math.isnan(extra) else float(extra),
                    "cancelled": bool(not res.operated[i]),
                    "bind_type": int(res.bind_type[i]),
                    "bind_pred": int(res.bind_pred[i]),
                }
            )
        out["plans"][name] = {
            "nodes": nodes,
            "extra_delay_total": float(np.nansum([x["extra_delay"] or 0.0 for x in nodes])),
        }
    return out


def examples_stage(network: Network, plans: dict[str, Plan], config: Config, station: str = "OPO") -> dict[str, Any]:
    """The legal duties and two-duty pairings that serve one outstation, for the pairing-network figure."""
    from airline_ops.planning.duties import build_pairing_space, enumerate_duties

    rules = PlanningRules.from_config(config, network)
    ds = enumerate_duties(network, rules)
    sp = build_pairing_space(ds, rules)
    flights = network.flights
    touch = {f.idx for f in flights if station in (f.origin, f.destination)}
    out_duties = [d for d in ds.duties if d.end_station == station and d.start_station == rules.base]
    back_duties = [d for d in ds.duties if d.start_station == station and d.end_station == rules.base]
    keep = {d.idx for d in out_duties + back_duties}
    pairs = [
        {"first": int(a), "second": int(b), "rest": int(r)}
        for a, b, r in zip(sp.pair_first, sp.pair_second, sp.pair_rest, strict=True)
        if int(a) in keep and int(b) in keep
    ]
    singles = [int(k) for k in sp.single if set(ds.duties[int(k)].flights) & touch]
    chosen: dict[str, list[list[list[int]]]] = {}
    for name, plan in plans.items():
        chosen[name] = [[list(d.flights) for d in p.duties] for p in plan.pairings if set(p.flights) & touch]

    def duty_json(d: Any) -> dict[str, Any]:
        return {
            "idx": d.idx,
            "flights": [flights[f].number for f in d.flights],
            "flight_ids": list(d.flights),
            "report_utc": d.report_utc,
            "length": d.length,
            "fdp": d.fdp,
            "start": d.start_station,
            "end": d.end_station,
        }

    return {
        "station": station,
        "out_duties": [duty_json(d) for d in out_duties],
        "back_duties": [duty_json(d) for d in back_duties],
        "single_duties_touching": [duty_json(ds.duties[k]) for k in singles],
        "pairs": pairs,
        "chosen": chosen,
        "n_duties_network": ds.size,
        "n_pairings_network": sp.size,
    }


def quick_quantiles(values: list[float]) -> list[float]:
    """5th, 25th, 50th, 75th and 95th percentiles."""
    return quantiles(values, (0.05, 0.25, 0.5, 0.75, 0.95))
