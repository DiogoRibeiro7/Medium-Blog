"""Analyses added after the main results, kept apart from the registered experiment.

The sweep in the main run reports each robust plan's mean costs, not paired differences.
This module re-plans a few of those plans on the main network and evaluates them with the
same seeds as the integrated plan, so that their differences come with paired intervals:

- the penalty without a buffer (b = 0, w = 0.5), the plan with the lowest expected total
  cost under baseline delays in the sweep;
- the largest buffer and weight (b = 45, w = 2), the lowest under the stressed day;
- buffers alone (w = 0) at 0 and 45 minutes, under the calm and the stressed delay climates;
- the penalties learned from a simulated history (H5): a point forecast and the empirical
  distribution of each flight's arrival delay, rebuilt exactly as in the main run.

It also reports where each plan puts its slack. Nothing here is a registered hypothesis;
results are written to ``details.json`` and labelled as descriptive in the article.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from airline_ops.config import Config
from airline_ops.errors import DataError
from airline_ops.experiments.runner import (
    HISTORY_SEED_OFFSET,
    PLANNING_SEED_OFFSET,
    connection_slack,
    evaluate_plan,
    load_records,
    penalty_function,
    summarise,
)
from airline_ops.experiments.stats import paired_difference
from airline_ops.logging_setup import get_logger
from airline_ops.network.generator import generate_network
from airline_ops.planning.plan import Plan, planned_cost
from airline_ops.planning.planners import (
    RobustSettings,
    check_plan,
    plan_integrated,
    plan_robust,
    plan_sequential,
)
from airline_ops.planning.rules import PlanningRules
from airline_ops.simulation.delays import DelayModel, EmpiricalPenalty
from airline_ops.simulation.scenarios import SCENARIOS, draw_events
from airline_ops.simulation.simulator import simulate, unroll

LOG = get_logger("experiments.details")
METRICS = ("total_cost", "operational_cost", "pax_delay_min", "missed_connection_pax", "cancelled")
VARIANTS = {
    "b0_w0.5": (0, 0.5),
    "b0_w1.0": (0, 1.0),
    "b15_w1.0": (15, 1.0),
    "b45_w2.0": (45, 2.0),
    "b45_w0.0": (45, 0.0),
}


def _model(results: Path, config: Config) -> DelayModel:
    calib_path = results / "calibration.json"
    if not calib_path.is_file():
        raise DataError(f"missing {calib_path}; run the experiment first")
    chosen = json.loads(calib_path.read_text(encoding="utf-8"))["chosen"]
    d = config.delays
    return DelayModel(d.mean_primary_min, float(chosen["p_delayed"]), float(chosen["sigma"]), d.block_noise_sd,
                      d.block_bias)  # fmt: skip


def _slack_profile(plan: Plan, network: Any, config: Config) -> dict[str, float]:
    s = connection_slack(plan, network, config)
    ac = np.array(s["aircraft_hub_turn_slack"])
    crew = np.array(s["crew_change_slack"]) if s["crew_change_slack"] else np.array([np.nan])
    return {
        "hub_turns": float(ac.size),
        "hub_turn_slack_median": float(np.median(ac)),
        "hub_turns_within_5_min": float(np.mean(ac <= 5.0)),
        "hub_turns_within_15_min": float(np.mean(ac <= 15.0)),
        "crew_changes": float(np.sum(np.isfinite(crew))),
        "crew_change_slack_median": float(np.nanmedian(crew)),
    }


def _learned_penalties(
    network: Any, config: Config, model: DelayModel, base_seed: int
) -> tuple[EmpiricalPenalty, EmpiricalPenalty]:
    """The point and distributional penalties of the main run's learning stage, rebuilt."""
    rules = PlanningRules.from_config(config, network)
    plan = plan_sequential(network, config)
    unrolled = unroll(network, plan, config.simulation.horizon_days, rules)
    n = network.n_flights
    samples: list[list[float]] = [[] for _ in range(n)]
    hist_seed = base_seed + HISTORY_SEED_OFFSET
    for rep in range(config.experiment.history_replications):
        ev = draw_events(SCENARIOS[0], network, np.random.default_rng([hist_seed, rep, 0]))
        res = simulate(network, plan, unrolled, config, model, ev, hist_seed, rep)
        for i in np.nonzero(res.operated)[0]:
            samples[int(i) % n].append(float(res.arr_delay[i]))
    arrays = [np.array(x) for x in samples]
    return EmpiricalPenalty(arrays, point=True), EmpiricalPenalty(arrays, point=False)


def run_details(config: Config, results: Path, contract: Path, cache_dir: Path) -> Path:
    """Re-plan the selected variants, evaluate them paired with the integrated plan, write details.json.

    Raises:
        DataError: When the main results are missing.

    """
    exp = config.experiment
    reps, base_seed = exp.replications, exp.seed
    records = load_records(config, contract, cache_dir)
    network = generate_network(config.network, records, exp.network_seeds[0])
    model = _model(results, config)
    excess = penalty_function(network, model, config.robust.penalty_samples, base_seed + PLANNING_SEED_OFFSET)
    plans: dict[str, Plan] = {"integrated": plan_integrated(network, config)}
    for name, (b, w) in VARIANTS.items():
        plans[name] = plan_robust(network, config, RobustSettings(b, w), excess, name=name)
    point, dist = _learned_penalties(network, config, model, base_seed)
    settings = RobustSettings(0, config.robust.penalty_weight)
    plans["point_forecast"] = plan_robust(network, config, settings, point.expected_excess_flight, name="point")
    plans["distribution"] = plan_robust(network, config, settings, dist.expected_excess_flight, name="dist")
    scen = (SCENARIOS[0], SCENARIOS[5])
    evals = {name: evaluate_plan(network, p, config, model, scen, reps, base_seed) for name, p in plans.items()}
    out: dict[str, Any] = {"replications": reps, "model": asdict(model), "plans": {}, "paired_vs_integrated": {}}
    for name, p in plans.items():
        out["plans"][name] = {
            "planned_cost_day": planned_cost(p, network, config.costs)["total"],
            "violations": len(check_plan(p, network, config)),
            "slack": _slack_profile(p, network, config),
            **{sc: summarise(rows) for sc, rows in evals[name].items()},
        }
        if name != "integrated":
            out["paired_vs_integrated"][name] = {
                sc: {k: paired_difference([r[k] for r in evals[name][sc]], [r[k] for r in evals["integrated"][sc]])
                     for k in METRICS}
                for sc in evals[name]
            }  # fmt: skip
    # Buffers alone under the calm and the stressed climates, paired with no buffer.
    climates: dict[str, Any] = {}
    for climate in (min(exp.delay_climates), max(exp.delay_climates)):
        m = replace(model, mean_primary_min=climate)
        base = evaluate_plan(network, plans["integrated"], config, m, (SCENARIOS[0],), reps, base_seed)["S0"]
        buf = evaluate_plan(network, plans["b45_w0.0"], config, m, (SCENARIOS[0],), reps, base_seed)["S0"]
        climates[f"{climate:g}"] = {
            k: paired_difference([r[k] for r in buf], [r[k] for r in base]) for k in ("total_cost", "operational_cost")
        }
        LOG.info("climate pair done", extra={"fields": {"climate": climate}})
    out["buffer45_minus_buffer0_by_climate"] = climates
    path = results / "details.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    LOG.info("details written", extra={"fields": {"path": str(path)}})
    return path
