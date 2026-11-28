"""Plain-text summary of the results: every number the article quotes, and the hypothesis checks.

The summary reads only the JSON files written by the runner, so it can be regenerated from a
results folder without rerunning the experiment. Costs are per day unless the line says
otherwise; intervals are 95% intervals over the replications.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from airline_ops.config import load_config
from airline_ops.errors import DataError
from airline_ops.logging_setup import get_logger

LOG = get_logger("experiments.summary")
FOLDER = Path(__file__).resolve().parents[3]
PLANS = ("greedy", "sequential", "integrated", "robust")


def _load(results: Path, name: str) -> Any:
    path = results / name
    if not path.is_file():
        raise DataError(f"missing results file {path}; run the experiment first")
    return json.loads(path.read_text(encoding="utf-8"))


def _ci(s: dict[str, float], scale: float = 1.0, fmt: str = ",.0f") -> str:
    return f"{s['mean'] / scale:{fmt}} [{s['lo'] / scale:{fmt}}, {s['hi'] / scale:{fmt}}]"


def _excludes_zero(s: dict[str, float]) -> bool:
    return bool(s["lo"] > 0 or s["hi"] < 0)


def _q(values: list[float]) -> str:
    if not values:
        return "n/a"
    q = np.quantile(np.array(values), [0.05, 0.5, 0.95])
    return f"p5 {q[0]:.0f}, median {q[1]:.0f}, p95 {q[2]:.0f}"


class _Writer:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def h(self, text: str) -> None:
        self.lines += ["", text, "-" * len(text)]

    def __call__(self, text: str = "") -> None:
        self.lines.append(text)


def _network(w: _Writer, net: dict[str, Any], calib: dict[str, Any]) -> None:
    w.h("Network")
    w(f"hub {net.get('hub', 'MUC')}, {net['spokes']} spokes, {net['flights_per_day']} flights per day")
    w(f"fleet {net['fleet']} (minimum fleet {net['minimum_fleet']}), {net['nightstop_spokes']} night-stop spokes")
    w(f"passengers per day {net['passengers_per_day']:,}, connecting {net['connecting_passengers_per_day']:,}")
    w(f"mean block time {net['mean_block_min']:.0f} min")
    w.h("Delay model calibration")
    t, c = calib["targets"], calib["chosen"]
    w(f"targets: D15 {t['d15']:.3f}, mean primary {t['mean_primary_min']} min, reactionary (validation) "
      f"{t['reactionary_share_validation']:.2f}")  # fmt: skip
    w(f"chosen: p_delayed {c['p']}, sigma {c['sigma']}, simulated D15 {c['d15']:.3f}, "
      f"mean departure delay {c['mean_dep_delay']:.1f} min, "
      f"reactionary share {c['reactionary_share']:.3f}")  # fmt: skip
    w(f"grid points {len(calib['grid'])}, replications per point {calib['replications']}")


def _plans(w: _Writer, plans: dict[str, Any]) -> None:
    w.h("Plans")
    for name in PLANS:
        p = plans[name]
        c = p["planned_cost"]
        notes = p["notes"]
        w(f"{name}: planned €{c['total']:,.0f}/day (ownership {c['ownership']:,.0f}, crew pay {c['crew_pay']:,.0f}, "
          f"hotel {c['hotel']:,.0f}, deadhead {c['deadhead']:,.0f}); aircraft {c['aircraft']:.0f}")  # fmt: skip
        w(f"  pairings {p['pairings']}, duties {p['duties']}, mean duty {p['mean_duty_h']:.2f} h, "
          f"mean FDP {p['mean_fdp_h']:.2f} h, mean pairing span {p['mean_pairing_span_h']:.1f} h")  # fmt: skip
        w(f"  block hours per aircraft-day {p['rotations']['block_hours_per_aircraft_day']:.2f}, "
          f"rotations {p['rotations']['rotations']:.0f}")  # fmt: skip
        w(f"  FDP margin to Table 2 (min): {_q(p['fdp_margin_min'])}")
        sl = p["slack"]
        w(f"  hub aircraft-turn slack above minimum (min): {_q(sl['aircraft_hub_turn_slack'])}")
        w(f"  crew aircraft-change slack above minimum (min): {_q(sl['crew_change_slack'])}")
        shown = f": {'; '.join(p['violations'][:6])}" if p["violations"] else ""
        w(f"  violations {len(p['violations'])}{shown}")
        w(f"  solve seconds {p.get('solve_seconds', float('nan')):.1f}, objective {p['objective']}, "
          f"LP bound {p['lower_bound']}")  # fmt: skip
        keys = ("cg_iterations", "columns_generated", "pairings_in_space", "duties", "gap_to_lp_bound",
                "short_connections", "routing_variables", "buffer_min", "penalty_weight")  # fmt: skip
        w("  " + ", ".join(f"{k} {notes[k]}" for k in keys if k in notes))


def _evaluation(w: _Writer, ev: dict[str, Any], horizon: int) -> None:
    w.h(f"Evaluation ({ev['replications']} replications, per day, mean [95% interval])")
    for sc, label in ev["scenarios"].items():
        w(f"{sc} {label}")
        for name in PLANS:
            s = ev["plans"][name][sc]
            w(f"  {name:<11} total {_ci(s['total_cost'], horizon)} | disruption {_ci(s['operational_cost'], horizon)}"
              f" | pax delay min {_ci(s['pax_delay_min'], horizon)}"
              f" | missed pax {_ci(s['missed_connection_pax'], horizon, '.1f')}"
              f" | cancelled {_ci(s['cancelled'], horizon, '.2f')}")  # fmt: skip
            w(f"  {'':<11} D15 {s['d15']['mean']:.3f} A15 {s['a15']['mean']:.3f} mean dep delay "
              f"{s['mean_dep_delay']['mean']:.1f} reactionary {s['reactionary_share']['mean']:.3f} "
              f"recovery {s['recovery_min']['mean']:.0f} min footprint {s['footprint']['mean']:.1f} "
              f"EU261 €{s['eu261']['mean'] / horizon:,.0f} swaps {s['swaps']['mean']:.2f} "
              f"standby {s['standby_calls']['mean']:.2f} discretion {s['discretion_uses']['mean']:.2f} "
              f"ferries {s['ferries']['mean']:.2f} "
              f"EUROCONTROL-ref €{s['eurocontrol_reference']['mean'] / horizon:,.0f} "
              f"aircraft util {s['aircraft_utilization_h']['mean']:.2f} h "
              f"crew util {s['crew_utilization']['mean']:.3f}")  # fmt: skip
    w.h("Paired differences against sequential (per day)")
    for name, by_sc in ev["paired_vs_sequential"].items():
        for sc, m in by_sc.items():
            parts = [f"{k} {_ci(v, horizon, ',.1f')}{' *' if _excludes_zero(v) else ''}" for k, v in m.items()]
            w(f"{name} − sequential {sc}: " + "; ".join(parts))
    if "paired_robust_vs_integrated" in ev:
        w.h("Paired differences robust − integrated (per day)")
        for sc, m in ev["paired_robust_vs_integrated"].items():
            parts = [f"{k} {_ci(v, horizon, ',.1f')}{' *' if _excludes_zero(v) else ''}" for k, v in m.items()]
            w(f"{sc}: " + "; ".join(parts))
    w("(* the 95% interval excludes zero)")


def _hypotheses(
    w: _Writer, plans: dict[str, Any], ev: dict[str, Any], climates: dict[str, Any] | None,
    learning: dict[str, Any] | None, horizon: int,
) -> None:  # fmt: skip
    w.h("Hypotheses (PLAN.md)")
    seq, integ = plans["sequential"]["planned_cost"]["total"], plans["integrated"]["planned_cost"]["total"]
    saving = (seq - integ) / seq
    w(f"H1: integrated planned €{integ:,.0f} vs sequential €{seq:,.0f}; saving {saving:.3%}; "
      f"holds: {integ <= seq + 1e-6 and saving < 0.02}")  # fmt: skip
    d = ev["paired_vs_sequential"]["integrated"]["S0"]["operational_cost"]
    w(f"H2: integrated − sequential disruption cost S0 {_ci(d, horizon, ',.1f')} per day; "
      f"integrated not lower: {not d['hi'] < 0}")  # fmt: skip
    r_s = ev["paired_vs_sequential"]["robust"]["S0"]["total_cost"]
    r_i = ev.get("paired_robust_vs_integrated", {}).get("S0", {}).get("total_cost")
    ok = r_s["hi"] < 0 and (r_i is not None and r_i["hi"] < 0)
    w(f"H3: robust − sequential total S0 {_ci(r_s, horizon, ',.1f')}; robust − integrated total S0 "
      f"{_ci(r_i, horizon, ',.1f') if r_i else 'n/a'}; holds: {ok}")  # fmt: skip
    if climates:
        rows = [r for r in climates["rows"] if "buffer" in r]
        best: dict[float, tuple[int, float]] = {}
        for r in rows:
            v = r["S0"]["total_cost"]["mean"] / horizon
            if r["climate"] not in best or v < best[r["climate"]][1]:
                best[r["climate"]] = (r["buffer"], v)
        text = ", ".join(f"{c:g} min → {b} min" for c, (b, _v) in sorted(best.items()))
        lo_c, hi_c = min(best), max(best)
        w(f"H4: cost-minimising buffer by climate: {text}; holds: {best[hi_c][0] > best[lo_c][0]}")
        for c in sorted(best):
            cells = sorted((r for r in rows if r["climate"] == c), key=lambda r: r["buffer"])
            w(f"    climate {c:g}: " + ", ".join(
                f"b{r['buffer']} €{r['S0']['total_cost']['mean'] / horizon:,.0f}" for r in cells))  # fmt: skip
        for r in climates["rows"]:
            if "plan" in r:
                w(f"    climate {r['climate']:g} {r['plan']}: total {_ci(r['S0']['total_cost'], horizon)}, "
                  f"disruption {_ci(r['S0']['operational_cost'], horizon)}")  # fmt: skip
    if learning:
        v = learning["variants"]
        w(f"H5: history replications {learning['history_replications']}, mean |forecast| "
          f"{learning['mean_abs_forecast']:.1f} min, share of flights with mean delay > 10 min "
          f"{learning['share_flights_mean_above_10']:.2f}")  # fmt: skip
        for name, rec in v.items():
            w(f"    {name}: planned €{rec['planned_cost_day']:,.0f}, total S0 {_ci(rec['S0']['total_cost'], horizon)}, "
              f"disruption S0 {_ci(rec['S0']['operational_cost'], horizon)}, total S5 "
              f"{_ci(rec['S5']['total_cost'], horizon)}, violations {rec['violations']}")  # fmt: skip
        if {"no_penalty", "point_forecast", "distribution"} <= set(v):
            none = v["no_penalty"]["S0"]["total_cost"]["mean"]
            pt = v["point_forecast"]["S0"]["total_cost"]["mean"]
            di = v["distribution"]["S0"]["total_cost"]["mean"]
            w(f"    point − none {(pt - none) / horizon:+,.0f}/day, "
              f"distribution − none {(di - none) / horizon:+,.0f}/day"
              " (unpaired difference of means; see the intervals above)")  # fmt: skip


def _breakeven(w: _Writer, ev: dict[str, Any], horizon: int) -> None:
    """Share of disrupted days at which the robust plan's extra cost on normal days pays for itself.

    Descriptive, added after the results: with D0 the mean robust-minus-reference total cost per day
    under baseline delays and Dk the same under scenario k, a year with a share p of k-days has an
    expected difference (1 - p) D0 + p Dk, which is zero at p = D0 / (D0 - Dk).
    """
    w.h("Break-even share of disrupted days (descriptive, added after the results)")
    refs = {"sequential": ev["paired_vs_sequential"]["robust"]}
    if "paired_robust_vs_integrated" in ev:
        refs["integrated"] = ev["paired_robust_vs_integrated"]
    for ref, diffs in refs.items():
        d0 = diffs["S0"]["total_cost"]["mean"] / horizon
        parts = []
        for sc, m in diffs.items():
            if sc == "S0":
                continue
            dk = m["total_cost"]["mean"] / horizon
            if d0 > 0 > dk:
                parts.append(f"{sc} {d0 / (d0 - dk):.1%}")
            elif d0 <= 0:
                parts.append(f"{sc} robust cheaper already at baseline")
            else:
                parts.append(f"{sc} never (robust not cheaper on that day)")
        w(f"robust vs {ref}: baseline difference {d0:+,.0f}/day; " + "; ".join(parts))


def _details(w: _Writer, det: dict[str, Any], horizon: int) -> None:
    w.h("Added after the results: paired re-evaluations (details.json, per day)")
    for name, rec in det["plans"].items():
        sl = rec["slack"]
        w(f"{name}: planned €{rec['planned_cost_day']:,.0f}, violations {rec['violations']}, hub turns "
          f"{sl['hub_turns']:.0f} (median slack {sl['hub_turn_slack_median']:.1f} min, within 5 min "
          f"{sl['hub_turns_within_5_min']:.1%}), crew changes {sl['crew_changes']:.0f} (median slack "
          f"{sl['crew_change_slack_median']:.1f} min), crew utilisation {rec['S0']['crew_utilization']['mean']:.3f}, "
          f"discretion/day {rec['S0']['discretion_uses']['mean'] / horizon:.2f}")  # fmt: skip
    for name, by_sc in det["paired_vs_integrated"].items():
        for sc, m in by_sc.items():
            parts = [f"{k} {_ci(v, horizon, ',.1f')}{' *' if _excludes_zero(v) else ''}" for k, v in m.items()]
            w(f"{name} − integrated {sc}: " + "; ".join(parts))
    for climate, m in det["buffer45_minus_buffer0_by_climate"].items():
        parts = [f"{k} {_ci(v, horizon, ',.1f')}" for k, v in m.items()]
        w(f"climate {climate}: buffer 45 − buffer 0 (both w = 0): " + "; ".join(parts))


def _sweep(w: _Writer, sweep: dict[str, Any], horizon: int) -> None:
    w.h("Buffer and penalty sweep (S0 and S5, per day)")
    for r in sweep["rows"]:
        if not r.get("feasible"):
            w(f"b{r['buffer']} w{r['weight']}: infeasible or invalid: {r.get('reason') or r.get('violations')}")
            continue
        w(f"b{r['buffer']:>2} w{r['weight']:<3} planned €{r['planned_cost_day']:,.0f} aircraft {r['aircraft']:.0f} "
          f"| S0 total {_ci(r['S0']['total_cost'], horizon)} disruption {_ci(r['S0']['operational_cost'], horizon)} "
          f"| S5 total {_ci(r['S5']['total_cost'], horizon)} "
          f"| solve {r['solve_seconds']:.1f}s gap {r['gap']}")  # fmt: skip
    w("Pareto front (planned cost vs S0 disruption cost): "
      + ", ".join(f"b{p['buffer']} w{p['weight']}" for p in sweep["pareto_front"]))  # fmt: skip
    for sc in ("S0", "S5"):
        feas = [r for r in sweep["rows"] if r.get("feasible")]
        best = min(feas, key=lambda r: r[sc]["total_cost"]["mean"])
        w(f"lowest expected total cost in {sc}: b{best['buffer']} w{best['weight']} "
          f"€{best[sc]['total_cost']['mean'] / horizon:,.0f}/day")  # fmt: skip


def _structural(w: _Writer, st: dict[str, Any], horizon: int) -> None:
    w.h("Structural replications (other network seeds)")
    for row in st["rows"]:
        if "error" in row:
            w(f"seed {row['seed']}: error {row['error']}")
            continue
        w(f"seed {row['seed']}: {row['flights']} flights, fleet {row['fleet']}")
        base = row["plans"]["sequential"]
        for name in PLANS:
            p = row["plans"][name]
            dt = (p["S0"]["total_cost"]["mean"] - base["S0"]["total_cost"]["mean"]) / horizon
            w(f"  {name:<11} planned €{p['planned_cost_day']:,.0f} violations {p['violations']} "
              f"S0 total €{p['S0']['total_cost']['mean'] / horizon:,.0f} (vs seq {dt:+,.0f}) "
              f"S0 disruption €{p['S0']['operational_cost']['mean'] / horizon:,.0f} "
              f"S1 disruption €{p['S1']['operational_cost']['mean'] / horizon:,.0f} "
              f"S5 disruption €{p['S5']['operational_cost']['mean'] / horizon:,.0f} "
              f"solve {p['solve_seconds']:.1f}s")  # fmt: skip


def write_summary(results: Path) -> Path:
    """Write ``summary.txt`` in the results folder and return its path.

    Raises:
        DataError: When a results file the summary needs is missing.

    """
    run = _load(results, "run.json")
    horizon = load_config(FOLDER / run["config"]).simulation.horizon_days
    w = _Writer()
    w(f"Results summary: config {run['config']}, run time {run['seconds'] / 60:.1f} min")
    w(f"delay model: {run['model']}")
    net, calib, plans, ev = (_load(results, n) for n in ("network.json", "calibration.json", "plans.json",
                                                         "evaluation.json"))  # fmt: skip
    _network(w, net, calib)
    _plans(w, plans)
    _evaluation(w, ev, horizon)
    names = ("sweep", "climates", "learning", "structural", "propagation", "examples", "details")
    optional = {n: _load(results, f"{n}.json") if (results / f"{n}.json").is_file() else None for n in names}
    _hypotheses(w, plans, ev, optional["climates"], optional["learning"], horizon)
    _breakeven(w, ev, horizon)
    if optional["sweep"]:
        _sweep(w, optional["sweep"], horizon)
    if optional["details"]:
        _details(w, optional["details"], horizon)
    if optional["structural"]:
        _structural(w, optional["structural"], horizon)
    if optional["propagation"]:
        prop = optional["propagation"]
        w.h("Propagation example (S1, one replication)")
        w(f"delayed inbound flights: {prop['delayed_flights']}")
        for name, p in prop["plans"].items():
            canc = sum(1 for n in p["nodes"] if n["cancelled"])
            w(
                f"{name}: {len(p['nodes'])} flights changed, {canc} cancelled, "
                f"{p['extra_delay_total']:,.0f} extra minutes"
            )
    if optional["examples"]:
        ex = optional["examples"]
        w.h(f"Pairing network through {ex['station']}")
        w(f"duties into {ex['station']} {len(ex['out_duties'])}, out of it {len(ex['back_duties'])}, "
          f"overnight links {len(ex['pairs'])}; network: {ex['n_duties_network']} duties, "
          f"{ex['n_pairings_network']} pairings")  # fmt: skip
        for name, chosen in ex["chosen"].items():
            w(f"{name}: pairings touching {ex['station']}: {chosen}")
    out = results / "summary.txt"
    out.write_text("\n".join(w.lines).lstrip() + "\n", encoding="utf-8")
    LOG.info("summary written", extra={"fields": {"path": str(out)}})
    return out
