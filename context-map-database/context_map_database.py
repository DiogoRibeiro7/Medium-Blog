"""A category-theoretic context map for LLM agents, built as a database and tested.

Companion code for the Medium article on reading the "Transportive AI" proposal (contexts
as a category, meanings as fibres of a functor, tokens attached by a profunctor, the whole
assembled by the Grothendieck construction) as the functorial data model of databases.
Prints every number in the article and draws every figure. Calls no language model.

Sections
  1  The deploy schema as a finitely presented category: morphisms, hom-sets, confluence.
  2  The instance as a functor; the category of elements as one flat table.
  3  The conversation: a context-blind rule, the context-aware query, and the schema with
     one extra arrow.
  4  The three checks of the proposal as queries: a declared law, a commuting square,
     a loop.
  5  The same checks in SQL on random instances, and profunctor composition as a join
     followed by connected components.
  6  Learned transports: loop drift, its formula, and the loop check as a statistical test.
  7  Grounding: what happens when the agent does not know which context it is in.

Run:  python context_map_database.py > results/output.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import agreement
import deploy_example as D
import drift
import figures
import grounding
import sql
from cset import Instance

HERE = Path(__file__).resolve().parent
SEED = 20261122


def rounded(obj, sig: int = 10):
    """Floats to `sig` significant digits, so the JSON does not depend on the last bits that
    BLAS libraries are free to differ in from one machine to another."""
    if isinstance(obj, float):
        return float(f"{obj:.{sig}g}")
    if isinstance(obj, dict):
        return {k: rounded(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v, sig) for v in obj]
    return obj


def header(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def main() -> None:
    # ------------------------------------------------------------------------------
    header("1  The deploy schema as a finitely presented category")
    for label, cp in (("first schema", False), ("with w: Cloud -> Production", True)):
        S = D.schema(cp)
        M = S.morphisms()
        print(f"\n{label}: {len(S.objects)} objects, {len(S.arrows)} generating arrows, "
              f"{len(S.equations)} equation(s); rewriting confluent: {S.is_confluent()}")
        print(f"  morphisms in all: {sum(len(v) for v in M.values())}")
        for (a, b), ps in M.items():
            if ps:
                print(f"  Hom({a}, {b}) = {{{', '.join(str(p) for p in ps)}}}")
        print(f"  Hom(Staging, Production) empty: {not M[('Staging', 'Production')]}")
        con = sql.schema_connection(S)
        print(f"  SQL recursive query, Staging reaches Production: {sql.reachable(con, 'Staging', 'Production')}")

    # ------------------------------------------------------------------------------
    header("2  The instance as a functor; the category of elements as one table")
    I = D.instance()
    S = I.schema
    print("rows per table:", {c: len(I.rows[c]) for c in S.objects}, "total", sum(len(t) for t in I.rows.values()))
    print("typing errors:", len(I.typing_errors()))
    print("path-equation violations:", I.all_violations())
    print("is a functor:", I.is_functor())
    print("\nThe path equation as SQL:")
    print("  " + sql.equation_query(S, *S.equations[0]))
    con = sql.load(I)
    print("SQL violations:", sql.equation_violations(con, S))
    loop = S.path("Incident", "i", "u")
    print("\nThe loop Incident -> Production -> Incident (i;u), row by row:")
    for r in I.rows["Incident"]:
        print(f"  {r:12s} -> {I.apply(loop, r):12s}  ({I.rows['Incident'][I.apply(loop, r)]['meaning']})")
    moved = [r for r in I.rows["Incident"] if I.apply(loop, r) != r]
    print(f"  rows that do not come back as themselves: {len(moved)} of {len(I.rows['Incident'])}: {moved}")
    back = S.path("Production", "u", "i")
    print(f"The loop Production -> Incident -> Production (u;i) moves "
          f"{sum(I.apply(back, r) != r for r in I.rows['Production'])} rows.")
    flat = I.elements()
    print(f"\nflat table: {len(flat)} rows, columns {list(flat[0].keys())}")
    print("round trip flat -> instance identical:", Instance.from_elements(S, flat, sql.ATTRS).same_as(I))
    fcon = sql.load_flat(S, flat)
    print("flat-table conditions broken (python, sql):", len(I.fibration_errors_flat(flat)),
          len(sql.flat_target_errors(fcon, S)))
    print("\n" + sql.flat_ddl(S))
    tests = [
        ("a Staging row that also fills the Incident column",
         {"eid": "bad1", "context": "Staging", "token": "approve", "fk_s": "c_approve", "fk_i": "p_note"}),
        ("a Staging row with no s column", {"eid": "bad2", "context": "Staging", "token": "approve"}),
        ("a Staging row whose s column points at a Production row",
         {"eid": "bad3", "context": "Staging", "token": "approve", "fk_s": "p_note"}),
    ]
    for label, row in tests:
        print(f"  insert {label}: {sql.try_insert_flat(fcon, S, row)}")
    print("  cross-row query after the inserts:", sql.flat_target_errors(fcon, S))

    # ------------------------------------------------------------------------------
    header("3  The conversation")
    for t, ctx, text, fact in D.CONVERSATION:
        print(f"  turn {t:2d}  {ctx:10s}  {text:40s}  {'records ' + fact if fact else ''}")
    for label, cp in (("first schema", None), ("with w, sending the generic approval to 'release authorised'", "leak"),
                      ("with w, sending it to the note that grants nothing", "safe")):
        inst = D.instance(cp)
        print(f"\n{label}: functor {inst.is_functor()}")
        for r in D.run_conversation(inst):
            ev = "; ".join(f"{f} via {p} -> {row}" for f, p, row in r["evidence"]) or "none"
            print(f"  turn {r['turn']:2d} in {r['context']:10s} asks {r['asks']:8s}  context-aware: "
                  f"{'carry out' if r['aware'] else 'refuse':9s}  context-blind: "
                  f"{'carry out' if r['blind'] else 'refuse':9s}  evidence: {ev}")
            if r["turn"] == 8:
                print(f"           state at turn 8: {r['state']}")

    # ------------------------------------------------------------------------------
    header("4  The three checks as queries")
    print("Declared law: a release authorisation in Production can only be the image of one that")
    print("was already there (rows carried to one along a morphism without being such an image).")
    for label, cp in (("first schema", None), ("with w, leak", "leak"), ("with w, safe", "safe")):
        inst = D.instance(cp)
        py = D.law_violations(inst)
        sq = sql.law_violations(sql.load(inst), inst.schema, "Production", "release")
        print(f"  {label}: {len(py)} violating rows (python) / {len(sq)} (sql), same: {py == sq}")
        for v in py:
            print(f"     {v}")
    print("\nCommuting squares: an update is natural when it commutes with every foreign key.")
    for label, alpha in (("revoke every approval", D.revoke_everywhere()),
                         ("revoke approvals inside the incident only", D.revoke_in_incident_only())):
        py = sorted(I.naturality_violations(alpha))
        sq = sql.naturality_violations(sql.load(I), S, alpha)
        print(f"  {label}: {len(py)} failing squares, sql agrees: {py == sq}")
        for v in py:
            print(f"     arrow {v[0]}, row {v[1]}: move then update -> {v[2]}; update then move -> {v[3]}")
    print("\nLoops: the path equation u;i = id holds on every Production row (section 2); the loop i;u")
    print("legitimately moves the specific hotfix approvals to the generic note.")

    # ------------------------------------------------------------------------------
    header("5  The same checks in SQL on random instances")
    ag = agreement.run_instances(SEED, n=400)
    for k, v in ag.items():
        print(f"  {k}: {v}")
    rc = agreement.run_reachability(7, n=300)
    print("random schemas, reachability (python BFS vs recursive SQL):", rc)
    co = agreement.run_coends(11, n=200)
    print("profunctor composition over a diamond-shaped category (coend by union-find vs SQL):")
    for k, v in co.items():
        print(f"  {k}: {v}")
    print(f"  join overcounts the composite by a factor of {co['join_total'] / co['coend_total']:.2f}")

    # ------------------------------------------------------------------------------
    header("6  Learned transports")
    eps = float(np.finfo(float).eps)
    print(f"written maps: max |B A - I| within ten machine epsilons ({10 * eps:.1e}): "
          f"{drift.strict_drift(SEED) < 10 * eps}")  # the exact residual depends on the BLAS
    chk = drift.sampler_check(SEED)
    print("exact sampler vs brute-force least squares:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in chk.items()})
    curve = drift.drift_curve(SEED)
    print(f"\n{'d':>4} {'m':>4} {'n':>6} {'reps':>5} {'RMS sim':>9} {'formula':>9} {'ratio':>6} {'5%':>8} {'95%':>8} {'min':>8}")
    worst = 0.0
    for r in curve:
        ratio = r["rms_sim"] / r["rms_analytic"]
        worst = max(worst, abs(ratio - 1))
        print(f"{r['d']:4d} {r['m']:4d} {r['n']:6d} {r['reps']:5d} {r['rms_sim']:9.5f} {r['rms_analytic']:9.5f} "
              f"{ratio:6.3f} {r['q05']:8.5f} {r['q95']:8.5f} {r['min']:8.5f}")
    print(f"largest relative gap between simulation and formula: {pct(worst, 2)}")
    pw = drift.power_study(SEED)
    print(f"\nloop check as a test: d = {pw['d']}, m = {pw['m']}, sigma = {pw['sigma']}, alpha = {pw['alpha']}")
    print("violation multiples of the null RMS drift:", pw["multiples"])
    for b in pw["by_n"]:
        print(f"  n = {b['n']:6d}: null RMS {b['null_rms']:.5f} (formula {b['analytic_rms']:.5f}), "
              f"threshold {b['threshold']:.5f}, size {b['size']:.3f}, null CV {pct(b['null_cv'], 2)}")
        print(f"     power: {[round(p, 3) for p in b['power']]}")
        print(f"     80% power at {b['k80']:.3f} x null drift = {b['delta80']:.5f}")
        print(f"     noise overstated by 10%: power {[round(p, 3) for p in b['power_noise_overestimated_10pct']]}; "
              f"80% power at {b['k80_noise_overestimated_10pct']:.3f} x null drift")
        print(f"     false alarms when the noise is misjudged: {b['size_misjudged']}")

    # ------------------------------------------------------------------------------
    header("7  Grounding")
    gs = grounding.study(SEED)
    print(f"cost rule threshold t = {gs['t_cost']:.2f} (unpermitted action costs 9 refusals)")
    for L in gs["levels"]:
        print(f"\ncontext accuracy {pct(L['accuracy'], 2)} (scale {L['scale']:.3f}); mean top posterior "
              f"{L['mean_max_posterior']:.3f}, overconfident {L['mean_max_posterior_overconfident']:.3f}")
        for name in ("argmax", "bayes_calibrated", "bayes_overconfident"):
            r = L[name]
            print(f"  {name:20s} unpermitted carried out {pct(r['unauthorised'], 2):>7s}   permitted refused {pct(r['refused'], 2):>7s}")
        print(f"  identity from the confusion matrix: {pct(L['identity_unauthorised'], 2)}")
        print("  sources (argmax):", {k: pct(v) for k, v in L["argmax_sources"].items()})
    audit = grounding.database_audit()
    print("\ndatabase audit of argmax decisions:", audit)

    # ------------------------------------------------------------------------------
    header("Figures")
    for p in (figures.figure_schema(), figures.figure_drift(curve, pw), figures.figure_grounding(gs)):
        print("  wrote", Path(p).relative_to(HERE))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "summary.json").write_text(json.dumps(rounded(
        {"agreement": ag, "reachability": rc, "coend": co,
         "drift": {"curve": curve, "power": pw},
         "grounding": {"t_cost": gs["t_cost"], "levels": [{k: v for k, v in L.items() if k not in ("curve", "thresholds")} for L in gs["levels"]]},
         "audit": audit}), indent=1) + "\n")


if __name__ == "__main__":
    main()
