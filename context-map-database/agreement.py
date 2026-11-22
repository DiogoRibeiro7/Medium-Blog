"""Random instances, checked twice: once by the category-theoretic code, once by SQL.

Every check in the article is computed both ways on many random instances of the deploy
schemas, some of them corrupted on purpose, and on random profunctors over a small free
category. The two must return the same rows every time.
"""

from __future__ import annotations

import numpy as np

import sql
from cset import CoProfunctor, Instance, Profunctor
from deploy_example import CONTEXTS, law_violations, schema
from fincat import Arrow, Path, Presentation

GRANTS = (None, None, "release", "hotfix", "test_deploy")


def random_instance(rng: np.random.Generator, cloud_push: bool, corrupt: float) -> Instance:
    S = schema(cloud_push)
    n = {c: int(rng.integers(2, 10)) for c in CONTEXTS}
    if cloud_push:
        n["Production"] = max(n["Production"], n["Cloud"])
    n["Incident"] = max(n["Incident"], n["Production"])
    inst = Instance(S)
    ids = {c: [f"{c[0].lower()}{k}" for k in range(n[c])] for c in CONTEXTS}
    for c in CONTEXTS:
        for rid in ids[c]:
            inst.add(c, rid, token=str(rng.choice(["deploy", "approve"])), meaning=rid,
                     grants=GRANTS[int(rng.integers(len(GRANTS)))])

    def pick(c):
        return ids[c][int(rng.integers(len(ids[c])))]

    for x in ids["Staging"]:
        inst.link("s", x, pick("Cloud"))
    for x in ids["Production"]:
        inst.link("p", x, pick("Cloud"))
    u_img = rng.permutation(ids["Incident"])[: n["Production"]]
    for x, y in zip(ids["Production"], u_img):
        inst.link("u", x, str(y))
    for y in ids["Incident"]:
        inst.link("i", y, pick("Production"))
    for x, y in zip(ids["Production"], u_img):
        inst.link("i", str(y), x)  # u;i = id
    if cloud_push:
        w_img = rng.permutation(ids["Production"])[: n["Cloud"]]
        for c, x in zip(ids["Cloud"], w_img):
            inst.link("w", c, str(x))
            inst.link("p", str(x), c)  # w;p = id
    for a in S.arrows:
        for x in list(inst.fk[a.name]):
            if rng.random() < corrupt:
                inst.link(a.name, x, pick(a.tgt))
    return inst


def random_update(rng: np.random.Generator, inst: Instance) -> dict[str, dict[str, str]]:
    alpha = {}
    for c, rows in inst.rows.items():
        ids = list(rows)
        alpha[c] = {x: ids[int(rng.integers(len(ids)))] for x in ids if rng.random() < 0.3}
    return alpha


def run_instances(seed: int = 20261122, n: int = 400) -> dict:
    rng = np.random.default_rng(seed)
    out = {"instances": 0, "functors": 0, "equation_rows_flagged": 0, "equation_mismatches": 0,
           "naturality_squares_flagged": 0, "naturality_mismatches": 0,
           "law_rows_flagged": 0, "law_mismatches": 0, "flat_roundtrip_failures": 0,
           "flat_condition_failures": 0}
    for cloud_push in (False, True):
        for k in range(n):
            corrupt = (0.0, 0.05, 0.2)[k % 3]
            inst = random_instance(rng, cloud_push, corrupt)
            S = inst.schema
            out["instances"] += 1
            out["functors"] += inst.is_functor()
            con = sql.load(inst)
            py_eq = {k2: v for k2, v in inst.all_violations().items()}
            sq_eq = sql.equation_violations(con, S)
            out["equation_rows_flagged"] += sum(len(v) for v in py_eq.values())
            out["equation_mismatches"] += py_eq != sq_eq
            alpha = random_update(rng, inst)
            py_n = sorted(inst.naturality_violations(alpha))
            sq_n = sql.naturality_violations(con, S, alpha)
            out["naturality_squares_flagged"] += len(py_n)
            out["naturality_mismatches"] += py_n != sq_n
            if inst.is_functor():
                py_l = law_violations(inst)
                sq_l = sql.law_violations(con, S, "Production", "release")
                out["law_rows_flagged"] += len(py_l)
                out["law_mismatches"] += py_l != sq_l
            flat = inst.elements()
            back = Instance.from_elements(S, flat, ("token", "meaning", "grants"))
            out["flat_roundtrip_failures"] += not back.same_as(inst)
            fcon = sql.load_flat(S, flat)
            out["flat_condition_failures"] += bool(inst.fibration_errors_flat(flat)) or bool(
                sql.flat_target_errors(fcon, S))
    return out


def run_reachability(seed: int = 7, n: int = 300) -> dict:
    rng = np.random.default_rng(seed)
    mism, pairs, reach = 0, 0, 0
    for _ in range(n):
        k = int(rng.integers(3, 8))
        objs = tuple(f"o{j}" for j in range(k))
        arrows = []
        for j in range(int(rng.integers(1, 2 * k))):
            a, b = rng.choice(k, 2)
            arrows.append(Arrow(f"g{j}", objs[a], objs[b]))
        S = Presentation(objs, tuple(arrows))
        con = sql.schema_connection(S)
        for a in objs:
            for b in objs:
                pairs += 1
                r1, r2 = S.reachable(a, b), sql.reachable(con, a, b)
                reach += r1
                mism += r1 != r2
    return {"schemas": n, "pairs": pairs, "reachable": reach, "mismatches": mism}


def diamond() -> Presentation:
    """A free category with two different paths b0 -> b3."""
    objs = ("b0", "b1", "b2", "b3")
    arrows = (Arrow("f1", "b0", "b1"), Arrow("f2", "b0", "b2"), Arrow("g1", "b1", "b3"), Arrow("g2", "b2", "b3"))
    return Presentation(objs, arrows)


def random_profunctors(rng: np.random.Generator, B: Presentation, A=("a0", "a1"), C=("c0", "c1")):
    prow, qrow = {}, {}
    for b in B.objects:
        for a in A:
            for k in range(int(rng.integers(1, 4))):
                prow[f"p_{b}_{a}_{k}"] = (a, b)
        for c in C:
            for k in range(int(rng.integers(1, 4))):
                qrow[f"q_{b}_{c}_{k}"] = (b, c)
    pact, qact = {}, {}
    for g in B.arrows:
        pact[g.name] = {}
        for x, (a, b) in prow.items():
            if b == g.src:
                opts = [y for y, (a2, b2) in prow.items() if b2 == g.tgt and a2 == a]
                pact[g.name][x] = opts[int(rng.integers(len(opts)))]
        qact[g.name] = {}
        for y, (b, c) in qrow.items():
            if b == g.tgt:
                opts = [z for z, (b2, c2) in qrow.items() if b2 == g.src and c2 == c]
                qact[g.name][y] = opts[int(rng.integers(len(opts)))]
    return Profunctor(B, prow, pact), CoProfunctor(B, qrow, qact)


def run_coends(seed: int = 11, n: int = 200) -> dict:
    from cset import coend_compose, join_count
    rng = np.random.default_rng(seed)
    B = diamond()
    mism, cells, overcount_cells, join_total, coend_total = 0, 0, 0, 0, 0
    for _ in range(n):
        P, Q = random_profunctors(rng, B)
        py = coend_compose(P, Q)
        sq = sql.coend_sql(P, Q)
        jn = join_count(P, Q)
        mism += py != sq
        for key in jn:
            cells += 1
            overcount_cells += jn[key] != py[key]
            join_total += jn[key]
            coend_total += py[key]
    # With no arrows in the middle the coend is the join.
    discrete = Presentation(B.objects, ())
    disc_equal = 0
    for _ in range(50):
        P, Q = random_profunctors(rng, discrete)
        disc_equal += coend_compose(P, Q) == join_count(P, Q) == sql.coend_sql(P, Q)
    return {"pairs_of_profunctors": n, "python_sql_mismatches": mism, "cells": cells,
            "cells_where_join_overcounts": overcount_cells, "join_total": join_total,
            "coend_total": coend_total, "discrete_middle_equal": disc_equal, "discrete_trials": 50}
