"""Small, fast checks of the claims the article rests on."""

import numpy as np

import agreement
import deploy_example as D
import drift
import grounding
import sql
from cset import Instance


def test_schemas_are_finite_and_confluent():
    for cp in (False, True):
        S = D.schema(cp)
        assert S.is_confluent()
        assert sum(len(v) for v in S.morphisms().values()) == (18 if cp else 10)


def test_one_arrow_opens_a_path():
    assert D.schema(False).hom("Staging", "Production") == []
    assert [str(p) for p in D.schema(True).hom("Staging", "Production")] == ["s;w"]


def test_instance_is_a_functor_and_flattens():
    I = D.instance()
    assert I.is_functor()
    flat = I.elements()
    assert Instance.from_elements(I.schema, flat, sql.ATTRS).same_as(I)
    assert not I.fibration_errors_flat(flat)


def test_loop_moves_only_the_specific_hotfixes():
    I = D.instance()
    loop = I.schema.path("Incident", "i", "u")
    assert [r for r in I.rows["Incident"] if I.apply(loop, r) != r] == ["i_hotfix_42", "i_hotfix_43"]


def test_conversation_decisions():
    aware = {cp: [r["aware"] for r in D.run_conversation(D.instance(cp))] for cp in (None, "leak", "safe")}
    assert aware[None] == [True, False, False]
    assert aware["leak"] == [True, True, False]
    assert aware["safe"] == [True, False, False]
    assert all(r["blind"] for r in D.run_conversation(D.instance()))


def test_law_and_squares_agree_with_sql():
    for cp in (None, "leak", "safe"):
        inst = D.instance(cp)
        assert D.law_violations(inst) == sql.law_violations(sql.load(inst), inst.schema, "Production", "release")
    I = D.instance()
    assert I.naturality_violations(D.revoke_everywhere()) == []
    assert len(I.naturality_violations(D.revoke_in_incident_only())) == 6


def test_random_agreement():
    out = agreement.run_instances(3, n=30)
    assert out["equation_mismatches"] == out["naturality_mismatches"] == out["law_mismatches"] == 0
    assert out["flat_roundtrip_failures"] == out["flat_condition_failures"] == 0
    assert agreement.run_reachability(3, n=30)["mismatches"] == 0
    co = agreement.run_coends(3, n=20)
    assert co["python_sql_mismatches"] == 0 and co["discrete_middle_equal"] == co["discrete_trials"]


def test_drift_formula():
    rng = np.random.default_rng(0)
    d, m, n, s = 16, 32, 600, 0.1
    A = drift.orthonormal(rng, m, d)
    vals = np.array([drift.one_loop(rng, A, A.T, n, s) for _ in range(800)])
    assert abs(np.sqrt(np.mean(vals ** 2)) / drift.analytic_rms(s, d, m, n) - 1) < 0.03


def test_grounding_identity():
    s = grounding.scale_for_accuracy(0.9, n=50_000)
    c, post, a, legit = grounding.draw(np.random.default_rng(1), 100_000, s)
    r = grounding.rates(c, post, a, legit, "argmax")
    acc = (post.argmax(1) == c).mean()
    # an unpermitted request is carried out only if the context is misread as the one context
    # that permits it: about one third of the misreadings
    assert abs(r["unauthorised"] - (1 - acc) / 3) < 0.01
