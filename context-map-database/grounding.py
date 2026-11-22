"""What the map cannot check: which context the conversation is in.

The database answers "is this action permitted in context c?" exactly. The agent does not
know c; a context classifier gives it a posterior over the four contexts. Every error in
this simulation comes from that posterior, and the database reports nothing wrong, because
every decision it is asked to check is consistent with the context it was told.

Observation model: o = s * mu_c + N(0, I_2), with Incident placed closest to Production, so
that the dangerous confusion (an incident turn read as a production turn) is the commonest.
The posterior is exact Bayes, hence calibrated; an overconfident classifier sharpens it with
temperature 0.5. The scale s sets the classifier's accuracy.
"""

from __future__ import annotations

import sqlite3

import numpy as np

CONTEXTS = ("Cloud", "Staging", "Production", "Incident")
PRIOR = np.array([0.10, 0.30, 0.40, 0.20])
MU = np.array([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [1.5, 0.5]])
ACTION = ("change", "test_deploy", "release", "hotfix")  # the one action each context permits
LEGIT_SHARE = 0.5


def draw(rng: np.random.Generator, n: int, scale: float, temperature: float = 1.0):
    c = rng.choice(4, size=n, p=PRIOR)
    o = scale * MU[c] + rng.standard_normal((n, 2))
    logp = np.log(PRIOR) - 0.5 * ((o[:, None, :] - scale * MU[None, :, :]) ** 2).sum(-1)
    logp = logp / temperature
    logp -= logp.max(1, keepdims=True)
    post = np.exp(logp)
    post /= post.sum(1, keepdims=True)
    legit = rng.random(n) < LEGIT_SHARE
    other = (c + rng.integers(1, 4, size=n)) % 4  # one of the three actions c does not permit
    a = np.where(legit, c, other)  # index of the requested action (= the context that permits it)
    return c, post, a, legit


def rates(c, post, a, legit, rule: str, t: float = 0.5):
    if rule == "argmax":
        permit = post.argmax(1) == a
    elif rule == "bayes":
        permit = post[np.arange(len(a)), a] >= t
    else:
        raise ValueError(rule)
    return {"unauthorised": float(permit[~legit].mean()), "refused": float((~permit[legit]).mean())}


def scale_for_accuracy(target: float, seed: int = 1, n: int = 200_000) -> float:
    lo, hi = 0.2, 20.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        c, post, *_ = draw(np.random.default_rng(seed), n, mid)
        acc = float((post.argmax(1) == c).mean())
        lo, hi = (mid, hi) if acc < target else (lo, mid)
    return 0.5 * (lo + hi)


def study(seed: int = 20261122, accuracies=(0.80, 0.90, 0.95, 0.99), n: int = 400_000,
          cost_ratio: float = 9.0) -> dict:
    """Unauthorised and refused rates for the argmax rule and the expected-cost rule.

    The expected-cost rule permits when P(action permitted) >= t with t = C_u / (C_u + C_r);
    cost_ratio = C_u / C_r = 9 gives t = 0.9."""
    t_cost = cost_ratio / (1 + cost_ratio)
    out = {"t_cost": t_cost, "levels": []}
    thresholds = np.concatenate([np.linspace(0.01, 0.99, 99), [0.995, 0.999]])
    for acc in accuracies:
        s = scale_for_accuracy(acc)
        rng = np.random.default_rng(seed)
        c, post, a, legit = draw(rng, n, s)
        _, post_oc, _, _ = draw(np.random.default_rng(seed), n, s, temperature=0.5)
        realised = float((post.argmax(1) == c).mean())
        am = rates(c, post, a, legit, "argmax")
        cal = rates(c, post, a, legit, "bayes", t_cost)
        oc = rates(c, post_oc, a, legit, "bayes", t_cost)
        curve = [rates(c, post, a, legit, "bayes", float(t)) for t in thresholds]
        # where the argmax rule's unauthorised actions come from: (true, believed) pairs
        believed = post.argmax(1)
        bad = (~legit) & (believed == a)
        pairs = {}
        for i in range(4):
            for j in range(4):
                k = int((bad & (c == i) & (believed == j)).sum())
                if k:
                    pairs[f"{CONTEXTS[i]} read as {CONTEXTS[j]}"] = k / int(bad.sum())
        # the identity behind it: unauthorised = sum over true c of P(c | illegit) *
        # P(believed = context permitting the request | c), from the confusion matrix
        conf = np.array([[np.mean(believed[c == i] == j) for j in range(4)] for i in range(4)])
        p_c = np.array([np.mean(c[~legit] == i) for i in range(4)])
        identity = float(sum(p_c[i] * np.mean([conf[i, j] for j in range(4) if j != i]) for i in range(4)))
        out["levels"].append({"target_accuracy": acc, "scale": s, "accuracy": realised,
                              "argmax": am, "bayes_calibrated": cal, "bayes_overconfident": oc,
                              "curve": curve, "thresholds": [float(x) for x in thresholds],
                              "argmax_sources": dict(sorted(pairs.items(), key=lambda kv: -kv[1])),
                              "identity_unauthorised": identity,
                              "mean_max_posterior": float(post.max(1).mean()),
                              "mean_max_posterior_overconfident": float(post_oc.max(1).mean())})
    return out


def database_audit(seed: int = 5, n: int = 20_000, accuracy: float = 0.95) -> dict:
    """Run argmax decisions through a database that records each executed action against the
    grant row of the context the agent believed it was in, then run the integrity checks."""
    s = scale_for_accuracy(accuracy)
    c, post, a, legit = draw(np.random.default_rng(seed), n, s)
    believed = post.argmax(1)
    con = sqlite3.connect(":memory:", isolation_level=None)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("CREATE TABLE grant_row (id TEXT PRIMARY KEY, context TEXT NOT NULL, action TEXT NOT NULL)")
    con.executemany("INSERT INTO grant_row VALUES (?, ?, ?)",
                    [(f"g_{ctx}", ctx, ACTION[k]) for k, ctx in enumerate(CONTEXTS)])
    con.execute("CREATE TABLE execution (turn INTEGER PRIMARY KEY, context TEXT NOT NULL, action TEXT NOT NULL, "
                "grant_id TEXT NOT NULL REFERENCES grant_row(id))")
    rows = [(t, CONTEXTS[believed[t]], ACTION[a[t]], f"g_{CONTEXTS[believed[t]]}")
            for t in range(n) if believed[t] == a[t]]
    con.execute("BEGIN")
    con.executemany("INSERT INTO execution VALUES (?, ?, ?, ?)", rows)
    con.execute("COMMIT")
    fk = con.execute("PRAGMA foreign_key_check").fetchall()
    mismatch = con.execute("SELECT COUNT(*) FROM execution e JOIN grant_row g ON g.id = e.grant_id "
                           "WHERE g.context IS NOT e.context OR g.action IS NOT e.action").fetchone()[0]
    executed = len(rows)
    wrong = int(((believed == a) & ~legit).sum())
    return {"turns": n, "accuracy": float((believed == c).mean()), "executed": executed,
            "unauthorised_executed": wrong, "foreign_key_violations": len(fk), "grant_mismatches": int(mismatch)}
