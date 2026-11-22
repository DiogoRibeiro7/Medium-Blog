"""The deploy example as a schema, an instance and a conversation.

Contexts: Cloud contains Staging and Production; Production contains Incident. Moving out
of a context (pop) generalises a reading; moving into Incident (push) specialises one. The
only path equation says that entering the incident and leaving again changes nothing in
Production: u;i = id_Production. Leaving and re-entering is allowed to lose information,
and does: an approval for a specific hotfix comes back as a note with no grant.

A second schema adds one arrow, w: Cloud -> Production, with the section equation
w;p = id_Cloud. It looks harmless. It creates a path Staging -> Cloud -> Production.
"""

from __future__ import annotations

from cset import Instance
from fincat import Arrow, Path, Presentation

CONTEXTS = ("Cloud", "Staging", "Production", "Incident")


def schema(cloud_push: bool = False) -> Presentation:
    arrows = [
        Arrow("s", "Staging", "Cloud"),
        Arrow("p", "Production", "Cloud"),
        Arrow("i", "Incident", "Production"),
        Arrow("u", "Production", "Incident"),
    ]
    eqs = [(Path("Production", "Production", ("u", "i")), Path("Production", "Production", ()))]
    if cloud_push:
        arrows.append(Arrow("w", "Cloud", "Production"))
        eqs.append((Path("Cloud", "Cloud", ("w", "p")), Path("Cloud", "Cloud", ())))
    return Presentation(CONTEXTS, tuple(arrows), tuple(eqs))


ROWS = {
    "Cloud": [
        ("c_deploy", "deploy", "change the infrastructure", None),
        ("c_approve", "approve", "an approval was recorded in a sub-environment", None),
        ("c_revoked", "approve", "an approval was revoked", None),
    ],
    "Staging": [
        ("s_deploy", "deploy", "ship the build to staging", None),
        ("s_approve", "approve", "test procedure approved", "test_deploy"),
        ("s_revoked", "approve", "an approval was revoked", None),
    ],
    "Production": [
        ("p_deploy", "deploy", "release to users", None),
        ("p_approve", "approve", "release authorised", "release"),
        ("p_note", "approve", "an approval was recorded inside an incident", None),
        ("p_revoked", "approve", "an approval was revoked", None),
    ],
    "Incident": [
        ("i_deploy", "deploy", "apply an approved hotfix", None),
        ("i_suspended", "approve", "release authorisation, suspended for the incident", None),
        ("i_hotfix_42", "approve", "hotfix 42 approved", "hotfix"),
        ("i_hotfix_43", "approve", "hotfix 43 approved", "hotfix"),
        ("i_note", "approve", "an approval recorded in an earlier incident", None),
        ("i_revoked", "approve", "an approval was revoked", None),
    ],
}

LINKS = {
    "s": {"s_deploy": "c_deploy", "s_approve": "c_approve", "s_revoked": "c_revoked"},
    "p": {"p_deploy": "c_deploy", "p_approve": "c_approve", "p_note": "c_approve", "p_revoked": "c_revoked"},
    "i": {"i_deploy": "p_deploy", "i_suspended": "p_approve", "i_hotfix_42": "p_note",
          "i_hotfix_43": "p_note", "i_note": "p_note", "i_revoked": "p_revoked"},
    "u": {"p_deploy": "i_deploy", "p_approve": "i_suspended", "p_note": "i_note", "p_revoked": "i_revoked"},
}

# What "deploy" asks for in each context.
REQUESTED = {"Cloud": "change", "Staging": "test_deploy", "Production": "release", "Incident": "hotfix"}


def instance(cloud_push: str | None = None) -> Instance:
    """cloud_push: None (first schema), 'leak' (w sends the generic approval to the release
    authorisation) or 'safe' (w sends it to the note that grants nothing)."""
    S = schema(cloud_push is not None)
    inst = Instance(S)
    for c, rows in ROWS.items():
        for rid, token, meaning, grants in rows:
            inst.add(c, rid, token=token, meaning=meaning, grants=grants)
    for a, m in LINKS.items():
        for x, y in m.items():
            inst.link(a, x, y)
    if cloud_push is not None:
        target = {"leak": "p_approve", "safe": "p_note"}[cloud_push]
        for x, y in {"c_deploy": "p_deploy", "c_approve": target, "c_revoked": "p_revoked"}.items():
            inst.link("w", x, y)
    return inst


# A conversation: (turn, context the turn happens in, utterance, fact recorded or None).
CONVERSATION = [
    (1, "Production", "We are shipping v2.3 this week.", None),
    (2, "Staging", "The test procedure is approved.", "s_approve"),
    (3, "Staging", "Run the integration tests.", None),
    (4, "Incident", "Incident opened: payment latency.", None),
    (5, "Incident", "Hotfix 42 is approved.", "i_hotfix_42"),
    (6, "Incident", "Deploy?", None),
    (7, "Production", "Incident closed.", None),
    (8, "Production", "Deploy?", None),
    (9, "Incident", "New incident opened: cache errors.", None),
    (10, "Incident", "Deploy?", None),
]


POPS = {"s", "p", "i"}  # arrows that leave a context for the one containing it


def decide(inst: Instance, context: str, state: list[tuple[str, str]]):
    """Context-aware decision: allowed iff some fact in the state, carried along a morphism of
    the schema from where it sits to the current context, grants what deploy asks for."""
    want = REQUESTED[context]
    evidence = []
    for ctx, fact in state:
        for p in inst.schema.hom(ctx, context):
            row = inst.apply(p, fact)
            if row is not None and inst.rows[context][row]["grants"] == want:
                evidence.append((fact, str(p), row))
    return bool(evidence), evidence


def decide_blind(inst: Instance, recorded: list[tuple[str, str]]) -> bool:
    """The failure mode: 'something was approved', with no context predicate."""
    return any(inst.rows[ctx][f]["grants"] is not None for ctx, f in recorded)


def run_conversation(inst: Instance) -> list[dict]:
    """Replay the conversation. Facts sit in the context where they were recorded; when the
    conversation leaves a context for the one containing it, the facts recorded inside are
    carried out along that arrow (an incident's approvals become notes in Production)."""
    S = inst.schema
    state: list[tuple[str, str]] = []
    recorded: list[tuple[str, str]] = []
    out, prev = [], None
    for turn, ctx, text, fact in CONVERSATION:
        if prev is not None and prev != ctx:
            pop = [a for a in S.arrows if a.name in POPS and a.src == prev and a.tgt == ctx]
            if pop:
                state = [(ctx, inst.fk[pop[0].name][f]) if c == prev else (c, f) for c, f in state]
        if fact:
            state.append((ctx, fact))
            recorded.append((ctx, fact))
        if text == "Deploy?":
            ok, ev = decide(inst, ctx, state)
            out.append({"turn": turn, "context": ctx, "asks": REQUESTED[ctx], "aware": ok,
                        "evidence": ev, "blind": decide_blind(inst, recorded), "state": list(state)})
        prev = ctx
    return out


# Updates given table by table as old row -> new row.
def revoke_everywhere() -> dict[str, dict[str, str]]:
    alpha = {}
    for c, rows in ROWS.items():
        rev = [r for r, *_ in rows if r.endswith("revoked")][0]
        alpha[c] = {rid: rev for rid, token, *_ in rows if token == "approve" and rid != rev}
    return alpha


def revoke_in_incident_only() -> dict[str, dict[str, str]]:
    alpha = {c: {} for c in ROWS}
    alpha["Incident"] = revoke_everywhere()["Incident"]
    return alpha


def law_violations(inst: Instance, target: str = "Production", grant: str = "release") -> list[tuple]:
    """A grant in `target` may only come from a grant that was already in `target`: rows that a
    morphism carries to such a grant without being the image of one."""
    S, out = inst.schema, []
    homs = S.morphisms()
    sources = [z for z, a in inst.rows[target].items() if a["grants"] == grant]
    for c in S.objects:
        allowed = {inst.apply(m2, z) for m2 in homs[(target, c)] for z in sources}
        for m in homs[(c, target)]:
            for x in inst.rows[c]:
                y = inst.apply(m, x)
                if inst.rows[target][y]["grants"] == grant and x not in allowed:
                    out.append((c, x, str(m), y))
    return sorted(out)
