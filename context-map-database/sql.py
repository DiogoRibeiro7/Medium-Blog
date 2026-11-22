"""The same structures in SQLite: tables, foreign keys, and constraint queries.

Each object becomes a table and each generating arrow a NOT NULL foreign-key column on its
source table. A path equation becomes a query that joins along both sides and returns the
rows where they end in different places. The category of elements becomes one table with a
context column, CHECK constraints for the columns that only rows of one context may fill,
and a query for the cross-row condition a CHECK cannot express. Hom-set emptiness becomes a
recursive query over the arrows of the schema.
"""

from __future__ import annotations

import sqlite3

from cset import CoProfunctor, Instance, Profunctor
from fincat import Path, Presentation

ATTRS = ("token", "meaning", "grants")


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def schema_ddl(S: Presentation) -> list[str]:
    stmts = []
    for c in S.objects:
        cols = ["id TEXT PRIMARY KEY"] + [f"{a} TEXT" for a in ATTRS]
        for arr in S.arrows:
            if arr.src == c:
                cols.append(f"{q(arr.name)} TEXT NOT NULL REFERENCES {q(arr.tgt)}(id)")
        stmts.append(f"CREATE TABLE {q(c)} ({', '.join(cols)})")
    stmts.append("CREATE TABLE schema_arrow (name TEXT PRIMARY KEY, src TEXT NOT NULL, tgt TEXT NOT NULL)")
    return stmts


def schema_connection(S: Presentation) -> sqlite3.Connection:
    """Only the arrows of a schema, for questions about the schema itself."""
    con = sqlite3.connect(":memory:", isolation_level=None)
    con.execute("CREATE TABLE schema_arrow (name TEXT PRIMARY KEY, src TEXT NOT NULL, tgt TEXT NOT NULL)")
    con.executemany("INSERT INTO schema_arrow VALUES (?, ?, ?)", [(a.name, a.src, a.tgt) for a in S.arrows])
    return con


def load(I: Instance) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:", isolation_level=None)
    con.execute("PRAGMA foreign_keys = ON")
    S = I.schema
    for stmt in schema_ddl(S):
        con.execute(stmt)
    con.executemany("INSERT INTO schema_arrow VALUES (?, ?, ?)", [(a.name, a.src, a.tgt) for a in S.arrows])
    con.execute("BEGIN")
    con.execute("PRAGMA defer_foreign_keys = ON")  # tables reference each other in cycles
    for c in S.objects:
        outs = [a for a in S.arrows if a.src == c]
        cols = ["id", *ATTRS, *[a.name for a in outs]]
        sql = f"INSERT INTO {q(c)} ({', '.join(q(x) for x in cols)}) VALUES ({', '.join('?' * len(cols))})"
        for rid, attrs in I.rows[c].items():
            con.execute(sql, (rid, *[attrs.get(k) for k in ATTRS], *[I.fk[a.name][rid] for a in outs]))
    con.execute("COMMIT")
    return con


def _walk(S: Presentation, p: Path, start_alias: str, tag: str) -> tuple[list[str], str]:
    joins, here_alias, here_obj = [], start_alias, p.src
    for k, name in enumerate(p.word):
        arr = S.by_name[name]
        nxt = f"{tag}{k}"
        joins.append(f"JOIN {q(arr.tgt)} {nxt} ON {nxt}.id = {here_alias}.{q(name)}")
        here_alias, here_obj = nxt, arr.tgt
    return joins, f"{here_alias}.id"


def equation_query(S: Presentation, lhs: Path, rhs: Path) -> str:
    """SELECT the ids of rows of the source table where the two paths disagree."""
    j1, end1 = _walk(S, lhs, "x", "l")
    j2, end2 = _walk(S, rhs, "x", "r")
    return (f"SELECT x.id FROM {q(lhs.src)} x " + " ".join(j1 + j2)
            + f" WHERE {end1} IS NOT {end2} ORDER BY x.id")


def equation_violations(con: sqlite3.Connection, S: Presentation) -> dict[str, list[str]]:
    return {f"{l} = {r}": [row[0] for row in con.execute(equation_query(S, l, r))] for l, r in S.equations}


REACH_SQL = """
WITH RECURSIVE reach(obj) AS (
    SELECT ? UNION SELECT s.tgt FROM reach r JOIN schema_arrow s ON s.src = r.obj
)
SELECT EXISTS (SELECT 1 FROM reach WHERE obj = ?)
"""


def reachable(con: sqlite3.Connection, a: str, b: str) -> bool:
    """Whether Hom(a, b) is non-empty: some chain of foreign keys leads from a to b."""
    return bool(con.execute(REACH_SQL, (a, b)).fetchone()[0])


def naturality_query(S: Presentation, changed: dict[str, dict[str, str]]) -> tuple[str, list[tuple]]:
    """Squares that fail for an update given as old id -> new id per table."""
    con_rows = []
    for c, m in changed.items():
        for old, new in m.items():
            con_rows.append((c, old, new))
    parts = []
    for arr in S.arrows:
        parts.append(f"""
        SELECT '{arr.name}' AS arrow, x.id AS row,
               COALESCE(a1.new, x.{q(arr.name)}) AS route1,
               x2.{q(arr.name)} AS route2
        FROM {q(arr.src)} x
        LEFT JOIN alpha a1 ON a1.tbl = '{arr.tgt}' AND a1.old = x.{q(arr.name)}
        LEFT JOIN alpha a0 ON a0.tbl = '{arr.src}' AND a0.old = x.id
        JOIN {q(arr.src)} x2 ON x2.id = COALESCE(a0.new, x.id)
        WHERE COALESCE(a1.new, x.{q(arr.name)}) IS NOT x2.{q(arr.name)}""")
    return " UNION ALL ".join(parts) + " ORDER BY arrow, row", con_rows


def naturality_violations(con: sqlite3.Connection, S: Presentation, alpha: dict[str, dict[str, str]]):
    sql, rows = naturality_query(S, alpha)
    con.execute("DROP TABLE IF EXISTS alpha")
    con.execute("CREATE TEMP TABLE alpha (tbl TEXT, old TEXT, new TEXT)")
    con.executemany("INSERT INTO alpha VALUES (?, ?, ?)", rows)
    return [tuple(r) for r in con.execute(sql)]


# -- category of elements: one table with a context column -----------------------------
def flat_ddl(S: Presentation) -> str:
    cols = ["eid TEXT PRIMARY KEY", "context TEXT NOT NULL"] + [f"{a} TEXT" for a in ATTRS]
    checks = []
    for arr in S.arrows:
        cols.append(f"fk_{arr.name} TEXT REFERENCES element(eid)")
        checks.append(f"CHECK ((context = '{arr.src}') = (fk_{arr.name} IS NOT NULL))")
    return f"CREATE TABLE element ({', '.join(cols + checks)})"


def load_flat(S: Presentation, flat: list[dict]) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:", isolation_level=None)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute(flat_ddl(S))
    cols = ["eid", "context", *ATTRS, *[f"fk_{a.name}" for a in S.arrows]]
    con.execute("BEGIN")
    con.execute("PRAGMA defer_foreign_keys = ON")
    con.executemany(f"INSERT INTO element ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    [tuple(r.get(k) for k in cols) for r in flat])
    con.execute("COMMIT")
    return con


def flat_target_errors(con: sqlite3.Connection, S: Presentation) -> list[tuple[str, str]]:
    """The cross-row condition: an arrow's column must point to a row over its target."""
    parts = [f"SELECT e.eid, 'fk_{a.name}' FROM element e JOIN element t ON t.eid = e.fk_{a.name} "
             f"WHERE t.context IS NOT '{a.tgt}'" for a in S.arrows]
    return [tuple(r) for r in con.execute(" UNION ALL ".join(parts))]


def try_insert_flat(con: sqlite3.Connection, S: Presentation, row: dict) -> str:
    cols = ["eid", "context", *ATTRS, *[f"fk_{a.name}" for a in S.arrows]]
    try:
        con.execute(f"INSERT INTO element ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    tuple(row.get(k) for k in cols))
        return "accepted"
    except sqlite3.IntegrityError as exc:
        return f"rejected ({exc})"


# -- profunctor composition: a join followed by connected components ----------------------
def coend_sql(P: Profunctor, Q: CoProfunctor) -> dict[tuple[str, str], int]:
    con = sqlite3.connect(":memory:", isolation_level=None)
    con.execute("CREATE TABLE p (id TEXT PRIMARY KEY, a TEXT, b TEXT)")
    con.execute("CREATE TABLE q (id TEXT PRIMARY KEY, b TEXT, c TEXT)")
    con.execute("CREATE TABLE p_act (arrow TEXT, x TEXT, gx TEXT)")
    con.execute("CREATE TABLE q_act (arrow TEXT, y TEXT, gy TEXT)")
    con.execute("CREATE TABLE arrow (name TEXT, src TEXT, tgt TEXT)")
    con.executemany("INSERT INTO p VALUES (?, ?, ?)", [(k, *v) for k, v in P.rows.items()])
    con.executemany("INSERT INTO q VALUES (?, ?, ?)", [(k, *v) for k, v in Q.rows.items()])
    con.executemany("INSERT INTO p_act VALUES (?, ?, ?)", [(g, x, y) for g, m in P.act.items() for x, y in m.items()])
    con.executemany("INSERT INTO q_act VALUES (?, ?, ?)", [(g, x, y) for g, m in Q.act.items() for x, y in m.items()])
    con.executemany("INSERT INTO arrow VALUES (?, ?, ?)", [(g.name, g.src, g.tgt) for g in P.B.arrows])
    sql = """
    WITH RECURSIVE
    pair(x, y, a, c) AS (SELECT p.id, q.id, p.a, q.c FROM p JOIN q ON p.b = q.b),
    edge(x1, y1, x2, y2) AS (
        SELECT pa.gx, y2.id, x.id, qa.gy
        FROM arrow g
        JOIN p x ON x.b = g.src
        JOIN p_act pa ON pa.arrow = g.name AND pa.x = x.id
        JOIN q y2 ON y2.b = g.tgt
        JOIN q_act qa ON qa.arrow = g.name AND qa.y = y2.id
    ),
    und(x1, y1, x2, y2) AS (SELECT * FROM edge UNION SELECT x2, y2, x1, y1 FROM edge),
    reach(x0, y0, x, y) AS (
        SELECT x, y, x, y FROM pair
        UNION
        SELECT r.x0, r.y0, u.x2, u.y2 FROM reach r JOIN und u ON u.x1 = r.x AND u.y1 = r.y
    ),
    rep(x, y, rx) AS (
        SELECT x, y, MIN(x0 || '|' || y0) FROM reach GROUP BY x, y
    )
    SELECT p.a, p.c, COUNT(DISTINCT r.rx) FROM pair p JOIN rep r ON r.x = p.x AND r.y = p.y
    GROUP BY p.a, p.c
    """
    return {(a, c): n for a, c, n in con.execute(sql)}


# -- a declared law, written as a query ---------------------------------------------------
def law_query(S: Presentation, target: str, grant: str) -> str:
    """Rows anywhere that some morphism carries to a row of `target` granting `grant`, and
    that are not themselves the image of such a row of `target` along a morphism back.

    In words: a grant in `target` may only come from a grant that was already in `target`.
    The morphisms are enumerated from the schema and each becomes one join."""
    homs = S.morphisms()
    parts = []
    for c in S.objects:
        back = []
        for k, m2 in enumerate(homs[(target, c)]):
            joins, end = _walk(S, m2, "z", f"b{k}_")
            back.append(f"SELECT {end} FROM {q(target)} z " + " ".join(joins) + f" WHERE z.grants = '{grant}'")
        allowed = " UNION ".join(back) if back else "SELECT NULL WHERE 0"
        for k, m in enumerate(homs[(c, target)]):
            joins, end = _walk(S, m, "x", f"f{k}_")
            end_alias = end.split(".")[0] if m.word else "x"
            parts.append(
                f"SELECT '{c}', x.id, '{m}', {end} FROM {q(c)} x " + " ".join(joins)
                + f" WHERE {end_alias}.grants = '{grant}' AND x.id NOT IN ({allowed})")
    return " UNION ALL ".join(parts)


def law_violations(con: sqlite3.Connection, S: Presentation, target: str, grant: str) -> list[tuple]:
    return sorted(tuple(r) for r in con.execute(law_query(S, target, grant)))
