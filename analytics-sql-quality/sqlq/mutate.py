"""Single-point mutants of SQL models, generated on the sqlglot syntax tree.

Each operator looks at one node and proposes zero or more replacements. A
mutant is the model with exactly one node replaced. Mutants whose generated
SQL equals the original, or another mutant, are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

DIALECT = "duckdb"


@dataclass(frozen=True)
class Mutant:
    model: str
    operator: str
    detail: str
    sql: str


ROR = {
    exp.EQ: [exp.NEQ],
    exp.NEQ: [exp.EQ],
    exp.LT: [exp.LTE, exp.GT],
    exp.LTE: [exp.LT, exp.GT],
    exp.GT: [exp.GTE, exp.LT],
    exp.GTE: [exp.GT, exp.LT],
}
AGG = {exp.Avg: exp.Median, exp.Max: exp.Min, exp.Min: exp.Max, exp.Sum: exp.Avg}
ARITH = {exp.Add: exp.Sub, exp.Sub: exp.Add, exp.Mul: exp.Div, exp.Div: exp.Mul}
COMPARISONS = tuple(ROR)
# Columns an analyst can plausibly confuse with each other in this schema.
SWAPS = {
    "dep_delay_min": "arr_delay_min",
    "arr_delay_min": "dep_delay_min",
    "origin": "dest",
    "dest": "origin",
}


def _conjuncts(node: exp.Expression) -> list[exp.Expression]:
    return list(node.flatten()) if isinstance(node, exp.And) else [node]


def _and_all(parts: list[exp.Expression]) -> exp.Expression:
    out = parts[0]
    for p in parts[1:]:
        out = exp.and_(out, p, copy=False)
    return out


def proposals(node: exp.Expression):
    """Yield (operator, detail, replacement_or_callable) for one node."""
    cls = type(node)
    if cls in ROR:
        for alt in ROR[cls]:
            yield "relational operator", f"{node.key} -> {alt.key}", alt(this=node.this.copy(), expression=node.expression.copy())
    if isinstance(node, exp.Literal) and not node.is_string and isinstance(node.parent, COMPARISONS):
        value = int(node.this) if node.this.lstrip("-").isdigit() else None
        if value is not None:
            for delta in (-1, 1):
                yield "boundary literal", f"{value} -> {value + delta}", exp.Literal.number(value + delta)
    if isinstance(node, exp.And):
        yield "logical connector", "and -> or", exp.or_(node.this.copy(), node.expression.copy())
    if isinstance(node, exp.Or):
        yield "logical connector", "or -> and", exp.and_(node.this.copy(), node.expression.copy())
    if isinstance(node, exp.Join):
        side = (node.args.get("side") or "").upper()
        kind = (node.args.get("kind") or "").upper()
        if side == "LEFT":
            new = node.copy()
            new.set("side", None)
            new.set("kind", "INNER")
            yield "join type", "left -> inner", new
        elif kind in ("", "INNER") and node.args.get("on") is not None:
            new = node.copy()
            new.set("kind", None)
            new.set("side", "LEFT")
            yield "join type", "inner -> left", new
        on = node.args.get("on")
        if on is not None and len(_conjuncts(on)) > 1:
            parts = _conjuncts(on)
            for i, p in enumerate(parts):
                new = node.copy()
                new.set("on", _and_all([q.copy() for j, q in enumerate(parts) if j != i]))
                yield "join key dropped", f"drop {p.sql(DIALECT)}", new
    if isinstance(node, exp.Where) and not isinstance(node.parent, exp.Filter):
        parts = _conjuncts(node.this)
        if len(parts) == 1:
            yield "predicate dropped", f"drop where {node.this.sql(DIALECT)}", None
        else:
            for i, p in enumerate(parts):
                new = exp.Where(this=_and_all([q.copy() for j, q in enumerate(parts) if j != i]))
                yield "predicate dropped", f"drop {p.sql(DIALECT)}", new
    if isinstance(node, exp.Filter):
        yield "filter dropped", f"drop filter {node.expression.this.sql(DIALECT)}", node.this.copy()
    if isinstance(node, exp.Count) and not isinstance(node.this, exp.Star) and node.this is not None:
        if isinstance(node.this, exp.Distinct):
            yield "distinct dropped", "count(distinct x) -> count(x)", exp.Count(this=node.this.expressions[0].copy())
        yield "count argument", f"count({node.this.sql(DIALECT)}) -> count(*)", exp.Count(this=exp.Star())
    if cls in AGG:
        yield "aggregate", f"{node.key} -> {AGG[cls].key}", AGG[cls](this=node.this.copy())
    if cls in ARITH:
        yield "arithmetic", f"{node.key} -> {ARITH[cls].key}", ARITH[cls](this=node.this.copy(), expression=node.expression.copy())
    if isinstance(node, exp.Is) and isinstance(node.expression, exp.Null):
        if isinstance(node.parent, exp.Not):
            pass
        else:
            yield "null check", "is null -> is not null", exp.Not(this=node.copy())
    if isinstance(node, exp.Not) and not isinstance(node.this, exp.Exists):
        yield "negation dropped", f"not {node.this.sql(DIALECT)} -> {node.this.sql(DIALECT)}", node.this.copy()
    if isinstance(node, exp.Coalesce):
        yield "null handling", "coalesce(x, y) -> x", node.this.copy()
    if isinstance(node, exp.Nullif):
        yield "null handling", "nullif(x, y) -> x", node.this.copy()
    if isinstance(node, exp.Cast) and node.to.this == exp.DataType.Type.DOUBLE:
        yield "cast dropped", "cast(x as double) -> x", node.this.copy()
    if isinstance(node, exp.Select) and node.args.get("distinct") is not None:
        new = node.copy()
        new.set("distinct", None)
        yield "distinct dropped", "select distinct -> select", new
    if isinstance(node, exp.Case) and len(node.args.get("ifs") or []) > 1:
        for i, branch in enumerate(node.args["ifs"]):
            new = node.copy()
            new.set("ifs", [b.copy() for j, b in enumerate(node.args["ifs"]) if j != i])
            yield "case branch dropped", f"drop when {branch.this.sql(DIALECT)}", new
    if (
        isinstance(node, exp.Column)
        and node.name in SWAPS
        and not isinstance(node.parent, (exp.Select, exp.Group))
    ):
        new = node.copy()
        new.set("this", exp.to_identifier(SWAPS[node.name]))
        yield "column swapped", f"{node.name} -> {SWAPS[node.name]}", new
    if isinstance(node, exp.Not) and isinstance(node.this, exp.Exists):
        new = _not_exists_to_not_in(node)
        if new is not None:
            yield "not exists -> not in", "rewrite as not in", new


def _not_exists_to_not_in(node: exp.Not) -> exp.Expression | None:
    """NOT EXISTS (SELECT .. FROM t WHERE t.a = o.b AND rest) ->
    o.b NOT IN (SELECT t.a FROM t WHERE rest)."""
    sub = node.this.this
    if not isinstance(sub, exp.Select) or sub.args.get("where") is None:
        return None
    inner = {t.alias_or_name for t in sub.find_all(exp.Table)}
    parts = _conjuncts(sub.args["where"].this)
    for i, p in enumerate(parts):
        if isinstance(p, exp.EQ) and isinstance(p.this, exp.Column) and isinstance(p.expression, exp.Column):
            a, b = p.this, p.expression
            if a.table in inner and b.table not in inner:
                inner_col, outer_col = a, b
            elif b.table in inner and a.table not in inner:
                inner_col, outer_col = b, a
            else:
                continue
            rest = [q.copy() for j, q in enumerate(parts) if j != i]
            new_sub = sub.copy()
            new_sub.set("expressions", [inner_col.copy()])
            new_sub.set("where", exp.Where(this=_and_all(rest)) if rest else None)
            return exp.Not(this=exp.In(this=outer_col.copy(), query=exp.Subquery(this=new_sub)))
    return None


def mutants(model: str, sql: str) -> tuple[str, list[Mutant]]:
    """Return (round-tripped original SQL, list of distinct mutants)."""
    tree = sqlglot.parse_one(sql, read=DIALECT)
    original = tree.sql(dialect=DIALECT, pretty=True)
    n_nodes = sum(1 for _ in tree.walk())
    seen = {original}
    out: list[Mutant] = []
    for idx in range(n_nodes):
        node = list(tree.walk())[idx]
        for operator, detail, _ in list(proposals(node)):
            copy = tree.copy()
            target = list(copy.walk())[idx]
            # recompute the proposal on the copy so replacements belong to it
            for op2, det2, repl in proposals(target):
                if (op2, det2) != (operator, detail):
                    continue
                if repl is None:
                    target.pop()
                else:
                    target.replace(repl)
                break
            text = copy.sql(dialect=DIALECT, pretty=True)
            if text in seen:
                continue
            seen.add(text)
            out.append(Mutant(model, operator, detail, text))
    return original, out
