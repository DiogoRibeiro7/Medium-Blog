"""Instances on a presented category (C-sets), the category of elements, and profunctors.

An instance assigns each object a table of rows and each generating arrow a function
between tables (a foreign key). It is a functor C -> Set exactly when every function is
total and well typed and every path equation holds row by row. The checks here say which
rows break which equation, so that the same answers can be compared with SQL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fincat import Path, Presentation


@dataclass
class Instance:
    schema: Presentation
    rows: dict[str, dict[str, dict]] = field(default_factory=dict)  # object -> row id -> attributes
    fk: dict[str, dict[str, str]] = field(default_factory=dict)  # arrow -> (row id -> row id)

    def __post_init__(self) -> None:
        for c in self.schema.objects:
            self.rows.setdefault(c, {})
        for a in self.schema.arrows:
            self.fk.setdefault(a.name, {})

    # -- building ----------------------------------------------------------------
    def add(self, obj: str, rid: str, **attrs) -> None:
        if any(rid in t for t in self.rows.values()):
            raise ValueError(f"row id {rid} already used")
        self.rows[obj][rid] = dict(attrs)

    def link(self, arrow: str, src_row: str, tgt_row: str) -> None:
        self.fk[arrow][src_row] = tgt_row

    def copy(self) -> "Instance":
        return Instance(self.schema, {c: {r: dict(v) for r, v in t.items()} for c, t in self.rows.items()},
                        {a: dict(m) for a, m in self.fk.items()})

    # -- evaluation --------------------------------------------------------------
    def apply(self, p: Path, rid: str) -> str | None:
        here = rid
        for name in p.word:
            here = self.fk[name].get(here)
            if here is None:
                return None
        return here

    def typing_errors(self) -> list[str]:
        errs = []
        for a in self.schema.arrows:
            src_rows, tgt_rows = self.rows[a.src], self.rows[a.tgt]
            for r in src_rows:
                if r not in self.fk[a.name]:
                    errs.append(f"{a.name}: row {r} of {a.src} has no value")
                elif self.fk[a.name][r] not in tgt_rows:
                    errs.append(f"{a.name}: row {r} points outside {a.tgt}")
            for r in self.fk[a.name]:
                if r not in src_rows:
                    errs.append(f"{a.name}: value set for row {r} that is not in {a.src}")
        return errs

    def equation_violations(self, lhs: Path, rhs: Path) -> list[str]:
        """Rows of the source table on which the two sides of a path equation disagree."""
        return sorted(r for r in self.rows[lhs.src] if self.apply(lhs, r) != self.apply(rhs, r))

    def all_violations(self) -> dict[str, list[str]]:
        return {f"{l} = {r}": self.equation_violations(l, r) for l, r in self.schema.equations}

    def is_functor(self) -> bool:
        return not self.typing_errors() and not any(self.all_violations().values())

    def transport(self, rid: str, src: str, tgt: str) -> set[str]:
        """Every row of tgt that rid reaches along some morphism src -> tgt."""
        return {self.apply(p, rid) for p in self.schema.hom(src, tgt)} - {None}

    # -- updates as instance maps --------------------------------------------------
    def naturality_violations(self, alpha: dict[str, dict[str, str]]) -> list[tuple[str, str, str, str]]:
        """Squares that fail for a map alpha: F => F given table by table.

        For each arrow g: c -> c' and row x of c, compare alpha_c' (g x) with g (alpha_c x).
        Returns (arrow, row, one route, the other route) for each failing square.
        """
        bad = []
        for a in self.schema.arrows:
            for x in self.rows[a.src]:
                route1 = alpha[a.tgt].get(self.fk[a.name][x], self.fk[a.name][x])
                route2 = self.fk[a.name][alpha[a.src].get(x, x)]
                if route1 != route2:
                    bad.append((a.name, x, route1, route2))
        return bad

    # -- category of elements (Grothendieck construction) ------------------------------
    def elements(self) -> list[dict]:
        """One flat table: a row per element, a context column, and one foreign-key column
        per arrow, filled only on rows whose context is that arrow's source."""
        flat = []
        for c in self.schema.objects:
            for r, attrs in self.rows[c].items():
                row = {"eid": r, "context": c, **attrs}
                for a in self.schema.arrows:
                    row[f"fk_{a.name}"] = self.fk[a.name].get(r) if a.src == c else None
                flat.append(row)
        return flat

    @classmethod
    def from_elements(cls, schema: Presentation, flat: list[dict], attr_names: tuple[str, ...]) -> "Instance":
        inst = cls(schema)
        for row in flat:
            inst.add(row["context"], row["eid"], **{k: row[k] for k in attr_names if k in row})
        for row in flat:
            for a in schema.arrows:
                v = row.get(f"fk_{a.name}")
                if v is not None:
                    inst.link(a.name, row["eid"], v)
        return inst

    def fibration_errors_flat(self, flat: list[dict]) -> list[str]:
        """The conditions that make a flat table the category of elements of a functor:
        an arrow's column is filled exactly on rows over its source, and points to a row
        over its target."""
        by_id = {r["eid"]: r for r in flat}
        errs = []
        for row in flat:
            for a in self.schema.arrows:
                v = row.get(f"fk_{a.name}")
                if (row["context"] == a.src) != (v is not None):
                    errs.append(f"{row['eid']}: fk_{a.name} filled on the wrong context")
                if v is not None and by_id.get(v, {}).get("context") != a.tgt:
                    errs.append(f"{row['eid']}: fk_{a.name} points off {a.tgt}")
        return errs

    def same_as(self, other: "Instance") -> bool:
        return self.rows == other.rows and {a: m for a, m in self.fk.items()} == {a: m for a, m in other.fk.items()}


# -- profunctors ---------------------------------------------------------------------
@dataclass
class Profunctor:
    """A set-valued profunctor P: A^op x B -> Set with A discrete and B presented.

    Stored as the category of elements: rows (id, a, b), covariant in b along the arrows of B
    (a function per arrow on rows over its source) and constant in a.
    """

    B: Presentation
    rows: dict[str, tuple[str, str]]  # id -> (a, b)
    act: dict[str, dict[str, str]]  # arrow of B -> (row id -> row id)


@dataclass
class CoProfunctor:
    """Q: B^op x C -> Set with C discrete: contravariant in b (functions go from rows over an
    arrow's target to rows over its source)."""

    B: Presentation
    rows: dict[str, tuple[str, str]]  # id -> (b, c)
    act: dict[str, dict[str, str]]


def coend_compose(P: Profunctor, Q: CoProfunctor) -> dict[tuple[str, str], int]:
    """Size of (Q o P)(a, c) = coend over b of P(a, b) x Q(b, c).

    Pairs (x, y) with x over b and y over b are glued along every arrow g: b -> b' by
    (P(g) x, y') ~ (x, Q(g) y') for y' over b'. Returns the number of classes for each (a, c).
    """
    pairs = []
    index = {}
    for x, (a, b) in P.rows.items():
        for y, (b2, c) in Q.rows.items():
            if b == b2:
                index[(x, y)] = len(pairs)
                pairs.append((a, c))
    parent = list(range(len(pairs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for g in P.B.arrows:
        for x, (a, b) in P.rows.items():
            if b != g.src:
                continue
            gx = P.act[g.name][x]
            for y2, (b2, c) in Q.rows.items():
                if b2 != g.tgt:
                    continue
                gy = Q.act[g.name][y2]
                i, j = find(index[(gx, y2)]), find(index[(x, gy)])
                if i != j:
                    parent[i] = j
    out: dict[tuple[str, str], set[int]] = {}
    for k, ac in enumerate(pairs):
        out.setdefault(ac, set()).add(find(k))
    return {ac: len(s) for ac, s in out.items()}


def join_count(P: Profunctor, Q: CoProfunctor) -> dict[tuple[str, str], int]:
    """The join alone: pairs (x, y) over the same b, with no gluing."""
    out: dict[tuple[str, str], int] = {}
    for x, (a, b) in P.rows.items():
        for y, (b2, c) in Q.rows.items():
            if b == b2:
                out[(a, c)] = out.get((a, c), 0) + 1
    return out
