"""The table rules of each contract, judged on a profile, in a strict and a tolerant variant.

Seven kinds, as in Part 3 but written only from the dictionaries:

- schema: every field present under the dictionary's name with the kind of type its description implies
  (tolerant: names in any case, an integer accepted where a float is implied); a field with a ``since`` month
  is required only from that month; columns the dictionary does not list do not count against the rule;
- not null: no empty value in a required field;
- ranges: no negative amount, distance or duration, and timestamps in the order their descriptions imply;
- code lists: listed codes only, and zones from the zone lookup (1 to 265);
- month window: every pick-up inside the month of the file;
- unique rows: no exact duplicate row;
- reconciliation (yellow): the total equals the sum of its charges, to the cent.

Strict fires on any violating row; tolerant fires when, for some field, more than 1% of rows violate.
"""

from __future__ import annotations

from dataclasses import dataclass

from feed.contracts import CONTRACTS, TOLERANCE, Contract

KINDS = ("schema", "not null", "ranges", "code lists", "month window", "unique rows", "reconciliation")


@dataclass
class Rule:
    kind: str
    variant: str  # strict | tolerant
    fired: bool
    share: float  # largest share of violating rows over the fields the rule covers (schema: number of problems)
    detail: str

    @property
    def name(self) -> str:
        return f"{self.kind} ({self.variant})"


def type_kind(duck: str) -> str:
    t = duck.upper()
    if t in ("NULL", '"NULL"'):
        return "null"
    if any(x in t for x in ("INT", "HUGEINT")):
        return "integer"
    if any(x in t for x in ("DOUBLE", "FLOAT", "DECIMAL", "REAL")):
        return "float"
    if "TIMESTAMP" in t or t == "DATE":
        return "timestamp"
    if "VARCHAR" in t or "STRING" in t or "BLOB" in t:
        return "string"
    return t.lower()


def schema_problems(p: dict, c: Contract, tolerant: bool) -> list[str]:
    have = {name: type_kind(t) for name, t in p["schema"]}
    low = {name.lower(): (name, k) for name, k in have.items()}
    out = []
    for f in c.fields:
        if f.since and p["month"] < f.since:
            continue
        if tolerant:
            if f.name.lower() not in low:
                out.append(f"missing {f.name}")
                continue
            got = low[f.name.lower()][1]
            ok = got == f.kind or (f.kind == "float" and got == "integer")
        else:
            if f.name not in have:
                alt = low.get(f.name.lower())
                out.append(f"missing {f.name}" + (f" (found {alt[0]})" if alt else ""))
                continue
            got = have[f.name]
            ok = got == f.kind
        if not ok:
            out.append(f"{f.name}: {got}, expected {f.kind}")
    return out


def judge(p: dict, c: Contract | None = None) -> list[Rule]:
    c = c or CONTRACTS[p["feed"]]
    n = max(p["n_rows"], 1)
    cnt = p["counts"]
    out: list[Rule] = []

    for variant in ("strict", "tolerant"):
        probs = schema_problems(p, c, variant == "tolerant")
        out.append(Rule("schema", variant, bool(probs), float(len(probs)), "; ".join(probs[:6])))

    def group(prefixes: tuple[str, ...], kind: str, fields: set[str] | None = None) -> None:
        items = [
            (k, v)
            for k, v in cnt.items()
            if k.split("|")[0] in prefixes and (fields is None or k.split("|")[1] in fields)
        ]
        total = sum(v for _, v in items)
        worst = max(items, key=lambda kv: kv[1], default=("", 0))
        detail = ", ".join(
            f"{k.split('|', 1)[1]} {v / n:.4%}" for k, v in sorted(items, key=lambda kv: -kv[1])[:4] if v
        )
        out.append(Rule(kind, "strict", total > 0, worst[1] / n, detail))
        out.append(Rule(kind, "tolerant", worst[1] / n > TOLERANCE, worst[1] / n, detail))

    required = {f.name for f in c.fields if f.required and not (f.since and p["month"] < f.since)}
    present = {name.lower() for name, _ in p["schema"]}
    missing_required = [f for f in required if f.lower() not in present]
    group(("null",), "not null", required)
    if missing_required:  # a required field that is not delivered is empty on every row
        out[-2] = Rule("not null", "strict", True, 1.0, f"not delivered: {', '.join(sorted(missing_required))}")
        out[-1] = Rule("not null", "tolerant", True, 1.0, out[-2].detail)
    group(("neg", "order"), "ranges")
    group(("code", "zone"), "code lists")
    group(("window",), "month window")
    group(("unique",), "unique rows")
    if c.total:
        group(("total",), "reconciliation")
    return out
