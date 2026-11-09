"""Data tests (generic and singular) and unit tests with fixtures.

Every test returns a TestResult. Generic and singular tests run against built
models in a schema; unit tests build the models they need from fixture rows in
a schema of their own.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import yaml

from . import project

RAW = ("raw_flights", "raw_planes", "raw_weather", "raw_airports", "raw_airlines")


@dataclass(frozen=True)
class TestResult:
    name: str
    level: str  # schema | reconciliation | unit
    model: str
    severity: str  # error | warn
    passed: bool
    failing_rows: int
    error: str = ""


# ---------------------------------------------------------------- generic tests
def compile_generic(model: str, spec: dict) -> tuple[str, str]:
    kind = spec["test"]
    if kind == "not_null":
        cond = " or ".join(f"{c} is null" for c in spec["columns"])
        return f"not_null({','.join(spec['columns'])})", f"select * from {model} where {cond}"
    if kind == "unique":
        cols = ", ".join(spec["columns"])
        sql = f"select {cols} from {model} group by {cols} having count(*) > 1"
        return f"unique({','.join(spec['columns'])})", sql
    if kind == "accepted_values":
        values = ", ".join("'" + str(v).replace("'", "''") + "'" for v in spec["values"])
        col = spec["column"]
        sql = f"select * from {model} where {col} is not null and {col} not in ({values})"
        return f"accepted_values({col})", sql
    if kind == "accepted_range":
        col = spec["column"]
        sql = f"select * from {model} where {col} < {spec['min']} or {col} > {spec['max']}"
        return f"accepted_range({col})", sql
    if kind == "relationships":
        col, to, field = spec["column"], spec["to"], spec["field"]
        sql = (
            f"select child.{col} from {model} as child "
            f"left join {to} as parent on child.{col} = parent.{field} "
            f"where child.{col} is not null and parent.{field} is null"
        )
        return f"relationships({col}->{to}.{field})", sql
    if kind == "expression":
        return f"expression({spec['where']})", f"select * from {model} where {spec['where']}"
    if kind == "row_count":
        sql = f"select 1 where (select count(*) from {model}) <> {spec['equals']}"
        return f"row_count={spec['equals']}", sql
    raise ValueError(f"unknown test {kind}")


def run_query_test(con, name, level, model, severity, sql) -> TestResult:
    try:
        n = con.execute(f"select count(*) from ({sql})").fetchone()[0]
        return TestResult(name, level, model, severity, n == 0, n)
    except duckdb.Error as exc:  # a test that cannot run counts as a failure
        return TestResult(name, level, model, severity, False, -1, type(exc).__name__)


def run_schema_tests(con, schema_file: Path, models: set[str] | None = None):
    spec = yaml.safe_load(schema_file.read_text())
    out = []
    for model, tests in spec.items():
        if models is not None and model not in models:
            continue
        for t in tests:
            name, sql = compile_generic(model, t)
            sev = t.get("severity", "error")
            out.append(run_query_test(con, f"{model}:{name}", "schema", model, sev, sql))
    return out


def run_singular_tests(con, folder: Path):
    out = []
    for path in sorted(folder.glob("*.sql")):
        sql = path.read_text()
        model = sorted(project.referenced_tables(sql) - {"stg_flights", "stg_planes"} or {"?"})[0]
        out.append(run_query_test(con, path.stem, "reconciliation", model, "error", sql))
    return out


# ------------------------------------------------------------------- unit tests
def load_unit_cases(folder: Path, name: str = "cases.yml") -> tuple[list[dict], dict]:
    cases = yaml.safe_load((folder / name).read_text())
    defaults = yaml.safe_load((folder / "defaults.yml").read_text())
    return cases, defaults


_RAW_COLUMNS: dict[str, list] = {}


def _create_fixture_tables(con, schema: str, given: dict, defaults: dict) -> None:
    con.execute(f"create schema {schema}")
    for table in RAW:
        if table not in _RAW_COLUMNS:  # the naive loader always yields the same types
            _RAW_COLUMNS[table] = con.execute(f"describe main.{table}").fetchall()
        cols = _RAW_COLUMNS[table]
        con.execute(f"create table {schema}.{table} as select * from main.{table} limit 0")
        for row in given.get(table, []):
            full = {**defaults[table], **row}
            exprs = []
            for name, ctype, *_ in cols:
                value = full[name]
                if value is None:
                    exprs.append(f"cast(null as {ctype})")
                else:
                    text = str(value).replace("'", "''")
                    exprs.append(f"cast('{text}' as {ctype})")
            con.execute(f"insert into {schema}.{table} select {', '.join(exprs)}")


def _same(expected, actual) -> bool:
    if isinstance(actual, (dt.date, dt.datetime)):
        return str(expected) == actual.isoformat()[: len(str(expected))]
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if actual is None or (isinstance(actual, float) and math.isnan(actual)):
            return False
        try:
            return math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-6)
        except (TypeError, ValueError):
            return False
    return str(expected) == str(actual)


def compare_rows(expected: list[dict], actual: list[dict]) -> bool:
    if len(expected) != len(actual):
        return False
    remaining = list(actual)
    for exp_row in expected:
        match = next(
            (a for a in remaining if all(_same(v, a.get(k)) for k, v in exp_row.items())),
            None,
        )
        if match is None:
            return False
        remaining.remove(match)
    return True


def run_unit_tests(con, models: dict[str, project.Model], cases, defaults, tag="u"):
    out = []
    for i, case in enumerate(cases):
        schema = f"{tag}_{i}"
        target = case["model"]
        try:
            _create_fixture_tables(con, schema, case["given"], defaults)
            needed = project.upstream(models, target)
            project.build(con, models, schema, search_path=schema, only=needed)
            cur = con.execute(f"select * from {schema}.{target}")
            names = [d[0] for d in cur.description]
            actual = [dict(zip(names, r)) for r in cur.fetchall()]
            ok = compare_rows(case["expect"], actual)
            out.append(TestResult(case["name"], "unit", target, "error", ok, 0 if ok else 1))
        except duckdb.Error as exc:
            out.append(TestResult(case["name"], "unit", target, "error", False, -1, type(exc).__name__))
        finally:
            con.execute(f"drop schema if exists {schema} cascade")
            con.execute("set search_path = 'main'")
    return out
