"""A folder of SQL models, each a SELECT in <name>.sql, built in dependency order."""

from __future__ import annotations

from dataclasses import dataclass
from graphlib import TopologicalSorter
from pathlib import Path

import duckdb
import sqlglot
from sqlglot import exp


@dataclass(frozen=True)
class Model:
    name: str
    sql: str
    deps: frozenset[str]


def referenced_tables(sql: str) -> set[str]:
    tree = sqlglot.parse_one(sql, read="duckdb")
    ctes = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    return {t.name for t in tree.find_all(exp.Table)} - ctes


def load_project(folder: Path) -> dict[str, Model]:
    sources = {p.stem: p.read_text() for p in sorted(folder.glob("*.sql"))}
    return {
        name: Model(name, sql, frozenset(referenced_tables(sql) & sources.keys()))
        for name, sql in sources.items()
    }


def build_order(models: dict[str, Model]) -> list[str]:
    graph = {name: set(m.deps) for name, m in models.items()}
    return list(TopologicalSorter(graph).static_order())


def downstream(models: dict[str, Model], name: str) -> set[str]:
    out, frontier = {name}, {name}
    while frontier:
        frontier = {m for m, mod in models.items() if mod.deps & frontier} - out
        out |= frontier
    return out


def upstream(models: dict[str, Model], name: str) -> set[str]:
    out, frontier = {name}, {name}
    while frontier:
        frontier = set().union(*(models[m].deps for m in frontier)) - out
        out |= frontier
    return out


def build(
    con: duckdb.DuckDBPyConnection,
    models: dict[str, Model],
    schema: str,
    search_path: str,
    only: set[str] | None = None,
) -> None:
    """Materialise models as tables in ``schema``; unqualified names resolve
    through ``search_path`` (the schema itself first, then the raw tables)."""
    con.execute(f"create schema if not exists {schema}")
    con.execute(f"set search_path = '{search_path}'")
    for name in build_order(models):
        if only is not None and name not in only:
            continue
        con.execute(f"create or replace table {schema}.{name} as {models[name].sql}")
