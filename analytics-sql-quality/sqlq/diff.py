"""Row-level comparison of two relations that share a key: what a reviewer
should see next to a code diff."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataDiff:
    rows_left: int
    rows_right: int
    only_left: int
    only_right: int
    changed: int
    unchanged: int

    @property
    def touched(self) -> int:
        return self.only_left + self.only_right + self.changed

    @property
    def touched_share(self) -> float:
        total = self.touched + self.unchanged
        return self.touched / total if total else 0.0


def data_diff(con, left: str, right: str, key: list[str], digits: int = 9) -> DataDiff:
    """Rows only on the left, only on the right, and matched rows whose
    non-key values differ (floats compared after rounding to ``digits``)."""
    types = {r[0]: r[1] for r in con.execute(f"describe {right}").fetchall()}
    left_cols = [r[0] for r in con.execute(f"describe {left}").fetchall()]
    values = [c for c in left_cols if c in types and c not in key]

    def norm(alias: str, col: str) -> str:
        if types[col] in ("DOUBLE", "FLOAT") or types[col].startswith("DECIMAL"):
            return f"round(cast({alias}.{col} as double), {digits})"
        return f"cast({alias}.{col} as varchar)"

    on = " and ".join(f"l.{k} is not distinct from r.{k}" for k in key)
    differs = " or ".join(f"{norm('l', c)} is distinct from {norm('r', c)}" for c in values)
    differs = differs or "false"
    sql = f"""
        with l as (select *, true as present_ from {left}),
             r as (select *, true as present_ from {right}),
             j as (
                 select l.present_ as lp, r.present_ as rp, ({differs}) as diff
                 from l full outer join r on {on}
             )
        select
            (select count(*) from {left}),
            (select count(*) from {right}),
            count(*) filter (where lp and rp is null),
            count(*) filter (where rp and lp is null),
            count(*) filter (where lp and rp and diff),
            count(*) filter (where lp and rp and not diff)
        from j
    """
    return DataDiff(*con.execute(sql).fetchone())
