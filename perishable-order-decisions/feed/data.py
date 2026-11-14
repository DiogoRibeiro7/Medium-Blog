"""FreshRetailNet-50K as arrays: one row per store-product series, one column per day (PLAN.md, "Data")."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = "Dingdong-Inc/FreshRetailNet-50K"
FILES = ("train", "eval")
IDS = (
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
)
DAILY = ("sale_amount", "stock_hour6_22_cnt", "discount", "holiday_flag", "activity_flag")
HOURLY = ("hours_sale", "hours_stock_status")
HOURS = 24


@dataclass
class Panel:
    dates: np.ndarray  # (T,) str, ascending
    test: np.ndarray  # (T,) bool, the eval split
    ids: dict[str, np.ndarray]  # (n,) int64 each
    sales: np.ndarray  # (n, T) float64
    oos_6_22: np.ndarray  # (n, T) int16
    discount: np.ndarray  # (n, T) float32
    holiday: np.ndarray  # (n, T) float32
    activity: np.ndarray  # (n, T) float32
    hours_sale: np.ndarray  # (n, T, 24) float32
    status: np.ndarray  # (n, T, 24) int8

    @property
    def n(self) -> int:
        return self.sales.shape[0]

    @property
    def days(self) -> int:
        return self.sales.shape[1]


def _hourly(col: pa.ChunkedArray, dtype) -> np.ndarray:
    arr = col.combine_chunks()
    offsets = np.asarray(arr.offsets)
    if not np.all(np.diff(offsets) == HOURS):
        raise ValueError("an hourly list does not have 24 values")
    return np.asarray(arr.flatten().to_numpy(zero_copy_only=False), dtype=dtype).reshape(-1, HOURS)


def read(path: Path, split: str) -> dict[str, np.ndarray]:
    t = pq.read_table(path, columns=[*IDS, "dt", *DAILY, *HOURLY])
    out = {c: t.column(c).to_numpy() for c in (*IDS, *DAILY)}
    out["dt"] = np.asarray(t.column("dt").to_pylist(), dtype=str)
    out["hours_sale"] = _hourly(t.column("hours_sale"), np.float32)
    out["hours_stock_status"] = _hourly(t.column("hours_stock_status"), np.int8)
    out["split"] = np.full(len(out["dt"]), split == "eval")
    return out


def build(parts: list[dict[str, np.ndarray]]) -> tuple[Panel, list[str]]:
    """Stack the splits into a series x day panel; return it with the structural checks as text lines."""
    cols = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.lexsort((cols["dt"], cols["product_id"], cols["store_id"]))
    cols = {k: v[order] for k, v in cols.items()}
    dates = np.unique(cols["dt"])
    T = len(dates)
    if len(cols["dt"]) % T:
        raise ValueError("rows are not a whole number of series x dates")
    n = len(cols["dt"]) // T
    if not np.array_equal(cols["dt"].reshape(n, T), np.broadcast_to(dates, (n, T))):
        raise ValueError("series do not share the same dates")
    key = cols["store_id"].astype(np.int64) * 1_000_000 + cols["product_id"].astype(np.int64)
    if len(np.unique(key)) != n:
        raise ValueError("store-product pairs are not unique series")
    split = cols["split"].reshape(n, T)
    if not (split == split[:1]).all():
        raise ValueError("the split differs between series")
    test = split[0]
    if test.any() and not (np.argmax(test) == T - test.sum() and test[-test.sum() :].all()):
        raise ValueError("eval dates are not all after train dates")

    def daily(c: str, dtype) -> np.ndarray:
        return cols[c].reshape(n, T).astype(dtype)

    ids = {c: cols[c].reshape(n, T)[:, 0].astype(np.int64) for c in IDS}
    for c in IDS:
        if not (cols[c].reshape(n, T) == ids[c][:, None]).all():
            raise ValueError(f"{c} changes within a series")
    panel = Panel(
        dates=dates,
        test=test,
        ids=ids,
        sales=daily("sale_amount", np.float64),
        oos_6_22=daily("stock_hour6_22_cnt", np.int16),
        discount=daily("discount", np.float32),
        holiday=daily("holiday_flag", np.float32),
        activity=daily("activity_flag", np.float32),
        hours_sale=cols["hours_sale"].reshape(n, T, HOURS),
        status=cols["hours_stock_status"].reshape(n, T, HOURS),
    )
    lines = [
        f"rows {len(cols['dt'])}; series {n}; dates {T} ({dates[0]} to {dates[-1]}); "
        f"train dates {int((~test).sum())}, eval dates {int(test.sum())} (from {dates[test][0]})",
        f"stores {len(np.unique(ids['store_id']))}; products {len(np.unique(ids['product_id']))}; "
        f"cities {len(np.unique(ids['city_id']))}",
        "every hourly list has 24 values; every series has the same dates; eval dates follow train dates",
    ]
    hourly_total = panel.hours_sale.sum(axis=2, dtype=np.float64)
    gap = np.abs(hourly_total - panel.sales)
    lines.append(
        f"daily sales against the sum of hourly sales: median absolute gap {np.median(gap):.6f}, "
        f"99th percentile {np.quantile(gap, 0.99):.6f}, share above 1e-3: {np.mean(gap > 1e-3):.4f}"
    )
    return panel, lines


def load(data_dir: Path) -> tuple[Panel, list[str]]:
    return build([read(data_dir / f"{s}.parquet", s) for s in FILES])
