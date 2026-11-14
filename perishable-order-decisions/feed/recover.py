"""Out-of-stock hours and recovered demand (PLAN.md, "Recovered demand")."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feed.data import Panel

MIN_CLEAN_DAYS = 7
MIN_SHARE = 0.2
AGREEMENT = 0.99


@dataclass
class Coding:
    out_value: int  # the status value that means out of stock
    first: int  # first hour counted by stock_hour6_22_cnt
    last: int  # last hour counted (inclusive)
    agreement: float  # share of days on which the count matches


def stock_coding(panel: Panel) -> Coding:
    """Find which status value and which hour range reproduce stock_hour6_22_cnt."""
    best: Coding | None = None
    for out_value in (1, 0):
        out = panel.status == out_value
        for first, last in ((6, 22), (6, 21), (7, 22)):
            count = out[:, :, first : last + 1].sum(axis=2)
            agreement = float(np.mean(count == panel.oos_6_22))
            if best is None or agreement > best.agreement:
                best = Coding(out_value, first, last, agreement)
    assert best is not None
    if best.agreement < AGREEMENT:
        raise ValueError(f"no coding of the hourly status reproduces stock_hour6_22_cnt (best {best})")
    return best


def out_hours(panel: Panel, coding: Coding) -> np.ndarray:
    return panel.status == coding.out_value


@dataclass
class Profiles:
    share: np.ndarray  # (n, 24) hourly share of a day's sales
    source: np.ndarray  # (n,) 0 own, 1 third category, 2 all series
    clean_mean: np.ndarray  # (n,) mean sales on fully in-stock fit days (nan if none)


def profiles(panel: Panel, fit: np.ndarray) -> Profiles:
    """Hourly profiles from the fully in-stock days of the fit window."""
    clean = (panel.oos_6_22 == 0) & fit[None, :]
    hs = panel.hours_sale.astype(np.float64) * clean[:, :, None]
    own = hs.sum(axis=1)  # (n, 24)
    total = own.sum(axis=1)
    days = clean.sum(axis=1)
    ok = (days >= MIN_CLEAN_DAYS) & (total > 0)
    share = np.zeros_like(own)
    share[ok] = own[ok] / total[ok, None]
    source = np.where(ok, 0, 1)
    glob = own.sum(axis=0)
    glob = glob / glob.sum()
    cat = panel.ids["third_category_id"]
    for c in np.unique(cat[~ok]):
        members = cat == c
        pooled = own[members].sum(axis=0)
        rows = members & ~ok
        if pooled.sum() > 0:
            share[rows] = pooled / pooled.sum()
        else:
            share[rows] = glob
            source[rows] = 2
    clean_sales = np.where(clean, panel.sales, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        clean_mean = np.where(days > 0, clean_sales / days, np.nan)
    return Profiles(share, source, clean_mean)


@dataclass
class Recovery:
    demand: np.ndarray  # (n, T)
    stockout: np.ndarray  # (n, T) bool: some out-of-stock hour
    fallback: np.ndarray  # (n, T) bool: observed share below MIN_SHARE
    observed_share: np.ndarray  # (n, T)


def recover(panel: Panel, out: np.ndarray, prof: Profiles) -> Recovery:
    """Recovered daily demand for every series-day; days without out-of-stock hours keep their sales."""
    stockout = out.any(axis=2)
    observed_share = np.einsum("nth,nh->nt", (~out).astype(np.float64), prof.share)
    instock_sales = np.where(out, 0.0, panel.hours_sale.astype(np.float64)).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = np.where(observed_share > 0, instock_sales / observed_share, 0.0)
    fallback = stockout & (observed_share < MIN_SHARE)
    level = np.nan_to_num(prof.clean_mean, nan=0.0)[:, None]
    demand = panel.sales.copy()
    partial = stockout & ~fallback
    demand[partial] = np.maximum(panel.sales, scaled)[partial]
    demand[fallback] = np.maximum(panel.sales, np.broadcast_to(level, panel.sales.shape))[fallback]
    return Recovery(demand, stockout, fallback, observed_share)
