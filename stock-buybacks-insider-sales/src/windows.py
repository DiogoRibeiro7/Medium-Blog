"""Trading-day windows around an event (weekdays only; exchange holidays ignored)."""
import numpy as np
import pandas as pd


def shift_bd(date, n):
    return pd.Timestamp(np.busday_offset(np.datetime64(pd.Timestamp(date).date()), n, roll="forward"))


def window_stats(sales, event_date, pre=10, post=10):
    """sales: DataFrame with tx_date (Timestamp), shares, owner_cik, ceo.
    Pre = [-pre, -1], post = [0, +post] trading days around the event date."""
    d0 = pd.Timestamp(event_date)
    pre_lo, pre_hi = shift_bd(d0, -pre), shift_bd(d0, -1)
    post_hi = shift_bd(d0, post)
    a = sales[(sales.tx_date >= pre_lo) & (sales.tx_date <= pre_hi)]
    b = sales[(sales.tx_date >= d0) & (sales.tx_date <= post_hi)]
    return {
        "sellers_pre": a.owner_cik.nunique(), "sellers_post": b.owner_cik.nunique(),
        "shares_pre": float(a.shares.sum()), "shares_post": float(b.shares.sum()),
        "ceo_sells_pre": int(a.ceo.any()) if len(a) else 0,
        "ceo_sells_post": int(b.ceo.any()) if len(b) else 0,
    }
