"""Difference-in-differences of insider selling: announcements against earnings placebos.

For every event the change is post minus pre. The announcement effect is the
firm-level mean change at announcements minus the firm-level mean change at
earnings dates without a buyback authorization, which absorbs the trading-window
calendar. Inference: firm bootstrap.
"""
import numpy as np
import pandas as pd

OUTCOMES = ["sellers", "ceo_sells"]


def event_changes(ev):
    ev = ev.copy()
    ev["d_sellers"] = ev.sellers_post - ev.sellers_pre
    ev["d_ceo_sells"] = ev.ceo_sells_post - ev.ceo_sells_pre
    ev["d_log_shares"] = np.log1p(ev.shares_post) - np.log1p(ev.shares_pre)
    return ev


def firm_means(ev, kind, col):
    return ev[ev.kind == kind].groupby("cik")[col].mean()


def did(ev, ann_kind="announcement", col="d_sellers", n_boot=2000, seed=7):
    """Firm-level DiD with a firm bootstrap. Firms need both event kinds."""
    ev = event_changes(ev)
    a = firm_means(ev, ann_kind, col)
    p = firm_means(ev, "earnings_placebo", col)
    both = a.index.intersection(p.index)
    if len(both) < 5:
        return {"n_firms": int(len(both)), "estimate": None, "ci": None}
    diff = (a.loc[both] - p.loc[both]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(n_boot)]
    return {
        "n_firms": int(len(both)),
        "estimate": float(diff.mean()),
        "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "announcement_mean_change": float(a.loc[both].mean()),
        "placebo_mean_change": float(p.loc[both].mean()),
    }


def summary(ev):
    out = {}
    for kind in ("announcement", "announcement_with_earnings"):
        for col in ("d_sellers", "d_ceo_sells", "d_log_shares"):
            out[f"{kind}:{col}"] = did(ev, kind, col)
    return out
