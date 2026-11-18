"""Data loading, campaign panels and estimators for the coupon-campaign article.

Everything here is deterministic given the cached Complete Journey tables. The numbers the
article quotes are printed by coupon_numbers.py, which calls these functions.

Conventions
-----------
* A campaign week k covers the seven days start + 7k ... start + 7k + 6. Week 0 starts on the
  campaign's first day. Only weeks that end on or before 31 December 2017 are used.
* The baseline is the eight weeks before the start; panels can carry 16 or 24 weeks of history
  for the trend design and the placebos. The campaign period is weeks 0 ... ceil(length / 7) - 1.
  The post-period is the eight weeks after that.
* A unit is a household and a campaign. A household enters a campaign's panel if it bought
  anything in the eight pre-period weeks (an "active" household). Recipients are treated,
  everyone else is a control for that campaign.
* Every estimate is a difference in differences of unit means: a per-household change, recipients
  minus controls. The designs differ in the change (from the eight weeks before, or from the
  household's own fitted trend) and in whether controls are reweighted to the recipients within
  cells of earlier behaviour (coarsened exact matching).
* Sampling uncertainty comes from a bootstrap that resamples households, so a household that
  appears in several campaigns is resampled as one cluster.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / ".cache"
PRE_WEEKS = 8
POST_WEEKS = 8
FIRST_DAY = pd.Timestamp("2017-01-01")
LAST_DAY = pd.Timestamp("2017-12-31")   # receipts run into the first hours of 1 Jan 2018; whole days end here


@dataclass
class Data:
    tx: pd.DataFrame            # one row per receipt line
    desc: pd.DataFrame          # one row per campaign: type, start, end
    recipients: pd.DataFrame    # campaign_id, household_id
    coupons: pd.DataFrame       # coupon_upc, product_id, campaign_id
    redemptions: pd.DataFrame   # household_id, coupon_upc, campaign_id, date
    households: pd.Index        # every household with a receipt


def load(cache=CACHE):
    """Read the cached tables and add the few derived columns every section uses."""
    tx = pd.read_parquet(cache / "transactions.parquet")
    tx["date"] = tx["transaction_timestamp"].dt.normalize()
    tx = tx.drop(columns=["transaction_timestamp"])
    for col in ["household_id", "basket_id", "product_id", "store_id"]:
        tx[col] = tx[col].astype(str)
    desc = pd.read_parquet(cache / "campaign_descriptions.parquet")
    desc["campaign_id"] = desc["campaign_id"].astype(str)
    desc["campaign_type"] = desc["campaign_type"].astype(str)
    desc["start"] = pd.to_datetime(desc["start_date"])
    desc["end"] = pd.to_datetime(desc["end_date"])
    desc["order"] = desc["campaign_id"].astype(int)
    desc = desc.sort_values(["start", "order"]).drop(columns="order").reset_index(drop=True)
    rec = pd.read_parquet(cache / "campaigns.parquet").astype(str)
    cou = pd.read_parquet(cache / "coupons.parquet").astype(str)
    red = pd.read_parquet(cache / "coupon_redemptions.parquet")
    red = red.assign(household_id=red["household_id"].astype(str), coupon_upc=red["coupon_upc"].astype(str),
                     campaign_id=red["campaign_id"].astype(str), date=pd.to_datetime(red["redemption_date"]))
    red = red.drop(columns=["redemption_date"])
    households = pd.Index(sorted(tx["household_id"].unique()))
    return Data(tx, desc, rec, cou, red, households)


def campaign_weeks(row):
    """Number of campaign weeks, and the last complete week index available in the data."""
    n_dur = int(np.ceil(((row.end - row.start).days + 1) / 7))
    k_max = int(((LAST_DAY - row.start).days - 6) // 7)
    return n_dur, k_max


def eligible(desc, pre_weeks=PRE_WEEKS):
    """Campaigns with pre_weeks full weeks before the start and at least one complete campaign week."""
    ok = []
    for row in desc.itertuples():
        n_dur, k_max = campaign_weeks(row)
        if row.start - pd.Timedelta(weeks=pre_weeks) >= FIRST_DAY and k_max >= 0:
            ok.append(row.campaign_id)
    return ok


@dataclass
class Panel:
    campaign_id: str
    campaign_type: str
    households: np.ndarray       # household id per unit
    treated: np.ndarray          # bool per unit
    weeks: np.ndarray            # week indices, -8 ... k_last
    n_dur_obs: int               # campaign weeks observed
    n_post_obs: int              # post weeks observed
    n_dur: int                   # campaign weeks by the calendar
    y: dict                      # outcome name -> (units x weeks) array
    redemption_trips: np.ndarray  # distinct (household, day) redemptions in the observed campaign weeks, per unit
    redemption_rows: np.ndarray   # coupon redemptions in the observed campaign weeks, per unit
    overlap_a: np.ndarray        # unit also received a Type A campaign that overlaps this campaign's weeks
    n_recipients: int            # recipients of the campaign, active or not
    n_recipients_inactive: int   # recipients with no receipt in the pre-period (outside the panel)
    redemptions_inactive: int    # redemptions by those recipients in the observed campaign weeks


def build_panel(data, campaign_id, covered=None, pre_weeks=PRE_WEEKS):
    """Unit-by-week outcomes for one campaign.

    covered: set of product ids that the campaign's coupons cover. Defaults to the coupon table.
    pre_weeks: weeks of history before the start kept in the panel (8, 16 or 24). Households
    enter the panel if they bought anything in the last eight of them.
    Outcomes: cov_trips (baskets with a covered product), cov_spend (sales value of covered
    products), all_trips (baskets), all_spend (sales value of everything).
    """
    row = data.desc.loc[data.desc.campaign_id == campaign_id].iloc[0]
    n_dur, k_max = campaign_weeks(row)
    k_dur_last = min(n_dur - 1, k_max)
    k_last = min(n_dur + POST_WEEKS - 1, k_max)
    lo = row.start - pd.Timedelta(weeks=pre_weeks)
    hi = row.start + pd.Timedelta(weeks=k_last + 1)
    w = data.tx[(data.tx.date >= lo) & (data.tx.date < hi)]
    w = w.assign(k=((w.date - row.start).dt.days // 7).astype(int))
    active = np.array(sorted(w.loc[(w.k < 0) & (w.k >= -PRE_WEEKS), "household_id"].unique()))
    weeks = np.arange(-pre_weeks, k_last + 1)
    if covered is None:
        covered = set(data.coupons.loc[data.coupons.campaign_id == campaign_id, "product_id"])
    is_cov = w.product_id.isin(covered)

    def grid(frame, value):
        s = frame.groupby(["household_id", "k"])[value].agg("nunique" if value == "basket_id" else "sum")
        return s.unstack(fill_value=0).reindex(index=active, columns=weeks, fill_value=0).to_numpy(float)

    y = {"cov_trips": grid(w[is_cov], "basket_id"), "cov_spend": grid(w[is_cov], "sales_value"),
         "all_trips": grid(w, "basket_id"), "all_spend": grid(w, "sales_value")}
    rec = set(data.recipients.loc[data.recipients.campaign_id == campaign_id, "household_id"])
    treated = np.isin(active, list(rec))
    # redemptions in the observed campaign weeks
    red = data.redemptions[(data.redemptions.campaign_id == campaign_id) & (data.redemptions.date >= row.start)
                           & (data.redemptions.date < row.start + pd.Timedelta(weeks=k_dur_last + 1))]
    rows = red.groupby("household_id").size().reindex(active, fill_value=0).to_numpy(float)
    trips = red.drop_duplicates(["household_id", "date"]).groupby("household_id").size()
    trips = trips.reindex(active, fill_value=0).to_numpy(float)
    inactive = rec - set(active)
    red_inactive = int(red.household_id.isin(inactive).sum())
    # overlap with Type A campaigns
    a_ids = []
    for other in data.desc.itertuples():
        if other.campaign_type == "Type A" and other.campaign_id != campaign_id:
            if other.start <= row.start + pd.Timedelta(weeks=k_dur_last + 1) and other.end >= row.start:
                a_ids.append(other.campaign_id)
    a_hh = set(data.recipients.loc[data.recipients.campaign_id.isin(a_ids), "household_id"])
    overlap = np.isin(active, list(a_hh))
    return Panel(campaign_id, row.campaign_type, active, treated, weeks, k_dur_last + 1,
                 max(0, k_last - k_dur_last), n_dur, y, trips, rows, overlap, len(rec), len(inactive), red_inactive)


def window_mean(panel, outcome, lo, hi):
    """Per-unit mean of an outcome over campaign weeks lo..hi inclusive; NaN when none of them is observed."""
    wk = panel.weeks
    mask = (wk >= lo) & (wk <= hi)
    y = panel.y[outcome]
    return y[:, mask].mean(1) if mask.any() else np.full(len(y), np.nan)


def campaign_span(panel):
    """First and last observed campaign week."""
    return 0, panel.n_dur_obs - 1


def post_span(panel):
    """First and last post-campaign week (the eight weeks after the calendar end); may be unobserved."""
    return panel.n_dur, panel.n_dur + POST_WEEKS - 1


def event_labels(pre_weeks=PRE_WEEKS):
    """Positions of a pooled event study: each pre week, the first five campaign weeks, eight post weeks."""
    return [f"w{k}" for k in range(-pre_weeks, 0)] + [f"d{k}" for k in range(1, 6)] + \
           [f"p{k}" for k in range(1, POST_WEEKS + 1)]


def event_changes(panel, outcome, base_lo, base_hi, labels):
    """Per-unit outcome at each event position minus the unit's mean over the baseline weeks."""
    y = panel.y[outcome]
    wk = panel.weeks
    base = window_mean(panel, outcome, base_lo, base_hi)
    out = np.full((len(y), len(labels)), np.nan)
    for j, lab in enumerate(labels):
        if lab[0] == "w":
            k = int(lab[1:])
        elif lab[0] == "d":
            k = int(lab[1:]) - 1
            if k >= panel.n_dur_obs:
                continue
        else:
            k = panel.n_dur + int(lab[1:]) - 1
        col = wk == k
        if col.any():
            out[:, j] = y[:, col][:, 0] - base
    return out


def trend_residual(panel, outcome, fit_lo, fit_hi, tgt_lo, tgt_hi):
    """Per-unit mean over weeks tgt_lo..tgt_hi minus the unit's own straight line fitted to weeks fit_lo..fit_hi.

    Returns (residual, number of target weeks observed). The line is ordinary least squares on
    the weekly values, so each household is compared with where its own trajectory was heading.
    """
    wk = panel.weeks
    y = panel.y[outcome]
    f = (wk >= fit_lo) & (wk <= fit_hi)
    t = wk[f].astype(float)
    tb = t.mean()
    yb = y[:, f].mean(1)
    slope = ((y[:, f] - yb[:, None]) * (t - tb)).sum(1) / ((t - tb) ** 2).sum()
    g = (wk >= tgt_lo) & (wk <= tgt_hi)
    if not g.any():
        return np.full(len(y), np.nan), 0
    return y[:, g].mean(1) - (yb + slope * (wk[g].mean() - tb)), int(g.sum())


def _bins_pre(panel, lo, hi):
    wk = panel.weeks
    mask = (wk >= lo) & (wk <= hi)
    return mask


def cells_known_coupons(panel, lo=-PRE_WEEKS, hi=-1):
    """Matching cells for campaigns whose recipients all got the same coupons (Types B and C).

    Baskets with a covered product in weeks lo..hi (0, 1, 2, 3-5, 6+), total spend in those weeks
    (quintiles among the campaign's active households), and whether the household also had a
    Type A campaign running at the same time.
    """
    mask = _bins_pre(panel, lo, hi)
    cov = panel.y["cov_trips"][:, mask].sum(1)
    cov_bin = np.digitize(cov, [1, 2, 3, 6])
    spend = panel.y["all_spend"][:, mask].sum(1)
    spend_bin = np.digitize(spend, np.quantile(spend, [0.2, 0.4, 0.6, 0.8]))
    return cov_bin * 100 + spend_bin * 10 + panel.overlap_a.astype(int)


def cells_targeted(panel, lo=-PRE_WEEKS, hi=-1):
    """Matching cells for Type A campaigns: total spend in weeks lo..hi (deciles) by number of baskets (terciles)."""
    mask = _bins_pre(panel, lo, hi)
    spend = panel.y["all_spend"][:, mask].sum(1)
    trips = panel.y["all_trips"][:, mask].sum(1)
    sb = np.digitize(spend, np.quantile(spend, np.linspace(0.1, 0.9, 9)))
    tb = np.digitize(trips, np.quantile(trips, [1 / 3, 2 / 3]))
    return sb * 10 + tb


def did(d, treated, cells, hh_pos, W):
    """Matched difference in differences for one campaign, for every bootstrap draw at once.

    d       per-unit change (n,) or (n, m); columns that are all NaN are returned as NaN
    treated bool per unit
    cells   matching cell per unit
    hh_pos  position of each unit's household in the household index (for the bootstrap weights)
    W       (B, n_households) household multiplicities; a row of ones gives the point estimate
    Returns (att, n_treated): att has shape (B,) or (B, m); n_treated is the bootstrap-weighted
    number of recipients that have controls in their cell.
    """
    two_d = d.ndim == 2
    dd = d if two_d else d[:, None]
    col_ok = ~np.isnan(dd).all(0)
    dd = np.nan_to_num(dd)
    B = W.shape[0]
    num = np.zeros((B, dd.shape[1]))
    den = np.zeros(B)
    for c in np.unique(cells):
        it = np.flatnonzero((cells == c) & treated)
        ic = np.flatnonzero((cells == c) & ~treated)
        if len(it) == 0 or len(ic) == 0:
            continue
        wt = W[:, hh_pos[it]]
        wc = W[:, hh_pos[ic]]
        st = wt.sum(1)
        sc = wc.sum(1)
        ok = (st > 0) & (sc > 0)
        mt = (wt @ dd[it]) / np.where(st > 0, st, 1)[:, None]
        mc = (wc @ dd[ic]) / np.where(sc > 0, sc, 1)[:, None]
        num += np.where(ok[:, None], st[:, None] * (mt - mc), 0.0)
        den += np.where(ok, st, 0.0)
    att = num / np.where(den > 0, den, np.nan)[:, None]
    att[:, ~col_ok] = np.nan
    return (att if two_d else att[:, 0]), den


def bootstrap_weights(n_households, draws, seed):
    """Household multiplicities for a nonparametric cluster bootstrap; row 0 is all ones (the estimate)."""
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n_households, np.full(n_households, 1.0 / n_households), size=draws).astype(float)
    return np.vstack([np.ones((1, n_households)), W])


def bh(pvalues):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out


def redemption_lines(data):
    """Receipt lines that carry a campaign coupon: same household, same day, a product the coupon covers,
    and a manufacturer coupon discount on the line. One redemption can match more than one line."""
    m = data.redemptions.merge(data.coupons, on=["coupon_upc", "campaign_id"])
    m = m.merge(data.tx[["household_id", "product_id", "date", "basket_id", "sales_value", "retail_disc",
                         "coupon_disc", "coupon_match_disc", "quantity"]],
                on=["household_id", "product_id", "date"])
    m = m[m.coupon_disc > 0]
    return m
