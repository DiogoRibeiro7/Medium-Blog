"""Building blocks for the mirror-statistics test of fund intermediation through Ireland and Luxembourg.

The balance of payments records a security by the residence of its holder. When an investment fund
resident in Ireland or Luxembourg (the "hubs") buys a US share, the EU records a purchase of a US
security; when an investor outside the EU buys a share of that fund, the EU records a non-EU purchase
of an EU security. The statistics of the hubs do not say who holds their fund shares, but the other
member states report what they hold in each hub. Non-EU holdings are the residual:

    held outside the EU = held by all non-residents of the hub - held by residents of other members

Three member states do not publish the bilateral cell for every year (France and Austria for both
hubs, Luxembourg for Ireland), and Germany's quarterly flows by partner stop in 2020. Their holdings
are imputed from their published intra-EU totals, and bounds assign those totals to the hubs either in
full or not at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HUBS = ("IE", "LU")
MEMBERS = ("AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT",
           "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK")
F, F5, F51, F52, F3 = "FA__P__F", "FA__P__F5", "FA__P__F51", "FA__P__F52", "FA__P__F3"
EU, EXT, WRL, US = "EU27_2020", "EXT_EU27_2020", "WRL_REST", "US"
FIRST, LAST = 2015, 2025          # window of the flow analysis, as in the main series
POS_FIRST = 2014                  # first year-end with an EU27 extra-EU position


@dataclass
class Data:
    flows: pd.Series          # member flows, index (geo, partner, item, side, quarter)
    positions: pd.Series      # member positions, index (geo, partner, item, side, year)
    eu_flows: pd.Series       # EU27 flows, index (partner, item, side, sector, quarter)
    eu_positions: pd.Series   # EU27 positions, index (partner, item, side, year)
    fin: pd.Series            # financial balance sheets, index (geo, sector, finpos, item, year)


def load(data_dir: Path | str = "data") -> Data:
    d = Path(data_dir)
    fl = pd.read_csv(d / "member_flows.csv", low_memory=False)
    fl = fl[(fl.freq == "Q") & (fl.sector10 == "S1") & (fl.sectpart == "S1")]
    po = pd.read_csv(d / "member_positions.csv", low_memory=False)
    po = po[(po.freq == "A") & (po.sector10 == "S1") & (po.sectpart == "S1")]
    po = po.assign(time=po.time.astype(int))
    ef = pd.read_csv(d / "eu_flows.csv", low_memory=False)
    ef = ef[ef.sectpart == "S1"]
    ep = pd.read_csv(d / "eu_positions.csv", low_memory=False)
    ep = ep[(ep.sector10 == "S1") & (ep.freq == "A")].assign(time=lambda x: x.time.astype(int))
    fa = pd.read_csv(d / "fin_accounts.csv", low_memory=False).assign(time=lambda x: x.time.astype(int))
    return Data(
        flows=fl.set_index(["geo", "partner", "bop_item", "stk_flow", "time"]).value.sort_index(),
        positions=po.set_index(["geo", "partner", "bop_item", "stk_flow", "time"]).value.sort_index(),
        eu_flows=ef.set_index(["partner", "bop_item", "stk_flow", "sector10", "time"]).value.sort_index(),
        eu_positions=ep.set_index(["partner", "bop_item", "stk_flow", "time"]).value.sort_index(),
        fin=fa.set_index(["geo", "sector", "finpos", "na_item", "time"]).value.sort_index(),
    )


def quarters(first: int = FIRST, last: int = LAST) -> list[str]:
    return [f"{y}-Q{q}" for y in range(first, last + 1) for q in range(1, 5)]


def years(first: int = POS_FIRST, last: int = LAST) -> list[int]:
    return list(range(first, last + 1))


def _get(s: pd.Series, key: tuple, index: list) -> pd.Series:
    try:
        out = s.loc[key]
    except KeyError:
        return pd.Series(np.nan, index=index, dtype=float)
    return out.reindex(index).astype(float)


def flow(d: Data, geo: str, partner: str, item: str, side: str) -> pd.Series:
    """Quarterly net purchases (side 'ASS') or net incurrence of liabilities ('LIAB'), EUR million."""
    return _get(d.flows, (geo, partner, item, side), quarters())


def position(d: Data, geo: str, partner: str, item: str, side: str) -> pd.Series:
    """End-of-year position ('A_LE' assets, 'L_LE' liabilities), EUR million."""
    return _get(d.positions, (geo, partner, item, side), years())


def eu_flow(d: Data, partner: str, item: str, side: str, sector: str = "S1") -> pd.Series:
    return _get(d.eu_flows, (partner, item, side, sector), quarters())


def eu_position(d: Data, partner: str, item: str, side: str) -> pd.Series:
    return _get(d.eu_positions, (partner, item, side), years())


def annual(q: pd.Series) -> pd.Series:
    """Sum quarters into calendar years; a year with a missing quarter is missing."""
    yr = q.index.str[:4].astype(int)
    grouped = q.groupby(yr)
    return grouped.sum(min_count=4).where(grouped.count() == 4)


# ---------------------------------------------------------------------------------------------
# Who buys US securities
# ---------------------------------------------------------------------------------------------

def us_purchases(d: Data, item: str = F) -> pd.DataFrame:
    """Quarterly purchases of US securities by member (columns), with Luxembourg's equity derived as
    total minus debt where only those two are published."""
    cols = {}
    for g in MEMBERS:
        s = flow(d, g, US, item, "ASS")
        if g == "LU" and item == F5 and s.isna().all():
            s = flow(d, g, US, F, "ASS") - flow(d, g, US, F3, "ASS")
        cols[g] = s
    return pd.DataFrame(cols)


def us_holdings(d: Data, item: str = F) -> pd.DataFrame:
    """End-of-year holdings of US securities by member, same convention as ``us_purchases``."""
    cols = {}
    for g in MEMBERS:
        s = position(d, g, US, item, "A_LE")
        if g == "LU" and item == F5 and s.isna().all():
            s = position(d, g, US, F, "A_LE") - position(d, g, US, F3, "A_LE")
        cols[g] = s
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------------------------
# Intra-EU holdings and purchases of each member, with Luxembourg derived from the EU27 aggregate
# ---------------------------------------------------------------------------------------------

def _lu_residual(eu_total: pd.Series, others: pd.DataFrame) -> pd.Series:
    """EU27 total minus the 26 other members; missing if any of them is missing."""
    return (eu_total - others.sum(axis=1, min_count=1)).where(others.notna().all(axis=1))


def extra_eu_ratio(d: Data, geo: str, item: str = F5) -> float:
    """Share of non-EU securities in a member's foreign holdings in the first year it publishes both."""
    ext = position(d, geo, EXT, item, "A_LE")
    wrl = position(d, geo, WRL, item, "A_LE")
    ok = ext.notna() & wrl.notna() & (wrl != 0)
    return float((ext[ok] / wrl[ok]).iloc[0]) if ok.any() else np.nan


def _extra_eu(d: Data, kind: str, item: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Holdings ('pos') or purchases ('flow') of non-EU securities by the 26 members other than
    Luxembourg. A member's unpublished periods (positions: Greece to 2018, Malta to 2016, Belgium and
    Bulgaria in 2014; flows: Greece to 2019 and some quarters for Malta) are filled with its total
    times its first published non-EU share, so that Luxembourg's residual exists in every period.
    Returns (values, mask of filled cells)."""
    get, side = (position, "A_LE") if kind == "pos" else (flow, "ASS")
    vals, filled = {}, {}
    for g in MEMBERS:
        if g == "LU":
            continue
        ext = get(d, g, EXT, item, side)
        miss = ext.isna()
        if miss.any():
            fill = get(d, g, WRL, item, side).fillna(0.0) * extra_eu_ratio(d, g, item)
            ext = ext.fillna(fill)
        vals[g], filled[g] = ext, miss
    return pd.DataFrame(vals), pd.DataFrame(filled)


def filled_cells(d: Data, item: str = F5) -> dict[str, int]:
    """Number of member-periods whose non-EU holdings or purchases were filled (for the summary)."""
    return {kind: int(_extra_eu(d, kind, item)[1].loc[lambda x: x.index.map(str) >= str(FIRST)].sum().sum())
            for kind in ("pos", "flow")}


def _intra_eu(d: Data, kind: str, item: str) -> pd.DataFrame:
    get, side = (position, "A_LE") if kind == "pos" else (flow, "ASS")
    eu_get = eu_position if kind == "pos" else eu_flow
    ext, _ = _extra_eu(d, kind, item)
    cols = {}
    for g in MEMBERS:
        if g == "LU":
            continue
        pub = get(d, g, EU, item, side)
        cols[g] = pub.fillna(get(d, g, WRL, item, side) - ext[g])
    lu_ext = _lu_residual(eu_get(d, EXT, item, side), ext)
    cols["LU"] = get(d, "LU", WRL, item, side) - lu_ext
    return pd.DataFrame(cols)[list(MEMBERS)]


def intra_eu_positions(d: Data, item: str = F5) -> pd.DataFrame:
    """Each member's holdings of securities issued in other EU members, end of year; Luxembourg's are
    its total minus its non-EU holdings, which are the EU27 total minus the other 26 members."""
    return _intra_eu(d, "pos", item)


def intra_eu_flows(d: Data, item: str = F5) -> pd.DataFrame:
    """Each member's quarterly net purchases of securities issued in other EU members."""
    return _intra_eu(d, "flow", item)


# ---------------------------------------------------------------------------------------------
# Who holds the hubs' securities (positions)
# ---------------------------------------------------------------------------------------------

def holdings_of_hub(d: Data, hub: str, item: str = F5) -> pd.DataFrame:
    """Holdings of the hub's securities by each other member, end of year (NaN where unpublished)."""
    return pd.DataFrame({g: position(d, g, hub, item, "A_LE") for g in MEMBERS if g != hub})


def reporter_share(held: pd.DataFrame, intra: pd.DataFrame) -> pd.Series:
    """Weighted share of the hub in the intra-EU holdings of the members that publish both."""
    both = held.notna() & intra[held.columns].notna()
    return held.where(both).sum(axis=1) / intra[held.columns].where(both).sum(axis=1)


def hub_positions(d: Data, hub: str, item: str = F5) -> pd.DataFrame:
    """Non-resident holdings of the hub's securities split into EU holders (published and imputed)
    and the residual held outside the EU, with bounds, end of year, EUR million."""
    held = holdings_of_hub(d, hub, item)
    intra = intra_eu_positions(d, item)
    share = reporter_share(held, intra)
    missing = held.isna()
    miss_intra = intra[held.columns].where(missing)
    out = pd.DataFrame(index=held.index)
    out["nonresident"] = position(d, hub, WRL, item, "L_LE")
    out["eu_reported"] = held.sum(axis=1)
    out["eu_imputed"] = miss_intra.mul(share, axis=0).sum(axis=1)
    out["eu_missing_cap"] = miss_intra.clip(lower=0).sum(axis=1)
    out["n_missing"] = missing.sum(axis=1)
    out["reporter_share"] = share
    out["eu"] = out.eu_reported + out.eu_imputed
    out["non_eu"] = out.nonresident - out.eu
    out["non_eu_high"] = out.nonresident - out.eu_reported
    out["non_eu_low"] = out.nonresident - out.eu_reported - out.eu_missing_cap
    return out


def missing_holders(d: Data, item: str = F5) -> dict[str, pd.DataFrame]:
    """For each hub, the intra-EU holdings of the members whose bilateral cell is missing."""
    intra = intra_eu_positions(d, item)
    out = {}
    for h in HUBS:
        held = holdings_of_hub(d, h, item)
        out[h] = intra[held.columns].where(held.isna())
    return out


def combined_positions(d: Data, item: str = F5) -> pd.DataFrame:
    """Both hubs together; the lower bound counts each missing member's intra-EU holdings once."""
    parts = {h: hub_positions(d, h, item) for h in HUBS}
    out = sum(p[["nonresident", "eu_reported", "eu_imputed", "eu", "non_eu", "non_eu_high"]] for p in parts.values())
    miss = missing_holders(d, item)
    either = pd.concat(miss.values(), axis=1)
    cap = either.T.groupby(level=0).max().T.clip(lower=0).sum(axis=1)
    out["eu_missing_cap"] = cap
    out["non_eu_low"] = out.nonresident - out.eu_reported - cap
    return out


# ---------------------------------------------------------------------------------------------
# Who buys the hubs' securities (flows)
# ---------------------------------------------------------------------------------------------

def _own_share(d: Data, hub: str, item: str) -> pd.DataFrame:
    """Each member's share of the hub in its own intra-EU holdings at end of year (where published),
    otherwise the reporters' weighted share."""
    held = holdings_of_hub(d, hub, item)
    intra = intra_eu_positions(d, item)[held.columns]
    own = held / intra
    fill = reporter_share(held, intra_eu_positions(d, item))
    return own.apply(lambda col: col.fillna(fill)).clip(lower=0, upper=1)


def hub_flows(d: Data, hub: str, item: str = F5) -> pd.DataFrame:
    """Quarterly net purchases of the hub's securities by non-residents, by other EU members
    (published and imputed) and by investors outside the EU, with bounds, EUR million.

    A missing bilateral flow is imputed as the member's intra-EU net purchases in the quarter times
    the hub's share of its intra-EU holdings at the end of the previous year."""
    q = quarters()
    held = pd.DataFrame({g: flow(d, g, hub, item, "ASS") for g in MEMBERS if g != hub})
    intra = intra_eu_flows(d, item)[held.columns]
    share_y = _own_share(d, hub, item)
    prev = pd.Index(q).str[:4].astype(int) - 1
    share_q = share_y.reindex(prev).set_axis(q)
    missing = held.isna()
    miss_intra = intra.where(missing)
    out = pd.DataFrame(index=q)
    out["nonresident"] = flow(d, hub, WRL, item, "LIAB")
    out["eu_reported"] = held.sum(axis=1)
    out["eu_imputed"] = (miss_intra * share_q).sum(axis=1)
    out["eu_missing_neg"] = miss_intra.clip(upper=0).sum(axis=1)
    out["eu_missing_pos"] = miss_intra.clip(lower=0).sum(axis=1)
    out["n_missing"] = missing.sum(axis=1)
    out["eu"] = out.eu_reported + out.eu_imputed
    out["non_eu"] = out.nonresident - out.eu
    out["non_eu_high"] = out.nonresident - out.eu_reported - out.eu_missing_neg
    out["non_eu_low"] = out.nonresident - out.eu_reported - out.eu_missing_pos
    return out


def combined_flows(d: Data, item: str = F5) -> pd.DataFrame:
    """Both hubs; a member missing for both is bounded once by its intra-EU net purchases."""
    q = quarters()
    parts = {h: hub_flows(d, h, item) for h in HUBS}
    out = sum(p[["nonresident", "eu_reported", "eu_imputed", "eu", "non_eu"]] for p in parts.values())
    miss = {}
    for h in HUBS:
        held = pd.DataFrame({g: flow(d, g, h, item, "ASS") for g in MEMBERS if g != h})
        miss[h] = intra_eu_flows(d, item)[held.columns].where(held.isna())
    either = pd.concat(miss.values(), axis=1).T.groupby(level=0).max().T.reindex(q)
    out["non_eu_high"] = out.nonresident - out.eu_reported - either.clip(upper=0).sum(axis=1)
    out["non_eu_low"] = out.nonresident - out.eu_reported - either.clip(lower=0).sum(axis=1)
    return out


# ---------------------------------------------------------------------------------------------
# Ownership of the hubs' funds and attribution of their US purchases
# ---------------------------------------------------------------------------------------------

def fund_shares_outstanding(d: Data, hub: str) -> pd.Series:
    """Investment fund shares issued by residents of the hub, held by anyone, end of year (financial
    balance sheets, total economy, liabilities)."""
    return _get(d.fin, (hub, "S1", "LIAB", "F52"), years())


def company_share_ratio(d: Data, hub: str) -> pd.Series:
    """Share of company shares (as opposed to fund shares) in other members' holdings of the hub's
    equity, among members that publish both."""
    a51 = pd.DataFrame({g: position(d, g, hub, F51, "A_LE") for g in MEMBERS if g != hub})
    a5 = pd.DataFrame({g: position(d, g, hub, F5, "A_LE") for g in MEMBERS if g != hub})
    both = a51.notna() & a5.notna()
    return a51.where(both).sum(axis=1) / a5.where(both).sum(axis=1)


def non_eu_fund_shares(d: Data, hub: str, bound: str = "central", correct_companies: bool = True) -> pd.Series:
    """Fund shares of the hub held outside the EU, end of year.

    Holdings of the hub's equity and fund shares outside the EU (``hub_positions``) include the
    shares of companies resident in the hub. Those are removed by subtracting the hub's company-share
    liabilities net of the part EU members hold."""
    hp = hub_positions(d, hub, F5)
    col = {"central": "non_eu", "low": "non_eu_low", "high": "non_eu_high"}[bound]
    nonres = hp[col]
    if not correct_companies:
        return nonres
    l51 = position(d, hub, WRL, F51, "L_LE")
    eu51 = company_share_ratio(d, hub) * hp.eu
    return nonres - (l51 - eu51).clip(lower=0)


def non_eu_ownership(d: Data, hub: str, bound: str = "central", correct_companies: bool = True,
                     denominator: str = "all") -> pd.Series:
    """Share of the hub's fund shares owned outside the EU, end of year.

    ``denominator='all'`` divides by fund shares held by anyone, residents of the hub included;
    ``'nonresident'`` divides by fund shares held by non-residents only."""
    num = non_eu_fund_shares(d, hub, bound, correct_companies)
    if denominator == "all":
        den = fund_shares_outstanding(d, hub)
    else:
        den = position(d, hub, WRL, F52, "L_LE")
    if not correct_companies:
        den = den + position(d, hub, WRL, F51, "L_LE")
    return (num / den).clip(0, 1)


def ownership_low_joint(d: Data, us_weights: pd.DataFrame) -> pd.DataFrame:
    """Lower bound of the non-EU ownership shares of both hubs taken together: each member missing
    for both hubs has its whole intra-EU holdings assigned to the hub where it lowers the non-EU
    attribution most (the hub with more US securities per fund share)."""
    miss = missing_holders(d, F5)
    cen = {h: hub_positions(d, h, F5) for h in HUBS}
    den = {h: fund_shares_outstanding(d, h) for h in HUBS}
    extra = {h: pd.Series(0.0, index=years()) for h in HUBS}
    w = pd.DataFrame({h: us_weights[h] / den[h] for h in HUBS})
    for g in MEMBERS:
        cols = [h for h in HUBS if g in miss[h].columns and g != h]
        vals = {h: miss[h][g].clip(lower=0) for h in cols}
        for y in years():
            avail = [h for h in cols if pd.notna(vals[h].get(y))]
            if not avail:
                continue
            best = max(avail, key=lambda h: w.loc[y, h] if pd.notna(w.loc[y, h]) else -1)
            extra[best].loc[y] += vals[best].loc[y]
    out = {}
    for h in HUBS:
        nonres = cen[h].nonresident - cen[h].eu_reported - extra[h]
        l51 = position(d, h, WRL, F51, "L_LE")
        eu51 = company_share_ratio(d, h) * (cen[h].eu_reported + extra[h])
        out[h] = ((nonres - (l51 - eu51).clip(lower=0)) / den[h]).clip(0, 1)
    return pd.DataFrame(out)


def attribution(us_hub: pd.DataFrame, omega: pd.DataFrame) -> pd.Series:
    """Hub purchases or holdings of US securities times the non-EU ownership share, summed over hubs."""
    return (us_hub[list(HUBS)] * omega[list(HUBS)]).sum(axis=1, min_count=len(HUBS))


def mid_year(omega: pd.DataFrame) -> pd.DataFrame:
    """Average of the start- and end-of-year ownership shares, for weighting a year's flows."""
    return (omega + omega.shift(1)) / 2


# ---------------------------------------------------------------------------------------------
# Co-movement
# ---------------------------------------------------------------------------------------------

def ols_hac(y: np.ndarray, x: np.ndarray, lags: int = 4) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS with an intercept and Newey-West standard errors. Returns (coef, se, r2); the intercept is
    the first element."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(x, float)])
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ X.T @ y
    u = y - X @ b
    s = (X * u[:, None]).T @ (X * u[:, None])
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        g = (X[lag:] * u[lag:, None]).T @ (X[:-lag] * u[:-lag, None])
        s += w * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    r2 = 1 - (u @ u) / ((y - y.mean()) @ (y - y.mean()))
    return b, np.sqrt(np.diag(cov)), float(r2)
