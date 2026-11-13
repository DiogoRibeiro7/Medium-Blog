"""Building blocks for the category-level test of the PPP drift inside Europe.

The drift of economy i in category c over [a, b], against the EU27 as reference, in log points x 100:

    drift_ic = 100 * [ ln(P_ic(b) / P_ic(a)) - ln(PPP_ic(b) / PPP_ic(a)) - ln(P_EU,c(b) / P_EU,c(a)) ]

with P the implicit deflator of household consumption in category c (national currency), PPP the
Eurostat purchasing power parity for the same category (national currency per PPS, EU27 = 1) and
P_EU,c the EU27 deflator in euro. Against any other reference the drift changes by a constant across
economies, so the slope of the drift on the 2005 price level does not depend on the reference.

The household-consumption drift splits exactly into the twelve COICOP divisions, weighted by each
economy's average expenditure shares in a and b, plus three aggregation terms (see ``decompose``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EU27 = ("AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT",
        "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK")
OTHER = ("CH", "NO", "IS", "RS", "AL", "BA")
EUROPE = EU27 + OTHER            # every Eurostat-group economy with household deflators by purpose
BALKANS = ("RS", "AL", "BA")
REF = "EU27_2020"
START, END = 2005, 2019

# (PPP category, COICOP code, label, kind). Kinds: goods, services, mixed, and "public" for health and
# education, whose PPPs cover government provision while the deflators cover household purchases only.
DIVISIONS = [
    ("A0101", "CP01", "Food and non-alcoholic beverages", "goods"),
    ("A0102", "CP02", "Alcoholic beverages and tobacco", "goods"),
    ("A0103", "CP03", "Clothing and footwear", "goods"),
    ("A0104", "CP04", "Housing, water, electricity, gas", "services"),
    ("A0105", "CP05", "Furnishings and household equipment", "goods"),
    ("A0106", "CP06", "Health", "public"),
    ("A0107", "CP07", "Transport", "mixed"),
    ("A0108", "CP08", "Communication", "services"),
    ("A0109", "CP09", "Recreation and culture", "mixed"),
    ("A0110", "CP10", "Education", "public"),
    ("A0111", "CP11", "Restaurants and hotels", "services"),
    ("A0112", "CP12", "Miscellaneous goods and services", "services"),
]
GROUPS = [
    ("A010101", "CP011", "Food", "goods"),
    ("A010201", "CP021", "Alcoholic beverages", "goods"),
    ("A010202", "CP022", "Tobacco", "goods"),
    ("A010301", "CP031", "Clothing", "goods"),
    ("A010302", "CP032", "Footwear", "goods"),
    ("A010405", "CP045", "Electricity, gas and other fuels", "goods"),
    ("A010501", "CP051", "Furniture and floor coverings", "goods"),
    ("A010503", "CP053", "Household appliances", "goods"),
    ("A010701", "CP071", "Purchase of vehicles", "goods"),
    ("A010901", "CP091", "Audio-visual and IT equipment", "goods"),
    ("A010703", "CP073", "Transport services", "services"),
]
# Eurostat's own goods and services split of household consumption: PPP category -> durability items
DURABILITY = {
    "P0101": ("Consumer goods", ("P311_S14", "P312_S14", "P313_S14")),
    "P010101": ("Non-durable goods", ("P313_S14",)),
    "P010102": ("Semi-durable goods", ("P312_S14",)),
    "P010103": ("Durable goods", ("P311_S14",)),
    "P0201": ("Consumer services", ("P314_S14",)),
}


@dataclass
class Data:
    ppp: pd.Series      # index (geo, ppp_cat, na_item, year)
    defl: pd.Series     # index (geo, coicop, unit, year)
    fcs: pd.Series      # index (geo, na_item, unit, year)


def load(data_dir: Path | str = "data") -> Data:
    d = Path(data_dir)
    p = pd.read_csv(d / "ppp.csv", low_memory=False)
    q = pd.read_csv(d / "deflators.csv", low_memory=False)
    f = pd.read_csv(d / "durability.csv", low_memory=False)
    return Data(
        ppp=p.set_index(["geo", "ppp_cat", "na_item", "time"]).value.astype(float).sort_index(),
        defl=q.set_index(["geo", "coicop", "unit", "time"]).value.astype(float).sort_index(),
        fcs=f.set_index(["geo", "na_item", "unit", "time"]).value.astype(float).sort_index(),
    )


def _get(s: pd.Series, key: tuple) -> pd.Series:
    try:
        return s.loc[key]
    except KeyError:
        return pd.Series(dtype=float)


def ppp(d: Data, geo: str, cat: str, item: str = "PPP_EU27_2020") -> pd.Series:
    return _get(d.ppp, (geo, cat, item))


def deflator(d: Data, geo: str, coicop: str) -> pd.Series:
    unit = "PD20_EUR" if geo == REF else "PD20_NAC"
    return _get(d.defl, (geo, coicop, unit))


def nominal(d: Data, geo: str, coicop: str) -> pd.Series:
    return _get(d.defl, (geo, coicop, "CP_MNAC"))


def tornqvist(parts: list[tuple[pd.Series, pd.Series]]) -> pd.Series:
    """Chained Törnqvist price index (first common year = 1) from (nominal, price) pairs."""
    common = sorted(set.intersection(*[set(pd.concat([c, p], axis=1).dropna().index) for c, p in parts]))
    total = sum(c.loc[common] for c, _ in parts)
    out = pd.Series(0.0, index=common)
    for t0, t1 in zip(common[:-1], common[1:]):
        out[t1] = out[t0] + sum(0.5 * (c[t0] / total[t0] + c[t1] / total[t1]) * np.log(p[t1] / p[t0])
                                for c, p in parts)
    return np.exp(out)


def durability_deflator(d: Data, geo: str, items: tuple[str, ...]) -> pd.Series:
    """Deflator of a durability aggregate: current prices over chain-linked volumes per item, chained
    as a Törnqvist index across items."""
    cur, vol = ("CP_MEUR", "CLV20_MEUR") if geo == REF else ("CP_MNAC", "CLV20_MNAC")
    parts = [(_get(d.fcs, (geo, i, cur)), _get(d.fcs, (geo, i, cur)) / _get(d.fcs, (geo, i, vol))) for i in items]
    return tornqvist(parts)


def log_change(s: pd.Series, a: int, b: int) -> float:
    try:
        return float(np.log(s[b] / s[a]))
    except (KeyError, ZeroDivisionError):
        return np.nan


def price_level(d: Data, year: int = START, cat: str = "E011", geos=EUROPE) -> pd.Series:
    """Log price level index (EU27 = 100) of a category in a year."""
    return np.log(pd.Series({g: ppp(d, g, cat, "PLI_EU27_2020").get(year, np.nan) for g in geos}))


def drift_parts(d: Data, cat: str, coicop: str | None = None, a: int = START, b: int = END,
                geos=EUROPE) -> pd.DataFrame:
    """Drift of each economy in one category and its two parts: the change in the deflator relative
    to the EU27 ('deflator') and the change in the PPP ('ppp'); drift = deflator - ppp."""
    if cat in DURABILITY:
        items = DURABILITY[cat][1]
        ref = log_change(durability_deflator(d, REF, items), a, b)
        defl = {g: log_change(durability_deflator(d, g, items), a, b) for g in geos}
    else:
        ref = log_change(deflator(d, REF, coicop), a, b)
        defl = {g: log_change(deflator(d, g, coicop), a, b) for g in geos}
    out = pd.DataFrame({
        "deflator": 100 * (pd.Series(defl) - ref),
        "ppp": 100 * pd.Series({g: log_change(ppp(d, g, cat), a, b) for g in geos}),
    })
    out["drift"] = out.deflator - out.ppp
    return out


def ols(y: pd.Series, x: pd.Series) -> dict:
    """Slope of y on x with an intercept; HC1 standard error; correlation; n."""
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    n = len(df)
    X = np.column_stack([np.ones(n), df.x.values])
    beta = np.linalg.lstsq(X, df.y.values, rcond=None)[0]
    u = df.y.values - X @ beta
    xtx = np.linalg.inv(X.T @ X)
    cov = xtx @ ((X * (u ** 2)[:, None]).T @ X) @ xtx * n / (n - 2)
    se = float(np.sqrt(cov[1, 1]))
    return {"slope": float(beta[1]), "se": se, "t": float(beta[1] / se), "r": float(np.corrcoef(df.x, df.y)[0, 1]),
            "n": n, "intercept": float(beta[0])}


def decompose(d: Data, a: int = START, b: int = END, geos=EUROPE) -> pd.DataFrame:
    """Exact split of each economy's household-consumption drift into share-weighted division drifts
    and three aggregation terms:

        drift_H = sum_c s_c * drift_c + A - B - C

    A = change in the household deflator minus the share-weighted change in the division deflators;
    B = change in the household PPP minus the share-weighted change in the division PPPs;
    C = change in the EU27 household deflator minus the change in the EU27 division deflators
        weighted with the economy's own shares.
    s_c is the economy's average share of division c in a and b (current prices). The columns 'B_ppp'
    and 'C_ref' hold -B and -C, so all columns add up to 'drift'."""
    rows = {}
    ref = {co: log_change(deflator(d, REF, co), a, b) for _, co, _, _ in DIVISIONS}
    ref_h = log_change(deflator(d, REF, "TOTAL"), a, b)
    for g in geos:
        cp = {co: nominal(d, g, co) for _, co, _, _ in DIVISIONS}
        try:
            tot = {t: sum(cp[co][t] for co in cp) for t in (a, b)}
        except KeyError:
            continue
        s = {co: 0.5 * (cp[co][a] / tot[a] + cp[co][b] / tot[b]) for co in cp}
        dp = {co: log_change(deflator(d, g, co), a, b) for co in cp}
        dq = {co: log_change(ppp(d, g, cat), a, b) for cat, co, _, _ in DIVISIONS}
        dph, dqh = log_change(deflator(d, g, "TOTAL"), a, b), log_change(ppp(d, g, "E011"), a, b)
        r = {"drift": 100 * (dph - dqh - ref_h)}
        for co in cp:
            r[co] = 100 * s[co] * (dp[co] - dq[co] - ref[co])
        r["A_deflator"] = 100 * (dph - sum(s[co] * dp[co] for co in cp))
        r["B_ppp"] = -100 * (dqh - sum(s[co] * dq[co] for co in cp))
        r["C_reference"] = -100 * (ref_h - sum(s[co] * ref[co] for co in cp))
        for co in cp:
            r[f"share_{co}"] = s[co]
        rows[g] = r
    return pd.DataFrame(rows).T


def composition_term(d: Data, a: int = START, b: int = END, geos=EUROPE) -> pd.Series:
    """Change in each economy's division shares times the division's log price relative to the
    economy's household-consumption PPP (average of a and b), with a minus sign: positive when spending
    moves towards divisions that are cheap relative to the economy's average. If household PPPs were
    Törnqvist aggregates of division PPPs with the economy's own weights, this would be the part of
    -B that comes from changing weights."""
    out = {}
    for g in geos:
        cp = {co: nominal(d, g, co) for _, co, _, _ in DIVISIONS}
        try:
            tot = {t: sum(cp[co][t] for co in cp) for t in (a, b)}
            rel = {co: 0.5 * sum(np.log(ppp(d, g, cat)[t] / ppp(d, g, "E011")[t]) for t in (a, b))
                   for cat, co, _, _ in DIVISIONS}
        except KeyError:
            continue
        out[g] = -100 * sum((cp[co][b] / tot[b] - cp[co][a] / tot[a]) * rel[co] for co in cp)
    return pd.Series(out)


def relative_price_spread(d: Data, year: int = START, geos=EUROPE) -> pd.Series:
    """Expenditure-weighted standard deviation of division price levels relative to the economy's
    household-consumption price level, in log points."""
    out = {}
    for g in geos:
        cp = {co: nominal(d, g, co).get(year, np.nan) for _, co, _, _ in DIVISIONS}
        tot = sum(cp.values())
        rel = {co: np.log(ppp(d, g, cat).get(year, np.nan) / ppp(d, g, "E011").get(year, np.nan))
               for cat, co, _, _ in DIVISIONS}
        w = np.array([cp[co] / tot for co in cp])
        v = np.array([rel[co] for co in cp])
        out[g] = float(100 * np.sqrt(np.sum(w * (v - np.sum(w * v)) ** 2)))
    return pd.Series(out)


def housing_ex_energy(d: Data, a: int = START, b: int = END, geos=EUROPE) -> pd.DataFrame:
    """Drift of housing without electricity, gas and fuels (mostly actual and imputed rents). Both the
    deflator and the PPP of the remainder are backed out of the division and the energy group as if the
    division were a geometric mean of the two with the economy's average energy share in a and b. This
    is an approximation: neither the chain-linked deflator nor the EKS-aggregated PPP is exactly a
    geometric mean of its parts."""
    def share(geo: str) -> float:
        c4, c45 = nominal(d, geo, "CP04"), nominal(d, geo, "CP045")
        try:
            return 0.5 * sum(c45[t] / c4[t] for t in (a, b))
        except KeyError:
            return np.nan

    def rest(total: float, energy: float, s: float) -> float:
        return (total - s * energy) / (1 - s)

    s_ref = share(REF)
    ref = rest(log_change(deflator(d, REF, "CP04"), a, b), log_change(deflator(d, REF, "CP045"), a, b), s_ref)
    rows = {}
    for g in geos:
        s = share(g)
        if np.isnan(s):
            continue
        dp = rest(log_change(deflator(d, g, "CP04"), a, b), log_change(deflator(d, g, "CP045"), a, b), s)
        dq = rest(log_change(ppp(d, g, "A0104"), a, b), log_change(ppp(d, g, "A010405"), a, b), s)
        rows[g] = {"deflator": 100 * (dp - ref), "ppp": 100 * dq, "energy_share": s}
    out = pd.DataFrame(rows).T
    out["drift"] = out.deflator - out.ppp
    return out


def spearman(x: pd.Series, y: pd.Series) -> float:
    df = pd.concat([x, y], axis=1).dropna()
    return float(df.rank().corr().iloc[0, 1])
