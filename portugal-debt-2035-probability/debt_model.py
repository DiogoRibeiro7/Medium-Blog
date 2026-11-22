"""A small annual model of Portugal's debt ratio, 2027-2045.

All ratios are in % of GDP, all rates in %.

    gap_t  = rho * gap_{t-1} + (1 - h) * s_t            output gap; s_t is a demand shock
    y_t    = pot_t + h * s_t + (gap_t - gap_{t-1})       real growth; h is the scarring share
    pi_t   = pi_base_t + kappa * gap_t + v_t             GDP deflator
    g_t    = (1 + y_t)(1 + pi_t) - 1                     nominal growth
    i_t    = (1 - s) * i_{t-1} + s * m_t                 effective rate; s is the repricing share
    int_t  = i_t * d_{t-1} / (1 + g_t)                   interest bill
    pb_t   = pbs_t + eps * gap_t - disc_t                primary balance
    bal_t  = pb_t - int_t                                headline balance
    d_t    = d_{t-1} / (1 + g_t) + int_t - pb_t + sfa_t  debt ratio

If the headline deficit exceeds 3% of GDP, the structural primary balance is raised by
0.5 pp a year from the following year, the minimum adjustment under an excessive
deficit procedure.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pandas as pd

YEARS = list(range(2027, 2046))

# Starting point, end-2026: Government target in the 2027 budget (Commission autumn 2026: 86.5).
D0 = 87.5
# Effective interest rate in 2026: 2.0% of GDP interest bill on end-2025 debt of 89.2% of GDP.
I0 = 2.35
# Stock-flow adjustment: 2027 set so the 2027 debt ratio equals the budget's 84.5%;
# afterwards close to the 1996-2025 mean (0.34 pp).
SFA_2027, SFA_LATER = 1.06, 0.30


def potential_growth(year: int) -> float:
    path = {2027: 2.1, 2028: 2.1, 2029: 1.8, 2030: 1.6}
    return path.get(year, 1.3 if year <= 2035 else 1.1)


def marginal_rate(year: int) -> float:
    """Average cost of new debt: about 3.4% on 2026 issuance, drifting to 3.8%."""
    return min(3.4 + 0.08 * (year - 2027), 3.8)


def base_deflator(year: int) -> float:
    return 2.2 if year == 2027 else 2.0


@dataclass
class Scenario:
    key: str
    label: str
    pbs: dict
    shock: dict = field(default_factory=dict)
    spread: dict = field(default_factory=dict)
    disc: dict = field(default_factory=dict)
    sfa_extra: dict = field(default_factory=dict)
    eps: float = 0.5
    rho: float = 0.6
    h: float = 0.3
    kappa: float = 0.3
    s_reprice: float = 0.14
    edp: bool = True

    def copy(self, **changes) -> Scenario:
        out = copy.deepcopy(self)
        for k, v in changes.items():
            setattr(out, k, v)
        return out


def flat(level: float) -> dict:
    return {y: level for y in YEARS}


def drift(start: float = 2.5, end: float = 1.8, by: int = 2029) -> dict:
    out = {}
    for y in YEARS:
        if y <= 2027:
            out[y] = start
        elif y >= by:
            out[y] = end
        else:
            out[y] = start + (end - start) * (y - 2027) / (by - 2027)
    return out


def run(sc: Scenario, d0: float = D0, i0: float = I0, draws: dict | None = None) -> pd.DataFrame:
    """draws: optional {year: (u, v, dm, dsfa)} historical shock vectors (Monte Carlo)."""
    d, i, gap, mdev, edp_add = d0, i0, 0.0, 0.0, 0.0
    rows = []
    for y in YEARS:
        s = sc.shock.get(y, 0.0)
        v = dm = dsfa = 0.0
        if draws is not None:
            u, v, dm, dsfa = draws[y]
            s += u
        gap_new = sc.rho * gap + (1 - sc.h) * s
        scar = sc.h * (min(s, 0.0) if draws is not None else s)
        yr = potential_growth(y) + scar + (gap_new - gap)
        pi = base_deflator(y) + sc.kappa * gap_new + v
        g = (1 + yr / 100) * (1 + pi / 100) - 1
        mdev = 0.7 * mdev + dm
        m = marginal_rate(y) + sc.spread.get(y, 0.0) + mdev
        i = (1 - sc.s_reprice) * i + sc.s_reprice * m
        intr = i / 100 * d / (1 + g)
        pb = sc.pbs[y] + edp_add + sc.eps * gap_new - sc.disc.get(y, 0.0)
        bal = pb - intr
        sfa = (SFA_2027 if y == 2027 else SFA_LATER) + sc.sfa_extra.get(y, 0.0) + dsfa
        snowball = intr - d * g / (1 + g)
        d = d / (1 + g) + intr - pb + sfa
        rows.append(
            dict(
                year=y,
                real_growth=yr,
                deflator=pi,
                nominal_growth=100 * g,
                gap=gap_new,
                marginal_rate=m,
                effective_rate=i,
                interest=intr,
                primary_balance=pb,
                balance=bal,
                sfa=sfa,
                snowball=snowball,
                debt=d,
                edp_adjustment=edp_add,
            )
        )
        if sc.edp and bal < -3.0:
            edp_add += 0.5
        gap = gap_new
    return pd.DataFrame(rows).set_index("year")


def scenarios(pb_level: float = 2.5) -> dict[str, Scenario]:
    hold = flat(pb_level)
    return {
        "A": Scenario("A", f"2027 budget stance held (primary surplus {pb_level:.1f}%)", hold),
        "B": Scenario("B", "Drift to the Commission's 1.8% structural primary surplus", drift()),
        "R1": Scenario("R1", "Mild recession in 2027 (2003-type)", hold, shock={2027: -3.0}),
        "R2": Scenario(
            "R2",
            "Severe recession in 2027 (2009-type)",
            hold,
            shock={2027: -5.5},
            spread={2027: 1.0, 2028: 1.0},
            disc={2027: 1.0, 2028: 0.5},
            sfa_extra={2027: 1.0},
        ),
        "R3": Scenario(
            "R3",
            "Severe recession with bond-market stress (2009-12-type)",
            hold,
            shock={2027: -5.5, 2028: -3.0},
            spread={2027: 2.5, 2028: 2.5, 2029: 2.5, 2030: 1.0},
            disc={2027: 1.0},
            sfa_extra={2027: 1.5, 2028: 1.0},
        ),
    }


def shift_scenario(sc: Scenario, start: int) -> Scenario:
    """Move a recession scenario so that its first shock year is `start`."""
    lag = start - 2027

    def mv(dct):
        return {y + lag: v for y, v in dct.items() if y + lag in YEARS}

    return sc.copy(shock=mv(sc.shock), spread=mv(sc.spread), disc=mv(sc.disc), sfa_extra=mv(sc.sfa_extra))


def required_primary_surplus(sc: Scenario, target: float = 60.0, year: int = 2035) -> float:
    """Constant primary surplus from 2028 that brings debt to `target` in `year`."""
    lo, hi = -2.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        pbs = {y: (sc.pbs[y] if y == 2027 else mid) for y in YEARS}
        if run(sc.copy(pbs=pbs)).loc[year, "debt"] > target:
            lo = mid
        else:
            hi = mid
    return hi
