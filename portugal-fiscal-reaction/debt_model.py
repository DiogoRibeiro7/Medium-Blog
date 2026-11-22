"""Annual debt-ratio model for Portugal, 2027-2045, with a fiscal policy rule.

Same macro block as portugal-debt-2035-probability. All ratios in % of GDP, rates in %.

    gap_t  = rho_gap * gap_{t-1} + (1 - h) * s_t
    y_t    = pot_t + h * s_t + (gap_t - gap_{t-1})
    pi_t   = pi_base_t + kappa * gap_t + v_t
    g_t    = (1 + y_t)(1 + pi_t) - 1
    i_t    = (1 - s) * i_{t-1} + s * m_t
    int_t  = i_t * d_{t-1} / (1 + g_t)
    d_t    = d_{t-1} / (1 + g_t) + int_t - pb_t + sfa_t

Primary balance:
    fixed policy  pb_t = pbs_t + eps * gap_t - disc_t
    rule policy   pb_t = a + rho * d_{t-1} + gamma * gap_t + phi * pb_{t-1} + e_t - disc_t   (from 2028)

In both, if the headline deficit exceeds 3% of GDP the primary balance is raised by a further
0.5 pp a year from the following year (minimum adjustment under an excessive deficit procedure).
The 2027 primary balance is the 2027 budget's 2.5% of GDP under every policy.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pandas as pd

YEARS = list(range(2027, 2046))
D0 = 87.5  # end-2026 debt, Government target in the 2027 budget (Commission autumn 2026: 86.5)
I0 = 2.35  # effective interest rate in 2026
PB_2027 = 2.5  # primary surplus in the 2027 budget
SFA_2027, SFA_LATER = 1.06, 0.30


def potential_growth(year: int) -> float:
    path = {2027: 2.1, 2028: 2.1, 2029: 1.8, 2030: 1.6}
    return path.get(year, 1.3 if year <= 2035 else 1.1)


def marginal_rate(year: int) -> float:
    return min(3.4 + 0.08 * (year - 2027), 3.8)


def base_deflator(year: int) -> float:
    return 2.2 if year == 2027 else 2.0


@dataclass
class Policy:
    key: str
    label: str
    kind: str = "fixed"  # "fixed" or "rule"
    pbs: float = 2.5  # fixed primary surplus from 2028
    a: float = 0.0
    rho: float = 0.0
    gamma: float = 0.5
    phi: float = 0.0
    eps: float = 0.5  # automatic stabiliser under the fixed policy
    shock: dict = field(default_factory=dict)
    spread: dict = field(default_factory=dict)
    disc: dict = field(default_factory=dict)
    sfa_extra: dict = field(default_factory=dict)
    rho_gap: float = 0.6
    h: float = 0.3
    kappa: float = 0.3
    s_reprice: float = 0.14
    edp: bool = True

    def copy(self, **changes) -> Policy:
        out = copy.deepcopy(self)
        for k, v in changes.items():
            setattr(out, k, v)
        return out

    def long_run_pb(self, debt: float) -> float:
        """Primary balance the rule settles at for a given debt ratio, closed output gap."""
        return (self.a + self.rho * debt) / (1 - self.phi)


def anchored_intercept(rho: float, phi: float, debt: float, pb: float) -> float:
    """Intercept that makes the rule's long-run primary balance equal pb at the given debt."""
    return pb * (1 - phi) - rho * debt


def run(p: Policy, d0: float = D0, i0: float = I0, draws: dict | None = None) -> pd.DataFrame:
    """draws: optional {year: (u, v, dm, dsfa, e)} historical shock vectors (Monte Carlo)."""
    d, i, gap, mdev, edp_add, pb_prev = d0, i0, 0.0, 0.0, 0.0, PB_2027
    rows = []
    for y in YEARS:
        s = p.shock.get(y, 0.0)
        v = dm = dsfa = e = 0.0
        if draws is not None:
            u, v, dm, dsfa, e = draws[y]
            s += u
        gap_new = p.rho_gap * gap + (1 - p.h) * s
        scar = p.h * (min(s, 0.0) if draws is not None else s)
        yr = potential_growth(y) + scar + (gap_new - gap)
        pi = base_deflator(y) + p.kappa * gap_new + v
        g = (1 + yr / 100) * (1 + pi / 100) - 1
        mdev = 0.7 * mdev + dm
        m = marginal_rate(y) + p.spread.get(y, 0.0) + mdev
        i = (1 - p.s_reprice) * i + p.s_reprice * m
        intr = i / 100 * d / (1 + g)
        if y == 2027:
            pb = PB_2027 + p.eps * gap_new
        elif p.kind == "rule":
            pb = p.a + p.rho * d + p.gamma * gap_new + p.phi * pb_prev + e
        else:
            pb = p.pbs + p.eps * gap_new
        pb += edp_add - p.disc.get(y, 0.0)
        bal = pb - intr
        sfa = (SFA_2027 if y == 2027 else SFA_LATER) + p.sfa_extra.get(y, 0.0) + dsfa
        d_new = d / (1 + g) + intr - pb + sfa
        rows.append(
            dict(
                year=y,
                real_growth=yr,
                nominal_growth=100 * g,
                gap=gap_new,
                effective_rate=i,
                interest=intr,
                primary_balance=pb,
                balance=bal,
                sfa=sfa,
                debt=d_new,
                edp_adjustment=edp_add,
            )
        )
        if p.edp and bal < -3.0:
            edp_add += 0.5
        d, gap, pb_prev = d_new, gap_new, pb
    return pd.DataFrame(rows).set_index("year")


def steady_state_debt(p: Policy, i_lr: float, g_lr: float, sfa: float = SFA_LATER) -> float:
    """Debt ratio at which the rule's long-run primary balance just stabilises debt."""
    slope_req = (i_lr - g_lr) / (100 + g_lr)  # debt-stabilising pb per point of debt
    # (a + rho d)/(1 - phi) = slope_req * d + sfa
    lr_slope = p.rho / (1 - p.phi)
    lr_level = p.a / (1 - p.phi)
    return (sfa - lr_level) / (lr_slope - slope_req)
