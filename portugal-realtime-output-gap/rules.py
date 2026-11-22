"""Portugal's debt ratio, 2027-2045, under alternative fiscal rules, vectorised over simulated futures.

The macro block is the one used in portugal-debt-2035-probability and portugal-fiscal-reaction.
All ratios are in % of GDP and all rates in %. For each future and year t:

    gap_t  = rho_gap * gap_{t-1} + (1 - h) * s_t - mu * (sp_{t-1} - sp_{t-2})
    y_t    = pot_t + h * min(s_t, 0) + (gap_t - gap_{t-1})
    pi_t   = pi_base_t + kappa * gap_t + v_t
    g_t    = (1 + y_t)(1 + pi_t) - 1
    i_t    = (1 - s) * i_{t-1} + s * m_t,  m_t = marginal_t + mdev_t,  mdev_t = 0.7 mdev_{t-1} + dm_t
    int_t  = i_t * d_{t-1} / (1 + g_t)
    d_t    = d_{t-1} / (1 + g_t) + int_t - pb_t + sfa_t

sp is the structural primary balance, pb - one-offs - 0.5 * gap; mu is the fiscal multiplier,
applied with a one-year lag. Rules that use the output gap see it in real time:

    gap_rt = gap_t - gap_miss * (gap_t - rho_gap * gap_{t-1}) + gap_bias + nu_t,
    nu_t   = gap_noise_rho * nu_{t-1} + gap_noise_sd * sqrt(1 - gap_noise_rho^2) * z_t,

so they miss a share of the year's innovation, carry a bias, and add persistent noise. The
parameters are estimated from the Commission's archived releases in run_analysis.py.

The 2027 primary balance is the 2027 budget's 2.5% of GDP under every rule. From 2028 each rule
sets the primary balance:

    fixed       pb = level + eps * gap_rt
    debt        pb = level + kappa_d * (d_{t-1} - d_ref) + eps * gap_rt
    structural  pb = level + int_t + eps * gap_rt          (cyclically adjusted headline balance)
    spending    pb = R_t - E_t,  R_t = R + eta * gap_t,  E_t = E_{t-1} (1 + x) / (1 + g_t)

where x, the growth of nominal primary spending, is fixed for each plan of plan_years years. At the
start of each plan (anchor=True) x is set so that, with no further shocks, the structural primary
balance reaches `level` + kappa_d * (d - d_ref) by the plan's last year, with d the debt ratio when
the plan is set; within the plan nothing is revised except, if indexed, for realised inflation.
With anchor=False, x is expected nominal potential growth minus `level` and the level of spending
is never re-anchored.

Every rule then adds the year's fiscal surprise e_t (one-off operations, bank support and other
misses), optionally repays past shortfalls over `amortise` years (each negative surprise in equal
instalments, positive surprises credited against them first, as in the Swiss amortisation
account), and adds the corrective arm: 0.5 pp a year more from the year after each headline
deficit above 3% of GDP.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

YEARS = list(range(2027, 2046))
H = len(YEARS)
D0 = 87.5  # end-2026 debt, Government target in the 2027 budget
I0 = 2.35  # effective interest rate in 2026
PB_2027 = 2.5  # primary surplus in the 2027 budget
SFA_2027, SFA_LATER = 1.06, 0.30
REVENUE = 43.0  # general government revenue, % of GDP (AMECO, 2025)
SEMI_ELASTICITY = 0.5  # response of the primary balance to the output gap
D_REF = 84.5  # debt ratio in the 2027 budget


def potential_growth(year: int) -> float:
    path = {2027: 2.1, 2028: 2.1, 2029: 1.8, 2030: 1.6}
    return path.get(year, 1.3 if year <= 2035 else 1.1)


def marginal_rate(year: int) -> float:
    return min(3.4 + 0.08 * (year - 2027), 3.8)


def base_deflator(year: int) -> float:
    return 2.2 if year == 2027 else 2.0


def expected_nominal_growth(year: int) -> float:
    return ((1 + potential_growth(year) / 100) * (1 + base_deflator(year) / 100) - 1) * 100


@dataclass(frozen=True)
class Macro:
    rho_gap: float = 0.6
    h: float = 0.3
    kappa: float = 0.3
    s_reprice: float = 0.14
    mu: float = 0.5  # fiscal multiplier on the structural primary balance, one-year lag
    gap_miss: float = 0.5  # share of the year's output-gap innovation that real-time estimates miss
    gap_bias: float = 0.0  # average real-time error in the output gap, points
    gap_noise_sd: float = 0.0  # standard deviation of the remaining real-time error, points
    gap_noise_rho: float = 0.0  # its persistence from one year to the next
    edp: bool = True

    def copy(self, **kw) -> Macro:
        return replace(self, **kw)


@dataclass(frozen=True)
class Rule:
    key: str
    label: str
    kind: str  # fixed | debt | structural | spending
    level: float = 2.5  # fixed: pb; debt: pb at d_ref; structural: headline balance; spending: margin x below growth
    eps: float = SEMI_ELASTICITY  # room left to the cycle (fixed, debt, structural)
    kappa: float = 0.0  # debt rule, or spending plans: pp of primary balance per point of debt
    plan_years: int = 4  # spending rule: length of each nominal plan
    indexed: bool = False  # spending rule: index the path to realised inflation each year
    anchor: bool = True  # spending rule: re-anchor the level of spending at the start of each plan
    amortise: int = 0  # offset each fiscal surprise over this many following years (0: never)
    d_ref: float = D_REF

    def copy(self, **kw) -> Rule:
        return replace(self, **kw)


def zero_draws(n: int = 1) -> np.ndarray:
    return np.zeros((n, H, 5))


def simulate(rule: Rule, macro: Macro, draws: np.ndarray, noise: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """draws: (n, H, 5) array of yearly shock vectors (u, v, dm, dsfa, e); noise: optional (n, H) array of
    standard normal draws for the real-time error. Returns (n, H) arrays."""
    n = draws.shape[0]
    nu = np.zeros(n)
    scale = macro.gap_noise_sd * np.sqrt(1 - macro.gap_noise_rho**2)
    d = np.full(n, D0)
    i = np.full(n, I0)
    gap = np.zeros(n)
    mdev = np.zeros(n)
    edp_add = np.zeros(n)
    sp1 = sp2 = None  # structural primary balance one and two years back
    expo = np.zeros(n)  # primary expenditure ratio (spending rule)
    x = 0.0
    eta = SEMI_ELASTICITY - (REVENUE - PB_2027) / 100
    gap_rt_prev = np.zeros(n)
    owed: list[np.ndarray] = []  # negative surprises, each repaid in equal instalments
    credit = np.zeros(n)  # positive surprises, credited against repayments
    out = {k: np.empty((n, H)) for k in ("debt", "pb", "bal", "gap", "sp", "interest", "growth", "edp")}
    for t, y in enumerate(YEARS):
        u, v, dm, dsfa, e = (draws[:, t, k] for k in range(5))
        imp = (sp1 - sp2) if sp2 is not None else 0.0
        gap_new = macro.rho_gap * gap + (1 - macro.h) * u - macro.mu * imp
        if noise is not None and macro.gap_noise_sd > 0:
            nu = macro.gap_noise_rho * nu + scale * noise[:, t] if t else macro.gap_noise_sd * noise[:, t]
        gap_rt = gap_new - macro.gap_miss * (gap_new - macro.rho_gap * gap) + macro.gap_bias + nu
        scar = macro.h * np.minimum(u, 0.0)
        yr = potential_growth(y) + scar + (gap_new - gap)
        pi = base_deflator(y) + macro.kappa * gap_new + v
        g = (1 + yr / 100) * (1 + pi / 100) - 1
        mdev = 0.7 * mdev + dm
        i = (1 - macro.s_reprice) * i + macro.s_reprice * (marginal_rate(y) + mdev)
        intr = i / 100 * d / (1 + g)
        rev = REVENUE + eta * gap_new
        if y == 2027:
            pb = PB_2027 + SEMI_ELASTICITY * gap_new
            expo = rev - pb
            one_off = np.zeros(n)
        else:
            one_off = e.copy()
            if rule.kind == "fixed":
                pb = rule.level + rule.eps * gap_rt
            elif rule.kind == "debt":
                pb = rule.level + rule.kappa * (d - rule.d_ref) + rule.eps * gap_rt
            elif rule.kind == "structural":
                pb = rule.level + intr + rule.eps * gap_rt
            elif rule.kind == "spending":
                if (y - 2028) % rule.plan_years == 0:
                    span = range(y, y + rule.plan_years)
                    growth = np.prod([1 + expected_nominal_growth(z) / 100 for z in span])
                    if rule.anchor:
                        target = rule.level + rule.kappa * (d - rule.d_ref)
                        structural_exp = expo * (1 + gap_rt_prev / 100)
                        x = 100 * (((REVENUE - target) / structural_exp * growth) ** (1 / rule.plan_years) - 1)
                    else:
                        x = 100 * (growth ** (1 / rule.plan_years) - 1) - rule.level
                xt = x + (pi - base_deflator(y)) if rule.indexed else x
                expo = expo * (1 + xt / 100) / (1 + g)
                pb = rev - expo
            else:
                raise ValueError(rule.kind)
            if rule.amortise:
                due = sum(owed[-rule.amortise :]) / rule.amortise if owed else np.zeros(n)
                used = np.minimum(credit, due)
                credit = credit - used + np.maximum(e, 0.0)
                pb = pb + due - used
                owed.append(np.maximum(-e, 0.0))
            pb = pb + edp_add + one_off
        sp = pb - one_off - SEMI_ELASTICITY * gap_new
        bal = pb - intr
        sfa = (SFA_2027 if y == 2027 else SFA_LATER) + dsfa
        d_new = d / (1 + g) + intr - pb + sfa
        for k, val in (
            ("debt", d_new), ("pb", pb), ("bal", bal), ("gap", gap_new), ("sp", sp),
            ("interest", intr), ("growth", 100 * g), ("edp", edp_add),
        ):  # fmt: skip
            out[k][:, t] = val
        if macro.edp:
            edp_add = edp_add + 0.5 * (bal < -3.0)
        sp2 = sp if sp1 is None else sp1  # the 2027 impulse is set to zero
        sp1 = sp
        d, gap, gap_rt_prev = d_new, gap_new, gap_rt
    return out


WINDOW = (2028, 2035)


def effort(sim: dict[str, np.ndarray], window: tuple[int, int] = WINDOW) -> float:
    """Average primary balance over the window, averaged over futures."""
    a, b = YEARS.index(window[0]), YEARS.index(window[1]) + 1
    return float(sim["pb"][:, a:b].mean())


def calibrate(
    rule: Rule, macro: Macro, draws: np.ndarray, target: float, lo=-8.0, hi=12.0, tol=1e-6, noise=None
) -> Rule:
    """Set the rule's level so that its average primary balance over 2028-2035 equals target.

    Uses the same draws at every step (common random numbers), so the map from level to effort
    is deterministic and monotone, and bisection converges."""
    sign = 1.0 if effort(simulate(rule.copy(level=hi), macro, draws, noise)) > effort(
        simulate(rule.copy(level=lo), macro, draws, noise)
    ) else -1.0  # fmt: skip
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f = effort(simulate(rule.copy(level=mid), macro, draws, noise)) - target
        if abs(f) < tol:
            break
        if (f > 0) == (sign > 0):
            hi = mid
        else:
            lo = mid
    return rule.copy(level=mid)


def metrics(sim: dict[str, np.ndarray], window: tuple[int, int] = WINDOW, tighten: float = 1.0) -> dict:
    """The three axes of the comparison, plus supporting numbers, over the window."""
    a, b = YEARS.index(window[0]), YEARS.index(window[1]) + 1
    j35, j45 = YEARS.index(2035), YEARS.index(2045)
    debt, bal, gap, sp = sim["debt"], sim["bal"][:, a:b], sim["gap"][:, a:b], sim["sp"]
    dsp = (sp[:, a:b] - sp[:, a - 1 : b - 1]) if a > 0 else np.diff(sp, axis=1)[:, : b - 1]
    forced = (dsp >= tighten) & (gap < 0)
    q = np.percentile(debt[:, j35], [10, 50, 90])
    return {
        "effort": float(sim["pb"][:, a:b].mean()),
        "p_at_or_below_60_2035": float((debt[:, j35] <= 60).mean()),
        "p10_2035": float(q[0]),
        "median_2035": float(q[1]),
        "p90_2035": float(q[2]),
        "sd_2035": float(debt[:, j35].std()),
        "p_above_90_2035": float((debt[:, j35] > 90).mean()),
        "median_2045": float(np.median(debt[:, j45])),
        "p_breach": float((bal < -3.0).any(axis=1).mean()),
        "breach_years": float((bal < -3.0).sum(axis=1).mean()),
        "p_forced_tightening": float(forced.any(axis=1).mean()),
        "mean_abs_discretionary_change": float(np.abs(dsp).mean()),
        "p90_largest_tightening": float(np.percentile(dsp.max(axis=1), 90)),
        "mean_gap": float(gap.mean()),
        "mean_worst_gap": float(gap.min(axis=1).mean()),
        "p_gap_below_minus_4": float((gap < -4).any(axis=1).mean()),
        "p_edp": float((sim["edp"][:, b - 1] > 0).mean()),
    }
