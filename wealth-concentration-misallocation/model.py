"""Heterogeneous-entrepreneur economy with collateral constraints.

Two pieces:

1. A static allocation problem (``allocate``). N owners hold wealth a_i and run a firm with
   productivity z_i and technology y_i = z_i k_i^gamma, gamma < 1. Capital is rented on a
   market at rate R; each owner can rent at most lam * a_i (a collateral constraint, as in
   Evans and Jovanovic 1989, Buera and Shin 2013, Moll 2014). Total capital equals total
   wealth. With lam = inf the allocation is first best and depends on z only; with a finite
   lam it depends on the joint distribution of (a, z). Efficiency is output relative to the
   frictionless output with the same capital and the same productivities.

2. A dynamic economy (``Economy``) of dynasties that adds labour, saving and inheritance.
   Firm technology is y = z (k^alpha n^(1-alpha))^eta; every household supplies one unit of
   labour; households save a share s_w of wage income and s_k of capital income (the rich
   save more: Dynan, Skinner and Zeldes 2004); productivity follows an AR(1)
   within a life and is redrawn, with correlation rho_g, when a dynasty passes to an heir,
   while wealth passes in full (less any estate tax). Taxes on capital income or on wealth
   can be switched on; both are deducted from disposable capital income before saving, so
   the two differ only in their base. Each period the factor markets clear exactly.

Everything is deterministic given the seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm, rankdata

# ---------------------------------------------------------------------------------------
# distributions


def lognormal_sigma_for_top_share(share: float, p: float = 0.01) -> float:
    """Log-sd of a lognormal whose top-p share of the total equals ``share``.

    For a lognormal with log-sd sigma, the top-p share is 1 - Phi(Phi^{-1}(1 - p) - sigma).
    """
    return float(norm.ppf(1 - p) - norm.ppf(1 - share))


def pearson_for_spearman(rho_s: float) -> float:
    """Gaussian-copula correlation that delivers a given Spearman rank correlation."""
    return float(2 * np.sin(np.pi * rho_s / 6))


def top_share(x: np.ndarray, p: float = 0.01) -> float:
    xs = np.sort(x)[::-1]
    m = max(1, int(round(p * len(xs))))
    return float(xs[:m].sum() / xs.sum())


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def coupled_draws(n: int, top1: float, sigma_z: float, rho_s: float,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Wealth (mean 1, lognormal with a target top-1% share) and productivity coupled by a
    Gaussian copula with Spearman correlation ``rho_s``."""
    sigma_a = lognormal_sigma_for_top_share(top1)
    r = pearson_for_spearman(rho_s)
    e1 = rng.standard_normal(n)
    e2 = rng.standard_normal(n)
    a = np.exp(sigma_a * e1)
    a /= a.mean()
    z = np.exp(sigma_z * (r * e1 + np.sqrt(1 - r * r) * e2))
    return a, z


# ---------------------------------------------------------------------------------------
# static allocation


@dataclass
class Allocation:
    k: np.ndarray
    R: float
    output: float
    first_best: float
    constrained: float  # share of owners at the collateral limit

    @property
    def efficiency(self) -> float:
        return self.output / self.first_best


def first_best_output(z: np.ndarray, K: float, gamma: float) -> float:
    return float(np.sum(z ** (1 / (1 - gamma))) ** (1 - gamma) * K ** gamma)


def _demand(logR: float, loga_lam: np.ndarray, logz_g: np.ndarray, gamma: float) -> np.ndarray:
    # unconstrained demand: k = (gamma z / R)^(1/(1-gamma)); capped at lam * a
    return np.exp(np.minimum(loga_lam, (logz_g - logR) / (1 - gamma)))


def allocate(a: np.ndarray, z: np.ndarray, lam: float, gamma: float) -> Allocation:
    """Competitive allocation of K = sum(a) under k_i <= lam * a_i."""
    K = float(a.sum())
    Y_fb = first_best_output(z, K, gamma)
    if not np.isfinite(lam):
        w = z ** (1 / (1 - gamma))
        k = K * w / w.sum()
        R = float(gamma * np.sum(w) ** (1 - gamma) * K ** (gamma - 1))
        return Allocation(k, R, float(np.sum(z * k ** gamma)), Y_fb, 0.0)
    loga_lam = np.log(lam * a)
    logz_g = np.log(gamma * z)

    def excess(logR: float) -> float:
        return float(_demand(logR, loga_lam, logz_g, gamma).sum() / K - 1.0)

    lo, hi = -40.0, 40.0
    if excess(lo) <= 0:  # every owner constrained even at R -> 0 (lam = 1)
        k = lam * a * (K / (lam * a).sum())
        R = float(np.min(gamma * z * k ** (gamma - 1)))
        return Allocation(k, R, float(np.sum(z * k ** gamma)), Y_fb, 1.0)
    logR = brentq(excess, lo, hi, xtol=1e-12, maxiter=500)
    k = _demand(logR, loga_lam, logz_g, gamma)
    k *= K / k.sum()
    constrained = float(np.mean(k >= lam * a * (1 - 1e-9)))
    return Allocation(k, float(np.exp(logR)), float(np.sum(z * k ** gamma)), Y_fb, constrained)


# ---------------------------------------------------------------------------------------
# dynamic economy


@dataclass
class Params:
    n: int = 10_000
    alpha: float = 1 / 3       # capital share inside the firm's input bundle
    eta: float = 0.75          # span of control (returns to scale of the firm)
    lam: float = 1.25          # collateral limit: k <= lam * a
    delta: float = 0.06        # depreciation
    s_w: float = 0.05          # saving rate out of wage income
    s_k: float = 0.50          # saving rate out of capital income (profits + rent, after tax)
    sigma_z: float = 0.5       # stationary sd of log productivity
    psi: float = 0.98          # annual persistence of log productivity within a life
    death: float = 1 / 40      # annual probability that a dynasty passes to an heir
    rho_g: float = 0.3         # correlation of log productivity between parent and heir
    tau_k: float = 0.0         # tax on capital income (profits + rental income)
    tau_a: float = 0.0         # annual tax on wealth
    tau_e: float = 0.0         # estate tax on wealth passed to an heir


@dataclass
class State:
    a: np.ndarray
    logz: np.ndarray
    w: float = 1.0
    R: float = 0.1


@dataclass
class Period:
    output: float
    first_best: float
    K: float
    w: float
    R: float
    top1_wealth: float
    top10_wealth: float
    coupling: float          # Spearman(a, z)
    capital_coupling: float  # Spearman(k, z)
    revenue: float
    constrained: float

    @property
    def efficiency(self) -> float:
        return self.output / self.first_best


@dataclass
class Economy:
    p: Params
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def initial_state(self) -> State:
        n = self.p.n
        logz = self.p.sigma_z * self.rng.standard_normal(n)
        a = np.ones(n) * 3.0
        return State(a=a, logz=logz)

    # -- one period of market clearing ------------------------------------------------
    def clear(self, st: State) -> tuple[np.ndarray, np.ndarray, float, float]:
        p = self.p
        mu, nu = p.alpha * p.eta, (1 - p.alpha) * p.eta
        gamma = mu / (1 - nu)
        z = np.exp(st.logz)
        K = float(st.a.sum())
        N = float(p.n)
        loga_lam = np.log(p.lam * st.a)
        logz_n = st.logz / (1 - nu)

        def capital_given_w(logw: float) -> tuple[np.ndarray, float]:
            # labour is chosen freely given k; the optimised firm is Z k^gamma - R k
            logZ = np.log(1 - nu) + (nu / (1 - nu)) * (np.log(nu) - logw) + logz_n
            logZg = np.log(gamma) + logZ

            def excess(logR: float) -> float:
                return float(np.exp(np.minimum(loga_lam, (logZg - logR) / (1 - gamma))).sum() / K - 1)

            logR = brentq(excess, -30.0, 30.0, xtol=1e-11, maxiter=500)
            k = np.exp(np.minimum(loga_lam, (logZg - logR) / (1 - gamma)))
            return k * (K / k.sum()), logR

        def labour_excess(logw: float) -> float:
            k, _ = capital_given_w(logw)
            logn = (np.log(nu) + st.logz + mu * np.log(k) - logw) / (1 - nu)
            return float(np.exp(logn).sum() / N - 1)

        c = np.log(st.w)
        lo, hi = c - 0.5, c + 0.5
        while labour_excess(lo) < 0:
            lo -= 1.0
        while labour_excess(hi) > 0:
            hi += 1.0
        logw = brentq(labour_excess, lo, hi, xtol=1e-11, maxiter=500)
        k, logR = capital_given_w(logw)
        n = np.exp((np.log(nu) + st.logz + mu * np.log(k) - logw) / (1 - nu))
        n *= N / n.sum()
        st.w, st.R = float(np.exp(logw)), float(np.exp(logR))
        return k, n, st.w, st.R

    def first_best(self, logz: np.ndarray, K: float) -> float:
        p = self.p
        return float(np.sum(np.exp(logz / (1 - p.eta))) ** (1 - p.eta)
                     * (K ** p.alpha * p.n ** (1 - p.alpha)) ** p.eta)

    # -- one period of production, taxes, saving, ageing ----------------------------
    def step(self, st: State) -> Period:
        p = self.p
        k, n, w, R = self.clear(st)
        z = np.exp(st.logz)
        y = z * (k ** p.alpha * n ** (1 - p.alpha)) ** p.eta
        profit = y - R * k - w * n
        cap_income = profit + R * st.a
        tax_k = p.tau_k * cap_income
        tax_a = p.tau_a * st.a
        K = float(st.a.sum())
        rec = Period(
            output=float(y.sum()), first_best=self.first_best(st.logz, K), K=K, w=w, R=R,
            top1_wealth=top_share(st.a, 0.01), top10_wealth=top_share(st.a, 0.10),
            coupling=spearman(st.a, z), capital_coupling=spearman(k, z),
            revenue=float(tax_k.sum() + tax_a.sum()),
            constrained=float(np.mean(k >= p.lam * st.a * (1 - 1e-9))),
        )
        # Both taxes reduce disposable capital income, which is saved at the same rate s_k, so
        # the two regimes differ only in their base (income vs the stock of wealth).
        a_next = (1 - p.delta) * st.a + p.s_k * (cap_income - tax_k - tax_a) + p.s_w * w
        a_next = np.maximum(a_next, 1e-6)
        # productivity: AR(1) within a life
        eps = self.rng.standard_normal(p.n)
        logz_next = p.psi * st.logz + np.sqrt(1 - p.psi ** 2) * p.sigma_z * eps
        # succession: wealth passes (less estate tax); productivity is redrawn around the parent's
        dies = self.rng.random(p.n) < p.death
        if dies.any():
            e = self.rng.standard_normal(int(dies.sum()))
            logz_next[dies] = p.rho_g * st.logz[dies] + np.sqrt(1 - p.rho_g ** 2) * p.sigma_z * e
            estate = p.tau_e * a_next[dies]
            rec.revenue += float(estate.sum())
            a_next[dies] -= estate
        st.a, st.logz = a_next, logz_next
        return rec

    def run(self, years: int, st: State | None = None) -> tuple[State, list[Period]]:
        st = st or self.initial_state()
        hist = [self.step(st) for _ in range(years)]
        return st, hist


def summarise(hist: list[Period], last: int) -> dict[str, float]:
    tail = hist[-last:]
    keys = ["output", "first_best", "K", "w", "R", "top1_wealth", "top10_wealth",
            "coupling", "capital_coupling", "revenue", "constrained"]
    out = {k: float(np.mean([getattr(h, k) for h in tail])) for k in keys}
    out["efficiency"] = float(np.mean([h.efficiency for h in tail]))
    return out
