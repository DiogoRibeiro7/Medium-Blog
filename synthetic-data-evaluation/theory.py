"""Closed-form coverage for a correctly specified generator.

Let U be the sampling variance of the real-data estimate (n_obs rows) and
k = n_syn / n_obs. An estimate from one synthetic data set of n_syn rows
varies around the generator's parameters with variance U / k, and the
standard error computed from that data set estimates U / k as well, so
ubar is about U / k.

Simple synthesis (parameters fixed at their estimates):
    Var(q_l)  = U + U/k                       (estimation + synthesis noise)
    Var(qbar) = U + U/(k m)

Proper synthesis (parameters redrawn, with variance about U, before each set):
    Var(q_l)  = U + U + U/k
    Var(qbar) = U + (U + U/k)/m

Dividing by the variance each rule uses gives the factor r by which the rule
understates the variance, and the nominal 95% interval then covers with
probability 2 Phi(1.96 / sqrt(r)) - 1 (inference.predicted_coverage).
"""

from __future__ import annotations

from inference import predicted_coverage


def true_variance(k: float, m: int, proper: bool) -> float:
    """Var(qbar) in units of ubar (= U / k)."""
    if proper:
        return k + (k + 1.0) / m
    return k + 1.0 / m


def rule_variance(rule: str, k: float, m: int) -> float:
    """Variance each rule uses, in units of ubar."""
    return {
        "naive": 1.0 / m,
        "simple": k + 1.0 / m,
        "proper": k + (1.0 + k) / m,
    }[rule]


def coverage(rule: str, k: float, m: int, proper: bool) -> float:
    return predicted_coverage(true_variance(k, m, proper) / rule_variance(rule, k, m))


def size_coverage(k: float, proper: bool) -> float:
    """One synthetic data set of k n_obs rows analysed as if it were real."""
    return coverage("naive", k, 1, proper)
