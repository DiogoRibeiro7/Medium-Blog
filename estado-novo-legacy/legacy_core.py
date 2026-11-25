"""Functions behind the numbers on Portugal's long-run gap and the Estado Novo's schooling legacy. Pure: tables in, numbers out.

Four ingredients:

* relative income: a country's GDP per head divided by that of a reference group, the group's GDP over its population;
* schooling gains under the regime: the gain in adult years of schooling between two dates, regressed on the starting level
  across European countries, so that each country is judged against what countries starting at the same level achieved;
* human capital from schooling: Penn World Table's Mincerian index, exp(phi(s)), with returns of 13.4% a year for the first
  four years, 10.1% for the next four and 6.8% beyond eight (Psacharopoulos 1994, as used by PWT);
* development accounting: output per worker y = A^(1/(1-a)) * (K/Y)^(a/(1-a)) * h, so the log gap to a reference splits
  into a capital-intensity term, a human-capital term and a residual productivity term (Hall and Jones 1999, Caselli 2005).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA = 1.0 / 3.0  # capital share in the accounting, the conventional value

# Western European reference group for the long-run income series: twelve countries with continuous Maddison data from 1900.
W12 = ["AUT", "BEL", "DNK", "FIN", "FRA", "DEU", "ITA", "NLD", "NOR", "SWE", "CHE", "GBR"]

ISO2_TO_ISO3 = {
    "BE": "BEL", "BG": "BGR", "CZ": "CZE", "DK": "DNK", "DE": "DEU", "EE": "EST", "IE": "IRL", "EL": "GRC", "ES": "ESP",
    "FR": "FRA", "HR": "HRV", "IT": "ITA", "CY": "CYP", "LV": "LVA", "LT": "LTU", "LU": "LUX", "HU": "HUN", "MT": "MLT",
    "NL": "NLD", "AT": "AUT", "PL": "POL", "PT": "PRT", "RO": "ROU", "SI": "SVN", "SK": "SVK", "FI": "FIN", "SE": "SWE",
}  # fmt: skip
EU27_ISO2 = list(ISO2_TO_ISO3)

NAMES = {
    "PRT": "Portugal", "ESP": "Spain", "GRC": "Greece", "ITA": "Italy", "IRL": "Ireland", "FIN": "Finland", "POL": "Poland",
    "HUN": "Hungary", "CZE": "Czechia", "SVK": "Slovakia", "SVN": "Slovenia", "EST": "Estonia", "LVA": "Latvia",
    "LTU": "Lithuania", "ROU": "Romania", "BGR": "Bulgaria", "HRV": "Croatia", "AUT": "Austria", "BEL": "Belgium",
    "DNK": "Denmark", "FRA": "France", "DEU": "Germany", "NLD": "Netherlands", "NOR": "Norway", "SWE": "Sweden",
    "CHE": "Switzerland", "GBR": "United Kingdom", "TUR": "Turkey", "RUS": "Russia", "ALB": "Albania", "LUX": "Luxembourg",
    "CYP": "Cyprus", "MLT": "Malta", "EU27_2020": "EU27",
}  # fmt: skip


def relative_to_group(gdp: pd.DataFrame, pop: pd.DataFrame, group: list[str], min_members: int | None = None) -> pd.DataFrame:
    """GDP per head of every column relative to the group's aggregate GDP per head, in percent.

    gdp and pop are wide (years by country codes). The group's per-head figure is the sum of its GDP over the sum of its
    population, computed only in years where at least min_members members (default: all) report both.
    """
    need = len(group) if min_members is None else min_members
    g = gdp[group]
    p = pop[group]
    both = g.notna() & p.notna()
    ref = g.where(both).sum(axis=1) / p.where(both).sum(axis=1)
    ref[both.sum(axis=1) < need] = np.nan
    return 100.0 * (gdp / pop).div(ref, axis=0)


def annual_change(series: pd.Series, start: int, end: int) -> float:
    """Average annual change, in percent, of a positive series between two years (compound rate)."""
    a, b = series.get(start, np.nan), series.get(end, np.nan)
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0 or end <= start:
        return float("nan")
    return float(100.0 * ((b / a) ** (1.0 / (end - start)) - 1.0))


def ols(y, X) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares coefficients and residuals; X must already contain the constant column."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def catch_up_residuals(start: pd.Series, end: pd.Series) -> pd.DataFrame:
    """Gain from start to end regressed on the start level; residual = gain beyond what the start level predicts.

    Returns one row per country with start, gain, predicted gain and residual, plus the slope and intercept in attrs.
    """
    d = pd.DataFrame({"start": start, "end": end}).dropna()
    d["gain"] = d["end"] - d["start"]
    beta, res = ols(d["gain"], np.column_stack([np.ones(len(d)), d["start"]]))
    d["predicted_gain"] = d["gain"] - res
    d["residual"] = res
    d.attrs["intercept"], d.attrs["slope"] = float(beta[0]), float(beta[1])
    return d


def mincer_phi(years) -> np.ndarray:
    """Penn World Table's log human capital per worker for average years of schooling (piecewise-linear returns)."""
    s = np.asarray(years, dtype=float)
    return 0.134 * np.minimum(s, 4.0) + 0.101 * np.clip(s - 4.0, 0.0, 4.0) + 0.068 * np.maximum(s - 8.0, 0.0)


def human_capital_ratio(years_a, years_b) -> np.ndarray:
    """Output per worker at schooling a relative to schooling b, all else equal: exp(phi(a) - phi(b))."""
    return np.exp(mincer_phi(years_a) - mincer_phi(years_b))


def accounting(y: float, ky: float, h: float, y_ref: float, ky_ref: float, h_ref: float, alpha: float = ALPHA) -> dict:
    """Split ln(y / y_ref) into capital intensity, human capital and productivity, the three terms summing to the total.

    y: output per worker; ky: capital over output; h: human capital per worker. The productivity term is the residual.
    Also returns the output per worker the country would have, relative to the reference, with the reference's h.
    """
    total = np.log(y / y_ref)
    k_term = alpha / (1.0 - alpha) * np.log(ky / ky_ref)
    h_term = np.log(h / h_ref)
    a_term = total - k_term - h_term
    return {
        "total": float(total),
        "capital": float(k_term),
        "human_capital": float(h_term),
        "productivity": float(a_term),
        "relative_output": float(np.exp(total)),
        "relative_output_with_ref_h": float(np.exp(total - h_term)),
    }


def geometric_mean(values) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.exp(np.mean(np.log(v)))) if len(v) else float("nan")


def convergence_fit(initial: pd.Series, final: pd.Series, schooling: pd.Series) -> dict:
    """ln(final/initial) = b0 + b1 ln(initial) + b2 schooling, by least squares. Returns coefficients and fitted values."""
    d = pd.DataFrame({"y0": initial, "y1": final, "s": schooling}).dropna()
    growth = np.log(d["y1"] / d["y0"])
    X = np.column_stack([np.ones(len(d)), np.log(d["y0"]), d["s"]])
    beta, res = ols(growth, X)
    fitted_level = d["y0"] * np.exp(X @ beta)
    return {"beta": beta, "residual": pd.Series(res, index=d.index), "fitted_level": fitted_level, "data": d}


def predict_level(beta, initial: float, schooling: float) -> float:
    """Final level implied by the convergence fit for a given initial level and schooling."""
    return float(initial * np.exp(beta[0] + beta[1] * np.log(initial) + beta[2] * schooling))


def bootstrap_convergence(d: pd.DataFrame, draws: int, seed: int) -> np.ndarray:
    """Coefficient draws from a nonparametric (pairs) bootstrap of the convergence fit; columns b0, b1, b2."""
    rng = np.random.default_rng(seed)
    n = len(d)
    out = np.empty((draws, 3))
    y = np.log(d["y1"].to_numpy() / d["y0"].to_numpy())
    X = np.column_stack([np.ones(n), np.log(d["y0"].to_numpy()), d["s"].to_numpy()])
    for i in range(draws):
        idx = rng.integers(0, n, n)
        out[i] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
    return out


def first_year_reaching(series: pd.Series, level: float) -> float:
    """First year in which a series reaches at least `level`; NaN if never."""
    hit = series[series >= level]
    return float(hit.index.min()) if len(hit) else float("nan")


def cohort_shift(shares: list[float], incoming: float, decades: int) -> list[float]:
    """Shares of four ten-year age bands (25-34 to 55-64) after `decades` decades.

    Each cohort keeps its share as it ages one band per decade, the oldest band leaves, and every new
    25-34 band has the share `incoming`. Nobody gains or loses a qualification and nobody migrates.
    """
    bands = list(shares)
    for _ in range(decades):
        bands = [incoming] + bands[:-1]
    return bands
