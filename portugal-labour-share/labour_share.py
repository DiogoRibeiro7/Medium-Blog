"""Portuguese labour share under alternative definitions.

Companion code for the Medium article on how the labour share of Portuguese
GDP depends on measurement choices. Every number quoted in the article is
written to ``results/summary.txt``; the tables behind the figures are written
to ``results/*.csv``.

Run from this folder::

    python labour_share.py

The data are the Eurostat snapshot in ``data/`` (see ``fetch.py`` and the
README). The sample is 1995-2024; 2025 is in the snapshot but provisional and
is not used.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

FIRST_YEAR = 1995
LAST_YEAR = 2024
HAC_LAGS = 3
TRIM = 0.15
N_BOOT = 1999
SEED = 20261021

NUMERATORS = (
    "raw",
    "employment",
    "hours",
    "mixed_all",
    "mixed_two_thirds",
    "mixed_rest",
)
DENOMINATORS = ("market_prices", "factor_cost", "factor_cost_ex_government")

NUMERATOR_LABELS = {
    "raw": "compensation of employees only",
    "employment": "self-employed paid the average employee wage (per person)",
    "hours": "self-employed paid the average employee wage (per hour)",
    "mixed_all": "all mixed income counted as labour",
    "mixed_two_thirds": "two thirds of mixed income counted as labour",
    "mixed_rest": "mixed income split like the rest of the economy",
}
DENOMINATOR_LABELS = {
    "market_prices": "GDP at market prices",
    "factor_cost": "GDP at factor cost",
    "factor_cost_ex_government": "factor cost, government excluded",
}

PEERS = ("PT", "ES", "IT", "EL")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_gdp_panel() -> pd.DataFrame:
    """Return the income-side GDP components for every country in the snapshot."""
    gdp = pd.read_csv(DATA / "nama_10_gdp.csv")
    gdp = gdp[(gdp["year"] >= FIRST_YEAR) & (gdp["year"] <= LAST_YEAR)]
    return gdp.sort_values(["geo", "year"]).reset_index(drop=True)


def load_portugal() -> pd.DataFrame:
    """Return every Portuguese series needed for the labour-share variants.

    Returns
    -------
    pd.DataFrame
        One row per year with national accounts aggregates (``nama_10_gdp``),
        employment (``nama_10_a10_e``), general government accounts
        (``gov_10a_main``) and total-economy operating surplus and mixed income
        (``nasa_10_nf_tr``), all in current prices, million euro, or thousands
        of persons and hours.
    """
    gdp = load_gdp_panel()
    frame = gdp[gdp["geo"] == "PT"].drop(columns="geo")
    for name in ("nama_10_a10_e_PT.csv", "gov_10a_main_PT.csv", "nasa_10_nf_tr_PT.csv"):
        frame = frame.merge(pd.read_csv(DATA / name), on="year", how="inner")
    frame = frame[(frame["year"] >= FIRST_YEAR) & (frame["year"] <= LAST_YEAR)]
    frame = frame.sort_values("year").reset_index(drop=True)
    if len(frame) != LAST_YEAR - FIRST_YEAR + 1:
        raise ValueError("Portuguese panel has missing years.")
    return frame


def data_checks(pt: pd.DataFrame, panel: pd.DataFrame) -> dict[str, float]:
    """Check the accounting identities the analysis relies on.

    Returns
    -------
    dict[str, float]
        Largest absolute gaps, in million euro.
    """
    gap = panel["B1GQ"] - (panel["D1"] + panel["B2A3G"] + panel["D2X3"])
    return {
        "gdp_identity_max_gap": float(gap.abs().max()),
        "nama_vs_nasa_operating_surplus_max_gap": float(
            (pt["B2A3G"] - pt["B2A3G_S1"]).abs().max()
        ),
        "surplus_plus_mixed_income_max_gap": float(
            (pt["B2A3G_S1"] - pt["B2G_S1"] - pt["B3G_S1"]).abs().max()
        ),
    }


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Components:
    """Building blocks shared by every labour-share variant (million euro)."""

    gdp: pd.Series
    factor_cost: pd.Series
    government_factor_income: pd.Series
    compensation: pd.Series
    government_compensation: pd.Series
    mixed_income: pd.Series
    self_employed_labour: dict[str, pd.Series]


def components(pt: pd.DataFrame) -> Components:
    """Compute the components of the labour-share variants.

    Government factor income is government gross value added less other net
    taxes on production, which equals government compensation of employees
    plus its gross operating surplus. Self-employed labour income is the part
    of mixed income (B.3g) that each rule treats as labour.
    """
    factor_cost = pt["D1"] + pt["B2A3G"]
    government = pt["B1G"] - pt["D29PAY"] + pt["D39REC"]
    self_employed = pt["EMP_DC_THS_PER"] - pt["SAL_DC_THS_PER"]
    self_employed_hours = pt["EMP_DC_THS_HW"] - pt["SAL_DC_THS_HW"]
    pay_per_employee = pt["D1"] / pt["SAL_DC_THS_PER"]
    pay_per_hour = pt["D1"] / pt["SAL_DC_THS_HW"]
    rest_share = pt["D1"] / (pt["D1"] + pt["B2G_S1"])
    mixed = pt["B3G_S1"]
    labour = {
        "raw": 0.0 * mixed,
        "employment": self_employed * pay_per_employee,
        "hours": self_employed_hours * pay_per_hour,
        "mixed_all": mixed,
        "mixed_two_thirds": (2.0 / 3.0) * mixed,
        "mixed_rest": rest_share * mixed,
    }
    return Components(
        gdp=pt["B1GQ"],
        factor_cost=factor_cost,
        government_factor_income=government,
        compensation=pt["D1"],
        government_compensation=pt["D1PAY"],
        mixed_income=mixed,
        self_employed_labour=labour,
    )


def labour_share(c: Components, numerator: str, denominator: str) -> pd.Series:
    """Return one labour-share variant, in percent."""
    if denominator == "market_prices":
        top, bottom = c.compensation, c.gdp
    elif denominator == "factor_cost":
        top, bottom = c.compensation, c.factor_cost
    elif denominator == "factor_cost_ex_government":
        top = c.compensation - c.government_compensation
        bottom = c.factor_cost - c.government_factor_income
    else:
        raise ValueError(f"Unknown denominator {denominator!r}")
    return 100.0 * (top + c.self_employed_labour[numerator]) / bottom


def mixed_income_share(c: Components, denominator: str) -> pd.Series:
    """Return mixed income as a percent of the chosen denominator."""
    bottom = {
        "market_prices": c.gdp,
        "factor_cost": c.factor_cost,
        "factor_cost_ex_government": c.factor_cost - c.government_factor_income,
    }[denominator]
    return 100.0 * c.mixed_income / bottom


# ---------------------------------------------------------------------------
# Time-series inference
# ---------------------------------------------------------------------------


def rng_for(label: str) -> np.random.Generator:
    """Return a generator seeded by the global seed and a test label.

    Each test gets its own stream, so a test reported twice gives the same
    p-value and adding a test does not change the others.
    """
    return np.random.default_rng([SEED, zlib.crc32(label.encode("utf-8"))])


def ar1(residuals: np.ndarray) -> tuple[float, float]:
    """Fit an AR(1) to residuals; return (rho, innovation standard deviation)."""
    r = np.asarray(residuals, dtype=float)
    rho = float(np.sum(r[1:] * r[:-1]) / np.sum(r[:-1] ** 2))
    innovations = r[1:] - rho * r[:-1]
    return rho, float(np.std(innovations, ddof=1))


def simulate_ar1(
    rng: np.random.Generator, n_rep: int, n: int, rho: float, sigma: float, burn: int = 200
) -> np.ndarray:
    """Simulate ``n_rep`` AR(1) paths of length ``n`` (stationary start via burn-in)."""
    shocks = rng.normal(0.0, sigma, size=(n_rep, n + burn))
    path = np.zeros_like(shocks)
    for j in range(1, n + burn):
        path[:, j] = rho * path[:, j - 1] + shocks[:, j]
    return path[:, burn:]


def residual_maker(design: np.ndarray) -> np.ndarray:
    """Return M = I - X(X'X)^{-1}X' for a design matrix X."""
    q, _ = np.linalg.qr(design)
    return np.eye(design.shape[0]) - q @ q.T


def trend_test(y: np.ndarray, label: str) -> dict[str, float]:
    """Linear trend with HAC and AR(1)-bootstrap p-values.

    The bootstrap p-value compares the OLS t-statistic of the slope with its
    distribution under a no-trend null whose errors follow the AR(1) fitted to
    the residuals of the trend model.
    """
    n = len(y)
    t = np.arange(n, dtype=float)
    design = sm.add_constant(t)
    hac = sm.OLS(y, design).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    ols = sm.OLS(y, design).fit()
    rho, sigma = ar1(ols.resid)
    sims = simulate_ar1(rng_for("trend:" + label), N_BOOT, n, rho, sigma) + y.mean()
    x = t - t.mean()
    slopes = sims @ x / (x @ x)
    fitted = sims.mean(axis=1, keepdims=True) + np.outer(slopes, x)
    s2 = ((sims - fitted) ** 2).sum(axis=1) / (n - 2)
    t_boot = slopes / np.sqrt(s2 / (x @ x))
    t_obs = ols.params[1] / ols.bse[1]
    p_boot = (1 + np.sum(np.abs(t_boot) >= abs(t_obs))) / (N_BOOT + 1)
    return {
        "slope": float(ols.params[1]),
        "p_hac": float(hac.pvalues[1]),
        "rho": rho,
        "p_ar1_bootstrap": float(p_boot),
    }


@dataclass(frozen=True)
class BreakDesigns:
    """Null and alternative designs for a single-break scan."""

    years: np.ndarray
    null: np.ndarray
    candidates: list[int]
    alternatives: list[np.ndarray]


def break_designs(years: np.ndarray, spec: str) -> BreakDesigns:
    """Build the designs for a break scan.

    ``spec`` is ``"level"`` (trend plus a level shift), ``"level_slope"``
    (trend plus a level shift and a change in slope, the interrupted
    time-series form) or ``"mean"`` (a shift in the mean, no trend).
    """
    n = len(years)
    t = np.arange(n, dtype=float)
    null = np.ones((n, 1)) if spec == "mean" else np.column_stack([np.ones(n), t])
    first = int(np.floor(TRIM * n))
    candidates = list(range(first, n - first + 1))
    alternatives = []
    for k in candidates:
        cols = [null, (t >= k).astype(float)]
        if spec == "level_slope":
            cols.append(np.clip(t - k, 0.0, None))
        elif spec not in ("level", "mean"):
            raise ValueError(f"Unknown spec {spec!r}")
        alternatives.append(np.column_stack(cols))
    return BreakDesigns(years=years, null=null, candidates=candidates, alternatives=alternatives)


def f_profile(y: np.ndarray, designs: BreakDesigns) -> np.ndarray:
    """F statistics for a break at each candidate date; ``y`` may be (n,) or (reps, n)."""
    y2 = np.atleast_2d(y)
    n = y2.shape[1]
    m0 = residual_maker(designs.null)
    ssr0 = np.einsum("ij,jk,ik->i", y2, m0, y2)
    out = np.empty((y2.shape[0], len(designs.candidates)))
    for j, design in enumerate(designs.alternatives):
        m1 = residual_maker(design)
        ssr1 = np.einsum("ij,jk,ik->i", y2, m1, y2)
        q = design.shape[1] - designs.null.shape[1]
        out[:, j] = ((ssr0 - ssr1) / q) / (ssr1 / (n - design.shape[1]))
    return out if np.ndim(y) == 2 else out[0]


def break_test(
    y: np.ndarray, years: np.ndarray, spec: str, label: str, bootstrap: bool = True
) -> dict[str, float]:
    """Single break chosen by the data, with naive and bootstrap p-values.

    The naive p-value is the HAC p-value of the shift at the selected date,
    as if the date had been fixed in advance. The bootstrap p-values compare
    the supremum F statistic with its distribution under the no-break null,
    with AR(1) errors whose persistence is taken either from the no-break
    model's residuals (more persistent) or from the selected break model's
    residuals (less persistent).
    """
    designs = break_designs(years, spec)
    profile = f_profile(y, designs)
    j = int(np.argmax(profile))
    k = designs.candidates[j]
    alt = designs.alternatives[j]
    fit_alt = sm.OLS(y, alt).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    shift_col = designs.null.shape[1]
    result = {
        "spec": spec,
        "break_year": int(years[k]),
        "shift": float(fit_alt.params[shift_col]),
        "naive_p": float(fit_alt.pvalues[shift_col]),
        "sup_f": float(profile[j]),
    }
    if not bootstrap:
        return result
    fit_null = sm.OLS(y, designs.null).fit()
    rho_null, sigma_null = ar1(fit_null.resid)
    rho_alt, sigma_alt = ar1(sm.OLS(y, alt).fit().resid)
    for label_rho, rho, sigma in (("null", rho_null, sigma_null), ("alt", rho_alt, sigma_alt)):
        rng = rng_for(f"break:{label}:{spec}:{years[-1]}:{label_rho}")
        sims = simulate_ar1(rng, N_BOOT, len(y), rho, sigma) + fit_null.fittedvalues[None, :]
        sup = f_profile(sims, designs).max(axis=1)
        result[f"rho_{label_rho}"] = rho
        result[f"p_boot_{label_rho}"] = float((1 + np.sum(sup >= profile[j])) / (N_BOOT + 1))
        result[f"crit95_{label_rho}"] = float(np.quantile(sup, 0.95))
    return result


def fixed_break(y: np.ndarray, years: np.ndarray, year: int) -> tuple[float, float]:
    """Level shift at a fixed year in the interrupted time-series form (HAC p-value)."""
    t = np.arange(len(y), dtype=float)
    k = int(np.where(years == year)[0][0])
    design = np.column_stack([np.ones_like(t), t, (t >= k).astype(float), np.clip(t - k, 0, None)])
    fit = sm.OLS(y, design).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return float(fit.params[2]), float(fit.pvalues[2])


# ---------------------------------------------------------------------------
# Decompositions
# ---------------------------------------------------------------------------


def government_shift_share(pt: pd.DataFrame, c: Components, start: int, end: int) -> dict[str, float]:
    """Split the change in the factor-cost labour share between two years.

    With w the government's share of factor income, g the government labour
    share and p the labour share of the rest of the economy, LS = w g + (1-w) p
    and, with bars for two-year averages,
    dLS = w_bar dg + (1 - w_bar) dp + dw (g_bar - p_bar), exactly.
    """
    idx = {int(y): i for i, y in enumerate(pt["year"])}
    i, j = idx[start], idx[end]
    w = c.government_factor_income / c.factor_cost
    g = c.government_compensation / c.government_factor_income
    p = (c.compensation - c.government_compensation) / (c.factor_cost - c.government_factor_income)
    w_bar, g_bar, p_bar = (w[i] + w[j]) / 2, (g[i] + g[j]) / 2, (p[i] + p[j]) / 2
    parts = {
        "total": 100 * (c.compensation[j] / c.factor_cost[j] - c.compensation[i] / c.factor_cost[i]),
        "within_government": 100 * w_bar * (g[j] - g[i]),
        "within_rest": 100 * (1 - w_bar) * (p[j] - p[i]),
        "between": 100 * (w[j] - w[i]) * (g_bar - p_bar),
        "government_labour_share_start": 100 * g[i],
        "government_labour_share_end": 100 * g[j],
        "rest_labour_share_start": 100 * p[i],
        "rest_labour_share_end": 100 * p[j],
        "government_weight_start": 100 * w[i],
        "government_weight_end": 100 * w[j],
    }
    residual = parts["total"] - parts["within_government"] - parts["within_rest"] - parts["between"]
    if abs(residual) > 1e-9:
        raise AssertionError("Shift-share decomposition does not add up.")
    return parts


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class Report:
    """Collect lines for ``results/summary.txt``."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def section(self, title: str) -> None:
        self.lines += ["", title, "-" * len(title)]

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def write(self, path: Path) -> None:
        path.write_text("\n".join(self.lines).lstrip() + "\n", encoding="utf-8")


def fmt_p(p: float) -> str:
    """Format a p-value with enough digits to tell small values apart."""
    return f"{p:.4f}" if p >= 0.0001 else f"{p:.1e}"


def main() -> None:
    """Run the full analysis and write the results."""
    RESULTS.mkdir(exist_ok=True)
    panel = load_gdp_panel()
    pt = load_portugal()
    c = components(pt)
    years = pt["year"].to_numpy()
    rep = Report()

    rep.add("Portuguese labour share under alternative definitions")
    rep.add(f"Sample {FIRST_YEAR}-{LAST_YEAR}, annual, current prices. Bootstrap replications: {N_BOOT}, seed {SEED}.")

    rep.section("0. Data checks (million euro)")
    for key, value in data_checks(pt, panel).items():
        rep.add(f"{key}: {value:.1f}")

    # 1. Composition --------------------------------------------------------
    rep.section("1. Self-employment and mixed income")
    se = pt["EMP_DC_THS_PER"] - pt["SAL_DC_THS_PER"]
    se_share = 100 * se / pt["EMP_DC_THS_PER"]
    mi_gdp = 100 * c.mixed_income / c.gdp
    mi_per_se = c.mixed_income / se
    pay = pt["D1"] / pt["SAL_DC_THS_PER"]
    ratio = mi_per_se / pay
    rep.add(f"self-employed share of employment: {se_share.iloc[0]:.1f}% ({FIRST_YEAR}) -> {se_share.iloc[-1]:.1f}% ({LAST_YEAR})")
    rep.add(f"self-employed persons (thousands): {se.iloc[0]:.0f} -> {se.iloc[-1]:.0f}; employees: {pt['SAL_DC_THS_PER'].iloc[0]:.0f} -> {pt['SAL_DC_THS_PER'].iloc[-1]:.0f}")
    rep.add(f"mixed income, % of GDP: {mi_gdp.iloc[0]:.2f} -> {mi_gdp.iloc[-1]:.2f} (change {mi_gdp.iloc[-1] - mi_gdp.iloc[0]:+.2f} pp)")
    rep.add(f"mixed income per self-employed / compensation per employee: {ratio.iloc[0]:.2f} ({FIRST_YEAR}), {ratio.iloc[-1]:.2f} ({LAST_YEAR}), range {ratio.min():.2f}-{ratio.max():.2f}")
    for rule in ("employment", "hours", "mixed_rest"):
        alpha = c.self_employed_labour[rule] / c.mixed_income
        rep.add(f"implied labour fraction of mixed income, {rule}: {alpha.iloc[0]:.2f} ({FIRST_YEAR}), {alpha.iloc[-1]:.2f} ({LAST_YEAR}), range {alpha.min():.2f}-{alpha.max():.2f}")

    # 2. Variant grid -------------------------------------------------------
    rep.section("2. Eighteen labour shares (percent; slope in pp per year)")
    rows = []
    long_rows = []
    for num in NUMERATORS:
        for den in DENOMINATORS:
            s = labour_share(c, num, den)
            tt = trend_test(s.to_numpy(), f"PT:{num}:{den}")
            peak = int(s.to_numpy().argmax())
            rows.append({
                "numerator": num,
                "denominator": den,
                "share_first": s.iloc[0],
                "share_last": s.iloc[-1],
                "change": s.iloc[-1] - s.iloc[0],
                "peak": s.iloc[peak],
                "peak_year": int(years[peak]),
                **tt,
            })
            long_rows += [{"year": int(y), "numerator": num, "denominator": den, "share": v} for y, v in zip(years, s)]
    grid = pd.DataFrame(rows)
    grid.to_csv(RESULTS / "variants.csv", index=False, float_format="%.4f")
    pd.DataFrame(long_rows).to_csv(RESULTS / "shares_long.csv", index=False, float_format="%.4f")
    for r in rows:
        rep.add(
            f"{r['numerator']:<17}{r['denominator']:<27}"
            f"{r['share_first']:6.2f} -> {r['share_last']:6.2f}  change {r['change']:+5.2f}  "
            f"peak {r['peak']:.2f} ({r['peak_year']})  slope {r['slope']:+.3f}  "
            f"p_HAC {fmt_p(r['p_hac'])}  rho {r['rho']:.2f}  p_AR1boot {fmt_p(r['p_ar1_bootstrap'])}"
        )
    adjusted = grid[grid["numerator"] != "raw"]
    raw = grid[grid["numerator"] == "raw"]
    rep.add()
    rep.add(f"raw shares: change range {raw['change'].min():+.2f} to {raw['change'].max():+.2f}; slope range {raw['slope'].min():+.3f} to {raw['slope'].max():+.3f}; HAC p range {fmt_p(raw['p_hac'].min())} to {fmt_p(raw['p_hac'].max())}")
    rep.add(f"adjusted shares: change range {adjusted['change'].min():+.2f} to {adjusted['change'].max():+.2f}; slope range {adjusted['slope'].min():+.3f} to {adjusted['slope'].max():+.3f}")
    rep.add(f"adjusted shares: max HAC p {fmt_p(adjusted['p_hac'].max())}; AR(1)-bootstrap p range {fmt_p(adjusted['p_ar1_bootstrap'].min())} to {fmt_p(adjusted['p_ar1_bootstrap'].max())}; count below 0.05: {int((adjusted['p_ar1_bootstrap'] < 0.05).sum())} of {len(adjusted)}")
    rep.add(f"adjusted shares: peak years {sorted(adjusted['peak_year'].unique().tolist())}")

    # 3. Break-even labour fraction -----------------------------------------
    rep.section("3. Labour fraction of mixed income at which the 1995-2024 change is zero")
    rep.add("change(alpha) = change in compensation share + alpha * change in mixed-income share")
    for den in DENOMINATORS:
        comp = labour_share(c, "raw", den)
        mi = mixed_income_share(c, den)
        d_comp = comp.iloc[-1] - comp.iloc[0]
        d_mi = mi.iloc[-1] - mi.iloc[0]
        alpha_star = -d_comp / d_mi
        rep.add(
            f"{den:<27} compensation share change {d_comp:+.2f}, mixed-income share change {d_mi:+.2f} "
            f"-> alpha* {alpha_star:.2f}"
        )

    # 4. Government ---------------------------------------------------------
    rep.section("4. Government")
    w = 100 * c.government_factor_income / c.factor_cost
    g = 100 * c.government_compensation / c.government_factor_income
    peak = int(w.to_numpy().argmax())
    rep.add(f"government share of factor income: {w.iloc[0]:.1f}% ({FIRST_YEAR}), peak {w.iloc[peak]:.1f}% ({years[peak]}), {w.iloc[-1]:.1f}% ({LAST_YEAR})")
    rep.add(f"government labour share (compensation / factor income): mean {g.mean():.1f}%, range {g.min():.1f}-{g.max():.1f}")
    rest = labour_share(c, "raw", "factor_cost_ex_government")
    rep.add(f"labour share outside government (raw): {rest.iloc[0]:.1f}% -> {rest.iloc[-1]:.1f}%")
    idx = {int(y): i for i, y in enumerate(years)}
    for a, b in ((2010, 2012), (2011, 2012), (2010, 2013)):
        parts = government_shift_share(pt, c, a, b)
        rep.add(
            f"{a}->{b}: factor-cost share {parts['total']:+.2f} pp = within government {parts['within_government']:+.2f} "
            f"+ within rest {parts['within_rest']:+.2f} + smaller government {parts['between']:+.2f}; "
            f"government contribution {100 * (parts['within_government'] + parts['between']) / parts['total']:.0f}%"
        )
        rep.add(
            f"    government labour share {parts['government_labour_share_start']:.1f} -> {parts['government_labour_share_end']:.1f}; "
            f"rest {parts['rest_labour_share_start']:.1f} -> {parts['rest_labour_share_end']:.1f}; "
            f"government weight {parts['government_weight_start']:.1f} -> {parts['government_weight_end']:.1f}"
        )
    d_gov = pt["D1PAY"].iloc[idx[2012]] - pt["D1PAY"].iloc[idx[2010]]
    d_all = pt["D1"].iloc[idx[2012]] - pt["D1"].iloc[idx[2010]]
    d_emp = 100 * (pt["SAL_DC_THS_PER"].iloc[idx[2012]] / pt["SAL_DC_THS_PER"].iloc[idx[2010]] - 1)
    rep.add(f"2010->2012: compensation {d_all:+.0f} MEUR, of which government {d_gov:+.0f} MEUR ({100 * d_gov / d_all:.0f}%); employees {d_emp:+.1f}%")
    mkt = labour_share(c, "raw", "market_prices")
    fc = labour_share(c, "raw", "factor_cost")
    rep.add(f"2010->2012 change at market prices {mkt.iloc[idx[2012]] - mkt.iloc[idx[2010]]:+.2f} pp, at factor cost {fc.iloc[idx[2012]] - fc.iloc[idx[2010]]:+.2f} pp")
    tax = 100 * pt["D2X3"] / pt["B1GQ"]
    rep.add(f"net taxes on production and imports, % of GDP: 2010 {tax.iloc[idx[2010]]:.2f}, 2011 {tax.iloc[idx[2011]]:.2f}, 2012 {tax.iloc[idx[2012]]:.2f}, 2013 {tax.iloc[idx[2013]]:.2f}")

    # 5. Breaks in Portugal ---------------------------------------------------
    rep.section("5. Break chosen by the data (Portugal)")
    levels = ", ".join(f"{y} {mkt.iloc[idx[y]]:.1f}" for y in (2010, 2011, 2012, 2013, 2019, 2020, 2024))
    rep.add(f"raw share at market prices: {levels}")
    rep.add(
        f"2019->2020: compensation {100 * (pt['D1'].iloc[idx[2020]] / pt['D1'].iloc[idx[2019]] - 1):+.1f}%, "
        f"GDP {100 * (pt['B1GQ'].iloc[idx[2020]] / pt['B1GQ'].iloc[idx[2019]] - 1):+.1f}%"
    )
    rep.add("Selected year by specification and sample end (no bootstrap):")
    break_series: dict[str, Callable[[], pd.Series]] = {
        "raw/market_prices": lambda: labour_share(c, "raw", "market_prices"),
        "raw/factor_cost": lambda: labour_share(c, "raw", "factor_cost"),
        "raw/ex_government": lambda: labour_share(c, "raw", "factor_cost_ex_government"),
        "employment/market_prices": lambda: labour_share(c, "employment", "market_prices"),
        "employment/ex_government": lambda: labour_share(c, "employment", "factor_cost_ex_government"),
    }
    selection_rows = []
    for end in (2019, 2023, 2024):
        keep = years <= end
        for spec in ("mean", "level", "level_slope"):
            picks = []
            for name, make in break_series.items():
                res = break_test(make().to_numpy()[keep], years[keep], spec, f"PT:{name}", bootstrap=False)
                picks.append(f"{name}={res['break_year']}")
                selection_rows.append({"sample_end": end, "spec": spec, "series": name, "break_year": res["break_year"]})
            rep.add(f"  end {end}, {spec:<11}: " + "  ".join(picks))
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(RESULTS / "break_selection.csv", index=False)
    rep.add(f"distinct years selected across the {len(selection)} cases: {sorted(selection['break_year'].unique().tolist())}")

    rep.add()
    rep.add(f"Full sample {FIRST_YEAR}-{LAST_YEAR}, with bootstrap:")
    break_rows = []
    profile_rows = []
    for name, make in break_series.items():
        y = make().to_numpy()
        for spec in ("level", "level_slope"):
            res = break_test(y, years, spec, f"PT:{name}")
            res["series"] = name
            break_rows.append(res)
            rep.add(
                f"  {name:<25}{spec:<12} year {res['break_year']}  shift {res['shift']:+.2f}  naive p {fmt_p(res['naive_p'])}  "
                f"supF {res['sup_f']:.1f}  boot p {res['p_boot_alt']:.3f} (rho {res['rho_alt']:.2f}) / "
                f"{res['p_boot_null']:.3f} (rho {res['rho_null']:.2f})  5% crit {res['crit95_alt']:.1f} / {res['crit95_null']:.1f}"
            )
            if name == "raw/market_prices":
                designs = break_designs(years, spec)
                for k, f in zip(designs.candidates, f_profile(y, designs)):
                    profile_rows.append({"spec": spec, "year": int(years[k]), "f": f,
                                         "crit95_alt": res["crit95_alt"], "crit95_null": res["crit95_null"]})
    pd.DataFrame(break_rows).to_csv(RESULTS / "breaks_portugal.csv", index=False, float_format="%.4f")
    pd.DataFrame(profile_rows).to_csv(RESULTS / "break_profile_raw_market.csv", index=False, float_format="%.4f")
    rep.add(
        "F needed for 5% at a date fixed in advance (one restriction, level spec): "
        f"{stats.f.ppf(0.95, 1, len(years) - 3):.2f}; (two restrictions, level+slope spec): "
        f"{stats.f.ppf(0.95, 2, len(years) - 4):.2f}"
    )

    # 6. The original 2012 test and peers -----------------------------------
    rep.section("6. Fixed 2012 break (interrupted time series, HAC) and breaks chosen by the data, by country")
    cross_rows = []
    for geo in PEERS:
        sub = panel[panel["geo"] == geo].reset_index(drop=True)
        yrs = sub["year"].to_numpy()
        for den, series in (
            ("market_prices", 100 * sub["D1"] / sub["B1GQ"]),
            ("factor_cost", 100 * sub["D1"] / (sub["D1"] + sub["B2A3G"])),
        ):
            y = series.to_numpy()
            shift12, p12 = fixed_break(y, yrs, 2012)
            row = {"geo": geo, "denominator": den, "shift_2012": shift12, "p_2012_hac": p12,
                   "change": y[-1] - y[0], "first": y[0], "last": y[-1]}
            for spec in ("level", "level_slope"):
                res = break_test(y, yrs, spec, f"{geo}:raw/{den}")
                row.update({f"{spec}_year": res["break_year"], f"{spec}_shift": res["shift"],
                            f"{spec}_naive_p": res["naive_p"], f"{spec}_p_alt": res["p_boot_alt"],
                            f"{spec}_p_null": res["p_boot_null"]})
            cross_rows.append(row)
            rep.add(
                f"{geo} {den:<14} {row['first']:.1f} -> {row['last']:.1f}; fixed 2012 shift {shift12:+.2f} (p {fmt_p(p12)}); "
                f"chosen: level {row['level_year']} ({row['level_shift']:+.2f}, naive p {fmt_p(row['level_naive_p'])}, boot p {row['level_p_alt']:.3f}/{row['level_p_null']:.3f}); "
                f"level+slope {row['level_slope_year']} ({row['level_slope_shift']:+.2f}, naive p {fmt_p(row['level_slope_naive_p'])}, boot p {row['level_slope_p_alt']:.3f}/{row['level_slope_p_null']:.3f})"
            )
    pd.DataFrame(cross_rows).to_csv(RESULTS / "cross_country.csv", index=False, float_format="%.4f")

    # 7. Trough and recovery --------------------------------------------------
    rep.section("7. Trough and rise since, all eighteen shares")
    wide = pd.DataFrame(long_rows).pivot_table(index="year", columns=["numerator", "denominator"], values="share")
    troughs = wide.idxmin()
    rep.add(f"trough years: {({int(k): int(v) for k, v in troughs.value_counts().items()})}")
    rise = wide.loc[LAST_YEAR] - wide.min()
    fall = wide.loc[2010] - wide.loc[2013]
    rep.add(f"rise from trough to {LAST_YEAR}: {rise.min():.2f} to {rise.max():.2f} pp")
    rep.add(f"fall 2010->2013: {fall.min():.2f} to {fall.max():.2f} pp")
    rep.add(f"rise exceeds the 2010->2013 fall in {int((rise > fall).sum())} of {len(rise)} shares")
    xg = wide[("raw", "factor_cost_ex_government")]
    rep.add(f"raw share outside government: {xg.min():.1f}% ({int(xg.idxmin())}) -> {xg.loc[LAST_YEAR]:.1f}% ({LAST_YEAR})")

    rep.write(RESULTS / "summary.txt")
    print("\n".join(rep.lines))


if __name__ == "__main__":
    main()
