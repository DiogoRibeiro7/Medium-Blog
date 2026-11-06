"""Model experiments for the food and housing cases.

    python scenarios.py

Writes ``results/summary.txt`` and the CSV files behind every figure. All
numbers in the article that come from the model are printed in the summary.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import evidence
import model as m

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

GINI = 0.309  # Eurostat ilc_di12, Portugal, 2025
FOOD_BASELINE_PRICED_OUT = 0.03  # households who cannot afford their basket at the reference price
HOUSING_BASELINE_PRICED_OUT = 0.05  # households who cannot afford a dwelling at the reference price
SHARE_65_PLUS = None  # filled from the farm structure data in main()

FOOD_CAP = m.Capacity()  # defaults documented in model.py
PHI0 = 0.10
DELTA0 = 0.025


def food_world(M: float = 1.0, phi: float = PHI0, delta: float = DELTA0) -> m.FoodModel:
    hh = m.Households.from_gini(GINI, 0.4)
    p0 = m.reference_price(hh, FOOD_BASELINE_PRICED_OUT)
    return m.FoodModel(hh=hh, cap=FOOD_CAP, p0=p0, p_w=1.0, phi=m.const(phi), delta=m.const(delta), M=m.const(M))


# ---------------------------------------------------------------------------
# Food: fold, hysteresis, erosion and shock
# ---------------------------------------------------------------------------


def fold_open(cap: m.Capacity, p_w: float, q_w: float, delta: float) -> tuple[float, float]:
    """Largest effort premium at which a viable domestic sector exists when borrowing is unlimited.

    With unlimited borrowing at ``p_w`` the price is ``p_w`` for every K below ``q_w``,
    so dK/dt = 0 needs p_w - c0(K) - phi = pi_s * atanh(delta (K + K_g) / (g_max K)).
    The upper branch exists while phi is below the maximum of the left side minus the
    right side over K. Returns (phi_c, K at the fold).
    """
    K = np.linspace(1e-3, q_w, 20000)
    arg = delta * (K + cap.K_g) / (cap.g_max * K)
    ok = arg < 1.0
    c0 = cap.c_inf + cap.dc / (1.0 + (K / cap.K_h) ** cap.n) + cap.beta * K * K
    need = np.full_like(K, np.inf)
    need[ok] = cap.pi_s * np.arctanh(arg[ok])
    slack = p_w - c0 - need
    i = int(np.argmax(slack))
    return float(slack[i]), float(K[i])


def bifurcation(world: m.FoodModel, phis: np.ndarray, M: float) -> pd.DataFrame:
    rows = []
    for phi in phis:
        for K, stable in world.equilibria(float(phi), DELTA0, M):
            p, b = world.price(K, M)
            rows.append({"phi": phi, "M": M, "K": K, "stable": bool(stable), "price": p, "borrowed": b})
    return pd.DataFrame(rows)


def erosion_runs(world: m.FoodModel) -> dict[str, pd.DataFrame]:
    """phi rises from 0.10 to 0.35 over 40 years; a three-year disruption halves borrowed supply."""
    phi = m.ramp(PHI0, 0.35, 0.0, 40.0)
    K0 = max(k for k, s in world.equilibria(PHI0, DELTA0, 1.0) if s)
    out = {}
    specs = {
        "open_shock_year50": dict(M=m.step(1.0, 0.5, 50.0, 53.0)),
        "open_shock_year5": dict(M=m.step(1.0, 0.5, 5.0, 8.0)),
        "capped_M030": dict(M=m.const(0.30)),
    }
    for name, kw in specs.items():
        w = replace(world, phi=phi, **kw)
        r = w.simulate(K0, 0.0, 70.0, 0.05)
        out[name] = pd.DataFrame(r)
    return out


def alarm_years(df: pd.DataFrame, K0: float) -> dict[str, float | None]:
    t, K, p = df["t"].to_numpy(), df["K"].to_numpy(), df["price"].to_numpy()

    def first(mask):
        idx = np.flatnonzero(mask)
        return float(t[idx[0]]) if len(idx) else None

    return {
        "price_up_2pct": first(p > 1.02),
        "capacity_down_20pct": first(K < 0.8 * K0),
        "capacity_below_10pct_of_need": first(K < 0.10),
    }


def shock_exposure(world: m.FoodModel, cuts=(0.15, 0.30, 0.50)) -> pd.DataFrame:
    """Priced-out share during a disruption that removes a share of what was being borrowed."""
    rows = []
    q_w = world._q_w
    for cut in cuts:
      for K in np.linspace(0.0, 1.0, 201):
        borrowed_before = max(q_w - K, 0.0)
        M_shock = (1.0 - cut) * borrowed_before
        p, _ = world.price(K, M_shock)
        rows.append(
            {
                "cut": cut,
                "K": K,
                "price": p,
                "priced_out": float(world.dem.priced_out(p)),
                "unmet": float(world.dem.unmet_need(p)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Food: retirement wave (rate-induced tipping), vectorised over many runs
# ---------------------------------------------------------------------------


def simulate_food_batch(
    hh: m.Households,
    p0: float,
    caps: dict[str, np.ndarray],
    phi: np.ndarray,
    M: np.ndarray,
    delta_of_t,
    K0: np.ndarray,
    t1: float,
    dt: float = 0.1,
    p_w: float = 1.0,
) -> np.ndarray:
    """RK4 for many independent food economies at once. Returns K at t1."""
    dem = m.Demand(hh, p0)
    q_w = float(dem.total(p_w))

    def price(K):
        p = np.full_like(K, p_w)
        hi = K >= q_w
        if hi.any():
            p[hi] = dem.inverse(K[hi])
        lo = (~hi) & (K + M < q_w)
        if lo.any():
            p[lo] = dem.inverse(K[lo] + M[lo])
        return p

    def f(t, K):
        K = np.maximum(K, 0.0)
        c = caps["c_inf"] + phi + caps["dc"] / (1.0 + (K / caps["K_h"]) ** caps["n"]) + caps["beta"] * K * K
        mg = price(K) - c
        x = np.tanh(mg / caps["pi_s"])
        d = delta_of_t(t)
        up = K * (caps["g_max"] * x * K / (K + caps["K_g"]) - d) + caps["eps"] * x
        down = K * (caps["l_max"] * x - d)
        return np.where(mg >= 0, up, down)

    K = K0.copy()
    n = int(round(t1 / dt))
    for i in range(n):
        t = i * dt
        k1 = f(t, K)
        k2 = f(t + dt / 2, K + dt / 2 * k1)
        k3 = f(t + dt / 2, K + dt / 2 * k2)
        k4 = f(t + dt, K + dt * k3)
        K = np.maximum(K + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
    return K


def wave_rate(share_leaving: float, years: float) -> float:
    """Exit rate that removes ``share_leaving`` of capacity in ``years``."""
    return -np.log(1.0 - share_leaving) / years


def retirement_wave_grid(world: m.FoodModel, share65: float) -> pd.DataFrame:
    """Final capacity after a retirement wave, over the effort premium and the wave's duration.

    The wave removes the 65+ share of capacity over ``W`` years (on top of nothing:
    the exit rate during the wave is the larger of normal attrition and the rate
    that empties that cohort in ``W`` years). Each economy starts on its upper
    branch and is followed for 60 years after the wave ends.
    """
    hh, p0 = world.hh, world.p0
    phis = np.round(np.arange(0.10, 0.2801, 0.01), 3)
    Ws = np.array([3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40], dtype=float)
    Ms = np.array([1.0, 0.3])
    PP, WW, MM = (a.ravel() for a in np.meshgrid(phis, Ws, Ms, indexing="ij"))
    K0 = np.empty_like(PP)
    cache: dict[tuple[float, float], float] = {}
    for i, (ph, Mv) in enumerate(zip(PP, MM)):
        key = (float(ph), float(Mv))
        if key not in cache:
            cache[key] = max(k for k, s in world.equilibria(key[0], DELTA0, key[1]) if s)
        K0[i] = cache[key]
    caps = {k: np.full(PP.shape, getattr(world.cap, k)) for k in ("c_inf", "dc", "K_h", "n", "beta", "g_max", "l_max", "pi_s", "K_g", "eps")}
    rates = np.maximum(DELTA0, wave_rate(share65, WW))
    start = 5.0

    def delta_of_t(t):
        return np.where((t >= start) & (t < start + WW), rates, DELTA0)

    K_end = simulate_food_batch(hh, p0, caps, PP, MM, delta_of_t, K0, start + WW.max() + 60.0)
    return pd.DataFrame({"phi": PP, "W_years": WW, "M": MM, "wave_rate": rates, "K_start": K0, "K_end": K_end, "recovered": K_end > 0.5 * K0})


def monte_carlo(world: m.FoodModel, share65: float, n: int = int(__import__("os").environ.get("MC_N", 2000)), seed: int = 20261106) -> pd.DataFrame:
    """Draw the structural parameters of the capacity curve and record what happens."""
    rng = np.random.default_rng(seed)
    hh, p0 = world.hh, world.p0
    q_w = world._q_w
    draw = {
        "c_inf": rng.uniform(0.45, 0.65, n),
        "dc": rng.uniform(0.30, 0.90, n),
        "K_h": rng.uniform(0.15, 0.40, n),
        "n": rng.uniform(2.0, 6.0, n),
        "beta": rng.uniform(0.20, 0.50, n),
        "g_max": rng.uniform(0.05, 0.09, n),
        "l_max": rng.uniform(0.06, 0.15, n),
        "pi_s": rng.uniform(0.05, 0.20, n),
        "K_g": rng.uniform(0.05, 0.20, n),
        "eps": np.full(n, FOOD_CAP.eps),
        "delta0": rng.uniform(0.020, 0.030, n),
    }
    rows = []
    for i in range(n):
        cap = m.Capacity(**{k: float(draw[k][i]) for k in ("c_inf", "dc", "K_h", "n", "beta", "g_max", "l_max", "pi_s", "K_g", "eps")})
        phi_c, K_fold = fold_open(cap, 1.0, q_w, float(draw["delta0"][i]))
        rows.append({"phi_c": phi_c, "K_fold": K_fold, "restart_premium_at_fold": cap.cost(0.0, phi_c) - 1.0})
    df = pd.concat([pd.DataFrame(draw), pd.DataFrame(rows)], axis=1)
    viable = df["phi_c"] > PHI0 + 0.02  # a domestic sector exists at today's premium, with some room
    sub = df[viable].reset_index(drop=True)
    caps = {k: sub[k].to_numpy() for k in ("c_inf", "dc", "K_h", "n", "beta", "g_max", "l_max", "pi_s", "K_g", "eps")}
    # an economy that has used up three quarters of its headroom to the fold
    phi_run = PHI0 + 0.75 * (sub["phi_c"].to_numpy() - PHI0)
    for M in (1.0, 0.3):
        K0 = np.empty(len(sub))
        for j in range(len(sub)):
            cap = m.Capacity(**{k: float(caps[k][j]) for k in caps})
            w = replace(world, cap=cap)
            st = [k for k, s in w.equilibria(float(phi_run[j]), float(sub["delta0"][j]), M) if s]
            K0[j] = max(st)
        for W in (5.0, 10.0, 15.0, 20.0):
            d0 = sub["delta0"].to_numpy()
            rate = np.maximum(d0, wave_rate(share65, W))

            def delta_of_t(t, rate=rate, d0=d0, W=W):
                return np.where((t >= 5.0) & (t < 5.0 + W), rate, d0)

            K_end = simulate_food_batch(hh, p0, caps, phi_run, np.full(len(sub), M), delta_of_t, K0, 5.0 + W + 60.0)
            sub[f"K0_M{M:g}_W{W:g}"] = K0
            sub[f"Kend_M{M:g}_W{W:g}"] = K_end
            sub[f"recovered_M{M:g}_W{W:g}"] = K_end > 0.5 * K0
    sub.attrs["n_draws"] = n
    sub.attrs["n_viable"] = int(viable.sum())
    return sub


# ---------------------------------------------------------------------------
# Housing: fit to Portugal 2008-2025, then counterfactuals
# ---------------------------------------------------------------------------

HOUSING_CAP = m.Capacity(c_inf=0.55, dc=0.60, K_h=0.35, n=4.0, beta=0.30)
YEAR0 = 2000.0


def housing_world(params: dict, **override) -> m.HousingModel:
    hh = m.Households.from_gini(GINI, 0.4)
    p0 = m.reference_price(hh, HOUSING_BASELINE_PRICED_OUT)
    g_hh = params["household_growth"]
    kw = dict(
        hh=hh,
        cap=HOUSING_CAP,
        p0=p0,
        growth_need=lambda t: g_hh[0] if t < 2019 - YEAR0 else g_hh[1],
        cost_shock=m.step(0.0, params["kappa"], 2008 - YEAR0, 2014 - YEAR0),
        nonprimary=lambda t: 0.44
        * (
            1.0
            if t < 2009 - YEAR0
            else params["x_bust"]
            if t < 2014 - YEAR0
            else params["x_bust"] + (params["x_boom"] - params["x_bust"]) * min(1.0, (t - (2014 - YEAR0)) / 11.0)
        ),
    )
    kw.update(override)
    return m.HousingModel(**kw)


def run_housing(params: dict, t1_year: float = 2031.0, **override) -> pd.DataFrame:
    w = housing_world(params, **override)
    # steady state before the bust, with the pre-bust parameters
    pre = housing_world(params, cost_shock=m.const(0.0), nonprimary=m.const(0.44))
    y0 = pre.steady_state(0.0)
    r = w.simulate(y0, 0.0, t1_year - YEAR0, 0.05)
    df = pd.DataFrame(r)
    df["year"] = YEAR0 + df["t"]
    return df


def annual(df: pd.DataFrame, years) -> pd.DataFrame:
    idx = [int(np.argmin(np.abs(df["year"].to_numpy() - y))) for y in years]
    out = df.iloc[idx].copy()
    out["year"] = list(years)
    return out.set_index("year")


def fit_housing(obs: pd.DataFrame, household_growth: tuple[float, float]) -> tuple[dict, float, pd.DataFrame]:
    """Least squares on log capacity (construction employment) and log price-to-wage, 2008-2025."""
    years = list(range(2008, 2026))
    target_k = np.log(obs.loc[years, "construction_employment"] / obs.loc[2008, "construction_employment"])
    target_p = np.log(obs.loc[years, "price_to_wage"] / obs.loc[2008, "price_to_wage"])

    def loss(theta):
        kappa, x_bust, x_boom = theta
        if not (0.0 <= kappa <= 1.0 and 0.3 <= x_bust <= 1.2 and 0.8 <= x_boom <= 4.0):
            return 1e3
        params = {"kappa": kappa, "x_bust": x_bust, "x_boom": x_boom, "household_growth": household_growth}
        a = annual(run_housing(params, 2026.0), years)
        mk = np.log(a["k"] / a.loc[2008, "k"])
        mp = np.log(a["price"] / a.loc[2008, "price"])
        return float(np.mean((mk.to_numpy() - target_k.to_numpy()) ** 2) + np.mean((mp.to_numpy() - target_p.to_numpy()) ** 2))

    grid = []
    for kappa in (0.2, 0.35, 0.5):
        for x_bust in (0.6, 0.8, 1.0):
            for x_boom in (1.5, 2.0, 2.5):
                grid.append(((kappa, x_bust, x_boom), loss((kappa, x_bust, x_boom))))
    start = min(grid, key=lambda g: g[1])[0]
    res = minimize(loss, np.array(start), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 250})
    kappa, x_bust, x_boom = res.x
    params = {"kappa": float(kappa), "x_bust": float(x_bust), "x_boom": float(x_boom), "household_growth": household_growth}
    a = annual(run_housing(params, 2026.0), years)
    fitted = pd.DataFrame(
        {
            "model_capacity": a["k"] / a.loc[2008, "k"],
            "data_capacity": np.exp(target_k).to_numpy(),
            "model_price": a["price"] / a.loc[2008, "price"],
            "data_price": np.exp(target_p).to_numpy(),
            "model_priced_out": a["priced_out"],
        },
        index=years,
    )
    return params, float(res.fun), fitted


def housing_counterfactuals(params: dict) -> dict[str, pd.DataFrame]:
    sym = replace(HOUSING_CAP, g_max=HOUSING_CAP.l_max + DELTA0 + 0.0)
    runs = {
        "fitted": run_housing(params),
        "floor_80pct_2008_2016": run_housing(params, K_floor=m.step(0.0, 0.8, 2008 - YEAR0, 2017 - YEAR0)),
        "rebuild_as_fast_as_loss": run_housing(params, cap=sym),
        "no_bust": run_housing({**params, "kappa": 0.0, "x_bust": 1.0}),
        "bust_until_2018": run_housing(params, cost_shock=m.step(0.0, params["kappa"], 2008 - YEAR0, 2019 - YEAR0)),
    }
    return runs


# ---------------------------------------------------------------------------


def main() -> None:
    RES.mkdir(exist_ok=True)
    ev = evidence.load_all()
    share65 = float(ev["farm_age"].loc[("PT", 2023), "ge65_share"]) / 100.0
    lines: list[str] = []
    say = lines.append

    world = food_world()
    hh, p0 = world.hh, world.p0
    q_w = world._q_w
    say("== Demand calibration")
    say(f"Gini {GINI} -> lognormal sigma {hh.sigma:.4f}")
    say(f"food reference price: {FOOD_BASELINE_PRICED_OUT:.0%} priced out; need bought at p0 = {float(hh.demand(p0)):.4f}")
    say(f"food discretionary layer x = {world.x} per household at p0, eta = {world.eta}; price elasticity of total demand at p0: {world.dem.elasticity():.3f}")
    say(f"need-only demand elasticity at p0: {float((np.log(hh.demand(p0*1.0001))-np.log(hh.demand(p0*0.9999)))/(np.log(1.0001)-np.log(0.9999))):.3f}")
    for r in (1.25, 1.5, 2.0, 3.0):
        say(f"  food price x{r:g}: total demand {float(world.dem.total(r)):.3f}; priced out {float(hh.priced_out(r * p0)):.3f}, unmet need {float(hh.unmet_need(r * p0)):.3f}")
    hh_h = m.Households.from_gini(GINI, 0.4)
    p0_h = m.reference_price(hh_h, HOUSING_BASELINE_PRICED_OUT)
    for r in (1.25, 1.5, 1.65, 2.0):
        say(f"  housing price x{r:g}: priced out {float(hh_h.priced_out(r * p0_h)):.3f}")
    static = pd.DataFrame({"price_ratio": np.linspace(0.5, 4.0, 351)})
    static["food_total_demand"] = world.dem.total(static.price_ratio.to_numpy())
    static["food_priced_out"] = hh.priced_out(static.price_ratio * p0)
    static["food_unmet"] = hh.unmet_need(static.price_ratio * p0)
    static["housing_priced_out"] = hh_h.priced_out(static.price_ratio * p0_h)
    static.to_csv(RES / "demand_curves.csv", index=False)

    say("\n== Food: equilibria and the fold")
    phi_c, K_fold = fold_open(world.cap, 1.0, q_w, DELTA0)
    say(f"open borrowing (M >= {q_w:.3f}): fold at phi_c = {phi_c:.4f}, K at fold = {K_fold:.4f}")
    K_today = max(k for k, s in world.equilibria(PHI0, DELTA0, 1.0) if s)
    say(f"phi0 = {PHI0}: anchored capacity {K_today:.4f} of need; borrowed {q_w - K_today:.4f}")
    say(f"restart needs price above c(0) = {world.cap.cost(0.0, phi_c):.4f} p0 at phi_c (world price 1.0)")
    say(f"restart premium at phi_c: {world.cap.cost(0.0, phi_c) - 1.0:.4f} p0 per unit; at phi0: {world.cap.cost(0.0, PHI0) - 1.0:.4f}")
    phis = np.round(np.arange(0.0, 0.401, 0.0025), 4)
    bif = pd.concat([bifurcation(world, phis, 1.0), bifurcation(world, phis, 0.3)], ignore_index=True)
    bif.to_csv(RES / "food_bifurcation.csv", index=False)
    for phi in (0.10, 0.20, 0.25, 0.27):
        st = bif[(bif.M == 1.0) & (np.isclose(bif.phi, phi)) & bif.stable & (bif.K > 0)]
        if len(st):
            say(f"  open, phi {phi:.2f}: upper branch K = {st.K.max():.4f}, price {st.price.iloc[-1]:.3f}")
    capped = bif[(bif.M == 0.3) & bif.stable & (bif.K > 0.2)].groupby("phi").K.max()
    say(f"capped (M = 0.3): upper branch at phi 0.40 = {capped.loc[capped.index.max()]:.4f}")

    say("\n== Food: erosion with a disruption")
    runs = erosion_runs(world)
    for name, df in runs.items():
        df.to_csv(RES / f"food_erosion_{name}.csv", index=False)
    al = alarm_years(runs["open_shock_year50"], K_today)
    say(f"open world, phi 0.10 -> 0.35 over 40 years: {al}")
    al_c = alarm_years(runs["capped_M030"], K_today)
    say(f"capped world (M = 0.3): {al_c}")
    for name in ("open_shock_year50", "open_shock_year5"):
        df = runs[name]
        win = df[(df.t >= (50 if name.endswith("50") else 5)) & (df.t < (53 if name.endswith("50") else 8))]
        say(
            f"  {name}: anchored K at shock start {win.K.iloc[0]:.4f}; peak price {win.price.max():.3f} p0; "
            f"peak priced out {win.priced_out.max():.3f}; peak unmet need {win.unmet.max():.3f}"
        )
    end = runs["open_shock_year50"].iloc[-1]
    say(f"  open world after the disruption, year 70: K {end.K:.5f}, price {end.price:.3f}")
    cend = runs["capped_M030"].iloc[-1]
    say(f"  capped world, year 70: K {cend.K:.4f}, price {cend.price:.3f} p0, priced out {cend.priced_out:.3f}")
    expo = shock_exposure(world)
    expo.to_csv(RES / "food_shock_exposure.csv", index=False)
    for cut in (0.15, 0.30, 0.50):
        e = expo[expo.cut == cut]
        for K in (0.0, 0.25, 0.5, 0.75, 0.9):
            row = e.iloc[int(np.argmin(np.abs(e.K - K)))]
            say(f"  disruption removes {cut:.0%} of borrowed supply, anchored K {K:.2f}: price {row.price:.3f} p0, priced out {row.priced_out:.3f}, unmet {row.unmet:.3f}")

    say("\n== Food: retirement wave")
    say(f"Portugal 2023, farm managers aged 65+: {share65:.3f}")
    for W in (10, 15, 20):
        say(f"  exit rate if they all leave within {W} years: {wave_rate(share65, W):.4f} per year")
    rw = retirement_wave_grid(world, share65)
    rw.to_csv(RES / "food_retirement_wave.csv", index=False)
    for M in (1.0, 0.3):
        for phi in (0.10, 0.18, 0.22, 0.25, 0.27):
            sub = rw[(rw.M == M) & np.isclose(rw.phi, phi)].sort_values("W_years")
            fail = sub[~sub.recovered]
            crit = fail.W_years.max() if len(fail) else None
            say(f"  M = {M:g}, phi {phi:.2f}: start K {sub.K_start.iloc[0]:.4f}; longest wave that is not survived: {crit} years")

    say("\n== Food: Monte Carlo over the capacity curve")
    mc = monte_carlo(world, share65)
    mc.to_csv(RES / "food_monte_carlo.csv", index=False)
    say(f"draws {mc.attrs['n_draws']}, with a domestic sector viable above phi0 + 0.02: {mc.attrs['n_viable']}")
    say(f"  headroom phi_c - phi0: median {np.median(mc.phi_c - PHI0):.4f}, 10-90% {np.quantile(mc.phi_c - PHI0, 0.1):.4f} to {np.quantile(mc.phi_c - PHI0, 0.9):.4f}")
    say(f"  restart premium at the fold: median {mc.restart_premium_at_fold.median():.4f} p0; 10-90% {mc.restart_premium_at_fold.quantile(0.1):.4f} to {mc.restart_premium_at_fold.quantile(0.9):.4f}; share <= 0 (no lock-in) {(mc.restart_premium_at_fold <= 0).mean():.3f}")
    say(f"  anchored capacity at the fold: median {mc.K_fold.median():.4f}")
    for W in (5.0, 10.0, 15.0, 20.0):
        o = 1.0 - mc[f"recovered_M1_W{W:g}"].mean()
        c = 1.0 - mc[f"recovered_M0.3_W{W:g}"].mean()
        say(f"  wave over {W:g} years: share that never recovers, open {o:.3f}, capped {c:.3f}")

    say("\n== Housing: fit to Portugal, 2008-2025")
    obs = ev["housing_obs"]
    hg = ev["household_growth"]
    say(f"household growth from Eurostat lfst_hhnhtych: {hg[0]:.4f} (2006-2019), {hg[1]:.4f} (2019-2025)")
    params, sse, fitted = fit_housing(obs, hg)
    fitted.to_csv(RES / "housing_fit.csv")
    say(f"fitted: kappa {params['kappa']:.3f} p0, non-primary demand x{params['x_bust']:.3f} in 2009-2013, x{params['x_boom']:.3f} by 2025")
    say(f"mean squared log error per point (both series): {sse:.5f}; RMSE {np.sqrt(sse / 2):.4f}")
    for y in (2013, 2014, 2019, 2025):
        r = fitted.loc[y]
        say(f"  {y}: capacity model {r.model_capacity:.3f} data {r.data_capacity:.3f}; price model {r.model_price:.3f} data {r.data_price:.3f}; priced out {r.model_priced_out:.3f}")
    (RES / "housing_params.json").write_text(json.dumps(params, indent=2))
    cf = housing_counterfactuals(params)
    for name, df in cf.items():
        df.to_csv(RES / f"housing_{name}.csv", index=False)
        a = annual(df, [2008, 2014, 2025, 2030])
        say(
            f"  {name}: capacity 2014 {a.loc[2014, 'k']:.3f}, 2025 {a.loc[2025, 'k']:.3f}, 2030 {a.loc[2030, 'k']:.3f}; "
            f"price 2025 {a.loc[2025, 'price']:.3f} p0; priced out 2025 {a.loc[2025, 'priced_out']:.3f}, 2030 {a.loc[2030, 'priced_out']:.3f}"
        )

    text = "\n".join(lines) + "\n"
    (RES / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
