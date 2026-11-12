"""Simulation experiments. Writes results/sim_*.csv and results/model_summary.txt.

    python experiments.py            # full run (a few minutes on four cores)
    python experiments.py --quick    # small grid for the tests

E1  Static allocation: efficiency over (top-1% wealth share x Spearman(wealth, productivity)),
    with the collateral limit of the dynamic baseline, plus slices for looser credit.
E2  Dynamic economy: steady states as the correlation of productivity between parent and
    heir (rho_g) goes from 0 to 1.
E3  Dynamic economy: a 25% tax on capital income against a wealth tax that raises the same
    revenue in steady state.
E4  Dynamic economy: a technology shock re-ranks productivity at t = 0; paths of coupling,
    efficiency and concentration under no tax, the capital income tax and the wealth tax.
E5  Dynamic economy: persistence of productivity within a life (psi) crossed with rho_g = 0
    and rho_g = 1, i.e. when does inheritance of wealth without inheritance of ability bite?
E6  Sensitivity of the E4 transition (no tax) to the two settings that drive self-financing:
    the saving rate out of capital income (s_k) and the persistence of productivity (psi).
"""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from model import (Economy, Params, allocate, coupled_draws, lognormal_sigma_for_top_share,
                   spearman, summarise, top_share)
from scipy.stats import norm

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT = Path(__file__).resolve().parent / "results"
BASE = Params()
NU = (1 - BASE.alpha) * BASE.eta
GAMMA = BASE.alpha * BASE.eta / (1 - NU)      # reduced-form capital exponent at a given wage
SIGMA_EFF = BASE.sigma_z / (1 - NU)           # dispersion of the reduced-form productivity

BURN, AVG = 350, 100


# ---------------------------------------------------------------------------------------
# E1


def _static_cell(args):
    lam, top1, rho, seed, n = args
    a, z = coupled_draws(n, top1, SIGMA_EFF, rho, np.random.default_rng(seed))
    al = allocate(a, z, lam, GAMMA)
    return dict(lam=lam, top1_target=top1, rho_target=rho, seed=seed,
                top1=top_share(a), rho=spearman(a, z), efficiency=al.efficiency,
                constrained=al.constrained)


def e1(pool, quick: bool) -> pd.DataFrame:
    n = 4_000 if quick else 20_000
    seeds = range(2 if quick else 4)
    top1s = np.round(np.arange(0.04, 0.501, 0.06 if quick else 0.02), 3)
    rhos = np.round(np.arange(-0.2, 1.001, 0.3 if quick else 0.05), 3)
    jobs = [(BASE.lam, t, r, s, n) for t in top1s for r in rhos for s in seeds]
    for lam in (2.0, 4.0, 10.0):
        jobs += [(lam, t, r, s, n) for t in top1s for r in (0.0, 0.5, 0.9, 1.0) for s in seeds]
    rows = list(pool.map(_static_cell, jobs, chunksize=8))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# dynamic helpers


def _steady(args):
    params, seed, burn, avg = args
    eco = Economy(params, np.random.default_rng(seed))
    _, hist = eco.run(burn + avg)
    out = summarise(hist, avg)
    out.update(seed=seed, rho_g=params.rho_g, tau_k=params.tau_k, tau_a=params.tau_a)
    return out


def _transition(args):
    params, seed, burn, horizon = args
    eco = Economy(params, np.random.default_rng(seed))
    st, _ = eco.run(burn)
    # technology shock: productivity re-ranked, independent of wealth and of past productivity
    st.logz = params.sigma_z * eco.rng.standard_normal(params.n)
    rows = []
    for t in range(horizon):
        h = eco.step(st)
        rows.append(dict(t=t, seed=seed, tau_k=params.tau_k, tau_a=params.tau_a,
                         efficiency=h.efficiency, coupling=h.coupling, top1=h.top1_wealth,
                         top10=h.top10_wealth, output=h.output, wage=h.w))
    return rows


def e2(pool, quick: bool) -> pd.DataFrame:
    n = 3_000 if quick else BASE.n
    burn, avg = (150, 40) if quick else (BURN, AVG)
    seeds = range(1 if quick else 3)
    grid = (0.0, 0.5, 1.0) if quick else (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    jobs = [(replace(BASE, n=n, rho_g=g), s, burn, avg) for g in grid for s in seeds]
    return pd.DataFrame(list(pool.map(_steady, jobs)))


def revenue_neutral_wealth_tax(pool, tau_k: float, quick: bool) -> tuple[float, float, float]:
    """Wealth tax that raises the same steady-state revenue (share of output) as tau_k."""
    n = 3_000 if quick else BASE.n
    burn, avg = (150, 40) if quick else (BURN, AVG)
    seeds = range(1 if quick else 3)

    def rev_share(p: Params) -> float:
        res = list(pool.map(_steady, [(p, s, burn, avg) for s in seeds]))
        return float(np.mean([r["revenue"] / r["output"] for r in res]))

    target = rev_share(replace(BASE, n=n, tau_k=tau_k))
    lo, hi = 0.0, 0.08
    f_lo, f_hi = 0.0 - target, rev_share(replace(BASE, n=n, tau_a=hi)) - target
    for _ in range(4 if quick else 8):  # bracketed secant (regula falsi with Illinois step)
        x = hi - f_hi * (hi - lo) / (f_hi - f_lo)
        fx = rev_share(replace(BASE, n=n, tau_a=x)) - target
        if abs(fx) < 2e-4:
            return x, target, fx + target
        if fx * f_hi < 0:
            lo, f_lo = hi, f_hi
        else:
            f_lo /= 2
        hi, f_hi = x, fx
    return hi, target, f_hi + target


def e3(pool, quick: bool, tau_k: float, tau_a: float) -> pd.DataFrame:
    n = 3_000 if quick else BASE.n
    burn, avg = (150, 40) if quick else (BURN, AVG)
    seeds = range(1 if quick else 3)
    regimes = [replace(BASE, n=n), replace(BASE, n=n, tau_k=tau_k), replace(BASE, n=n, tau_a=tau_a)]
    jobs = [(p, s, burn, avg) for p in regimes for s in seeds]
    return pd.DataFrame(list(pool.map(_steady, jobs)))


def e4(pool, quick: bool, tau_k: float, tau_a: float) -> pd.DataFrame:
    n = 3_000 if quick else BASE.n
    burn, horizon = (150, 40) if quick else (BURN, 120)
    seeds = range(1 if quick else 3)
    regimes = [replace(BASE, n=n), replace(BASE, n=n, tau_k=tau_k), replace(BASE, n=n, tau_a=tau_a)]
    jobs = [(p, s, burn, horizon) for p in regimes for s in seeds]
    rows = [r for chunk in pool.map(_transition, jobs) for r in chunk]
    return pd.DataFrame(rows)


def e5(pool, quick: bool) -> pd.DataFrame:
    n = 3_000 if quick else BASE.n
    burn, avg = (150, 40) if quick else (BURN, AVG)
    seeds = range(1 if quick else 3)
    psis = (0.85, 0.98) if quick else (0.80, 0.90, 0.95, 0.98)
    jobs = [(replace(BASE, n=n, psi=psi, rho_g=g), s, burn, avg)
            for psi in psis for g in (0.0, 1.0) for s in seeds]
    res = list(pool.map(_steady, jobs))
    for r, (p, *_rest) in zip(res, jobs):
        r["psi"] = p.psi
    return pd.DataFrame(res)


def _transition_metrics(args):
    params, seed, burn, horizon, pre = args
    eco = Economy(params, np.random.default_rng(seed))
    st, hist = eco.run(burn)
    tail = hist[-pre:]
    c0 = float(np.mean([h.coupling for h in tail]))
    e0 = float(np.mean([h.efficiency for h in tail]))
    y0 = float(np.mean([h.output for h in tail]))
    st.logz = params.sigma_z * eco.rng.standard_normal(params.n)
    path = [eco.step(st) for _ in range(horizon)]
    c = np.array([h.coupling for h in path])
    eff = np.array([h.efficiency for h in path])
    y = np.array([h.output for h in path])

    def first(cond: np.ndarray) -> float:
        return float(np.argmax(cond)) if cond.any() else float("nan")

    return dict(s_k=params.s_k, psi=params.psi, seed=seed, pre_coupling=c0, pre_efficiency=e0,
                pre_top1=float(np.mean([h.top1_wealth for h in tail])),
                t_half=first(c >= 0.5 * c0), t_90=first(c >= 0.9 * c0),
                eff_drop=e0 - eff[0], eff_gap_10=e0 - eff[10], eff_gap_25=e0 - eff[25],
                output_years_lost=float(np.sum((y0 - y) / y0)))


def e6(pool, quick: bool) -> pd.DataFrame:
    n = 3_000 if quick else BASE.n
    burn, horizon, pre = (150, 40, 20) if quick else (BURN, 120, 50)
    seeds = range(1 if quick else 3)
    sks = (0.3, 0.5) if quick else (0.3, 0.4, 0.5)
    psis = (0.90, 0.98) if quick else (0.90, 0.95, 0.98)
    jobs = [(replace(BASE, n=n, s_k=sk, psi=psi), s, burn, horizon, pre)
            for sk in sks for psi in psis for s in seeds]
    return pd.DataFrame(list(pool.map(_transition_metrics, jobs)))


# ---------------------------------------------------------------------------------------
# summary


def regime_name(tk: float, ta: float) -> str:
    return "none" if tk == 0 and ta == 0 else ("capital_income" if tk > 0 else "wealth")


def summary(s1, s2, s3, s4, s5, s6, tau_k, tau_a, target, achieved) -> str:
    L = []
    w = L.append
    w("# Simulation results (every number quoted in the article)")
    w("")
    w(f"Baseline parameters: {BASE}")
    w(f"Reduced-form capital exponent at a given wage gamma = {GAMMA:.4f}; "
      f"dispersion of reduced-form productivity = {SIGMA_EFF:.4f}")
    k_top1 = 1 - norm.cdf(norm.ppf(0.99) - SIGMA_EFF / (1 - GAMMA))
    w(f"Top-1% share of capital in the first-best allocation (analytic) = {k_top1:.4f}")
    w("")
    w("## E1 static allocation (lam = baseline)")
    m = s1[s1.lam == BASE.lam].groupby(["top1_target", "rho_target"]).mean(numeric_only=True).reset_index()
    for rho in (0.0, 0.5, 0.9, 1.0):
        sl = m[np.isclose(m.rho_target, rho)].sort_values("top1_target")
        if sl.empty:
            continue
        best = sl.loc[sl.efficiency.idxmax()]
        w(f"rho={rho:.2f}: efficiency at top1=0.04 {sl.efficiency.iloc[0]:.4f}; at top1=0.20 "
          f"{np.interp(0.20, sl.top1_target, sl.efficiency):.4f}; at top1=0.30 "
          f"{np.interp(0.30, sl.top1_target, sl.efficiency):.4f}; at top1=0.50 {sl.efficiency.iloc[-1]:.4f}; "
          f"max {best.efficiency:.4f} at top1={best.top1_target:.2f}")
    # how much does coupling matter, holding concentration at 0.25?
    at25 = m[np.isclose(m.top1_target, 0.28) | np.isclose(m.top1_target, 0.22)].groupby("rho_target").efficiency.mean()
    if len(at25):
        w(f"Efficiency at top1 ~ 0.25, rho from {at25.index.min():.2f} to {at25.index.max():.2f}: "
          f"{at25.iloc[0]:.4f} -> {at25.iloc[-1]:.4f}")
    # sign of d efficiency / d top1 by rho (average slope over 0.10..0.40)
    for rho in sorted(m.rho_target.unique()):
        sl = m[np.isclose(m.rho_target, rho) & (m.top1_target >= 0.10) & (m.top1_target <= 0.40)]
        if len(sl) > 2:
            slope = np.polyfit(sl.top1_target, sl.efficiency, 1)[0]
            w(f"  slope d(eff)/d(top1) over 0.10-0.40 at rho={rho:.2f}: {slope:+.4f} per unit share")
    w("")
    w("## E1 slices for looser credit (rho -> efficiency at top1 = 0.04 / 0.30 / 0.50)")
    ms = s1.groupby(["lam", "rho_target", "top1_target"]).efficiency.mean().reset_index()
    for lam in sorted(ms.lam.unique()):
        for rho in (0.0, 0.5, 0.9, 1.0):
            sl = ms[(ms.lam == lam) & np.isclose(ms.rho_target, rho)].sort_values("top1_target")
            if sl.empty:
                continue
            w(f"lam={lam:g} rho={rho:.1f}: {sl.efficiency.iloc[0]:.4f} / "
              f"{np.interp(0.30, sl.top1_target, sl.efficiency):.4f} / {sl.efficiency.iloc[-1]:.4f}")
    w("")
    w("## E2 dynamic steady states by rho_g (means over seeds)")
    g = s2.groupby("rho_g").mean(numeric_only=True)
    for rho_g, r in g.iterrows():
        w(f"rho_g={rho_g:.1f}: efficiency {r.efficiency:.4f}, coupling {r.coupling:.4f}, "
          f"capital coupling {r.capital_coupling:.4f}, top1 {r.top1_wealth:.4f}, top10 {r.top10_wealth:.4f}, "
          f"output {r.output:.1f}, wage {r.w:.4f}, R {r.R:.4f}, K/Y {r.K / r.output:.3f}, "
          f"constrained {r.constrained:.4f}")
    gs = s2.groupby("rho_g").efficiency.agg(["min", "max"])
    w("efficiency range across seeds: " + "; ".join(f"{i:.1f}: {a:.4f}-{b:.4f}" for i, (a, b) in gs.iterrows()))
    w("")
    w("## E3 taxes, steady state (means over seeds)")
    w(f"capital income tax tau_k = {tau_k:.3f}; revenue-neutral wealth tax tau_a = {tau_a:.5f} "
      f"(target revenue/output {target:.5f}, achieved {achieved:.5f})")
    s3 = s3.assign(regime=[regime_name(a, b) for a, b in zip(s3.tau_k, s3.tau_a)])
    t = s3.groupby("regime").mean(numeric_only=True)
    base = t.loc["none"]
    for name in ("none", "capital_income", "wealth"):
        r = t.loc[name]
        w(f"{name}: efficiency {r.efficiency:.4f}, coupling {r.coupling:.4f}, top1 {r.top1_wealth:.4f}, "
          f"top10 {r.top10_wealth:.4f}, output {r.output:.1f} ({100 * (r.output / base.output - 1):+.2f}% vs none), "
          f"K {r.K:.0f}, wage {r.w:.4f} ({100 * (r.w / base.w - 1):+.2f}%), revenue/output {r.revenue / r.output:.5f}")
    ci, wt = t.loc["capital_income"], t.loc["wealth"]
    w(f"wealth tax vs capital income tax: output {100 * (wt.output / ci.output - 1):+.2f}%, "
      f"efficiency {wt.efficiency - ci.efficiency:+.4f}, coupling {wt.coupling - ci.coupling:+.4f}, "
      f"top1 {wt.top1_wealth - ci.top1_wealth:+.4f}, wage {100 * (wt.w / ci.w - 1):+.2f}%, "
      f"K {100 * (wt.K / ci.K - 1):+.2f}%")
    w("")
    w("## E4 re-ranking shock at t = 0 (means over seeds)")
    s4 = s4.assign(regime=[regime_name(a, b) for a, b in zip(s4.tau_k, s4.tau_a)])
    p = s4.groupby(["regime", "t"]).mean(numeric_only=True).reset_index()
    for name in ("none", "capital_income", "wealth"):
        q = p[p.regime == name].set_index("t")
        pre = t.loc[name] if name in t.index else None
        target_c = pre.coupling if pre is not None else np.nan
        half = q.coupling >= 0.5 * target_c
        t_half = int(q.index[half.values.argmax()]) if half.any() else -1
        w(f"{name}: coupling t=0 {q.coupling.iloc[0]:.4f}, t=10 {q.coupling.loc[10]:.4f}, "
          f"t=25 {q.coupling.loc[min(25, q.index.max())]:.4f}, t=50 {q.coupling.loc[min(50, q.index.max())]:.4f}, "
          f"end {q.coupling.iloc[-1]:.4f}; years to half of pre-shock coupling {t_half}; "
          f"efficiency t=0 {q.efficiency.iloc[0]:.4f}, t=25 {q.efficiency.loc[min(25, q.index.max())]:.4f}, "
          f"end {q.efficiency.iloc[-1]:.4f}; top1 t=0 {q.top1.iloc[0]:.4f}, t=25 {q.top1.loc[min(25, q.index.max())]:.4f}")
    w("")
    w("## E5 persistence of productivity x inheritance of ability (means over seeds)")
    g5 = s5.groupby(["psi", "rho_g"]).mean(numeric_only=True)
    for psi in sorted(s5.psi.unique()):
        r0, r1 = g5.loc[(psi, 0.0)], g5.loc[(psi, 1.0)]
        w(f"psi={psi:.2f}: rho_g=0 efficiency {r0.efficiency:.4f}, coupling {r0.coupling:.4f}, top1 {r0.top1_wealth:.4f}; "
          f"rho_g=1 efficiency {r1.efficiency:.4f}, coupling {r1.coupling:.4f}, top1 {r1.top1_wealth:.4f}; "
          f"gap efficiency {r1.efficiency - r0.efficiency:+.4f}, output {100 * (r1.output / r0.output - 1):+.2f}%")
    w("")
    w("## E6 transition after re-ranking, no tax: sensitivity to s_k and psi (means over seeds)")
    g6 = s6.groupby(["s_k", "psi"]).mean(numeric_only=True)
    for (sk, psi), r in g6.iterrows():
        w(f"s_k={sk:.1f} psi={psi:.2f}: pre-shock coupling {r.pre_coupling:.4f}, efficiency {r.pre_efficiency:.4f}, "
          f"top1 {r.pre_top1:.4f}; years to half coupling {r.t_half:.1f}, to 90% {r.t_90:.1f}; "
          f"efficiency drop at t=0 {r.eff_drop:.4f}, gap at t=10 {r.eff_gap_10:.4f}, at t=25 {r.eff_gap_25:.4f}; "
          f"cumulative output lost over 120 years {r.output_years_lost:.3f} years of pre-shock output")
    w(f"range of years to half coupling across the grid: {g6.t_half.min():.1f} to {g6.t_half.max():.1f}; "
      f"to 90%: {g6.t_90.min():.1f} to {g6.t_90.max():.1f}; output-years lost: "
      f"{g6.output_years_lost.min():.3f} to {g6.output_years_lost.max():.3f}")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tau_k = 0.25
    with ProcessPoolExecutor(args.workers) as pool:
        s1 = e1(pool, args.quick)
        s2 = e2(pool, args.quick)
        tau_a, target, achieved = revenue_neutral_wealth_tax(pool, tau_k, args.quick)
        s3 = e3(pool, args.quick, tau_k, tau_a)
        s4 = e4(pool, args.quick, tau_k, tau_a)
        s5 = e5(pool, args.quick)
        s6 = e6(pool, args.quick)
    s1.to_csv(args.out / "sim_static_surface.csv", index=False, float_format="%.6f")
    s2.to_csv(args.out / "sim_inheritance.csv", index=False, float_format="%.6f")
    s3.to_csv(args.out / "sim_taxes.csv", index=False, float_format="%.6f")
    s4.to_csv(args.out / "sim_reranking.csv", index=False, float_format="%.6f")
    s5.to_csv(args.out / "sim_persistence.csv", index=False, float_format="%.6f")
    s6.to_csv(args.out / "sim_transition_sensitivity.csv", index=False, float_format="%.6f")
    text = summary(s1, s2, s3, s4, s5, s6, tau_k, tau_a, target, achieved)
    (args.out / "model_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
