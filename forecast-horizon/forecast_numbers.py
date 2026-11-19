"""Every number in the article, printed in order, plus the tables in results/ that figures.py draws.

python fetch_data.py               # once, for the reanalysis sample
python forecast_numbers.py > output.txt
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

import chaos_core as cc

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DAYS = cc.DAYS_PER_UNIT


def lorenz63_section():
    print("== Lorenz's rounding (Lorenz 1963 model)")
    dt = 0.01
    x = cc.run(cc.lorenz63, np.array([1.0, 1.0, 1.0]), dt, 5000)
    lam = cc.lyapunov(cc.lorenz63, x, dt, 200_000, seed=1)
    print(f"largest Lyapunov exponent: {lam:.3f} per time unit; errors double every {math.log(2) / lam:.2f} time units")

    clim, _ = cc.sample(cc.lorenz63, x, dt, 100_000, 10)
    rng = np.random.default_rng(2)
    i, j = rng.integers(0, len(clim), (2, 20_000))
    typical = float(np.sqrt(((clim[i] - clim[j]) ** 2).sum(axis=1)).mean())
    print(f"typical distance between two states on the attractor: {typical:.1f}")

    start = np.round(x, 6)
    rounded = np.round(start, 3)
    print(f"state kept to six decimals: {start[0]:.6f}, {start[1]:.6f}, {start[2]:.6f}")
    print(f"state rounded to three:     {rounded[0]:.3f}, {rounded[1]:.3f}, {rounded[2]:.3f}")
    print(f"initial difference: {cc.norm(start - rounded):.6f}")
    steps = 4000
    a, b = start.copy(), rounded.copy()
    rows = [(0.0, a[0], b[0], cc.norm(a - b))]
    for k in range(1, steps + 1):
        a = cc.rk4(cc.lorenz63, a, dt)
        b = cc.rk4(cc.lorenz63, b, dt)
        rows.append((k * dt, a[0], b[0], cc.norm(a - b)))
    r = pd.DataFrame(rows, columns=["t", "x_full", "x_rounded", "distance"])
    r.to_csv(RESULTS / "rounding.csv", index=False)
    t1 = float(r.t[(r.distance >= 1).idxmax()])
    t_half = float(r.t[(r.distance >= 0.5 * typical).idxmax()])
    print(f"the two runs first differ by 1 unit at t = {t1:.1f}, and by half the typical distance at t = {t_half:.1f}")

    level = 0.5 * typical
    starts = clim[rng.choice(len(clim), 300, replace=False)]
    eps = 10.0 ** -np.arange(1, 13)
    hz, _, _ = cc.horizons(cc.lorenz63, starts, np.ones(3), eps, dt, 3500, 5, slice(None), level / math.sqrt(3), 3)
    gain = np.diff(hz)  # eps falls from 1e-1 to 1e-12, so each step is a tenfold reduction
    print(
        f"each tenfold reduction of the initial error adds {gain.mean():.2f} time units on average "
        f"(range {gain.min():.2f} to {gain.max():.2f}); ln(10)/exponent = {math.log(10) / lam:.2f}"
    )
    return lam


def one_scale_section():
    print("\n== One scale (Lorenz 1996 model, 40 variables, forcing 8)")
    dt = 0.01
    rng = np.random.default_rng(4)
    x = cc.run(cc.lorenz96, 8 + 0.1 * rng.standard_normal(40), dt, 5000)
    clim, x = cc.sample(cc.lorenz96, x, dt, 200_000, 50)
    sd, mean = cc.fixed(clim.std()), cc.fixed(clim.mean())
    print(f"climatological mean {mean:.2f}, standard deviation {sd:.2f}; one time unit = {DAYS:.0f} days")
    lam = cc.lyapunov(cc.lorenz96, x, dt, 100_000, seed=5)
    print(
        f"largest Lyapunov exponent: {lam:.3f} per time unit; small errors double every "
        f"{math.log(2) / lam * DAYS:.2f} days"
    )
    sat = math.sqrt(2) * sd
    level = 0.5 * sat
    print(f"error of a random guess from the climate: {sat:.2f}; horizon = time to reach half of it, {level:.2f}")

    starts = clim[rng.choice(len(clim), 300, replace=False)]
    eps = 10.0 ** -np.arange(8, 0, -1)
    hz, times, curves = cc.horizons(cc.lorenz96, starts, sd, eps, dt, 1500, 10, slice(None), level, 6)
    for e, h in zip(eps, hz, strict=True):
        print(f"  initial error {e:.0e} of the natural variability: horizon {h * DAYS:5.1f} days")
    g = cc.days_per_decade(eps, hz)
    print(f"days gained by each tenfold reduction: mean {g.mean():.1f} (range {g.min():.1f} to {g.max():.1f})")
    print(f"ln(10) / exponent, in days: {math.log(10) / lam * DAYS:.1f}")
    pd.DataFrame(
        {f"eps_{e:.0e}": c for e, c in zip(eps, curves, strict=True)} | {"days": times * DAYS, "sat": sat}
    ).to_csv(RESULTS / "one_scale_growth.csv", index=False)
    return clim, mean, sd, eps, hz, lam


def two_scale_section():
    print("\n== Two scales (Lorenz 1996 two-scale model: 36 slow and 360 fast variables)")
    m = cc.TwoScale()
    dt = 0.005
    rng = np.random.default_rng(7)
    s0 = np.concatenate([rng.standard_normal(m.k), 0.1 * rng.standard_normal(m.k * m.j)])
    s = cc.run(m, s0, dt, 4000)
    clim, s = cc.sample(m, s, dt, 40_000, 50)
    sdx, sdy = cc.fixed(clim[:, : m.k].std()), cc.fixed(clim[:, m.k :].std())
    print(f"standard deviation: slow variables {sdx:.2f}, fast variables {sdy:.3f}")
    lam = cc.lyapunov(m, s, dt, 4000, seed=8)
    print(
        f"largest Lyapunov exponent of the full system: {lam:.1f} per time unit; the fastest errors double every "
        f"{math.log(2) / lam * DAYS * 24:.1f} hours"
    )
    satx = math.sqrt(2) * sdx
    level = 0.5 * satx
    starts = clim[rng.choice(len(clim), 120, replace=False)]
    eps = 10.0 ** -np.arange(8, -1, -1)
    out = {}
    for mode in ("slow", "fast"):
        scales = np.concatenate(
            [np.full(m.k, sdx if mode == "slow" else 0.0), np.full(m.k * m.j, sdy if mode == "fast" else 0.0)]
        )
        hz, _, _ = cc.horizons(m, starts, scales, eps, dt, 2000, 10, slice(0, m.k), level, 9 if mode == "slow" else 10)
        out[mode] = hz
        print(f"  initial error in the {mode} variables only:")
        for e, h in zip(eps, hz, strict=True):
            print(f"    {e:.0e} of their natural variability: horizon of the slow variables {h * DAYS:5.1f} days")
        g = cc.days_per_decade(eps, hz)
        print(
            f"    days per tenfold reduction: from 1e-01 to 1e-02 {g[-2]:.1f}; "
            f"below 1e-03, mean {g[:5].mean():.1f} (range {g[:5].min():.1f} to {g[:5].max():.1f})"
        )
        print(
            f"    gain from 1e-03 to 1e-08 (a hundred thousand times more precise): "
            f"{(out[mode][0] - out[mode][5]) * DAYS:.1f} days"
        )
    return eps, out


def ensemble_section(clim, mean, sd):
    print("\n== Ensembles (one-scale model, perfect model)")
    rng = np.random.default_rng(11)
    truths = clim[rng.choice(len(clim), 200, replace=False)]
    init_err, members = 0.05, 50
    rows, (truth, single, ens) = cc.ensemble_experiment(
        cc.lorenz96, truths, mean, sd, init_err, members, 0.01, 600, 10, 12
    )
    e = pd.DataFrame(rows, columns=["t", "rmse_single", "rmse_mean", "spread", "acc_single", "acc_mean", "crpss"])
    e["days"] = e.t * DAYS
    e.to_csv(RESULTS / "ensemble_skill.csv", index=False)
    print(f"200 cases, {members} members, starting error {init_err:.0%} of the natural variability, 30 days")

    def first(col, below):
        hit = e[e[col] < below]
        return float(hit.days.iloc[0]) if len(hit) else float("nan")

    print(
        f"anomaly correlation below 0.6: single forecast at day {first('acc_single', 0.6):.1f}, "
        f"ensemble mean at day {first('acc_mean', 0.6):.1f}"
    )
    print(
        f"single forecast worse than forecasting the climate average (error above {sd:.2f}) from day "
        f"{float(e.days[e.rmse_single > sd].iloc[0]):.1f}"
    )
    print(
        f"ensemble mean error at day 30: {e.rmse_mean.iloc[-1] / sd:.2f} of the natural variability; "
        f"single forecast {e.rmse_single.iloc[-1] / sd:.2f}"
    )
    print(
        "probability skill against climatology (CRPS skill score) at days 5, 10, 15, 20: "
        + ", ".join(f"{float(e.crpss[np.isclose(e.days, d)].iloc[0]):.2f}" for d in (5, 10, 15, 20))
    )
    print(
        f"CRPS skill score first below 0.05 at day {first('crpss', 0.05):.1f}, below 0.01 at day "
        f"{first('crpss', 0.01):.1f}"
    )
    ratio = (e.spread / e.rmse_mean)[e.days >= 1]
    print(f"spread / error of the ensemble mean, days 1 to 30: {ratio.min():.2f} to {ratio.max():.2f}")
    return e


def plume_section(mean, sd, clim):
    """One case, kept member by member, for the plume figure."""
    rng = np.random.default_rng(13)
    truth = clim[rng.integers(len(clim))].copy()
    s = 0.05 * sd
    analysis = truth + rng.standard_normal(40) * s
    ens = analysis[None, :] + rng.standard_normal((50, 40)) * s
    single = analysis.copy()
    rows = []
    for i in range(601):
        if i > 0:
            truth = cc.rk4(cc.lorenz96, truth, 0.01)
            single = cc.rk4(cc.lorenz96, single, 0.01)
            ens = cc.rk4(cc.lorenz96, ens, 0.01)
        if i % 5 == 0:
            rows.append([i * 0.01 * DAYS, truth[0], single[0], *ens[:, 0]])
    cols = ["days", "truth", "single"] + [f"m{k}" for k in range(50)]
    pd.DataFrame(rows, columns=cols).to_csv(RESULTS / "ensemble_plume.csv", index=False)


def analogue_section():
    print("\n== The real atmosphere: analogues (NCEP/NCAR reanalysis sample, 2013-2014)")
    air, lat, lon, hours = cc.load_air(HERE / "data" / "air_temperature.nc")
    a, doy = cc.daily_anomalies(air, lat, hours)
    print(
        f"{a.shape[0]} days, {air.shape[1]} x {air.shape[2]} grid points from {lat.min():.0f} to {lat.max():.0f} N "
        f"and {360 - lon.max():.0f} to {360 - lon.min():.0f} W; near-surface air temperature, daily means, "
        f"seasonal cycle removed"
    )
    sat = cc.saturation(a)
    curves, ok, d = cc.analogue_divergence(a, doy)
    print(
        f"saturation (two unrelated days): {sat:.2f} K; median same-season pair at least 30 days apart "
        f"{np.median(d[ok]) / sat:.2f} of it"
    )
    start = curves[:, 0] / sat
    print(
        f"closest analogue of each day ({len(curves)} days): median distance {np.median(start):.2f} of saturation, "
        f"closest of all {start.min():.2f}"
    )
    real = cc.relative_curve(curves, sat)
    late = real[10:].mean()
    print(
        "pooled distance of analogue pairs by day, as a share of saturation: "
        + ", ".join(f"day {k} {real[k]:.2f}" for k in (0, 1, 2, 3, 5, 10, 20))
    )
    print(f"level after day 10: {late:.2f}; first reaches 95% of it on day {int(np.argmax(real >= 0.95 * late))}")
    best = curves[start <= np.quantile(start, 0.1)]
    b = cc.relative_curve(best, sat)
    print(
        f"best tenth of analogues ({len(best)}): start {b[0]:.2f}, day 3 {b[3]:.2f}, after day 10 {b[10:].mean():.2f}"
    )

    r = 20
    p, share = cc.leading_components(a, r)
    psat = cc.saturation(p)
    pc_curves, _, _ = cc.analogue_divergence(p, doy)
    pc_real = cc.relative_curve(pc_curves, psat)
    sims = []
    for rep in range(5):
        q = cc.var1_surrogate(p, len(p), seed=20 + rep)
        qc, _, _ = cc.analogue_divergence(q, doy)
        sims.append(cc.relative_curve(qc, cc.saturation(q)))
    sim = np.mean(sims, axis=0)
    print(
        f"in the leading {r} patterns ({share:.0%} of the variance), real against a linear stochastic copy "
        "(memory plus random forcing, no chaos):"
    )
    print("  real:      " + ", ".join(f"day {k} {pc_real[k]:.2f}" for k in (0, 1, 2, 3, 5, 10)))
    print("  surrogate: " + ", ".join(f"day {k} {sim[k]:.2f}" for k in (0, 1, 2, 3, 5, 10)))
    print(f"  largest gap over days 0 to 20: {np.abs(pc_real - sim).max():.2f}")
    pd.DataFrame(
        {"day": np.arange(len(real)), "real": real, "best_tenth": b, "pc_real": pc_real, "pc_surrogate": sim}
    ).to_csv(RESULTS / "analogues.csv", index=False)


def literature_section():
    print("\n== Doubling times of small errors reported over the years")
    lit = pd.read_csv(HERE / "literature" / "doubling_times.csv")
    for row in lit.itertuples():
        hi = f" to {row.doubling_days_high:.2f}" if not pd.isna(row.doubling_days_high) else ""
        print(f"  {row.year}: {row.source}, {row.system}: {row.doubling_days:g}{hi} days")


def main():
    RESULTS.mkdir(exist_ok=True)
    lorenz63_section()
    clim, mean, sd, eps, hz, _ = one_scale_section()
    eps2, two = two_scale_section()
    pd.DataFrame({"model": "one scale", "start": "all", "eps": eps, "days": hz * DAYS}).pipe(
        lambda one: pd.concat(
            [one]
            + [pd.DataFrame({"model": "two scales", "start": k, "eps": eps2, "days": v * DAYS}) for k, v in two.items()]
        )
    ).to_csv(RESULTS / "precision.csv", index=False)
    ensemble_section(clim, mean, sd)
    plume_section(mean, sd, clim)
    analogue_section()
    literature_section()


if __name__ == "__main__":
    main()
