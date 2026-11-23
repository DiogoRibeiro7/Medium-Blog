"""Print every number used in the article, in the order the article uses them, and save the arrays the figures need.

python bayes_numbers.py > output.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

import bayes_core as B

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SEED = 20261123
A, Bp = B.PROTOCOL_PRIOR
X, N = B.VACCINE_CASES, B.TOTAL_CASES


def rngs(k):
    return [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(k)]


R = rngs(12)


def pct(v, d=1):
    return f"{100 * float(v):.{d}f}%"


def head(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# ----------------------------------------------------------------------------------------------- 1 prior


def vaccine_prior():
    head("1. The vaccine trial: what the protocol's prior says before any case is counted")
    print(f"Prior on theta (share of cases in the vaccine arm): Beta({A}, {Bp:g}); worth a + b = {A + Bp:.6f} cases")
    mean = A / (A + Bp)
    print(f"Prior mean of theta {mean:.4f}; VE at that theta {pct(B.ve_from_theta(mean))}")
    med = stats.beta.ppf(0.5, A, Bp)
    print(f"Prior median of theta {med:.4f}; prior median VE {pct(B.ve_from_theta(med))}")
    for ve in (0.3, 0.5, 0.9, 0.0):
        p = stats.beta.cdf(float(B.theta_from_ve(ve)), A, Bp)
        print(f"Prior P(VE > {ve:.0%}) = {pct(p)}")
    print(f"Prior P(VE < 0%) (vaccine harmful) = {pct(1 - stats.beta.cdf(0.5, A, Bp))}")
    lo, hi = B.ve_from_theta(stats.beta.ppf([0.975, 0.025], A, Bp))
    print(f"Prior central 95% interval for VE: {pct(lo)} to {pct(hi)}")
    p_neg100 = 1 - stats.beta.cdf(float(B.theta_from_ve(-1.0)), A, Bp)
    print(f"Prior P(VE < -100%) (vaccine at least doubles the risk) = {pct(p_neg100)}")
    p_neg300 = 1 - stats.beta.cdf(float(B.theta_from_ve(-3.0)), A, Bp)
    print(f"Prior P(VE < -300%) (vaccine at least quadruples the risk) = {pct(p_neg300)}")
    pp = stats.betabinom(N, A, Bp)
    print(f"Prior predictive of vaccine-arm cases among {N}: mean {pp.mean():.1f}")
    print(f"  P(x <= {X}) = {pct(pp.cdf(X))}; P(x >= {N // 2}) = {pct(pp.sf(N // 2 - 1))}")
    print(f"  central 90% interval {int(pp.ppf(0.05))} to {int(pp.ppf(0.95))}")
    rng = R[0]
    draws = B.ve_from_theta(rng.beta(A, Bp, 200_000))
    np.savez_compressed(RES / "prior_ve.npz", ve=draws, x=np.arange(N + 1), pmf=pp.pmf(np.arange(N + 1)))


# ----------------------------------------------------------------------------------------------- 2 posterior three ways


def vaccine_posterior():
    head("2. The posterior three ways: grid, sampler and conjugate algebra")
    pa, pb = B.conjugate(X, N, A, Bp)
    print(f"Data: {X} vaccine-arm cases of {N}. Conjugate posterior Beta({pa:.6f}, {pb:g})")
    qs = [0.5, 0.025, 0.975]
    exact = B.ve_quantiles(X, N, A, Bp, qs)
    print(
        f"Conjugate, equal follow-up: VE median {pct(exact[0], 2)},"
        f" 95% interval {pct(exact[1], 2)} to {pct(exact[2], 2)}"
    )
    th, w = B.grid_posterior(X, N, A, Bp)
    grid = B.ve_from_theta(B.weighted_quantile(th, w, [0.5, 0.975, 0.025]))
    print(
        f"Grid of {th.size:,} points:      VE median {pct(grid[0], 2)},"
        f" 95% interval {pct(grid[1], 2)} to {pct(grid[2], 2)}"
    )
    rng = R[1]
    draws, acc = B.metropolis_theta(
        np.full(4, X), N, A, Bp, rng=rng, iters=22_000, step=0.6, start=np.array([-3.0, -1.0, 0.0, 1.0]), burn=2_000
    )
    ve_s = B.ve_from_theta(draws)
    smp = np.quantile(ve_s, qs)
    print(
        f"Metropolis, 4 chains x 20,000:  VE median {pct(smp[0], 2)}, 95% interval {pct(smp[1], 2)} to {pct(smp[2], 2)}"
        f"  (acceptance {acc:.2f}, split R-hat {B.split_rhat(draws):.4f})"
    )
    diffs = np.abs(np.concatenate([grid - exact, smp - exact]))
    print(f"Largest gap from the conjugate answer: {100 * diffs.max():.2f} percentage points")
    adj = B.ve_quantiles(X, N, A, Bp, qs, r=B.SURV_RATIO)
    point = 1 - (X / 2.214) / ((N - X) / 2.222)
    print(
        f"Adjusted for surveillance time (2.214 vs 2.222 thousand person-years): point estimate {pct(point)},"
        f" posterior median {pct(adj[0])}, 95% credible interval {pct(adj[1])} to {pct(adj[2])}"
    )
    print("Published (Polack et al. 2020): 95.0%, 95% credible interval 90.3% to 97.6%")
    miss = stats.beta.sf(float(B.theta_from_ve(B.VE_BAR, B.SURV_RATIO)), pa, pb)
    print(
        f"Posterior P(VE <= 30%) = {miss:.1e}; P(VE > 90%) = {pct(B.prob_ve_above(X, N, A, Bp, 0.9, B.SURV_RATIO), 2)}"
    )
    np.savez_compressed(RES / "posterior_ve.npz", theta=th, w=w, sampler=ve_s[::5].ravel())


# ----------------------------------------------------------------------------------------------- 3 accumulation


def accumulation():
    head("3. Watching the posterior as the cases accumulate")
    print("Planned looks and the largest vaccine-arm count that still counts as success (design, equal follow-up):")
    bounds = {}
    for n in B.LOOKS:
        thr = B.THRESHOLD_FINAL if n == B.LOOKS[-1] else B.THRESHOLD_INTERIM
        x = B.success_boundary(n, threshold=thr)
        bounds[n] = x
        print(
            f"  {n:>3} cases: success if at most {x:>2} in the vaccine arm (P > {thr});"
            f" VE at the boundary {pct(1 - x / (n - x))}"
        )
    print("Published VE thresholds at 32, 92 and 120 cases: 76.9%, 62.7% and 58.8%")

    rng = R[2]
    orders = B.random_orders(rng, 5_000)
    cum = np.cumsum(orders, axis=1)
    n = np.arange(1, N + 1)
    # order invariance on the grid, one case at a time
    th, w0 = B.grid_posterior(0, 0, A, Bp)
    batch = B.grid_posterior(X, N, A, Bp)[1]
    worst = 0.0
    for k in range(3):
        w = w0.copy()
        for c in orders[k]:
            w = B.grid_update(th, w, bool(c))
        worst = max(worst, float(np.abs(w - batch).max() / batch.max()))
    print(
        f"Three random orders updated one case at a time on the grid vs all at once: largest relative gap {worst:.1e}"
    )

    hyper = stats.hypergeom(N, X, 32)
    print(f"Chance that a random order puts 7 or 8 of the 8 vaccine-arm cases among the first 32: {hyper.sf(6):.1e}")
    for look in B.LOOKS[:-1]:
        share = np.mean(cum[:, look - 1] <= bounds[look])
        print(f"  share of 5,000 random orders that pass the {look}-case look: {pct(share)}")
    p_above = B.prob_ve_above(cum, n, A, Bp)
    first = np.argmax(p_above > B.THRESHOLD_INTERIM, axis=1) + 1
    print(
        f"If the rule were checked after every case: first crossing at case {int(np.median(first))} (median),"
        f" range {first.min()} to {first.max()}"
    )
    for look in (10, 32, 62, 94, 170):
        lo = B.ve_quantiles(cum[:, look - 1], look, A, Bp, 0.025)
        hi = B.ve_quantiles(cum[:, look - 1], look, A, Bp, 0.975)
        med = B.ve_quantiles(cum[:, look - 1], look, A, Bp, 0.5)
        print(
            f"  after {look:>3} cases: median posterior VE {pct(np.median(med))};"
            f" 95% interval width {100 * np.median(hi - lo):.1f} points (median over orders),"
            f" lower end {pct(np.median(lo))}"
        )

    print("Decision rule under a vaccine that is exactly 30% effective (200,000 simulated trials):")
    interims = [B.THRESHOLD_INTERIM] * 4 + [B.THRESHOLD_FINAL]
    fs = B.false_success_rate(R[3], B.LOOKS, interims)
    print(f"  protocol, five looks: {pct(fs, 2)} called a success")
    fs1 = B.false_success_rate(R[3], (164,), (B.THRESHOLD_FINAL,))
    print(f"  final look only (164 cases, 0.986): {pct(fs1, 2)}")
    every = tuple(range(1, 165))
    fse = B.false_success_rate(R[3], every, [B.THRESHOLD_INTERIM] * 163 + [B.THRESHOLD_FINAL])
    print(f"  checked after every case (0.995, then 0.986 at 164): {pct(fse, 2)}")
    for ve in (0.6, 0.95):
        pw = B.false_success_rate(R[4], B.LOOKS, interims, ve=ve)
        print(f"  protocol, five looks, true VE {ve:.0%}: {pct(pw, 1)} called a success")

    keep = orders[:200]
    kc = np.cumsum(keep, axis=1)
    np.savez_compressed(
        RES / "accumulation.npz",
        n=n,
        p_above=B.prob_ve_above(kc, n, A, Bp),
        med=B.ve_quantiles(kc, n, A, Bp, 0.5),
        lo=B.ve_quantiles(kc, n, A, Bp, 0.025),
        hi=B.ve_quantiles(kc, n, A, Bp, 0.975),
        looks=np.array(B.LOOKS),
    )


# ----------------------------------------------------------------------------------------------- 4 prior sensitivity

PRIORS = {
    "protocol": (A, Bp),
    "flat": (1.0, 1.0),
    # centred on VE = 0 (theta = 0.5) and worth 100 cases
    "sceptic": (50.0, 50.0),
    # centred on VE = 90% (theta = 1/11) and worth 20 cases
    "enthusiast": (20.0 / 11.0, 200.0 / 11.0),
}


def sensitivity():
    head("4. How much the prior matters, and when")
    rng = R[5]
    order = B.random_orders(rng, 1)[0]
    cum = np.cumsum(order)
    n = np.arange(1, N + 1)
    print(
        f"One random order of arrival (seed {SEED}); vaccine-arm cases after 10/32/62/94/170: "
        f"{[int(cum[k - 1]) for k in (10, 32, 62, 94, 170)]}"
    )
    out = {}
    for name, (a, b) in PRIORS.items():
        pm = a / (a + b)
        print(
            f"{name:>10}: Beta({a:.4g}, {b:.4g}), worth {a + b:.1f} cases,"
            f" prior VE at mean theta {pct(B.ve_from_theta(pm))}"
        )
        for look in (10, 32, 94, 170):
            q = B.ve_quantiles(cum[look - 1], look, a, b, [0.5, 0.025, 0.975])
            print(f"            after {look:>3}: VE median {pct(q[0])}, 95% interval {pct(q[1])} to {pct(q[2])}")
        wgt = (a + b) / (a + b + N)
        post_mean = (a + X) / (a + b + N)
        blend = wgt * pm + (1 - wgt) * X / N
        print(
            f"            prior weight at {N} cases {pct(wgt)}; posterior mean theta {post_mean:.5f}"
            f" = {wgt:.4f} x {pm:.4f} + {1 - wgt:.4f} x {X / N:.4f} = {blend:.5f}"
        )
        out[name] = np.vstack([B.ve_quantiles(cum, n, a, b, q) for q in (0.5, 0.025, 0.975)])
    for look in (32, 170):
        meds = [B.ve_quantiles(cum[look - 1], look, a, b, 0.5) for a, b in PRIORS.values()]
        print(f"Spread of the four posterior medians after {look} cases: {100 * (max(meds) - min(meds)):.1f} points")
    np.savez_compressed(RES / "sensitivity.npz", n=n, **out)


# ----------------------------------------------------------------------------------------------- 5 O-rings


def orings():
    head("5. The O-rings: prior predictive, posterior and extrapolation to 31 F")
    f = B.load_flights(HERE / "data" / "rdatasets" / "csv" / "vcd" / "SpaceShuttle.csv")
    print(f"{f.temp.size} flights with a known count; launch temperatures {f.temp.min():.0f} to {f.temp.max():.0f} F")
    print(f"{int(f.distressed.sum())} distressed O-rings in all, on {int((f.distressed > 0).sum())} flights")
    cold = f.temp < 65
    print(f"Flights below 65 F: {int(cold.sum())}, all with distress: {bool((f.distressed[cold] > 0).all())}")
    print(
        f"Coldest flight in the data {f.temp.min():.0f} F; Challenger at {B.LAUNCH_F:.0f} F,"
        f" {f.temp.min() - B.LAUNCH_F:.0f} degrees colder"
    )

    rng = R[6]
    for prior in (B.VAGUE, B.WEAKLY_INFORMATIVE):
        for t in (70.0, B.LAUNCH_F):
            s = B.prior_extreme_share(prior, rng, temp_f=t)
            print(f"Prior {prior.name:>18}: share of draws with chance at {t:.0f} F below 1% or above 99%: {pct(s)}")
    a_w, b_w = B.WEAKLY_INFORMATIVE.draw(rng, 400_000)
    q70 = np.quantile(B.p_at(a_w, b_w, 70.0), [0.05, 0.5, 0.95])
    print(
        f"Weakly informative prior, chance per ring at 70 F: median {pct(q70[1])},"
        f" 90% interval {pct(q70[0])} to {pct(q70[2])}"
    )

    m = B.mle(f)
    print(
        f"Maximum likelihood: log-odds at 70 F {m[0]:.3f}, slope {m[1]:.3f} per 10 F ({m[1] / 10:.4f} per F);"
        f" chance per ring at 31 F {pct(B.p_at(m[0], m[1], B.LAUNCH_F))}"
    )

    temps = np.linspace(25, 85, 121)
    bands = {}
    for prior in (B.WEAKLY_INFORMATIVE, B.VAGUE):
        g = B.grid_posterior_2d(f, prior)
        a_d, b_d = B.grid_draws(g, R[7], 200_000)
        print(f"Posterior, {prior.name} prior (grid 500 x 500, edge mass {B.grid_edge_mass(g):.1e}):")
        bq = np.quantile(b_d, [0.05, 0.5, 0.95])
        print(
            f"  slope per 10 F: median {bq[1]:.2f}, 90% interval {bq[0]:.2f} to {bq[2]:.2f};"
            f" P(slope < 0) = {pct(np.mean(b_d < 0))}"
        )
        print(f"  odds of distress per ring multiply by {np.exp(-bq[1]):.2f} (median) for every 10 F colder")
        for t in (81.0, 70.0, 53.0, B.LAUNCH_F):
            pq = np.quantile(B.p_at(a_d, b_d, t), [0.05, 0.5, 0.95])
            print(f"  chance per ring at {t:.0f} F: median {pct(pq[1])}, 90% interval {pct(pq[0])} to {pct(pq[2])}")
        p31 = B.p_at(a_d, b_d, B.LAUNCH_F)
        print(
            f"  at 31 F: expected distressed rings of 6 {6 * p31.mean():.1f};"
            f" posterior predictive P(at least one of 6) {pct(np.mean(1 - (1 - p31) ** 6))}"
        )
        curves = B.p_at(a_d[:4000, None], b_d[:4000, None], temps[None, :])
        bands[prior.name] = np.quantile(curves, [0.05, 0.5, 0.95], axis=0)

    ca, cb, acc = B.metropolis_2d(
        f, B.WEAKLY_INFORMATIVE, rng=R[8], chains=4, iters=26_000, burn=2_000, step=(0.45, 0.4)
    )
    p31s = B.p_at(ca, cb, B.LAUNCH_F)
    q = np.quantile(p31s, [0.05, 0.5, 0.95])
    print(
        f"Metropolis cross-check (weakly informative, 4 x 24,000): chance per ring at 31 F median {pct(q[1])},"
        f" 90% interval {pct(q[0])} to {pct(q[2])}; acceptance {acc:.2f};"
        f" split R-hat slope {B.split_rhat(cb):.4f}, intercept {B.split_rhat(ca):.4f}"
    )

    print("How the answer at 31 F depends on the prior's slope scale (intercept prior as weakly informative):")
    for sd in (0.5, 1.0, 2.0, 5.0, 10.0):
        pr = B.NormalPrior(f"slope sd {sd}", -3.0, 1.5, 0.0, sd)
        g = B.grid_posterior_2d(f, pr)
        a_d, b_d = B.grid_draws(g, R[9], 100_000)
        p31 = np.quantile(B.p_at(a_d, b_d, B.LAUNCH_F), [0.05, 0.5, 0.95])
        p70 = np.quantile(B.p_at(a_d, b_d, 70.0), [0.05, 0.5, 0.95])
        print(
            f"  slope sd {sd:>4}: at 31 F median {pct(p31[1])} (90%: {pct(p31[0])} to {pct(p31[2])});"
            f" at 70 F median {pct(p70[1])} (90%: {pct(p70[0])} to {pct(p70[2])})"
        )

    inc = f.distressed > 0
    fi = B.Flights(f.temp[inc], f.distressed[inc])
    g = B.grid_posterior_2d(fi, B.WEAKLY_INFORMATIVE)
    a_d, b_d = B.grid_draws(g, R[10], 200_000)
    bq = np.quantile(b_d, [0.05, 0.5, 0.95])
    print(
        f"Only the {inc.sum()} flights with distress (weakly informative prior): slope median {bq[1]:.2f},"
        f" 90% interval {bq[0]:.2f} to {bq[2]:.2f}; P(slope < 0) = {pct(np.mean(b_d < 0))};"
        f" chance per ring at 31 F median {pct(np.median(B.p_at(a_d, b_d, B.LAUNCH_F)))}"
    )

    a_v, b_v = B.VAGUE.draw(R[11], 60)
    a_i, b_i = B.WEAKLY_INFORMATIVE.draw(R[11], 60)
    np.savez_compressed(
        RES / "orings.npz",
        temp=f.temp,
        distressed=f.distressed,
        temps=temps,
        wi=bands["weakly informative"],
        vague=bands["vague"],
        prior_vague=B.p_at(a_v[:, None], b_v[:, None], temps[None, :]),
        prior_wi=B.p_at(a_i[:, None], b_i[:, None], temps[None, :]),
    )
    return f


def fake_data_check(f):
    head("6. Checking the O-ring model on fake data with a known answer")
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(13)[12])
    k = 400
    a_true, b_true = B.WEAKLY_INFORMATIVE.draw(rng, k)
    qa, qb = [], []
    for i in range(k):
        y = rng.binomial(B.RINGS, B.p_at(a_true[i], b_true[i], f.temp))
        g = B.grid_posterior_2d(B.Flights(f.temp, y), B.WEAKLY_INFORMATIVE, a_range=(-10, 4), b_range=(-6, 5), m=301)
        u = B.posterior_quantile_of_truth(g, a_true[i], b_true[i])
        qa.append(u[0])
        qb.append(u[1])
    qa, qb = np.array(qa), np.array(qb)
    for name, u in (("intercept", qa), ("slope", qb)):
        c50 = np.mean((u > 0.25) & (u < 0.75))
        c90 = np.mean((u > 0.05) & (u < 0.95))
        ks = stats.kstest(u, "uniform").pvalue
        print(
            f"{name:>9}: {k} fake datasets; 50% intervals cover {pct(c50)}, 90% intervals cover {pct(c90)};"
            f" uniformity of posterior quantiles, KS p = {ks:.2f}"
        )


# ----------------------------------------------------------------------------------------------- 7 calibration


def calibration():
    head("7. Checking the sampler without algebra: simulation-based calibration")
    print("2,000 simulated trials: theta from the protocol prior, 170 cases, 99 posterior draws kept per trial")
    out = {}
    for i, setup in enumerate(B.SAMPLERS):
        rng = np.random.default_rng(np.random.SeedSequence(SEED + 7).spawn(3)[i])
        ranks, covered, acc = B.sbc(setup, rng)
        counts, p = B.rank_uniformity(ranks)
        outer = np.mean((ranks < 5) | (ranks > 94))
        print(
            f"{setup.name:>20}: acceptance {acc:.2f}; 90% intervals cover {pct(covered.mean())};"
            f" ranks in the outer two bins {pct(outer)} (10.0% expected); chi-square p = {p:.1e}"
        )
        out[setup.name.replace(" ", "_")] = counts
    np.savez_compressed(RES / "sbc.npz", **out)


def main():
    RES.mkdir(exist_ok=True)
    print("Companion numbers for the article on teaching Bayesian statistics problem first.")
    print(f"Seed {SEED}. NumPy {np.__version__}.")
    vaccine_prior()
    vaccine_posterior()
    accumulation()
    sensitivity()
    f = orings()
    fake_data_check(f)
    calibration()


if __name__ == "__main__":
    main()
