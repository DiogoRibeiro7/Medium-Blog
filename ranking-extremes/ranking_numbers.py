"""Every number in the article, printed in order. Tables for the figures go to .cache/results/.

python fetch_data.py
python ranking_numbers.py > output.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

import fetch_data as fd
import ranking_core as rc
import snapshot as sn

HERE = Path(__file__).resolve().parent
RESULTS = HERE / ".cache" / "results"
DRAWS = 20_000
RATINGS = ("services", "happiness")
LABEL = {"services": "satisfaction with tax-paid services", "happiness": "happiness"}
Z95, Z998 = 1.959964, 3.090232


def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def q(x, digits=2):
    """Median and 90% interval of draws, as text."""
    lo, hi = rc.interval(x)
    return f"{np.median(x):.{digits}f} (90%: {lo:.{digits}f} to {hi:.{digits}f})"


def fit(y, n, X=None, seed=0, draws=DRAWS, **grid):
    X = rc.design(len(y)) if X is None else X
    return rc.posterior_draws(np.asarray(y, float), np.asarray(n, float), X, draws, np.random.default_rng(seed), **grid)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    d = sn.table()
    n = d["n"].to_numpy()
    thin = d["thin"].to_numpy()
    N = len(d)

    # ------------------------------------------------------------------------------------------------
    section("1. The snapshot")
    print(f"source: {sn.SOURCE_URL}, read {sn.READ_ON}")
    print(f"canonical SHA-256 of the transcribed table: {sn.canonical_sha256()}")
    print(
        f"countries listed: {N}; rated reports in them: {n.sum()} of {sn.RATED_REPORTS} rated, {sn.APPROVED_REPORTS} "
        "approved"
    )
    print(f"rated reports per country: min {n.min()}, median {np.median(n):.0f}, max {n.max()}")
    print(f"thin countries (5-9 reports): {thin.sum()} of {N} ({thin.mean():.0%} of countries)")
    print(f"reports in thin countries: {n[thin].sum()} of {n.sum()} ({n[thin].sum() / n.sum():.1%})")
    print(
        f"countries with 30+ reports: {(n >= 30).sum()}, holding {n[n >= 30].sum()} reports "
        f"({n[n >= 30].sum() / n.sum():.1%})"
    )
    print("thin countries:", ", ".join(f"{c} ({k})" for c, k in zip(d.country[thin], n[thin], strict=True)))
    for v in RATINGS:
        y = d[v].to_numpy()
        print(
            f"{v}: report-weighted mean {np.average(y, weights=n):.2f}; unweighted mean {y.mean():.2f}; "
            f"SD of country means {y.std(ddof=1):.2f}; range {y.min():.2f} to {y.max():.2f}"
        )
        print(
            f"  SD of country means, thin countries {y[thin].std(ddof=1):.2f}; 10-29 reports "
            f"{y[(n >= 10) & (n < 30)].std(ddof=1):.2f}; 30+ reports {y[n >= 30].std(ddof=1):.2f}"
        )

    # ------------------------------------------------------------------------------------------------
    section("2. Who sits at the ends of the published rankings")
    observed_T = {}
    for v in RATINGS:
        y = d[v].to_numpy()
        top, bottom = rc.extreme_positions(y, 3)
        print(f"{v}:")
        print("  top 3:   ", "; ".join(f"{d.country[i]} {y[i]:.2f} (n={n[i]})" for i in top))
        print("  bottom 3:", "; ".join(f"{d.country[i]} {y[i]:.2f} (n={n[i]})" for i in bottom))
        observed_T[v] = int(rc.thin_at_extremes(y, thin, 3))
        print(f"  thin countries among the six end positions: {observed_T[v]}")
    print(f"expected number if the six positions were filled at random: {6 * thin.mean():.2f}")
    both = rc.thin_at_extremes(d["services"].to_numpy(), thin, 3) + rc.thin_at_extremes(
        d["happiness"].to_numpy(), thin, 3
    )
    print(f"both rankings together: {both} of 12 end positions go to thin countries")

    # ------------------------------------------------------------------------------------------------
    section("3. How much of the spread is countries and how much is sampling noise")
    fits = {}
    for k, v in enumerate(RATINGS):
        y = d[v].to_numpy()
        f = fit(y, n, seed=10 + k)
        fits[v] = f
        s, t, mu = f["sigma"], f["tau"], f["beta"][:, 0]
        print(f"{v} (uniform priors: sigma on 0.5-4, tau on 0-3):")
        print(f"  sigma, SD of single ratings within a country: {q(s)}")
        print(f"  tau, SD of true country means:                 {q(t)}")
        print(f"  mu, mean of true country means:                {q(mu)}")
        print(f"  P(tau < 0.25) = {np.mean(t < 0.25):.3f}; P(tau < 0.5) = {np.mean(t < 0.5):.3f}")
        se5 = s / np.sqrt(5)
        se112 = s / np.sqrt(112)
        print(f"  standard error of a country mean: n=5 {q(se5)}; n=112 {q(se112)}")
        noise = (s[:, None] ** 2 / n).mean(1)
        share = noise / (noise + t**2)
        print(f"  share of the variance of published country means that is sampling noise: {q(share)}")
        share_thin = (s[:, None] ** 2 / n[thin]).mean(1)
        share_thin = share_thin / (share_thin + t**2)
        print(f"  same, among thin countries only: {q(share_thin)}")
    print()
    # cross-check of sigma for services from the page's share of ratings of 8 or more
    from scipy import stats

    share8 = (0.39 * 304 + 0.42 * 478) / 782
    m = np.average(d["services"], weights=n)
    total_sd = (7.5 - m) / stats.norm.ppf(1 - share8)
    tau_med = np.median(fits["services"]["tau"])
    print(
        f"cross-check, services: the page reports 39% (remote, 304) and 42% (on-site, 478) of ratings at 8+, "
        f"so {share8:.1%} overall;"
    )
    print(
        f"  with mean {m:.2f} and a rounded normal, the total SD of single ratings is {total_sd:.2f}, "
        f"and the within-country SD sqrt(total^2 - tau^2) = {np.sqrt(total_sd**2 - tau_med**2):.2f}"
    )
    share8h = (0.65 * 304 + 0.48 * 478) / 782
    mh = np.average(d["happiness"], weights=n)
    print(
        f"  happiness: {share8h:.1%} of ratings at 8+ with a mean of {mh:.2f}: more than half above a mean "
        "below 7.5, so the distribution is skewed to the left and the normal shortcut does not apply"
    )

    # ------------------------------------------------------------------------------------------------
    section("4. Funnel plots: limits around the report-weighted mean at the posterior median of sigma")
    funnel_rows = []
    for v in RATINGS:
        y = d[v].to_numpy()
        s_med = float(np.median(fits[v]["sigma"]))
        centre = float(np.average(y, weights=n))
        lo95, hi95 = rc.funnel_limits(centre, s_med, n, Z95)
        lo998, hi998 = rc.funnel_limits(centre, s_med, n, Z998)
        out95 = (y < lo95) | (y > hi95)
        out998 = (y < lo998) | (y > hi998)
        print(f"{v}: centre {centre:.2f}, sigma {s_med:.2f}")
        print(
            f"  half-width of the 95% limits: n=5 {Z95 * s_med / np.sqrt(5):.2f}, n=112 "
            f"{Z95 * s_med / np.sqrt(112):.2f}"
        )
        print(
            f"  outside the 95% limits ({out95.sum()}):",
            ", ".join(
                f"{d.country[i]} (n={n[i]}, {'above' if y[i] > centre else 'below'})" for i in np.where(out95)[0]
            ),
        )
        print(
            f"  outside the 99.8% limits ({out998.sum()}):",
            ", ".join(
                f"{d.country[i]} (n={n[i]}, {'above' if y[i] > centre else 'below'})" for i in np.where(out998)[0]
            ),
        )
        print(f"  thin countries outside the 95% limits: {(out95 & thin).sum()} of {thin.sum()}")
        z = (y - centre) / (s_med / np.sqrt(n))
        for i in rc.extreme_positions(y, 1)[0].tolist() + rc.extreme_positions(y, 1)[1].tolist():
            print(f"  {d.country[i]}: z = {z[i]:+.2f}")
        for i in range(N):
            funnel_rows.append(
                {
                    "rating": v,
                    "country": d.country[i],
                    "n": n[i],
                    "mean": y[i],
                    "z": z[i],
                    "centre": centre,
                    "sigma": s_med,
                }
            )
    pd.DataFrame(funnel_rows).to_csv(RESULTS / "funnel.csv", index=False)

    # ------------------------------------------------------------------------------------------------
    section("5. Shrinkage and rank intervals")
    rank_rows = []
    for v in RATINGS:
        y = d[v].to_numpy()
        f = fits[v]
        s_med, t_med = np.median(f["sigma"]), np.median(f["tau"])
        w = rc.shrinkage_weight(np.array([5, 9, 31, 112]), s_med, t_med)
        print(
            f"{v}: weight on a country's own mean at the posterior medians: n=5 {w[0]:.2f}, n=9 {w[1]:.2f}, n=31 "
            f"{w[2]:.2f}, n=112 {w[3]:.2f}"
        )
        theta = f["theta"]
        r = rc.ranks_desc(theta)
        lo, hi = rc.rank_interval(r)
        pm = theta.mean(0)
        raw_rank = rc.ranks_desc(y)
        pm_rank = rc.ranks_desc(pm)
        p_top = (r == 1).mean(0)
        p_bottom = (r == N).mean(0)
        p_top3 = (r <= 3).mean(0)
        p_bot3 = (r >= N - 2).mean(0)
        order = np.argsort(pm_rank)
        print(
            f"  {'country':15s} {'n':>4s} {'raw':>5s} {'rank':>4s} {'shrunk':>6s} {'rank':>4s}  90% rank  P(1st)  "
            "P(last)  P(top3)  P(bottom3)"
        )
        for i in order:
            print(
                f"  {d.country[i]:15s} {n[i]:4d} {y[i]:5.2f} {raw_rank[i]:4d} {pm[i]:6.2f} {pm_rank[i]:4d}  "
                f"{lo[i]:2d} to {hi[i]:2d}"
                f"  {p_top[i]:6.3f}  {p_bottom[i]:7.3f}  {p_top3[i]:7.3f}  {p_bot3[i]:10.3f}"
            )
            rank_rows.append(
                {
                    "rating": v,
                    "country": d.country[i],
                    "n": n[i],
                    "thin": thin[i],
                    "raw": y[i],
                    "raw_rank": raw_rank[i],
                    "shrunk": pm[i],
                    "shrunk_rank": pm_rank[i],
                    "theta_lo": np.quantile(theta[:, i], 0.05),
                    "theta_hi": np.quantile(theta[:, i], 0.95),
                    "rank_lo": lo[i],
                    "rank_hi": hi[i],
                    "p_top": p_top[i],
                    "p_bottom": p_bottom[i],
                }
            )
        width = hi - lo + 1
        print(
            f"  width of the 90% rank interval: median {np.median(width):.0f} places; thin "
            f"{np.median(width[thin]):.0f}; 30+ reports {np.median(width[n >= 30]):.0f}"
        )
        top_half = np.where(hi <= 10)[0]
        bot_half = np.where(lo >= N - 9)[0]
        print("  90% rank interval inside the top 10:", ", ".join(d.country[top_half]) or "none")
        print("  90% rank interval inside the bottom 10:", ", ".join(d.country[bot_half]) or "none")
        most_top = np.argmax(p_top)
        most_bot = np.argmax(p_bottom)
        print(
            f"  most likely first: {d.country[most_top]} ({p_top[most_top]:.3f}); most likely last: "
            f"{d.country[most_bot]} ({p_bottom[most_bot]:.3f})"
        )
        print(f"  P(a thin country is truly first) = {p_top[thin].sum():.3f}; truly last = {p_bottom[thin].sum():.3f}")
        changed = np.sum(np.abs(raw_rank - pm_rank) >= 5)
        print(f"  countries whose place moves 5 or more from the raw table: {changed}")
        print(f"  Spearman correlation of raw and shrunk order: {rc.spearman(y, pm):.3f}")
    pd.DataFrame(rank_rows).to_csv(RESULTS / "ranks.csv", index=False)

    # pairwise questions the table invites
    ths = fits["services"]["theta"]
    thh = fits["happiness"]["theta"]
    ix = {c: i for i, c in enumerate(d.country)}
    print()
    print(
        f"services: P(Luxembourg > Switzerland) = {np.mean(ths[:, ix['Luxembourg']] > ths[:, ix['Switzerland']]):.3f}"
    )
    print(f"services: P(Greece < Ireland) = {np.mean(ths[:, ix['Greece']] < ths[:, ix['Ireland']]):.3f}")
    print(
        "services: P(Portugal below the mean of true country means) = "
        f"{np.mean(ths[:, ix['Portugal']] < fits['services']['beta'][:, 0]):.3f}"
    )
    print(f"happiness: P(Kosovo > Germany) = {np.mean(thh[:, ix['Kosovo']] > thh[:, ix['Germany']]):.3f}")
    print(f"happiness: P(Slovenia < Germany) = {np.mean(thh[:, ix['Slovenia']] < thh[:, ix['Germany']]):.3f}")
    print(
        "happiness: P(Germany below the mean of true country means) = "
        f"{np.mean(thh[:, ix['Germany']] < fits['happiness']['beta'][:, 0]):.3f}"
    )

    # ------------------------------------------------------------------------------------------------
    section("6. Would sampling noise alone put the small countries at the ends? Simulation, 20,000 tables each")
    sim_rows = []
    for k, v in enumerate(RATINGS):
        f = fits[v]
        rng = np.random.default_rng(100 + k)
        reps = 20_000
        s_med = np.median(f["sigma"])
        mu_med = np.median(f["beta"][:, 0])
        # (a) no differences between countries
        y0, _ = rc.simulate_means(n, np.full(reps, mu_med), np.full(reps, s_med), np.zeros(reps), rng)
        T0 = rc.thin_at_extremes(y0, thin, 3)
        top0 = thin[np.argmax(y0, 1)]
        bot0 = thin[np.argmin(y0, 1)]
        # (b) posterior predictive: countries differ as estimated, every parameter drawn from the posterior
        pick = rng.integers(0, len(f["sigma"]), reps)
        y1, th1 = rc.simulate_means(n, f["beta"][pick, 0], f["sigma"][pick], f["tau"][pick], rng)
        T1 = rc.thin_at_extremes(y1, thin, 3)
        top1 = thin[np.argmax(y1, 1)]
        bot1 = thin[np.argmin(y1, 1)]
        true_top1 = thin[np.argmax(th1, 1)]
        true_bot1 = thin[np.argmin(th1, 1)]
        same_top = np.argmax(y1, 1) == np.argmax(th1, 1)
        same_bot = np.argmin(y1, 1) == np.argmin(th1, 1)
        print(f"{v} (observed: {observed_T[v]} of 6 end positions thin):")
        print(
            f"  (a) no country differences: mean {T0.mean():.2f} thin at the ends; P(>= observed) = "
            f"{np.mean(T0 >= observed_T[v]):.3f}; "
            f"P(first is thin) = {top0.mean():.3f}; P(last is thin) = {bot0.mean():.3f}"
        )
        print(
            f"  (b) countries differ as estimated: mean {T1.mean():.2f}; P(>= observed) = "
            f"{np.mean(T1 >= observed_T[v]):.3f}; "
            f"P(first is thin) = {top1.mean():.3f}; P(last is thin) = {bot1.mean():.3f}"
        )
        print(
            f"      in the same tables, the truly best country is thin with P = {true_top1.mean():.3f} and the truly "
            "worst with "
            f"P = {true_bot1.mean():.3f}"
        )
        print(
            f"      the published first is the true first in {same_top.mean():.1%} of tables; last is the true last "
            f"in {same_bot.mean():.1%}"
        )
        for t_val in range(7):
            sim_rows.append(
                {
                    "rating": v,
                    "T": t_val,
                    "no_differences": np.mean(T0 == t_val),
                    "as_estimated": np.mean(T1 == t_val),
                    "observed": observed_T[v],
                }
            )
    pd.DataFrame(sim_rows).to_csv(RESULTS / "extremes_simulation.csv", index=False)
    print(
        f"share of thin countries among all {N}: {thin.mean():.3f} (what a position would get if size did not matter)"
    )

    # ------------------------------------------------------------------------------------------------
    section("7. Sensitivity to sigma, fixed instead of estimated")
    for v in RATINGS:
        y = d[v].to_numpy()
        print(f"{v}:")
        for s_fix in (1.5, 2.0, 2.5, 3.0):
            f = fit(y, n, seed=200, draws=10_000, sigma_grid=np.array([s_fix, s_fix + 1e-9]))
            r = rc.ranks_desc(f["theta"])
            lo, hi = rc.rank_interval(r)
            pm = f["theta"].mean(0)
            centre = np.average(y, weights=n)
            lo95, hi95 = rc.funnel_limits(centre, s_fix, n, Z95)
            out95 = (y < lo95) | (y > hi95)
            top_raw = int(np.argmax(y))
            bot_raw = int(np.argmin(y))
            print(
                f"  sigma {s_fix:.1f}: tau {np.median(f['tau']):.2f}; outside 95% funnel {out95.sum()} "
                f"(thin {(out95 & thin).sum()}); raw first {d.country[top_raw]} -> shrunk place "
                f"{rc.ranks_desc(pm)[top_raw]}, "
                f"P(1st) {np.mean(r[:, top_raw] == 1):.3f}; raw last {d.country[bot_raw]} -> "
                f"{rc.ranks_desc(pm)[bot_raw]}, "
                f"P(last) {np.mean(r[:, bot_raw] == N):.3f}; median rank-interval width {np.median(hi - lo + 1):.0f}"
            )

    # ------------------------------------------------------------------------------------------------
    section("8. The tax question, with sampling noise in the model")
    for k, v in enumerate(RATINGS):
        y = d[v].to_numpy()
        tax = d["tax"].to_numpy(float)
        X = rc.design(N, tax)
        f = fit(y, n, X, seed=300 + k)
        b10 = f["beta"][:, 1] * 10
        f0 = fits[v]
        print(f"{v}:")
        print(
            f"  Pearson r of published country means with tax: {rc.pearson(y, tax):+.3f} (page quotes "
            f"{'+0.06' if v == 'services' else '-0.46'})"
        )
        print(f"  same, thin countries excluded: {rc.pearson(y[~thin], tax[~thin]):+.3f}")
        print(f"  slope per 10 points of effective tax: {q(b10)}; P(slope < 0) = {np.mean(b10 < 0):.3f}")
        r2 = 1 - np.median(f["tau"]) ** 2 / np.median(f0["tau"]) ** 2
        print(f"  tau without tax {q(f0['tau'])}; with tax {q(f['tau'])}")
        print(f"  1 - (tau with / tau without)^2 at the medians: {r2:+.0%}")
        rt = rc.pearson(f0["theta"], np.broadcast_to(tax, f0["theta"].shape))
        print(f"  correlation of true country means with tax (draws of the model without tax): {q(rt)}")

    # ------------------------------------------------------------------------------------------------
    section("9. Do the engineers' service ratings agree with a population survey? OECD 2023")
    oecd = fd.build_oecd()
    m = d.merge(oecd, on="iso3", how="inner")
    print(f"countries in both: {len(m)} ({m['thin'].sum()} thin): {', '.join(m.country)}")
    print(
        "OECD measures: percent satisfied with healthcare, education, administrative services; composite = their mean"
    )
    for col in ("healthcare", "education", "administration", "composite"):
        print(
            f"  {col:15s} range {m[col].min():.1f} to {m[col].max():.1f}; Pearson with published services "
            f"{rc.pearson(m['services'], m[col]):+.3f}; Spearman {rc.spearman(m['services'], m[col]):+.3f}"
        )
    ym, nm = m["services"].to_numpy(), m["n"].to_numpy()
    f0 = fit(ym, nm, seed=400)
    pm0 = f0["theta"].mean(0)
    print(
        f"  composite vs shrunk means (model without OECD): Pearson {rc.pearson(pm0, m['composite']):+.3f}; Spearman "
        f"{rc.spearman(pm0, m['composite']):+.3f}"
    )
    th = m["thin"].to_numpy()
    gap = np.abs(ym - np.poly1d(np.polyfit(m["composite"], ym, 1))(m["composite"]))
    print(
        f"  distance from the least-squares line: thin countries {gap[th].mean():.2f} points; others "
        f"{gap[~th].mean():.2f}"
    )
    X = rc.design(len(m), m["composite"])
    f1 = fit(ym, nm, X, seed=401)
    b10 = f1["beta"][:, 1] * 10
    print(f"  hierarchical slope per 10 points of OECD composite: {q(b10)}; P(slope > 0) = {np.mean(b10 > 0):.3f}")
    t0, t1 = np.median(f0["tau"]), np.median(f1["tau"])
    print(f"  tau on these {len(m)} countries without OECD {q(f0['tau'])}; with OECD {q(f1['tau'])}")
    print(f"  1 - (tau with / tau without)^2 at the medians: {1 - t1**2 / t0**2:+.0%}")
    rmat = rc.pearson(f1["theta"], np.broadcast_to(m["composite"].to_numpy(float), f1["theta"].shape))
    print(f"  correlation of true country means with the composite (draws of the model with OECD): {q(rmat)}")
    m[["country", "iso3", "n", "thin", "services", "composite", "healthcare", "education", "administration"]].assign(
        shrunk=pm0
    ).to_csv(RESULTS / "oecd.csv", index=False)

    # ------------------------------------------------------------------------------------------------
    section("10. A forecast the page can check: each thin country at the ends doubles its number of reports")
    print("updated mean = (published mean + mean of n new reports) / 2; new reports drawn from the posterior")
    for k, v in enumerate(RATINGS):
        y = d[v].to_numpy()
        f = fits[v]
        rng = np.random.default_rng(700 + k)
        top, bottom = rc.extreme_positions(y, 3)
        ends = [i for i in np.concatenate([top, bottom]) if thin[i]]
        print(f"{v}:")
        for i in ends:
            new = f["theta"][:, i] + f["sigma"] / np.sqrt(n[i]) * rng.standard_normal(len(f["sigma"]))
            upd = (y[i] + new) / 2
            mu = f["beta"][:, 0]
            closer = np.mean(np.abs(upd - mu) < np.abs(y[i] - mu))
            lo, hi = np.quantile(upd, [0.1, 0.9])
            print(
                f"  {d.country[i]:11s} n={n[i]} -> {2 * n[i]}: published {y[i]:.2f}; updated median "
                f"{np.median(upd):.2f} "
                f"(80%: {lo:.2f} to {hi:.2f}); P(closer to the middle) = {closer:.2f}"
            )

    # ------------------------------------------------------------------------------------------------
    section("11. Does the method work on ratings like these? 300 simulated tables of whole-number 0-10 ratings")
    for k, (v, s_lat, t_lat, target) in enumerate((("services", 2.5, 0.85, 6.8), ("happiness", 1.8, 0.25, 7.4))):
        rng = np.random.default_rng(500 + k)
        locs = np.arange(0.0, 13.0, 0.01)
        curve = rc.expected_rating_curve(locs, s_lat, np.random.default_rng(600 + k), size=200_000)
        m_lat = float(np.interp(target, curve, locs))
        reps = 300
        cover, rank_cover, width = [], [], []
        top_raw, top_pm, rho_raw, rho_pm, mae_raw, mae_pm, T_sim = [], [], [], [], [], [], []
        sd_single = []
        for _ in range(reps):
            loc = m_lat + t_lat * rng.standard_normal(N)
            truth = np.interp(loc, locs, curve)
            means = np.empty(N)
            for i in range(N):
                r_i = rc.ratings(loc[i], n[i], s_lat, rng)
                means[i] = r_i.mean()
                if i == 0:
                    sd_single.append(rc.ratings(loc[i], 2000, s_lat, rng).std())
            f = rc.posterior_draws(means, n.astype(float), rc.design(N), 4000, rng)
            lo, hi = np.quantile(f["theta"], [0.05, 0.95], axis=0)
            cover.append(np.mean((truth >= lo) & (truth <= hi)))
            rr = rc.ranks_desc(f["theta"])
            rlo, rhi = rc.rank_interval(rr)
            true_rank = rc.ranks_desc(truth)
            rank_cover.append(np.mean((true_rank >= rlo) & (true_rank <= rhi)))
            width.append(np.median(rhi - rlo + 1))
            pm = f["theta"].mean(0)
            top_raw.append(np.argmax(means) == np.argmax(truth))
            top_pm.append(np.argmax(pm) == np.argmax(truth))
            rho_raw.append(rc.spearman(means, truth))
            rho_pm.append(rc.spearman(pm, truth))
            mae_raw.append(np.mean(np.abs(rc.ranks_desc(means) - true_rank)))
            mae_pm.append(np.mean(np.abs(rc.ranks_desc(pm) - true_rank)))
            T_sim.append(rc.thin_at_extremes(means, thin, 3))
        print(
            f"{v}-like: latent SD {s_lat}, between-country SD {t_lat}, mean rating {target}; skewed, rounded, "
            "clipped to 0-10; "
            f"SD of single ratings about {np.mean(sd_single):.2f}"
        )
        print(f"  coverage of the 90% intervals for true country means: {np.mean(cover):.3f}")
        print(
            f"  coverage of the 90% rank intervals: {np.mean(rank_cover):.3f}; median width {np.mean(width):.1f} places"
        )
        print(f"  published first is the true first: raw {np.mean(top_raw):.1%}, shrunk {np.mean(top_pm):.1%}")
        print(f"  Spearman with the true order: raw {np.mean(rho_raw):.3f}, shrunk {np.mean(rho_pm):.3f}")
        print(f"  mean absolute rank error: raw {np.mean(mae_raw):.2f}, shrunk {np.mean(mae_pm):.2f}")
        print(f"  thin countries at the six end positions of the raw table: mean {np.mean(T_sim):.2f}")


if __name__ == "__main__":
    main()
