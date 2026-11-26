"""Every number in the article, printed in order, and the arrays the figures use (in results/).

    python multiverse_numbers.py > output.txt

Reads the four national YRBS waves from data/ (see fetch_data.py). About a minute on a laptop.
"""

import json
import sys
from dataclasses import replace

import numpy as np

import yrbs_core as Y

RESULTS = Y.HERE / "results"
DRAWS = 4000
SEED = 20261126
Z = 1.959963984540054


def section(title):
    print()
    print(f"## {title}")


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def median_iqr(v):
    q = np.percentile(v, [25, 50, 75])
    return f"median {q[1]:.3f} (IQR {q[0]:.3f} to {q[2]:.3f})"


def main():
    rng = np.random.default_rng(SEED)
    waves = {w: Y.load_wave(w) for w in Y.WAVES}
    designs = {w: Y.psu_index(d["stratum"], d["psu"]) for w, d in waves.items()}
    summary = {}

    # ----------------------------------------------------------------------------------------------
    section("1. The four waves and their questions")
    for w, d in waves.items():
        cid, s_of, s_cnt = designs[w]
        print(
            f"{w}: {len(d['sad'])} students, {len(s_of)} PSUs in {len(s_cnt)} strata; technology item(s): "
            + ", ".join(Y.MEASURES[w])
        )
    print("Persistent sadness or hopelessness, weighted prevalence (design-based 95% CI):")
    trend = {}
    for w, d in waves.items():
        cid, s_of, s_cnt = designs[w]
        row = {}
        for lab, m in (("all", np.ones(len(d["sad"]), bool)), ("girls", d["female"] == 1), ("boys", d["female"] == 0)):
            ok = m & ~np.isnan(d["sad"]) & ~np.isnan(d["weight"])
            p, se = Y.weighted_prevalence(d["sad"][ok], d["weight"][ok], cid[ok], s_of, s_cnt)
            row[lab] = (p, se)
        trend[w] = row
        print(
            f"  {w}: all {pct(row['all'][0])}, girls {pct(row['girls'][0])} "
            f"({pct(row['girls'][0] - Z * row['girls'][1])} to {pct(row['girls'][0] + Z * row['girls'][1])}), "
            f"boys {pct(row['boys'][0])}"
        )
    summary["trend"] = {str(w): {k: list(v) for k, v in r.items()} for w, r in trend.items()}
    for w in (2021, 2023):
        d = waves[w]
        cid, s_of, s_cnt = designs[w]
        for lab, m in (("all", np.ones(len(d["sad"]), bool)), ("girls", d["female"] == 1), ("boys", d["female"] == 0)):
            ok = m & ~np.isnan(d["poor_mh"]) & ~np.isnan(d["weight"])
            p, _ = Y.weighted_prevalence(d["poor_mh"][ok], d["weight"][ok], cid[ok], s_of, s_cnt)
            print(f"  poor mental health most of the time or always, {w}, {lab}: {pct(p)}")

    print("Heavy use, weighted share of those who answered the item:")
    heavy = {}
    for w, d in waves.items():
        cid, s_of, s_cnt = designs[w]
        for m in Y.MEASURES[w]:
            x = d[m]
            ok = ~np.isnan(x) & ~np.isnan(d["weight"])
            hi_thr, top_thr = (5.0, 6.0) if m == "social" else (3.0, 5.0)
            out = {}
            for lab, sub in (("all", ok), ("girls", ok & (d["female"] == 1)), ("boys", ok & (d["female"] == 0))):
                ph, _ = Y.weighted_prevalence((x[sub] >= hi_thr).astype(float), d["weight"][sub], cid[sub], s_of, s_cnt)
                pt, _ = Y.weighted_prevalence(
                    (x[sub] >= top_thr).astype(float), d["weight"][sub], cid[sub], s_of, s_cnt
                )
                out[lab] = (ph, pt)
            heavy[f"{m} {w}"] = out
            hl, tl = (
                ("several times a day or more", "about once an hour or more")
                if m == "social"
                else ("3+ hours", "5+ hours")
            )
            print(
                f"  {m} {w} (answered by {int(ok.sum())}): {hl} {pct(out['all'][0])} (girls {pct(out['girls'][0])}, "
                f"boys {pct(out['boys'][0])}); {tl} {pct(out['all'][1])} (girls {pct(out['girls'][1])}, "
                f"boys {pct(out['boys'][1])})"
            )
    summary["heavy"] = {k: {s: list(v) for s, v in r.items()} for k, r in heavy.items()}

    # ----------------------------------------------------------------------------------------------
    section("2. Checks against CDC's published 2023 figures")
    d = waves[2023]
    cid, s_of, s_cnt = designs[2023]
    social = d["social"]
    answered = ~np.isnan(social)
    print(
        f"Answered the social media item: {int(answered.sum())} of {len(social)} "
        f"({int((~answered).sum())} missing); CDC reports 15,203 and 4,900"
    )
    for lab, m in (
        ("all", answered),
        ("girls", answered & (d["female"] == 1)),
        ("boys", answered & (d["female"] == 0)),
    ):
        p, se = Y.weighted_prevalence((social[m] >= 5).astype(float), d["weight"][m], cid[m], s_of, s_cnt)
        print(f"  frequent use (several times a day or more), {lab}: {pct(p)} ({pct(p - Z * se)} to {pct(p + Z * se)})")
    print("  CDC (Young et al. 2024): 77.0% overall, 81.8% girls, 72.9% boys")
    cdc = Y.Spec(2023, "social", "high_vs_rest", "sad", "demographic_identity", "all", "survey")
    for o in ("sad", "consider", "plan", "attempt"):
        f = Y.fit_spec(d, replace(cdc, outcome=o), designs[2023])
        print(
            f"  adjusted risk ratio, frequent vs less frequent use, {o}: {f.rr:.2f} "
            f"(risk difference {100 * f.b:.1f} points, n = {f.n})"
        )
    print(
        "  CDC adjusted prevalence ratios (logistic, predicted marginals, adjusted for age): "
        "sad 1.35, consider 1.21, plan 1.16, attempt 1.11"
    )

    # ----------------------------------------------------------------------------------------------
    section("3. Dose and response")
    dose = {}
    for w, m, levels in (
        (2023, "social", list(range(8))),
        (2021, "screen", [0.5, 1, 2, 3, 4, 5]),
        (2019, "tv", [0, 0.5, 1, 2, 3, 4, 5]),
        (2019, "games", [0, 0.5, 1, 2, 3, 4, 5]),
    ):
        dd = waves[w]
        cid, s_of, s_cnt = designs[w]
        res = {}
        for lab, sexmask in (("girls", dd["female"] == 1), ("boys", dd["female"] == 0)):
            rows = []
            for lv in levels:
                ok = sexmask & (dd[m] == lv) & ~np.isnan(dd["sad"])
                p, se = Y.weighted_prevalence(dd["sad"][ok], dd["weight"][ok], cid[ok], s_of, s_cnt)
                rows.append((float(lv), p, se, int(ok.sum())))
            res[lab] = rows
            txt = ", ".join(f"{lv:g}: {pct(p, 0)} (n={n})" for lv, p, se, n in rows)
            print(f"  {m} {w}, {lab}, sad by level: {txt}")
        dose[f"{m} {w}"] = res
    summary["dose"] = dose
    g = dose["social 2023"]["girls"]
    print(
        f"  girls 2023: sad {pct(g[0][1])} among non-users, {pct(g[3][1])} a few times a week, "
        f"{pct(g[7][1])} more than once an hour"
    )

    # ----------------------------------------------------------------------------------------------
    section("4. The specification curve")
    available = {w: {o for o in Y.OUTCOMES if o in waves[w]} for w in Y.WAVES}
    specs = Y.grid(available)
    fits = [Y.fit_spec(waves[s.wave], s, designs[s.wave]) for s in specs]
    beta = np.array([f.beta for f in fits])
    b = np.array([f.b for f in fits])
    se_d = np.array([f.se_design for f in fits])
    se_n = np.array([f.se_naive for f in fits])
    scale = np.array([f.sx / f.sy for f in fits])
    t_d, t_n = b / se_d, b / se_n
    nobs = np.array([f.n for f in fits])
    rr = np.array([f.rr for f in fits])
    fork = {k: np.array([getattr(s, k) for s in specs]) for k in Y.FORKS}
    fork["measure"] = np.array([s.wave_measure for s in specs])
    print(f"Specifications: {len(specs)} (analysis samples {nobs.min()} to {nobs.max()} students)")
    print(f"Standardised coefficient: {median_iqr(beta)}; range {beta.min():.3f} to {beta.max():.3f}")
    print(f"Positive (more use, more of the outcome): {pct((beta > 0).mean())}; negative: {(beta < 0).sum()}")
    print(
        f"Significant and positive at 5%, design-based SE: {pct((t_d > Z).mean())}; "
        f"significant and negative: {int((t_d < -Z).sum())}"
    )
    print(f"Significant and positive at 5%, naive SE: {pct((t_n > Z).mean())}")
    print(f"Median design SE / naive SE: {np.median(se_d / se_n):.2f}")
    print(f"Variance 'explained', beta squared: {median_iqr(beta**2 * 100)} (per cent)")
    sw = fork["weighting"] == "survey"
    print(f"Survey-weighted only ({sw.sum()}): {median_iqr(beta[sw])}")

    print("By fork, median standardised coefficient [share significant and positive, design SE]:")
    by_level = {}
    for k in Y.FORKS:
        parts = []
        for lv in dict.fromkeys(fork[k]):
            m = fork[k] == lv
            by_level.setdefault(k, {})[str(lv)] = [float(np.median(beta[m])), float((t_d[m] > Z).mean()), int(m.sum())]
            parts.append(f"{lv} {np.median(beta[m]):.3f} [{pct((t_d[m] > Z).mean(), 0)}]")
        print(f"  {k}: " + "; ".join(parts))
    summary["by_level"] = by_level

    phi, r2 = Y.shapley_r2(beta, fork)
    print(f"Share of the spread of the coefficients explained by the six forks together: {pct(r2)}")
    for k in sorted(phi, key=phi.get, reverse=True):
        print(f"  {k}: {pct(phi[k])} of the variance ({pct(phi[k] / r2)} of what the forks explain)")
    summary["shapley"] = {k: float(v) for k, v in phi.items()}
    summary["shapley_r2"] = r2

    # Within-wave, within-measure spreads: how far apart can two analysts of the same question get?
    for lab in dict.fromkeys(fork["measure"]):
        m = (fork["measure"] == lab) & sw
        print(
            f"  {lab}, survey-weighted: beta {beta[m].min():.3f} to {beta[m].max():.3f}, "
            f"median {np.median(beta[m]):.3f}"
        )

    # Paired comparisons, holding every other choice fixed.
    index = {s.key(): i for i, s in enumerate(specs)}

    def paired(field, a, bb):
        ratios, diffs = [], []
        for i, s in enumerate(specs):
            if getattr(s, field) != a:
                continue
            j = index.get(replace(s, **{field: bb}).key())
            if j is None:
                continue
            diffs.append(beta[j] - beta[i])
            if beta[i] > 0.01:
                ratios.append(beta[j] / beta[i])
        return np.array(diffs), np.array(ratios)

    dif, rat = paired("controls", "demographic_identity", "pathways")
    print(
        f"Adding sleep, bullying and physical activity to demographics + identity: beta changes by a median "
        f"{np.median(dif):.3f}; median ratio {np.median(rat):.2f} ({len(dif)} pairs)"
    )
    dif, rat = paired("controls", "none", "demographic")
    print(f"Adding demographics to no controls: median change {np.median(dif):.3f}; ratio {np.median(rat):.2f}")
    dif, rat = paired("controls", "demographic", "demographic_identity")
    print(f"Adding sexual identity to demographics: median change {np.median(dif):.3f}; ratio {np.median(rat):.2f}")
    dif, _ = paired("sample", "boys", "girls")
    print(
        f"Girls minus boys, same choices otherwise: median {np.median(dif):.3f}; "
        f"girls larger in {pct((dif > 0).mean())}"
    )
    dif, _ = paired("coding", "high_vs_rest", "high_vs_light")
    print(
        f"Light users instead of everyone else as the comparison (high threshold): median change {np.median(dif):.3f}"
    )
    dif, _ = paired("coding", "top_vs_rest", "top_vs_light")
    print(f"Same at the top threshold: median change {np.median(dif):.3f}")
    dif, _ = paired("weighting", "survey", "unweighted")
    print(
        f"Unweighted minus weighted: median change {np.median(dif):.3f}; "
        f"|change| > 0.02 in {pct((np.abs(dif) > 0.02).mean())}"
    )

    # The grid's own weighting: five outcomes, three of them suicide-risk items.
    w_eq = np.zeros(len(specs))
    for o in Y.OUTCOMES:
        m = fork["outcome"] == o
        w_eq[m] = 1.0 / m.sum()
    order = np.argsort(beta)
    cw = np.cumsum(w_eq[order]) / w_eq.sum()
    med_eq = float(beta[order][np.searchsorted(cw, 0.5)])
    print(f"Median with each outcome weighted equally instead of each specification: {med_eq:.3f}")

    # Defensible subset: demographic controls (with or without identity), survey weights.
    defn = np.isin(fork["controls"], ("demographic", "demographic_identity")) & sw
    print(
        f"Defensible subset ({defn.sum()} specs: demographic controls, survey weights): {median_iqr(beta[defn])}; "
        f"significant and positive {pct((t_d[defn] > Z).mean())}"
    )
    for o in Y.OUTCOMES:
        m = defn & (fork["outcome"] == o)
        print(f"  {o}: {median_iqr(beta[m])}")
    for sx in ("girls", "boys"):
        m = defn & (fork["sample"] == sx)
        print(f"  {sx}: {median_iqr(beta[m])}")

    # ----------------------------------------------------------------------------------------------
    section("5. Joint inference")
    z_by_wave = {}
    for w in Y.WAVES:
        idx = np.array([i for i, s in enumerate(specs) if s.wave == w])
        z_by_wave[w] = (idx, np.vstack([fits[i].z for i in idx]))
    bnull, tnull = Y.joint_null(z_by_wave, se_d, scale, DRAWS, rng)
    for lab, m in (("all specifications", np.ones(len(specs), bool)), ("defensible subset", defn)):
        obs_med = float(np.median(beta[m]))
        obs_cnt = int((t_d[m] > Z).sum())
        null_med = np.median(bnull[:, m], axis=1)
        null_cnt = (tnull[:, m] > Z).sum(axis=1)
        p_med = (1 + (null_med >= obs_med).sum()) / (DRAWS + 1)
        p_cnt = (1 + (null_cnt >= obs_cnt).sum()) / (DRAWS + 1)
        print(
            f"{lab}: observed median {obs_med:.3f}, null 97.5th percentile {np.percentile(null_med, 97.5):.3f}, "
            f"p = {p_med:.4f}; observed significant-positive {obs_cnt} of {m.sum()}, null median "
            f"{np.median(null_cnt):.0f} (97.5th {np.percentile(null_cnt, 97.5):.0f}), p = {p_cnt:.4f}"
        )
        summary[f"null_{lab.split()[0]}"] = {
            "null_median": null_med.tolist(),
            "observed": obs_med,
            "null_count": null_cnt.tolist(),
            "observed_count": obs_cnt,
        }
    print(f"({DRAWS} multiplier-bootstrap draws, seed {SEED})")

    # ----------------------------------------------------------------------------------------------
    section("6. One estimate, five ways to say it")
    for sx in ("all", "girls", "boys"):
        f = Y.fit_spec(waves[2023], replace(cdc, sample=sx), designs[2023])
        lo, hi = Y.besd(f.beta)
        print(
            f"2023, frequent social media use and sadness, {sx}: risk difference {100 * f.b:.1f} points "
            f"(95% CI {100 * (f.b - Z * f.se_design):.1f} to {100 * (f.b + Z * f.se_design):.1f}); "
            f"risk {pct(f.p_ref)} -> {pct(f.p_ref + f.b)}; risk ratio {f.rr:.2f}; beta {f.beta:.3f}; "
            f"beta^2 {100 * f.beta**2:.2f}% of variance; BESD {pct(lo)} vs {pct(hi)}; "
            f"E-value {float(Y.e_value(f.rr)):.2f}"
        )
        summary[f"cdc_{sx}"] = {"b": f.b, "rr": f.rr, "beta": f.beta, "p_ref": f.p_ref, "se": f.se_design}
    binm = (fork["coding"] != "linear") & defn & np.isfinite(rr)
    ev = Y.e_value(rr[binm])
    print(
        f"Binary-exposure specifications in the defensible subset: {binm.sum()}; risk ratio "
        f"{median_iqr(rr[binm])}; E-value {median_iqr(ev)}; max E-value {ev.max():.2f}"
    )

    # Comparators from the same survey, same model: other things that go with sadness.
    d = waves[2023]
    d_cmp = dict(d)
    d_cmp["short_sleep"] = np.where(np.isnan(d["sleep"]), np.nan, (d["sleep"] <= 7).astype(float))
    d_cmp["inactive"] = np.where(np.isnan(d["active"]), np.nan, (d["active"] == 0).astype(float))
    d_cmp["frequent_social"] = np.where(np.isnan(d["social"]), np.nan, (d["social"] >= 5).astype(float))
    cmp_out = {}
    for xname, lab in (
        ("frequent_social", "frequent social media use"),
        ("ebullied", "electronically bullied"),
        ("bullied", "bullied at school"),
        ("short_sleep", "7 hours of sleep or fewer"),
        ("inactive", "no day with 60 minutes of activity"),
    ):
        for sx in ("girls", "boys"):
            sp = Y.Spec(2023, xname, "linear", "sad", "demographic_identity", sx, "survey")
            f = Y.fit_spec(d_cmp, sp, designs[2023])
            xm = d_cmp[xname]
            sexm = d["female"] == (1 if sx == "girls" else 0)
            ok = sexm & ~np.isnan(xm) & ~np.isnan(d["sad"]) & (xm == 0)
            p_ref = Y.wmean(d["sad"][ok], d["weight"][ok])
            cmp_out[f"{lab}|{sx}"] = [f.beta, f.b, f.se_design]
            print(
                f"  2023 {sx}, sadness and {lab}: beta {f.beta:.3f}, risk difference {100 * f.b:.1f} points "
                f"(unadjusted risk without it {pct(p_ref)})"
            )
    summary["comparators"] = cmp_out

    # ----------------------------------------------------------------------------------------------
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for s, f in zip(specs, fits, strict=True):
        rows.append(
            {
                "wave": s.wave,
                "measure": s.measure,
                "coding": s.coding,
                "outcome": s.outcome,
                "controls": s.controls,
                "sample": s.sample,
                "weighting": s.weighting,
                "beta": f.beta,
                "b": f.b,
                "se_design": f.se_design,
                "se_naive": f.se_naive,
                "n": f.n,
                "rr": f.rr,
            }
        )
    (RESULTS / "specs.json").write_text(json.dumps(rows))
    (RESULTS / "summary.json").write_text(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
