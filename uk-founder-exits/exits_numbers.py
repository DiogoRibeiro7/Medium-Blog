"""Every number in the article, from the cohort table built by ch_build.py.

    python exits_numbers.py --cohort cohort.csv --report build_report.json --out output.txt

The cohort table is company-level and stays on the runner; output.txt holds aggregates only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import exits_core as ec

GROUPS = ["British", "non-British", "mixed"]
HORIZONS = [2.0, 4.0, 6.0]
LABELS = {
    "foreign_acquisition": "sold to a foreign owner",
    "flip": "moved under a foreign parent of its own",
    "uk_acquisition": "sold to a UK owner",
    "offshore": "moved under an offshore parent",
    "closure": "closed",
    "any_foreign": "under foreign or offshore control (sale, flip or offshore parent)",
    "next_allotment": "allotted shares again",
    "any_exit": "sold, moved under a foreign parent or closed first",
}
COVARIATES = ["founder_group", "cohort_year", "sic_group", "region", "founders", "allotments_first_year"]
REFERENCE = {"founder_group": "non-British", "cohort_year": "2016", "sic_group": "software", "region": "rest of UK"}
FOREIGN = {"foreign_acquisition": "any_foreign", "flip": "any_foreign", "offshore": "any_foreign"}
FOREIGN_SALE = {"flip": "foreign_acquisition", "offshore": "foreign_acquisition"}


def load(path):
    df = pd.read_csv(path, dtype={"company_number": str}, keep_default_na=False)
    df["cohort_year"] = df["cohort_year"].astype(str)
    df["founders"] = df["founders"].astype(int).clip(upper=4)
    df["allotments_first_year"] = df["allotments_first_year"].astype(int).clip(upper=5)
    return df


def pct(x):
    return f"{100 * x:.1f}%"


def interval(est, lo, hi, scale=100, unit="%"):
    return f"{scale * est:.1f}{unit} (95% interval {scale * lo:.1f} to {scale * hi:.1f})"


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, *parts):
        self.lines.append(" ".join(str(p) for p in parts))

    def section(self, title):
        self.lines += ["", title, "-" * len(title)]

    def text(self):
        return "\n".join(self.lines).strip() + "\n"


def composition(say, df):
    say.section("Cohort")
    say(f"Companies in the cohort: {len(df):,}")
    for g in GROUPS:
        f = df[df.founder_group == g]
        say(
            f"{g}: {len(f):,} ({pct(len(f) / len(df))}); London {pct((f.region == 'London').mean())}; "
            f"Oxford and Cambridge {pct((f.region == 'Oxford and Cambridge').mean())}; "
            f"mean founders with significant control {f.founders.mean():.2f}; "
            f"mean allotments in the first year {f.allotments_first_year.mean():.2f}; "
            f"a founder living outside the UK {pct(f.founder_abroad.mean())}"
        )
    say("Founder origin:", ", ".join(f"{k} {v:,}" for k, v in df.founder_origin.value_counts().items()))
    say("Incorporation year:", ", ".join(f"{k} {v:,}" for k, v in df.cohort_year.value_counts().sort_index().items()))
    say("Activity:", ", ".join(f"{k} {v:,}" for k, v in df.sic_group.value_counts().items()))
    say("Entry with a resolution within 60 days (priced-round proxy):", pct((df.round_entry != "").mean()))


def events(say, out):
    say.section("First events after the first allotment")
    say(f"Median follow-up from the first allotment: {out.time.median():.2f} years; longest {out.time.max():.2f}")
    tab = pd.crosstab(out.cause, out.founder_group).reindex(columns=GROUPS, fill_value=0)
    for cause, row in tab.iterrows():
        say(f"{LABELS.get(cause, cause)}:", ", ".join(f"{g} {row[g]:,}" for g in GROUPS))


def cif_section(say, out, causes, title, b, seed, weights=None, levels=GROUPS):
    say.section(title)
    res = ec.bootstrap_cif(out, "founder_group", causes, HORIZONS, b=b, seed=seed, weights=weights, levels=levels)
    for cause in causes:
        say(f"{LABELS.get(cause, cause)}")
        for h in HORIZONS:
            parts = []
            for g in res.group.unique():
                r = res[(res.group == g) & (res.cause == cause) & (res.years == h)].iloc[0]
                parts.append(f"{g} {interval(r.estimate, r.low, r.high)}")
            say(f"  {h:g} years: " + "; ".join(parts))
    return res


def cox_section(say, out, causes, title):
    say.section(title)
    for cause in causes:
        if (out.cause == cause).sum() < 20:
            say(f"{LABELS.get(cause, cause)}: fewer than 20 events, not modelled")
            continue
        s = ec.cause_specific_cox(out, cause, COVARIATES, REFERENCE)
        rows = []
        for g in ("British", "mixed"):
            key = f"founder_group[{g}]"
            if key in s.index:
                r = s.loc[key]
                rows.append(f"{g} vs non-British HR {r.hr:.2f} ({r.low:.2f} to {r.high:.2f}, p = {r.p:.3g})")
        say(f"{LABELS.get(cause, cause)} ({(out.cause == cause).sum():,} events): " + "; ".join(rows))


def ph_check(say, out, cause):
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    x = ec.design(out, COVARIATES, REFERENCE)
    data = x.assign(time=out["time"].to_numpy(), event=(out["cause"] == cause).astype(int).to_numpy())
    cph = CoxPHFitter().fit(data, "time", "event")
    test = proportional_hazard_test(cph, data, time_transform="rank")
    p = test.summary.loc["founder_group[British]", "p"] if "founder_group[British]" in test.summary.index else np.nan
    say(f"Proportional hazards, {LABELS.get(cause, cause)}: rank-time test for the British coefficient p = {p:.3g}")


def fine_gray_section(say, out, cause, b, seed):
    x = ec.design(out, COVARIATES, REFERENCE)
    cols = list(x.columns)
    j = cols.index("founder_group[British]")
    beta, cov = ec.fine_gray(out.time.to_numpy(), out.cause.to_numpy(), x.to_numpy(), cause)
    rng = np.random.default_rng(seed)
    draws = []
    n = len(out)
    for _ in range(b):
        i = rng.integers(0, n, n)
        xb = x.iloc[i]
        keep = xb.std().to_numpy() > 0
        if not keep[j]:
            continue
        bb, _ = ec.fine_gray(out.time.to_numpy()[i], out.cause.to_numpy()[i], xb.to_numpy()[:, keep], cause)
        draws.append(bb[int(np.sum(keep[:j]))])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    se = np.sqrt(cov[j, j])
    say(
        f"Fine-Gray, {LABELS.get(cause, cause)}: British vs non-British subdistribution HR {np.exp(beta[j]):.2f} "
        f"(bootstrap 95% interval {np.exp(lo):.2f} to {np.exp(hi):.2f}; model-based "
        f"{np.exp(beta[j] - 1.96 * se):.2f} to {np.exp(beta[j] + 1.96 * se):.2f}; {len(draws)} resamples)"
    )
    return np.exp(beta[j]), np.exp(lo), np.exp(hi)


def states_section(say, out, years):
    say.section(f"Where companies stand {years:g} years after the first allotment")
    st = ec.states_at(out, years, ec.CAUSES)
    say(f"Companies followed at least {years:g} years: {len(st):,}")
    order = [*ec.CAUSES, "independent, raised again", "independent, no further allotment"]
    for g in GROUPS:
        f = st[st.founder_group == g]
        if not len(f):
            continue
        shares = f.state.value_counts(normalize=True).reindex(order, fill_value=0)
        say(f"{g} ({len(f):,}): " + "; ".join(f"{LABELS.get(k, k)} {pct(v)}" for k, v in shares.items()))
    shares = {
        g: {k: float(v) for k, v in st[st.founder_group == g].state.value_counts(normalize=True).items()}
        | {"n": int((st.founder_group == g).sum())}
        for g in GROUPS
    }
    a, b = st[st.founder_group == "British"], st[st.founder_group == "non-British"]
    for state in ("foreign_acquisition", "closure", "independent, no further allotment"):
        pa, pb = (a.state == state).mean(), (b.state == state).mean()
        se = np.sqrt(pa * (1 - pa) / len(a) + pb * (1 - pb) / len(b))
        say(
            f"British minus non-British, {LABELS.get(state, state)}: "
            f"{interval(pa - pb, pa - pb - 1.96 * se, pa - pb + 1.96 * se, unit=' points')}"
        )
    return shares


def adjusted_section(say, out, causes, b, seed):
    say.section("Covariate-balanced cumulative incidence, British vs non-British founders")
    two = out[out.founder_group.isin(["British", "non-British"])].copy()
    two["british"] = (two.founder_group == "British").astype(int)
    covs = [c for c in COVARIATES if c != "founder_group"]
    w, ps = ec.balancing_weights(two, "british", covs, REFERENCE)
    two["w"] = w
    x = ec.design(two, covs, REFERENCE)
    worst = max(abs(ec.standardised_difference(x.assign(t=two.british), c, "t", w)) for c in x.columns)
    raw = max(abs(ec.standardised_difference(x.assign(t=two.british), c, "t")) for c in x.columns)
    say(f"Largest standardised difference in covariates: {raw:.3f} before weighting, {worst:.3f} after")
    say(f"Propensity of British founding: range {ps.min():.2f} to {ps.max():.2f} after trimming at 1% and 99%")
    title = "Weighted (weights held fixed across resamples)"
    return cif_section(say, two, causes, title, b, seed, "w", ["British", "non-British"])


def origin_section(say, out, b, seed):
    say.section("By founder origin, 5 years after the first allotment")
    m = out.assign(cause=out.cause.replace(FOREIGN))
    levels = [lv for lv in m.founder_origin.value_counts().index if (m.founder_origin == lv).sum() >= 100]
    levels = ["British", *[lv for lv in levels if lv != "British"]]
    res = ec.bootstrap_cif(m, "founder_origin", ["any_foreign", "closure"], [5.0], b=b, seed=seed, levels=levels)
    for cause in ("any_foreign", "closure"):
        say(LABELS[cause])
        for g in levels:
            r = res[(res.group == g) & (res.cause == cause)].iloc[0]
            say(f"  {g} ({(m.founder_origin == g).sum():,}): {interval(r.estimate, r.low, r.high)}")
    return res


def sensitivity_section(say, df):
    say.section("Sensitivity of the British vs non-British hazard ratios")
    specs = {
        "main (first allotment)": (df, "entry", None),
        "origin at the first allotment with a resolution": (df, "round_entry", None),
        "flips and offshore parents counted as foreign sales": (df, "entry", FOREIGN_SALE),
        "founders all living in the UK": (df[df.founder_abroad == 0], "entry", None),
        "London only": (df[df.region == "London"], "entry", None),
        "software only": (df[df.sic_group == "software"], "entry", None),
        "one founder with significant control": (df[df.founders == 1], "entry", None),
        "incorporated 2016 to 2018": (df[df.cohort_year.isin(["2016", "2017", "2018"])], "entry", None),
        "incorporated 2019 to 2020": (df[df.cohort_year.isin(["2019", "2020"])], "entry", None),
    }
    rows = []
    for name, (d, entry, merge) in specs.items():
        out = ec.outcomes(d, entry=entry, merge=merge)
        out = out[out.founder_group.isin(GROUPS)]
        parts = []
        for cause in ("foreign_acquisition", "uk_acquisition", "closure"):
            if (out.cause == cause).sum() < 20:
                parts.append(f"{cause}: too few events")
                continue
            try:
                s = ec.cause_specific_cox(out, cause, COVARIATES, REFERENCE).loc["founder_group[British]"]
            except Exception as e:  # a small subset can fail to converge; report it rather than stop
                parts.append(f"{cause}: not estimated ({type(e).__name__})")
                continue
            parts.append(f"{cause} {s.hr:.2f} ({s.low:.2f} to {s.high:.2f})")
            rows.append((name, cause, s.hr, s.low, s.high, len(out)))
        say(f"{name} ({len(out):,} companies): " + "; ".join(parts))
    return pd.DataFrame(rows, columns=["spec", "cause", "hr", "low", "high", "n"])


def follow_on_section(say, df, b, seed):
    causes = ["next_allotment", "any_exit"]
    merge = {c: "any_exit" for c in ec.CAUSES}
    out = ec.outcomes(df, causes=["next_allotment", *ec.CAUSES], merge=merge)
    out = out[out.founder_group.isin(GROUPS)]
    cif_section(say, out, causes, "Raising again: a further allotment more than a year after the first", b, seed)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--fg-bootstrap", type=int, default=200)
    ap.add_argument("--curve-bootstrap", type=int, default=200)
    ap.add_argument("--figdata", help="write the aggregates the figures need to this JSON file")
    args = ap.parse_args(argv)
    say = Report()
    report = json.loads(Path(args.report).read_text())
    say("UK technology companies that allotted shares within three years of incorporation, by founder nationality")
    say(f"Companies House: PSC snapshot {report['snapshot']}; follow-up censored on {report['censor_date']}")
    say(f"Companies found by the search: {report['searched_companies']:,}")
    say(f"Search chunks capped by the API: {report['capped_search_chunks']}")
    say("Left out:", "; ".join(f"{k} {v:,}" for k, v in report["excluded"].items()))
    df = load(args.cohort)
    composition(say, df)
    out = ec.outcomes(df)
    events(say, out)
    cif_section(say, out, ec.CAUSES, "Cumulative incidence by founder group (Aalen-Johansen)", args.bootstrap, 1)
    merged = out.assign(cause=out.cause.replace(FOREIGN))
    cif_section(say, merged, ["any_foreign"], "Any foreign or offshore control", args.bootstrap, 2)
    title = "Cause-specific Cox models, adjusted for year, activity, region, team and early allotments"
    cox_section(say, out, ec.CAUSES, title)
    cox_section(say, merged, ["any_foreign"], "Cause-specific Cox model, any foreign or offshore control")
    say.section("Model checks and subdistribution hazards")
    for cause in ("foreign_acquisition", "closure"):
        ph_check(say, out, cause)
    fg = {c: fine_gray_section(say, out, c, args.fg_bootstrap, 3) for c in ("foreign_acquisition", "closure")}
    fg["any_foreign"] = fine_gray_section(say, merged, "any_foreign", args.fg_bootstrap, 4)
    adjusted = adjusted_section(say, out, ["foreign_acquisition", "uk_acquisition", "closure"], args.bootstrap, 5)
    states = states_section(say, out, 5.0)
    origin = origin_section(say, out, args.bootstrap, 6)
    follow = follow_on_section(say, df, args.bootstrap, 7)
    sens = sensitivity_section(say, df)
    Path(args.out).write_text(say.text())
    if args.figdata:
        grid = [round(0.1 * i, 1) for i in range(81)]
        b = args.curve_bootstrap
        curves = pd.concat(
            [
                ec.bootstrap_cif(out, "founder_group", ec.CAUSES, grid, b=b, seed=8, levels=GROUPS),
                ec.bootstrap_cif(merged, "founder_group", ["any_foreign"], grid, b=b, seed=9, levels=GROUPS),
                ec.bootstrap_cif(follow, "founder_group", ["next_allotment"], grid, b=b, seed=10, levels=GROUPS),
            ]
        )
        figdata = {
            "snapshot": report["snapshot"],
            "censor_date": report["censor_date"],
            "n": {g: int((out.founder_group == g).sum()) for g in GROUPS},
            "curves": curves.to_dict(orient="records"),
            "adjusted": adjusted.to_dict(orient="records"),
            "states5": states,
            "origin5": origin.to_dict(orient="records"),
            "sensitivity": sens.to_dict(orient="records"),
            "fine_gray": {k: [float(v) for v in vals] for k, vals in fg.items()},
        }
        Path(args.figdata).write_text(json.dumps(figdata, indent=1))
    print(say.text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
