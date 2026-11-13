"""Synthesis of the wealth-origin series.

Reads the committed results of the five companion projects in this repository and writes

    results/estimates.csv   the headline coefficient under every design, on the same footing
    results/ledger.csv      every objection the series tested, the test used, and a verdict
    results/summary.txt     every number quoted in the closing article

No data are fetched and nothing is re-estimated: each number is read from a results file
that the companion project's own CI reproduces. The verdicts are computed from those
numbers with the rules in VERDICT RULES below, so a reader can disagree with a rule and
change it in one place.

    python synthesis.py
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RES = HERE / "results"

SOURCES = {
    "audit": REPO / "wealth-concentration-misallocation" / "results",
    "panel": REPO / "wealth-origin-growth" / "results",
    "sectors": REPO / "wealth-origin-sectors" / "results",
    "pandemic": REPO / "wealth-origin-pandemic" / "results",
    "successions": REPO / "wealth-origin-successions" / "results",
}

# Short names for the five studies, in the order they were run. The article refers to
# them by title; these labels are what the figures print.
STUDIES = {
    "audit": "Simulation and\ncross-country audit",
    "panel": "Within-country\npanel",
    "sectors": "Sector\ntest",
    "pandemic": "Pandemic\nexposure",
    "successions": "Succession\nevents",
}

Z95 = 1.96  # two-sided 95% normal quantile, rounded as in the companion projects
Z80 = 0.8416212335729143  # one-sided 80% power quantile

PASSED, OPEN, FAILED = "passed", "open", "failed"


def read(study: str, name: str) -> pd.DataFrame:
    return pd.read_csv(SOURCES[study] / name)


def one(df: pd.DataFrame, **match) -> pd.Series:
    """The single row of df whose columns equal the given values."""
    mask = np.ones(len(df), dtype=bool)
    for col, val in match.items():
        mask &= (df[col] == val).to_numpy()
    rows = df.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {match}, found {len(rows)}")
    return rows.iloc[0]


def ci(b: float, se: float) -> tuple[float, float]:
    return b - Z95 * se, b + Z95 * se


def fmt(x: float, d: int = 2) -> str:
    return f"{x:+.{d}f}"


def fmt_ci(b: float, lo: float, hi: float, d: int = 2) -> str:
    return f"{fmt(b, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


# --------------------------------------------------------------------------------------
# VERDICT RULES
#
# The series' claim is that a larger inherited share of billionaire wealth goes with
# slower TFP growth, so the expected sign of every headline coefficient is negative.
# Each ledger row is a test that claim has to pass: 'passed' means it survived the test,
# 'failed' means it did not, 'open' means the test could not tell.
# --------------------------------------------------------------------------------------

def verdict_estimate(b: float, lo: float, hi: float, reference: float | None = None) -> str:
    """A coefficient re-estimated under a new control or sample.

    passed: negative and the 95% interval excludes zero.
    failed: the estimate keeps less than half of the reference (or changes sign), and the
            interval includes zero; with no reference, a positive estimate.
    open:   anything else (negative, interval includes zero, at least half retained).
    """
    if b < 0 and hi < 0:
        return PASSED
    if reference is not None and b / reference < 0.5:
        return FAILED
    if reference is None and b > 0:
        return FAILED
    return OPEN


def verdict_p(p: float) -> str:
    """A one-sided p-value in the expected direction: below 0.05 passed, below 0.10 open."""
    if p < 0.05:
        return PASSED
    if p < 0.10:
        return OPEN
    return FAILED


def verdict_reverse_timing(diff_hi: float, back_hi: float) -> str:
    """Forward coefficient minus the same coefficient on past growth (a placebo).

    passed: forward minus backward is negative with an interval that excludes zero.
    failed: the backward (placebo) coefficient is itself negative with an interval that
            excludes zero, i.e. composition 'predicts' growth that had already happened.
    open:   neither.
    """
    if diff_hi < 0:
        return PASSED
    if back_hi < 0:
        return FAILED
    return OPEN


def verdict_pretrend(b: float, lo: float, hi: float, post: float) -> str:
    """An event-study lead. The objection is that growth was already unusual before the event.

    passed: the whole 95% interval lies within plus or minus half the post-event effect
            (an equivalence test, so a wide interval cannot pass by being uninformative).
    failed: the interval excludes zero.
    open:   anything else.
    """
    bound = abs(post) / 2
    if -bound <= lo and hi <= bound:
        return PASSED
    if lo > 0 or hi < 0:
        return FAILED
    return OPEN


def verdict_placebo_event(b: float, lo: float, hi: float, main: float) -> str:
    """Events that should not matter (an elderly founder leaves with no heir on the list).

    passed: the placebo is not negative, or its interval includes zero and it is under half
            the main effect.
    failed: the placebo is negative with an interval that excludes zero.
    open:   anything else.
    """
    if hi < 0:
        return FAILED
    if b >= 0 or abs(b) < abs(main) / 2:
        return PASSED
    return OPEN


def verdict_leave_one_out(share_ci_excl_zero: float, all_negative: bool) -> str:
    """Dropping one country at a time.

    passed: every estimate negative and at least 90% of intervals exclude zero.
    open:   every estimate negative.
    failed: some estimate not negative.
    """
    if all_negative and share_ci_excl_zero >= 0.9:
        return PASSED
    if all_negative:
        return OPEN
    return FAILED


def verdict_ratio(ratio: float) -> str:
    """Two designs' estimates of the same quantity. passed within a factor of 2, open
    within a factor of 3, failed beyond."""
    r = max(ratio, 1 / ratio) if ratio > 0 else math.inf
    if r <= 2:
        return PASSED
    if r <= 3:
        return OPEN
    return FAILED


def verdict_sign_test(median: float, n_neg: int, n_pos: int) -> str:
    """A set of specifications on the original claim (concentration, not origin).

    failed: the median is not negative, or significant positives outnumber negatives.
    passed: the median is negative and significant negatives outnumber positives.
    """
    if median >= 0 or n_pos > n_neg:
        return FAILED
    if n_neg > n_pos:
        return PASSED
    return OPEN


# --------------------------------------------------------------------------------------
# Estimates on one footing: TFP growth over three years, the 63 'billionaire countries'
# (56 with TFP), country and year fixed effects, inherited-minus-self-made coefficient in
# percentage points of annual growth per log point.
# --------------------------------------------------------------------------------------

@dataclass
class Estimate:
    key: str
    study: str
    panel: str  # "composition" or "event"
    label: str
    b: float
    se: float
    n_obs: float
    without_2020: bool = False

    @property
    def lo(self) -> float:
        return ci(self.b, self.se)[0]

    @property
    def hi(self) -> float:
        return ci(self.b, self.se)[1]


def composition_estimates() -> list[Estimate]:
    out: list[Estimate] = []

    specs = read("panel", "specs.csv")
    r = one(specs, horizon=3, outcome="tfp", transform="log1p", sample="billionaire_countries",
            controls="income", estimator="fe")
    out.append(Estimate("panel", "panel", "composition", "Within-country panel, income controlled",
                        r["diff"], r["se_diff"], r["n_obs"]))
    pre = read("panel", "pre_covid.csv")
    r = one(pre, outcome="tfp", sample="billionaire_countries", horizon=3, windows="ends by 2019")
    out.append(Estimate("panel_pre2020", "panel", "composition",
                        "Within-country panel, income controlled", r["diff"], r["se"], r["n_obs"],
                        without_2020=True))

    fits = read("sectors", "fits.csv")
    r = one(fits, outcome="tfp", horizon=3, sample="billionaire_countries", estimator="fe")
    out.append(Estimate("sectors6", "sectors", "composition", "+ six sector-group shares",
                        r["diff_sector_ctrl"], r["se_diff_sector_ctrl"], r["n_obs"]))
    loy = read("sectors", "leave_one_year_out.csv")
    r2 = one(loy, sample="billionaire_countries", horizon=3, dropped_year=2020)
    out.append(Estimate("sectors6_no2020", "sectors", "composition", "+ six sector-group shares",
                        r2["b"], r2["se"], np.nan, without_2020=True))
    allsec = read("sectors", "all_sector_controls.csv")
    r = one(allsec, outcome="tfp", horizon=3, sample="billionaire_countries")
    out.append(Estimate("sectors18", "sectors", "composition", "+ all 18 sector shares",
                        r["all"], r["se_all"], np.nan))

    mc = read("pandemic", "measurement_checks.csv")
    labels = {
        "wealth, as in the main fits": ("exposure", "+ four pandemic exposures x list year"),
        "wealth averaged over lists t-1 and t": ("twolist", "Two-list average, exposures controlled"),
        "headcount per 10 million people": ("headcount", "Headcount measure, exposures controlled"),
    }
    for measure, (key, label) in labels.items():
        for drop in (False, True):
            r = one(mc, sample="billionaire_countries", horizon=3, measure=measure,
                    without_2020_list=drop)
            out.append(Estimate(key + ("_no2020" if drop else ""), "pandemic", "composition", label,
                                r["b"], r["se"], r["n_obs"], without_2020=drop))
    return out


def event_estimates() -> list[Estimate]:
    ev = read("successions", "event_studies.csv")
    rob = read("successions", "robustness.csv")
    out: list[Estimate] = []

    def from_ev(key: str, shock: str, label: str) -> None:
        r = one(ev, shock=shock, outcome="tfp", sample="billionaire countries")
        out.append(Estimate(key, "successions", "event", label, r["post"], r["se_post"], r["n_obs"]))

    def from_rob(key: str, shock: str, variant: str, label: str, without_2020: bool = False) -> None:
        r = one(rob, shock=shock, variant=variant)
        out.append(Estimate(key, "successions", "event", label, r["post"], r["se_post"], r["n_obs"],
                            without_2020=without_2020))

    from_ev("succ_all", "strict", "All matched successions")
    from_rob("succ_all_no2020", "strict", "growth years 2020 and 2021 left out",
             "All matched successions", without_2020=True)
    from_ev("succ_old", "old founders", "Founders aged 70 or more")
    from_rob("succ_old_no2020", "old founders", "growth years 2020 and 2021 left out",
             "Founders aged 70 or more", without_2020=True)
    from_rob("succ_exposure", "strict", "pandemic exposure x year as controls",
             "All successions, pandemic exposure x year")
    from_ev("succ_old_any", "old exits, any", "Any exit of a founder aged 70+")
    from_ev("succ_old_noheir", "old exits without heirs", "Exit aged 70+, no heir listed")
    return out


def estimates_frame() -> pd.DataFrame:
    rows = []
    for e in composition_estimates() + event_estimates():
        rows.append({"key": e.key, "study": e.study, "panel": e.panel, "label": e.label,
                     "without_2020": e.without_2020, "b": e.b, "se": e.se, "lo": e.lo, "hi": e.hi,
                     "n_obs": e.n_obs})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Power of the succession design for an effect of the size the panel implies
# --------------------------------------------------------------------------------------

def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def type_s_m(theta: float, se: float, z: float = Z95) -> dict[str, float]:
    """Power, sign error and exaggeration ratio of a two-sided test (Gelman and Carlin 2014).

    The estimate is normal with mean theta and standard deviation se. Among estimates that
    reach significance, type_s is the share with the wrong sign and exaggeration is the mean
    absolute estimate divided by |theta|. Closed forms from the truncated normal.
    """
    c = z * se
    a = (c - theta) / se  # upper cut, standardised
    b = (-c - theta) / se  # lower cut, standardised
    p_hi, p_lo = 1 - _Phi(a), _Phi(b)
    power = p_hi + p_lo
    e_hi = theta * p_hi + se * _phi(a)  # E[X; X > c]
    e_lo = -(theta * p_lo - se * _phi(b))  # E[-X; X < -c]
    wrong = p_hi if theta < 0 else p_lo
    return {"power": power, "type_s": wrong / power, "exaggeration": (e_hi + e_lo) / power / abs(theta)}


def succession_power() -> dict[str, float]:
    """How many succession events would detect an effect of the size the panel predicts.

    The panel's prediction for each matched succession (predicted_events.csv) is a change in
    annual TFP growth; the event study's post coefficient is a three-year cumulative change
    per log point of the shock d. The panel-implied post coefficient for one event is
    therefore 3 * predicted / d; its median over events is the effect size to detect.

    Assumes the standard error of the post coefficient falls with the square root of the
    number of shock country-years and that new events look like the old ones.
    """
    pe = read("successions", "predicted_events.csv")
    pe = pe[pe["d"] > 0]
    implied = 3 * pe["predicted"] / pe["d"]
    target = float(implied.median())

    ev = one(read("successions", "event_studies.csv"), shock="strict", outcome="tfp",
             sample="billionaire countries")
    se_now, post = float(ev["se_post"]), float(ev["post"])
    se_needed = abs(target) / (Z95 + Z80)
    multiple = (se_now / se_needed) ** 2

    shocks = read("successions", "succession_shocks.csv")
    strict = shocks[(shocks["shock"] == "strict") & (shocks["pct_gdp"] > 0)]
    n_shock_cy = len(strict)
    # the first list that can reveal a succession is 2012 (it needs the 2011 list before it)
    n_list_years = int(strict["year"].max()) - 2012 + 1

    tm = type_s_m(target, se_now)

    # Sensitivity: the single-list panel coefficient behind the predictions may be attenuated
    # by noise in one list's valuations. On the same country-years, the two-list average is
    # larger than the single-list estimate by this factor; scale the target by it.
    mc = read("pandemic", "measurement_checks.csv")
    same = one(mc, sample="billionaire_countries", horizon=3, without_2020_list=False,
               measure="wealth, lists 2012 onwards (same sample as the two-list average)")
    avg = one(mc, sample="billionaire_countries", horizon=3, without_2020_list=False,
              measure="wealth averaged over lists t-1 and t")
    attenuation = float(avg["b"] / same["b"])
    target_adj = target * attenuation
    tm_adj = type_s_m(target_adj, se_now)
    event_median = float((post / 3) * pe["d"].median())
    panel_median = float(pe["predicted"].median())
    return {
        "n_events_with_prediction": len(pe),
        "panel_implied_post": target,
        "event_post": post,
        "se_post": se_now,
        "se_needed": se_needed,
        "multiple_of_events": multiple,
        "power_now": tm["power"],
        "type_s": tm["type_s"],
        "exaggeration": tm["exaggeration"],
        "observed_ratio": post / target,
        "attenuation_factor": attenuation,
        "panel_implied_post_adjusted": target_adj,
        "power_adjusted": tm_adj["power"],
        "exaggeration_adjusted": tm_adj["exaggeration"],
        "observed_ratio_adjusted": post / target_adj,
        "multiple_of_events_adjusted": (se_now / (abs(target_adj) / (Z95 + Z80))) ** 2,
        "years_of_lists_needed_adjusted": n_list_years * (se_now / (abs(target_adj) / (Z95 + Z80))) ** 2,
        "shock_country_years": n_shock_cy,
        "list_years": n_list_years,
        "shock_country_years_needed": n_shock_cy * multiple,
        "years_of_lists_needed": n_list_years * multiple,
        "event_median_per_year": event_median,
        "panel_median_per_year": panel_median,
        "ratio_event_to_panel": event_median / panel_median,
    }


# --------------------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------------------

def ledger_frame(est: pd.DataFrame, power: dict[str, float]) -> pd.DataFrame:
    E = est.set_index("key")
    rows: list[dict] = []

    def add(objection: str, study: str, test: str, statistic: str, verdict: str, rule: str) -> None:
        rows.append({"objection": objection, "study": study, "test": test, "statistic": statistic,
                     "verdict": verdict, "rule": rule, "cell": cell_text(statistic, rule)})

    # 1. The original claim: concentration itself
    ps = read("audit", "panel_specs.csv")
    w = ps[(ps["estimator"] == "within") & ps["measure"].str.startswith("wealth")]
    med = float(w["beta"].median())
    n_neg = int((w["hi"] < 0).sum())
    n_pos = int((w["lo"] > 0).sum())
    sc = one(read("audit", "spec_curve_test.csv"), subset="wealth_observed")
    add("Top-1% wealth share predicts slower growth within countries", "audit",
        f"within-country estimator, {len(w)} wealth specifications",
        f"median {fmt(med, 3)}; significant {n_neg} negative, {n_pos} positive; "
        f"source-based wealth data median {fmt(sc['median'], 3)} (joint p={sc['p_stouffer']:.3f})",
        verdict_sign_test(med, n_neg, n_pos), "sign test")

    # 2. Does inherited control go with slower TFP growth at all?
    gap = "Inherited control goes with slower TFP growth"
    bs = read("audit", "billionaire_specs.csv")
    r = one(bs, horizon=5, outcome="g_tfp", transform="log1p", sample="all", model="composition")
    d = r["b_inherited_pct"] - r["b_self_made_pct"]
    diff_lo, diff_hi = _audit_diff_interval()
    add(gap, "audit", f"pooled 5-year windows, {int(r['n_obs'])} country-waves in {int(r['n_countries'])} countries",
        fmt_ci(d, diff_lo, diff_hi), verdict_estimate(d, diff_lo, diff_hi), "estimate")
    e = E.loc["panel"]
    add(gap, "panel", "country and year fixed effects, 13 annual lists",
        fmt_ci(e.b, e.lo, e.hi), verdict_estimate(e.b, e.lo, e.hi), "estimate")
    sa = E.loc["succ_all"]
    add(gap, "successions", "TFP growth over the three years after control passes to heirs",
        fmt_ci(sa.b, sa.lo, sa.hi), verdict_estimate(sa.b, sa.lo, sa.hi), "estimate")

    # 3. Reverse timing
    pp = one(read("panel", "pre_post_test.csv"), outcome="tfp", transform="log1p",
             sample="billionaire_countries", horizon=3)
    plac = one(read("panel", "placebo.csv"), outcome="tfp", transform="log1p",
               sample="billionaire_countries", offset=-3)
    add("Not a reflection of earlier growth", "panel",
        "same coefficient on growth t-3 to t; forward minus backward",
        f"backward {fmt(plac['diff'])}; forward minus backward {fmt_ci(pp['forward_minus_backward'], pp['lo'], pp['hi'])}",
        verdict_reverse_timing(pp["hi"], plac["hi"]), "reverse timing")
    sec_fb = _sectors_forward_backward()
    add("Not a reflection of earlier growth", "sectors",
        "sector-controlled coefficient, forward minus backward",
        f"backward {fmt(sec_fb['backward'])}; forward minus backward {fmt_ci(sec_fb['diff'], sec_fb['lo'], sec_fb['hi'])}",
        verdict_reverse_timing(sec_fb["hi"], sec_fb["back_hi"]), "reverse timing")
    mpp = one(read("pandemic", "measures_pre_post.csv"), measure="two-list average", horizon=3)
    back_lo, back_hi = ci(mpp["backward"], mpp["se_backward"])
    add("Not a reflection of earlier growth", "pandemic",
        "two-list average, exposures controlled, forward minus backward",
        f"backward {fmt(mpp['backward'])}; forward minus backward {fmt_ci(mpp['fwd_minus_bwd'], mpp['lo'], mpp['hi'])}",
        verdict_reverse_timing(mpp["hi"], back_hi), "reverse timing")
    rb = one(read("successions", "robustness.csv"), shock="strict", variant="baseline")
    km1_lo, km1_hi = ci(rb["k_m1"], rb["se_k_m1"])
    add("Not a reflection of earlier growth", "successions",
        "growth in the year before the founder leaves the list (k=-1)",
        f"{fmt_ci(rb['k_m1'], km1_lo, km1_hi)} against a post effect of {fmt(rb['post'])}",
        verdict_pretrend(rb["k_m1"], km1_lo, km1_hi, rb["post"]), "pre-trend equivalence")

    # 4. One country
    loo = read("panel", "leave_one_out.csv")
    loo = loo[(loo["outcome"] == "tfp") & (loo["sample"] == "billionaire_countries") & (loo["horizon"] == 3)]
    # countries without TFP data leave the fit unchanged; keep only drops that change it
    loo = loo[~(np.isclose(loo["diff"], loo["full_diff"]) & np.isclose(loo["se"], loo["full_se"]))]
    share = float((loo["hi"] < 0).mean())
    add("Not driven by one country", "panel", f"drop each of {len(loo)} countries",
        f"range {fmt(loo['diff'].min())} to {fmt(loo['diff'].max())}; interval excludes zero in {int((loo['hi'] < 0).sum())} of {len(loo)}",
        verdict_leave_one_out(share, bool((loo["diff"] < 0).all())), "leave one out")
    sl = read("sectors", "leave_one_out.csv")
    sl = sl[(sl["model"] == "diff_sector_ctrl") & (sl["sample"] == "billionaire_countries") & (sl["horizon"] == 3)]
    share = float((sl["hi"] < 0).mean())
    add("Not driven by one country", "sectors", f"drop each of {len(sl)} countries, sector-controlled",
        f"range {fmt(sl['b'].min())} to {fmt(sl['b'].max())}; interval excludes zero in {int((sl['hi'] < 0).sum())} of {len(sl)}",
        verdict_leave_one_out(share, bool((sl["b"] < 0).all())), "leave one out")
    rl = one(read("successions", "robustness.csv"), shock="strict", variant="leave one country with a succession out")
    n_c, n_ok = int(rl["n_obs"]), int(rl["loo_ci_excl_zero"])
    add("Not driven by one country", "successions", f"drop each of {n_c} countries with a succession",
        f"range {fmt(rl['loo_furthest'])} to {fmt(rl['loo_closest'])}; interval excludes zero in {n_ok} of {n_c}",
        verdict_leave_one_out(n_ok / n_c, rl["loo_closest"] < 0), "leave one out")

    # 5. Sector mix
    s6, s18, p0 = E.loc["sectors6"], E.loc["sectors18"], E.loc["panel"]
    req = read("sectors", "sector_required.csv")
    add("Not explained by the sector mix", "sectors", "all 18 sector shares as controls",
        f"{fmt_ci(s18.b, s18.lo, s18.hi)} (six groups {fmt(s6.b)}); required sector coefficient outside its interval in {int(req['excluded'].sum())} of {len(req)}",
        verdict_estimate(s18.b, s18.lo, s18.hi, p0.b), "estimate vs reference")

    # 6. Pandemic exposure
    ex = E.loc["exposure"]
    add("Not explained by pandemic exposure", "pandemic", "four exposures x list year",
        f"{fmt_ci(ex.b, ex.lo, ex.hi)} (without exposures {fmt(s6.b)})",
        verdict_estimate(ex.b, ex.lo, ex.hi, s6.b), "estimate vs reference")
    se_, sa = E.loc["succ_exposure"], E.loc["succ_all"]
    add("Not explained by pandemic exposure", "successions", "exposure x year in the event study",
        f"{fmt_ci(se_.b, se_.lo, se_.hi)} (baseline {fmt(sa.b)})",
        verdict_estimate(se_.b, se_.lo, se_.hi, sa.b), "estimate vs reference")

    # 7. Dependence on the 2020 list / pandemic years
    pr = E.loc["panel_pre2020"]
    add("Survives without the 2020 list", "panel", "lists 2011-2016 only (windows end by 2019)",
        f"{fmt_ci(pr.b, pr.lo, pr.hi)} (all lists {fmt(p0.b)})",
        verdict_estimate(pr.b, pr.lo, pr.hi, p0.b), "estimate vs reference")
    sn = E.loc["sectors6_no2020"]
    add("Survives without the 2020 list", "sectors", "2020 list dropped",
        f"{fmt_ci(sn.b, sn.lo, sn.hi)} (all lists {fmt(s6.b)})",
        verdict_estimate(sn.b, sn.lo, sn.hi, s6.b), "estimate vs reference")
    tn, t0, wn = E.loc["twolist_no2020"], E.loc["twolist"], E.loc["exposure_no2020"]
    add("Survives without the 2020 list", "pandemic", "two-list average, 2020 list dropped",
        f"{fmt_ci(tn.b, tn.lo, tn.hi)} (single list without 2020: {fmt(wn.b)})",
        verdict_estimate(tn.b, tn.lo, tn.hi, t0.b), "estimate vs reference")
    an = E.loc["succ_all_no2020"]
    add("Survives without the 2020 list", "successions", "growth years 2020 and 2021 left out",
        f"{fmt_ci(an.b, an.lo, an.hi)} (baseline {fmt(sa.b)})",
        verdict_estimate(an.b, an.lo, an.hi, sa.b), "estimate vs reference")

    # 8. Valuation noise in single lists
    hc = E.loc["headcount"]
    add("Not an artefact of one list's valuations", "pandemic", "average of lists t-1 and t",
        f"{fmt_ci(t0.b, t0.lo, t0.hi)} (headcount measure {fmt(hc.b)})",
        verdict_estimate(t0.b, t0.lo, t0.hi, ex.b), "estimate vs reference")

    # 9. Visible in GDP per capita
    jt = one(read("panel", "joint_test.csv"), subset="gdp_per_capita")
    add("Also shows in GDP per capita growth", "panel", "joint test over 72 GDP specifications",
        f"median {fmt(jt['median'], 3)}, Stouffer p={jt['p_stouffer']:.3f}",
        verdict_p(jt["p_stouffer"]), "joint test")
    jt = _sectors_joint("diff_sector_ctrl", "gdp_per_capita")
    add("Also shows in GDP per capita growth", "sectors", "joint test over 12 GDP specifications",
        f"median {fmt(jt['median'], 3)}, Stouffer p={jt['p']:.3f}", verdict_p(jt["p"]), "joint test")
    jt = one(read("pandemic", "average_joint_test.csv"), coefficient="avg", subset="gdp_per_capita")
    add("Also shows in GDP per capita growth", "pandemic", "joint test, two-list average, 12 GDP specifications",
        f"median {fmt(jt['median'], 3)}, Stouffer p={jt['p_stouffer']:.3f}",
        verdict_p(jt["p_stouffer"]), "joint test")
    g = one(read("successions", "event_studies.csv"), shock="strict", outcome="gdppc",
            sample="billionaire countries")
    glo, ghi = ci(g["post"], g["se_post"])
    add("Also shows in GDP per capita growth", "successions", "event study on GDP per capita growth",
        fmt_ci(g["post"], glo, ghi), verdict_estimate(g["post"], glo, ghi), "estimate")

    # 10. Holds across the whole specification set (TFP)
    jt = one(read("panel", "joint_test.csv"), subset="tfp")
    add("Holds across the whole specification set", "panel", "joint test over 72 TFP specifications",
        f"median {fmt(jt['median'], 3)}, Stouffer p={jt['p_stouffer']:.3f}",
        verdict_p(jt["p_stouffer"]), "joint test")
    jt = _sectors_joint("diff_sector_ctrl", "tfp")
    add("Holds across the whole specification set", "sectors", "joint test, sector-controlled, 12 TFP specifications",
        f"median {fmt(jt['median'], 3)}, Stouffer p={jt['p']:.3f}", verdict_p(jt["p"]), "joint test")
    jt = one(read("pandemic", "average_joint_test.csv"), coefficient="avg", subset="tfp")
    add("Holds across the whole specification set", "pandemic", "joint test, two-list average, 12 TFP specifications",
        f"median {fmt(jt['median'], 3)}, Stouffer p={jt['p_stouffer']:.3f}",
        verdict_p(jt["p_stouffer"]), "joint test")

    # 11. Timing beats reshuffled timing
    pm = one(read("successions", "permutation.csv"), shock="strict", outcome="tfp",
             sample="billionaire countries")
    add("Beats randomly reshuffled timing", "successions", "1,000 within-country reshuffles of shock years",
        f"share of draws at or below {fmt(pm['observed_post'])}: {pm['p_le']:.3f}",
        verdict_p(pm["p_le"]), "permutation")

    # 12. Exits that should not matter
    nh = E.loc["succ_old_noheir"]
    add("Founder exits without heirs show no drop", "successions", "founders aged 70+ leaving with no heir listed",
        f"{fmt_ci(nh.b, nh.lo, nh.hi)} (successions {fmt(sa.b)})",
        verdict_placebo_event(nh.b, nh.lo, nh.hi, sa.b), "placebo events")

    # 13. Do the designs agree on size?
    add("Event study and panel agree on size", "successions",
        "event-study effect vs panel prediction, median succession",
        f"{power['event_median_per_year']:+.3f} vs {power['panel_median_per_year']:+.3f} pp a year (ratio {power['ratio_event_to_panel']:.1f})",
        verdict_ratio(power["ratio_event_to_panel"]), "ratio")

    return pd.DataFrame(rows)


_INTERVAL = re.compile(r"[+-]\d+\.\d+ \[[+-]\d+\.\d+, [+-]\d+\.\d+\]")


def cell_text(statistic: str, rule: str) -> str:
    """The short form of a statistic printed in the ledger figure."""
    if rule == "leave one out":
        m = re.search(r"excludes zero in (\d+) of (\d+)", statistic)
        return f"{m.group(1)} of {m.group(2)} drops"
    if rule in ("joint test", "permutation"):
        m = re.search(r"(?:p=|: )(\d\.\d{3})$", statistic)
        return f"p = {m.group(1)}"
    if rule == "sign test":
        m = re.search(r"median ([+-]\d+\.\d+)", statistic)
        return f"median {m.group(1)}"
    if rule == "ratio":
        m = re.search(r"ratio (\d+\.\d)", statistic)
        return f"{m.group(1)} times"
    m = _INTERVAL.search(statistic)
    if m is None:
        raise ValueError(f"no interval in {statistic!r}")
    return m.group(0)


def _audit_diff_interval() -> tuple[float, float]:
    """The audit's country-cluster bootstrap interval for inherited minus self-made.

    The audit writes it only to its summary text (the CSV holds the two coefficients), so it
    is parsed from the line for 5-year TFP growth, log1p, all countries.
    """
    text = (SOURCES["audit"] / "analysis_summary.txt").read_text()
    for line in text.splitlines():
        if line.startswith("h=5 g_tfp log1p all composition"):
            part = line.split("inherited - self-made")[1]
            inside = part.split("[")[1].split("]")[0]
            lo, hi = (float(x) for x in inside.split(","))
            return lo, hi
    raise ValueError("audit difference interval not found")


def _sectors_forward_backward() -> dict[str, float]:
    """Forward minus backward for the sector-controlled coefficient.

    pre_post.csv holds the backward point estimate but not its interval. The reverse-timing
    rule only asks whether the backward interval lies entirely below zero; when the backward
    estimate is not negative its upper limit is at least the estimate itself, which is
    enough to answer. A negative backward estimate would need the interval, so it stops.
    """
    pp = one(read("sectors", "pre_post.csv"), model="diff_sector_ctrl", outcome="tfp",
             sample="billionaire_countries", horizon=3)
    back = float(pp["backward"])
    if back < 0:
        raise ValueError("negative backward estimate: its interval is needed for the verdict")
    return {"backward": back, "diff": float(pp["forward_minus_backward"]), "lo": float(pp["lo"]),
            "hi": float(pp["hi"]), "back_hi": back}


def _sectors_joint(coef: str, subset: str) -> dict[str, float]:
    jt = read("sectors", "joint_test.csv")
    r = one(jt, coefficient=coef, subset=subset)
    return {"median": float(r["median"]), "p": float(r["p_stouffer"])}


# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------

def summary_lines(est: pd.DataFrame, led: pd.DataFrame, power: dict[str, float]) -> list[str]:
    L = ["# Synthesis of the wealth-origin series (every number quoted in the closing article)", ""]

    L.append("## Headline coefficient under each design (TFP, 3 years, billionaire countries, fixed effects;")
    L.append("## pp of annual growth per log point of inherited relative to self-made billionaire wealth)")
    for _, r in est[est["panel"] == "composition"].iterrows():
        tag = "without the 2020 list" if r["without_2020"] else "all lists"
        if r["key"] == "panel_pre2020":
            tag = "lists 2011-2016 only"
        n = "" if np.isnan(r["n_obs"]) else f", n={int(r['n_obs'])}"
        L.append(f"  {r['study']:<9} {r['label']} ({tag}{n}): {fmt_ci(r['b'], r['lo'], r['hi'], 3)}")
    full = est[(est["panel"] == "composition") & ~est["without_2020"]]
    L.append(f"full-sample estimates: {len(full)}; range {fmt(full['b'].min(), 3)} to {fmt(full['b'].max(), 3)}; "
             f"all intervals exclude zero: {bool((full['hi'] < 0).all())}")
    drop = est[(est["panel"] == "composition") & est["without_2020"]]
    L.append(f"estimates without the 2020 list (or before it): {len(drop)}; intervals excluding zero "
             f"{int((drop['hi'] < 0).sum())} ({', '.join(drop.loc[drop['hi'] < 0, 'key'])})")
    ratio = (drop.set_index("key")["b"].rename(lambda k: k.replace("_no2020", "").replace("_pre2020", ""))
             / full.set_index("key")["b"])
    ratio = ratio.dropna()
    L.append("share of the full-sample estimate kept when the 2020 list is dropped: "
             + "; ".join(f"{k} {v:.2f}" for k, v in ratio.items()))
    L.append("")

    L.append("## Succession event study (cumulative change in TFP growth over the event year and the two after, pp per log point of the shock)")
    for _, r in est[est["panel"] == "event"].iterrows():
        tag = "growth years 2020-21 left out" if r["without_2020"] else "all years"
        L.append(f"  {r['label']} ({tag}, n={int(r['n_obs'])}): {fmt_ci(r['b'], r['lo'], r['hi'], 3)}")
    L.append("")

    L.append("## Power of the succession design for an effect of the size the panel implies")
    L.append(f"successions with a panel prediction: {power['n_events_with_prediction']}")
    L.append(f"panel-implied post coefficient (median over events of 3 x predicted / shock): {power['panel_implied_post']:+.3f}")
    L.append(f"event-study post coefficient: {power['event_post']:+.3f}, standard error {power['se_post']:.3f}")
    L.append(f"standard error needed for 80% power at the 5% level: {power['se_needed']:.3f}")
    L.append(f"multiple of the current number of succession events needed: {power['multiple_of_events']:.1f}")
    L.append(f"power of the current design against the panel-implied effect: {power['power_now']:.3f}")
    L.append(f"if the true effect were the panel-implied one: significant estimates have the wrong sign with "
             f"probability {power['type_s']:.3f} and overstate it by a factor of {power['exaggeration']:.2f} on average; "
             f"the observed estimate is {power['observed_ratio']:.2f} times it")
    L.append(f"allowing for attenuation of the panel coefficient (two-list over single-list estimate on the same "
             f"country-years: {power['attenuation_factor']:.3f}): target {power['panel_implied_post_adjusted']:+.3f}, "
             f"power {power['power_adjusted']:.3f}, exaggeration {power['exaggeration_adjusted']:.2f}, observed "
             f"{power['observed_ratio_adjusted']:.2f} times the target, events needed {power['multiple_of_events_adjusted']:.1f} times, about {power['years_of_lists_needed_adjusted']:.0f} years of lists")
    L.append(f"country-years with a matched succession (all countries): {power['shock_country_years']} over "
             f"{power['list_years']} list years; needed at the same event sizes: "
             f"{power['shock_country_years_needed']:.0f}, about {power['years_of_lists_needed']:.0f} years of lists at the current rate")
    L.append(f"median succession: event-implied {power['event_median_per_year']:+.4f} pp a year, panel-predicted "
             f"{power['panel_median_per_year']:+.4f}; ratio {power['ratio_event_to_panel']:.2f}")
    L.append("")

    L.append("## Ledger (objection / study: test -> statistic -> verdict [rule])")
    for _, r in led.iterrows():
        L.append(f"{r['objection']} / {r['study']}: {r['test']} -> {r['statistic']} -> {r['verdict']} [{r['rule']}]")
    L.append("")
    L.append("## Verdict counts")
    counts = led.groupby(["study", "verdict"]).size().unstack(fill_value=0)
    for study in STUDIES:
        if study in counts.index:
            c = counts.loc[study]
            L.append(f"  {study}: passed {int(c.get(PASSED, 0))}, open {int(c.get(OPEN, 0))}, failed {int(c.get(FAILED, 0))}")
    tot = led["verdict"].value_counts()
    L.append(f"  all: tests {len(led)}, objections {led['objection'].nunique()}; passed {int(tot.get(PASSED, 0))}, "
             f"open {int(tot.get(OPEN, 0))}, failed {int(tot.get(FAILED, 0))}")
    by_obj = led.groupby("objection", sort=False)["verdict"].agg(lambda v: ",".join(v))
    L.append("  by objection: " + "; ".join(f"{k}: {v}" for k, v in by_obj.items()))
    return L


def main() -> None:
    RES.mkdir(exist_ok=True)
    est = estimates_frame()
    power = succession_power()
    led = ledger_frame(est, power)
    est.to_csv(RES / "estimates.csv", index=False, float_format="%.6f")
    led.to_csv(RES / "ledger.csv", index=False)
    (RES / "power.json").write_text(json.dumps({k: round(float(v), 6) for k, v in power.items()}, indent=1) + "\n")
    lines = summary_lines(est, led, power)
    (RES / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
