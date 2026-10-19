"""Every number the article quotes, read from the committed results, and the audit.

``collect_values`` is the single source of the manuscript's and the Medium article's
numbers: each value is read from a committed table (or, for the two hyperparameters that
no table stores, from the generated Bayesian report) and keeps its provenance. ``audit``
re-runs the snapshot audits, checks that outputs which should agree do agree, and checks
that the numbers quoted in the narrative documents match the tables.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eu_okun.contract import EU27

MINUS = "−"
FOCUS = ("ES", "PT", "NL", "LU")
VERDICTS = {"robust_negative": "robust negative",
            "negative_precision_dependent": "negative; significance specification-dependent",
            "sign_unstable": "sign-unstable"}


@dataclass(frozen=True)
class Value:
    """One quoted value: raw number or text, LaTeX and Markdown forms, provenance."""

    key: str
    raw: float | int | str
    tex: str
    md: str
    source: str


def _sign(x: float, digits: int, signed: bool) -> str:
    text = f"{abs(x):.{digits}f}"
    if x < 0 and float(text) != 0:
        return "-" + text
    return ("+" + text) if signed else text


def num(key: str, x: float, digits: int, source: str, signed: bool = False) -> Value:
    plain = _sign(x, digits, signed)
    return Value(key, x, f"${plain}$", plain.replace("-", MINUS), source)


def pct(key: str, x: float, digits: int, source: str, signed: bool = False) -> Value:
    """x is a fraction; shown as a percentage."""
    plain = _sign(100 * x, digits, signed)
    return Value(key, x, f"${plain}$\\%", plain.replace("-", MINUS) + "%", source)


def interval(key: str, low: float, high: float, digits: int, source: str,
             signed: bool = True) -> Value:
    def one(x: float) -> str:
        if math.isinf(x):
            return "-\\infty" if x < 0 else "+\\infty"
        return _sign(x, digits, signed)

    def md_one(x: float) -> str:
        if math.isinf(x):
            return f"{MINUS}∞" if x < 0 else "+∞"
        return _sign(x, digits, signed).replace("-", MINUS)

    return Value(key, f"[{low}, {high}]", f"$[{one(low)}, {one(high)}]$",
                 f"[{md_one(low)}, {md_one(high)}]", source)


def text(key: str, s: str | int, source: str) -> Value:
    return Value(key, s, str(s), str(s), source)


# ----------------------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------------------


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _one(rows: Sequence[Mapping[str, str]], **match: str) -> dict[str, str]:
    found = [dict(r) for r in rows if all(r[k] == v for k, v in match.items())]
    if len(found) != 1:
        raise ValueError(f"Expected one row matching {match}, found {len(found)}.")
    return found[0]


def _report_number(path: Path, pattern: str) -> list[float]:
    match = re.search(pattern, path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"Pattern not found in {path.name}: {pattern}")
    return [float(g.replace(MINUS, "-")) for g in match.groups()]


def _names(items: Sequence[str]) -> str:
    items = list(items)
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def accepted_specs(docs: Path) -> list[str]:
    diagnostics = json.loads((docs / "bayes" / "diagnostics.json").read_text(encoding="utf-8"))
    return [spec for spec, d in diagnostics.items() if d["accepted"]]


def claim_verdicts(docs: Path) -> dict[str, str]:
    """BAYES_PROTOCOL.md, section 7: supported if the rule holds in every accepted fit."""
    rows = _csv(docs / "bayes" / "claims.csv")
    accepted = accepted_specs(docs)
    out = {}
    for claim in dict.fromkeys(r["claim"] for r in rows):
        mine = [r for r in rows if r["claim"] == claim and r["spec"] in accepted]
        primary = next(r for r in mine if r["spec"] == "primary")
        if all(r["holds"] == "True" for r in mine):
            out[claim] = "supported"
        elif primary["contradicted"] == "True":
            out[claim] = "contradicted"
        else:
            out[claim] = "not resolved"
    return out


# ----------------------------------------------------------------------------------
# Values
# ----------------------------------------------------------------------------------


def collect_values(root: Path) -> dict[str, Value]:
    """Every value quoted in the manuscript and the Medium article."""
    docs = root / "docs"
    out: dict[str, Value] = {}

    def add(v: Value) -> None:
        if v.key in out:
            raise ValueError(f"Duplicate value key {v.key}.")
        out[v.key] = v

    # Data ----------------------------------------------------------------------------
    manifest = json.loads((root / "snapshots" / "eurostat-2026-10-15" /
                           "snapshot_manifest.json").read_text(encoding="utf-8"))
    add(text("n_countries", len(manifest["countries"]), "snapshot manifest"))
    add(text("snapshot_date", "2026-10-15", "snapshots/eurostat-2026-10-15"))
    for name, key in (("eurostat-2026-10-15/snapshot_manifest.json", "hash_quarterly"),
                      ("external-shocks-2026-10-16/manifest.json", "hash_shocks"),
                      ("eurostat-monthly-2026-10-18/snapshot_manifest.json", "hash_monthly"),
                      ("shift-share-2026-10-18/manifest.json", "hash_shiftshare"),
                      ("eurostat-deflator-2026-10-18/manifest.json", "hash_deflator"),
                      ("heterogeneity-2026-10-18/manifest.json", "hash_heterogeneity"),
                      ("gdp-measurement-2026-10-19/manifest.json", "hash_gdpm")):
        digest = hashlib.sha256((root / "snapshots" / name).read_bytes()).hexdigest()
        add(Value(key, digest, f"\\texttt{{{digest[:16]}}}", f"`{digest[:16]}`",
                  f"snapshots/{name}"))

    # Classical -----------------------------------------------------------------------
    src = "docs/review/robustness.csv"
    rob = _csv(docs / "review" / "robustness.csv")
    add(text("cl_n_neg_excluded", sum(float(r["headline_excluded_high"]) < 0 for r in rob), src))
    add(text("cl_n_robust", sum(r["verdict"] == "robust_negative" for r in rob), src))
    shares = sorted(float(r["pandemic_leverage_share"]) for r in rob)
    add(pct("cl_pandemic_share_median", shares[len(shares) // 2], 0, src))
    add(text("cl_n_full_outside", sum(r["full_outside_excluded_ci"] == "True" for r in rob), src))
    grid = _csv(docs / "review" / "grid.csv")

    def k2(sample: str, geo: str) -> float:
        return float(_one(grid, sample=sample, geo=geo, gdp_lags="2",
                          participation="False")["gdp_sum"])

    add(text("cl_n_attenuated", sum(abs(k2("full", g)) < abs(k2("pandemic_excluded", g))
                                    for g in EU27), "docs/review/grid.csv"))
    for geo in FOCUS:
        for sample, tag in (("full", "full"), ("pandemic_excluded", "excl"),
                            ("pre_2020", "pre")):
            r = _one(grid, sample=sample, geo=geo, gdp_lags="2", participation="False")
            add(num(f"cl_{geo}_{tag}", float(r["gdp_sum"]), 2, "docs/review/grid.csv", True))
            add(interval(f"cl_{geo}_{tag}_ci", float(r["ci_low"]), float(r["ci_high"]), 2,
                         "docs/review/grid.csv"))
        add(text(f"cl_{geo}_verdict", VERDICTS.get(_one(rob, geo=geo)["verdict"],
                                                   _one(rob, geo=geo)["verdict"]), src))
    add(text("cl_n_core", _one(rob, geo="AT")["core_estimates"], src))
    # Holm over the 27 two-sided normal tests of the headline sums (not a registered rule;
    # it shows how much the 22 depends on testing each country separately).
    from statistics import NormalDist

    head = [_one(grid, sample="pandemic_excluded", geo=g, gdp_lags="2", participation="False")
            for g in EU27]
    pvals = sorted(2 * (1 - NormalDist().cdf(abs(float(r["gdp_sum"]) / float(r["se"]))))
                   for r in head)
    holm = 0
    for i, p in enumerate(pvals):
        if p > 0.05 / (len(pvals) - i):
            break
        holm += 1
    add(text("cl_n_neg_holm", holm, "docs/review/grid.csv"))
    add(text("cl_n_quarters", _one(grid, sample="pandemic_excluded", geo="AT", gdp_lags="2",
                                   participation="False")["n"], "docs/review/grid.csv"))

    small = json.loads((docs / "small-sample" / "summary.json").read_text(encoding="utf-8"))
    add(text("ss_published_holm", small["published_holm"]["4"], "docs/small-sample/summary.json"))
    add(text("ss_survivors", len(small["survivors"]), "docs/small-sample/summary.json"))
    add(pct("ss_size_normal", small["size_pooled"]["asymptotic"], 1,
            "docs/small-sample/summary.json"))
    add(pct("ss_size_bootstrap", small["size_pooled"]["bootstrap"], 1,
            "docs/small-sample/summary.json"))
    hi = json.loads((docs / "headline-inference" / "summary.json").read_text(encoding="utf-8"))
    src = "docs/headline-inference/summary.json"
    add(text("hi_negative", hi["bootstrap_negative"], src))
    add(text("hi_negative_holm", hi["bootstrap_negative_holm"], src))
    add(text("hi_lost", _names(hi["lost"]), src))
    add(text("hi_point_negative", hi["point_negative"], src))
    add(text("hi_boot_below", hi["boot_intervals_below_zero"], src))
    add(pct("hi_size_normal", hi["size_pooled"]["normal"], 1, src))
    add(pct("hi_size_boot", hi["size_pooled"]["bootstrap"], 1, src))
    add(text("hi_over_rejecting", hi["normal_over_rejecting"], src))
    add(num("hi_rho_gap", hi["rho_gap_pre_median"], 2, src, True))
    for bw in ("2", "8"):
        add(text(f"hi_bw{bw}_holm", hi["bandwidth_negative"][bw]["holm"], src))

    # Hierarchical Bayes --------------------------------------------------------------
    report = docs / "HIERARCHICAL_BAYES.md"
    mu = _report_number(report, r"mu_theta, the unweighted mean of the country sums, has median "
                                r"(-?[\d.]+) and 90% interval \[(-?[\d.]+), (-?[\d.]+)\]")
    tau = _report_number(report, r"tau_theta: median (-?[\d.]+), 90% interval "
                                 r"\[\+?(-?[\d.]+), \+?(-?[\d.]+)\]")
    src = "docs/HIERARCHICAL_BAYES.md (generated from saved posterior draws)"
    add(num("by_mu", mu[0], 3, src, True))
    add(num("by_mu_abs", abs(mu[0]), 2, src))
    add(interval("by_mu_ci", mu[1], mu[2], 3, src))
    add(num("by_tau", tau[0], 3, src))
    add(interval("by_tau_ci", tau[1], tau[2], 3, src, signed=False))
    est = _csv(docs / "bayes" / "country_estimates.csv")
    accepted = accepted_specs(docs)
    add(text("by_n_specs", len(json.loads((docs / "bayes" / "diagnostics.json").read_text(
        encoding="utf-8"))), "docs/bayes/diagnostics.json"))
    add(text("by_n_accepted", len(accepted), "docs/bayes/diagnostics.json"))
    primary = [r for r in est if r["spec"] == "primary"]
    add(text("by_n_neg_primary", sum(r["sign"] == "negative" for r in primary),
             "docs/bayes/country_estimates.csv"))
    robust = [g for g in EU27 if all(_one(est, spec=s, geo=g)["sign"] == "negative"
                                     for s in accepted)]
    add(text("by_n_neg_all", len(robust), "docs/bayes/country_estimates.csv"))
    add(text("by_n_smaller", sum(abs(float(r["theta_median"])) < abs(float(r["classical_theta"]))
                                 for r in primary), "docs/bayes/country_estimates.csv"))
    for geo in FOCUS:
        r = _one(est, spec="primary", geo=geo)
        add(num(f"by_{geo}", float(r["theta_median"]), 2, "docs/bayes/country_estimates.csv",
                True))
        add(interval(f"by_{geo}_ci", float(r["theta_low"]), float(r["theta_high"]), 2,
                     "docs/bayes/country_estimates.csv"))
        add(num(f"by_{geo}_ols", float(r["classical_theta"]), 2,
                "docs/bayes/country_estimates.csv", True))
        add(num(f"by_{geo}_lagshare", float(r["lag_share_median"]), 2,
                "docs/bayes/country_estimates.csv"))
    claims = _csv(docs / "bayes" / "claims.csv")
    verdicts = claim_verdicts(docs)
    for claim, key in (("Luxembourg weak", "lu"), ("Netherlands delayed", "nl"),
                       ("Spain large", "es")):
        add(text(f"claim_{key}_verdict", verdicts[claim], "docs/bayes/claims.csv"))
        row = _one(claims, claim=claim, spec="primary")
        add(num(f"claim_{key}_p", float(row["probability"]), 3, "docs/bayes/claims.csv"))
        add(text(f"claim_{key}_holds", sum(r["holds"] == "True" for r in claims
                                           if r["claim"] == claim and r["spec"] in accepted),
                 "docs/bayes/claims.csv"))
    surviving = _csv(docs / "bayes-robustness" / "surviving_pairs.csv")
    add(text("by_pairs_surviving", sum(r["set"] == "A" for r in surviving),
             "docs/bayes-robustness/surviving_pairs.csv"))
    pairs = _csv(docs / "bayes-robustness" / "pairwise_probabilities.csv")
    credible = 0
    for r in pairs:
        if r["spec"] != "primary":
            continue
        for g in EU27:
            if g > r["row"]:
                p = float(r[g])
                credible += p >= 0.95 or p <= 0.05
    add(text("by_pairs_primary", credible, "docs/bayes-robustness/pairwise_probabilities.csv"))
    add(text("by_pairs_total", len(EU27) * (len(EU27) - 1) // 2, "EU-27"))
    widths = sorted(float(r["rank_high"]) - float(r["rank_low"]) for r in primary)
    add(text("by_rank_width", int(widths[len(widths) // 2]), "docs/bayes/country_estimates.csv"))
    focus = _csv(docs / "bayes-robustness" / "focus_pairs.csv")
    for r in focus:
        add(text(f"pair_{r['country_a']}_{r['country_b']}", r["verdict"].replace("_", " "),
                 "docs/bayes-robustness/focus_pairs.csv"))
    diagnostics = json.loads((docs / "bayes" / "diagnostics.json").read_text(encoding="utf-8"))
    check = diagnostics["primary"]["predictive_check"]
    add(num("by_lag1_obs", check["observed"]["lag1_autocorrelation"], 2,
            "docs/bayes/diagnostics.json"))
    add(num("by_lag1_rep", check["replicated_mean"]["lag1_autocorrelation"], 2,
            "docs/bayes/diagnostics.json"))
    add(pct("by_cov90", check["coverage_90"], 0, "docs/bayes/diagnostics.json"))
    dyn = json.loads((docs / "bayes-dynamics" / "dynamics.json").read_text(encoding="utf-8"))
    add(num("dyn_p", dyn["specifications"]["ardl_dynamic_factor"]["lag1_p_value"], 3,
            "docs/bayes-dynamics/dynamics.json"))
    add(num("dyn_factor_corr", dyn["specifications"]["ardl_dynamic_factor"][
        "factor_corr_mean_gdp"], 2, "docs/bayes-dynamics/dynamics.json", True))
    reg = json.loads((docs / "bayes-regimes" / "regimes.json").read_text(encoding="utf-8"))
    md = reg["specifications"]["regime_shift"]["mu_delta"]
    add(num("reg_delta", md["median"], 3, "docs/bayes-regimes/regimes.json", True))
    add(interval("reg_delta_ci", md["low"], md["high"], 3, "docs/bayes-regimes/regimes.json"))
    tau_delta = _report_number(docs / "BAYES_REGIMES.md",
                               r"\| `regime_shift` \|[^|]+\|[^|]+\|[^|]+\| \+?([\d.]+) \[")
    add(num("reg_tau", tau_delta[0], 2, "docs/BAYES_REGIMES.md"))

    # Which conclusions depend on the model and on acceptance ----------------------------
    add(num("by_lag1_p", check["p_value"]["lag1_autocorrelation"], 3,
            "docs/bayes/diagnostics.json"))
    mu_range = _report_number(report, r"across accepted fits the median of mu_theta ranges "
                                      r"from (-?\d+\.\d+) to (-?\d+\.\d+)")
    add(num("by_mu_lo", mu_range[0], 2, src, True))
    add(num("by_mu_hi", mu_range[1], 2, src, True))
    factor = _report_number(report, r"correlated with average GDP growth \(median (-?[\d.]+)\)")
    add(num("by_factor_corr", factor[0], 2, src, True))
    dyn_report = docs / "BAYES_DYNAMICS.md"
    adf = _report_number(dyn_report, r"`ardl_dynamic_factor` (-?[\d.]+) \[(-?[\d.]+), "
                                     r"(-?[\d.]+)\]")
    add(num("dyn_mu", adf[0], 2, "docs/BAYES_DYNAMICS.md", True))
    add(interval("dyn_mu_ci", adf[1], adf[2], 2, "docs/BAYES_DYNAMICS.md"))
    add(text("dyn_n_neg", len(dyn["specifications"]["ardl_dynamic_factor"]["credibly_negative"]),
             "docs/bayes-dynamics/dynamics.json"))
    for spec, key in (("ardl", "ardl"), ("ardl_dynamic_factor", "adf")):
        lu = _report_number(dyn_report, rf"\| Luxembourg weak \| `{spec}` \| ([\d.]+) \|")
        add(num(f"dyn_lu_{key}_p", lu[0], 3, "docs/BAYES_DYNAMICS.md"))
        add(text(f"dyn_lu_{key}", dyn["specifications"][spec]["claims"]["Luxembourg weak"],
                 "docs/bayes-dynamics/dynamics.json"))
    short = _report_number(dyn_report, r"in `ardl_dynamic_factor` the short-run `theta_LU` "
                                       r"gives ([\d.]+)")
    add(num("dyn_lu_adf_short_p", short[0], 2, "docs/BAYES_DYNAMICS.md"))
    unpooled = diagnostics["no_pooling"]
    add(num("by_unpooled_rhat", unpooled["max_rhat"], 4, "docs/bayes/diagnostics.json"))
    from eu_okun.review_report import COUNTRY_NAMES

    index = re.search(r"\[(\d+)\]", unpooled["max_rhat_parameter"])
    if index is None:
        raise ValueError("Unexpected parameter name in docs/bayes/diagnostics.json")
    add(text("by_unpooled_country", COUNTRY_NAMES[EU27[int(index.group(1))]],
             "docs/bayes/diagnostics.json"))
    with_unpooled = [g for g in robust
                     if _one(est, spec="no_pooling", geo=g)["sign"] == "negative"]
    add(text("by_n_neg_unpooled", len(with_unpooled), "docs/bayes/country_estimates.csv"))
    rep_dir = docs / "bayes" / "replication-linux"
    rep_diag = json.loads((rep_dir / "diagnostics.json").read_text(encoding="utf-8"))
    rep_est = _csv(rep_dir / "country_estimates.csv")
    rep_robust = [g for g in EU27 if all(_one(rep_est, spec=s, geo=g)["sign"] == "negative"
                                         for s, d in rep_diag.items() if d["accepted"])]
    add(text("by_n_neg_linux", len(rep_robust),
             "docs/bayes/replication-linux/country_estimates.csv"))
    differ = [g for g in robust if g not in with_unpooled]
    add(text("by_unpooled_differ", _names(differ), "docs/bayes/country_estimates.csv"))
    add(text("by_unpooled_n_differ", len(differ), "docs/bayes/country_estimates.csv"))
    p_unpooled = [float(_one(est, spec="no_pooling", geo=g)["p_negative"]) for g in differ]
    add(num("by_unpooled_p_lo", min(p_unpooled), 2, "docs/bayes/country_estimates.csv"))
    add(num("by_unpooled_p_hi", max(p_unpooled), 2, "docs/bayes/country_estimates.csv"))
    ols_clear = [g for g in differ if float(_one(est, spec="primary", geo=g)["classical_high"]) < 0]
    add(text("by_unpooled_ols", _names(ols_clear), "docs/bayes/country_estimates.csv"))
    add(text("by_unpooled_ols_n", len(ols_clear), "docs/bayes/country_estimates.csv"))

    # Prediction ------------------------------------------------------------------------
    pc = _csv(docs / "predictive" / "comparisons.csv")
    fam = [r for r in pc if r["kind"] == "family"]
    add(text("pc_decided", sum(r["verdict"] != "not distinguishable" for r in fam),
             "docs/predictive/comparisons.csv"))
    add(text("pc_total", len(fam), "docs/predictive/comparisons.csv"))
    for ev, model_b, key in (("E1", "zero_change", "e1_zero"), ("E2", "ols_dl", "e2_ols"),
                             ("E2", "zero_change", "e2_zero")):
        r = _one(fam, evaluation=ev, model_a="bayes_primary", model_b=model_b)
        add(pct(f"pc_{key}", float(r["ratio"]) - 1, 1, "docs/predictive/comparisons.csv", True))
        add(num(f"pc_{key}_holm", float(r["p_holm"]), 3, "docs/predictive/comparisons.csv"))
    fc = _csv(docs / "forecast-comparison" / "comparisons.csv")
    ffam = [r for r in fc if r["kind"] == "family"]
    add(text("fc_decided", sum(r["verdict"] != "not distinguishable" for r in ffam),
             "docs/forecast-comparison/comparisons.csv"))
    add(text("fc_total", len(ffam), "docs/forecast-comparison/comparisons.csv"))
    for a, b, key in (("bayes_gdp_ar1", "gdp_unemployment_ar1", "gdp"),
                      ("bayes_ar1", "unemployment_ar1", "ar1"),
                      ("bayes_gdp_ar1", "zero_change", "zero"),
                      ("bayes_gdp_ar1_factor", "bayes_gdp_ar1", "factor")):
        r = _one(ffam, model_a=a, model_b=b)
        add(pct(f"fc_{key}", float(r["ratio"]) - 1, 1, "docs/forecast-comparison/comparisons.csv",
                True))
        add(interval(f"fc_{key}_ci", 100 * (float(r["ratio_low"]) - 1),
                     100 * (float(r["ratio_high"]) - 1), 1,
                     "docs/forecast-comparison/comparisons.csv"))
        add(num(f"fc_{key}_holm", float(r["p_holm"]), 3,
                "docs/forecast-comparison/comparisons.csv"))
    fx = [r for r in fc if r["kind"] == "sensitivity: excluding pandemic"]
    add(text("fcx_decided", sum(r["verdict"] != "not distinguishable" for r in fx),
             "docs/forecast-comparison/comparisons.csv"))
    for a, b, key in (("bayes_gdp_ar1", "gdp_unemployment_ar1", "gdp"),
                      ("bayes_ar1", "unemployment_ar1", "ar1"),
                      ("bayes_gdp_ar1", "zero_change", "zero"),
                      ("bayes_gdp_ar1", "bayes_ar1", "lags"),
                      ("bayes_gdp_ar1_factor", "bayes_gdp_ar1", "factor")):
        r = _one(fx, model_a=a, model_b=b)
        add(pct(f"fcx_{key}", float(r["ratio"]) - 1, 1,
                "docs/forecast-comparison/comparisons.csv", True))
        add(num(f"fcx_{key}_holm", float(r["p_holm"]), 3,
                "docs/forecast-comparison/comparisons.csv"))
    gains = sorted(float(r["ratio"]) - 1 for r in fx if r["verdict"] != "not distinguishable")
    add(pct("fcx_gain_lo", -gains[-1], 1, "docs/forecast-comparison/comparisons.csv"))
    add(pct("fcx_gain_hi", -gains[0], 1, "docs/forecast-comparison/comparisons.csv"))
    add(num("fcx_holm_max", max(float(r["p_holm"]) for r in fx
                                if r["verdict"] != "not distinguishable"), 3,
            "docs/forecast-comparison/comparisons.csv"))
    add(text("fc_origins", ffam[0]["quarters"], "docs/forecast-comparison/comparisons.csv"))
    add(text("fcx_origins", fx[0]["quarters"], "docs/forecast-comparison/comparisons.csv"))
    add(text("pc_quarters", fam[0]["quarters"], "docs/predictive/comparisons.csv"))
    ar1 = _one(fc, kind="descriptive", model_a="unemployment_ar1", model_b="zero_change",
               loss="crps", targets="all")
    add(pct("fc_ar1_vs_zero", float(ar1["ratio"]) - 1, 1,
            "docs/forecast-comparison/comparisons.csv", True))
    nofail = _one(pc, kind="sensitivity", evaluation="E1", model_a="bayes_primary",
                  model_b="zero_change")
    add(num("pc_e1_zero_holm_nofail", float(nofail["p_holm"]), 3,
            "docs/predictive/comparisons.csv"))

    # Identification --------------------------------------------------------------------
    ok = _csv(docs / "identification" / "okun_sets.csv")
    prim = [r for r in ok if r["variant"] in ("MP", "OIL")]
    add(num("id_max_t", max(abs(float(r["first_stage_t"])) for r in prim), 2,
            "docs/identification/okun_sets.csv"))
    add(text("id_unbounded", sum(r["kind"] != "bounded" for r in prim),
             "docs/identification/okun_sets.csv"))
    add(text("id_sets", len(prim), "docs/identification/okun_sets.csv"))
    pr = _csv(docs / "identification" / "predictability.csv")
    add(num("id_pred_oil", float(_one(pr, shock="OIL")["p_value"]), 3,
            "docs/identification/predictability.csv"))
    add(num("id_pred_mp", float(_one(pr, shock="MP")["p_value"]), 3,
            "docs/identification/predictability.csv"))
    ms = _csv(docs / "monthly-identification" / "ratio_sets.csv")
    mprim = [r for r in ms if r["variant"] in ("MP", "OIL")]
    add(num("mo_max_t", max(abs(float(r["first_stage_t"])) for r in mprim), 2,
            "docs/monthly-identification/ratio_sets.csv"))
    add(text("mo_unbounded", sum(r["kind"] != "bounded" for r in mprim),
             "docs/monthly-identification/ratio_sets.csv"))
    mp = _csv(docs / "monthly-identification" / "predictability.csv")
    add(num("mo_pred_oil", float(_one(mp, shock="OIL")["p_value"]), 3,
            "docs/monthly-identification/predictability.csv"))
    from eu_okun.monthly_data import UNEMPLOYMENT_METHOD

    add(text("mo_n_direct", sum(m == "D" for m in UNEMPLOYMENT_METHOD.values()),
             "eu_okun.monthly_data.UNEMPLOYMENT_METHOD (MONTHLY_IDENTIFICATION_PROTOCOL.md)"))
    resp = _csv(docs / "identification" / "responses.csv")
    u0 = _one(resp, variant="MP", outcome="unemployment", h="0")
    add(num("id_mp_u0", float(u0["estimate"]), 2, "docs/identification/responses.csv", True))
    add(interval("id_mp_u0_ci", float(u0["low90"]), float(u0["high90"]), 2,
                 "docs/identification/responses.csv"))
    fe = _csv(docs / "shift-share" / "feasibility.csv")
    for s in ("P1", "P2", "P3"):
        add(num(f"ss_{s}_t", float(_one(fe, set=s)["expected_t"]), 2,
                "docs/shift-share/feasibility.csv"))
    add(text("ss_decision", _one(fe, set="P1")["decision"], "docs/shift-share/feasibility.csv"))
    sp = _csv(docs / "sign-restrictions" / "specifications.csv")
    add(pct("sr_nonempty", float(_one(sp, spec="primary")["nonempty_share"]), 0,
            "docs/sign-restrictions/specifications.csv"))
    add(pct("sr_nonempty_h01", float(_one(sp, spec="h01")["nonempty_share"]), 1,
            "docs/sign-restrictions/specifications.csv"))
    sm = _csv(docs / "sign-restrictions" / "summaries.csv")
    for shock, key in (("demand", "dem"), ("technology", "tech")):
        r = _one(sm, spec="primary", target=f"{shock}:okun:4")
        add(num(f"sr_{key}_uniform", float(r["single_median"]), 2,
                "docs/sign-restrictions/summaries.csv", True))
        bounded = math.isfinite(float(r["region_lower"])) and math.isfinite(
            float(r["region_upper"]))
        add(text(f"sr_{key}_bounded", "bounded" if bounded else "unbounded",
                 "docs/sign-restrictions/summaries.csv"))
    for shock, key in (("demand", "dem"), ("technology", "tech"), ("labour_supply", "ls")):
        fev = _one(sm, spec="primary", target=f"{shock}:fevd_du:8")
        add(pct(f"sr_{key}_fevd_hi", float(fev["region_upper"]), 0,
                "docs/sign-restrictions/summaries.csv"))
        add(pct(f"sr_{key}_fevd_lo", float(fev["region_lower"]), 0,
                "docs/sign-restrictions/summaries.csv"))
    gdp8 = _one(sm, spec="primary", target="demand:gdp:8")
    add(interval("sr_dem_gdp8_ci", float(gdp8["region_lower"]), float(gdp8["region_upper"]), 2,
                 "docs/sign-restrictions/summaries.csv"))
    both = {out[f"sr_{k}_bounded"].raw for k in ("dem", "tech")}
    add(text("sr_sets", f"both {both.pop()}" if len(both) == 1 else
             f"{out['sr_dem_bounded'].raw} for demand and {out['sr_tech_bounded'].raw} for "
             "technology", "docs/sign-restrictions/summaries.csv"))
    orders = _csv(docs / "sign-restrictions" / "orderings.csv")
    add(text("sr_n_orderings", len(orders), "docs/sign-restrictions/orderings.csv"))
    for h in (4, 8):
        vals = [float(r[f"okun_h{h}"]) for r in orders]
        add(num(f"sr_order_min_h{h}", max(vals), 3, "docs/sign-restrictions/orderings.csv", True))
        add(num(f"sr_order_max_h{h}", min(vals), 3, "docs/sign-restrictions/orderings.csv", True))

    # Case study -------------------------------------------------------------------------
    hp = [r for r in _csv(docs / "heterogeneity" / "predictions.csv") if r["setting"] == "primary"]
    add(text("cs_supported", sum(r["verdict"] == "supported" for r in hp),
             "docs/heterogeneity/predictions.csv"))
    add(text("cs_total", len(hp), "docs/heterogeneity/predictions.csv"))
    add(text("cs_contradicted", sum(r["verdict"] == "contradicted" for r in hp),
             "docs/heterogeneity/predictions.csv"))
    for pkey, key in (("P1.1", "p11"), ("P1.3", "p13"), ("P2.1a", "p21a"), ("P2.2", "p22"),
                      ("P2.3b", "p23b"), ("P3.1a", "p31a"), ("P4.1 ES", "p41es"),
                      ("P5.2", "p52")):
        r = _one(hp, key=pkey)
        add(num(f"cs_{key}", float(r["estimate"]), 2, "docs/heterogeneity/predictions.csv", True))
        add(interval(f"cs_{key}_ci", float(r["low"]), float(r["high"]), 2,
                     "docs/heterogeneity/predictions.csv"))
        add(num(f"cs_{key}_holm", float(r["holm"]), 3, "docs/heterogeneity/predictions.csv"))
        add(text(f"cs_{key}_verdict", r["verdict"], "docs/heterogeneity/predictions.csv"))
    every = _csv(docs / "heterogeneity" / "predictions.csv")
    for pkey, key in (("P2.2", "p22"), ("P3.1a", "p31a"), ("P4.1 ES", "p41es")):
        add(text(f"cs_{key}_settings", ", ".join(
            r["setting"].replace("_", "-") for r in every
            if r["key"] == pkey and r["setting"] != "primary" and r["verdict"] == "supported")
            or "none", "docs/heterogeneity/predictions.csv"))
    pairs = [r for r in _csv(docs / "heterogeneity" / "pairs.csv")
             if r["setting"] == "primary" and r["measure"] != "neg_du"]
    for status, key in (("remains", "remain"), ("disappears", "disappear"),
                        ("reverses", "reverse")):
        add(text(f"cs_pairs_{key}", sum(r["status"] == status for r in pairs),
                 "docs/heterogeneity/pairs.csv"))
    for a, b in (("ES", "PT"), ("ES", "NL"), ("ES", "LU"), ("NL", "LU")):
        add(text(f"cs_pair_{a}_{b}_h", _one(pairs, a=a, b=b, measure="h")["status"],
                 "docs/heterogeneity/pairs.csv"))
    el = _csv(docs / "heterogeneity" / "elasticities.csv")
    for geo, measure in (("LU", "prod"), ("LU", "e_dom_pe"), ("LU", "e_res"), ("NL", "prod"),
                         ("ES", "prod"), ("ES", "e_dom"), ("ES", "h"), ("PT", "h"),
                         ("PT", "dp"), ("NL", "e_dom"), ("PT", "du")):
        r = _one(el, geo=geo, measure=measure)
        add(num(f"cs_{geo}_{measure}", float(r["elasticity"]), 2,
                "docs/heterogeneity/elasticities.csv", True))
        add(interval(f"cs_{geo}_{measure}_ci", float(r["low"]), float(r["high"]), 2,
                     "docs/heterogeneity/elasticities.csv"))
    cor = _csv(docs / "heterogeneity" / "correlations.csv")
    for variant, indicator, key in (("classical", "epl_r", "eplr_cl"),
                                    ("Bayesian, without LU and IE", "epl_t", "eplt_noluie"),
                                    ("Bayesian", "epl_t", "eplt_by")):
        r = _one(cor, variant=variant, indicator=indicator)
        add(num(f"cs_{key}_rho", float(r["rho"]), 2, "docs/heterogeneity/correlations.csv", True))
        add(num(f"cs_{key}_holm", float(r["holm"]), 3, "docs/heterogeneity/correlations.csv"))
        add(text(f"cs_{key}_n", r["n"], "docs/heterogeneity/correlations.csv"))
    add(text("cs_n_corr", len(cor), "docs/heterogeneity/correlations.csv"))

    # Monthly smoothing in the quarterly rate (SMOOTHING_PROTOCOL.md) ----------------
    sm = json.loads((docs / "smoothing" / "summary.json").read_text(encoding="utf-8"))
    src = "docs/smoothing/summary.json"
    add(text("sm_n_average", len(sm["average_of_monthly"]), src))
    add(text("sm_smoothed_average", sum(g in ("DE", "NL", "EE", "PT", "HR", "LU")
                                        for g in sm["average_of_monthly"]), src))
    add(num("sm_misfit_all", sm["misfit"]["observed_all"], 2, src, True))
    add(num("sm_misfit_without", sm["misfit"]["observed_without_smoothed"], 2, src, True))
    add(num("sm_misfit_smoothed", sm["misfit"]["observed_smoothed"], 2, src, True))
    add(num("sm_misfit_direct", sm["misfit"]["observed_direct"], 2, src, True))
    add(text("sm_p1_verdict", sm["verdicts"]["P1"], src))
    add(text("sm_p2_verdict", sm["verdicts"]["P2"], src))
    add(num("sm_lu_auto", sm["luxembourg"]["autocorrelation"], 2, src, True))
    add(num("sm_lu_corr", sm["luxembourg"]["change_corr"], 2, src))
    inh = {r["geo"]: r for r in _csv(docs / "smoothing" / "inheritance.csv")}
    add(num("sm_nl_corr", float(inh["NL"]["change_corr"]), 2, "docs/smoothing/inheritance.csv"))
    tests = {r["key"]: r for r in _csv(docs / "smoothing" / "tests.csv")
             if r["setting"] == "primary"}
    for key in ("P1", "P2"):
        add(num(f"sm_{key.lower()}_smoothed", float(tests[key]["mean_smoothed"]), 2,
                "docs/smoothing/tests.csv", True))
        add(num(f"sm_{key.lower()}_direct", float(tests[key]["mean_comparison"]), 2,
                "docs/smoothing/tests.csv", True))

    # Measurement noise or finance (GDP_MEASUREMENT_PROTOCOL.md) ---------------------
    src = "docs/gdp-measurement/predictions.csv"
    gm = [r for r in _csv(docs / "gdp-measurement" / "predictions.csv")
          if r["setting"] == "primary"]
    add(text("gm_total", len(gm), src))
    add(text("gm_supported", sum(r["verdict"] == "supported" for r in gm), src))
    add(text("gm_contradicted", sum(r["verdict"] == "contradicted" for r in gm), src))
    for pkey, key in (("P1.1 LU", "p11_lu"), ("P1.2 LU", "p12_lu"), ("P1.1 IE", "p11_ie"),
                      ("P1.2 IE", "p12_ie"), ("P2.1 LU", "p21_lu"), ("P2.2 LU", "p22_lu")):
        r = _one(gm, key=pkey)
        add(num(f"gm_{key}_focus", float(r["focus_value"]), 2, src, True))
        add(num(f"gm_{key}_median", float(r["group_median"]), 2, src, True))
        add(num(f"gm_{key}", float(r["estimate"]), 2, src, True))
        add(interval(f"gm_{key}_ci", float(r["low"]), float(r["high"]), 2, src))
        holm_p = float(r["holm"])
        add(text(f"gm_{key}_holm", "below 0.001", src) if holm_p < 0.001
            else num(f"gm_{key}_holm", holm_p, 3, src))
        add(text(f"gm_{key}_verdict", r["verdict"], src))
    st = {r["geo"]: r for r in _csv(docs / "gdp-measurement" / "statistics.csv")}
    src = "docs/gdp-measurement/statistics.csv"
    for stat in ("c_q", "c_y", "prod", "prod_n"):
        add(num(f"gm_lu_{stat}", float(st["LU"][stat]), 2, src, True))
    de = {r["geo"]: r for r in _csv(docs / "gdp-measurement" / "descriptive.csv")}
    src = "docs/gdp-measurement/descriptive.csv"
    add(pct("gm_lu_fin_va", float(de["LU"]["finance_share_va"]), 0, src))
    add(pct("gm_lu_fin_hours", float(de["LU"]["finance_share_hours"]), 0, src))
    others = sorted(float(r["finance_share_va"]) for g, r in de.items() if g != "LU")
    add(pct("gm_fin_va_next", others[-1], 0, src))
    add(text("cs_n_corr_contra", sum(r["verdict"] == "contradicted" for r in cor),
             "docs/heterogeneity/correlations.csv"))

    # Post-hoc pandemic-window sensitivity (not registered) ------------------------------
    ws = _csv(docs / "window-sensitivity" / "window_sensitivity.csv")
    src = "docs/window-sensitivity/window_sensitivity.csv"
    for design, key in (("quarterly LP", "q"), ("monthly LP", "m")):
        for margin in ("0", "1"):
            mine = [r for r in ws if r["design"] == design and r["margin"] == margin]
            add(num(f"ws_{key}{margin}_max_t",
                    max(abs(float(r["value"])) for r in mine
                        if r["statistic"] == "first_stage_t"), 2, src))
            add(text(f"ws_{key}{margin}_unbounded",
                     sum(r["value"] != "bounded" for r in mine if r["statistic"] == "set"), src))
    svar = [r for r in ws if r["design"] == "sign-restricted SVAR" and r["margin"] == "1"]
    add(pct("ws_sr_nonempty", float(next(r["value"] for r in svar
                                         if r["statistic"] == "nonempty_share")), 0, src))
    regions = {r["value"] for r in svar if r["statistic"] == "robust_region"}
    add(text("ws_sr_regions", f"all {regions.pop()}" if len(regions) == 1 else "mixed", src))
    return out


# ----------------------------------------------------------------------------------
# Audit
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    group: str
    name: str
    passed: bool
    detail: str


def _snapshot_checks(root: Path) -> list[Check]:
    from eu_okun.audit import audit_snapshot
    from eu_okun.external_shocks import audit_shock_snapshot
    from eu_okun.gdp_measurement_data import audit_gdp_measurement
    from eu_okun.heterogeneity_data import audit_heterogeneity
    from eu_okun.monthly_data import audit_monthly_snapshot
    from eu_okun.shift_share import audit_shift_share
    from eu_okun.sign_restrictions import audit_deflator

    snap = root / "snapshots"
    audits: list[tuple[str, Callable[[Path], Any], Path]] = [
        ("Eurostat quarterly 2026-10-15", audit_snapshot, snap / "eurostat-2026-10-15"),
        ("External shocks 2026-10-16", audit_shock_snapshot, snap / "external-shocks-2026-10-16"),
        ("Eurostat monthly 2026-10-18", audit_monthly_snapshot,
         snap / "eurostat-monthly-2026-10-18"),
        ("OECD shift-share 2026-10-18", audit_shift_share, snap / "shift-share-2026-10-18"),
        ("Eurostat deflator 2026-10-18", audit_deflator, snap / "eurostat-deflator-2026-10-18"),
        ("Case-study inputs 2026-10-18", audit_heterogeneity, snap / "heterogeneity-2026-10-18"),
        ("Value added and hours by industry 2026-10-19", audit_gdp_measurement,
         snap / "gdp-measurement-2026-10-19"),
    ]
    out = []
    for name, fn, path in audits:
        try:
            fn(path)
            out.append(Check("snapshot", name, True, "hashes match; processed files re-derive"))
        except Exception as exc:  # noqa: BLE001 - every failure is reported as a failed check
            out.append(Check("snapshot", name, False, f"{type(exc).__name__}: {exc}"))
    archived = snap / "ci-results-38754743850"
    try:
        audit_archived_outputs(archived)
        out.append(Check("snapshot", "Archived CI outputs, run 38754743850", True,
                         "hashes match"))
    except Exception as exc:  # noqa: BLE001 - every failure is reported as a failed check
        out.append(Check("snapshot", "Archived CI outputs, run 38754743850", False,
                         f"{type(exc).__name__}: {exc}"))
    return out


def audit_archived_outputs(directory: Path) -> None:
    """Every file of a frozen output snapshot matches its manifest, and nothing is extra."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    listed = {f["file"]: f for f in manifest["files"]}
    present = {p.relative_to(directory).as_posix() for p in directory.rglob("*.json")
               if p.name != "manifest.json"}
    if present != set(listed):
        raise ValueError(f"files differ from the manifest: {sorted(present ^ set(listed))}")
    for name, f in listed.items():
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != f["sha256"] or len(data) != f["bytes"]:
            raise ValueError(f"{name} does not match its recorded hash")


def _gap_detail(gap: float, tolerance: float) -> str:
    """The exact gap only on failure: rounding noise differs between platforms."""
    if gap < tolerance:
        return f"largest difference below {tolerance:.0e}"
    return f"largest difference {gap:.1e}"


def _consistency_checks(root: Path) -> list[Check]:
    docs = root / "docs"
    out = []
    est = _csv(docs / "bayes" / "country_estimates.csv")
    summ = _csv(docs / "bayes-robustness" / "country_summaries.csv")
    gaps = [abs(float(_one(est, spec="primary", geo=g)["theta_median"])
                - float(_one(summ, spec="primary", geo=g)["median"])) for g in EU27]
    out.append(Check("consistency", "Bayesian primary medians: report table = robustness table",
                     max(gaps) < 1e-12, _gap_detail(max(gaps), 1e-12)))
    # The Bayesian report's OLS column uses the likelihood's rows: 2009-2019 with the terms
    # that rely on a flagged Eurostat value masked. Rebuild those rows and re-fit.
    from eu_okun.baseline import _read_panel
    from eu_okun.bayes_data import build_bayes_data
    from eu_okun.bayes_fit import classical_on_same_rows
    from eu_okun.review import read_flags

    csv_path = root / "snapshots" / "eurostat-2026-10-15" / "processed" / "eu27_quarterly.csv"
    refit = classical_on_same_rows(build_bayes_data(_read_panel(csv_path), read_flags(csv_path)))
    gaps = [abs(float(_one(est, spec="primary", geo=g)["classical_theta"]) - refit[g]["theta"])
            for g in EU27]
    out.append(Check("consistency", "OLS column of the Bayesian report = OLS re-fitted on the "
                     "same (flag-masked, 2009-2019) rows", max(gaps) < 1e-9,
                     _gap_detail(max(gaps), 1e-9)))
    verdicts = claim_verdicts(docs)
    report = (docs / "HIERARCHICAL_BAYES.md").read_text(encoding="utf-8")
    quoted = re.search(r"Luxembourg weak: \*\*(.+?)\*\*; Netherlands delayed: \*\*(.+?)\*\*; "
                       r"Spain large: \*\*(.+?)\*\*", report)
    expected = (verdicts["Luxembourg weak"], verdicts["Netherlands delayed"],
                verdicts["Spain large"])
    out.append(Check("consistency", "Claim verdicts recomputed from claims.csv and acceptance = "
                     "report", quoted is not None and quoted.groups() == expected,
                     f"recomputed {expected}"))
    values = collect_values(root)
    quoted_neg = re.search(r"Credibly negative in every accepted specification: (\d+) countries",
                           report)
    out.append(Check("consistency", "Countries credibly negative in every accepted fit = report",
                     quoted_neg is not None and int(quoted_neg.group(1)) == values[
                         "by_n_neg_all"].raw, f"recomputed {values['by_n_neg_all'].raw}"))
    for design, registered in (("q", "id_max_t"), ("m", "mo_max_t")):
        gap = abs(float(values[f"ws_{design}0_max_t"].raw) - float(values[registered].raw))
        out.append(Check("consistency", f"Window sensitivity at margin 0 = registered "
                         f"{registered}", gap < 1e-4, _gap_detail(gap, 1e-4)))
    fc = (docs / "FORECAST_COMPARISON.md").read_text(encoding="utf-8")
    out.append(Check("consistency", "Forecast comparisons decided = report",
                     f"{values['fc_decided'].raw} of {values['fc_total'].raw} pre-registered "
                     "comparisons are decided" in fc, f"{values['fc_decided'].raw} of "
                     f"{values['fc_total'].raw}"))
    return out


def narrative_quotes(values: Mapping[str, Value]) -> list[tuple[str, str]]:
    """(document, text that must appear in it): numbers quoted outside generated reports."""
    v = {k: x.md.replace(MINUS, "-") for k, x in values.items()}
    return [
        ("README.md", f"expected first-stage t is {v['ss_P1_t']}"),
        ("README.md", f"(P2: t = {v['ss_P2_t']})"),
        ("README.md", f"(P3: t = {v['ss_P3_t']})"),
        ("README.md", f"The largest first-stage |t| is {v['mo_max_t']} with monthly"),
        ("README.md", f"against {v['id_max_t']} with quarterly GDP"),
        ("README.md", f"elasticity {v['cs_LU_prod']}"),
        ("README.md", f"({v['cs_NL_prod']} against {v['cs_ES_prod']})"),
        ("README.md", f"({v['cs_PT_h']} against {v['cs_ES_h']})"),
        ("README.md", f"demand {v['sr_dem_uniform']}, technology {v['sr_tech_uniform']}"),
        ("README.md", f"between {v['sr_order_min_h4']} and {v['sr_order_max_h4']} at 4 "
                      f"quarters and between {v['sr_order_min_h8']} and {v['sr_order_max_h8']} "
                      "at 8 quarters"),
        ("docs/IDENTIFICATION_STUDY.md", f"|t| at most {v['id_max_t']}"),
        ("docs/IDENTIFICATION_STUDY.md", f"(p = {v['mo_pred_oil']})"),
        ("docs/IDENTIFICATION_STUDY.md", f"(p = {v['id_pred_oil']})"),
        ("docs/IDENTIFICATION_STUDY.md", f"expected first-stage t is {v['ss_P1_t']}"),
        ("docs/IDENTIFICATION_STUDY.md", f"between {v['sr_order_min_h4']} and "
                                         f"{v['sr_order_max_h4']} at 4 quarters and between "
                                         f"{v['sr_order_min_h8']} and {v['sr_order_max_h8']} "
                                         "at 8 quarters"),
        ("ROADMAP.md", f"first-stage t is {v['ss_P1_t']} against 3.16"),
        ("ROADMAP.md", f"None of the {v['ss_published_holm']} published Holm-significant changes"),
        ("README.md", f"rejects {v['ss_size_normal']} of the time at a nominal 5%"),
        ("docs/IDENTIFICATION_STUDY.md",
         f"the unemployment change spans {v['sr_dem_fevd_lo'].rstrip('%')}-"
         f"{v['sr_dem_fevd_hi']}"),
    ]


def _narrative_checks(root: Path, values: Mapping[str, Value]) -> list[Check]:
    out = []
    for doc, quote in narrative_quotes(values):
        body = (root / doc).read_text(encoding="utf-8").replace(MINUS, "-")
        flat = re.sub(r"\s+", " ", body)
        out.append(Check("narrative", f"{doc}: \"{quote}\"", quote in flat,
                         "quoted as in the tables" if quote in flat else "not found"))
    return out


def audit(root: Path) -> list[Check]:
    """Snapshot audits, consistency between outputs, and quoted numbers in the documents."""
    values = collect_values(root)
    return _snapshot_checks(root) + _consistency_checks(root) + _narrative_checks(root, values)
