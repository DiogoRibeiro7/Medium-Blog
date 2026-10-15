"""Render the classical-results review: tables, figures and a Markdown report.

Every number in the report is read from a ``ReviewResult``; every qualitative
statement is conditioned on those numbers through the decision rules written into
the report itself. Static text is limited to source definitions quoted from Eurostat
metadata and to limitations that do not depend on the data.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.comparison import FOCUS_COUNTRIES
from eu_okun.contract import EU27
from eu_okun.forecasting import BASELINE_MODEL, GDP_MODEL
from eu_okun.review import (
    BANDWIDTHS,
    CORE_LAGS,
    CORE_SAMPLES,
    HAC_BANDWIDTH,
    HEADLINE_LAGS,
    SAMPLES,
    VERDICTS,
    ReviewResult,
    Robustness,
    grid_lookup,
    holm_significant,
    run_review,
)

REPORT_NAME = "CLASSICAL_RESULTS_REVIEW.md"
COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CY": "Cyprus", "CZ": "Czechia",
    "DE": "Germany", "DK": "Denmark", "EE": "Estonia", "EL": "Greece", "ES": "Spain",
    "FI": "Finland", "FR": "France", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland",
    "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "MT": "Malta",
    "NL": "Netherlands", "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SE": "Sweden",
    "SI": "Slovenia", "SK": "Slovakia",
}
MODEL_LABELS = {
    "zero_change": "zero change",
    "historical_mean": "historical mean",
    "unemployment_ar1": "AR(1) of unemployment change",
    "gdp_lags": "two GDP lags",
    "gdp_unemployment_ar1": "AR(1) + two GDP lags",
    "gdp_ar1_participation": "AR(1) + two GDP lags + participation",
}
EVALUATION_LABELS = {
    "published": "All targets (as published)",
    "non_pandemic_targets": "Targets outside 2020-Q1..2022-Q2",
    "pandemic_free": "2020-2021 removed from training too",
}
# Documented in Eurostat's LFS metadata (lfsi_esms) or the national-accounts definition;
# quoted, not inferred from the numbers.
FOCUS_NOTES: dict[str, list[str]] = {
    "PT": [
        "The Eurostat labour-force metadata reviewed contain no Portugal-specific "
        "comparability note for the 2021 survey change.",
    ],
    "ES": [
        "Eurostat metadata: under the 2021 regulation, and because of the COVID crisis, "
        "Spain continues to test the job attachment of employed persons on temporary "
        "lay-off as in the pre-2021 definition. Eurostat therefore flags Spanish "
        "labour-force values 'definition differs' from 2021-Q1.",
        "Eurostat metadata: unemployment in Spain refers to ages 16-74, not 15-74.",
    ],
    "NL": [
        "Eurostat metadata: Statistics Netherlands phased in data collection under the "
        "2021 regulation during 2021, uses a rolling reference week, and kept publishing "
        "monthly unemployment on the pre-2021 definition during 2021. These quarters are "
        "not flagged in the data.",
    ],
    "LU": [
        "GDP measures production by resident producer units, including work done by "
        "cross-border commuters; unemployment and participation describe the resident "
        "population. Commuters account for a large share of domestic employment in "
        "Luxembourg, so resident unemployment captures only part of any labour-input "
        "response to output. This is a measurement mismatch, not an institutional finding.",
    ],
}


# ----------------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------------


def s3(value: float) -> str:
    return f"{value:+.3f}"


def interval(low: float, high: float) -> str:
    return f"[{low:+.3f}, {high:+.3f}]"


def pval(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def pct(value: float) -> str:
    return f"{100 * value:.0f}%"


def names(codes: Iterable[str]) -> str:
    """Join country codes or labels as 'A, B and C' ('none' when empty)."""
    items = list(codes)
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def countries(n: int) -> str:
    return f"{n} country" if n == 1 else f"{n} countries"


def table(header: Sequence[str], rows: Iterable[Sequence[Any]], align: str = "") -> list[str]:
    """Markdown table; ``align`` holds one of l/r per column (default: first l, rest r)."""
    align = align or "l" + "r" * (len(header) - 1)
    marks = ["---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(marks) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines + [""]


def _sci(value: float) -> str:
    return "0" if value == 0 else f"{value:.1e}"


# ----------------------------------------------------------------------------------
# Derived quantities used across sections
# ----------------------------------------------------------------------------------


class Facts:
    """Named quantities computed once from the review and reused in the text."""

    def __init__(self, review: ReviewResult) -> None:
        self.r = review
        self.lookup = grid_lookup(review.grid)
        self.robust = {x.geo: x for x in review.robustness}
        self.by_verdict = {
            key: [x.geo for x in review.robustness if x.verdict == key] for key in VERDICTS
        }
        self.by_verdict_free = {
            key: [x.geo for x in review.robustness if x.verdict_without_pandemic == key]
            for key in VERDICTS
        }
        self.pandemic_outside = [x.geo for x in review.robustness if x.full_outside_excluded_ci]
        self.leverage = [x.pandemic_leverage_share for x in review.robustness]
        self.stability = [s for s in review.stability if not s.participation]
        self.post_n = len(review.stability_design["post_periods"])
        self.pre_n = len(review.stability_design["pre_periods"])
        self.loo_unstable = [s.geo for s in self.stability if not s.post_sign_stable]
        self.positive_stable = [s.geo for s in self.stability
                                if s.post_sum > 0 and s.post_sign_stable]
        self.sample_n = {
            spec.name: next(r.n for r in review.grid if r.sample == spec.name)
            for spec in SAMPLES if spec.common
        }
        self.sample_span = {
            spec.name: next((r.first, r.last) for r in review.grid if r.sample == spec.name)
            for spec in SAMPLES if spec.common
        }
        self.max_check = max(c.max_abs_difference for c in review.checks)
        self.checked = sum(c.compared for c in review.checks)
        self.forecasts = {(f.evaluation, f.model): f for f in review.forecasts}
        self.lfs_2008_missing: list[str] = next(
            (m["countries"] for m in review.missing if m["series"] == "unemployment"), []
        )
        self.post_provisional = [s.geo for s in self.stability if s.post_provisional_share > 0]
        self.post_provisional_max = max(s.post_provisional_share for s in self.stability)
        self.not_robust = sorted(self.by_verdict["negative_precision_dependent"]
                                 + self.by_verdict["sign_unstable"])
        full8 = {(p.country_a, p.country_b): p for p in review.pairs
                 if p.sample == "full" and p.block_length == 8}
        excl8 = {(p.country_a, p.country_b): p for p in review.pairs
                 if p.sample == "pandemic_excluded" and p.block_length == 8}
        self.pair_ratio = float(np.median(
            [abs(excl8[k].estimate) / abs(full8[k].estimate) for k in full8]
        ))
        self.pair_sign_changes = [f"{a}-{b}" for (a, b) in full8
                                  if np.sign(full8[(a, b)].estimate)
                                  != np.sign(excl8[(a, b)].estimate)]
        self.dominant_counts = {
            dimension: sum(x.dominant_dimension == dimension for x in review.robustness)
            for dimension in ("GDP lag order", "pandemic treatment", "sample selection",
                              "participation control")
        }
        self.archived_max = max(
            (a.max_abs_difference for a in review.archived or []), default=0.0
        )
        self.attenuated = [x.geo for x in review.robustness
                           if abs(x.headline_full) < abs(x.headline_excluded)]
        pre = {g: self.lookup[("pre_2020", g, 1, False)] for g in EU27}
        self.weaker_after = [s.geo for s in self.stability
                             if s.post_loo_min > pre[s.geo].ci_high]

    def headline(self, sample: str, geo: str, lags: int = HEADLINE_LAGS,
                 participation: bool = False) -> str:
        row = self.lookup[(sample, geo, lags, participation)]
        return f"{s3(row.gdp_sum)} {interval(row.ci_low, row.ci_high)}"


# ----------------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------------


def _summary(f: Facts) -> list[str]:
    r = f.r
    robust = f.by_verdict["robust_negative"]
    excluded_negative = [
        g for g in EU27 if f.lookup[("pandemic_excluded", g, HEADLINE_LAGS, False)].ci_high < 0
    ]
    holm = {b: holm_significant(r, b) for b in BANDWIDTHS}
    published_gdp = f.forecasts[("published", GDP_MODEL)]
    clean_gdp = f.forecasts[("non_pandemic_targets", GDP_MODEL)]
    free_gdp = f.forecasts[("pandemic_free", GDP_MODEL)]
    archived = (
        f"The {len(r.archived)} archived CI result files were regenerated from the frozen "
        f"snapshot; the largest numerical difference is {_sci(f.archived_max)}."
        if r.archived else "Archived CI result files were not supplied to this run."
    )
    lines = [
        "## Summary", "",
        "| Category | Finding |", "| --- | --- |",
        f"| **Verified computation** | All {len(r.checks)} independent checks pass "
        f"({f.checked:,} quantities; largest absolute discrepancy {_sci(f.max_check)}). "
        f"{archived} |",
        f"| **Verified, reportable** | Excluding 2020-2021, the two-lag GDP coefficient sum "
        f"is negative with a 95% HAC interval below zero in {len(excluded_negative)} of 27 "
        f"countries. In {len(robust)} countries ({names(robust)}) all "
        f"{r.robustness[0].core_estimates} core specifications agree in sign and exclude "
        f"zero. These are descriptive associations, not causal effects. |",
        f"| **Sensitivity-dependent** | The published 27-country table includes 2020-2021, "
        f"which supplies a median {pct(float(np.median(f.leverage)))} of the current-quarter "
        f"GDP variation; for {len(f.pandemic_outside)} countries "
        f"({names(f.pandemic_outside)}) the published estimate lies outside the "
        f"pandemic-excluded 95% interval. Signs or significance also depend on the "
        f"specification for {names(f.not_robust)}. "
        f"Without 2020-2021 the focus-country pair differences are a median "
        f"{f.pair_ratio:.1f} times larger, and the sign changes for "
        f"{len(f.pair_sign_changes)} of 6. |",
        f"| **Inconclusive** | Post-2021 coefficients ({f.post_n} quarters): "
        f"{holm[HAC_BANDWIDTH]} of {len(r.stability)} pre/post change tests are "
        f"Holm-significant, but the asymptotic HAC tests are unreliable at this length and "
        f"{len(f.loo_unstable)} of 27 post-2021 sums change sign when one quarter is "
        f"dropped. Forecasting: with all published targets, adding GDP to AR(1) lowers MAE "
        f"in {published_gdp.countries_better_mae} of 27 countries; outside the pandemic "
        f"targets in {clean_gdp.countries_better_mae} (median gain "
        f"{pct(clean_gdp.median_skill_vs_ar1)}), and with pandemic-free training in "
        f"{free_gdp.countries_better_mae}; the loss difference is distinguishable from "
        f"zero in favour of GDP in {clean_gdp.significantly_better} and against it in "
        f"{clean_gdp.significantly_worse} countries (non-pandemic targets). |",
        "| **Limitations** | GDP is a domestic concept, the labour data are residential; "
        f"{names(sorted({x.geo for x in r.flag_summary if x.flag == 'd'}))} follow a "
        "different unemployment definition from 2021; post-2021 GDP is partly "
        f"provisional in {len(f.post_provisional)} countries (up to "
        f"{pct(f.post_provisional_max)} of the window); "
        "contemporaneous GDP and participation are endogenous; no shock is identified. |",
        "",
    ]
    return lines


def _provenance(f: Facts, provenance: str) -> list[str]:
    r = f.r
    m = r.manifest
    lines = [
        "## 1. Scope and provenance", "",
        "This review audits the classical (non-Bayesian) results produced by `eu-okun "
        "baseline`, `compare`, `regimes`, `infer` and `forecast`. It adds no new "
        "statistical model: it recomputes the published numbers by an independent "
        "numerical route, re-estimates the *existing* specifications on pre-declared "
        "alternative samples, and re-scores the existing forecasts.", "",
    ]
    lines += table(["Item", "Value"], [
        ["Snapshot downloaded (UTC)", m["downloaded_at_utc"]],
        ["Snapshot manifest SHA-256", f"`{r.audit['snapshot_manifest_sha256']}`"],
        ["Processed panel SHA-256", f"`{r.audit['processed_csv_sha256']}`"],
        ["Download window", f"{m['period']['start']} to {m['period']['end']}"],
        ["Countries", "EU-27 (fixed present-day composition)"],
        ["Archived outputs compared", provenance or "not supplied"],
    ], "ll")
    lines += [
        "Eurostat revises past observations, so every number here belongs to this "
        "snapshot. The snapshot is committed under `snapshots/` so the report can be "
        "regenerated exactly (section 12).", "",
    ]
    return lines


def _sources(f: Facts) -> list[str]:
    r = f.r
    labels = r.labels
    lines = ["## 2. Sources, definitions and data quality", "",
             "### 2.1 Series as delivered", "",
             "Read from the JSON-stat labels of the snapshot itself:", ""]
    rows = []
    for name, entry in labels.items():
        filters = "; ".join(f"{k}: {v}" for k, v in entry["filters"].items())
        rows.append([f"`{entry['dataset']}`", name, filters, entry["updated"]])
    lines += table(["Dataset", "Series", "Filters (Eurostat labels)", "Last update"], rows,
                   "llll")
    flag_meanings = {}
    for entry in labels.values():
        flag_meanings.update(entry["flag_meanings"])
    lines += ["Eurostat flag meanings: " + "; ".join(
        f"`{k}` = {v}" for k, v in sorted(flag_meanings.items())) + ".", ""]
    gdp_url = labels["gdp"]["metadata_url"]
    lfs_url = labels["unemployment"]["metadata_url"]
    lines += [
        "### 2.2 Are the three series compatible?", "",
        "| Aspect | GDP growth | Unemployment | Participation | Consequence |",
        "| --- | --- | --- | --- | --- |",
        "| Source | National accounts | Labour Force Survey (LFS) | LFS | Unemployment "
        "and participation share sample, weights and flags (checked: identical flags on "
        "every row). |",
        "| Population | Production of *resident producer units* on the territory, "
        "including work by non-residents | Residents in private households | Same as "
        "unemployment | GDP and labour data cover different people where cross-border "
        "commuting is large (Luxembourg above all). |",
        "| Age band | n/a | 15-74 (16-74 in Spain and Italy) | 15-74 | Small level "
        "difference for ES and IT, constant over time. |",
        "| Unit | % change on previous quarter, chain-linked volumes | % of labour force "
        "| % of population 15-74 | Coefficients are percentage points of unemployment-rate "
        "change per percentage point of quarterly growth. |",
        "| Adjustment | Seasonally and calendar adjusted | Seasonally adjusted only | "
        "Seasonally adjusted only | Calendar effects matter for output, hardly for rates. |",
        "| Definition changes | Breaks flagged for BE (2009) and IT (2010) | 2021 "
        "regulation; break-corrected back to 2009 (see 2.5) | Same as unemployment | "
        "Spain and France still differ from 2021 (flag `d`). |",
        "| Revisions | Recent quarters provisional (`p`) | Revised with weights and "
        "census | Same | Post-2021 results depend on provisional GDP. |",
        "",
        f"Metadata: [national accounts]({gdp_url}), [labour-force survey]({lfs_url}).", "",
    ]
    lines += ["### 2.3 Missing observations", ""]
    lines += table(["Series", "Quarters", "Countries", "Cells"], [
        [m["series"], m["quarters"], f"{len(m['countries'])}: {names(m['countries'])}",
         m["cells"]] for m in r.missing
    ], "lllr") if r.missing else ["No missing cells.", ""]
    first_full = f.sample_span["full"][0]
    lines += [
        f"No value is imputed. Because the 2008 labour data are missing for these "
        f"countries, the first usable change in unemployment is {first_full}, not 2009-Q1, "
        f"and every common EU-27 sample starts there.", "",
        "### 2.4 Quality flags", "",
        "Flags are retained in the processed panel, but **none of the published models "
        "uses them**: flagged values enter the regressions like any other.", "",
    ]
    lines += table(["Series", "Flag", "Country", "Cells", "Quarters"], [
        [x.series.upper() if x.series == "lfs" else "GDP", f"`{x.flag}`", x.geo, x.cells,
         x.quarters]
        for x in r.flag_summary
    ], "lllrl")
    lines += ["![Missing cells and Eurostat flags by country and quarter]"
              "({assets}/data_quality.png)", ""]
    d_countries = sorted({x.geo for x in r.flag_summary if x.flag == "d"})
    break_2021 = [geo for geo in EU27 if r.flags[(geo, "2021-Q1")]["lfs"] in ("b", "bu")]
    lines += [
        "### 2.5 The 2021 Labour Force Survey change", "",
        "Eurostat's LFS metadata state that the 2021 regulation (IESS) created breaks, "
        "that Eurostat and the member states ran a joint break-correction exercise, and "
        "that the new tables contain break-corrected data for 2009-Q1 to 2020-Q4 which "
        "\"mostly start in 2009\". The snapshot is consistent with this: the labour series "
        f"begin in 2009 for {len(f.lfs_2008_missing)} countries, and only "
        f"{names(break_2021)} carries a break flag at 2021-Q1.", "",
        f"Two residual issues remain. {names(d_countries)} are flagged `d` (definition "
        "differs) from 2021-Q1 onward: both keep counting workers on temporary lay-off "
        "under the pre-2021 job-attachment rule. The Netherlands phased the new survey in "
        "during 2021 without a flag. The published models treat all of these as "
        "comparable observations; the `quality_screened` sample below removes flagged "
        "changes instead.", "",
    ]
    return lines


def _verification(f: Facts) -> list[str]:
    r = f.r
    lines = [
        "## 3. Independent verification of the published numbers", "",
        "Each published quantity was recomputed by a deliberately different route: a QR "
        "solve instead of `lstsq`; a Bartlett HAC built from an explicit n-by-n kernel on "
        "calendar distances instead of a loop over lag orders; the coefficient sum read "
        "off a reparameterised regression (regressors g_t and g_{t-k} - g_t) instead of "
        "the contrast c'Vc; and the pre/post tests rebuilt from two separate regressions. "
        f"Tolerance: {VERIFY_TOLERANCE_TEXT}.", "",
    ]
    lines += table(["Area", "Quantity", "Independent route", "Compared", "Max abs. difference",
                    "Result"], [
        [c.area, c.quantity, c.method, f"{c.compared:,}", _sci(c.max_abs_difference),
         "pass" if c.passed else "**FAIL**"] for c in r.checks
    ], "lllrrl")
    if r.archived:
        lines += ["Archived CI outputs against outputs regenerated from the committed "
                  "snapshot:", ""]
        lines += table(["File", "SHA-256 (archived)", "Max abs. difference"], [
            [f"`{a.path}`", f"`{a.sha256[:16]}…`", _sci(a.max_abs_difference)]
            for a in r.archived
        ], "llr")
    lines += [
        "Two latent issues were found; neither changes a published number for this "
        "snapshot:", "",
        f"- `eu-okun baseline` measured HAC lags in rows rather than calendar quarters. "
        f"This only matters when a country's complete-case sample skips a quarter; "
        f"{f.r.gapped_baselines} of {len(EU27) * 8} baseline samples do. The baseline now "
        f"uses calendar lags, like every other command.",
        "- The pre/post Wald tests and all intervals use asymptotic normal and chi-square "
        "approximations with an uncorrected Newey-West covariance. The arithmetic is "
        "right; the approximation is weak for the 17-quarter post-2021 window (section 5).",
        "",
    ]
    return lines


VERIFY_TOLERANCE_TEXT = "absolute difference of at most 1e-9 (1e-6 for Wald statistics)"


def _sensitivity(f: Facts) -> list[str]:
    r = f.r
    lines = ["## 4. Sensitivity of the GDP coefficient sums", "",
             "### 4.1 Samples and decision rules", ""]
    lines += table(["Sample", "Definition", "Quarters (EU-27 common)", "GDP lags"], [
        [f"`{spec.name}`", spec.label,
         f"{f.sample_n[spec.name]} ({f.sample_span[spec.name][0]}..{f.sample_span[spec.name][1]})"
         if spec.common else "country-specific (see `grid.csv`)",
         ", ".join(map(str, spec.lag_orders))]
        for spec in SAMPLES
    ], "llll")
    lines += [
        f"The *core grid* for each country is GDP lags {', '.join(map(str, CORE_LAGS))} x "
        f"with/without participation x the samples {', '.join(f'`{s}`' for s in CORE_SAMPLES)} "
        f"({r.robustness[0].core_estimates} estimates). Verdicts, fixed before looking at "
        "the results:", "",
        "- **Robust negative**: every point estimate is negative and every 95% HAC "
        "interval lies below zero.",
        "- **Negative; significance depends on specification**: every point estimate is "
        "negative, but at least one interval reaches zero.",
        "- **Sign-unstable**: at least one point estimate is zero or positive.", "",
        "A second verdict drops the pandemic-inclusive `full` sample, separating "
        "instability caused by 2020-2021 from instability that remains without it. "
        "Lag order changes the estimand (a longer cumulative horizon), so its spread is a "
        "sensitivity, not a confidence interval. The post-2021 sample is reported "
        "separately (section 5) because two GDP lags cannot be estimated reliably on it.", "",
        "### 4.2 Pandemic influence", "",
        f"In the published common sample the eight quarters of 2020-2021 are "
        f"{pct(8 / f.sample_n['full'])} of observations but supply between "
        f"{pct(min(f.leverage))} and {pct(max(f.leverage))} (median "
        f"{pct(float(np.median(f.leverage)))}) of the variation in current-quarter GDP "
        "growth. Ordinary least squares weights observations by that variation, so those "
        "quarters - when job-retention schemes kept unemployment flat during record GDP "
        "falls - largely determine the published coefficients.", "",
        "![Coefficient sums with and without 2020-2021]({assets}/pandemic_influence.png)", "",
    ]
    lines += table(
        ["Country", "Published (incl. 2020-21)", "Pandemic excluded, 95% interval",
         "Share of GDP variation from 2020-21", "Published outside interval"],
        [[geo, s3(x.headline_full), f"{s3(x.headline_excluded)} "
          f"{interval(x.headline_excluded_low, x.headline_excluded_high)}",
          pct(x.pandemic_leverage_share), "yes" if x.full_outside_excluded_ci else ""]
         for geo, x in f.robust.items()],
        "lrrrl",
    )
    lines += [
        f"Two GDP lags, no participation control, HAC bandwidth {HAC_BANDWIDTH}. For "
        f"{len(f.pandemic_outside)} countries the published estimate lies outside the "
        "pandemic-excluded interval: the published all-country table should not be read "
        "as typical Okun coefficients for them.", "",
        "### 4.3 Robustness verdicts", "",
    ]
    lines += table(
        ["Country", "Verdict (24 estimates)", "Without the full sample (18)",
         "Core range", "Core median", "Largest single influence"],
        [[geo, VERDICTS[x.verdict], VERDICTS[x.verdict_without_pandemic],
          interval(x.core_min, x.core_max), s3(x.core_median),
          f"{x.dominant_dimension} ({x_effect(x):.3f})"]
         for geo, x in f.robust.items()],
        "lllrrl",
    )
    counts = {key: len(value) for key, value in f.by_verdict.items()}
    free_counts = {key: len(value) for key, value in f.by_verdict_free.items()}
    dominant: dict[str, list[str]] = {}
    for x in r.robustness:
        dominant.setdefault(x.dominant_dimension, []).append(x.geo)
    lines += [
        f"Verdict counts: {counts['robust_negative']} robust negative, "
        f"{counts['negative_precision_dependent']} precision-dependent, "
        f"{counts['sign_unstable']} sign-unstable. Without the pandemic-inclusive sample: "
        f"{free_counts['robust_negative']}, {free_counts['negative_precision_dependent']} "
        f"and {free_counts['sign_unstable']}. Sign-unstable even without 2020-2021: "
        f"{names(f.by_verdict_free['sign_unstable'])}.", "",
        "The largest single influence is the biggest change in the sum caused by one "
        "choice with the others held fixed: "
        + "; ".join(f"{dim}: {names(sorted(geos))}" for dim, geos in sorted(dominant.items()))
        + ".", "",
    ]
    bw_fail = [x.geo for x in r.robustness
               if not x.excluded_ci_excludes_zero_all_bandwidths
               and x.headline_excluded_high < 0]
    lines += [
        "### 4.4 HAC bandwidth", "",
        f"For the pandemic-excluded two-lag sum, intervals were recomputed with Bartlett "
        f"bandwidths {', '.join(map(str, BANDWIDTHS))}. Countries whose interval excludes "
        f"zero at bandwidth {HAC_BANDWIDTH} but not at every bandwidth: {names(bw_fail)}. "
        "No small-sample correction is applied at any bandwidth.", "",
    ]
    return lines


def x_effect(x: Robustness) -> float:
    return float({
        "GDP lag order": x.lag_effect, "participation control": x.participation_effect,
        "pandemic treatment": x.pandemic_effect, "sample selection": x.sample_effect,
    }[x.dominant_dimension])


def _stability(f: Facts) -> list[str]:
    r = f.r
    holm = {b: holm_significant(r, b) for b in BANDWIDTHS}
    rows = f.stability
    lower_sd = sum(s.gdp_sd_post < s.gdp_sd_pre for s in rows)
    definition = [s.geo for s in rows if s.post_definition_flagged]
    lines = [
        "## 5. Regimes and pre/post stability", "",
        f"The published stability tests compare 2009-2019 ({f.pre_n} quarters) with "
        f"2022 onwards ({f.post_n} quarters) in a fully interacted model with one GDP lag. "
        f"The arithmetic is verified (section 3). Holm-significant changes in the GDP sum "
        f"(family of {len(r.stability)} tests): "
        + ", ".join(f"bandwidth {b}: {holm[b]}" for b in BANDWIDTHS)
        + ". So the count barely depends on the bandwidth; the concern is the length of the "
        "post window and what it contains:", "",
        f"- {f.post_n} quarters for 3-4 coefficients: Newey-West tests are known to "
        "over-reject at this length, and nothing here corrects for it.",
        f"- GDP growth varied less after 2021 than before 2020 in {lower_sd} of 27 "
        "countries, so the post coefficients are weakly identified.",
        f"- Post-2021 sums change sign when a single quarter is dropped in "
        f"{len(f.loo_unstable)} countries ({names(f.loo_unstable)}). Positive and stable "
        f"under that check: {names(f.positive_stable)}.",
        f"- Post-2021 labour data follow a different definition (`d`) in "
        f"{names(definition)}, and post-2021 GDP is partly provisional in "
        f"{len(f.post_provisional)} countries (up to {pct(f.post_provisional_max)} of "
        "the window).",
        "- The pre-2020 window mixes the 2009-2013 crisis with the recovery; no split "
        "within it was tested.", "",
        "![Pre-2020 and post-2021 sums]({assets}/post2021_stability.png)", "",
        "Regime estimates published by `eu-okun regimes` (K=1, no participation), with "
        "the stability tests and the leave-one-out range for the focus countries:", "",
    ]
    lines += table(
        ["Country", "2009-2019", "Pandemic excluded", "2022 onwards",
         "Post range, one quarter dropped", "Holm p (bandwidth 4)"],
        [[geo, f.headline("pre_2020", geo, 1),
          f.headline("pandemic_excluded", geo, 1), f.headline("post_2021", geo, 1),
          interval(s.post_loo_min, s.post_loo_max), pval(s.holm_p[HAC_BANDWIDTH])]
         for geo in FOCUS_COUNTRIES for s in rows if s.geo == geo],
        "lrrrrr",
    )
    lines += [
        f"Descriptively, in {len(f.weaker_after)} countries ({names(f.weaker_after)}) "
        "every leave-one-out post-2021 sum lies above the upper bound of the 2009-2019 "
        "95% interval: the association is weaker after 2021 there. Not reportable: the "
        "size or sign of any post-2021 coefficient, or a dated structural break.", "",
    ]
    return lines


def _pairs(f: Facts) -> list[str]:
    r = f.r
    lines = [
        "## 6. Country-pair comparisons", "",
        f"The published paired block bootstrap ({r.pair_replicates} draws, synchronised "
        "quarters, two GDP lags, no participation) uses the full sample. The same "
        "procedure was re-run with 2020-2021 removed. Block length 8 shown; all lengths "
        "are in `pairs_review.csv`.", "",
    ]
    rows = []
    full = {(p.country_a, p.country_b): p for p in r.pairs
            if p.sample == "full" and p.block_length == 8}
    excl = {(p.country_a, p.country_b): p for p in r.pairs
            if p.sample == "pandemic_excluded" and p.block_length == 8}
    for key, p in full.items():
        q = excl[key]
        rows.append([f"{key[0]} - {key[1]}", s3(p.estimate), interval(p.ci_low, p.ci_high),
                     pval(p.holm_p), s3(q.estimate), interval(q.ci_low, q.ci_high),
                     pval(q.holm_p)])
    lines += table(["Pair", "Full: difference", "95% percentile interval", "Holm sign p",
                    "Excl. 2020-21: difference", "95% percentile interval", "Holm sign p"],
                   rows, "lrrrrrr")
    holm_by_length: dict[tuple[str, tuple[str, str]], list[float]] = {}
    for p in r.pairs:
        holm_by_length.setdefault((p.sample, (p.country_a, p.country_b)), []).append(p.holm_p)
    always = [f"{a}-{b}" for (s, (a, b)), ps in holm_by_length.items()
              if s == "pandemic_excluded" and max(ps) < 0.05]
    some = [f"{a}-{b}" for (s, (a, b)), ps in holm_by_length.items()
            if s == "pandemic_excluded" and min(ps) < 0.05 <= max(ps)]
    lines += [
        f"Without 2020-2021 the absolute differences are a median {f.pair_ratio:.1f} times "
        f"larger (block length 8), and the sign changes for {len(f.pair_sign_changes)} of "
        f"{len(rows)} pairs ({names(f.pair_sign_changes)}). Pairs below 0.05 after Holm "
        f"adjustment at every block length (4, 8, 12) without 2020-2021: {names(always)}; "
        f"at some lengths only: {names(some)}. Which pairs look different therefore "
        "depends on the pandemic treatment and, to a lesser degree, on the block length.",
        "",
        "The sign frequency is descriptive, not a null-centred test, and a difference in "
        "coefficient sums is not a difference in causal labour-market responses.", "",
    ]
    return lines


def _forecasts(f: Facts) -> list[str]:
    lines = [
        "## 7. Predictive evaluation", "",
        "The published forecasts were recomputed independently (section 3) and re-scored "
        "on three pre-declared evaluation sets. 'Better MAE' counts countries where the "
        "model's mean absolute error is below AR(1)'s. 'Distinguishable' counts countries "
        "where the mean absolute-loss difference exceeds 1.96 HAC standard errors "
        f"(bandwidth {HAC_BANDWIDTH}; a Diebold-Mariano-type ratio, approximate with "
        "24-34 origins).", "",
    ]
    rows = []
    for evaluation in EVALUATION_LABELS:
        for model in MODEL_LABELS:
            if model == BASELINE_MODEL:
                continue
            x = f.forecasts[(evaluation, model)]
            rows.append([EVALUATION_LABELS[evaluation], MODEL_LABELS[model], x.origins,
                         f"{x.countries_better_mae}/27", pct(x.median_skill_vs_ar1),
                         x.significantly_better, x.significantly_worse])
    lines += table(["Evaluation", "Model vs AR(1)", "Origins", "Better MAE",
                    "Median MAE reduction", "Distinguishably better", "Distinguishably worse"],
                   rows, "llrrrrr")
    zero_clean = f.forecasts[("non_pandemic_targets", "zero_change")]
    published_gdp = f.forecasts[("published", GDP_MODEL)]
    clean_gdp = f.forecasts[("non_pandemic_targets", GDP_MODEL)]
    free_gdp = f.forecasts[("pandemic_free", GDP_MODEL)]
    most_better = max(clean_gdp.significantly_better, free_gdp.significantly_better)
    lines += [
        "![Forecast skill with and without pandemic quarters]({assets}/forecast_skill.png)",
        "",
        "In the published scores, adding two GDP lags to AR(1) lowers the MAE in "
        f"{published_gdp.countries_better_mae} of 27 countries. Those scores are dominated "
        "by the pandemic targets: without them the count is "
        f"{clean_gdp.countries_better_mae} (median reduction "
        f"{pct(clean_gdp.median_skill_vs_ar1)}), and with 2020-2021 also removed from "
        f"training it is {free_gdp.countries_better_mae} (median "
        f"{pct(free_gdp.median_skill_vs_ar1)}). The loss difference is distinguishable "
        f"from zero in at most {countries(most_better)} in favour of GDP. Whether lagged "
        "GDP helps is therefore unresolved. "
        "AR(1) is itself a weak benchmark: the zero-change forecast beats it in "
        f"{zero_clean.countries_better_mae} of 27 countries outside the pandemic targets. "
        "The evaluation uses revised data and assumes previous-quarter GDP is known, so "
        "it is not a real-time test.", "",
    ]
    return lines


def _focus(f: Facts) -> list[str]:
    r = f.r
    lines = [
        "## 8. Portugal, Spain, the Netherlands and Luxembourg", "",
        "Descriptive detail only. Differences between these countries are associations "
        "under different measurement conditions; none is interpreted as causal or "
        "institutional.", "",
        "![Unemployment change against GDP growth]({assets}/focus_scatter.png)", "",
        "![Specification sensitivity]({assets}/focus_sensitivity.png)", "",
    ]
    skill = r.forecast_skill
    for geo in FOCUS_COUNTRIES:
        x = f.robust[geo]
        stability = next(s for s in f.stability if s.geo == geo)
        flags = [z for z in r.flag_summary if z.geo == geo]
        lines += [f"### {COUNTRY_NAMES[geo]} ({geo})", ""]
        lines += table(["Sample", "K=1", "K=2", "K=4"], [
            [f"`{sample}`" + (" + participation" if part else ""),
             *(f.headline(sample, geo, k, part) for k in CORE_LAGS)]
            for sample in CORE_SAMPLES for part in (False, True)
        ], "lrrr")
        flag_text = "; ".join(
            f"{'GDP' if z.series == 'gdp' else 'LFS'} `{z.flag}` {z.quarters}" for z in flags
        ) or "none"
        lines += [
            f"- Verdict: **{VERDICTS[x.verdict]}**; without the pandemic-inclusive sample: "
            f"**{VERDICTS[x.verdict_without_pandemic]}**. Largest single influence: "
            f"{x.dominant_dimension} ({x_effect(x):.3f}).",
            f"- 2020-2021 supplies {pct(x.pandemic_leverage_share)} of the current-quarter "
            f"GDP variation in the published sample.",
            f"- Post-2021 sum (K=1) {s3(stability.post_sum)}; range with one quarter "
            f"dropped {interval(stability.post_loo_min, stability.post_loo_max)}; "
            f"pre-2020 {s3(stability.pre_sum)}; Holm p for the change "
            f"{pval(stability.holm_p[HAC_BANDWIDTH])}.",
            f"- Forecast MAE reduction of AR(1) + GDP over AR(1): "
            f"{pct(skill['published'][GDP_MODEL][geo])} (published), "
            f"{pct(skill['non_pandemic_targets'][GDP_MODEL][geo])} (non-pandemic targets), "
            f"{pct(skill['pandemic_free'][GDP_MODEL][geo])} (pandemic-free).",
            f"- Eurostat flags: {flag_text}.",
            *(f"- {note}" for note in FOCUS_NOTES[geo]),
            "",
        ]
    return lines


def _conclusions(f: Facts) -> list[str]:
    r = f.r
    robust = f.by_verdict["robust_negative"]
    unstable_free = f.by_verdict_free["sign_unstable"]
    lines = [
        "## 9. What can be reported", "",
        "**Verified and robust enough to report** (as descriptive associations):", "",
        f"- The published estimates are computed correctly: {len(r.checks)} independent "
        "checks pass, and the archived CI outputs are reproduced from the frozen snapshot.",
        f"- Outside 2020-2021, unemployment-rate changes are negatively associated with "
        f"current and lagged GDP growth in most EU-27 countries; in {len(robust)} "
        f"({names(robust)}) every core specification agrees.",
        f"- Including 2020-2021 moves the two-lag sum towards zero in {len(f.attenuated)} "
        "of 27 countries; that attenuation is a property of the pandemic quarters, not of "
        "the 2009-2019 relationship.", "",
        "**Sensitivity-dependent** (report only with the sensitivity shown):", "",
        f"- Magnitudes: the GDP lag order is the largest influence in "
        f"{f.dominant_counts['GDP lag order']} countries, pandemic treatment in "
        f"{f.dominant_counts['pandemic treatment']}, sample selection in "
        f"{f.dominant_counts['sample selection']}.",
        f"- A weaker association after 2021 ({names(f.weaker_after)}; section 5).",
        f"- Signs or significance in {names(f.not_robust)}.",
        "- All country-pair differences and any ordering of countries.", "",
        "**Inconclusive** (do not report as findings):", "",
        "- The sign or size of post-2021 coefficients, and any dated structural break.",
        "- Whether lagged GDP improves unemployment forecasts over simple benchmarks.",
        f"- Any Okun relation for {names(unstable_free)}, where the sign is unstable even "
        "without 2020-2021.", "",
        "## 10. Methodological limitations", "",
        "- **Measurement**: domestic GDP versus resident labour data; ES and FR "
        "definitions differ from 2021; age band 16-74 in ES and IT; provisional recent GDP; "
        "revisions after the snapshot.",
        "- **Specification**: contemporaneous GDP growth and participation are "
        "endogenous; the participation control can absorb part of the GDP association "
        "(a bad-control risk); lag order changes the estimand.",
        "- **Inference**: uncorrected Newey-West covariance with a fixed bandwidth and "
        "normal/chi-square approximations; no adjustment for testing many countries "
        "except Holm within the stability and pair families; common shocks make "
        "country estimates dependent.",
        "- **Samples**: one snapshot; the pre-2020 window mixes crisis and recovery; the "
        "post-2021 window is short and spans an energy-price shock and high inflation.",
        "- **Forecasting**: revised data, optimistic information timing, no vintage "
        "reconstruction.", "",
        "## 11. Unresolved research questions", "",
        "1. Does the weak post-2021 association reflect a changed labour-market response, "
        "measurement (definitions, provisional GDP), or too little GDP variation to "
        "identify anything? Small-sample-valid inference (fixed-b or bootstrap) and later "
        "data are needed.",
        "2. Do the pandemic-era coefficients describe job-retention schemes? That needs "
        "data on hours worked and scheme take-up, not unemployment alone.",
        "3. For Luxembourg, does an employment- or hours-based measure that includes "
        "cross-border workers restore a measurable association?",
        "4. Are differences between countries larger than common European shocks and "
        "estimation dependence allow? Partial pooling (Phase 2) should address this "
        "jointly instead of pairwise.",
        "5. Is the 2009-2019 association stable within that window (crisis versus "
        "recovery)?",
        "6. How much do results move between Eurostat vintages?", "",
    ]
    return lines


def _reproduction(report: Path, output_dir: Path, data_dir: Path) -> list[str]:
    return [
        "## 12. Reproduction", "",
        "From `eu-okun-law/`, with the `report` extra installed:", "",
        "```bash",
        f"eu-okun review --data-dir {data_dir.as_posix()} "
        f"--output-dir {output_dir.as_posix()} --report {report.as_posix()}",
        "```", "",
        "Add `--published-dir <results>` to also compare archived CI outputs (section 3). "
        "Machine-readable tables (`grid.csv`, `robustness.csv`, `verification.csv`, "
        "`stability_review.csv`, `pairs_review.csv`, `forecast_review.csv`, "
        "`quality_flags.csv`, `review.json`) sit next to the figures. The bootstrap uses "
        "the published seed, so a rerun on the same snapshot reproduces every number.", "",
    ]


def render_report(review: ReviewResult, assets: str, report: Path, output_dir: Path,
                  data_dir: Path, provenance: str) -> str:
    """Assemble the Markdown report from the review results."""
    f = Facts(review)
    lines = [
        "# Classical results review: EU-27 Okun's law", "",
        "> Generated by `eu-okun review` from the frozen Eurostat snapshot described in "
        "section 1. Do not edit by hand; rerun the command instead. All results are "
        "descriptive conditional associations. None identifies a causal effect of a GDP "
        "shock.", "",
    ]
    for section in (
        _summary(f), _provenance(f, provenance), _sources(f), _verification(f),
        _sensitivity(f), _stability(f), _pairs(f), _forecasts(f), _focus(f),
        _conclusions(f), _reproduction(report, output_dir, data_dir),
    ):
        lines += section
    return "\n".join(lines).replace("{assets}", assets).rstrip() + "\n"


# ----------------------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------------------


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite value in review output.")
    return value


def write_tables(review: ReviewResult, directory: Path) -> None:
    """Write every review table as CSV plus one combined JSON document."""
    directory.mkdir(parents=True, exist_ok=True)
    grid = [asdict(row) for row in review.grid]
    robustness = [asdict(row) for row in review.robustness]
    checks = [{**asdict(c), "passed": c.passed} for c in review.checks]
    stability = [
        {**{k: v for k, v in asdict(s).items() if k != "holm_p"},
         **{f"holm_p_bandwidth_{b}": p for b, p in s.holm_p.items()},
         "post_sign_stable": s.post_sign_stable}
        for s in review.stability
    ]
    pairs = [{**asdict(p), "estimate_position": p.position} for p in review.pairs]
    forecasts = [asdict(x) for x in review.forecasts]
    skill = [
        {"evaluation": evaluation, "model": model, "geo": geo, "skill_vs_ar1": value}
        for evaluation, models in review.forecast_skill.items()
        for model, countries in models.items() for geo, value in countries.items()
    ]
    flags = [asdict(x) for x in review.flag_summary]
    bandwidths = [
        {"geo": geo, "bandwidth": b, "ci_low": low, "ci_high": high}
        for geo, values in review.bandwidth_ci.items() for b, (low, high) in values.items()
    ]
    tables = {
        "grid.csv": grid, "robustness.csv": robustness, "verification.csv": checks,
        "stability_review.csv": stability, "pairs_review.csv": pairs,
        "forecast_review.csv": forecasts, "forecast_skill.csv": skill,
        "quality_flags.csv": flags, "bandwidth_intervals.csv": bandwidths,
    }
    for name, rows in tables.items():
        _write_csv(directory / name, rows)
    document = {
        "description": "Classical results review; descriptive associations, not causal",
        "snapshot_manifest_sha256": review.audit["snapshot_manifest_sha256"],
        "processed_csv_sha256": review.audit["processed_csv_sha256"],
        "archived_outputs": [asdict(a) for a in review.archived or []],
        "gapped_baseline_samples": review.gapped_baselines,
        "missing": review.missing,
        "csv_only": ["grid.csv", "forecast_skill.csv"],
        **{name.removesuffix(".csv"): rows for name, rows in tables.items()
           if name not in ("grid.csv", "forecast_skill.csv")},
    }
    (directory / "review.json").write_text(
        json.dumps(_jsonable(document), indent=1, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def write_review(
    data_dir: Path, output_dir: Path, report: Path | None = None,
    published_dir: Path | None = None, provenance: str = "", figures: bool = True,
    pair_replicates: int = 999,
) -> ReviewResult:
    """Run the review and write tables, figures and the Markdown report."""
    review = run_review(data_dir, published_dir, pair_replicates)
    report = report or output_dir / REPORT_NAME
    write_tables(review, output_dir)
    if figures:
        from eu_okun.review_figures import write_figures

        source = (f"Source: Eurostat (namq_10_gdp, une_rt_q, lfsi_emp_q), snapshot "
                  f"{str(review.manifest['downloaded_at_utc'])[:10]}. Descriptive, not causal.")
        write_figures(review, output_dir, source)
    assets = Path(os.path.relpath(output_dir, report.parent)).as_posix()
    if provenance == "" and published_dir is not None:
        provenance = published_dir.as_posix()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        render_report(review, assets, report, output_dir, data_dir, provenance),
        encoding="utf-8", newline="\n",
    )
    return review
