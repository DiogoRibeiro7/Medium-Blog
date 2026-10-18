"""Tables, figures and the Markdown report of the monthly identification study.

Runs the pre-registered estimation of ``MONTHLY_IDENTIFICATION_PROTOCOL.md`` on the frozen
monthly Eurostat snapshot and the frozen external-shock snapshot, and writes
``MONTHLY_IDENTIFICATION_RESULTS.md``. Every number and every verdict in the report is
computed here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.external_shocks import audit_shock_snapshot, monthly_series
from eu_okun.monthly_data import (
    GROUPS,
    MANIFEST_FILE,
    UNEMPLOYMENT_METHOD,
    audit_monthly_snapshot,
    flag_counts,
    load_monthly_panel,
)
from eu_okun.monthly_structural import (
    FIRST_ORIGIN,
    LONG_FIRST_ORIGIN,
    OUTCOME_LABELS,
    OUTCOMES,
    PRIMARY,
    RATIO_HORIZONS,
    SHOCKS,
    VARIANTS,
    CountryResponse,
    MonthlyArrays,
    MonthlyResults,
    Placebo,
    country_responses,
    estimate_shock,
    in_euro_area,
    monthly_arrays,
    placebo,
    predictability,
    sample_countries,
)
from eu_okun.structural import OkunSet, Predictability, ShockSpec

ALPHA = 0.05
WEAK_T = 3.16
TABLE_HORIZONS = (0, 3, 6, 9, 12, 18, 24)
QUARTERLY_MATCH = {12: 4, 24: 8}         # monthly ratio horizon -> quarterly Okun horizon


@dataclass
class MonthlyReview:
    """Everything the report is generated from."""

    primary: dict[str, MonthlyResults]
    robustness: dict[str, MonthlyResults]
    ip_all: dict[str, MonthlyResults]
    predictability: dict[str, Predictability]
    placebo: dict[str, list[Placebo]]          # "MP: primary", "MP: all", ...
    countries: dict[str, list[CountryResponse]]
    quarterly: dict[str, dict[str, str]] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def doubtful(self, key: str) -> bool:
        """Protocol, section 6: a predictable shock makes identification doubtful."""
        return self.predictability[key].p_value < ALPHA

    def timing_doubtful(self, key: str) -> bool:
        """Protocol, section 6: a placebo rejection in the primary sample."""
        tests = self.placebo.get(f"{key}: primary", [])
        return any(p.p_value < ALPHA for p in tests)

    def label(self, key: str, spec: ShockSpec) -> str:
        if not spec.identified:
            return "contrast only (not an identified shock)"
        problems = []
        if self.doubtful(spec.key):
            problems.append("identification doubtful (shock predictable)")
        if self.timing_doubtful(spec.key):
            problems.append("timing doubtful (placebo rejects)")
        if key.endswith("all members as published"):
            problems.append("A7 doubtful by construction outside group D")
        return "; ".join(problems) if problems else "identified under A1-A5, A7"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quarterly_rows(directory: Path | None) -> dict[str, dict[str, str]]:
    """The committed quarterly Okun sets and unemployment responses (descriptive context)."""
    if directory is None or not (directory / "okun_sets.csv").exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with (directory / "okun_sets.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["variant"] in SHOCKS:
                out[f"{row['variant']}:okun:{row['h']}"] = row
    with (directory / "responses.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["variant"] in SHOCKS and row["outcome"] in ("gdp", "unemployment"):
                out[f"{row['variant']}:{row['outcome']}:{row['h']}"] = row
    return out


def build_review(monthly_dir: Path, shocks_dir: Path,
                 quarterly_dir: Path | None = None) -> MonthlyReview:
    """Audit both snapshots and run every pre-registered estimate."""
    manifest = audit_monthly_snapshot(monthly_dir)
    shock_manifest = audit_shock_snapshot(shocks_dir)
    panel = load_monthly_panel(monthly_dir)
    arrays = monthly_arrays(panel, manifest["period"]["start"], manifest["period"]["end"])
    shocks = monthly_series(shocks_dir)
    return review_from_arrays(arrays, shocks, quarterly_dir, monthly_dir, manifest,
                              shock_manifest)


def review_from_arrays(arrays: MonthlyArrays, shocks: Mapping[str, Mapping[str, float]],
                       quarterly_dir: Path | None = None, monthly_dir: Path | None = None,
                       manifest: Mapping[str, Any] | None = None,
                       shock_manifest: Mapping[str, Any] | None = None) -> MonthlyReview:
    """Every estimate of the protocol on prepared arrays."""
    primary = {key: estimate_shock(arrays, shocks, spec) for key, spec in SHOCKS.items()}
    robustness: dict[str, MonthlyResults] = {}
    for key, spec in SHOCKS.items():
        robustness[f"{key}: all members as published"] = estimate_shock(
            arrays, shocks, spec, replace(PRIMARY, sample="all"))
    for key, spec in VARIANTS.items():
        robustness[key] = estimate_shock(arrays, shocks, spec)
    for key, spec in SHOCKS.items():
        robustness[f"{key}: pandemic included"] = estimate_shock(
            arrays, shocks, spec, replace(PRIMARY, exclude_pandemic=False))
        robustness[f"{key}: six lags"] = estimate_shock(arrays, shocks, spec,
                                                        replace(PRIMARY, lags=6))
        robustness[f"{key}: Driscoll-Kraay"] = estimate_shock(
            arrays, shocks, spec, replace(PRIMARY, driscoll_kraay=True))
        robustness[f"{key}: long sample"] = estimate_shock(
            arrays, shocks, spec, replace(PRIMARY, first_origin=LONG_FIRST_ORIGIN))
    ip_all = {key: estimate_shock(arrays, shocks, spec,
                                  replace(PRIMARY, sample="all", du_controls=False),
                                  outcomes=("ip",))
              for key, spec in SHOCKS.items()}
    groupd = sample_countries(arrays, PRIMARY)
    tests = {key: predictability(arrays, shocks[spec.series], spec, groupd)
             for key, spec in {**SHOCKS, **VARIANTS}.items()}
    placebos = {}
    for key, spec in SHOCKS.items():
        for sample in ("primary", "all"):
            placebos[f"{key}: {sample}"] = placebo(arrays, shocks[spec.series], spec,
                                                   replace(PRIMARY, sample=sample),
                                                   primary[key].scale)
    countries = {key: country_responses(arrays, shocks[spec.series], spec, primary[key].scale,
                                        PRIMARY)
                 for key, spec in SHOCKS.items()}
    data: dict[str, Any] = {"months": [arrays.months[0], arrays.months[-1]]}
    provenance: dict[str, Any] = {}
    if monthly_dir is not None and manifest is not None:
        names = [arrays.countries[i] for i in groupd]
        data["flags_primary"] = {k: dict(v) for k, v in
                                 flag_counts(monthly_dir, FIRST_ORIGIN, names).items()}
        data["flags_primary_long"] = {k: dict(v) for k, v in
                                      flag_counts(monthly_dir, "2000-01", names).items()}
        data["flags_all"] = {k: dict(v) for k, v in
                             flag_counts(monthly_dir, FIRST_ORIGIN, arrays.countries).items()}
        provenance["eurostat_monthly_manifest_sha256"] = _sha256(monthly_dir / MANIFEST_FILE)
        provenance["eurostat_monthly_processed_sha256"] = manifest["processed_csv_sha256"]
        provenance["eurostat_monthly_raw"] = {
            name: {"dataset": entry["dataset"], "sha256": entry["sha256"],
                   "updated": entry["updated"], "url": entry["url"]}
            for name, entry in manifest["series"].items()}
        provenance["unemployment_method"] = {"source": manifest["unemployment_method"]["source"],
                                             "read": manifest["unemployment_method"]["read"]}
    if shock_manifest is not None:
        provenance["shock_files"] = {f["file"]: {"sha256": f["sha256"], "commit": f["commit"],
                                                 "licence": f["licence"]}
                                     for f in shock_manifest["files"]}
    if quarterly_dir is not None and (quarterly_dir / "okun_sets.csv").exists():
        provenance["quarterly_okun_sets_sha256"] = _sha256(quarterly_dir / "okun_sets.csv")
    provenance["sample_months"] = [arrays.months[0], arrays.months[-1]]
    return MonthlyReview(primary, robustness, ip_all, tests, placebos, countries,
                         _quarterly_rows(quarterly_dir), data, provenance)


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]], align: str = "") -> list[str]:
    align = align or "l" + "r" * (len(header) - 1)
    marks = ["---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(marks) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _band(estimate: float, low: float, high: float) -> str:
    return f"{estimate:+.3f} [{low:+.3f}, {high:+.3f}]"


def _horizons(values: Sequence[int]) -> str:
    return ", ".join(str(h) for h in values) if values else "none"


def distinguishable(results: MonthlyResults, outcome: str) -> list[int]:
    """Horizons at which the 90% band excludes zero."""
    return [r.h for r in results.responses if r.outcome == outcome
            and (r.low90 > 0 or r.high90 < 0)]


def _strength(t: float) -> str:
    return "weak" if abs(t) < WEAK_T else "not weak"


def _ratio(results: MonthlyResults, h: int) -> OkunSet:
    return next(o for o in results.ratios if o.h == h)


def compare(review: MonthlyReview) -> str:
    """Protocol, section 6: demand and supply are compared only if both sets are bounded."""
    statements = []
    for h in RATIO_HORIZONS:
        a, b = _ratio(review.primary["MP"], h), _ratio(review.primary["OIL"], h)
        if a.informative and b.informative:
            overlap = a.lower <= b.upper and b.lower <= a.upper
            statements.append(f"at h = {h} the demand and supply sets "
                              f"{'overlap' if overlap else 'do not overlap'}")
        else:
            missing = [s for s, o in (("monetary policy", a), ("oil supply", b))
                       if not o.informative]
            statements.append(f"at h = {h} no comparison is possible "
                              f"({' and '.join(missing)} uninformative)")
    return "**Demand against supply.** Under the protocol's rule, " + "; ".join(statements) + "."


def _countries_text(arrays_countries: Sequence[str], euro_only: bool) -> str:
    names = [g for g in arrays_countries if UNEMPLOYMENT_METHOD[g] == "D"]
    if euro_only:
        names = [g for g in names if in_euro_area(g, FIRST_ORIGIN)]
    return ", ".join(names)


def _flags(counts: Mapping[str, Mapping[str, int]]) -> str:
    parts = []
    for series, label in (("unemployment", "unemployment"), ("ip", "production")):
        flags = counts.get(series, {})
        text = ", ".join(f"{n} '{k}'" for k, n in sorted(flags.items())) or "none"
        parts.append(f"{label}: {text}")
    return "; ".join(parts)


def render(review: MonthlyReview, assets: str, command: str) -> str:
    """Assemble the Markdown report."""
    lines = ["# Identification with monthly outcomes: responses to the identified common "
             "shocks", "",
             "Generated by `eu-okun monthly-identification` from the frozen monthly Eurostat "
             "snapshot and the frozen external-shock snapshot. What is estimated and how it is "
             "judged was pre-registered in "
             "[MONTHLY_IDENTIFICATION_PROTOCOL.md](../MONTHLY_IDENTIFICATION_PROTOCOL.md), an "
             "amendment to the quarterly study "
             "([IDENTIFICATION_RESULTS.md](IDENTIFICATION_RESULTS.md)).", "",
             "**Scope.** The two common shocks of the quarterly study, at their native monthly "
             "frequency: ECB monetary policy surprises (demand; euro-area members) and OPEC oil "
             "supply surprises (supply, not productivity; EU countries). Outcomes are the "
             "monthly unemployment rate and industrial production (NACE B-D). The primary "
             "sample keeps only countries whose monthly unemployment figure is a direct "
             "monthly survey estimate (group D), because trends, moving averages and "
             "interpolation put later months into a month's value (assumption A7). The ratio "
             "`kappa_IP` is pp of unemployment per 1% of industrial production: it is **not** "
             "the GDP-based Okun coefficient. Responses are causal only under A1-A5 and A7.", ""]

    # Summary
    lines += ["## Summary", ""]
    for key, res in review.primary.items():
        spec = res.spec
        lines += [f"**{spec.label}** ({spec.unit}; {review.label(key, spec)}):", ""]
        for outcome in OUTCOMES:
            lines.append(f"- {OUTCOME_LABELS[outcome]}: 90% band excludes zero at months "
                         f"{_horizons(distinguishable(res, outcome))} (of 0-24).")
        for o in res.ratios:
            status = "informative" if o.informative else "uninformative"
            weak = " (weak first stage)" if abs(o.first_stage_t) < WEAK_T else ""
            lines.append(f"- `kappa_IP` at h = {o.h}: estimate {o.estimate:+.3f}, 90% "
                         f"Anderson-Rubin set {o.text()} ({status}); first-stage t "
                         f"{o.first_stage_t:+.2f}{weak}.")
        t = review.predictability[key]
        lines.append(f"- Predictability from lagged EU data (A3): F = {t.f_statistic:.2f}, "
                     f"p = {_p(t.p_value)}.")
        tests = review.placebo[f"{key}: primary"]
        lines.append("- Placebo, pre-shock change (A7): " + "; ".join(
            f"{OUTCOME_LABELS[p.outcome].split(' (')[0].lower()} t = {p.t:+.2f}, "
            f"p = {_p(p.p_value)}" for p in tests) + ".")
        hits = sum(len(distinguishable(res, o)) for o in OUTCOMES)
        lines.append(f"- Multiplicity: {hits} of {len(res.responses)} pointwise 90% bands "
                     f"exclude zero; about {0.1 * len(res.responses):.0f} would by chance if "
                     "the shock had no effect (bands are correlated across horizons).")
        lines.append("")
    comparison = compare(review)
    lines += [comparison, ""]
    lines += _relevance_summary(review)

    # 1. Data and design
    lines += ["## 1. Data and design", ""]
    months = review.data.get("months", ["", ""])
    lines += [f"Monthly Eurostat snapshot: {months[0]} to {months[-1]}. Lag-augmented local "
              "projections pooled over countries with country fixed effects; 12 lags of the "
              "shock, of the monthly unemployment change and of industrial-production growth; "
              "standard errors clustered by month; horizons 0-24 months; origins from "
              f"{FIRST_ORIGIN}. Observations whose window [t-12, t+h] touches "
              "2020-01..2021-12 are excluded.", ""]
    if "flags_primary" in review.data:
        lines += [f"Eurostat flags (stored, not used in estimation), group D countries from "
                  f"{FIRST_ORIGIN}: {_flags(review.data['flags_primary'])}. All countries "
                  f"from {FIRST_ORIGIN}: {_flags(review.data['flags_all'])}. Group D from "
                  f"2000-01 (long sample): {_flags(review.data['flags_primary_long'])}. "
                  "'b' marks a break in series, 'd' a different definition, 'p' a provisional "
                  "value.", ""]
    countries_all = sorted(UNEMPLOYMENT_METHOD)
    rows: list[list[Any]] = []
    for group, text in GROUPS.items():
        members = [g for g in countries_all if UNEMPLOYMENT_METHOD[g] == group]
        rows.append([group, text, ", ".join(members),
                     "primary" if group == "D" else "sensitivity only"])
    lines += ["Monthly unemployment by production method (Eurostat reference metadata for "
              "`une_rt_m`, fixed in the protocol before estimation):", ""]
    lines += _table(["Group", "Method", "Countries", "Use"], rows, "llll")
    rows = []
    for res in review.primary.values():
        first, last = res.response("ip", 0), res.response("ip", 24)
        rows.append([res.spec.label, res.spec.series, res.spec.unit,
                     _countries_text(countries_all, res.spec.euro_area_only),
                     f"{first.n} / {last.n}", f"{first.clusters} / {last.clusters}"])
    lines += _table(["Shock", "Series", "Normalisation", "Countries (group D)",
                     "Observations (h = 0 / 24)", "Months (h = 0 / 24)"], rows, "llllrr")

    # 2. Diagnostics
    lines += ["## 2. Diagnostics", "",
              "Predictability (A3): the monthly shock regressed on 12 lags of the cross-country "
              "means of the unemployment change and of industrial-production growth over the "
              "shock's group D countries; classical F-test of the 24 lag coefficients. "
              "p < 0.05 counts against A3.", ""]
    rows = [[key, str(t.n), f"{t.f_statistic:.2f}", _p(t.p_value), f"{t.autocorrelation:+.2f}"]
            for key, t in review.predictability.items()]
    lines += _table(["Series", "Months", "F", "p", "Lag-1 autocorrelation"], rows)
    lines += ["Placebo (A7, A3): the change from t-2 to t-1 regressed on the shock at t, every "
              "control shifted one month back; normalised as the shock, month-clustered. Under "
              "A3 and A7 the coefficient is zero. Only the primary sample decides the label; "
              "the all-members sample shows what smoothing and interpolation do.", ""]
    rows = []
    for name, tests in review.placebo.items():
        for p in tests:
            rows.append([name.split(":")[0], p.sample, OUTCOME_LABELS[p.outcome],
                         f"{p.estimate:+.4f}", f"{p.t:+.2f}", _p(p.p_value), str(p.n),
                         "rejects" if p.p_value < ALPHA else "does not reject"])
    lines += _table(["Shock", "Sample", "Outcome", "Pre-shock response", "t", "p",
                     "Observations", "At 5%"], rows, "lllrrrrl")
    lines += _relevance_table(review)

    # 3. Responses
    lines += ["## 3. Responses", "",
              "Pointwise 68% and 90% bands (figures) and 90% bands (tables). Industrial "
              "production is the cumulative log change from t-1 (%); unemployment is the change "
              "from t-1 (pp).", ""]
    for key, res in review.primary.items():
        rows = []
        for h in TABLE_HORIZONS:
            cells = [str(h)]
            for outcome in OUTCOMES:
                r = res.response(outcome, h)
                cells.append(_band(r.estimate, r.low90, r.high90))
            rows.append(cells)
        lines += [f"**{res.spec.label}** ({res.spec.unit}; {review.label(key, res.spec)}):",
                  ""]
        lines += _table(["Months", *(OUTCOME_LABELS[o] + " [90%]" for o in OUTCOMES)], rows)
        lines += [f"![Responses to the {res.spec.label}]({assets}/responses_{key}.png)", ""]
    lines += ["Secondary: industrial production over all members (euro-area members for "
              "monetary policy, EU-27 for oil), controls the lags of the shock and of "
              "production growth only. Industrial production has no timing problem, so this "
              "uses every country.", ""]
    rows = []
    for res in review.ip_all.values():
        cells = [res.spec.label]
        for h in (0, 6, 12, 24):
            r = res.response("ip", h)
            cells.append(_band(r.estimate, r.low90, r.high90))
        rows.append(cells)
    lines += _table(["Shock", "h = 0", "h = 6", "h = 12", "h = 24"], rows)

    # 4. Ratio
    lines += ["## 4. Shock-conditional ratio `kappa_IP`", "",
              "`kappa_IP(h)` = response of the unemployment change / response of cumulative "
              "industrial production (pp per 1%), estimated by LP-IV with the shock as "
              "instrument. The Anderson-Rubin (Fieller) set is valid however weak the "
              "instrument; an unbounded set means the data do not determine the ratio. "
              "`kappa_GDP = kappa_IP * (IP response / GDP response)`; that factor is not "
              "identified here, so no GDP-based Okun coefficient follows.", ""]
    rows = []
    for res in review.primary.values():
        for o in res.ratios:
            rows.append([res.spec.label, str(o.h), f"{o.estimate:+.3f}", o.text(),
                         f"{o.first_stage_t:+.2f}",
                         "informative" if o.informative else "uninformative"])
    lines += _table(["Shock", "Months", "Estimate", "90% AR set", "First-stage t", "Verdict"],
                    rows, "lrrlrl")
    lines += [comparison, ""]

    # 5. Robustness
    lines += ["## 5. Robustness (pre-registered)", ""]
    rows = []
    for name, res in review.robustness.items():
        cells = [name, review.label(name, res.spec)]
        for outcome in OUTCOMES:
            for h in RATIO_HORIZONS:
                r = res.response(outcome, h)
                cells.append(_band(r.estimate, r.low90, r.high90))
        for h in RATIO_HORIZONS:
            o = _ratio(res, h)
            cells.append(f"{o.text()} (t {o.first_stage_t:+.2f})")
        rows.append(cells)
    lines += _table(["Variant", "Status", "IP h=12 [90%]", "IP h=24 [90%]",
                     "Unemployment h=12 [90%]", "Unemployment h=24 [90%]",
                     "Ratio h=12 AR set (first-stage t)", "Ratio h=24 AR set (first-stage t)"],
                    rows, "llrrrrll")
    bounded = [name for name, res in review.robustness.items()
               if any(o.informative for o in res.ratios)]
    lines += [("Variants with a bounded ratio set at h = 12 or 24: " + ", ".join(bounded) + "."
               if bounded else "No robustness variant gives a bounded ratio set."), ""]

    # 6. Countries
    lines += ["## 6. Countries (descriptive)", "",
              "Country-by-country local projections at h = 12 (Newey-West, 13 lags), "
              "alphabetical. Unemployment estimates outside group D rest on smoothed or "
              "interpolated series (A7 doubtful by construction). Individual country series are "
              "short; no ranking and no per-country significance claims.", ""]
    for key, items in review.countries.items():
        by_geo: dict[str, dict[str, CountryResponse]] = {}
        for cr in items:
            by_geo.setdefault(cr.geo, {})[cr.outcome] = cr
        rows = []
        for geo in sorted(by_geo):
            ip, un = by_geo[geo]["ip"], by_geo[geo]["unemployment"]
            rows.append([geo, ip.group, str(ip.n), _band(ip.estimate, ip.low90, ip.high90),
                         _band(un.estimate, un.low90, un.high90)])
        lines += [f"**{review.primary[key].spec.label}**:", ""]
        lines += _table(["Country", "Group", "Observations", "IP h=12 [90%]",
                         "Unemployment h=12 [90%]"], rows, "llrrr")
    lines += [f"![Country responses at h = 12]({assets}/countries.png)", ""]

    # 7. Quarterly comparison
    lines += _quarterly_section(review)

    # 8. Conclusion
    informative = [(res.spec.label, o.h) for res in review.primary.values() for o in res.ratios
                   if o.informative]
    lines += ["## 8. What these results do and do not establish", ""]
    if informative:
        lines.append("- Bounded ratio sets in the primary specification: " + ", ".join(
            f"{label} at h = {h}" for label, h in informative) + ". Under A1-A5 and A7 they "
            "bound how much unemployment moves per 1% of industrial production moved by that "
            "shock.")
    else:
        lines.append("- No primary ratio set is bounded: with monthly outcomes too, the data do "
                     "not determine how much unemployment moves per unit of industrial "
                     "production moved by either shock.")
    lines += [
        "- Responses are causal only under A1-A5 and A7, and only where bands exclude zero; "
        "labels above record where a diagnostic counts against an assumption.",
        "- `kappa_IP` is not the GDP-based Okun coefficient; nothing here identifies the GDP "
        "response, productivity shocks, labour-supply shocks or country-specific GDP shocks.",
        "- The reduced-form Okun coefficients of the rest of this study remain weighted "
        "averages of shock-conditional coefficients with unknown weights "
        "(IDENTIFICATION_STUDY.md, section 1).", "",
        "## 9. Provenance and reproduction", ""]
    prov = review.provenance
    if "eurostat_monthly_manifest_sha256" in prov:
        raw = prov["eurostat_monthly_raw"]
        lines += [f"Monthly Eurostat snapshot manifest SHA-256 "
                  f"`{prov['eurostat_monthly_manifest_sha256']}`; processed CSV "
                  f"`{prov['eurostat_monthly_processed_sha256']}`; "
                  + "; ".join(f"`{e['dataset']}` updated {e['updated']}" for e in raw.values())
                  + ". Shock files, commits and the quarterly comparison file are hashed in "
                  "`provenance.json`.", ""]
    lines += ["```bash", "poetry install --extras \"bayes report structural\"", command, "```",
              ""]
    return "\n".join(lines)


def _relevance_rows(review: MonthlyReview) -> list[list[str]]:
    rows = []
    for key, res in review.primary.items():
        for o in res.ratios:
            ip_all = review.ip_all[key].response("ip", o.h)
            quarterly = review.quarterly.get(f"{key}:okun:{QUARTERLY_MATCH[o.h]}")
            q_text = (f"{float(quarterly['first_stage_t']):+.2f} (h = {QUARTERLY_MATCH[o.h]})"
                      if quarterly else "n/a")
            rows.append([res.spec.label, str(o.h), f"{o.first_stage_t:+.2f}",
                         _strength(o.first_stage_t), f"{ip_all.estimate / ip_all.se:+.2f}",
                         q_text])
    return rows


def _relevance_table(review: MonthlyReview) -> list[str]:
    lines = ["Relevance (A4): t-statistic of the cumulative industrial-production response, the "
             f"first stage of the ratio; |t| < {WEAK_T} is weak. For context, the secondary "
             "all-members production response and the quarterly GDP first stage of the "
             "identification study.", ""]
    return lines + _table(["Shock", "Months", "First-stage t (group D)", "Strength",
                           "IP t, all members", "Quarterly GDP first-stage t"],
                          _relevance_rows(review), "lrrlrr")


def _relevance_summary(review: MonthlyReview) -> list[str]:
    rows = _relevance_rows(review)
    text = "; ".join(f"{r[0]} at {r[1]} months t = {r[2]} (quarterly GDP {r[5]})" for r in rows)
    return [f"**Relevance.** First-stage t of cumulative industrial production: {text}.", ""]


def _quarterly_section(review: MonthlyReview) -> list[str]:
    lines = ["## 7. Comparison with the quarterly study (descriptive)", ""]
    if not review.quarterly:
        return lines + ["The committed quarterly results were not available.", ""]
    lines += ["Different outcomes (industrial production against GDP) and horizons (months "
              "against quarters): the comparison is about informativeness, not equal numbers.",
              ""]
    rows = []
    for key, res in review.primary.items():
        for h in RATIO_HORIZONS:
            qh = QUARTERLY_MATCH[h]
            okun = review.quarterly[f"{key}:okun:{qh}"]
            q_u = review.quarterly[f"{key}:unemployment:{qh}"]
            r = res.response("unemployment", h)
            o = _ratio(res, h)
            rows.append([res.spec.label, f"{qh} / {h}",
                         _band(float(q_u["estimate"]), float(q_u["low90"]), float(q_u["high90"])),
                         _band(r.estimate, r.low90, r.high90),
                         f"{float(okun['first_stage_t']):+.2f} / {o.first_stage_t:+.2f}",
                         f"{okun['set']} / {o.text()}"])
    lines += _table(["Shock", "Horizon (quarters / months)",
                     "Unemployment, quarterly [90%]", "Unemployment, monthly [90%]",
                     "First-stage t (GDP / IP)", "AR set (Okun GDP / kappa_IP)"], rows, "llrrrl")
    return lines


# ----------------------------------------------------------------------------------
# Figures and tables
# ----------------------------------------------------------------------------------


def write_figures(review: MonthlyReview, directory: Path, source: str) -> None:
    """Responses with bands per shock, and country responses at h = 12."""
    from eu_okun.review_figures import AQUA, BLUE, INK2, MUTED, SURFACE, _finish, _pyplot

    plt = _pyplot()
    for key, res in review.primary.items():
        fig, axes = plt.subplots(1, len(OUTCOMES), figsize=(10.0, 3.8))
        for ax, outcome in zip(axes, OUTCOMES, strict=True):
            rows = [r for r in res.responses if r.outcome == outcome]
            h = np.asarray([r.h for r in rows])
            ax.fill_between(h, [r.low90 for r in rows], [r.high90 for r in rows], color=BLUE,
                            alpha=0.15, lw=0)
            ax.fill_between(h, [r.low68 for r in rows], [r.high68 for r in rows], color=BLUE,
                            alpha=0.3, lw=0)
            ax.plot(h, [r.estimate for r in rows], color=BLUE, lw=2)
            ax.axhline(0, color=MUTED, lw=1)
            ax.set_title(OUTCOME_LABELS[outcome], fontsize=9.5)
            ax.set_xlabel("Months after the shock")
            ax.set_xticks(range(0, 25, 6))
        fig.suptitle(f"{res.spec.label}: {res.spec.unit}; group D countries; 68% and 90% bands",
                     x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK2)
        fig.subplots_adjust(wspace=0.25, top=0.82, bottom=0.2)
        _finish(plt, fig, directory / f"responses_{key}.png", source, credit_y=-0.06)

    keys = list(review.countries)
    fig, axes = plt.subplots(1, len(keys), figsize=(11.0, 8.6))
    for ax, key in zip(axes, keys, strict=True):
        items = sorted((cr for cr in review.countries[key] if cr.outcome == "unemployment"),
                       key=lambda cr: cr.geo)
        ys = np.arange(len(items))[::-1]
        for y, cr in zip(ys, items, strict=True):
            colour = BLUE if cr.group == "D" else MUTED
            ax.plot([cr.low90, cr.high90], [y, y], color=colour, lw=1.6, alpha=0.7,
                    solid_capstyle="round")
            ax.plot([cr.estimate], [y], "o", ms=5.5, color=colour, mec=SURFACE, mew=1)
        pooled = review.primary[key].response("unemployment", 12)
        ax.axvline(pooled.estimate, color=AQUA, lw=1.5, ls=(0, (3, 2)))
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{cr.geo} ({cr.group})" for cr in items], fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"{review.primary[key].spec.label}\nunemployment, h = 12 months (pp)",
                     fontsize=9.5)
    fig.supxlabel("Country estimate with 90% interval; blue: group D (direct monthly survey), "
                  "grey: other groups; dashed: pooled group D estimate", fontsize=9, color=INK2)
    fig.subplots_adjust(wspace=0.4, bottom=0.08)
    _finish(plt, fig, directory / "countries.png", source, credit_y=-0.02)


def write_tables(review: MonthlyReview, directory: Path) -> None:
    """Responses, ratio sets, diagnostics, country responses and provenance."""
    directory.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        with (directory / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    every = {**review.primary, **review.robustness}
    write("responses.csv", [{"variant": name, **asdict(r)}
                            for name, res in every.items() for r in res.responses])
    write("ratio_sets.csv", [{"variant": name, **asdict(o), "set": o.text()}
                             for name, res in every.items() for o in res.ratios])
    write("ip_all_members.csv", [{"variant": name, **asdict(r)}
                                 for name, res in review.ip_all.items() for r in res.responses])
    write("predictability.csv", [asdict(t) for t in review.predictability.values()])
    write("placebo.csv", [asdict(p) for tests in review.placebo.values() for p in tests])
    write("country_responses.csv", [asdict(r) for items in review.countries.values()
                                    for r in items])
    (directory / "provenance.json").write_text(json.dumps(review.provenance, indent=1) + "\n",
                                               encoding="utf-8", newline="\n")


def write_report(monthly_dir: Path, shocks_dir: Path, output_dir: Path,
                 report: Path | None = None, quarterly_dir: Path | None = None,
                 figures: bool = True) -> MonthlyReview:
    """Build the review and write tables, figures and the Markdown report."""
    review = build_review(monthly_dir, shocks_dir, quarterly_dir)
    report = report or output_dir / "MONTHLY_IDENTIFICATION_RESULTS.md"
    write_tables(review, output_dir)
    if figures:
        source = ("Source: Eurostat une_rt_m, sts_inpr_m; Jarocinski-Karadi (2020) and Kaenzig "
                  "(2021) shocks, CC BY 4.0. Causal only under MONTHLY_IDENTIFICATION_PROTOCOL.md "
                  "A1-A5, A7.")
        write_figures(review, output_dir, source)
    assets = Path(os.path.relpath(output_dir, report.parent)).as_posix()
    command = (f"poetry run eu-okun monthly-identification --data-dir {monthly_dir.as_posix()} "
               f"--shocks-dir {shocks_dir.as_posix()}"
               + (f" --quarterly-dir {quarterly_dir.as_posix()}" if quarterly_dir else "")
               + f" --output-dir {output_dir.as_posix()} --report {report.as_posix()}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(review, assets, command), encoding="utf-8", newline="\n")
    return review
