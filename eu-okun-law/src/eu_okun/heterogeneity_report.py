"""Comparative case-study report: Portugal, Spain, the Netherlands and Luxembourg.

Runs every computation of ``HETEROGENEITY_PROTOCOL.md`` on the frozen snapshots and the
committed classical and Bayesian estimates, and writes ``HETEROGENEITY_CASE_STUDY.md``
with its tables and figures. Every number and verdict in the report is computed here.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.audit import audit_snapshot
from eu_okun.baseline import _read_panel
from eu_okun.contract import EU27
from eu_okun.heterogeneity import (
    CRISIS,
    FOCUS,
    INDICATORS,
    LABELS,
    PRIMARY,
    SENSITIVITY,
    Correlation,
    CountryData,
    Evaluation,
    PairResult,
    Run,
    build_country,
    correlations,
    crisis,
    evaluate,
    indicators,
    lagged_share,
    pair_results,
    quarters_for,
    run,
)
from eu_okun.heterogeneity_data import SERIES, audit_heterogeneity, load_heterogeneity

MARGINS = ("prod", "hpw", "e_dom")
TABLE_MEASURES = ("prod", "hpw", "e_dom", "h", "e_res", "comm", "du", "dp")


@dataclass
class CaseStudy:
    """Everything the report is generated from."""

    primary: Run
    evaluations: list[Evaluation]
    pairs: list[PairResult]
    sensitivity: dict[str, tuple[list[Evaluation], list[PairResult]]]
    indicators: dict[str, dict[str, float]]
    correlations: list[Correlation]
    crisis: list[dict[str, Any]]
    coverage: dict[str, list[str]]
    theta: dict[str, dict[str, float]]
    provenance: dict[str, Any] = field(default_factory=dict)


def read_theta(review_dir: Path, robustness_dir: Path) -> dict[str, dict[str, float]]:
    """Committed pandemic-excluded estimates: Bayesian medians and classical K = 2 sums."""
    bayes, classical = {}, {}
    with (robustness_dir / "country_summaries.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["spec"] == "pandemic_excluded":
                bayes[row["geo"]] = float(row["median"])
    with (review_dir / "grid.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (row["sample"], row["gdp_lags"], row["participation"]) == (
                    "pandemic_excluded", "2", "False"):
                classical[row["geo"]] = float(row["gdp_sum"])
    return {"bayes": bayes, "classical": classical}


def study_from_data(data: Mapping[str, CountryData], quarters: Sequence[str],
                    epl: Mapping[str, Mapping[str, float]],
                    theta: Mapping[str, Mapping[str, float]], replicates: int = 1999,
                    permutations: int = 9999,
                    provenance: dict[str, Any] | None = None) -> CaseStudy:
    """Every registered computation on prepared data."""
    primary = run(data, quarters, PRIMARY, replicates)
    sens = {}
    for s in SENSITIVITY:
        r = run(data, quarters, s, replicates)
        sens[s.key] = (evaluate(r), pair_results(r))
    ind = indicators(data, quarters, epl, primary)
    corr = (correlations(theta["bayes"], ind, "Bayesian", permutations=permutations)
            + correlations(theta["classical"], ind, "classical", permutations=permutations)
            + correlations(theta["bayes"], ind, "Bayesian, without LU and IE",
                           exclude=("LU", "IE"), permutations=permutations))
    coverage = {m: sorted(g for g in EU27 if (g, m) in primary.estimates)
                for m in (*TABLE_MEASURES, "temp", "perm")}
    return CaseStudy(primary, evaluate(primary), pair_results(primary), sens, ind, corr,
                     crisis(data, quarters), coverage, {k: dict(v) for k, v in theta.items()},
                     provenance or {})


def build_study(base_dir: Path, snapshot: Path, review_dir: Path,
                robustness_dir: Path) -> CaseStudy:
    """Audit the snapshots and run the registered computation."""
    audit = audit_snapshot(base_dir)
    manifest = audit_heterogeneity(snapshot)
    base = _read_panel(base_dir / "processed" / "eu27_quarterly.csv")
    extra, epl = load_heterogeneity(snapshot)
    quarters = quarters_for()
    data = {geo: build_country(geo, quarters, base, extra) for geo in EU27}
    provenance = {"eurostat_manifest_sha256": audit["snapshot_manifest_sha256"],
                  "heterogeneity_files": {f["key"]: {"dataset": f["dataset"],
                                                     "sha256": f["sha256"], "url": f["url"],
                                                     "updated": f["updated"]}
                                          for f in manifest["files"]},
                  "downloaded_at_utc": manifest["downloaded_at_utc"]}
    return study_from_data(data, quarters, epl, read_theta(review_dir, robustness_dir),
                           provenance=provenance)


# ----------------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------------


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]], align: str = "") -> list[str]:
    align = align or "l" + "r" * (len(header) - 1)
    marks = ["---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(marks) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def _band(point: float, low: float, high: float, digits: int = 2) -> str:
    if not math.isfinite(point):
        return "n/a"
    return f"{point:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"


def _p(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _est(r: Run, geo: str, measure: str, digits: int = 2) -> str:
    e = r.get(geo, measure)
    return "n/a" if e is None else _band(e.point, *e.interval(), digits=digits)


def _median_point(r: Run, measure: str) -> float:
    values = [e.point for (g, m), e in r.estimates.items() if m == measure]
    return float(np.median(values)) if values else math.nan


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------


def render(cs: CaseStudy, assets: str, command: str) -> str:
    r = cs.primary
    ev = {e.key: e for e in cs.evaluations}
    lines = [
        "# Country heterogeneity: Portugal, Spain, the Netherlands and Luxembourg", "",
        "Generated by `eu-okun heterogeneity` from the frozen Eurostat and OECD snapshots and "
        "the committed classical and Bayesian estimates. Hypotheses, predictions, decision "
        "rules and the list of untestable hypotheses were pre-registered in "
        "[HETEROGENEITY_PROTOCOL.md](../HETEROGENEITY_PROTOCOL.md) before any computation.",
        "",
        "**How to read this report.** Three classes of evidence are kept apart:",
        "",
        "- **correlation**: cross-country associations (section 7) and descriptive "
        "companions;",
        "- **pattern consistent with a mechanism**: where output fluctuations are absorbed "
        "(margins, measurement concepts, contract types), sections 2-6 and 8;",
        "- **identified causal effect**: none. No identification strategy is available "
        "(IDENTIFICATION_STUDY.md), so no result here says that an institution *causes* a "
        "country's Okun coefficient.", "",
        "## Summary", "",
    ]
    rows = [[e.key, e.text, _band(e.estimate, e.low, e.high), _p(e.holm),
             f"**{e.verdict}**" if e.verdict != "not determined" else e.verdict]
            for e in cs.evaluations]
    lines += _table(["Prediction", "Statement", "Estimate [95%]", "Holm p", "Verdict"], rows,
                    "llrrl")
    counts = {v: sum(e.verdict == v for e in cs.evaluations)
              for v in ("supported", "contradicted", "not determined")}
    lines += [f"{counts['supported']} supported, {counts['contradicted']} contradicted, "
              f"{counts['not determined']} not determined, of {len(cs.evaluations)} "
              "pre-registered predictions (Holm over the determinable ones; P1.2 is the same "
              "test as P1.1, see the protocol's change log).", ""]
    for e in cs.evaluations:
        if e.note:
            lines.append(f"- {e.key}: {e.note}.")
    lines += ["", f"![Evidence map]({assets}/evidence.png)", ""]

    # 1. Data
    lines += ["## 1. Data, definitions and coverage", "",
              "All series are official and frozen with SHA-256 hashes; nothing is interpolated. "
              "Ratios combine numerator and denominator of the same concept (domestic with "
              "domestic, resident with resident). Countries with an estimate under the primary "
              "design:", ""]
    rows = [[LABELS[m], str(len(cs.coverage[m])),
             ", ".join(g for g in EU27 if g not in cs.coverage[m]) or "none"]
            for m in cs.coverage]
    lines += _table(["Measure", "Countries", "Missing"], rows, "lrl")
    sources = []
    for spec in SERIES.values():
        filters = ", ".join(f"{k}={v}" for k, v in spec.filters.items() if k != "freq")
        sources.append(f"`{spec.dataset}` ({filters})")
    lines += ["Sources: " + "; ".join(sources) + "; OECD `DSD_EPL@DF_EPL` (EPL_R, EPL_T, "
              "version 3, 2008). National statistical offices identified but not used "
              "(STATEC, CBS, INE Spain, INE Portugal): protocol, section 2.", ""]

    # 2. Margins
    lines += ["## 2. Margins of adjustment", "",
              "Two-lag elasticities to GDP growth (sum of the coefficients on g[t], g[t-1], "
              "g[t-2]), 95% block-bootstrap intervals. Output per hour, hours per worker and "
              "domestic employment sum to one by construction.", ""]
    rows = []
    for m in TABLE_MEASURES:
        rows.append([LABELS[m], *(_est(r, g, m) for g in FOCUS),
                     f"{_median_point(r, m):+.2f}"])
    lines += _table(["Measure", *FOCUS, "Median, all countries"], rows)
    lines += [f"![Margins of adjustment]({assets}/margins.png)", ""]

    # 3. Luxembourg
    lu_gap = [e for e in cs.evaluations if e.hypothesis == "H1"]
    others = sorted(e.point for (g, m), e in r.estimates.items() if m == "e_dom" and g != "LU")
    lu = r.get("LU", "e_dom")
    position = ""
    if lu is not None and others:
        q1, q3 = np.percentile(others, [25, 75])
        inside = q1 <= lu.point <= q3
        position = (f"LU's domestic-employment elasticity ({lu.point:+.2f}) lies "
                    f"{'inside' if inside else 'outside'} the interquartile range of the other "
                    f"countries [{q1:+.2f}, {q3:+.2f}] (descriptive companion).")
    lines += ["## 3. Luxembourg: production and residents (H1)", "",
              "GDP counts production by everyone working in Luxembourg; unemployment and "
              "participation count residents only. Domestic employment (`EMP_DC`) includes "
              "cross-border commuters, resident employment (`EMP_NC`) does not.", ""]
    rows = [[e.key, e.statistic, _band(e.estimate, e.low, e.high), _p(e.p_value), _p(e.holm),
             e.verdict] for e in lu_gap]
    lines += _table(["Prediction", "Statistic", "Estimate [95%]", "p", "Holm p", "Verdict"],
                    rows, "llrrrl")
    lines += [position, "",
              f"Luxembourg: domestic employment {_est(r, 'LU', 'e_dom_pe')}, resident "
              f"employment {_est(r, 'LU', 'e_res')}, unemployment {_est(r, 'LU', 'du')} pp.", "",
              f"![Domestic and resident employment]({assets}/luxembourg.png)", ""]

    # 4. Netherlands
    lines += ["## 4. The Netherlands: margins and timing (H2, H3)", ""]
    rows = [[e.key, e.statistic, _band(e.estimate, e.low, e.high), _p(e.holm), e.verdict]
            for e in cs.evaluations if e.hypothesis in ("H2", "H3")]
    lines += _table(["Prediction", "Statistic", "Estimate [95%]", "Holm p", "Verdict"], rows,
                    "llrrl")
    rows = []
    for g in FOCUS:
        if g in r.timing:
            share, draws, ok = lagged_share(r.timing[g])
            total = r.timing[g]
            total_draws = total.coef_draws[:, 1:].sum(axis=1)
            lo, hi = np.percentile(total_draws, [2.5, 97.5])
            rows.append([g, _band(float(total.coef[1:].sum()), lo, hi),
                         f"{share:.2f}" if ok else "not determinable"])
    lines += ["Unemployment response with four lags (pp per %): the lagged share is reported "
              "only where the sum's interval excludes zero. **Timing says nothing about "
              "temporary contracts.**", ""]
    lines += _table(["Country", "K = 4 sum [95%]", "Share from lags 1-4"], rows, "lrr")

    # 5. Temporary jobs
    lines += ["## 5. Temporary jobs as a buffer (H4)", "",
              "Year-on-year elasticities of temporary and permanent employees (LFS, not "
              "seasonally adjusted) to year-on-year GDP growth; changes spanning a break or "
              "low-reliability flag are dropped. A larger temporary elasticity shows where "
              "employment adjustment falls; it does not show that temporary contracts cause a "
              "country's Okun coefficient.", ""]
    rows = [[g, _est(r, g, "temp"), _est(r, g, "perm"),
             (lambda e: f"{e.verdict}" if e else "")(ev.get(f"P4.1 {g}"))]
            for g in FOCUS]
    lines += _table(["Country", "Temporary [95%]", "Permanent [95%]", "Verdict (P4.1)"], rows,
                    "lrrl")

    # 6. Portugal and Spain
    lines += ["## 6. Portugal and Spain (H5)", ""]
    rows = [[e.key, e.statistic, _band(e.estimate, e.low, e.high), _p(e.holm), e.verdict]
            for e in cs.evaluations if e.hypothesis == "H5"]
    lines += _table(["Prediction", "Statistic", "Estimate [95%]", "Holm p", "Verdict"], rows,
                    "llrrl")
    lines += [f"Crisis window {CRISIS[0]} to {CRISIS[1]} (cumulative changes, descriptive): GDP "
              "and the national-accounts margins from the start of the window; unemployment and "
              "participation from the first quarter their series exist.", ""]
    rows = [[c["geo"], f"{c['g']:+.1f}", f"{c['prod']:+.1f}", f"{c['hpw']:+.1f}",
             f"{c['e_dom']:+.1f}", f"{c['du']:+.1f} (from {c['du_from']})",
             f"{c['dp']:+.1f} (from {c['dp_from']})"] for c in cs.crisis]
    lines += _table(["Country", "GDP (%)", "Output per hour (%)", "Hours per worker (%)",
                     "Domestic employment (%)", "Unemployment rate (pp)",
                     "Participation rate (pp)"], rows)
    lines += [f"![Crisis decomposition]({assets}/crisis.png)", ""]

    # 7. Institutions
    lines += ["## 7. Institutions and structure across countries (H6, correlation only)", "",
              "Spearman correlation between the pandemic-excluded Okun coefficient and each "
              "indicator; permutation p-values, Holm over the six indicators within each "
              "variant. theta is negative: a positive correlation means a weaker unemployment "
              "response.", ""]
    rows = [[c.variant, c.label, "+" if c.predicted > 0 else "−", str(c.n), f"{c.rho:+.2f}",
             _p(c.p_value), _p(c.holm), c.verdict] for c in cs.correlations]
    lines += _table(["theta", "Indicator", "Predicted sign", "Countries", "rho", "p", "Holm p",
                     "Verdict"], rows, "llrrrrrl")
    lines += [f"![Institutions]({assets}/institutions.png)", ""]

    # 8. Pairs
    lines += ["## 8. Do the country differences remain under other measures?", "",
              "Elasticity differences (first minus second country) under four labour-input "
              "measures. A difference *remains* if its 95% interval excludes zero with the same "
              "sign as under unemployment (`-du`); *disappears* if the interval includes zero; "
              "*reverses* otherwise. Units differ across measures: only the persistence of the "
              "ordering is compared.", ""]
    rows = []
    for p in cs.pairs:
        rows.append([f"{p.a}-{p.b}", LABELS[p.measure],
                     _band(p.estimate, p.low, p.high), p.status])
    lines += _table(["Pair", "Measure", "Difference [95%]", "Status"], rows, "llrl")

    # 9. Sensitivity
    lines += ["## 9. Sensitivity (pre-registered)", "",
              "Verdict of every prediction under each sensitivity setting (each with its own "
              "Holm family).", ""]
    keys = list(cs.sensitivity)
    rows = []
    for e in cs.evaluations:
        cells = [e.key, e.verdict]
        for k in keys:
            match = next(x for x in cs.sensitivity[k][0] if x.key == e.key)
            cells.append(match.verdict)
        rows.append(cells)
    lines += _table(["Prediction", "Primary", *keys], rows, "l" * (2 + len(keys)))
    rows = []
    for k in keys:
        statuses = [p.status for p in cs.sensitivity[k][1] if p.measure != "neg_du"]
        rows.append([k, *(str(statuses.count(s)) for s in
                          ("remains", "disappears", "reverses"))])
    base = [p.status for p in cs.pairs if p.measure != "neg_du"]
    rows.insert(0, ["primary", *(str(base.count(s)) for s in ("remains", "disappears",
                                                              "reverses"))])
    lines += ["Pair differences under the alternative measures:", ""]
    lines += _table(["Setting", "Remain", "Disappear", "Reverse"], rows)

    # 10. Conclusions
    supported = [e for e in cs.evaluations if e.verdict == "supported"]
    contradicted = [e for e in cs.evaluations if e.verdict == "contradicted"]
    lines += ["## 10. What can and cannot be concluded", "",
              "**Patterns consistent with a mechanism (supported predictions):** "
              + ("; ".join(f"{e.key} ({e.text})" for e in supported) or "none") + ".", "",
              "**Contradicted predictions:** "
              + ("; ".join(f"{e.key} ({e.text})" for e in contradicted) or "none") + ".", "",
              "**Correlations (H6):** "
              + ("; ".join(f"{c.label} ({c.variant}: rho {c.rho:+.2f}, {c.verdict})"
                           for c in cs.correlations if c.verdict != "not determined")
                 or "no indicator is associated with the Okun coefficient after the Holm "
                    "adjustment, in any variant") + ".", "",
              "**Identified causal effects:** none.", "",
              "**Hypotheses that cannot be tested reliably** (documented in the protocol "
              "before any result):", "",
              "- that temporary contracts *cause* the Dutch or Spanish responses;",
              "- the effects of the 2012 labour-market reforms in Spain and Portugal (single "
              "events coinciding with the sovereign-debt crisis);",
              "- short-time work schemes (no harmonised quarterly take-up before 2020);",
              "- labour hoarding as distinct from unmeasured effort or capacity utilisation;",
              "- collective bargaining, minimum wages and benefit generosity;",
              "- the unemployment of Luxembourg's cross-border workers, recorded in their "
              "countries of residence;",
              "- sectoral composition.", "",
              "## 11. References", "",
              "- Ball, L., Leigh, D. and Loungani, P. (2017). Okun's law: fit at 50? *Journal of "
              "Money, Credit and Banking* 49(7).",
              "- Bentolila, S., Cahuc, P., Dolado, J. J. and Le Barbanchon, T. (2012). Two-tier "
              "labour markets in the Great Recession: France versus Spain. *Economic Journal* "
              "122(562).",
              "- Blanchard, O. and Portugal, P. (2001). What hides behind an unemployment rate: "
              "comparing Portuguese and U.S. labor markets. *American Economic Review* 91(1).",
              "- Ohanian, L. E. and Raffo, A. (2012). Aggregate hours worked in OECD countries: "
              "new measurement and implications for business cycles. *Journal of Monetary "
              "Economics* 59(1).",
              "- Okun, A. M. (1962). Potential GNP: its measurement and significance. "
              "*Proceedings of the Business and Economic Statistics Section, American "
              "Statistical Association*.",
              "- Eurostat: `namq_10_pe`, `namq_10_a10_e`, `lfsq_etgaed`, `lfsq_eegaed`, "
              "`lfsq_etpga`, `lfsq_eppga` (Eurostat database; reuse with acknowledgement).",
              "- OECD (2020). OECD Indicators of Employment Protection (EPL), version 3 "
              "(CC BY 4.0).", "",
              "## 12. Provenance and reproduction", ""]
    prov = cs.provenance
    if prov:
        lines += [f"Eurostat quarterly snapshot manifest `{prov['eurostat_manifest_sha256']}`; "
                  f"heterogeneity snapshot downloaded {prov['downloaded_at_utc']}, file hashes "
                  "and URLs in `provenance.json`. Bootstrap: 1999 synchronised circular block "
                  "resamples, block length 4, seed 20261018.", ""]
    lines += ["```bash", "poetry install --extras \"bayes report\"", command, "```", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------
# Figures and tables
# ----------------------------------------------------------------------------------


def write_figures(cs: CaseStudy, directory: Path, source: str) -> None:
    from eu_okun.review_figures import AQUA, BLUE, INK2, MUTED, ORANGE, SURFACE, _finish, _pyplot

    plt = _pyplot()
    r = cs.primary
    colour = {"supported": AQUA, "contradicted": ORANGE, "not determined": MUTED}

    # Evidence map: standardised statistics (mixed units are never put on one axis).
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    evs = cs.evaluations[::-1]
    for y, item in enumerate(evs):
        if math.isfinite(item.estimate) and item.sd > 0:
            z = item.estimate / item.sd
            ax.plot([z - 1.96, z + 1.96], [y, y], color=colour[item.verdict], lw=2, alpha=0.6)
            ax.plot([z], [y], "o", ms=6, color=colour[item.verdict], mec=SURFACE)
        else:
            ax.text(0, y, "not determinable", ha="center", va="center", fontsize=8,
                    color=MUTED)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(evs)))
    ax.set_yticklabels([f"{item.key} {item.hypothesis}" for item in evs], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Estimate / bootstrap sd, with ±1.96 (predicted direction: positive)")
    for name, shade in colour.items():
        ax.plot([], [], "o", color=shade, label=name)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Pre-registered predictions: strength and verdict (Holm-adjusted)",
                 fontsize=10)
    _finish(plt, fig, directory / "evidence.png", source)

    # Margins of adjustment.
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    width = 0.25
    for k, (m, shade) in enumerate(zip(MARGINS, (BLUE, ORANGE, AQUA), strict=True)):
        xs = np.arange(len(FOCUS)) + (k - 1) * width
        points, lows, highs = [], [], []
        for g in FOCUS:
            est = r.get(g, m)
            lo, hi = est.interval() if est else (math.nan, math.nan)
            points.append(est.point if est else math.nan)
            lows.append(lo)
            highs.append(hi)
        ax.bar(xs, points, width=width * 0.9, color=shade, label=LABELS[m])
        ax.errorbar(xs, points, yerr=[np.subtract(points, lows), np.subtract(highs, points)],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(FOCUS)))
    ax.set_xticklabels(FOCUS)
    ax.set_ylabel("Elasticity to GDP growth (two-lag sum)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Where output fluctuations are absorbed (the three margins sum to one)",
                 fontsize=10)
    _finish(plt, fig, directory / "margins.png", source)

    # Luxembourg: domestic against resident employment.
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for g in EU27:
        d, n = r.get(g, "e_dom_pe"), r.get(g, "e_res")
        if d is None or n is None:
            continue
        shade = ORANGE if g == "LU" else BLUE
        ax.plot([n.point], [d.point], "o", ms=7 if g == "LU" else 5, color=shade, mec=SURFACE)
        ax.annotate(g, (n.point, d.point), fontsize=7.5, color=INK2, xytext=(3, 3),
                    textcoords="offset points")
        if g == "LU":
            (dlo, dhi), (nlo, nhi) = d.interval(), n.interval()
            ax.plot([n.point, n.point], [dlo, dhi], color=shade, lw=1.5)
            ax.plot([nlo, nhi], [d.point, d.point], color=shade, lw=1.5)
    lims = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lims), max(lims)
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=(0, (3, 2)))
    ax.set_xlabel("Resident employment elasticity (EMP_NC)")
    ax.set_ylabel("Domestic employment elasticity (EMP_DC)")
    ax.set_title("Above the diagonal: production employment adjusts more than residents",
                 fontsize=9.5)
    _finish(plt, fig, directory / "luxembourg.png", source)

    # Crisis decomposition.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 3.8),
                                 gridspec_kw={"width_ratios": [2.2, 1]})
    items = (("g", "GDP"), ("prod", "Output per hour"), ("hpw", "Hours per worker"),
             ("e_dom", "Domestic employment"))
    for k, (row, shade) in enumerate(zip(cs.crisis, (BLUE, ORANGE), strict=False)):
        a1.bar(np.arange(len(items)) + (k - 0.5) * 0.35, [row[m] for m, _ in items],
               width=0.33, color=shade, label=row["geo"])
        a2.bar(np.arange(2) + (k - 0.5) * 0.35, [row["du"], row["dp"]], width=0.33, color=shade)
    a1.set_xticks(range(len(items)))
    a1.set_xticklabels([label for _, label in items], fontsize=8.5)
    a1.set_ylabel("Cumulative change (%)")
    a1.legend(fontsize=8)
    a1.axhline(0, color=MUTED, lw=1)
    a2.set_xticks(range(2))
    a2.set_xticklabels(["Unemployment", "Participation"], fontsize=8.5)
    a2.set_ylabel("Cumulative change (pp)")
    a2.axhline(0, color=MUTED, lw=1)
    fig.suptitle(f"Crisis window {CRISIS[0]}..{CRISIS[1]} (descriptive)", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold", color=INK2)
    fig.subplots_adjust(top=0.85, wspace=0.3)
    _finish(plt, fig, directory / "crisis.png", source, credit_y=-0.04)

    # Institutions.
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.6))
    primary = [c for c in cs.correlations if c.variant == "Bayesian"]
    for ax, (key, label, _), corr in zip(axes.ravel(), INDICATORS, primary, strict=True):
        for g in EU27:
            if key in cs.indicators.get(g, {}) and g in cs.theta["bayes"]:
                focus = g in FOCUS
                ax.plot([cs.indicators[g][key]], [cs.theta["bayes"][g]], "o",
                        ms=6 if focus else 4, color=ORANGE if focus else BLUE, mec=SURFACE)
                if focus:
                    ax.annotate(g, (cs.indicators[g][key], cs.theta["bayes"][g]), fontsize=7.5,
                                xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel(label, fontsize=8)
        ax.set_title(f"rho {corr.rho:+.2f}, Holm p {_p(corr.holm)}", fontsize=9)
    axes[0, 0].set_ylabel("Okun theta (Bayesian)")
    axes[1, 0].set_ylabel("Okun theta (Bayesian)")
    fig.suptitle("Correlation only: pandemic-excluded Okun coefficients against indicators",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK2)
    fig.subplots_adjust(hspace=0.45, wspace=0.3, top=0.88)
    _finish(plt, fig, directory / "institutions.png", source, credit_y=-0.02)


def write_tables(cs: CaseStudy, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        with (directory / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    est = []
    for (geo, m), e in sorted(cs.primary.estimates.items()):
        lo, hi = e.interval()
        est.append({"geo": geo, "measure": m, "elasticity": e.point, "low": lo, "high": hi,
                    "n": e.n})
    write("elasticities.csv", est)
    write("predictions.csv", [{"setting": "primary", **asdict(e)} for e in cs.evaluations]
          + [{"setting": k, **asdict(e)} for k, (evs, _) in cs.sensitivity.items()
             for e in evs])
    write("pairs.csv", [{"setting": "primary", **asdict(p)} for p in cs.pairs]
          + [{"setting": k, **asdict(p)} for k, (_, ps) in cs.sensitivity.items() for p in ps])
    write("correlations.csv", [asdict(c) for c in cs.correlations])
    write("indicators.csv", [{"geo": g, **{k: cs.indicators[g].get(k, "") for k, _, _ in
                                           INDICATORS}} for g in EU27])
    write("crisis.csv", cs.crisis)
    (directory / "provenance.json").write_text(json.dumps(cs.provenance, indent=1) + "\n",
                                               encoding="utf-8", newline="\n")


def write_report(base_dir: Path, snapshot: Path, review_dir: Path, robustness_dir: Path,
                 output_dir: Path, report: Path | None = None,
                 figures: bool = True) -> CaseStudy:
    cs = build_study(base_dir, snapshot, review_dir, robustness_dir)
    report = report or output_dir / "HETEROGENEITY_CASE_STUDY.md"
    write_tables(cs, output_dir)
    if figures:
        write_figures(cs, output_dir, "Source: Eurostat (national accounts, LFS), OECD EPL. "
                      "Associations only; HETEROGENEITY_PROTOCOL.md.")
    assets = Path(os.path.relpath(output_dir, report.parent)).as_posix()
    command = (f"poetry run eu-okun heterogeneity --data-dir {base_dir.as_posix()} "
               f"--snapshot {snapshot.as_posix()} --output-dir {output_dir.as_posix()} "
               f"--report {report.as_posix()}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(cs, assets, command), encoding="utf-8", newline="\n")
    return cs
