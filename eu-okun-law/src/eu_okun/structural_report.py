"""Tables, figures and the Markdown report of the identification study.

Runs the pre-registered estimation of ``IDENTIFICATION_PROTOCOL.md`` on the frozen
Eurostat and external-shock snapshots and writes ``IDENTIFICATION_RESULTS.md``. Every
number and every verdict in the report is computed here.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.audit import audit_snapshot
from eu_okun.baseline import _read_panel
from eu_okun.external_shocks import audit_shock_snapshot, load_quarterly
from eu_okun.structural import (
    OKUN_HORIZONS,
    OUTCOME_LABELS,
    OUTCOMES,
    PRIMARY,
    SHOCKS,
    VARIANTS,
    CountryResponse,
    Predictability,
    ShockResults,
    ShockSpec,
    country_responses,
    estimate_shock,
    panel_arrays,
    predictability,
)

PREDICTABILITY_ALPHA = 0.05
WEAK_T = 3.16


@dataclass
class IdentificationReview:
    """Everything the report is generated from."""

    primary: dict[str, ShockResults]
    robustness: dict[str, ShockResults]
    predictability: dict[str, Predictability]
    countries: dict[str, list[CountryResponse]]
    provenance: dict[str, Any] = field(default_factory=dict)

    def doubtful(self, key: str) -> bool:
        """Protocol, section 5: predictable shocks make identification doubtful."""
        return self.predictability[key].p_value < PREDICTABILITY_ALPHA


def build_review(eurostat_dir: Path, shocks_dir: Path) -> IdentificationReview:
    """Audit both snapshots and run every pre-registered estimate."""
    audit = audit_snapshot(eurostat_dir)
    shock_manifest = audit_shock_snapshot(shocks_dir)
    end = audit["analysis_period"]["end"]
    panel = _read_panel(eurostat_dir / "processed" / "eu27_quarterly.csv")
    arrays = panel_arrays(panel, "2008-Q1", str(end))
    shocks = load_quarterly(shocks_dir)
    primary = {key: estimate_shock(arrays, shocks, spec) for key, spec in SHOCKS.items()}
    robustness: dict[str, ShockResults] = {}
    for key, spec in VARIANTS.items():
        robustness[key] = estimate_shock(arrays, shocks, spec)
    for key, spec in SHOCKS.items():
        robustness[f"{key}: pandemic included"] = estimate_shock(
            arrays, shocks, spec, replace(PRIMARY, exclude_pandemic=False))
        robustness[f"{key}: two lags"] = estimate_shock(arrays, shocks, spec,
                                                        replace(PRIMARY, lags=2))
        robustness[f"{key}: Driscoll-Kraay"] = estimate_shock(
            arrays, shocks, spec, replace(PRIMARY, driscoll_kraay=True))
    tests = {}
    for key, spec in {**SHOCKS, **VARIANTS}.items():
        tests[key] = predictability(arrays, shocks[spec.series], spec)
    countries = {key: country_responses(arrays, shocks[spec.series], spec,
                                        primary[key].scale, PRIMARY)
                 for key, spec in SHOCKS.items()}
    provenance = {
        "eurostat_manifest_sha256": audit["snapshot_manifest_sha256"],
        "eurostat_processed_sha256": audit["processed_csv_sha256"],
        "shock_files": {f["file"]: {"sha256": f["sha256"], "commit": f["commit"],
                                    "licence": f["licence"]} for f in shock_manifest["files"]},
        "shock_quarterly_sha256": shock_manifest["quarterly_csv_sha256"],
        "sample_quarters": [arrays.quarters[0], arrays.quarters[-1]],
    }
    return IdentificationReview(primary, robustness, tests, countries, provenance)


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


def _horizons(values: Sequence[int]) -> str:
    return ", ".join(str(h) for h in values) if values else "none"


def distinguishable(results: ShockResults, outcome: str) -> list[int]:
    """Horizons at which the 90% band excludes zero."""
    return [r.h for r in results.responses if r.outcome == outcome
            and (r.low90 > 0 or r.high90 < 0)]


def _label(review: IdentificationReview, key: str, spec: ShockSpec) -> str:
    if not spec.identified:
        return "contrast only (not an identified shock)"
    return ("identification doubtful (shock predictable)" if review.doubtful(key)
            else "identified under A1-A6")


def render(review: IdentificationReview, assets: str, command: str) -> str:
    """Assemble the Markdown report."""
    lines = ["# Identification study: responses to externally identified shocks", "",
             "Generated by `eu-okun identification` from the frozen snapshots "
             "`snapshots/eurostat-2026-10-15` and `snapshots/external-shocks-2026-10-16`. "
             "What is estimated and how it is judged was pre-registered in "
             "[IDENTIFICATION_PROTOCOL.md](../IDENTIFICATION_PROTOCOL.md); why only this is "
             "estimated is argued in [IDENTIFICATION_STUDY.md](IDENTIFICATION_STUDY.md).", "",
             "**Scope.** Two common shocks identified outside this study: ECB monetary policy "
             "surprises (a demand shock, euro-area members) and OPEC oil supply surprises (a "
             "supply shock that is not a productivity shock, EU-27). Responses are causal "
             "only under the protocol's assumptions A1-A6. Productivity shocks, labour-supply "
             "shocks and country-specific GDP shocks are not identified, and every Okun "
             "coefficient estimated elsewhere in this study remains a reduced-form "
             "association.", ""]

    # Summary
    lines += ["## Summary", ""]
    for key, res in review.primary.items():
        spec = res.spec
        lines.append(f"**{spec.label}** ({spec.unit}; {_label(review, key, spec)}):")
        lines.append("")
        for outcome in OUTCOMES:
            lines.append(f"- {OUTCOME_LABELS[outcome]}: 90% band excludes zero at horizons "
                         f"{_horizons(distinguishable(res, outcome))} (of 0-8).")
        for o in res.okun:
            status = "informative" if o.informative else "uninformative"
            weak = " (weak first stage)" if abs(o.first_stage_t) < WEAK_T else ""
            lines.append(f"- Shock-conditional Okun coefficient at h = {o.h}: estimate "
                         f"{o.estimate:+.3f}, 90% Anderson-Rubin set {o.text()} ({status}); "
                         f"first-stage t {o.first_stage_t:+.2f}{weak}.")
        t = review.predictability[key]
        lines.append(f"- Predictability from lagged EU data (A3): F = {t.f_statistic:.2f}, "
                     f"p = {_p(t.p_value)}.")
        tests = len(res.responses)
        hits = sum(len(distinguishable(res, o)) for o in OUTCOMES)
        lines.append(f"- Multiplicity: {hits} of {tests} pointwise 90% bands exclude zero; "
                     f"about {0.1 * tests:.1f} would by chance if the shock had no effect.")
        lines.append("")
    comparison = _compare(review)
    lines += [comparison, ""]

    # 1. Design
    lines += ["## 1. Design", "",
              "Lag-augmented local projections, pooled over countries with country fixed "
              "effects; four lags of the shock, the unemployment change, GDP growth and the "
              "participation change; standard errors clustered by quarter; horizons 0-8. "
              "Observations whose window [t-4, t+h] touches 2020-Q1..2021-Q4 are excluded in "
              "the primary sample.", ""]
    rows: list[list[Any]] = []
    for res in review.primary.values():
        first = next(r for r in res.responses if r.h == 0)
        last = next(r for r in res.responses if r.h == 8)
        rows.append([res.spec.label, res.spec.series, res.spec.unit,
                     "euro-area members" if res.spec.euro_area_only else "EU-27",
                     f"{first.n} / {last.n}", f"{first.clusters} / {last.clusters}"])
    lines += _table(["Shock", "Series", "Normalisation", "Countries",
                     "Observations (h = 0 / 8)", "Quarters (h = 0 / 8)"], rows, "llllrr")

    # 2. Diagnostics
    lines += ["## 2. Diagnostics", "",
              "Predictability (A3): the quarterly shock regressed on four lags of the "
              "cross-country mean unemployment change and GDP growth; F-test of the eight "
              "lag coefficients. A p-value below 0.05 counts against A3.", ""]
    rows = [[key, str(review.predictability[key].n),
             f"{review.predictability[key].f_statistic:.2f}",
             _p(review.predictability[key].p_value),
             f"{review.predictability[key].autocorrelation:+.2f}"]
            for key in review.predictability]
    lines += _table(["Series", "Quarters", "F", "p", "Lag-1 autocorrelation"], rows)
    rows = []
    for res in review.primary.values():
        for o in res.okun:
            rows.append([res.spec.label, str(o.h), f"{o.first_stage_t:+.2f}",
                         "weak" if abs(o.first_stage_t) < WEAK_T else "not weak"])
    lines += ["Relevance (A4): the t-statistic of the cumulative GDP response (the first stage "
              f"of the Okun LP-IV); |t| < {WEAK_T} is weak.", ""]
    lines += _table(["Shock", "Horizon", "First-stage t", "Strength"], rows, "lrrl")

    # 3. Responses
    lines += ["## 3. Responses", "",
              "Pointwise 68% and 90% bands. Cumulative GDP is the sum of quarterly growth "
              "rates from t to t+h (%); unemployment and participation are changes from t-1 "
              "(pp).", ""]
    for key, res in review.primary.items():
        rows = []
        for h in (0, 2, 4, 6, 8):
            cells = [str(h)]
            for outcome in OUTCOMES:
                r = next(x for x in res.responses if x.h == h and x.outcome == outcome)
                cells.append(f"{r.estimate:+.3f} [{r.low90:+.3f}, {r.high90:+.3f}]")
            rows.append(cells)
        lines += [f"**{res.spec.label}** ({res.spec.unit}):", ""]
        lines += _table(["h", *(OUTCOME_LABELS[o] + " [90%]" for o in OUTCOMES)], rows)
        lines += [f"![Responses to the {res.spec.label}]({assets}/responses_{key}.png)", ""]

    # 4. Okun coefficients
    lines += ["## 4. Shock-conditional Okun coefficients", "",
              "kappa = response of the unemployment change / response of cumulative GDP "
              "(pp per %), estimated by LP-IV with the shock as instrument. The Anderson-Rubin "
              "(Fieller) set is valid however weak the instrument; an unbounded set means the "
              "data do not determine kappa.", ""]
    rows = []
    for res in review.primary.values():
        for o in res.okun:
            rows.append([res.spec.label, str(o.h), f"{o.estimate:+.3f}", o.text(),
                         "informative" if o.informative else "uninformative"])
    lines += _table(["Shock", "Horizon", "Estimate", "90% AR set", "Verdict"], rows, "lrrll")
    lines += [comparison, ""]

    # 5. Robustness
    lines += ["## 5. Robustness (pre-registered)", ""]
    rows = []
    for name, res in review.robustness.items():
        cells = [name, _label(review, res.spec.key, res.spec)]
        for outcome in ("gdp", "unemployment"):
            for h in OKUN_HORIZONS:
                r = next(x for x in res.responses if x.h == h and x.outcome == outcome)
                cells.append(f"{r.estimate:+.3f} [{r.low90:+.3f}, {r.high90:+.3f}]")
        cells += [next(o for o in res.okun if o.h == h).text() for h in OKUN_HORIZONS]
        rows.append(cells)
    lines += _table(["Variant", "Status", "GDP h=4 [90%]", "GDP h=8 [90%]",
                     "Unemployment h=4 [90%]", "Unemployment h=8 [90%]", "Okun h=4 AR set",
                     "Okun h=8 AR set"], rows, "llrrrrll")
    bounded = [name for name, res in review.robustness.items()
               if any(o.informative for o in res.okun)]
    lines += [("Variants with a bounded Okun set at h = 4 or 8: " + ", ".join(bounded) + ". "
               if bounded else "No variant gives a bounded Okun set. ")
              + "Variants that include 2020-2021 rest on quarters in which the identified "
              "surprises coincided with the pandemic collapse; the protocol excludes those "
              "windows because the exclusion restriction (A3) is least credible there.", ""]

    # 6. Countries
    lines += ["## 6. Countries (descriptive)", "",
              "Country-by-country local projections at h = 4 (Newey-West, h + 1 lags), "
              "alphabetical. Individual country series are short; no ranking and no "
              "per-country significance claims.", ""]
    for key, rows_c in review.countries.items():
        rows = []
        by_geo: dict[str, dict[str, CountryResponse]] = {}
        for cr in rows_c:
            by_geo.setdefault(cr.geo, {})[cr.outcome] = cr
        for geo in sorted(by_geo):
            g, u = by_geo[geo]["gdp"], by_geo[geo]["unemployment"]
            rows.append([geo, str(g.n), f"{g.estimate:+.3f} [{g.low90:+.3f}, {g.high90:+.3f}]",
                         f"{u.estimate:+.3f} [{u.low90:+.3f}, {u.high90:+.3f}]"])
        lines += [f"**{review.primary[key].spec.label}**:", ""]
        lines += _table(["Country", "Observations", "GDP h=4 [90%]", "Unemployment h=4 [90%]"],
                        rows)
    lines += [f"![Country responses at h = 4]({assets}/countries.png)", ""]

    # 7. Conclusion and reproduction
    lines += [
        "## 7. What these results do and do not establish", "",
        "- They describe the average responses of euro-area members (monetary policy) and "
        "of EU countries (oil supply news) to common shocks, causal under A1-A6 and only "
        "where bands exclude zero or AR sets are bounded.",
        "- They do not compare demand-driven with *productivity*-driven GDP changes: no "
        "productivity shock is identified.",
        "- They do not separate labour-supply from demand shocks, and they do not identify "
        "country-specific GDP shocks.",
        "- The reduced-form Okun coefficients of the rest of this study are weighted "
        "averages of shock-conditional coefficients with unknown weights "
        "(IDENTIFICATION_STUDY.md, section 1); nothing here turns them into causal "
        "effects.", "",
        "## 8. Provenance and reproduction", "",
        f"Eurostat snapshot manifest SHA-256 `{review.provenance['eurostat_manifest_sha256']}`; "
        f"quarterly shocks SHA-256 `{review.provenance['shock_quarterly_sha256']}`; raw shock "
        "files and commits in `provenance.json`.", "",
        "```bash",
        "poetry install --extras \"bayes report structural\"",
        command,
        "```", "",
    ]
    return "\n".join(lines)


def _compare(review: IdentificationReview) -> str:
    """Protocol, section 5: compare demand and supply Okun coefficients only if both
    are informative at the same horizon."""
    mp, oil = review.primary["MP"].okun, review.primary["OIL"].okun
    statements = []
    for a, b in zip(mp, oil, strict=True):
        if a.informative and b.informative:
            overlap = a.lower <= b.upper and b.lower <= a.upper
            statements.append(f"at h = {a.h} the demand and supply sets "
                              f"{'overlap' if overlap else 'do not overlap'}")
        else:
            missing = [s for s, o in (("monetary policy", a), ("oil supply", b))
                       if not o.informative]
            statements.append(f"at h = {a.h} no comparison is possible ({' and '.join(missing)}"
                              " uninformative)")
    return ("**Demand against supply.** Under the protocol's rule, "
            + "; ".join(statements) + ".")


# ----------------------------------------------------------------------------------
# Figures and tables
# ----------------------------------------------------------------------------------


def write_figures(review: IdentificationReview, directory: Path, source: str) -> None:
    """Responses with bands per shock, and country responses at h = 4."""
    from eu_okun.review_figures import AQUA, BLUE, INK2, MUTED, ORANGE, SURFACE, _finish, _pyplot

    plt = _pyplot()
    for key, res in review.primary.items():
        fig, axes = plt.subplots(1, len(OUTCOMES), figsize=(12.0, 3.8))
        for ax, outcome in zip(axes, OUTCOMES, strict=True):
            rows = [r for r in res.responses if r.outcome == outcome]
            h = np.asarray([r.h for r in rows])
            ax.fill_between(h, [r.low90 for r in rows], [r.high90 for r in rows], color=BLUE,
                            alpha=0.15, lw=0)
            ax.fill_between(h, [r.low68 for r in rows], [r.high68 for r in rows], color=BLUE,
                            alpha=0.3, lw=0)
            ax.plot(h, [r.estimate for r in rows], color=BLUE, lw=2, marker="o", ms=4,
                    mec=SURFACE, mew=1)
            ax.axhline(0, color=MUTED, lw=1)
            ax.set_title(OUTCOME_LABELS[outcome], fontsize=9.5)
            ax.set_xlabel("Quarters after the shock")
            ax.set_xticks(range(0, 9, 2))
        fig.suptitle(f"{res.spec.label}: {res.spec.unit}; 68% and 90% bands",
                     x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK2)
        fig.subplots_adjust(wspace=0.28, top=0.82, bottom=0.2)
        _finish(plt, fig, directory / f"responses_{key}.png", source, credit_y=-0.06)

    keys = list(review.countries)
    fig, axes = plt.subplots(1, len(keys), figsize=(11.0, 8.6), sharey=False)
    for ax, key, colour in zip(axes, keys, (BLUE, ORANGE), strict=True):
        country_rows = sorted((cr for cr in review.countries[key]
                               if cr.outcome == "unemployment"), key=lambda cr: cr.geo)
        ys = np.arange(len(country_rows))[::-1]
        for y, cr in zip(ys, country_rows, strict=True):
            ax.plot([cr.low90, cr.high90], [y, y], color=colour, lw=1.6, alpha=0.7,
                    solid_capstyle="round")
            ax.plot([cr.estimate], [y], "o", ms=5.5, color=colour, mec=SURFACE, mew=1)
        pooled = next(r for r in review.primary[key].responses
                      if r.h == 4 and r.outcome == "unemployment")
        ax.axvline(pooled.estimate, color=AQUA, lw=1.5, ls=(0, (3, 2)))
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([cr.geo for cr in country_rows], fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"{review.primary[key].spec.label}\nunemployment, h = 4 (pp)", fontsize=9.5)
    fig.supxlabel("Country estimate with 90% interval; dashed: pooled estimate", fontsize=9,
                  color=INK2)
    fig.subplots_adjust(wspace=0.35, bottom=0.08)
    _finish(plt, fig, directory / "countries.png", source, credit_y=-0.02)


def write_tables(review: IdentificationReview, directory: Path) -> None:
    """Responses, Okun sets, diagnostics, country responses and provenance."""
    directory.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: Sequence[dict[str, Any]]) -> None:
        with (directory / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    responses = [{"variant": name, **asdict(r)}
                 for name, res in {**review.primary, **review.robustness}.items()
                 for r in res.responses]
    write("responses.csv", responses)
    okun = [{"variant": name, **asdict(o), "set": o.text()}
            for name, res in {**review.primary, **review.robustness}.items() for o in res.okun]
    write("okun_sets.csv", okun)
    write("predictability.csv", [asdict(t) for t in review.predictability.values()])
    write("country_responses.csv", [asdict(r) for rows in review.countries.values()
                                    for r in rows])
    (directory / "provenance.json").write_text(json.dumps(review.provenance, indent=1) + "\n",
                                               encoding="utf-8", newline="\n")


def write_report(eurostat_dir: Path, shocks_dir: Path, output_dir: Path,
                 report: Path | None = None, figures: bool = True) -> IdentificationReview:
    """Build the review and write tables, figures and the Markdown report."""
    review = build_review(eurostat_dir, shocks_dir)
    report = report or output_dir / "IDENTIFICATION_RESULTS.md"
    write_tables(review, output_dir)
    if figures:
        source = ("Source: Eurostat (2026-10-15); Jarocinski-Karadi (2020) and Kaenzig (2021) "
                  "shocks, CC BY 4.0. Causal only under IDENTIFICATION_PROTOCOL.md A1-A6.")
        write_figures(review, output_dir, source)
    assets = Path(os.path.relpath(output_dir, report.parent)).as_posix()
    command = (f"poetry run eu-okun identification --data-dir {eurostat_dir.as_posix()} "
               f"--shocks-dir {shocks_dir.as_posix()} --output-dir {output_dir.as_posix()} "
               f"--report {report.as_posix()}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(review, assets, command), encoding="utf-8", newline="\n")
    return review

