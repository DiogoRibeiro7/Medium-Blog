"""Tables, figure and the Markdown report of the sign-restricted SVAR study.

Runs every specification registered in ``SIGN_RESTRICTIONS_PROTOCOL.md`` on the frozen
Eurostat snapshot and the frozen deflator snapshot, and writes
``SIGN_RESTRICTIONS_RESULTS.md``. Every number and verdict in the report is computed here.
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
from eu_okun.sign_restrictions import (
    FEVD_HORIZON,
    HORIZONS,
    PRIMARY,
    RATIO_HORIZONS,
    SHOCK_LABELS,
    SHOCKS,
    VARIABLES,
    VARIANTS,
    SpecResults,
    Summary,
    VarData,
    audit_deflator,
    estimate,
    load_deflator,
    recursive_orderings,
    var_data,
)

SIGNS = {1: "+", -1: "−"}


@dataclass
class SignReview:
    """Everything the report is generated from."""

    results: dict[str, SpecResults]
    orderings: list[dict[str, Any]]
    provenance: dict[str, Any] = field(default_factory=dict)


def review_from_data(data: VarData, draws: int = 1000, rotations: int = 20000,
                     check_draws: int = 100,
                     provenance: dict[str, Any] | None = None) -> SignReview:
    """Every registered specification and the 24 recursive orderings."""
    results = {PRIMARY.key: estimate(data, PRIMARY, draws, rotations,
                                     check_draws=check_draws)}
    for spec in VARIANTS:
        results[spec.key] = estimate(data, spec, draws, rotations)
    return SignReview(results, recursive_orderings(data, PRIMARY, draws), provenance or {})


def build_review(eurostat_dir: Path, deflator_dir: Path) -> SignReview:
    """Audit both snapshots and run the registered estimation."""
    audit = audit_snapshot(eurostat_dir)
    manifest = audit_deflator(deflator_dir)
    panel = _read_panel(eurostat_dir / "processed" / "eu27_quarterly.csv")
    data = var_data(panel, load_deflator(deflator_dir), "2008-Q1",
                    str(audit["analysis_period"]["end"]))
    provenance = {"eurostat_manifest_sha256": audit["snapshot_manifest_sha256"],
                  "deflator": {k: manifest[k] for k in ("url", "updated", "raw_sha256",
                                                        "processed_sha256",
                                                        "downloaded_at_utc")}}
    return review_from_data(data, provenance=provenance)


# ----------------------------------------------------------------------------------
# Formatting and verdicts
# ----------------------------------------------------------------------------------


def _num(value: float, digits: int = 3) -> str:
    if value == math.inf:
        return "+inf"
    if value == -math.inf:
        return "-inf"
    return f"{value:+.{digits}f}"


def region(s: Summary, digits: int = 3) -> str:
    return f"[{_num(s.region_lower, digits)}, {_num(s.region_upper, digits)}]"


def means(s: Summary, digits: int = 3) -> str:
    return f"[{_num(s.mean_lower, digits)}, {_num(s.mean_upper, digits)}]"


def single(s: Summary, digits: int = 3) -> str:
    return (f"{_num(s.single_median, digits)} [{_num(s.single_low90, digits)}, "
            f"{_num(s.single_high90, digits)}]")


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]], align: str = "") -> list[str]:
    align = align or "l" + "r" * (len(header) - 1)
    marks = ["---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(marks) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def q1_verdict(res: SpecResults, h: int) -> str:
    """Protocol, section 6: both regions bounded and not overlapping."""
    d, t = res.summary(f"demand:okun:{h}"), res.summary(f"technology:okun:{h}")
    if not (d.bounded and t.bounded):
        missing = [SHOCK_LABELS[k].lower() for k, s in (("demand", d), ("technology", t))
                   if not s.bounded]
        return f"not distinguishable ({' and '.join(missing)} region unbounded)"
    overlap = d.region_lower <= t.region_upper and t.region_lower <= d.region_upper
    return "not distinguishable (regions overlap)" if overlap else "distinguishable under S1-S3"


def persists(res: SpecResults, shock: str, h: int = 8) -> bool:
    """The unemployment response keeps its impact sign with the region excluding zero."""
    sign = res.spec.restrictions[shock].get(2, 0)
    s = res.summary(f"{shock}:unemployment:{h}")
    return (sign > 0 and s.region_lower > 0) or (sign < 0 and s.region_upper < 0)


def q2_text(res: SpecResults) -> str:
    d, ls = persists(res, "demand"), persists(res, "labour_supply")
    if d and ls:
        return "the distinction persists at h = 8 for both shocks"
    lost = [SHOCK_LABELS[k].lower() for k, ok in (("demand", d), ("labour_supply", ls))
            if not ok]
    return f"the distinction does not persist at h = 8 ({' and '.join(lost)} region includes zero)"


def conditional(res: SpecResults, text: str) -> str:
    """Protocol, section 6: verdicts of a specification the data mostly contradict."""
    if res.nonempty_share >= 0.5:
        return text
    return (f"{text} (conditional on the {100 * res.nonempty_share:.1f}% of draws that admit "
            "the restrictions)")


def plausibility(res: SpecResults) -> str:
    share = res.nonempty_share
    return (f"{100 * share:.1f}% of draws admit the restrictions"
            + (" (the data contradict them for most reduced forms)" if share < 0.5 else ""))


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------


def render(review: SignReview, assets: str, command: str) -> str:
    """Assemble the Markdown report."""
    p = review.results[PRIMARY.key]
    lines = [
        "# Sign-restricted structural VAR: demand, technology and labour-supply shocks", "",
        "Generated by `eu-okun sign-restrictions` from the frozen Eurostat snapshot and the "
        "frozen GDP-deflator snapshot. What is estimated and how it is judged was "
        "pre-registered in [SIGN_RESTRICTIONS_PROTOCOL.md](../SIGN_RESTRICTIONS_PROTOCOL.md); "
        "why this strategy was previously not estimated is argued in "
        "[IDENTIFICATION_STUDY.md](IDENTIFICATION_STUDY.md), section 4.3.", "",
        "**Scope.** Shocks are **set-identified** by impact sign restrictions (S1-S3). Every "
        "result below is the identified set: the range over all rotations consistent with "
        "the restrictions, summarised without a prior over rotations (Giacomini and "
        "Kitagawa 2021). Results are labelled \"set-identified under S1-S3\" and are causal "
        "only under those untestable restrictions. Recursive orderings are reported only as "
        "**orthogonalised, not causal** responses.", "",
        "## Summary", "",
        f"- **Plausibility.** {plausibility(p)}; on average {p.mean_admissible:.0f} admissible "
        f"rotations (with their assignments) per draw out of {p.rotations}.",
    ]
    for h in RATIO_HORIZONS:
        d, t = p.summary(f"demand:okun:{h}"), p.summary(f"technology:okun:{h}")
        lines.append(f"- **Demand against technology, h = {h}.** Okun ratio robust 90% region: "
                     f"demand {region(d)}, technology {region(t)}: {q1_verdict(p, h)}.")
    shares = "; ".join(f"{SHOCK_LABELS[k].lower()} {region(p.summary(f'{k}:fevd_du:8'), 2)}"
                       for k in SHOCKS)
    lines += [
        f"- **Labour supply against demand.** Imposed on impact; {q2_text(p)}. Share of "
        f"unemployment's forecast-error variance at h = {FEVD_HORIZON} (robust region): "
        f"{shares}.",
    ]
    finite = [r for r in review.orderings if all(math.isfinite(r[f"okun_h{h}"])
                                                 for h in RATIO_HORIZONS)]
    for h in RATIO_HORIZONS:
        vals = [r[f"okun_h{h}"] for r in finite]
        lines.append(f"- **Recursive orderings (orthogonalised, not causal), h = {h}.** The "
                     f"posterior-median Okun ratio of the \"GDP shock\" ranges from "
                     f"{min(vals):+.3f} to {max(vals):+.3f} across the 24 orderings.")
    if p.convergence:
        lines.append(f"- **Inner approximation.** Doubling the rotations on the first draws "
                     f"widens the sets by {100 * p.convergence['median_width_increase']:.1f}% "
                     f"(median), at most {100 * p.convergence['max_width_increase']:.1f}%.")
    lines += [""]

    # 1. Design
    lines += ["## 1. Design", "",
              f"Pooled panel VAR({p.spec.lags}) in (g, pi, du, dp) with country fixed effects; "
              f"{p.n} country-quarters from {p.countries} countries; windows touching "
              f"2020-2021 excluded. Flat-prior Normal-inverse-Wishart posterior: {p.draws} "
              f"stable draws ({p.discarded} explosive draws discarded); {p.rotations} Haar "
              "rotations per draw. One-standard-deviation shocks; GDP, unemployment and "
              "participation responses are cumulative (levels, % and pp).", ""]
    rows = []
    for shock in SHOCKS:
        signs = p.spec.restrictions[shock]
        rows.append([SHOCK_LABELS[shock], *(SIGNS.get(signs.get(v, 0), "free")
                                            for v in range(4))])
    lines += _table(["Shock (S1-S3, impact)", *VARIABLES], rows, "lllll")

    # 2. Plausibility
    lines += ["## 2. Plausibility of the restrictions", "",
              "Share of posterior draws for which at least one rotation satisfies every "
              "restriction (the posterior probability that the identified set is non-empty), "
              "and the mean number of admissible rotations per draw.", ""]
    lines += _table(["Specification", "Non-empty", "Admissible per draw", "Draws discarded",
                     "Reading"],
                    [[r.spec.label, f"{100 * r.nonempty_share:.1f}%",
                      f"{r.mean_admissible:.0f}", str(r.discarded),
                      "compatible with the reduced form" if r.nonempty_share >= 0.5 else
                      "contradicted by the data for most reduced forms"]
                     for r in review.results.values()], "lrrrl")

    # 3. Demand vs technology
    lines += ["## 3. Demand against technology", "",
              "Set of posterior means and robust 90% credible region (prior-free over "
              "rotations), and, for contrast, the uniform-rotation posterior median with its "
              "90% band (**prior-dependent**).", ""]
    rows = []
    for shock in ("demand", "technology"):
        for target, label in (("gdp", "GDP (cumulative, %)"),
                              ("unemployment", "Unemployment (pp)")):
            for h in HORIZONS:
                s = p.summary(f"{shock}:{target}:{h}")
                rows.append([SHOCK_LABELS[shock], label, str(h), means(s), region(s),
                             single(s)])
        for h in RATIO_HORIZONS:
            s = p.summary(f"{shock}:okun:{h}")
            rows.append([SHOCK_LABELS[shock], "Okun ratio (pp per %)", str(h), means(s),
                         region(s), single(s)])
    lines += _table(["Shock", "Target", "h", "Set of posterior means", "Robust 90% region",
                     "Uniform prior: median [90%]"], rows, "lllrrr")
    lines += [f"Verdicts: h = {h}: {q1_verdict(p, h)}." for h in RATIO_HORIZONS] + [""]

    # 4. Labour supply vs demand
    lines += ["## 4. Labour supply against demand", "",
              "The impact signs of unemployment differ by assumption. Beyond impact:", ""]
    rows = []
    for shock in ("labour_supply", "demand"):
        for target, label in (("gdp", "GDP (cumulative, %)"),
                              ("unemployment", "Unemployment (pp)"),
                              ("participation", "Participation (pp)")):
            for h in HORIZONS:
                s = p.summary(f"{shock}:{target}:{h}")
                rows.append([SHOCK_LABELS[shock], label, str(h), means(s), region(s),
                             single(s)])
    lines += _table(["Shock", "Target", "h", "Set of posterior means", "Robust 90% region",
                     "Uniform prior: median [90%]"], rows, "lllrrr")
    rows = [[SHOCK_LABELS[k], means(p.summary(f"{k}:fevd_du:8"), 2),
             region(p.summary(f"{k}:fevd_du:8"), 2), single(p.summary(f"{k}:fevd_du:8"), 2)]
            for k in SHOCKS]
    lines += [f"Share of unemployment's forecast-error variance at h = {FEVD_HORIZON}:", ""]
    lines += _table(["Shock", "Set of posterior means", "Robust 90% region",
                     "Uniform prior: median [90%]"], rows, "lrrr")
    lines += [f"Verdict: {q2_text(p)}.", ""]

    # 5. Robustness
    lines += ["## 5. Robustness (pre-registered)", ""]
    rows = []
    for r in review.results.values():
        rows.append([r.spec.label, f"{100 * r.nonempty_share:.1f}%",
                     region(r.summary("demand:okun:8")), region(r.summary("technology:okun:8")),
                     conditional(r, q1_verdict(r, 8)),
                     region(r.summary("labour_supply:unemployment:8")),
                     region(r.summary("demand:unemployment:8")), conditional(r, q2_text(r))])
    lines += _table(["Specification", "Non-empty", "Demand Okun h=8", "Technology Okun h=8",
                     "Question 1 (h = 8)", "Labour supply du h=8", "Demand du h=8",
                     "Question 2"], rows, "lrrrlrrl")

    # 6. Orderings
    lines += ["## 6. Recursive orderings (orthogonalised, not causal)", "",
              "Posterior median of the Okun ratio of the orthogonalised innovation to the g "
              "equation, for each Cholesky ordering. No ordering is defended: the spread "
              "measures how much an arbitrary choice moves the answer.", ""]
    lines += _table(["Ordering", *(f"Okun ratio h = {h}" for h in RATIO_HORIZONS)],
                    [[r["ordering"], *(f"{r[f'okun_h{h}']:+.3f}" for h in RATIO_HORIZONS)]
                     for r in review.orderings], "lrr")
    lines += [f"![Identified sets and orderings]({assets}/sets.png)", ""]

    # 7. Conclusion
    decided = [h for h in RATIO_HORIZONS if q1_verdict(p, h).startswith("distinguishable")]
    lines += ["## 7. What these results do and do not establish", "",
              ("- Demand- and technology-driven Okun ratios are distinguishable under S1-S3 at "
               "h = " + ", ".join(str(h) for h in decided) + "." if decided else
               "- Even under sign restrictions, the data do not distinguish demand- from "
               "technology-driven Okun ratios: the identified sets are too wide."),
              f"- Labour supply against demand: {q2_text(p)}; at impact the difference is "
              "assumed, not learned.",
              "- The uniform-rotation posterior looks sharper than the identified sets because "
              "it adds a prior over rotations; that sharpness is not evidence.",
              "- The recursive orderings give different \"GDP shock\" Okun ratios; none is "
              "causal.",
              "- Everything rests on the homogeneous pooled VAR and on S1-S3, which are not "
              "testable beyond their joint compatibility with the reduced form.", "",
              "## 8. Provenance and reproduction", ""]
    prov = review.provenance
    if prov:
        lines += [f"Eurostat snapshot manifest SHA-256 `{prov['eurostat_manifest_sha256']}`; "
                  f"deflator raw `{prov['deflator']['raw_sha256']}` (`namq_10_gdp`, "
                  f"updated {prov['deflator']['updated']}). Seed and settings are in "
                  "SIGN_RESTRICTIONS_PROTOCOL.md.", ""]
    lines += ["```bash", "poetry install --extras \"bayes report\"", command, "```", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------
# Figure and tables
# ----------------------------------------------------------------------------------


def write_figure(review: SignReview, directory: Path, source: str) -> None:
    """Robust regions of cumulative unemployment responses; ordering spread of the ratio."""
    from eu_okun.review_figures import AQUA, BLUE, INK2, MUTED, ORANGE, SURFACE, _finish, _pyplot

    plt = _pyplot()
    p = review.results[PRIMARY.key]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4),
                                   gridspec_kw={"width_ratios": [1.6, 1.0]})
    colours = {"demand": BLUE, "technology": ORANGE, "labour_supply": AQUA}
    for k, shock in enumerate(SHOCKS):
        for j, h in enumerate(HORIZONS):
            s = p.summary(f"{shock}:unemployment:{h}")
            x = j * 4 + k
            ax1.plot([x, x], [s.region_lower, s.region_upper], color=colours[shock], lw=6,
                     alpha=0.35, solid_capstyle="butt",
                     label=SHOCK_LABELS[shock] if j == 0 else None)
            ax1.plot([x], [s.single_median], "o", ms=5, color=colours[shock], mec=SURFACE)
    ax1.axhline(0, color=MUTED, lw=1)
    ax1.set_xticks([j * 4 + 1 for j in range(len(HORIZONS))])
    ax1.set_xticklabels([f"h = {h}" for h in HORIZONS])
    ax1.set_ylabel("Unemployment rate, cumulative (pp)")
    ax1.set_title("Robust 90% regions (bars); uniform-prior medians (dots)", fontsize=9.5)
    ax1.legend(loc="upper left", fontsize=8)
    for j, h in enumerate(RATIO_HORIZONS):
        vals = [r[f"okun_h{h}"] for r in review.orderings if math.isfinite(r[f"okun_h{h}"])]
        ax2.plot(np.full(len(vals), j), vals, "o", ms=4, color=MUTED, alpha=0.8)
    ax2.set_xticks(range(len(RATIO_HORIZONS)))
    ax2.set_xticklabels([f"h = {h}" for h in RATIO_HORIZONS])
    ax2.set_xlim(-0.6, len(RATIO_HORIZONS) - 0.4)
    ax2.axhline(0, color=MUTED, lw=1)
    ax2.set_title("24 recursive orderings: orthogonalised\n\"GDP shock\" Okun ratio "
                  "(not causal)", fontsize=9.5)
    fig.suptitle("Sign-restricted SVAR, set-identified under S1-S3", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold", color=INK2)
    fig.subplots_adjust(wspace=0.3, top=0.82, bottom=0.12)
    _finish(plt, fig, directory / "sets.png", source, credit_y=-0.04)


def write_tables(review: SignReview, directory: Path) -> None:
    """Summaries per specification and target, specification facts, orderings."""
    directory.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        with (directory / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write("summaries.csv", [{"spec": k, **asdict(s)} for k, r in review.results.items()
                            for s in r.summaries])
    write("specifications.csv", [{"spec": k, "label": r.spec.label, "lags": r.spec.lags,
                                  "n": r.n, "countries": r.countries, "draws": r.draws,
                                  "discarded": r.discarded, "rotations": r.rotations,
                                  "nonempty_share": r.nonempty_share,
                                  "mean_admissible": r.mean_admissible,
                                  "q1_h4": q1_verdict(r, 4), "q1_h8": q1_verdict(r, 8),
                                  "q2": q2_text(r)} for k, r in review.results.items()])
    write("orderings.csv", review.orderings)
    prov = {**review.provenance,
            "convergence": review.results[PRIMARY.key].convergence}
    (directory / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n",
                                               encoding="utf-8", newline="\n")


def write_report(eurostat_dir: Path, deflator_dir: Path, output_dir: Path,
                 report: Path | None = None, figures: bool = True) -> SignReview:
    """Build the review and write tables, figure and the Markdown report."""
    review = build_review(eurostat_dir, deflator_dir)
    report = report or output_dir / "SIGN_RESTRICTIONS_RESULTS.md"
    write_tables(review, output_dir)
    if figures:
        write_figure(review, output_dir, "Source: Eurostat (namq_10_gdp, une_rt_q, "
                     "lfsi_emp_q). Set-identified under SIGN_RESTRICTIONS_PROTOCOL.md S1-S3.")
    assets = Path(os.path.relpath(output_dir, report.parent)).as_posix()
    command = (f"poetry run eu-okun sign-restrictions --data-dir {eurostat_dir.as_posix()} "
               f"--deflator-dir {deflator_dir.as_posix()} --output-dir {output_dir.as_posix()} "
               f"--report {report.as_posix()}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(review, assets, command), encoding="utf-8", newline="\n")
    return review
