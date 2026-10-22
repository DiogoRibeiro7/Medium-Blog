"""Definition breaks and data vintages (``MEASUREMENT_PROTOCOL.md``).

The registered headline inference (``headline_inference``) is repeated on two other sets
of inputs: the rows that the study's definition-break rule keeps (the classical review's
``quality_screened`` sample), and the latest Eurostat release. The statistic, bootstraps
and seeds are the registered ones, so a change in a count comes from the inputs alone.
The revisions between the frozen snapshot and the new release are measured cell by cell.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eu_okun import headline_inference as hi
from eu_okun.contract import EU27
from eu_okun.headline_inference import ALPHA, HeadlineTest
from eu_okun.null_residuals import bh_adjust

SERIES = (("gdp", "gdp_growth_qoq_pct", "gdp_flag"),
          ("unemployment", "unemployment_rate_pct", "unemployment_flag"),
          ("participation", "participation_rate_pct", "participation_flag"))


# ----------------------------------------------------------------------------------
# Headline inference on other inputs
# ----------------------------------------------------------------------------------


def inference(data_dir: Path, sample: str, grid_csv: Path | None = None,
              draws: int = hi.B_MAIN) -> list[HeadlineTest]:
    """The registered headline tests on ``data_dir`` and ``sample``; reproduces the
    committed grid first when ``grid_csv`` is given."""
    heads, _ = hi.headline_designs(data_dir, sample)
    if grid_csv is not None:
        hi.check_published(heads, grid_csv, sample)
    return hi.headline_tests(heads, draws)


def counts(tests: Sequence[HeadlineTest]) -> dict[str, Any]:
    negative = [t.theta < 0 for t in tests]
    bh_sieve = bh_adjust([t.p_sieve for t in tests])
    bh_block = bh_adjust([t.p_block for t in tests])
    return {
        "point_negative": sum(negative),
        "normal": sum(n and t.p_normal <= ALPHA for n, t in zip(negative, tests, strict=True)),
        "normal_holm": sum(n and t.p_normal_holm <= ALPHA
                           for n, t in zip(negative, tests, strict=True)),
        "bootstrap": sum(t.negative for t in tests),
        "bootstrap_holm": sum(t.negative_holm for t in tests),
        "bootstrap_bh": sum(n and a <= ALPHA and b <= ALPHA
                            for n, a, b in zip(negative, bh_sieve, bh_block, strict=True)),
        "bootstrap_countries": [t.geo for t in tests if t.negative],
    }


def classification_changes(base: Sequence[HeadlineTest],
                           other: Sequence[HeadlineTest]) -> dict[str, list[str]]:
    """Countries whose sign or bootstrap significance differs between two runs."""
    b, o = {t.geo: t for t in base}, {t.geo: t for t in other}
    return {
        "sign": [g for g in EU27 if (b[g].theta < 0) != (o[g].theta < 0)],
        "bootstrap": [g for g in EU27 if b[g].negative != o[g].negative],
        "bootstrap_holm": [g for g in EU27 if b[g].negative_holm != o[g].negative_holm],
    }


# ----------------------------------------------------------------------------------
# Revisions between two releases
# ----------------------------------------------------------------------------------


def _panel(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return {(r["geo"], r["quarter"]): r for r in csv.DictReader(stream)}


def revisions(frozen: Path, new: Path) -> list[dict[str, Any]]:
    """Cell-by-cell differences, by series and country, over the frozen cells."""
    old = _panel(frozen / "processed" / "eu27_quarterly.csv")
    cur = _panel(new / "processed" / "eu27_quarterly.csv")
    rows = []
    for name, column, flag in SERIES:
        for geo in EU27:
            cells = [(k, old[k], cur.get(k)) for k in old if k[0] == geo and old[k][column]]
            diffs = [abs(float(c[column]) - float(o[column])) for _, o, c in cells
                     if c is not None and c[column]]
            changed = sum(d > 1e-12 for d in diffs)
            rows.append({
                "series": name, "geo": geo, "cells": len(cells),
                "changed": changed, "share_changed": changed / len(cells) if cells else 0.0,
                "mean_abs_revision": sum(diffs) / len(diffs) if diffs else 0.0,
                "max_abs_revision": max(diffs, default=0.0),
                "flag_changes": sum(c is not None and c[flag] != o[flag] for _, o, c in cells),
                "missing_now": sum(c is None or not c[column] for _, o, c in cells),
            })
    return rows


# ----------------------------------------------------------------------------------
# Study, report and files
# ----------------------------------------------------------------------------------


@dataclass
class Study:
    headline: list[HeadlineTest]
    screened: list[HeadlineTest]
    vintage: list[HeadlineTest]
    revisions: list[dict[str, Any]]
    releases: dict[str, dict[str, str]]


def _releases(directory: Path) -> dict[str, str]:
    manifest = json.loads((directory / "snapshot_manifest.json").read_text(encoding="utf-8"))
    return {name: f"{s['dataset']} {s['updated'][:10]} {s['sha256'][:12]}"
            for name, s in manifest["series"].items()}


def build_study(frozen: Path, new: Path, grid_csv: Path,
                draws: int = hi.B_MAIN) -> Study:
    return Study(inference(frozen, "pandemic_excluded", grid_csv, draws),
                 inference(frozen, "quality_screened", grid_csv, draws),
                 inference(new, "pandemic_excluded", None, draws),
                 revisions(frozen, new),
                 {"frozen": _releases(frozen), "new": _releases(new)})


def summary(st: Study) -> dict[str, Any]:
    gdp = [r for r in st.revisions if r["series"] == "gdp"]
    lfs = [r for r in st.revisions if r["series"] != "gdp"]
    return {
        "counts": {"headline": counts(st.headline), "screened": counts(st.screened),
                   "vintage": counts(st.vintage)},
        "changes": {"screened": classification_changes(st.headline, st.screened),
                    "vintage": classification_changes(st.headline, st.vintage)},
        "revisions": {
            "gdp_cells_changed": sum(r["changed"] for r in gdp),
            "gdp_cells": sum(r["cells"] for r in gdp),
            "gdp_countries_revised": [r["geo"] for r in gdp if r["changed"]],
            "gdp_max_abs": max(r["max_abs_revision"] for r in gdp),
            "lfs_cells_changed": sum(r["changed"] for r in lfs),
            "flag_changes": sum(r["flag_changes"] for r in st.revisions),
        },
        "max_sum_change": {
            "screened": max(abs(a.theta - b.theta) for a, b in zip(st.headline, st.screened,
                                                                    strict=True)),
            "vintage": max(abs(a.theta - b.theta) for a, b in zip(st.headline, st.vintage,
                                                                   strict=True)),
        },
        "spain_france": {t.geo: {"headline": t.theta, "screened": s.theta}
                         for t, s in zip(st.headline, st.screened, strict=True)
                         if t.geo in ("ES", "FR")},
        "releases": st.releases,
    }


def render(st: Study, command: str) -> str:
    s = summary(st)
    c = s["counts"]
    rev = s["revisions"]
    lines = [
        "# Definition breaks and data vintages", "",
        "Generated by `eu-okun measurement` under "
        "[MEASUREMENT_PROTOCOL.md](../MEASUREMENT_PROTOCOL.md), registered before the new "
        "release was downloaded or any estimate below was computed. The registered headline "
        "inference (two-lag sum, pandemic-excluded sample, null-imposed sieve and block "
        f"bootstraps, B = {hi.B_MAIN}, the registered seeds) is repeated on the rows that "
        "the study's definition-break rule keeps, and on the latest Eurostat release. "
        "Associations only.", "",
        "## The definition-break rule", "",
        "One rule (`review.flagged_change`): a labour-force break flag invalidates the change "
        "into the break quarter; a low-reliability or definition-differs flag invalidates "
        "every change touching the flagged quarter (ES and FR carry `d` from 2021-Q1); a GDP "
        "break flag on any regressor invalidates the row. The Bayesian likelihood always "
        "applies it; the classical review applies it in its `quality_screened` sample; the "
        "registered classical headline does not, and stays primary. The Dutch survey was "
        "phased in during 2021 without a flag; every headline row whose window touches "
        "2020-2021 is excluded, so it enters no headline row.", "",
        "## Counts", "",
        "| Count (of 27) | Headline | Under the rule | Latest release |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (("point_negative", "Negative sums"),
                       ("normal", "Below zero, normal reference"),
                       ("normal_holm", "Below zero, normal reference, Holm"),
                       ("bootstrap", "Negative under both bootstraps"),
                       ("bootstrap_holm", "Negative under both bootstraps, Holm"),
                       ("bootstrap_bh", "Negative under both bootstraps, Benjamini-Hochberg")):
        lines.append(f"| {label} | {c['headline'][key]} | {c['screened'][key]} | "
                     f"{c['vintage'][key]} |")
    ch = s["changes"]
    lines += [
        "", "Countries whose classification changes:", "",
        f"- under the rule: sign {', '.join(ch['screened']['sign']) or 'none'}; bootstrap "
        f"{', '.join(ch['screened']['bootstrap']) or 'none'}; bootstrap after Holm "
        f"{', '.join(ch['screened']['bootstrap_holm']) or 'none'};",
        f"- on the latest release: sign {', '.join(ch['vintage']['sign']) or 'none'}; "
        f"bootstrap {', '.join(ch['vintage']['bootstrap']) or 'none'}; bootstrap after Holm "
        f"{', '.join(ch['vintage']['bootstrap_holm']) or 'none'}.", "",
        f"Largest change of a country sum: {s['max_sum_change']['screened']:.3f} under the "
        f"rule, {s['max_sum_change']['vintage']:.3f} on the latest release. Spain: "
        f"{s['spain_france']['ES']['headline']:+.2f} headline, "
        f"{s['spain_france']['ES']['screened']:+.2f} under the rule; France: "
        f"{s['spain_france']['FR']['headline']:+.2f} and "
        f"{s['spain_france']['FR']['screened']:+.2f}.", "",
        "## The latest release", "",
        "| Series | Frozen | Latest |",
        "| --- | --- | --- |",
    ]
    for name in ("gdp", "unemployment", "participation"):
        lines.append(f"| {name} | {st.releases['frozen'][name]} | {st.releases['new'][name]} |")
    lines += [
        "",
        f"- GDP: {rev['gdp_cells_changed']} of {rev['gdp_cells']} cells revised, in "
        f"{len(rev['gdp_countries_revised'])} countries; largest absolute revision "
        f"{rev['gdp_max_abs']:.2f} pp.",
        f"- Unemployment and participation: {rev['lfs_cells_changed']} cells revised; "
        f"flag changes: {rev['flag_changes']}.",
        "- Eurostat publishes no historical vintages of these country series (only of EU and "
        "euro-area GDP), so this check covers only the revisions published since the freeze. "
        "It is not a test against the full revision history.", "",
        "## Every country", "",
        "| Country | Headline sum | Under the rule | Latest release |",
        "| --- | ---: | ---: | ---: |",
    ]
    for h, q, v in zip(st.headline, st.screened, st.vintage, strict=True):
        lines.append(f"| {h.geo} | {h.theta:+.3f} | {q.theta:+.3f} | {v.theta:+.3f} |")
    lines += ["", "## Reproduction", "", "```bash", command, "```", ""]
    return "\n".join(lines)


def _json(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json(v) for v in value]
    return value


def write_report(frozen: Path, new: Path, grid_csv: Path, output_dir: Path,
                 report: Path | None = None) -> Study:
    st = build_study(frozen, new, grid_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        with (output_dir / name).open("w", encoding="utf-8", newline="\n") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def rows(tests: Sequence[HeadlineTest], label: str) -> list[dict[str, Any]]:
        return [{"inputs": label, "geo": t.geo, "theta": t.theta, "se": t.se,
                 "p_normal": t.p_normal, "p_sieve": t.p_sieve, "p_block": t.p_block,
                 "boot_low": t.boot_low, "boot_high": t.boot_high, "negative": t.negative,
                 "negative_holm": t.negative_holm} for t in tests]

    write("breaks.csv", rows(st.headline, "headline") + rows(st.screened, "rule"))
    write("vintage.csv", rows(st.headline, "headline") + rows(st.vintage, "latest release"))
    write("revisions.csv", st.revisions)
    (output_dir / "summary.json").write_text(json.dumps(_json(summary(st)), indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    if report is not None:
        command = (f"poetry run eu-okun measurement --frozen {frozen.as_posix()} --new "
                   f"{new.as_posix()} --grid {grid_csv.as_posix()} --output-dir "
                   f"{output_dir.as_posix()} --report {report.as_posix()}")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render(st, command), encoding="utf-8", newline="\n")
    return st
