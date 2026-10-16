"""Command line: ``hansard-drift <command> [--config config/study.toml]``.

Stage 1 (needs the corpus):   fetch -> build -> sample
Labelling (by hand):          label --annotator NAME   [agreement]
Stage 2 (needs the labels):   estimate -> transfer -> figures -> report -> check
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_CONFIG = "config/study.toml"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="hansard-drift", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download the corpus from the Hugging Face hub")
    p = sub.add_parser("synth", help="write a synthetic corpus with known truth (tests and CI only)")
    p.add_argument("--days-per-era", type=int, default=30)
    sub.add_parser("build", help="roles, sitting order, dictionary tags, naive rates, lexical drift")
    p = sub.add_parser("sample", help="draw the blind validation sample (once)")
    p.add_argument("--force", action="store_true", help="redraw; only before any labelling has started")
    p = sub.add_parser("label", help="label the validation sample in the terminal")
    p.add_argument("--annotator", required=True)
    p = sub.add_parser("autolabel", help="label the sample from the synthetic truth (tests and CI only)")
    p.add_argument("--annotator", default="synthetic")
    sub.add_parser("agreement", help="Cohen's and Fleiss' kappa between annotators, overall and by era")
    sub.add_parser("adjudicate", help="majority vote of the annotator files into labels/labels.csv")
    sub.add_parser("estimate", help="corrected rates, precision and recall by era, decomposition, sides")
    sub.add_parser("transfer", help="dictionary against a learned classifier, within and across eras")
    sub.add_parser("figures", help="draw every figure the available results allow")
    sub.add_parser("report", help="write results/RESULTS.md")
    sub.add_parser("check", help="recompute key numbers by a second route")
    sub.add_parser("stage1", help="build, sample (if not drawn yet), figures, report, check")
    sub.add_parser("stage2", help="estimate, transfer, figures, report, check")

    args = parser.parse_args(argv)
    from .config import load_config

    cfg = load_config(args.config)
    cmd = args.command

    if cmd == "fetch":
        from .data import fetch

        rec = fetch(cfg)
        print(f"downloaded {len(rec['files'])} files at commit {rec['commit']}")
        return 0
    if cmd == "synth":
        from .synth import write_synthetic

        rec = write_synthetic(cfg, days_per_era=args.days_per_era)
        print(f"wrote a synthetic corpus of {rec['rows']} rows")
        return 0
    if cmd == "build":
        return _build(cfg)
    if cmd == "sample":
        return _sample(cfg, args.force)
    if cmd == "label":
        from .labels import annotate

        return annotate(cfg, args.annotator)
    if cmd == "autolabel":
        from .synth import autolabel

        print(f"labelled {autolabel(cfg, args.annotator)} items from the synthetic truth")
        return 0
    if cmd == "agreement":
        from .labels import agreement

        print(agreement(cfg).to_string(index=False))
        return 0
    if cmd == "adjudicate":
        from .labels import adjudicate

        out = adjudicate(cfg)
        print(
            f"{len(out)} items: {int((out['discipline'] == '1').sum())} positive, {int(out['unsure'].sum())} without a majority"
        )
        return 0
    if cmd == "estimate":
        return _estimate(cfg)
    if cmd == "transfer":
        from .transfer import transfer

        print(transfer(cfg).to_string(index=False))
        return 0
    if cmd == "figures":
        return _figures(cfg)
    if cmd == "report":
        from .report import write_report

        print(f"wrote {write_report(cfg)}")
        return 0
    if cmd == "check":
        return _check(cfg)
    if cmd == "stage1":
        rc = _build(cfg)
        if not (cfg.labels_dir / "design.csv").exists():
            rc = rc or _sample(cfg, False)
        else:
            print("validation sample already drawn; kept as it is")
        _figures(cfg)
        from .report import write_report

        write_report(cfg)
        return rc or _check(cfg)
    if cmd == "stage2":
        rc = _estimate(cfg)
        from .estimate import annotator_sensitivity
        from .labels import agreement, annotator_files
        from .report import write_report
        from .transfer import transfer

        if len(annotator_files(cfg)) >= 2:
            agreement(cfg)
            annotator_sensitivity(cfg)
        transfer(cfg)
        _figures(cfg)
        write_report(cfg)
        return rc or _check(cfg)
    parser.error(f"unknown command {cmd}")
    return 2


def _build(cfg) -> int:
    from .build import build

    s = build(cfg)
    print(
        f"{s['rows']:,} rows, {s['member_speeches']:,} Member speeches, {s['chair_speeches']:,} Chair speeches, "
        f"{s['discipline_tagged']:,} tagged as discipline"
    )
    return 0


def _sample(cfg, force: bool) -> int:
    from .sample import draw

    d = draw(cfg, force=force)
    print(f"drew {len(d)} speeches in {d.groupby(['era', 'd_tag', 'side']).ngroups} strata")
    return 0


def _estimate(cfg) -> int:
    from .estimate import estimate

    s = estimate(cfg)
    t = s["totals"]
    print(
        f"{s['labelled_items']} labels from {s['label_source']}; whole period: tagged {t['tagged']:,}, "
        f"estimated {t['interventions']['median']:,.0f} [{t['interventions']['lo']:,.0f}, {t['interventions']['hi']:,.0f}]"
    )
    return 0


def _figures(cfg) -> int:
    from .figures import all_figures

    names = all_figures(cfg)
    print("figures: " + (", ".join(names) if names else "none (no results yet)"))
    return 0


def _check(cfg) -> int:
    from .check import check

    problems = check(cfg)
    for p in problems:
        print(f"CHECK FAILED: {p}", file=sys.stderr)
    if not problems:
        print("all checks passed")
    return 1 if problems else 0
