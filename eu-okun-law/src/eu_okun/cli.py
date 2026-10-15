"""CLI for downloading and validating EU quarterly macroeconomic series."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from eu_okun.audit import audit_snapshot
from eu_okun.baseline import write_baselines
from eu_okun.comparison import write_comparisons
from eu_okun.dependence import run_country_pairs
from eu_okun.eurostat import EurostatDataError, save_snapshot
from eu_okun.forecasting import write_validation
from eu_okun.regimes import write_regimes
from eu_okun.review_report import write_review
from eu_okun.stability import write_stability


def main(argv: Sequence[str] | None = None) -> int:
    """Run validated Eurostat acquisition, or exit with a useful error."""
    parser = argparse.ArgumentParser(prog="eu-okun")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Download and validate Eurostat indicators")
    fetch.add_argument("--start-quarter", default="2008-Q1")
    fetch.add_argument("--end-quarter", default="2026-Q2")
    fetch.add_argument("--output-dir", type=Path, default=Path("data"))
    audit = commands.add_parser("audit", help="Audit an existing frozen Eurostat snapshot")
    audit.add_argument("--output-dir", type=Path, default=Path("data"))
    fit = commands.add_parser("baseline", help="Fit descriptive Okun OLS/HAC baselines")
    fit.add_argument("--data-dir", type=Path, default=Path("data"))
    fit.add_argument("--output", type=Path, default=Path("results/ols_hac.json"))
    compare = commands.add_parser("compare", help="Common-sample HAC country sensitivity")
    compare.add_argument("--data-dir", type=Path, default=Path("data"))
    compare.add_argument("--output-dir", type=Path, default=Path("results/comparison"))
    regime = commands.add_parser("regimes", help="Estimate economic-regime sensitivity")
    regime.add_argument("--data-dir", type=Path, default=Path("data"))
    regime.add_argument("--output-dir", type=Path, default=Path("results/regimes"))
    infer = commands.add_parser("infer", help="Stability and paired country inference")
    infer.add_argument("--data-dir", type=Path, default=Path("data"))
    infer.add_argument("--output-dir", type=Path, default=Path("results/inference"))
    forecast = commands.add_parser("forecast", help="Expanding-window forecast evaluation")
    forecast.add_argument("--data-dir", type=Path, default=Path("data"))
    forecast.add_argument("--output-dir", type=Path, default=Path("results/forecast"))
    review = commands.add_parser(
        "review", help="Critical review of the classical results (report, tables, figures)"
    )
    review.add_argument("--data-dir", type=Path, default=Path("data"))
    review.add_argument("--output-dir", type=Path, default=Path("results/review"))
    review.add_argument("--report", type=Path, default=None,
                        help="Markdown report path (default: <output-dir>/"
                             "CLASSICAL_RESULTS_REVIEW.md)")
    review.add_argument("--published-dir", type=Path, default=None,
                        help="Archived pipeline outputs to compare with the regenerated ones")
    review.add_argument("--provenance", default="",
                        help="Description of the archived outputs, quoted in the report")
    review.add_argument("--no-figures", action="store_true",
                        help="Skip figures (matplotlib not required)")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        try:
            manifest = save_snapshot(args.output_dir, args.start_quarter, args.end_quarter)
        except (EurostatDataError, ValueError, RuntimeError, OSError) as exc:
            parser.exit(2, f"Eurostat data acquisition failed: {exc}\n")
        print(f"Saved {manifest['country_quarter_rows']} country-quarter rows.")
        print(f"Manifest: {args.output_dir / 'snapshot_manifest.json'}")
        return 0
    if args.command == "audit":
        try:
            report = audit_snapshot(args.output_dir)
            target = args.output_dir / "coverage_audit.json"
            target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except (EurostatDataError, ValueError, OSError, KeyError, TypeError) as exc:
            parser.exit(2, f"Eurostat audit failed: {exc}\n")
        print(f"Audit complete: {target}")
        return 0
    if args.command == "baseline":
        try:
            count = write_baselines(args.data_dir, args.output)
        except (EurostatDataError, ValueError, OSError) as exc:
            parser.exit(2, f"Okun baseline failed: {exc}\n")
        print(f"Saved {count} descriptive regression fits to {args.output}")
        return 0
    if args.command == "compare":
        try:
            count = write_comparisons(args.data_dir, args.output_dir)
        except (EurostatDataError, ValueError, OSError) as exc:
            parser.exit(2, f"Comparative analysis failed: {exc}\n")
        print(f"Saved {count} common-sample comparisons to {args.output_dir}")
        return 0
    if args.command == "regimes":
        try:
            count = write_regimes(args.data_dir, args.output_dir)
        except (EurostatDataError, ValueError, OSError) as exc:
            parser.exit(2, f"Regime sensitivity failed: {exc}\n")
        print(f"Saved {count} regime specifications to {args.output_dir}")
        return 0
    if args.command == "infer":
        try:
            tests = write_stability(args.data_dir, args.output_dir)
            pairs = run_country_pairs(args.data_dir, args.output_dir)
        except (EurostatDataError, ValueError, OSError) as exc:
            parser.exit(2, f"Inference analysis failed: {exc}\n")
        print(f"Saved {tests} stability tests and {pairs} paired comparisons")
        return 0
    if args.command == "forecast":
        try:
            count = write_validation(args.data_dir, args.output_dir)
        except (EurostatDataError, ValueError, OSError) as exc:
            parser.exit(2, f"Forecast evaluation failed: {exc}\n")
        print(f"Saved {count} forecasts to {args.output_dir}")
        return 0
    if args.command == "review":
        try:
            result = write_review(
                args.data_dir, args.output_dir, args.report, args.published_dir,
                args.provenance, figures=not args.no_figures,
            )
        except (EurostatDataError, ValueError, OSError, RuntimeError) as exc:
            parser.exit(2, f"Classical results review failed: {exc}\n")
        failed = [check for check in result.checks if not check.passed]
        print(f"Review written to {args.output_dir}; {len(result.checks)} checks, "
              f"{len(failed)} failed")
        return 1 if failed else 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
