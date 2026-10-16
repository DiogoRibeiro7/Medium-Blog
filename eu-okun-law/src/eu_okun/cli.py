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
    bayes_fit = commands.add_parser(
        "bayes-fit", help="Fit the pre-registered hierarchical Bayesian specifications "
        "(expensive; needs the 'bayes' extra)")
    bayes_fit.add_argument("--data-dir", type=Path, default=Path("data"))
    bayes_fit.add_argument("--output-dir", type=Path, default=Path("results/bayes"))
    bayes_fit.add_argument("--specs", default="all",
                           help="Comma-separated specification names, or 'all'")
    bayes_fit.add_argument("--chains", type=int, default=4)
    bayes_fit.add_argument("--warmup", type=int, default=1000)
    bayes_fit.add_argument("--draws", type=int, default=1000)
    bayes_fit.add_argument("--target-accept", type=float, default=0.95)
    bayes_fit.add_argument("--seed", type=int, default=20261016)
    bayes_fit.add_argument("--retry-warmup", type=int, default=None,
                           help="Re-fit a failed specification once with this warm-up")
    bayes_report = commands.add_parser(
        "bayes-report", help="Tables, figures and report from saved Bayesian fits")
    bayes_report.add_argument("--fits-dir", type=Path, default=Path("results/bayes"))
    bayes_report.add_argument("--output-dir", type=Path, default=Path("results/bayes-report"))
    bayes_report.add_argument("--report", type=Path, default=None)
    bayes_report.add_argument("--no-figures", action="store_true")
    bayes_report.add_argument("--data-dir", type=Path, default=None,
                              help="Snapshot used by the fits (enables the factor checks)")
    bayes_report.add_argument("--replication-dir", type=Path, default=None,
                              help="Summary tables of an independent run to compare with")
    predictive_fit = commands.add_parser(
        "predictive-fit", help="Expanding-window refits of the Bayesian models for the "
        "predictive comparison (expensive; needs the 'bayes' extra)")
    predictive_fit.add_argument("--data-dir", type=Path, default=Path("data"))
    predictive_fit.add_argument("--output-dir", type=Path, default=Path("results/predictive"))
    predictive_fit.add_argument("--sets", default="all",
                                help="Comma-separated evaluation sets (E1, E2), or 'all'")
    predictive_fit.add_argument("--models", default="all",
                                help="Comma-separated Bayesian models, or 'all'")
    predictive_fit.add_argument("--targets", default=None,
                                help="Comma-separated target quarters (default: every target)")
    predictive_fit.add_argument("--chains", type=int, default=4)
    predictive_fit.add_argument("--warmup", type=int, default=1000)
    predictive_fit.add_argument("--draws", type=int, default=1000)
    predictive_fit.add_argument("--target-accept", type=float, default=0.95)
    predictive_fit.add_argument("--seed", type=int, default=20261016)
    predictive_fit.add_argument("--retry-warmup", type=int, default=3000,
                                help="Warm-up of the single retry of a failed refit (0: none)")
    predictive_fit.add_argument("--no-resume", action="store_true",
                                help="Refit origins that already have saved predictions")
    predictive_report = commands.add_parser(
        "predictive-report", help="Score, test and report the predictive comparison")
    predictive_report.add_argument("--data-dir", type=Path, default=Path("data"))
    predictive_report.add_argument("--fits-dir", type=Path, default=Path("results/predictive"))
    predictive_report.add_argument("--output-dir", type=Path,
                                   default=Path("results/predictive-report"))
    predictive_report.add_argument("--report", type=Path, default=None)
    predictive_report.add_argument("--no-figures", action="store_true")
    predictive_report.add_argument("--replication-dir", type=Path, default=None,
                                   help="Tables of an independent run to compare with")
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
    if args.command == "bayes-fit":
        from eu_okun.bayes_fit import SamplerSettings, run_specifications
        from eu_okun.bayes_model import SPECS

        names = list(SPECS) if args.specs == "all" else args.specs.split(",")
        settings = SamplerSettings(chains=args.chains, warmup=args.warmup, draws=args.draws,
                                   target_accept=args.target_accept, seed=args.seed)
        try:
            results = run_specifications(args.data_dir, args.output_dir, names, settings,
                                         retry_warmup=args.retry_warmup)
        except (EurostatDataError, ValueError, OSError) as exc:
            parser.exit(2, f"Bayesian fitting failed: {exc}\n")
        for name, diagnostics in results:
            status = "accepted" if diagnostics.accepted else "FAILED: " + "; ".join(
                diagnostics.failures())
            print(f"{name}: {status}")
        # Failed convergence is a reported scientific result, not a crash.
        return 0
    if args.command == "bayes-report":
        from eu_okun.bayes_report import write_report

        output = args.output_dir
        try:
            bayes_review = write_report(args.fits_dir, output, args.report,
                                        figures=not args.no_figures, data_dir=args.data_dir,
                                        replication_dir=args.replication_dir)
        except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Bayesian report failed: {exc}\n")
        print(f"Report for {len(bayes_review.fits)} fits written to {args.report or output}")
        return 0
    if args.command == "predictive-fit":
        from eu_okun.bayes_fit import SamplerSettings
        from eu_okun.bayes_predictive import run_predictive_refits
        from eu_okun.predictive import BAYES_MODELS, EVALUATION_SETS

        sets = list(EVALUATION_SETS) if args.sets == "all" else args.sets.split(",")
        models = list(BAYES_MODELS) if args.models == "all" else args.models.split(",")
        settings = SamplerSettings(chains=args.chains, warmup=args.warmup, draws=args.draws,
                                   target_accept=args.target_accept, seed=args.seed)
        try:
            records = run_predictive_refits(
                args.data_dir, args.output_dir, sets, models, settings,
                retry_warmup=args.retry_warmup or None,
                targets=None if args.targets is None else args.targets.split(","),
                resume=not args.no_resume)
        except (EurostatDataError, ValueError, OSError) as exc:
            parser.exit(2, f"Predictive refits failed: {exc}\n")
        failed_origins = [f"{r['evaluation']}/{r['model']}/{r['target']}" for r in records
                          if not r["accepted"]]
        print(f"{len(records)} origins fitted; failed after the retry: "
              f"{', '.join(failed_origins) if failed_origins else 'none'}")
        # Failed convergence is reported in the comparison, not treated as a crash.
        return 0
    if args.command == "predictive-report":
        from eu_okun.predictive_report import write_report as write_predictive_report

        try:
            comparison = write_predictive_report(args.data_dir, args.fits_dir, args.output_dir,
                                                 args.report, figures=not args.no_figures,
                                                 replication_dir=args.replication_dir)
        except (EurostatDataError, FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Predictive report failed: {exc}\n")
        decided = [t for t in comparison.family if t.verdict != "not distinguishable"]
        print(f"Predictive report written to {args.report or args.output_dir}; "
              f"{len(decided)} of {len(comparison.family)} pre-registered comparisons decided")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
