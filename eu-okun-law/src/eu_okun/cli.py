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
    bayes_export = commands.add_parser(
        "bayes-export", help="Write ArviZ NetCDF files (posterior.nc) for saved Bayesian fits")
    bayes_export.add_argument("--fits-dir", type=Path, default=Path("results/bayes"))
    bayes_export.add_argument("--data-dir", type=Path, default=Path("data"))
    bayes_export.add_argument("--specs", default="all",
                              help="Comma-separated specification names, or 'all' (found ones)")
    bayes_recovery = commands.add_parser(
        "bayes-recovery", help="Posterior recovery on synthetic data (expensive; 'bayes' extra)")
    bayes_recovery.add_argument("--data-dir", type=Path, default=Path("data"))
    bayes_recovery.add_argument("--output-dir", type=Path, default=Path("results/bayes-recovery"))
    bayes_recovery.add_argument("--specs", default="primary,time_effects,gaussian_iid")
    bayes_recovery.add_argument("--chains", type=int, default=4)
    bayes_recovery.add_argument("--warmup", type=int, default=1000)
    bayes_recovery.add_argument("--draws", type=int, default=1000)
    bayes_recovery.add_argument("--seed", type=int, default=20261016)
    bayes_checks = commands.add_parser(
        "bayes-checks", help="Quarter effects, LOO, prior predictive and recovery report")
    bayes_checks.add_argument("--fits-dir", type=Path, default=Path("results/bayes"))
    bayes_checks.add_argument("--data-dir", type=Path, default=Path("data"))
    bayes_checks.add_argument("--output-dir", type=Path, default=Path("results/bayes-checks"))
    bayes_checks.add_argument("--report", type=Path, default=None)
    bayes_checks.add_argument("--recovery-dir", type=Path, default=None)
    bayes_checks.add_argument("--no-figures", action="store_true")
    bayes_checks.add_argument("--replication-dir", type=Path, default=None,
                              help="Results of an independent run to compare with")
    bayes_robustness = commands.add_parser(
        "bayes-robustness", help="Robustness of country differences and ranks across fits")
    bayes_robustness.add_argument("--fits-dir", type=Path, default=Path("results/bayes"))
    bayes_robustness.add_argument("--output-dir", type=Path,
                                  default=Path("results/bayes-robustness"))
    bayes_robustness.add_argument("--report", type=Path, default=None)
    bayes_robustness.add_argument("--pairs-csv", type=Path,
                                  default=Path("docs/review/pairs_review.csv"),
                                  help="Classical paired bootstrap of the review")
    bayes_robustness.add_argument("--no-figures", action="store_true")
    bayes_robustness.add_argument("--replication-dir", type=Path, default=None,
                                  help="Robustness tables of an independent run")
    bayes_dynamics = commands.add_parser(
        "bayes-dynamics", help="Amendment 3: AR(1) factor and ARDL fits against the misfits")
    bayes_dynamics.add_argument("--fits-dir", type=Path, default=Path("results/bayes"))
    bayes_dynamics.add_argument("--data-dir", type=Path, default=Path("data"))
    bayes_dynamics.add_argument("--output-dir", type=Path,
                                default=Path("results/bayes-dynamics"))
    bayes_dynamics.add_argument("--report", type=Path, default=None)
    bayes_dynamics.add_argument("--no-figures", action="store_true")
    bayes_dynamics.add_argument("--replication-dir", type=Path, default=None,
                                help="Results (dynamics.json) of an independent run")
    small_sample = commands.add_parser(
        "small-sample", help="Bootstrap and size study of the pre/post stability tests "
        "(SMALL_SAMPLE_PROTOCOL.md)")
    small_sample.add_argument("--data-dir", type=Path, default=Path("data"))
    small_sample.add_argument("--output-dir", type=Path, default=Path("results/small-sample"))
    small_sample.add_argument("--report", type=Path, default=None)
    small_sample.add_argument("--review-dir", type=Path, default=Path("docs/review"),
                              help="Classical review tables (grid.csv, stability_review.csv)")
    small_sample.add_argument("--no-figures", action="store_true")
    regime_recovery = commands.add_parser(
        "bayes-regime-recovery", help="Amendment 4: recovery of a known post-2021 shift on "
        "synthetic data (expensive; 'bayes' extra)")
    regime_recovery.add_argument("--data-dir", type=Path, default=Path("data"))
    regime_recovery.add_argument("--output-dir", type=Path,
                                 default=Path("results/bayes-regime-recovery"))
    bayes_regimes = commands.add_parser(
        "bayes-regimes", help="Amendment 4: regime-specific and time-varying coefficients")
    bayes_regimes.add_argument("--fits-dir", type=Path, default=Path("results/bayes"))
    bayes_regimes.add_argument("--data-dir", type=Path, default=Path("data"))
    bayes_regimes.add_argument("--output-dir", type=Path, default=Path("results/bayes-regimes"))
    bayes_regimes.add_argument("--report", type=Path, default=None)
    bayes_regimes.add_argument("--review-dir", type=Path, default=Path("docs/review"),
                               help="Classical review tables (grid.csv, stability_review.csv)")
    bayes_regimes.add_argument("--recovery-dir", type=Path, default=None)
    bayes_regimes.add_argument("--no-figures", action="store_true")
    bayes_regimes.add_argument("--replication-dir", type=Path, default=None,
                               help="Results (regimes.json) of an independent run")
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
    forecast_fit = commands.add_parser(
        "forecast-bayes-fit", help="Refit the hierarchical Bayesian forecasting models at "
        "every origin of the forecasting framework (expensive; 'bayes' extra)")
    forecast_fit.add_argument("--data-dir", type=Path, default=Path("data"))
    forecast_fit.add_argument("--output-dir", type=Path, default=Path("results/forecast-bayes"))
    forecast_fit.add_argument("--models", default="all",
                              help="Comma-separated Bayesian forecasting models, or 'all'")
    forecast_fit.add_argument("--targets", default=None,
                              help="Comma-separated target quarters (default: every origin)")
    forecast_fit.add_argument("--chains", type=int, default=4)
    forecast_fit.add_argument("--warmup", type=int, default=1000)
    forecast_fit.add_argument("--draws", type=int, default=1000)
    forecast_fit.add_argument("--seed", type=int, default=20261016)
    forecast_fit.add_argument("--retry-warmup", type=int, default=3000,
                              help="Warm-up of the single retry of a failed refit (0: none)")
    forecast_fit.add_argument("--no-resume", action="store_true")
    forecast_report = commands.add_parser(
        "forecast-report", help="Score and compare Bayesian and classical forecasts")
    forecast_report.add_argument("--data-dir", type=Path, default=Path("data"))
    forecast_report.add_argument("--fits-dir", type=Path, default=Path("results/forecast-bayes"))
    forecast_report.add_argument("--output-dir", type=Path,
                                 default=Path("results/forecast-comparison"))
    forecast_report.add_argument("--report", type=Path, default=None)
    forecast_report.add_argument("--no-figures", action="store_true")
    forecast_report.add_argument("--replication-dir", type=Path, default=None,
                                 help="Tables of an independent run to compare with")
    fetch_shocks_cmd = commands.add_parser(
        "fetch-shocks", help="Download the pinned external shock series into a snapshot")
    fetch_shocks_cmd.add_argument("--output-dir", type=Path,
                                  default=Path("snapshots/external-shocks"))
    identification = commands.add_parser(
        "identification", help="Local projections on externally identified shocks")
    identification.add_argument("--data-dir", type=Path, default=Path("data"))
    identification.add_argument("--shocks-dir", type=Path,
                                default=Path("snapshots/external-shocks-2026-10-16"))
    identification.add_argument("--output-dir", type=Path,
                                default=Path("results/identification"))
    identification.add_argument("--report", type=Path, default=None)
    identification.add_argument("--no-figures", action="store_true")
    fetch_monthly = commands.add_parser(
        "fetch-monthly", help="Download monthly unemployment and production into a snapshot")
    fetch_monthly.add_argument("--start-month", default="2000-01")
    fetch_monthly.add_argument("--end-month", default=None,
                               help="Default: the last month with both series for the EU-27")
    fetch_monthly.add_argument("--output-dir", type=Path,
                               default=Path("snapshots/eurostat-monthly"))
    monthly = commands.add_parser(
        "monthly-identification",
        help="Monthly local projections on externally identified shocks")
    monthly.add_argument("--data-dir", type=Path,
                         default=Path("snapshots/eurostat-monthly-2026-10-18"))
    monthly.add_argument("--shocks-dir", type=Path,
                         default=Path("snapshots/external-shocks-2026-10-16"))
    monthly.add_argument("--quarterly-dir", type=Path, default=None,
                         help="Committed quarterly identification tables, for the comparison")
    monthly.add_argument("--output-dir", type=Path,
                         default=Path("results/monthly-identification"))
    monthly.add_argument("--report", type=Path, default=None)
    monthly.add_argument("--no-figures", action="store_true")
    fetch_ss = commands.add_parser(
        "fetch-shift-share", help="Download TiVA exposure and OECD partner GDP into a snapshot")
    fetch_ss.add_argument("--output-dir", type=Path, default=Path("snapshots/shift-share"))
    shift_share_cmd = commands.add_parser(
        "shift-share", help="Feasibility of a foreign-demand shift-share instrument")
    shift_share_cmd.add_argument("--data-dir", type=Path,
                                 default=Path("snapshots/shift-share-2026-10-18"))
    shift_share_cmd.add_argument("--eurostat-dir", type=Path,
                                 default=Path("snapshots/eurostat-2026-10-15"))
    shift_share_cmd.add_argument("--output-dir", type=Path, default=Path("results/shift-share"))
    shift_share_cmd.add_argument("--report", type=Path, default=None)
    shift_share_cmd.add_argument("--no-figures", action="store_true")
    fetch_deflator_cmd = commands.add_parser(
        "fetch-deflator", help="Download the quarterly GDP deflator growth into a snapshot")
    fetch_deflator_cmd.add_argument("--output-dir", type=Path,
                                    default=Path("snapshots/eurostat-deflator"))
    sign = commands.add_parser(
        "sign-restrictions", help="Sign-restricted SVAR with robust Bayesian identified sets")
    sign.add_argument("--data-dir", type=Path, default=Path("snapshots/eurostat-2026-10-15"))
    sign.add_argument("--deflator-dir", type=Path,
                      default=Path("snapshots/eurostat-deflator-2026-10-18"))
    sign.add_argument("--output-dir", type=Path, default=Path("results/sign-restrictions"))
    sign.add_argument("--report", type=Path, default=None)
    sign.add_argument("--no-figures", action="store_true")
    fetch_het = commands.add_parser(
        "fetch-heterogeneity", help="Download the case-study series into a snapshot")
    fetch_het.add_argument("--output-dir", type=Path, default=Path("snapshots/heterogeneity"))
    het = commands.add_parser(
        "heterogeneity", help="Case study of Portugal, Spain, the Netherlands and Luxembourg")
    het.add_argument("--data-dir", type=Path, default=Path("snapshots/eurostat-2026-10-15"))
    het.add_argument("--snapshot", type=Path,
                     default=Path("snapshots/heterogeneity-2026-10-18"))
    het.add_argument("--review-dir", type=Path, default=Path("docs/review"))
    het.add_argument("--robustness-dir", type=Path, default=Path("docs/bayes-robustness"))
    het.add_argument("--output-dir", type=Path, default=Path("results/heterogeneity"))
    het.add_argument("--report", type=Path, default=None)
    het.add_argument("--no-figures", action="store_true")
    cgrid = commands.add_parser(
        "core-grid", help="The core robustness grid without the participation control "
                          "(CORE_GRID_PROTOCOL.md)")
    cgrid.add_argument("--grid", type=Path, default=Path("docs/review/grid.csv"))
    cgrid.add_argument("--robustness", type=Path, default=Path("docs/review/robustness.csv"))
    cgrid.add_argument("--output-dir", type=Path, default=Path("results/core-grid"))
    cgrid.add_argument("--report", type=Path, default=None)
    nres = commands.add_parser(
        "null-residuals", help="Null-imposed bootstraps with errors from the unrestricted "
                               "residuals (post hoc, not registered)")
    nres.add_argument("--data-dir", type=Path, default=Path("snapshots/eurostat-2026-10-15"))
    nres.add_argument("--headline-dir", type=Path, default=Path("docs/headline-inference"))
    nres.add_argument("--small-sample-dir", type=Path, default=Path("docs/small-sample"))
    nres.add_argument("--output-dir", type=Path, default=Path("results/null-residuals"))
    nres.add_argument("--report", type=Path, default=None)
    hrs = commands.add_parser(
        "hours", help="Okun's law in hours, for every country, with and without the pandemic "
                      "(HOURS_PROTOCOL.md)")
    hrs.add_argument("--data-dir", type=Path, default=Path("snapshots/eurostat-2026-10-15"))
    hrs.add_argument("--snapshot", type=Path,
                     default=Path("snapshots/heterogeneity-2026-10-18"))
    hrs.add_argument("--case-study", type=Path,
                     default=Path("docs/heterogeneity/elasticities.csv"))
    hrs.add_argument("--output-dir", type=Path, default=Path("results/hours"))
    hrs.add_argument("--report", type=Path, default=None)
    smooth = commands.add_parser(
        "smoothing", help="Does the quarterly unemployment rate inherit monthly smoothing? "
                          "(SMOOTHING_PROTOCOL.md)")
    smooth.add_argument("--data-dir", type=Path, default=Path("snapshots/eurostat-2026-10-15"))
    smooth.add_argument("--monthly-dir", type=Path,
                        default=Path("snapshots/eurostat-monthly-2026-10-18"))
    smooth.add_argument("--review-dir", type=Path, default=Path("docs/review"))
    smooth.add_argument("--bayes-dir", type=Path, default=Path("docs/bayes"))
    smooth.add_argument("--output-dir", type=Path, default=Path("results/smoothing"))
    smooth.add_argument("--report", type=Path, default=None)
    fetch_gm = commands.add_parser(
        "fetch-gdp-measurement", help="Download value added and hours by industry into a "
                                      "snapshot (GDP_MEASUREMENT_PROTOCOL.md)")
    fetch_gm.add_argument("--output-dir", type=Path,
                          default=Path("snapshots/gdp-measurement"))
    gm = commands.add_parser(
        "gdp-measurement", help="Measurement noise or finance? (GDP_MEASUREMENT_PROTOCOL.md)")
    gm.add_argument("--data-dir", type=Path, default=Path("snapshots/eurostat-2026-10-15"))
    gm.add_argument("--snapshot", type=Path,
                    default=Path("snapshots/gdp-measurement-2026-10-19"))
    gm.add_argument("--output-dir", type=Path, default=Path("results/gdp-measurement"))
    gm.add_argument("--report", type=Path, default=None)
    headline = commands.add_parser(
        "headline-inference", help="Small-sample inference for the classical headline sums "
                                   "(HEADLINE_INFERENCE_PROTOCOL.md)")
    headline.add_argument("--data-dir", type=Path,
                          default=Path("snapshots/eurostat-2026-10-15"))
    headline.add_argument("--grid", type=Path, default=Path("docs/review/grid.csv"))
    headline.add_argument("--output-dir", type=Path,
                          default=Path("results/headline-inference"))
    headline.add_argument("--report", type=Path, default=None)
    headline.add_argument("--no-size", action="store_true",
                          help="Skip the size study (for a quick check)")
    window = commands.add_parser(
        "window-sensitivity", help="Post-hoc: identification results with the pandemic "
                                   "window widened by one period (not pre-registered)")
    window.add_argument("--data-dir", type=Path, default=Path("snapshots/eurostat-2026-10-15"))
    window.add_argument("--shocks-dir", type=Path,
                        default=Path("snapshots/external-shocks-2026-10-16"))
    window.add_argument("--monthly-dir", type=Path,
                        default=Path("snapshots/eurostat-monthly-2026-10-18"))
    window.add_argument("--deflator-dir", type=Path,
                        default=Path("snapshots/eurostat-deflator-2026-10-18"))
    window.add_argument("--svar-dir", type=Path, default=Path("docs/sign-restrictions"))
    window.add_argument("--shift-share-dir", type=Path,
                        default=Path("snapshots/shift-share-2026-10-18"))
    window.add_argument("--output-dir", type=Path, default=Path("results/window-sensitivity"))
    window.add_argument("--report", type=Path, default=None)
    pub = commands.add_parser(
        "publication", help="Audit the outputs and build the manuscript's values, tables, "
                            "figures, the Medium article and the manifests")
    pub_mode = pub.add_mutually_exclusive_group()
    pub_mode.add_argument("--build", action="store_true",
                          help="Write every generated file (the default)")
    pub_mode.add_argument("--check", action="store_true",
                          help="Fail if a committed generated file differs from the code's output")
    pub_mode.add_argument("--verify-bibliography", action="store_true",
                          help="Resolve every DOI and URL of references.bib (needs internet)")
    pub.add_argument("--root", type=Path, default=Path("."))
    pub.add_argument("--no-figures", action="store_true")
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
    if args.command == "bayes-export":
        from eu_okun.baseline import _read_panel
        from eu_okun.bayes_artifacts import export_fit
        from eu_okun.bayes_model import SPECS, design_for
        from eu_okun.review import read_flags

        names = ([n for n in SPECS if (args.fits_dir / n / "draws.npz").exists()]
                 if args.specs == "all" else args.specs.split(","))
        try:
            audit_snapshot(args.data_dir)
            csv_path = args.data_dir / "processed" / "eu27_quarterly.csv"
            panel, flags = _read_panel(csv_path), read_flags(csv_path)
            for name in names:
                spec = SPECS[name]
                data = design_for(spec, panel, flags)
                print(f"{name}: {export_fit(args.fits_dir / name, spec, data)}")
        except (EurostatDataError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Bayesian export failed: {exc}\n")
        return 0
    if args.command == "bayes-recovery":
        from eu_okun.bayes_checks import run_recovery
        from eu_okun.bayes_fit import SamplerSettings

        settings = SamplerSettings(chains=args.chains, warmup=args.warmup, draws=args.draws,
                                   seed=args.seed)
        try:
            recoveries = run_recovery(args.data_dir, args.output_dir, args.specs.split(","),
                                      settings)
        except (EurostatDataError, ValueError, OSError) as exc:
            parser.exit(2, f"Recovery run failed: {exc}\n")
        for rec in recoveries:
            inside = rec.coverage["theta"]
            print(f"{rec.spec}: {inside[1]}/{inside[2]} country sums inside 90% intervals; "
                  f"{'accepted' if rec.accepted else 'NOT accepted'}")
        return 0
    if args.command == "bayes-checks":
        from eu_okun.bayes_checks import write_checks_report

        try:
            checks = write_checks_report(args.fits_dir, args.data_dir, args.output_dir,
                                         args.report, args.recovery_dir,
                                         figures=not args.no_figures,
                                         replication_dir=args.replication_dir)
        except (EurostatDataError, FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Model checks failed: {exc}\n")
        print(f"Model checks written to {args.report or args.output_dir}: "
              f"{len(checks.comparisons)} LOO comparisons, {len(checks.prior)} prior checks")
        return 0
    if args.command == "bayes-robustness":
        from eu_okun.bayes_robustness import write_robustness_report

        try:
            robustness = write_robustness_report(
                args.fits_dir, args.output_dir, args.report, args.pairs_csv,
                figures=not args.no_figures, replication_dir=args.replication_dir)
        except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Robustness report failed: {exc}" + chr(10))
        print(f"Robustness report written to {args.report or args.output_dir}: "
              f"{int(robustness.surviving_a.sum())} pairs survive in set A")
        return 0
    if args.command == "bayes-dynamics":
        from eu_okun.bayes_dynamics import write_dynamics_report

        try:
            dynamics = write_dynamics_report(
                args.fits_dir, args.data_dir, args.output_dir, args.report,
                figures=not args.no_figures, replication_dir=args.replication_dir)
        except (EurostatDataError, FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Dynamics report failed: {exc}" + chr(10))
        print(f"Dynamics report written to {args.report or args.output_dir}: "
              f"{len(dynamics.comparisons)} LOO comparisons")
        return 0
    if args.command == "small-sample":
        from eu_okun.small_sample import write_report as write_small_sample

        try:
            study = write_small_sample(args.data_dir, args.output_dir, args.report,
                                       args.review_dir, figures=not args.no_figures)
        except (EurostatDataError, FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Small-sample study failed: {exc}" + chr(10))
        print(f"Small-sample report written to {args.report or args.output_dir}: "
              f"{sum(r.survives for r in study.tests)} of {len(study.tests)} changes survive")
        return 0
    if args.command == "bayes-regime-recovery":
        from eu_okun.bayes_regimes import run_regime_recovery

        try:
            recovered = run_regime_recovery(args.data_dir, args.output_dir)
        except (EurostatDataError, FileNotFoundError, ValueError, OSError) as exc:
            parser.exit(2, f"Regime recovery failed: {exc}" + chr(10))
        print(f"regime_shift recovery: {recovered.delta_inside_90}/{recovered.delta_total} "
              f"shifts inside 90%, known average shift "
              f"{'detected' if recovered.detected else 'not detected'}, "
              f"{'accepted' if recovered.accepted else 'NOT accepted'}")
        return 0
    if args.command == "bayes-regimes":
        from eu_okun.bayes_regimes import write_regimes_report

        try:
            regimes = write_regimes_report(
                args.fits_dir, args.data_dir, args.output_dir, args.review_dir, args.report,
                args.recovery_dir, figures=not args.no_figures,
                replication_dir=args.replication_dir)
        except (EurostatDataError, FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Regimes report failed: {exc}" + chr(10))
        print(f"Regimes report written to {args.report or args.output_dir}: "
              f"{len(regimes.comparisons)} LOO comparisons")
        return 0
    if args.command == "forecast-bayes-fit":
        from eu_okun.bayes_fit import SamplerSettings
        from eu_okun.forecast_bayes import FORECAST_SPECS, run_forecast_refits

        models = list(FORECAST_SPECS) if args.models == "all" else args.models.split(",")
        settings = SamplerSettings(chains=args.chains, warmup=args.warmup, draws=args.draws,
                                   seed=args.seed)
        try:
            records = run_forecast_refits(
                args.data_dir, args.output_dir, models, settings,
                retry_warmup=args.retry_warmup or None,
                targets=None if args.targets is None else args.targets.split(","),
                resume=not args.no_resume)
        except (EurostatDataError, ValueError, OSError) as exc:
            parser.exit(2, f"Forecast refits failed: {exc}" + chr(10))
        failed_refits = [f"{r['model']}/{r['target']}" for r in records if not r["accepted"]]
        print(f"{len(records)} origins fitted; failed after the retry: "
              f"{', '.join(failed_refits) if failed_refits else 'none'}")
        return 0
    if args.command == "forecast-report":
        from eu_okun.forecast_comparison import write_report as write_forecast_report

        try:
            forecasts = write_forecast_report(args.data_dir, args.fits_dir, args.output_dir,
                                              args.report, figures=not args.no_figures,
                                              replication_dir=args.replication_dir)
        except (EurostatDataError, FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Forecast report failed: {exc}" + chr(10))
        decided_forecasts = [t for t in forecasts.family
                             if t.verdict != "not distinguishable"]
        print(f"Forecast report written to {args.report or args.output_dir}; "
              f"{len(decided_forecasts)} of {len(forecasts.family)} comparisons decided")
        return 0
    if args.command == "fetch-shocks":
        from eu_okun.external_shocks import fetch_shocks

        try:
            manifest = fetch_shocks(args.output_dir)
        except (EurostatDataError, ValueError, OSError) as exc:
            parser.exit(2, f"Shock download failed: {exc}" + chr(10))
        print(f"Saved {len(manifest['files'])} shock files to {args.output_dir}")
        return 0
    if args.command == "identification":
        from eu_okun.structural_report import write_report as write_identification

        try:
            identified = write_identification(args.data_dir, args.shocks_dir, args.output_dir,
                                              args.report, figures=not args.no_figures)
        except (EurostatDataError, FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Identification study failed: {exc}" + chr(10))
        informative = [o for res in identified.primary.values() for o in res.okun
                       if o.informative]
        print(f"Identification results written to {args.report or args.output_dir}; "
              f"{len(informative)} informative shock-conditional Okun sets")
        return 0
    if args.command == "fetch-monthly":
        from eu_okun.monthly_data import save_monthly_snapshot

        try:
            monthly_manifest = save_monthly_snapshot(args.output_dir, args.start_month,
                                                     args.end_month)
        except (EurostatDataError, ValueError, RuntimeError, OSError) as exc:
            parser.exit(2, f"Monthly Eurostat acquisition failed: {exc}" + chr(10))
        period = monthly_manifest["period"]
        print(f"Saved {monthly_manifest['country_month_rows']} country-month rows "
              f"({period['start']} to {period['end']}) to {args.output_dir}")
        return 0
    if args.command == "monthly-identification":
        from eu_okun.monthly_report import write_report as write_monthly

        try:
            monthly_review = write_monthly(args.data_dir, args.shocks_dir, args.output_dir,
                                           args.report, quarterly_dir=args.quarterly_dir,
                                           figures=not args.no_figures)
        except (EurostatDataError, FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Monthly identification failed: {exc}" + chr(10))
        bounded = [o for res in monthly_review.primary.values() for o in res.ratios
                   if o.informative]
        print(f"Monthly identification results written to {args.report or args.output_dir}; "
              f"{len(bounded)} informative ratio sets")
        return 0
    if args.command == "fetch-shift-share":
        from eu_okun.shift_share import fetch_shift_share

        try:
            ss_manifest = fetch_shift_share(args.output_dir)
        except (EurostatDataError, ValueError, RuntimeError, OSError, KeyError) as exc:
            parser.exit(2, f"Shift-share download failed: {exc}" + chr(10))
        print(f"Saved {len(ss_manifest['files'])} OECD responses to {args.output_dir}")
        return 0
    if args.command == "shift-share":
        from eu_okun.shift_share_report import write_report as write_shift_share

        try:
            ss_review = write_shift_share(args.data_dir, args.eurostat_dir, args.output_dir,
                                          args.report, figures=not args.no_figures)
        except (EurostatDataError, FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Shift-share feasibility failed: {exc}" + chr(10))
        central = ss_review.results["P1"].central
        print(f"Shift-share feasibility written to {args.report or args.output_dir}; "
              f"P1 expected first-stage t {central.expected_t:.2f}: {ss_review.decision}")
        return 0
    if args.command == "fetch-deflator":
        from eu_okun.sign_restrictions import fetch_deflator

        try:
            deflator_manifest = fetch_deflator(args.output_dir)
        except (EurostatDataError, ValueError, RuntimeError, OSError) as exc:
            parser.exit(2, f"Deflator download failed: {exc}" + chr(10))
        print(f"Saved the GDP deflator ({deflator_manifest['period']['start']} to "
              f"{deflator_manifest['period']['end']}) to {args.output_dir}")
        return 0
    if args.command == "sign-restrictions":
        from eu_okun.sign_restrictions_report import q1_verdict
        from eu_okun.sign_restrictions_report import write_report as write_sign

        try:
            sign_review = write_sign(args.data_dir, args.deflator_dir, args.output_dir,
                                     args.report, figures=not args.no_figures)
        except (EurostatDataError, FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Sign-restriction study failed: {exc}" + chr(10))
        primary = sign_review.results["primary"]
        print(f"Sign-restriction results written to {args.report or args.output_dir}; "
              f"non-empty {100 * primary.nonempty_share:.1f}%; h = 8: "
              f"{q1_verdict(primary, 8)}")
        return 0
    if args.command == "fetch-heterogeneity":
        from eu_okun.heterogeneity_data import fetch_heterogeneity

        try:
            het_manifest = fetch_heterogeneity(args.output_dir)
        except (EurostatDataError, ValueError, RuntimeError, OSError) as exc:
            parser.exit(2, f"Case-study download failed: {exc}" + chr(10))
        print(f"Saved {len(het_manifest['files'])} responses to {args.output_dir}")
        return 0
    if args.command == "heterogeneity":
        from eu_okun.heterogeneity_report import write_report as write_case_study

        try:
            case = write_case_study(args.data_dir, args.snapshot, args.review_dir,
                                    args.robustness_dir, args.output_dir, args.report,
                                    figures=not args.no_figures)
        except (EurostatDataError, FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Case study failed: {exc}" + chr(10))
        verdicts = [e.verdict for e in case.evaluations]
        print(f"Case study written to {args.report or args.output_dir}; "
              f"{verdicts.count('supported')} supported, "
              f"{verdicts.count('contradicted')} contradicted, "
              f"{verdicts.count('not determined')} not determined")
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
    if args.command == "core-grid":
        from eu_okun.core_grid import summary as core_summary
        from eu_okun.core_grid import write_report as write_core_grid

        try:
            cg_rows = write_core_grid(args.grid, args.robustness, args.output_dir, args.report)
        except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Core-grid re-classification failed: {exc}" + chr(10))
        cs = core_summary(cg_rows)
        print(f"Robust negative without participation: {cs['robust']} "
              f"(published: {cs['robust_published']})")
        return 0
    if args.command == "null-residuals":
        from eu_okun.null_residuals import write_report as write_null_residuals

        try:
            nr = write_null_residuals(args.data_dir, args.headline_dir, args.small_sample_dir,
                                      args.output_dir, args.report)
        except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Null-residuals run failed: {exc}" + chr(10))
        head = nr["headline"]
        print(f"Headline, Holm both schemes: {head['registered']['holm_both']} registered, "
              f"{head['unrestricted']['holm_both']} with unrestricted errors; normal size "
              f"{head['size']['normal_registered']:.3f} -> {head['size']['normal']:.3f}")
        return 0
    if args.command == "hours":
        from eu_okun.hours import write_report as write_hours

        try:
            hours_study = write_hours(args.data_dir, args.snapshot, args.case_study,
                                      args.output_dir, args.report)
        except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Hours study failed: {exc}" + chr(10))
        for t in hours_study.tests:
            print(f"{t.key}: {t.estimate:+.3f} (Holm p {t.holm:.3f}) {t.verdict}")
        return 0
    if args.command == "smoothing":
        from eu_okun.smoothing import write_report as write_smoothing

        try:
            sm_study = write_smoothing(args.data_dir, args.monthly_dir, args.review_dir,
                                       args.bayes_dir, args.output_dir, args.report)
        except (EurostatDataError, FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Smoothing study failed: {exc}" + chr(10))
        print("; ".join(f"{t.key} {t.verdict}" for t in sm_study.primary) + "; averages of "
              f"monthly: {sum(i.average_of_monthly for i in sm_study.inheritance)} of "
              f"{len(sm_study.inheritance)}")
        return 0
    if args.command == "fetch-gdp-measurement":
        from eu_okun.gdp_measurement_data import fetch_gdp_measurement

        try:
            gm_manifest = fetch_gdp_measurement(args.output_dir)
        except (EurostatDataError, ValueError, RuntimeError, OSError) as exc:
            parser.exit(2, f"Download failed: {exc}" + chr(10))
        print(f"Saved {len(gm_manifest['files'])} responses to {args.output_dir}")
        return 0
    if args.command == "gdp-measurement":
        from eu_okun.gdp_measurement import write_report as write_gdp_measurement

        try:
            gm_study = write_gdp_measurement(args.snapshot, args.data_dir, args.output_dir,
                                             args.report)
        except (EurostatDataError, FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"GDP measurement study failed: {exc}" + chr(10))
        verdicts = [r.verdict for r in gm_study.primary]
        print(f"{verdicts.count('supported')} supported, "
              f"{verdicts.count('contradicted')} contradicted, "
              f"{verdicts.count('not determined')} not determined")
        return 0
    if args.command == "headline-inference":
        from eu_okun.headline_inference import write_report as write_headline

        try:
            hr = write_headline(args.data_dir, args.grid, args.output_dir, args.report,
                                size=not args.no_size)
        except (EurostatDataError, FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Headline inference failed: {exc}" + chr(10))
        hs = hr.summary()
        print(f"Negative under both bootstraps: {hs['bootstrap_negative']} of "
              f"{hs['countries']} (normal reference: {hs['normal_negative']}); after Holm "
              f"{hs['bootstrap_negative_holm']} (normal: {hs['normal_negative_holm']})")
        return 0
    if args.command == "window-sensitivity":
        from eu_okun.window_sensitivity import run as run_window_sensitivity

        try:
            rows = run_window_sensitivity(args.data_dir, args.shocks_dir, args.monthly_dir,
                                          args.deflator_dir, args.svar_dir, args.output_dir,
                                          args.report, shift_share_dir=args.shift_share_dir)
        except (EurostatDataError, FileNotFoundError, ValueError, OSError, KeyError) as exc:
            parser.exit(2, f"Window sensitivity failed: {exc}" + chr(10))
        print(f"{len(rows)} rows written to {args.output_dir / 'window_sensitivity.csv'}")
        return 0
    if args.command == "publication":
        from eu_okun import publication
        from eu_okun.publication_values import audit as audit_outputs

        root = args.root
        try:
            if args.verify_bibliography:
                bib = publication.verify_bibliography(
                    root / "publication" / "manuscript" / "references.bib")
                target = root / "publication" / "manifest" / "bibliography.json"
                target.write_text(publication.dump_json(bib), encoding="utf-8", newline="\n")
                problems = publication.bibliography_problems(bib)
                print(f"{len(bib)} references checked; {len(problems)} problems")
                for problem in problems:
                    print(f"  {problem}")
                return 1 if problems else 0
            audit_failed = [c for c in audit_outputs(root) if not c.passed]
            if args.check:
                stale = publication.check(root)
                for rel in stale:
                    print(f"out of date: {rel}")
            else:
                stale = []
                outputs = publication.build(root, figures_too=not args.no_figures)
                print(f"Wrote {len(outputs)} generated files under {root / 'publication'}")
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            parser.exit(2, f"Publication build failed: {exc}\n")
        for failure in audit_failed:
            print(f"audit FAIL: {failure.name}: {failure.detail}")
        print(f"Audit: {len(audit_failed)} failed checks; {len(stale)} stale files")
        return 1 if audit_failed or stale else 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
