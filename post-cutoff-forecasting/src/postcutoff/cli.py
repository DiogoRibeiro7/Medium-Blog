"""Command line: ``postcutoff <command> [--config config/experiment.toml]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_CONFIG = Path("config/experiment.toml")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="postcutoff", description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="download the load series from Energy-Charts and write the snapshot")
    p = sub.add_parser("synth", help="write a synthetic snapshot (tests and CI only)")
    p.add_argument("--end", default="2026-10-08")
    sub.add_parser("validate-data", help="report missing, masked and anomalous hours; fail on unmasked anomalies")
    p = sub.add_parser("smoke", help="run each model on a few origins and check the output")
    p.add_argument("--models", nargs="*")
    p.add_argument("--n", type=int, default=3)
    p = sub.add_parser("run", help="forecast every scored origin with every model and score it")
    p.add_argument("--models", nargs="*")
    p.add_argument("--force", action="store_true", help="recompute forecasts even if stored ones match")
    p.add_argument("--limit-per-window", type=int, default=None, help="score only the last N origins of each window")
    sub.add_parser("summarize", help="tables and bootstrap intervals from results/scores.csv")
    sub.add_parser("figures", help="draw the figures from results/ and data/")
    p = sub.add_parser("check", help="recompute key numbers by a second route; fail on any mismatch")
    p.add_argument("--no-figures", action="store_true")
    p = sub.add_parser("all", help="run, summarize, figures and check")
    p.add_argument("--models", nargs="*")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit-per-window", type=int, default=None)

    args = parser.parse_args(argv)
    from .config import load_config

    cfg = load_config(args.config)

    if args.command == "fetch":
        from .data import fetch_energy_charts, write_snapshot

        series, record = fetch_energy_charts(cfg)
        manifest = write_snapshot(series, cfg, record)
        print(
            f"wrote {cfg.snapshot}: {manifest['rows']} hours, {manifest['missing_hours']} missing, "
            f"{manifest['first_hour']} to {manifest['last_hour']}"
        )
        return 0

    if args.command == "synth":
        from .synth import write_synthetic

        manifest = write_synthetic(cfg, args.end)
        print(f"wrote synthetic snapshot {cfg.snapshot}: {manifest['rows']} hours")
        return 0

    if args.command == "validate-data":
        return validate_data(cfg)

    if args.command == "smoke":
        from .harness import smoke

        problems = smoke(cfg, args.models, args.n)
        return _report(problems, "smoke test")

    if args.command in ("run", "all"):
        from .harness import run

        run(cfg, args.models, force=args.force, limit_per_window=args.limit_per_window)
        if args.command == "run":
            return 0

    if args.command in ("summarize", "all"):
        from .summarize import summarize

        summarize(cfg)
        print(f"wrote {cfg.results_dir / 'summary.json'} and RESULTS.md")
        if args.command == "summarize":
            return 0

    if args.command in ("figures", "all"):
        from .figures import make_all

        for path in make_all(cfg):
            print(f"wrote {path}")
        if args.command == "figures":
            return 0

    if args.command in ("check", "all"):
        from .check import run_checks

        problems = run_checks(cfg, require_figures=not getattr(args, "no_figures", False))
        return _report(problems, "checks")
    return 0


def validate_data(cfg) -> int:
    from .data import anomalies, load_snapshot, prepare
    from .origins import build_origins

    series = load_snapshot(cfg)
    prep = prepare(series, cfg)
    flagged = anomalies(series, cfg)
    origins = build_origins(prep, cfg)
    print(f"snapshot {series.index.min()} to {series.index.max()}: {series.size} hours")
    print(
        f"missing hours: {int(prep.missing.sum())}; masked hours: {int(prep.masked.sum())}; "
        f"replaced in model inputs: {int(prep.imputed.sum())}"
    )
    print(origins.groupby(["window", "status"]).size().rename("origins").to_string())
    excluded = origins[origins["status"] == "excluded"]
    if not excluded.empty:
        print(excluded.groupby(["window", "reason"]).size().rename("excluded").to_string())
    unmasked = flagged[~flagged["masked"]]
    print(f"anomalous hours: {len(flagged)} ({len(unmasked)} not masked)")
    problems = []
    if not unmasked.empty:
        print(unmasked.to_string(index=False))
        problems.append("unmasked anomalous hours; add a mask to the config or explain them")
    if int(prep.missing.sum()) > cfg.max_missing_hours:
        problems.append(f"{int(prep.missing.sum())} missing hours, more than the {cfg.max_missing_hours} allowed")
    return _report(problems, "data validation")


def _report(problems: list[str], what: str) -> int:
    if problems:
        print(f"{what} failed:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"{what} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
