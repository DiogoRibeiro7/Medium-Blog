"""Command-line entry point.

Examples::

    python -m airline_ops run --config configs/experiment.toml --out results
    python -m airline_ops figures --results results --out figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from airline_ops.config import load_config
from airline_ops.errors import AirlineOpsError
from airline_ops.logging_setup import configure_logging, get_logger

LOG = get_logger("cli")
FOLDER = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = FOLDER.parent / "data_sources" / "openflights-airports.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="airline_ops", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the experiment and write results")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    run.add_argument("--cache", type=Path, default=FOLDER / ".cache")
    run.add_argument("--stages", nargs="*", default=None, help="subset of stages after planning")
    fig = sub.add_parser("figures", help="draw the figures from results")
    fig.add_argument("--results", type=Path, required=True)
    fig.add_argument("--out", type=Path, required=True)
    fig.add_argument("--case", type=Path, default=FOLDER / "configs" / "case_muc_opo.toml")
    summ = sub.add_parser("summary", help="write the text summary of the results")
    summ.add_argument("--results", type=Path, required=True)
    det = sub.add_parser("details", help="analyses added after the results (paired re-evaluations)")
    det.add_argument("--config", type=Path, required=True)
    det.add_argument("--results", type=Path, required=True)
    det.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    det.add_argument("--cache", type=Path, default=FOLDER / ".cache")
    dig = sub.add_parser("digest", help="digest of the inputs that determine the results")
    mode = dig.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", type=Path, help="exit 1 when this stored digest is out of date")
    mode.add_argument("--write", type=Path, help="write the current digest to this file")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the command line; returns a process exit code."""
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        if args.command == "run":
            from airline_ops.experiments.runner import run_experiment

            config = load_config(args.config)
            run_experiment(
                config, args.out, args.contract, args.cache, stages=set(args.stages) if args.stages else None
            )
        elif args.command == "figures":
            from airline_ops.viz.figures import draw_all

            draw_all(args.results, args.out, args.case)
        elif args.command == "summary":
            from airline_ops.experiments.summary import write_summary

            write_summary(args.results)
        elif args.command == "details":
            from airline_ops.experiments.details import run_details

            run_details(load_config(args.config), args.results, args.contract, args.cache)
        elif args.command == "digest":
            from airline_ops.experiments.digest import compute_digest, digest_matches

            if args.check is not None:
                current = digest_matches(FOLDER, args.check)
                LOG.info("inputs digest", extra={"fields": {"matches": current}})
                return 0 if current else 1
            args.write.parent.mkdir(parents=True, exist_ok=True)
            args.write.write_text(compute_digest(FOLDER) + "\n", encoding="utf-8")
    except AirlineOpsError as err:
        LOG.error("run failed", extra={"fields": {"error": str(err), "type": type(err).__name__}})
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
