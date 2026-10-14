"""CLI for downloading and validating EU quarterly macroeconomic series."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from eu_okun.eurostat import EurostatDataError, save_snapshot


def main(argv: Sequence[str] | None = None) -> int:
    """Run validated Eurostat acquisition, or exit with a useful error."""
    parser = argparse.ArgumentParser(prog="eu-okun")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Download and validate Eurostat indicators")
    fetch.add_argument("--start-quarter", default="2008-Q1")
    fetch.add_argument("--end-quarter", default="2026-Q2")
    fetch.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    if args.command == "fetch":
        try:
            manifest = save_snapshot(args.output_dir, args.start_quarter, args.end_quarter)
        except (EurostatDataError, ValueError, RuntimeError, OSError) as exc:
            parser.exit(2, f"Eurostat data acquisition failed: {exc}\n")
        print(f"Saved {manifest['country_quarter_rows']} country-quarter rows.")
        print(f"Manifest: {args.output_dir / 'snapshot_manifest.json'}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
