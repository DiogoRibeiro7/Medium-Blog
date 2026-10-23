"""Command-line interface: ``eu-wheat <command>``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from eu_wheat_weather.contract import DataError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eu-wheat", description="Heat, drought and wheat yields in EU regions.")
    commands = parser.add_subparsers(dest="command", required=True)
    crops = commands.add_parser(
        "fetch-crops", help="Download Eurostat wheat area and production by NUTS-2 and "
        "NUTS-1 region and freeze them as a snapshot")
    crops.add_argument("--output-dir", type=Path, required=True)
    bounds = commands.add_parser(
        "fetch-boundaries", help="Download the GISCO NUTS-2 boundaries, 2006-2024 "
        "(not committed; the manifest is)")
    bounds.add_argument("--output-dir", type=Path, default=Path("data/boundaries"))
    bounds.add_argument("--manifest", type=Path, required=True)
    weather = commands.add_parser(
        "fetch-weather", help="Download E-OBS daily temperature, precipitation and "
        "elevation, about 2.5 GB (not committed; the manifest is)")
    weather.add_argument("--output-dir", type=Path, default=Path("data/eobs"))
    weather.add_argument("--manifest", type=Path, required=True)
    audit = commands.add_parser(
        "audit", help="Check the crop snapshot and any local copies of the large inputs")
    audit.add_argument("--crops", type=Path, required=True)
    audit.add_argument("--weather-dir", type=Path, default=Path("data/eobs"))
    audit.add_argument("--weather-manifest", type=Path, default=None)
    audit.add_argument("--boundaries-dir", type=Path, default=Path("data/boundaries"))
    audit.add_argument("--boundaries-manifest", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch-crops":
            from eu_wheat_weather.crops import fetch_crops, save_snapshot

            manifest = save_snapshot(args.output_dir, fetch_crops())
            print(json.dumps(manifest["summary"]))
        elif args.command == "fetch-boundaries":
            from eu_wheat_weather.downloads import fetch_boundaries

            document = fetch_boundaries(args.output_dir, args.manifest)
            print(f"{len(document['files'])} boundary files; manifest {args.manifest}")
        elif args.command == "fetch-weather":
            from eu_wheat_weather.downloads import fetch_weather

            document = fetch_weather(args.output_dir, args.manifest)
            print(f"{len(document['files'])} E-OBS files; manifest {args.manifest}")
        elif args.command == "audit":
            from eu_wheat_weather.crops import audit_snapshot
            from eu_wheat_weather.downloads import verify

            print(f"crops: {json.dumps(audit_snapshot(args.crops))}")
            for label, directory, manifest in (
                    ("weather", args.weather_dir, args.weather_manifest),
                    ("boundaries", args.boundaries_dir, args.boundaries_manifest)):
                if manifest is not None:
                    checked = verify(directory, manifest)
                    print(f"{label}: {len(checked)} local files match {manifest}")
    except (DataError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"eu-wheat {args.command} failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
