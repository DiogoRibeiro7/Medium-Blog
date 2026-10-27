"""Command-line interface: ``eu-wheat <command>``."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from eu_wheat_weather.contract import (
    CALENDAR_FILE,
    CALENDAR_SNAPSHOT,
    CROP_NAMES,
    MAIZE_CALENDAR_FILE,
    DataError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eu-wheat", description="Heat, drought and wheat yields in EU regions.")
    commands = parser.add_subparsers(dest="command", required=True)
    crops = commands.add_parser(
        "fetch-crops", help="Download Eurostat area and production of a crop by NUTS-2 and "
        "NUTS-1 region and freeze them as a snapshot")
    crops.add_argument("--output-dir", type=Path, required=True)
    crops.add_argument("--crop", choices=sorted(CROP_NAMES.values()), default="wheat")
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
    nuts = commands.add_parser(
        "fetch-nuts-changes", help="Download Eurostat's official NUTS change tables "
        "(2006-2021) and freeze them as a snapshot")
    nuts.add_argument("--output-dir", type=Path, required=True)
    regions = commands.add_parser(
        "regions", help="Match every published region code to a NUTS-2021 region "
        "(METHODS.md, section 3.1)")
    regions.add_argument("--crops", type=Path, required=True)
    regions.add_argument("--nuts-changes", type=Path, required=True,
                         help="Snapshot of Eurostat's NUTS change tables")
    regions.add_argument("--boundaries-dir", type=Path, default=Path("data/boundaries"))
    regions.add_argument("--output", type=Path, default=Path("docs/panel/region_map.csv"))
    weather_agg = commands.add_parser(
        "weather", help="Aggregate E-OBS to NUTS-2021 regions by harvest year, window and "
        "weighting (METHODS.md, section 3.2)")
    weather_agg.add_argument("--eobs-dir", type=Path, default=Path("data/eobs"))
    weather_agg.add_argument("--boundaries-dir", type=Path, default=Path("data/boundaries"))
    weather_agg.add_argument("--output", type=Path, default=Path("docs/panel/weather.csv"))
    weather_agg.add_argument("--latitudes", type=Path,
                             default=Path("docs/panel/region_latitude.csv"))
    weather_agg.add_argument("--calendar", type=Path,
                             default=Path(CALENDAR_SNAPSHOT) / CALENDAR_FILE,
                             help="Crop calendar for the windows of Amendment A1")
    weather_agg.add_argument("--maturity", type=Path, default=Path("docs/panel/maturity.csv"))
    maize_weather = commands.add_parser(
        "maize-weather", help="Aggregate E-OBS to NUTS-2021 regions over each region's maize "
        "season, its placebo window and a fixed season (METHODS.md, section 9.1)")
    maize_weather.add_argument("--eobs-dir", type=Path, default=Path("data/eobs"))
    maize_weather.add_argument("--boundaries-dir", type=Path, default=Path("data/boundaries"))
    maize_weather.add_argument("--calendar", type=Path,
                               default=Path(CALENDAR_SNAPSHOT) / MAIZE_CALENDAR_FILE)
    maize_weather.add_argument("--output", type=Path, default=Path("docs/maize/weather.csv"))
    maize_weather.add_argument("--seasons", type=Path, default=Path("docs/maize/season.csv"))
    maize_panel = commands.add_parser(
        "maize-panel", help="Build the grain maize panel and its coverage report "
        "(METHODS.md, section 9.1; no estimate)")
    maize_panel.add_argument("--crops", type=Path, required=True)
    maize_panel.add_argument("--region-map", type=Path,
                             default=Path("docs/maize/region_map.csv"))
    maize_panel.add_argument("--weather", type=Path, default=Path("docs/maize/weather.csv"))
    maize_panel.add_argument("--seasons", type=Path, default=Path("docs/maize/season.csv"))
    maize_panel.add_argument("--output-dir", type=Path, default=Path("docs/maize"))
    maize_panel.add_argument("--report", type=Path, default=Path("docs/MAIZE_DATA.md"))
    calendar = commands.add_parser(
        "fetch-calendar", help="Download the GGCMI crop calendar (rainfed winter wheat) and "
        "freeze it as a snapshot (Amendment A1)")
    calendar.add_argument("--output-dir", type=Path, default=Path(CALENDAR_SNAPSHOT))
    panel = commands.add_parser(
        "panel", help="Build the region-year panel and its coverage report (no estimate)")
    panel.add_argument("--crops", type=Path, required=True)
    panel.add_argument("--region-map", type=Path, default=Path("docs/panel/region_map.csv"))
    panel.add_argument("--weather", type=Path, default=Path("docs/panel/weather.csv"))
    panel.add_argument("--output-dir", type=Path, default=Path("docs/panel"))
    panel.add_argument("--report", type=Path, default=Path("docs/DATA.md"))
    est = commands.add_parser(
        "estimate", help="Fit the registered models and write the results "
        "(METHODS.md, sections 4-6)")
    est.add_argument("--panel", type=Path, default=Path("docs/panel/panel.csv"))
    est.add_argument("--weather", type=Path, default=Path("docs/panel/weather.csv"))
    est.add_argument("--latitudes", type=Path, default=Path("docs/panel/region_latitude.csv"),
                     help="Region centroids, for the post-hoc July check only")
    est.add_argument("--output-dir", type=Path, default=Path("docs/results"))
    est.add_argument("--report", type=Path, default=Path("docs/RESULTS.md"))
    amend = commands.add_parser(
        "amend", help="Compute Amendment A1 (METHODS.md, section 8) and write its report, "
        "beside the registered results")
    amend.add_argument("--panel", type=Path, default=Path("docs/panel/panel.csv"))
    amend.add_argument("--weather", type=Path, default=Path("docs/panel/weather.csv"))
    amend.add_argument("--maturity", type=Path, default=Path("docs/panel/maturity.csv"))
    amend.add_argument("--crops", type=Path, required=True)
    amend.add_argument("--region-map", type=Path, default=Path("docs/panel/region_map.csv"))
    amend.add_argument("--registered", type=Path, default=Path("docs/results/results.json"))
    amend.add_argument("--output-dir", type=Path, default=Path("docs/amendment"))
    amend.add_argument("--report", type=Path, default=Path("docs/AMENDMENT.md"))
    pub = commands.add_parser(
        "publication", help="Write the article's values, text and figures from the "
        "committed results; --check compares instead")
    pub.add_argument("--root", type=Path, default=Path("."))
    pub.add_argument("--check", action="store_true")
    pub.add_argument("--no-figures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch-crops":
            from eu_wheat_weather.crops import fetch_crops, save_snapshot

            code = next(c for c, name in CROP_NAMES.items() if name == args.crop)
            manifest = save_snapshot(args.output_dir, fetch_crops(crop=code), crop=code)
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
            for label, directory, manifest_path in (
                    ("weather", args.weather_dir, args.weather_manifest),
                    ("boundaries", args.boundaries_dir, args.boundaries_manifest)):
                if manifest_path is not None:
                    checked = verify(directory, manifest_path)
                    print(f"{label}: {len(checked)} local files match {manifest_path}")
        elif args.command == "fetch-nuts-changes":
            from eu_wheat_weather.nuts_changes import fetch_nuts_changes

            nuts_manifest = fetch_nuts_changes(args.output_dir)
            print(f"{nuts_manifest['nuts2_rows']} NUTS-2 change rows, "
                  f"{nuts_manifest['pure_recodes']} pure recodes")
        elif args.command == "regions":
            from eu_wheat_weather.nuts_changes import audit_nuts_changes
            from eu_wheat_weather.panel import read_wheat, write_csv
            from eu_wheat_weather.regions import load_all, match_codes, match_records

            matches = match_codes({r.geo for r in read_wheat(args.crops)},
                                  load_all(args.boundaries_dir),
                                  audit_nuts_changes(args.nuts_changes))
            records = match_records(matches)
            write_csv(args.output, records, list(records[0]))
            print(json.dumps(dict(Counter(m.rule for m in matches))))
        elif args.command == "weather":
            from eu_wheat_weather.contract import FIRST_YEAR, LAST_YEAR
            from eu_wheat_weather.regions import load_all, target_regions
            from eu_wheat_weather.weather import (
                build_weather,
                region_latitudes,
                write_latitudes,
                write_maturity,
                write_weather,
            )

            regions = target_regions(load_all(args.boundaries_dir))
            write_latitudes(region_latitudes(regions), args.latitudes)
            rows, maturity = build_weather(args.eobs_dir, regions,
                                           range(FIRST_YEAR, LAST_YEAR + 1), args.calendar)
            write_weather(rows, args.output)
            write_maturity(maturity, args.maturity)
            print(f"{len(rows)} region-season rows written to {args.output}")
        elif args.command == "maize-weather":
            from eu_wheat_weather.contract import LAST_YEAR
            from eu_wheat_weather.maize import FIRST_YEAR as MAIZE_FIRST_YEAR
            from eu_wheat_weather.regions import load_all, target_regions
            from eu_wheat_weather.weather import (
                MAIZE_COLUMNS,
                SEASON_COLUMNS,
                build_maize_weather,
                write_table,
            )

            regions = target_regions(load_all(args.boundaries_dir))
            rows, seasons = build_maize_weather(args.eobs_dir, regions,
                                                range(MAIZE_FIRST_YEAR, LAST_YEAR + 1),
                                                args.calendar)
            write_table(rows, args.output, MAIZE_COLUMNS)
            write_table(seasons, args.seasons, SEASON_COLUMNS)
            print(f"{len(rows)} region-season rows written to {args.output}")
        elif args.command == "maize-panel":
            from eu_wheat_weather.maize import write_panel as write_maize_panel

            command = (f"poetry run eu-wheat maize-panel --crops {args.crops.as_posix()} "
                       f"--region-map {args.region_map.as_posix()} --weather "
                       f"{args.weather.as_posix()} --seasons {args.seasons.as_posix()} "
                       f"--output-dir {args.output_dir.as_posix()} --report "
                       f"{args.report.as_posix()}")
            summary = write_maize_panel(args.crops, args.region_map, args.weather,
                                        args.seasons, args.output_dir, args.report, command)
            print(f"{summary['panel_region_years']} region-years in "
                  f"{summary['panel_regions']} regions; excluded "
                  f"{summary['excluded_by_reason']}")
        elif args.command == "fetch-calendar":
            from eu_wheat_weather.downloads import fetch_calendar

            document = fetch_calendar(args.output_dir)
            print(f"{len(document['files'])} calendar file; manifest in {args.output_dir}")
        elif args.command == "panel":
            from eu_wheat_weather.panel import write_panel

            command = (f"poetry run eu-wheat panel --crops {args.crops.as_posix()} "
                       f"--region-map {args.region_map.as_posix()} --weather "
                       f"{args.weather.as_posix()} --output-dir {args.output_dir.as_posix()} "
                       f"--report {args.report.as_posix()}")
            summary = write_panel(args.crops, args.region_map, args.weather, args.output_dir,
                                  args.report, command)
            print(f"{summary['panel_region_years']} region-years in "
                  f"{summary['panel_regions']} regions; excluded {summary['excluded_by_reason']}")
        elif args.command == "estimate":
            from eu_wheat_weather.analysis import run, write_results

            command = (f"poetry run eu-wheat estimate --panel {args.panel.as_posix()} "
                       f"--weather {args.weather.as_posix()} --latitudes "
                       f"{args.latitudes.as_posix()} --output-dir "
                       f"{args.output_dir.as_posix()} --report {args.report.as_posix()}")
            results = run(args.panel, args.weather, args.latitudes)
            write_results(results, args.output_dir, args.report, command)
            print(json.dumps({k: v["verdict"] for k, v in results["decisions"].items()}))
            print(f"placebo significant under both methods: "
                  f"{results['E5']['significant_both']}")
        elif args.command == "amend":
            from eu_wheat_weather import amendment

            command = (f"poetry run eu-wheat amend --panel {args.panel.as_posix()} --weather "
                       f"{args.weather.as_posix()} --maturity {args.maturity.as_posix()} "
                       f"--crops {args.crops.as_posix()} --region-map "
                       f"{args.region_map.as_posix()} --registered "
                       f"{args.registered.as_posix()} --output-dir "
                       f"{args.output_dir.as_posix()} --report {args.report.as_posix()}")
            results = amendment.run(args.panel, args.weather, args.maturity, args.crops,
                                    args.region_map)
            registered = json.loads(args.registered.read_text(encoding="utf-8"))
            amendment.write_results(results, registered, args.output_dir, args.report, command)
            print(json.dumps({k: v["verdict"] for k, v in results["decisions"].items()}))
            print(f"amended placebo significant under both methods: "
                  f"{results['E5']['significant_both']}")
        elif args.command == "publication":
            from eu_wheat_weather.publication import check, write_publication

            if args.check:
                stale = check(args.root)
                if stale:
                    parser.exit(1, "Out of date: " + ", ".join(stale) + "\n")
                print("The article and its values are current.")
            else:
                written = write_publication(args.root, figures=not args.no_figures)
                print(f"{len(written)} files written")
    except (DataError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"eu-wheat {args.command} failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
