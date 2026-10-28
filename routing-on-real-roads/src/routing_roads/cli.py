"""Command line: ``routing-roads data`` builds the derived tables and docs/DATA.md."""

from __future__ import annotations

import argparse
from pathlib import Path

from routing_roads.contract import EXTRACT


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="routing-roads")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="Build the networks, points, matrices and days "
                               "from the pinned extract (no routing), and docs/DATA.md")
    data.add_argument("--pbf", type=Path, default=Path("data/raw") / EXTRACT)
    data.add_argument("--work", type=Path, default=Path("data/raw"),
                      help="Where the clipped OSM files go (not committed)")
    data.add_argument("--out", type=Path, default=Path("data/derived"))
    data.add_argument("--report", type=Path, default=Path("docs/DATA.md"))
    report = commands.add_parser("data-report", help="Regenerate docs/DATA.md from the "
                                 "derived tables")
    report.add_argument("--out", type=Path, default=Path("data/derived"))
    report.add_argument("--report", type=Path, default=Path("docs/DATA.md"))
    args = parser.parse_args(argv)

    from routing_roads.report import load, render

    if args.command == "data":
        from routing_roads.data import build

        build(args.pbf, args.out, args.work)
    command = (f"routing-roads data --pbf data/raw/{EXTRACT} --out "
               f"{args.out.as_posix()} --report {args.report.as_posix()}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(load(args.out), command), encoding="utf-8", newline="\n")
    print(f"wrote {args.report}")
