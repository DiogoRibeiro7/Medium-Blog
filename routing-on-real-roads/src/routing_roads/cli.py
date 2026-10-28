"""Command line: ``routing-roads data`` builds the derived tables and docs/DATA.md;
``route`` and ``estimate`` solve and report; ``publication`` writes the article."""

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
    route = commands.add_parser("route", help="Solve every registered instance, plan and seed "
                                "(sections 5 and 7) into results/solves.csv")
    route.add_argument("--derived", type=Path, default=Path("data/derived"))
    route.add_argument("--results", type=Path, default=Path("results"))
    route.add_argument("--workers", type=int, default=4)
    est = commands.add_parser("estimate", help="Compute section 6's estimands from "
                              "results/solves.csv and write docs/RESULTS.md")
    est.add_argument("--derived", type=Path, default=Path("data/derived"))
    est.add_argument("--results", type=Path, default=Path("results"))
    est.add_argument("--report", type=Path, default=Path("docs/RESULTS.md"))
    pub = commands.add_parser("publication", help="Write the article, its values and its "
                              "figures from the committed results (publication/)")
    pub.add_argument("--root", type=Path, default=Path("."))
    pub.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "publication":
        from routing_roads.publication import write_publication

        for name in write_publication(args.root, figures=not args.no_figures):
            print(f"wrote {name}")
        return

    from routing_roads.report import load, render

    if args.command == "route":
        from routing_roads.experiment import run

        print(f"wrote {run(args.derived, args.results, args.workers)}")
        return
    if args.command == "estimate":
        from routing_roads import estimates

        e = estimates.estimate(estimates.load_solves(args.results / "solves.csv"),
                               load(args.derived))
        command = (f"routing-roads estimate --derived {args.derived.as_posix()} --results "
                   f"{args.results.as_posix()} --report {args.report.as_posix()}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        estimates.write(e, args.results, args.report, command)
        print(f"wrote {args.report}")
        return
    if args.command == "data":
        from routing_roads.data import build

        build(args.pbf, args.out, args.work)
    command = (f"routing-roads data --pbf data/raw/{EXTRACT} --out "
               f"{args.out.as_posix()} --report {args.report.as_posix()}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(load(args.out), command), encoding="utf-8", newline="\n")
    print(f"wrote {args.report}")
