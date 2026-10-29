"""Command line: ``iberian-lp download`` fetches the drawn days' files from omie.es;
``iberian-lp data`` builds the derived tables and docs/DATA.md from them;
``iberian-lp programmes`` solves every programme from the derived tables;
``iberian-lp estimates`` computes the estimands and docs/RESULTS.md; and
``iberian-lp publication`` writes the article from them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="iberian-lp")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Fetch the drawn days' files (section 3)")
    download.add_argument("--raw", type=Path, default=Path("data/raw"))
    data = commands.add_parser("data", help="Build data/derived and docs/DATA.md from the "
                               "downloaded files (no programme is solved)")
    data.add_argument("--raw", type=Path, default=Path("data/raw"))
    data.add_argument("--out", type=Path, default=Path("data/derived"))
    data.add_argument("--report", type=Path, default=Path("docs/DATA.md"))
    report = commands.add_parser("data-report", help="Regenerate docs/DATA.md from the "
                                 "derived tables")
    report.add_argument("--out", type=Path, default=Path("data/derived"))
    report.add_argument("--report", type=Path, default=Path("docs/DATA.md"))
    programmes = commands.add_parser("programmes", help="Solve every programme (sections "
                                     "4-7) into results/programmes.csv")
    programmes.add_argument("--derived", type=Path, default=Path("data/derived"))
    programmes.add_argument("--results", type=Path, default=Path("results"))
    estimates = commands.add_parser("estimates", help="Compute the estimands (sections 5-7) "
                                    "into results/estimates.json and docs/RESULTS.md")
    estimates.add_argument("--derived", type=Path, default=Path("data/derived"))
    estimates.add_argument("--results", type=Path, default=Path("results"))
    estimates.add_argument("--report", type=Path, default=Path("docs/RESULTS.md"))
    publication = commands.add_parser("publication", help="Write the article, its values "
                                      "and figures into publication/")
    publication.add_argument("--root", type=Path, default=Path("."))
    publication.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "publication":
        from iberian_lp.publication import write_publication

        for name in write_publication(args.root, figures=not args.no_figures):
            print(f"wrote {name}")
        return

    if args.command == "download":
        from iberian_lp.contract import KINDS
        from iberian_lp.days import draw_days
        from iberian_lp.omie import fetch

        files = []
        for day in draw_days():
            for kind in KINDS:
                files.append(fetch(kind, day, args.raw))
                print(f"{files[-1]['file']}: {files[-1]['bytes']:,} bytes", flush=True)
        (args.raw / "downloads.json").write_text(json.dumps(files, indent=1) + "\n",
                                                 encoding="utf-8", newline="\n")
        return
    if args.command == "programmes":
        from iberian_lp.experiment import run

        checks = run(args.derived, args.results / "programmes.csv",
                     lambda line: print(line, flush=True))
        (args.results / "solver.json").write_text(json.dumps(checks, indent=1) + "\n",
                                                  encoding="utf-8", newline="\n")
        return
    if args.command == "estimates":
        from iberian_lp import estimates as e
        from iberian_lp.report import load

        solver = json.loads((args.results / "solver.json").read_text(encoding="utf-8"))
        results = e.estimate(e.load_programmes(args.results / "programmes.csv"),
                             load(args.derived), solver)
        e.write(results, args.results / "estimates.json")
        command = (f"iberian-lp estimates --derived {args.derived.as_posix()} --results "
                   f"{args.results.as_posix()} --report {args.report.as_posix()}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(e.render(results, command), encoding="utf-8", newline="\n")
        print(f"wrote {args.report}")
        return
    if args.command == "data":
        from iberian_lp.data import build

        build(args.raw, args.out)
    from iberian_lp.report import load, render

    command = (f"iberian-lp data --raw data/raw --out {args.out.as_posix()} --report "
               f"{args.report.as_posix()}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(load(args.out), command), encoding="utf-8", newline="\n")
    print(f"wrote {args.report}")
