"""Command line: swaptails <command> [options]. Run `swaptails --help` for the list."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

import numpy as np

from .config import load_config


def _countries(cfg, value):
    if not value:
        return list(cfg.countries)
    unknown = set(value) - set(cfg.countries)
    if unknown:
        raise SystemExit(f"unknown countries {sorted(unknown)}")
    return value


def cmd_fetch(cfg, args):
    from .data import fetch_all

    fetch_all(cfg, _countries(cfg, args.countries))


def cmd_synth(cfg, args):
    from .synth import synthesise

    synthesise(cfg, _countries(cfg, args.countries), seed=args.seed)
    print(f"synthetic data written to {cfg.cache_dir}")


def cmd_validate(cfg, args):
    """Missing and anomalous hours in the forecast period, and the reference forecast's coverage."""
    import pandas as pd

    from .data import anomalies, load_prepared

    report, failed = {}, False
    period_start = pd.Timestamp(cfg.forecast_first_day, tz="UTC")
    for k in _countries(cfg, args.countries):
        prep = load_prepared(cfg, k)
        in_period = (prep.local_date >= cfg.forecast_first_day) & (prep.local_date <= cfg.forecast_last_day)
        missing = int((np.isnan(prep.observed) & in_period & ~prep.masked).sum())
        an = anomalies(prep)
        an_period = an[~an["masked"] & (an["timestamp_utc"] >= period_start)]
        tso = None if prep.tso is None else float(np.isfinite(prep.tso[in_period]).mean())
        temp_missing = int(np.isnan(prep.temperature[in_period]).sum())
        report[k] = {
            "missing_hours_in_period": missing,
            "unmasked_anomalies_in_period": int(len(an_period)),
            "anomaly_examples": [str(t) for t in an_period["timestamp_utc"].head(10)],
            "reference_hour_coverage": tso,
            "temperature_missing_in_period": temp_missing,
        }
        if missing > cfg.max_missing_hours:
            failed = True
        print(f"{k}: {missing} missing hours, {len(an_period)} unmasked anomalous hours, reference coverage {tso}")
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "validation.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit(f"more than {cfg.max_missing_hours} missing hours in a country; see results/validation.json")


def cmd_notes(cfg, args):
    from .days import calendars, label_items

    cals = calendars(cfg, _countries(cfg, args.countries))
    items = label_items(cfg, cals, args.years or list(cfg.label_years))
    if args.date:
        items = [i for i in items if i[1] == date.fromisoformat(args.date)]
    print(f"{len(items)} candidate days")
    for country, d, reason, text in items[: args.show]:
        print(f"--- {country} {d} ({reason})\n{text}\n")


def cmd_labels(cfg, args):
    from .days import calendars, label_items
    from .llm import annotate

    if args.llm not in cfg.llms:
        raise SystemExit(f"unknown language model {args.llm}; configured: {sorted(cfg.llms)}")
    cals = calendars(cfg, _countries(cfg, args.countries))
    items = label_items(cfg, cals, args.years or list(cfg.label_years))
    if args.limit:
        items = items[: args.limit]
    annotate(cfg, args.llm, items)


def cmd_forecast(cfg, args):
    from .data import load_prepared
    from .days import calendars, day_table
    from .forecasters import load
    from .harness import run_one

    if args.model not in cfg.forecasters:
        raise SystemExit(f"unknown forecaster {args.model}; configured: {sorted(cfg.forecasters)}")
    spec = cfg.forecasters[args.model]
    variants = args.variants or cfg.variants_for(args.model)
    bad = set(variants) - set(cfg.variants_for(args.model))
    if bad:
        raise SystemExit(f"{args.model} does not take variants {sorted(bad)}")
    countries = _countries(cfg, args.countries)
    cals = calendars(cfg, countries)
    context = min(cfg.context_hours, spec.max_context)
    adapter = load(spec, context=context, horizon=32)
    cfg.forecasts_dir.mkdir(parents=True, exist_ok=True)
    prov = {
        "forecaster": args.model,
        "label": spec.label,
        "bloc": spec.bloc,
        "developer": spec.developer,
        **adapter.provenance(),
    }
    (cfg.forecasts_dir / f"provenance_{args.model}.json").write_text(
        json.dumps(prov, indent=2, default=str) + "\n", encoding="utf-8"
    )
    for k in countries:
        prep = load_prepared(cfg, k)
        table = day_table(cfg, k, cals[k], prep)
        for v in variants:
            if v.startswith("llm:"):
                from .days import coverage

                cov = coverage(table, v[4:])
                if cov < 1 and not args.allow_partial_labels:
                    raise SystemExit(
                        f"{v}: labels cover {cov:.1%} of candidate days in {k}; run `swaptails labels` first"
                    )
            run_one(cfg, adapter, args.model, v, prep, table, force=args.force)


EXTRAS = {
    "timestral": "timestral",
    "t0": "t0",
    "tirex": "tirex",
    "chronos2": "chronos2",
    "timesfm25": "timesfm25",
    "sundial": "sundial",
}


def cmd_plan(cfg, args):
    llms = args.llms.split() or list(cfg.llms)
    forecasters = args.forecasters.split() or list(cfg.forecasters)
    countries = _countries(cfg, args.countries.split())
    bad = (set(llms) - set(cfg.llms)) | (set(forecasters) - set(cfg.forecasters))
    if bad:
        raise SystemExit(f"unknown models {sorted(bad)}")
    # One forecasting job per forecaster and country keeps every job well inside the
    # six-hour limit of a hosted runner.
    plan = {
        "labels": [{"llm": k, "model": cfg.llms[k].model, "year": y} for k in llms for y in cfg.label_years],
        "forecast": [
            {"forecaster": f, "country": c, "extra": EXTRAS.get(cfg.forecasters[f].kind, "")}
            for f in forecasters
            for c in countries
        ],
    }
    print(json.dumps(plan))


def cmd_merge_labels(cfg, args):
    """Combine label files from several runs (one per shard), keeping one row per question."""
    from pathlib import Path

    cfg.labels_dir.mkdir(parents=True, exist_ok=True)
    for llm in cfg.llms:
        rows = {}
        for folder in args.folders:
            for path in sorted(Path(folder).rglob(f"{llm}.jsonl")):
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        r = json.loads(line)
                        rows[(r["country"], r["date"], r["cache_key"])] = line
            for path in sorted(Path(folder).rglob(f"{llm}.provenance.json")):
                (cfg.labels_dir / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        if rows:
            out = cfg.labels_dir / f"{llm}.jsonl"
            out.write_text("\n".join(rows[k] for k in sorted(rows)) + "\n", encoding="utf-8")
            print(f"{llm}: {len(rows)} answers")


def cmd_smoke(cfg, args):
    """Each forecaster on three origins of the first country, without writing anything."""
    from .data import load_prepared
    from .days import calendars, day_table
    from .forecasters import load
    from .harness import make_batch, origins

    key = _countries(cfg, args.countries)[0]
    prep = load_prepared(cfg, key)
    table = day_table(cfg, key, calendars(cfg, [key])[key], prep)
    for name in args.models or list(cfg.forecasters):
        spec = cfg.forecasters[name]
        context = min(cfg.context_hours, spec.max_context)
        items, _ = origins(cfg, prep, cfg.forecast_first_day, cfg.forecast_last_day, context)
        items = items[:3]
        adapter = load(spec, context=context, horizon=32)
        for v in ["none", "flag"] if spec.covariates else ["none"]:
            from .harness import hourly_codes

            codes = hourly_codes(prep, table, v) if v == "flag" else None
            horizon = max(o.horizon for o in items)
            q = adapter.forecast(make_batch(prep, items, context, horizon, v, codes), cfg.quantiles)
            ok = q.shape == (len(items), horizon, len(cfg.quantiles)) and np.isfinite(q).all()
            median = float(np.median(q[0, :, 1]))
            print(f"{name}|{v}: shape {q.shape}, finite {bool(np.isfinite(q).all())}, first-day median {median:.0f} MW")
            if not ok:
                raise SystemExit(f"{name}|{v} failed the smoke test")


def cmd_summarize(cfg, args):
    from .summarize import summarize

    summarize(cfg, args.countries)
    print(f"wrote {cfg.results_dir / 'summary.json'} and RESULTS.md")


def cmd_figures(cfg, args):
    from .figures import draw

    print("figures:", ", ".join(draw(cfg)))


def cmd_check(cfg, args):
    from .check import check

    problems = check(cfg, strict=not args.lenient)
    for p in problems:
        print("PROBLEM:", p)
    if problems:
        raise SystemExit(1)
    print("all checks passed")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="swaptails", description=__doc__)
    parser.add_argument("--config", default=None, help="experiment TOML (default config/experiment.toml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download load, operator forecasts and temperature forecasts")
    p.add_argument("--countries", nargs="*")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("synth", help="write synthetic data in place of the downloads (tests, dry runs)")
    p.add_argument("--countries", nargs="*")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("validate", help="missing and anomalous hours, reference coverage")
    p.add_argument("--countries", nargs="*")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("notes", help="print the calendar notes the language models read")
    p.add_argument("--countries", nargs="*")
    p.add_argument("--years", nargs="*", type=int)
    p.add_argument("--date")
    p.add_argument("--show", type=int, default=5)
    p.set_defaults(func=cmd_notes)

    p = sub.add_parser("labels", help="ask one language model about every candidate day")
    p.add_argument("--llm", required=True)
    p.add_argument("--countries", nargs="*")
    p.add_argument("--years", nargs="*", type=int)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_labels)

    p = sub.add_parser("forecast", help="run one forecaster with its covariate variants")
    p.add_argument("--model", required=True)
    p.add_argument("--variants", nargs="*")
    p.add_argument("--countries", nargs="*")
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow-partial-labels", action="store_true", help="run language-model variants anyway")
    p.add_argument("--skip-incomplete-labels", action="store_true", help="leave those variants out")
    p.set_defaults(func=cmd_forecast)

    p = sub.add_parser("merge-labels", help="combine label files from several shards")
    p.add_argument("folders", nargs="+")
    p.set_defaults(func=cmd_merge_labels)

    p = sub.add_parser("plan", help="print the job matrices for the GitHub Actions run, as JSON")
    p.add_argument("--llms", default="", help="space-separated subset (default: all)")
    p.add_argument("--forecasters", default="", help="space-separated subset (default: all)")
    p.add_argument("--countries", default="", help="space-separated subset (default: all)")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("smoke", help="each forecaster on three origins")
    p.add_argument("--models", nargs="*")
    p.add_argument("--countries", nargs="*")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("summarize", help="scores, intervals and tables")
    p.add_argument("--countries", nargs="*")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("figures", help="draw the figures from the summary")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("check", help="consistency checks on the results")
    p.add_argument("--lenient", action="store_true", help="tolerate missing runs and partial labels")
    p.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    args.func(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
