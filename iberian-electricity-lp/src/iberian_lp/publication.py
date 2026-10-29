"""The Medium article: every number comes from the committed tables and results.

``values`` reads ``results/estimates.json``, ``results/programmes.csv`` and
``data/derived`` and formats each quoted number once. ``render`` fills the ``{{key}}``
placeholders of ``publication/article.template.md``; an unknown or unused key is an error.
``write_publication`` writes the values, the article and the figures; ``check`` compares
the committed values and article with what the code produces.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from iberian_lp.contract import (
    DAYS_PER_MONTH,
    E1_THRESHOLD,
    E1_TOLERANCE,
    E3_MW,
    E3_SENSITIVITY_MW,
    FIRST_DAY,
    LAST_DAY,
    MIDDAY_HOURS,
    NON_CONVEX,
    SIMPLE,
    SIMULTANEOUS_LEVEL,
    VOLUME_TOLERANCE_MW,
)
from iberian_lp.estimates import Table, load_programmes
from iberian_lp.experiment import SPANISH
from iberian_lp.report import Derived, load, structure

PLACEHOLDER = re.compile(r"\{\{([a-z0-9_]+)\}\}")
MINUS = "−"
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")
TEMPLATE = "publication/article.template.md"
RELAXATIONS = (("all", "X"), ("blocks", "X_blocks"), ("sco", "X_sco"))


def eur(x: float) -> str:
    """A signed price change in EUR/MWh, with a true minus sign."""
    text = f"{x:+.2f}"
    return text.replace("-", MINUS) if float(text) != 0 else f"{abs(x):.2f}"


def size(x: float) -> str:
    """A price change quoted without its sign: the claims test checks the sign."""
    return f"{abs(x):.2f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def day_text(d: date) -> str:
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def word(n: int) -> str:
    return WORDS[n]


def table(root: Path) -> tuple[Table, Derived]:
    d = load(root / "data" / "derived")
    return Table(load_programmes(root / "results" / "programmes.csv"), d.days), d


def daily(t: Table) -> dict[str, Any]:
    """Per day, over the Spanish or joint zone: the mean δ of each relaxation, and per hour
    the share of periods priced at or below zero (no bootstrap)."""
    spanish = np.isin(t.zone, SPANISH)
    n = np.bincount(t.day[spanish], minlength=len(t.days))
    out: dict[str, Any] = {
        name: np.bincount(t.day[spanish], weights=t.nearest(p)[spanish],
                          minlength=len(t.days)) / n for name, p in RELAXATIONS}
    hours = sorted(set(t.hour[spanish]))
    out["nonpositive_by_hour"] = {
        int(h): float(np.mean(t.price[spanish & (t.hour == h)] <= 0)) for h in hours}
    return out


def _ci(x: Mapping[str, float]) -> dict[str, str]:
    return {"": eur(x["estimate"]), "_low": eur(x["low"]), "_high": eur(x["high"])}


def _put(v: dict[str, str], key: str, x: Mapping[str, float]) -> None:
    v |= {f"{key}{suffix}": text for suffix, text in _ci(x).items()}


def values(root: Path) -> dict[str, str]:
    e = json.loads((root / "results" / "estimates.json").read_text(encoding="utf-8"))
    t, d = table(root)
    e1, e2, e3, e4, desc, ph = (e[k] for k in ("E1", "E2", "E3", "E4", "descriptive",
                                                "post_hoc"))
    s = structure(d)
    sell = {k: sum(float(r[f"offered_sell_{k}"]) for r in d.periods) for k in
            (SIMPLE, *NON_CONVEX)}
    mid, out = ph["nonpositive"]["midday"], ph["nonpositive"]["outside"]
    v: dict[str, str] = {
        "n_days": str(e["days"]), "days_per_month": word(DAYS_PER_MONTH),
        "first_day": day_text(FIRST_DAY), "last_day": day_text(LAST_DAY),
        "n_periods": f"{desc['periods']:,}", "n_zone_periods": f"{desc['zone_periods']:,}",
        "n_programmes": f"{e['solver']['solves']:,}",
        "scipy_version": e["solver"]["versions"]["scipy"],
        "n_split": f"{desc['split']:,}", "split_pct": pct(desc["split"] / desc["periods"]),
        "split_days": str(s["split_days"]),
        "steps_median": f"{desc['steps_median']:,.0f}",
        "steps_summed": f"{desc['steps_summed_median']:,.0f}",
        "nc_offered_pct": pct(sum(sell[k] for k in NON_CONVEX) / sum(sell.values()), 0),
        **{f"acc_{k.lower()}": pct(desc["accepted_share"][k], 0) for k in NON_CONVEX},
        "simultaneous_level": f"{SIMULTANEOUS_LEVEL:.1%}",
        # E1
        "e1_threshold": f"{E1_THRESHOLD:.0%}", "e1_tolerance": f"{E1_TOLERANCE:.2f}",
        "volume_tolerance": f"{VOLUME_TOLERANCE_MW}",
        "wide_pct": pct(desc["wide"]["share"], 0),
        "wide_median": f"{desc['wide']['median_width']:.2f}",
        # E2
        "e2_moved": pct(e2["all"]["moved"]["estimate"], 0),
        "e2_down": f"{e2['all']['down']:,}", "e2_up": f"{e2['all']['up']:,}",
        "e2_drop": size(e2["all"]["mean"]["estimate"]),
        "e2_blocks_up": f"{e2['blocks']['up']:,}", "e2_blocks_down": f"{e2['blocks']['down']:,}",
        # E3
        "e3_mw": f"{E3_MW:,}", "e3_drop": size(e3["mean"]["estimate"]),
        "midday_from": f"{MIDDAY_HOURS[0] - 1}:00", "midday_to": f"{MIDDAY_HOURS[-1]}:00",
        "e3_outside_abs": f"{e3['outside_abs']['estimate']:.2f}",
        "e3_midday_abs": f"{e3['midday_abs']['estimate']:.2f}",
        "nonpos_midday_pct": pct(mid["count"] / mid["periods"], 0),
        "nonpos_other_pct": pct(out["count"] / out["periods"], 0),
        "pos_outside_abs": f"{ph['positive']['outside_abs']['estimate']:.2f}",
        "pos_midday_abs": f"{ph['positive']['midday_abs']['estimate']:.2f}",
        # E4
        "e4_imports_n": f"{e4['imports']['periods']:,}",
        "e4_exports_n": f"{e4['exports']['periods']:,}",
    }
    _put(v, "e2_mean", e2["all"]["mean"])
    _put(v, "e2_sco", e2["sco"]["mean"])
    _put(v, "e2_blocks", e2["blocks"]["mean"])
    _put(v, "e3_mean", e3["mean"])
    _put(v, "e3_g", e3["G"])
    _put(v, "pos_g", ph["positive"]["G"])
    small, large = E3_SENSITIVITY_MW
    v |= {"e3_small_mw": f"{small:,}", "e3_large_mw": f"{large:,}",
          "e3_small_drop": size(e3["sensitivity"][f"{small} MW"]["mean"]["estimate"]),
          "e3_large_drop": size(e3["sensitivity"][f"{large} MW"]["mean"]["estimate"])}
    _put(v, "e4_imports", e4["imports"])
    _put(v, "e4_exports", e4["exports"])
    return v


def render(template: str, values: Mapping[str, str]) -> str:
    used = set(PLACEHOLDER.findall(template))
    unknown = sorted(used - set(values))
    if unknown:
        raise KeyError(f"Unknown placeholders: {', '.join(unknown)}.")
    unused = sorted(set(values) - used)
    if unused:
        raise KeyError(f"Values the article does not use: {', '.join(unused)}.")
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def outputs(root: Path) -> dict[str, str]:
    """The text files the publication step writes, by path relative to ``root``."""
    v = values(root)
    header = ("<!-- Generated by iberian-lp publication from article.template.md and the "
              "committed results. Do not edit. -->\n")
    return {"publication/values.json": json.dumps(v, indent=1, ensure_ascii=False) + "\n",
            "publication/article.md": header + render(
                (root / TEMPLATE).read_text(encoding="utf-8"), v)}


def write_publication(root: Path, figures: bool = True) -> list[str]:
    written = []
    for name, text in outputs(root).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8", newline="\n")
        written.append(name)
    if figures:
        from iberian_lp.figures import draw_all

        e = json.loads((root / "results" / "estimates.json").read_text(encoding="utf-8"))
        t, d = table(root)
        paths = draw_all(root / "data" / "derived", d, t, e, root / "publication" / "figures")
        written += [p.relative_to(root).as_posix() for p in paths]
    return written


def check(root: Path) -> list[str]:
    """Generated text files that differ from what the code produces now."""
    return [name for name, text in outputs(root).items()
            if not (root / name).exists()
            or (root / name).read_text(encoding="utf-8") != text]
