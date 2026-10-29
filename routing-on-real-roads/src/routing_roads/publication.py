"""The Medium article: every number comes from the committed tables, solves and estimates.

``values`` reads ``results/estimates.json``, ``results/solves.csv`` and ``data/derived`` and
formats each quoted number once. ``render`` fills the ``{{key}}`` placeholders of
``publication/article.template.md``; an unknown or unused key is an error.
``write_publication`` writes the values, the article and the figures; ``check`` compares
the committed values and article with what the code produces.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from routing_roads.contract import (
    BUFFER_M,
    CAPACITY,
    DAYS,
    EXTRACT,
    ITERATIONS,
    PAIRS,
    SEEDS,
    SENSITIVITY_DAYS,
    SENSITIVITY_SEEDS,
    SIMULTANEOUS_LEVEL,
)
from routing_roads.estimates import (
    LENGTH_BANDS_M,
    costs,
    load_solves,
    noise_by_day,
    regret_by_day,
)
from routing_roads.experiment import AREAS, VARIANTS
from routing_roads.report import area_indices, load, matrix_summary, orders

PLACEHOLDER = re.compile(r"\{\{([a-z0-9_]+)\}\}")
MINUS = "−"
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")
TEMPLATE = "publication/article.template.md"
# "About as much as other legs of the same length": within 0.03 of the ratio of pairs of
# that length (docs/RESULTS.md, after the critical review).
LEGS_TYPICAL = 0.03


def pct(x: float, digits: int = 1) -> str:
    """A share as an unsigned percentage, with a true minus sign when negative."""
    text = f"{abs(100 * x):.{digits}f}"
    return (MINUS if x < 0 and float(text) != 0 else "") + text + "%"


def short(x: float) -> str:
    """A shortfall, quoted without its sign: the claims test checks that it is negative."""
    return pct(abs(x))


def points(x: float) -> str:
    """A change in percentage points, signed."""
    text = f"{100 * x:+.1f}"
    return text.replace("-", MINUS)



def floor_pct(x: float) -> str:
    """Rounded down, so that "never below" stays true of the rounded number."""
    return f"{math.floor(1000 * x) / 10:.1f}%"


def ceil_pct(x: float) -> str:
    """Rounded up, so that "never above" stays true of the rounded number."""
    return f"{math.ceil(1000 * x) / 10:.1f}%"


def word(n: int) -> str:
    return WORDS[n]


def _area(e: Mapping[str, Any], pair: str, area: str) -> Mapping[str, Any]:
    out: Mapping[str, Any] = e["areas"][f"{pair}_{area}"]
    return out


def _cells(rows: Sequence[dict[str, str]], pair: str, area: str) -> dict[str, Any]:
    return {f"{plan}_{kind}": costs(rows, "main", pair, area, plan, kind, DAYS, SEEDS)
            for plan in ("straight", "road") for kind in ("road", "straight")}


def daily(rows: Sequence[dict[str, str]], e: Mapping[str, Any]) -> dict[str, Any]:
    """Per area, the day-level series behind E1, E2 and E4 (no bootstrap)."""
    out = {}
    for pair in PAIRS:
        for area in AREAS:
            c = _cells(rows, pair.key, area)
            alpha = _area(e, pair.key, area)["alpha"]
            straight_mean = c["straight_straight"].mean(axis=1)
            out[f"{pair.key}_{area}"] = {
                "regret": regret_by_day(c["straight_road"], c["road_road"]),
                "noise": noise_by_day(c["road_road"]),
                "e_own": alpha * straight_mean / c["straight_road"].mean(axis=1) - 1,
                "e_best": alpha * straight_mean / c["road_road"].mean(axis=1) - 1,
                "kappa_straight": c["straight_road"].mean(axis=1) / straight_mean}
    return out


def _range(xs: Sequence[float], fmt: Any = pct) -> tuple[str, str]:
    return fmt(min(xs)), fmt(max(xs))


def values(root: Path) -> dict[str, str]:
    e = json.loads((root / "results" / "estimates.json").read_text(encoding="utf-8"))
    rows = load_solves(root / "results" / "solves.csv")
    d = load(root / "data" / "derived")
    days = daily(rows, e)
    keys = [(p.key, a) for p in PAIRS for a in AREAS]
    areas = [_area(e, *k) for k in keys]
    ph = e["post_hoc"]
    m = d.manifest
    lisbon, porto = (p.key for p in PAIRS)
    v: dict[str, str] = {
        "n_days": str(DAYS), "capacity": str(CAPACITY),
        "iterations": f"{ITERATIONS:,}", "n_seeds": word(len(SEEDS)),
        "buffer_km": f"{BUFFER_M / 1000:.0f}", "extract_file": EXTRACT,
        "extract_date": _extract_date(EXTRACT),
        "pyvrp_version": m["versions"]["pyvrp"], "osmnx_version": m["versions"]["osmnx"],
        "simultaneous_level": f"{SIMULTANEOUS_LEVEL:.2%}",
        "porto_bridges": word(sum(1 for s in m["pairs"][porto]["crossings"]["bridges"].values()
                                  if s >= 0.01)),
        "porto_bridge_share": pct(sum(s for s in m["pairs"][porto]["crossings"]["bridges"]
                                      .values() if s >= 0.01)),
    }
    v["e1_min"], v["e1_max"] = _range([a["E1"]["estimate"] for a in areas])
    v["noise_min"], v["noise_max"] = _range([a["noise"]["estimate"] for a in areas],
                                            lambda x: pct(x, 2))
    v["r_day_min"] = floor_pct(min(float(x["regret"].min()) for x in days.values()))
    v["noise_day_max"] = ceil_pct(max(float(x["noise"].max()) for x in days.values()))
    v["e4_own_min"], v["e4_own_max"] = _range([abs(a["E4"]["own"]["estimate"]) for a in areas])
    v["e4_best_min"], v["e4_best_max"] = _range(
        [abs(a["E4"]["best"]["estimate"]) for a in areas])
    v["alpha_min"], v["alpha_max"] = _range([a["alpha"] for a in areas], lambda x: f"{x:.2f}")
    v["kappa_road_min"], v["kappa_road_max"] = _range(
        [p["road"]["circuity"] for p in ph.values()], lambda x: f"{x:.2f}")
    v["kappa_straight_min"], v["kappa_straight_max"] = _range(
        [p["straight"]["circuity"] for p in ph.values()], lambda x: f"{x:.2f}")
    v["leg_min"], v["leg_max"] = _range([p["road"]["leg_straight_m"] / 1000 for p in ph.values()],
                                        lambda x: f"{x:.1f}")
    v["pair_min"], v["pair_max"] = _range([p["pair_straight_m"] / 1000 for p in ph.values()],
                                          lambda x: f"{x:.1f}")
    for (pair_key, area), a in zip(keys, areas, strict=True):
        k = f"{pair_key}_{area}"
        v |= {f"e1_{k}": pct(a["E1"]["estimate"]), f"e1_{k}_low": pct(a["E1"]["low"]),
              f"e1_{k}_high": pct(a["E1"]["high"]),
              f"e4_own_{k}": short(a["E4"]["own"]["estimate"]),
              f"km_{k}": f"{ph[k]['straight']['road_km'] - ph[k]['road']['road_km']:.1f}"}
    for pair in PAIRS:
        w = e["water"][pair.key]
        v |= {f"e3_{pair.key}": points(w["estimate"]), f"e3_{pair.key}_low": points(w["low"]),
              f"e3_{pair.key}_high": points(w["high"]),
              f"pi_{pair.key}": f"{pair.order_probability:g}",
              f"orders_up_{pair.key}": pct(_area(e, pair.key, "both")["clients"]
                                           / _area(e, pair.key, "city")["clients"] - 1, 0)}
        for area in AREAS:
            idx = area_indices(d, pair, area == "both")
            s = matrix_summary(d.road[pair.key], d.straight[pair.key], idx)
            v[f"orders_{pair.key}_{area}"] = f"{orders(d, pair, idx)['mean']:.0f}"
            if (pair.key, area) == (lisbon, "city"):
                v |= {"alpha_lisbon_city": f"{s['ratio']['p50']:.2f}",
                      "ratio_p5_lisbon_city": f"{s['ratio']['p5']:.2f}",
                      "ratio_p95_lisbon_city": f"{s['ratio']['p95']:.2f}"}
        pairs = m["pairs"][pair.key]
        v[f"nodes_{pair.key}"], v[f"edges_{pair.key}"] = (f"{pairs['nodes']:,}",
                                                          f"{pairs['edges']:,}")
    counts = {name: sum(1 for p in d.points if p["role"] == "pharmacy"
                        and p["municipality"] == name) for name in _municipalities()}
    v |= {"n_lisboa": str(counts["Lisboa"]), "n_almada": str(counts["Almada"]),
          "n_porto": str(counts["Porto"]), "n_gaia": str(counts["Vila Nova de Gaia"])}
    return v | _sensitivity_values(e) | _review_values(e)


def _review_values(e: Mapping[str, Any]) -> dict[str, str]:
    """After the critical review (docs/REVIEW.md): route direction, the straight-line plan's
    seed noise, the river without direction and in kilometres, and the legs."""
    r = e["after_review"]
    keys = [f"{p.key}_{a}" for p in PAIRS for a in AREAS]
    ra = r["areas"]
    lisbon, porto = (p.key for p in PAIRS)
    v: dict[str, str] = {}
    v["dir_min"], v["dir_max"] = _range([ra[k]["direction_share"] for k in keys],
                                        lambda x: pct(x, 0))
    v["dir_var_min"], v["dir_var_max"] = _range(
        [x["share"] for x in r["direction_by_variant"]], lambda x: pct(x, 0))
    v["same_min"], v["same_max"] = _range([ra[k]["same_routes"] for k in keys],
                                          lambda x: pct(x, 0))
    v["or_min"], v["or_max"] = _range([ra[k]["oriented"]["estimate"] for k in keys])
    v["noise_s_min"], v["noise_s_max"] = _range([ra[k]["noise_straight"]["estimate"]
                                                 for k in keys])
    v["noise_s_flat_max"] = pct(max(ra[k]["noise_straight_on_straight"] for k in keys), 2)
    v["times_s_min"], v["times_s_max"] = _range(
        [e["areas"][k]["E1"]["estimate"] / ra[k]["noise_straight"]["estimate"] for k in keys],
        lambda x: f"{x:.1f}")
    for k in keys:
        o = ra[k]["oriented"]
        v |= {f"or_{k}": pct(o["estimate"]), f"or_{k}_low": pct(o["low"]),
              f"or_{k}_high": pct(o["high"])}
    for pair in PAIRS:
        w = r["water"][pair.key]
        v |= {f"e3o_{pair.key}": points(w["oriented"]["estimate"]),
              f"e3o_{pair.key}_low": points(w["oriented"]["low"]),
              f"e3o_{pair.key}_high": points(w["oriented"]["high"])}
    w = r["water"][lisbon]["registered_at_95"]
    v |= {"e3_lisbon_95_low": points(w["low"]), "e3_lisbon_95_high": points(w["high"])}
    w = r["water"][porto]
    v |= {"e3_km_porto": f"{w['km']['estimate']:.1f}", "e3_km_porto_low": f"{w['km']['low']:.1f}",
          "e3_km_porto_high": f"{w['km']['high']:.1f}",
          "asym_porto_city": pct(w["asymmetry"]["city"]),
          "asym_porto_other": pct(w["asymmetry"]["other"]),
          "asym_porto_across": pct(w["asymmetry"]["across"])}
    v |= {f"road_km_porto_{a}": f"{e['post_hoc'][f'{porto}_{a}']['road']['road_km']:.0f}"
          for a in AREAS}
    bands = list(next(iter(ra.values()))["ratio"]["median_by_length"])
    v["short_min"], v["short_max"] = _range(
        [ra[k]["ratio"]["median_by_length"][bands[1]] for k in keys], lambda x: f"{x:.2f}")
    v["long_min"], v["long_max"] = _range(
        [ra[k]["ratio"]["median_by_length"][bands[-1]] for k in keys], lambda x: f"{x:.2f}")
    v["long_km"] = f"{LENGTH_BANDS_M[-1][0] / 1000:g}"
    v["ratio_mean_min"], v["ratio_mean_max"] = _range([ra[k]["ratio"]["mean"] for k in keys],
                                                      lambda x: f"{x:.2f}")
    legs = {k: ra[k]["legs"]["straight"] for k in keys}
    v["legs_typical"] = word(sum(1 for x in legs.values()
                                 if abs(x["circuity"] - x["like"]) < LEGS_TYPICAL))
    v["almada_legs"] = f"{legs[f'{lisbon}_both']['circuity']:.2f}"
    v["almada_like"] = f"{legs[f'{lisbon}_both']['like']:.2f}"
    steps = list(next(iter(r["capacity"].values())))
    v["cap_up_min"], v["cap_up_max"] = _range(
        [r["capacity"][k][steps[0]]["estimate"] for k in keys], lambda x: f"{100 * x:.1f}")
    v["cap_24_up"] = word(sum(1 for k in keys if r["capacity"][k][steps[-1]]["estimate"] > 0))
    return v


def _municipalities() -> list[str]:
    return [m for p in PAIRS for m in (p.city, p.other)]


def _extract_date(name: str) -> str:
    """portugal-261027.osm.pbf -> 27 October 2026."""
    stamp = re.search(r"-(\d{6})\.osm\.pbf$", name)
    if stamp is None:
        raise ValueError(f"No date in {name}.")
    yy, mm, dd = (int(stamp.group(1)[i:i + 2]) for i in (0, 2, 4))
    return f"{dd} {MONTHS[mm - 1]} {2000 + yy}"


def _sensitivity_values(e: Mapping[str, Any]) -> dict[str, str]:
    by = {(s["variant"], s["pair"]): s for s in e["sensitivity"]}
    variants = [x for x in VARIANTS if x.key != "main"]
    capacities = sorted({x.capacity for x in variants if x.capacity != CAPACITY})
    longer = max(x.iterations for x in variants)
    lisbon = PAIRS[0].key
    e1 = [s[k]["estimate"] for s in e["sensitivity"] for k in ("E1_city", "E1_both")]
    changes = [abs(by[("iterations_" + str(longer), p.key)][k]["estimate"]
                   - by[("main", p.key)][k]["estimate"])
               for p in PAIRS for k in ("E1_city", "E1_both", "E3")]
    lisbon_e3 = [by[(x.key, lisbon)]["E3"] for x in variants]
    v = {"n_variants": word(len(variants)), "sens_days": str(SENSITIVITY_DAYS),
         "sens_seeds": word(len(SENSITIVITY_SEEDS)),
         "cap_small": str(capacities[0]), "cap_large": str(capacities[-1]),
         "iter_large": f"{longer:,}", "effort_ratio": word(longer // ITERATIONS).capitalize(),
         "iter_max_change": f"{math.ceil(1000 * max(changes)) / 10:.1f}",
         "lisbon_e3_positive": word(sum(1 for x in lisbon_e3 if x["estimate"] > 0)),
         "lisbon_e3_above": word(sum(1 for x in lisbon_e3 if x["low"] > 0)),
         "porto_e3_with_zero": word(sum(1 for x in variants
                                        if by[(x.key, PAIRS[1].key)]["E3"]["low"] < 0
                                        < by[(x.key, PAIRS[1].key)]["E3"]["high"]))}
    v["sens_e1_min"], v["sens_e1_max"] = pct(min(e1)), pct(max(e1))
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
    header = ("<!-- Generated by routing-roads publication from article.template.md and the "
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
        from routing_roads.figures import draw_all

        e = json.loads((root / "results" / "estimates.json").read_text(encoding="utf-8"))
        paths = draw_all(load(root / "data" / "derived"),
                         load_solves(root / "results" / "solves.csv"), e,
                         root / "publication" / "figures")
        written += [p.relative_to(root).as_posix() for p in paths]
    return written


def check(root: Path) -> list[str]:
    """Generated text files that differ from what the code produces now."""
    return [name for name, text in outputs(root).items()
            if not (root / name).exists()
            or (root / name).read_text(encoding="utf-8") != text]


def per_day(root: Path) -> dict[str, Any]:
    """The day-level series, for the claims test."""
    e = json.loads((root / "results" / "estimates.json").read_text(encoding="utf-8"))
    return daily(load_solves(root / "results" / "solves.csv"), e)
