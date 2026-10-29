"""Amendment A1, section 8.2: yield series that are not regional measurements.

Five rules, each evaluated on the panel of section 3.1 (and, for Q3 and Q4, on the
published values behind it). They use areas, productions and yields only, never weather.
A region-year excluded by any rule leaves the panel; regions are then checked again
against the 10-year rule. Amendment A2 (section 10.2) adds Q6-Q8 for maize, evaluated on
the region-years that pass Q1-Q5.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from eu_wheat_weather.crops import WheatRow
from eu_wheat_weather.panel import MIN_YEARS
from eu_wheat_weather.regions import CodeMatch, HarmonisedRow

Q1_MIN_REGIONS, Q1_SPREAD = 3, 1.005      # largest over smallest yield
Q2_MIN_REGIONS, Q2_MAX_SD = 4, 0.02       # sample SD of the change in log yield
Q4_PRODUCTION_TOL, Q4_AREA_GAP = 0.01, 0.05
Q5_RUN, Q5_FACTOR = 3, 2.0
RULES = ("Q1", "Q2", "Q3", "Q4", "Q5")
# Amendment A2, section 10.2.
Q6_RUN, Q6_STEP = 3, 0.01         # three or more years, each within 1% of the year before
Q7_BREAK, Q7_MIN_OTHERS = 0.3, 2  # change in log yield against the country's median change
Q8_TOL = 0.001                    # area and production within 0.1% of another code's
RULES_A2 = ("Q6", "Q7", "Q8")

# The review's cases (docs/REVIEW.md, finding D1), to report which the rule catches.
REVIEW_CASES: tuple[tuple[str, tuple[str, ...], tuple[int, ...]], ...] = (
    ("Greece, pro-rated, 2000-2008", ("EL",), tuple(range(2000, 2009))),
    ("Netherlands, one figure, 2005", ("NL",), (2005,)),
    ("DK04, area understated, 2006-2009", ("DK04",), (2006, 2007, 2008, 2009)),
    ("NL22 and NL23, production swapped, 2004", ("NL22", "NL23"), (2004,)),
    ("ITF2, about 1 t/ha, 2012-2015", ("ITF2",), (2012, 2013, 2014, 2015)),
)


# The maize review's cases (docs/MAIZE_REVIEW.md, findings D3 and D5) for Q6-Q8.
REVIEW_CASES_A2: tuple[tuple[str, tuple[str, ...], tuple[int, ...]], ...] = (
    ("ITH4, repeated yields, 2015-2017", ("ITH4",), (2015, 2016, 2017)),
    ("ITG2, repeated yields, 2019-2021", ("ITG2",), (2019, 2020, 2021)),
    ("ITH4, break where its codes join in 2014/15", ("ITH4",), tuple(range(2010, 2015))),
    ("HR02, HR04's figures, 2013-2014", ("HR02",), (2013, 2014)),
)


@dataclass(frozen=True)
class Flag:
    geo: str
    year: int
    rule: str
    detail: str


def _by(rows: Iterable[Mapping[str, str]], *keys: str,
        ) -> dict[tuple[str, ...], list[Mapping[str, str]]]:
    out: dict[tuple[str, ...], list[Mapping[str, str]]] = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def q1_one_figure(panel: Sequence[Mapping[str, str]]) -> list[Flag]:
    flags = []
    for (country, year), rows in sorted(_by(panel, "country", "year").items()):
        yields = [float(r["yield_t_ha"]) for r in rows]
        if len(rows) >= Q1_MIN_REGIONS and max(yields) / min(yields) < Q1_SPREAD:
            detail = f"{len(rows)} regions of {country} within {max(yields) / min(yields) - 1:.2%}"
            flags += [Flag(r["geo"], int(year), "Q1", detail) for r in rows]
    return flags


def q2_pro_rated(panel: Sequence[Mapping[str, str]]) -> list[Flag]:
    log_yield = {(r["geo"], int(r["year"])): float(r["log_yield"]) for r in panel}
    flags = []
    for (country,), rows in sorted(_by(panel, "country").items()):
        years = sorted({int(r["year"]) for r in rows})
        regions = sorted({r["geo"] for r in rows})
        for t in years:
            both = [g for g in regions if (g, t - 1) in log_yield and (g, t) in log_yield]
            if len(both) < Q2_MIN_REGIONS:
                continue
            sd = statistics.stdev(log_yield[(g, t)] - log_yield[(g, t - 1)] for g in both)
            if sd < Q2_MAX_SD:
                detail = (f"{country} {t - 1}-{t}: SD of the change in log yield across "
                          f"{len(both)} regions {sd:.4f}")
                flags += [Flag(g, y, "Q2", detail) for g in both for y in (t - 1, t)]
    return flags


def q3_carried_forward(panel: Sequence[Mapping[str, str]],
                       harmonised: Sequence[HarmonisedRow]) -> list[Flag]:
    published = {(h.geo, h.year): (h.area_kha, h.production_kt) for h in harmonised
                 if h.area_kha is not None and h.production_kt is not None}
    flags = []
    for r in panel:
        geo, year = r["geo"], int(r["year"])
        before = published.get((geo, year - 1))
        if before is not None and before == published.get((geo, year)):
            flags.append(Flag(geo, year, "Q3", f"area {before[0]} and production "
                              f"{before[1]} as in {year - 1}"))
    return flags


def q4_sums_disagree(panel: Sequence[Mapping[str, str]], published: Sequence[WheatRow],
                     matches: Sequence[CodeMatch]) -> list[Flag]:
    target = {m.code: m.target for m in matches}
    in_panel = {(r["geo"], int(r["year"])) for r in panel}
    values = {(w.geo, w.year): (w.area_kha, w.production_kt) for w in published
              if w.area_kha is not None and w.production_kt is not None}
    flags = []
    for (code, year), (area, production) in sorted(values.items()):
        if len(code) != 3:
            continue
        children = sorted(g for (g, y) in values if y == year and len(g) == 4
                          and g.startswith(code))
        if len(children) < 2 or not area or not production:
            continue
        child_area = sum(values[(g, year)][0] or 0.0 for g in children)
        child_production = sum(values[(g, year)][1] or 0.0 for g in children)
        if (abs(child_production / production - 1) <= Q4_PRODUCTION_TOL
                and abs(child_area / area - 1) > Q4_AREA_GAP):
            detail = (f"{code} {year}: NUTS-2 production {child_production / production - 1:+.2%}"
                      f" of NUTS-1, area {child_area / area - 1:+.2%}")
            for g in children:
                t = target.get(g)
                if t is not None and (t, year) in in_panel:
                    flags.append(Flag(t, year, "Q4", detail))
    return flags


def q5_level_shifts(panel: Sequence[Mapping[str, str]]) -> list[Flag]:
    flags = []
    for (geo,), rows in sorted(_by(panel, "geo").items()):
        yields = {int(r["year"]): float(r["yield_t_ha"]) for r in rows}
        shifted = set()
        for year, y in yields.items():
            others = [v for t, v in yields.items() if t != year]
            if not others:                    # a single year: nothing to compare with
                continue
            other = statistics.median(others)
            if y < other / Q5_FACTOR or y > other * Q5_FACTOR:
                shifted.add(year)
        runs: list[list[int]] = []
        for year in sorted(shifted):
            if runs and year == runs[-1][-1] + 1:
                runs[-1].append(year)
            else:
                runs.append([year])
        for run in runs:
            if len(run) >= Q5_RUN:
                flags += [Flag(geo, t, "Q5", f"{len(run)} years from {run[0]} below half or "
                               f"above twice the median of the other years") for t in run]
    return flags


def q6_repeated_yields(panel: Sequence[Mapping[str, str]]) -> list[Flag]:
    """A run of three or more consecutive years, each yield within 1% of the year before:
    the years of the run after the first."""
    flags = []
    for (geo,), rows in sorted(_by(panel, "geo").items()):
        yields = {int(r["year"]): float(r["yield_t_ha"]) for r in rows}
        runs: list[list[int]] = []
        for year in sorted(yields):
            if (runs and year == runs[-1][-1] + 1
                    and abs(yields[year] / yields[runs[-1][-1]] - 1) < Q6_STEP):
                runs[-1].append(year)
            else:
                runs.append([year])
        for run in runs:
            if len(run) >= Q6_RUN:
                flags += [Flag(geo, t, "Q6", f"{len(run)} years from {run[0]}, each within "
                               f"{Q6_STEP:.0%} of the year before") for t in run[1:]]
    return flags


def q7_join_breaks(panel: Sequence[Mapping[str, str]], sources: Mapping[tuple[str, int], str],
                   recoded: set[str]) -> list[Flag]:
    """Where a region's series joins two codes by an official recode: if its change in log
    yield at the join differs from the country's other regions' median change by more
    than 0.3, the years before the join."""
    log_yield = {(r["geo"], int(r["year"])): float(r["log_yield"]) for r in panel}
    country = {r["geo"]: r["country"] for r in panel}
    flags = []
    for geo in sorted(country):
        years = sorted(y for (g, y) in log_yield if g == geo)
        for j in years:
            before, after = sources.get((geo, j - 1)), sources.get((geo, j))
            if (geo, j - 1) not in log_yield or before == after or not {before, after} & recoded:
                continue
            others = [g for g in country if g != geo and country[g] == country[geo]
                      and (g, j - 1) in log_yield and (g, j) in log_yield]
            if len(others) < Q7_MIN_OTHERS:
                continue
            median = statistics.median(log_yield[(g, j)] - log_yield[(g, j - 1)] for g in others)
            change = log_yield[(geo, j)] - log_yield[(geo, j - 1)]
            if abs(change - median) > Q7_BREAK:
                detail = (f"{before} to {after} in {j}: change in log yield {change:+.3f}, "
                          f"the country's median {median:+.3f}")
                flags += [Flag(geo, t, "Q7", detail) for t in years if t < j]
    return flags


def q8_other_codes_figures(panel: Sequence[Mapping[str, str]],
                           sources: Mapping[tuple[str, int], str],
                           published: Sequence[WheatRow], matches: Sequence[CodeMatch],
                           equivalent: Mapping[str, str] | None = None) -> list[Flag]:
    """A region-year whose area and production both equal, within 0.1%, another code's of
    the same country and year. Not counted: the region's own source, codes that map to
    the region, and a parent code whose only NUTS-2 region it is: in NUTS 2021, or, for
    an older code, in its own version (``equivalent``: code -> NUTS-2021 region)."""
    target = {m.code: m.target for m in matches}
    nuts2021 = {m.target for m in matches if m.target}
    values: dict[tuple[str, int], list[tuple[str, float, float]]] = defaultdict(list)
    for w in published:
        if w.area_kha and w.production_kt:
            values[(w.geo[:2], w.year)].append((w.geo, w.area_kha, w.production_kt))
    flags = []
    for r in panel:
        geo, year = r["geo"], int(r["year"])
        area, production = float(r["area_kha"]), float(r["production_kt"])
        for code, a, p in values.get((geo[:2], year), []):
            if (code == sources.get((geo, year)) or target.get(code) == geo
                    or (equivalent or {}).get(code) == geo):
                continue
            if geo.startswith(code) and not any(n != geo and n.startswith(code)
                                                for n in nuts2021):
                continue
            if abs(a / area - 1) <= Q8_TOL and abs(p / production - 1) <= Q8_TOL:
                flags.append(Flag(geo, year, "Q8", f"area {area} and production {production} "
                                  f"as {code}'s"))
                break
    return flags


def apply_rule(panel: Sequence[Mapping[str, str]], harmonised: Sequence[HarmonisedRow],
               published: Sequence[WheatRow], matches: Sequence[CodeMatch],
               ) -> tuple[list[Mapping[str, str]], list[Flag], list[str]]:
    """The panel after the rule, every flag, and the regions the 10-year re-check drops."""
    flags = (q1_one_figure(panel) + q2_pro_rated(panel)
             + q3_carried_forward(panel, harmonised)
             + q4_sums_disagree(panel, published, matches) + q5_level_shifts(panel))
    out = {(f.geo, f.year) for f in flags}
    kept = [r for r in panel if (r["geo"], int(r["year"])) not in out]
    years = Counter(r["geo"] for r in kept)
    short = sorted(g for g, n in years.items() if n < MIN_YEARS)
    return [r for r in kept if r["geo"] not in short], flags, short


def review_cases(flags: Sequence[Flag], panel: Sequence[Mapping[str, str]],
                 cases: Sequence[tuple[str, tuple[str, ...], tuple[int, ...]]] = REVIEW_CASES,
                 ) -> list[dict[str, object]]:
    """For each of the review's cases: its region-years in the panel, and how many the
    rule excludes, by rule."""
    flagged: dict[tuple[str, int], set[str]] = defaultdict(set)
    for f in flags:
        flagged[(f.geo, f.year)].add(f.rule)
    out = []
    for label, prefixes, years in cases:
        rows = [(r["geo"], int(r["year"])) for r in panel
                if int(r["year"]) in years and r["geo"].startswith(prefixes)]
        caught = [k for k in rows if k in flagged]
        rules = sorted({rule for k in caught for rule in flagged[k]})
        out.append({"case": label, "in_panel": len(rows), "excluded": len(caught),
                    "rules": "+".join(rules)})
    return out
