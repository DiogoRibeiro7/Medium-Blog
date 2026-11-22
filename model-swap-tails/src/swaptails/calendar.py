"""Holiday calendars, the days sent to the language models, and the notes they read.

Holidays come from the `holidays` package, in each country's own language:

- national: public holidays observed in the whole country;
- regional: public holidays observed in some subdivisions only (German states, Spanish
  autonomous communities, Austrian states, Portuguese districts, Italian provinces,
  Alsace-Moselle);
- optional: holidays the package files under "optional" or "bank" (Carnival in Portugal,
  Liberation Day in the Netherlands, bank holidays in Belgium), which many people take
  off but which are not a statutory day off for everyone.

Each local date gets one calendar category, used to split the scores:
national_holiday > regional_holiday > optional_holiday > adjacent (one day either side of a
national holiday) > year_end (24 December to 1 January) > ordinary_workday / ordinary_weekend.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import Config, Country

OPTIONAL_CATEGORIES = ("optional", "bank")
CATEGORIES = (
    "national_holiday",
    "regional_holiday",
    "optional_holiday",
    "adjacent",
    "year_end",
    "ordinary_workday",
    "ordinary_weekend",
)
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


@dataclass
class DayCalendar:
    national: list[str] = field(default_factory=list)
    regional: dict[str, list[str]] = field(default_factory=dict)  # holiday name -> subdivision names
    optional: list[str] = field(default_factory=list)


@dataclass
class CountryCalendar:
    country: Country
    days: dict[date, DayCalendar]
    n_subdivisions: int
    first: date
    last: date

    def get(self, d: date) -> DayCalendar:
        return self.days.get(d, DayCalendar())

    def is_national(self, d: date) -> bool:
        return bool(self.get(d).national)

    def is_regional(self, d: date) -> bool:
        return bool(self.get(d).regional)

    def is_optional(self, d: date) -> bool:
        return bool(self.get(d).optional)

    def is_holiday(self, d: date) -> bool:
        day = self.get(d)
        return bool(day.national or day.regional or day.optional)


def subdivision_names(holiday_class) -> dict[str, str]:
    """Subdivision code -> readable name (the first alias the package lists, else the code)."""
    names: dict[str, str] = {}
    for alias, code in getattr(holiday_class, "subdivisions_aliases", {}).items():
        names.setdefault(code, alias)
    return {code: names.get(code, code) for code in getattr(holiday_class, "subdivisions", ())}


def build_calendar(country: Country, first: date, last: date) -> CountryCalendar:
    import holidays

    years = list(range(first.year, last.year + 1))
    base = holidays.country_holidays(country.holidays, years=years, categories=("public",))
    cls = type(base)
    supported = getattr(cls, "supported_categories", ("public",))
    optional_cats = tuple(c for c in OPTIONAL_CATEGORIES if c in supported)
    days: dict[date, DayCalendar] = {}

    def day(d: date) -> DayCalendar:
        return days.setdefault(d, DayCalendar())

    for d, name in sorted(base.items()):
        for part in str(name).split("; "):
            day(d).national.append(part)

    names = subdivision_names(cls)
    subdivisions = [s for s in names if s not in country.exclude_subdivisions]
    if country.include_subdivisions:
        unknown = set(country.include_subdivisions) - set(names)
        if unknown:
            raise ValueError(f"{country.key}: unknown subdivisions {sorted(unknown)}")
        subdivisions = [s for s in subdivisions if s in country.include_subdivisions]
    for sub in subdivisions:
        regional = holidays.country_holidays(country.holidays, subdiv=sub, years=years, categories=("public",))
        for d, name in sorted(regional.items()):
            national = set(day(d).national) if d in days else set()
            for part in str(name).split("; "):
                if part in national:
                    continue
                day(d).regional.setdefault(part, [])
                if names[sub] not in day(d).regional[part]:
                    day(d).regional[part].append(names[sub])

    if optional_cats:
        optional = holidays.country_holidays(country.holidays, years=years, categories=optional_cats)
        for d, name in sorted(optional.items()):
            for part in str(name).split("; "):
                if part not in day(d).national and part not in day(d).optional:
                    day(d).optional.append(part)

    days = {d: v for d, v in days.items() if first <= d <= last and (v.national or v.regional or v.optional)}
    return CountryCalendar(country, days, len(subdivisions), first, last)


def in_year_end(d: date, start: tuple[int, int], end: tuple[int, int]) -> bool:
    """Whether d falls in a window that wraps the new year, such as 20 December to 6 January."""
    md = (d.month, d.day)
    return md >= start or md <= end


def categorise(cal: CountryCalendar, dates: list[date]) -> dict[date, str]:
    out = {}
    for d in dates:
        if cal.is_national(d):
            c = "national_holiday"
        elif cal.is_regional(d):
            c = "regional_holiday"
        elif cal.is_optional(d):
            c = "optional_holiday"
        elif cal.is_national(d - timedelta(days=1)) or cal.is_national(d + timedelta(days=1)):
            c = "adjacent"
        elif in_year_end(d, (12, 24), (1, 1)):
            c = "year_end"
        elif d.weekday() < 5:
            c = "ordinary_workday"
        else:
            c = "ordinary_weekend"
        out[d] = c
    return out


def ordinary_mask(cal: CountryCalendar, dates: list[date], cfg: Config) -> dict[date, bool]:
    """Days usable as references for the ex-post day type: no holiday of any kind, not next to
    a national holiday, and outside the whole year-end window."""
    out = {}
    for d in dates:
        special = (
            cal.is_holiday(d)
            or cal.is_national(d - timedelta(days=1))
            or cal.is_national(d + timedelta(days=1))
            or in_year_end(d, cfg.year_end_from, cfg.year_end_to)
        )
        out[d] = not special
    return out


def _stable_rng(*parts) -> np.random.Generator:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def candidate_days(cal: CountryCalendar, year: int, cfg: Config) -> dict[date, str]:
    """Days the language models are asked about in one year, with the reason each was chosen.

    Reasons: 'holiday' (a holiday of any kind), 'near_holiday' (within the margin of one),
    'year_end' (the year-end window), 'control' (ordinary days drawn at random).
    """
    first, last = date(year, 1, 1), date(year, 12, 31)
    dates = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    out: dict[date, str] = {}
    margin = cfg.candidate_margin_days
    for d in dates:
        if cal.is_holiday(d):
            out[d] = "holiday"
    for d in dates:
        if d in out:
            continue
        if any(cal.is_holiday(d + timedelta(days=k)) for k in range(-margin, margin + 1) if k):
            out[d] = "near_holiday"
        elif in_year_end(d, cfg.year_end_from, cfg.year_end_to):
            out[d] = "year_end"
    pool = [d for d in dates if d not in out]
    rng = _stable_rng(cfg.seed, cal.country.key, year)
    n = min(cfg.control_days_per_year, len(pool))
    for i in sorted(rng.choice(len(pool), size=n, replace=False)):
        out[pool[i]] = "control"
    return dict(sorted(out.items()))


def _fmt(d: date) -> str:
    return f"{WEEKDAYS[d.weekday()]} {d.day} {d.strftime('%B')} {d.year}"


def _describe(cal: CountryCalendar, d: date, max_named: int) -> str:
    day = cal.get(d)
    parts = []
    if day.national:
        parts.append("nationwide public holiday: " + "; ".join(day.national))
    for name, regions in day.regional.items():
        n = len(regions)
        where = f"{n} of {cal.n_subdivisions} {cal.country.subdivision_term}"
        if n <= max_named:
            where += " (" + ", ".join(regions) + ")"
        parts.append(f"public holiday in {where}: {name}")
    if day.optional:
        parts.append("optional or bank holiday, not a statutory day off for everyone: " + "; ".join(day.optional))
    return "; ".join(parts) if parts else "no holiday"


def note(cal: CountryCalendar, d: date, max_named: int = 8) -> str:
    """The calendar note a language model reads for one day. Holiday names stay in the
    country's own language; the frame is in English."""
    c = cal.country
    lines = [
        f"Country: {c.name} ({c.holidays}). Holiday names are in the country's own language.",
        f"Date: {_fmt(d)}.",
        f"On this date: {_describe(cal, d, max_named)}.",
    ]
    for k, label in ((-2, "Two days before"), (-1, "Day before"), (1, "Day after"), (2, "Two days after")):
        other = d + timedelta(days=k)
        lines.append(f"{label} ({_fmt(other)}): {_describe(cal, other, max_named)}.")
    return "\n".join(lines)


def weekday_rule(d: date) -> str:
    return "workday" if d.weekday() < 5 else ("saturday_like" if d.weekday() == 5 else "sunday_like")


def flag_rule(cal: CountryCalendar, d: date) -> str:
    """The label a nationwide holiday flag implies: a holiday is a Sunday."""
    return "sunday_like" if cal.is_national(d) else weekday_rule(d)


def date_range(first: date, last: date) -> list[date]:
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def calendar_frame(cal: CountryCalendar, cfg: Config, first: date, last: date) -> pd.DataFrame:
    dates = date_range(first, last)
    cats = categorise(cal, dates)
    ordinary = ordinary_mask(cal, dates, cfg)
    return pd.DataFrame(
        {
            "date": dates,
            "weekday": [d.weekday() for d in dates],
            "category": [cats[d] for d in dates],
            "national": [cal.is_national(d) for d in dates],
            "ordinary": [ordinary[d] for d in dates],
            "weekday_rule": [weekday_rule(d) for d in dates],
            "flag_rule": [flag_rule(cal, d) for d in dates],
        }
    ).set_index("date")
