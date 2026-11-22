"""One row per country and local day: calendar, ex-post truth, and every model's label.

The labels each covariate variant uses for a day:
- flag: national holiday (sunday_like) or the weekday;
- llm:<model>: the model's answer on the days it was asked about; the weekday on every
  other day and wherever its answer was invalid;
- oracle: the ex-post truth, falling back to the weekday where it cannot be computed.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .calendar import CountryCalendar, build_calendar, calendar_frame, candidate_days, note
from .config import Config
from .data import Prepared, daily_mean
from .daytype import truth_labels
from .llm import latest_labels


def calendars(cfg: Config, countries: list[str]) -> dict[str, CountryCalendar]:
    """Holiday calendars covering the data and every label year, with a week to spare."""
    first = min(cfg.first_day, date(min(cfg.label_years), 1, 1)) - timedelta(days=7)
    last = max(cfg.last_day, date(max(cfg.label_years), 12, 31)) + timedelta(days=7)
    return {k: build_calendar(cfg.countries[k], first, last) for k in countries}


def label_items(cfg: Config, cals: dict[str, CountryCalendar], years: list[int]) -> list[tuple[str, date, str, str]]:
    items = []
    for key, cal in cals.items():
        for year in years:
            for d, reason in candidate_days(cal, year, cfg).items():
                items.append((key, d, reason, note(cal, d, cfg.max_regions_named)))
    return items


def day_table(cfg: Config, key: str, cal: CountryCalendar, prep: Prepared | None) -> pd.DataFrame:
    """Everything known about each local day of one country."""
    table = calendar_frame(cal, cfg, cfg.first_day, cfg.last_day)
    reasons = {}
    for year in sorted({d.year for d in table.index}):
        reasons.update(candidate_days(cal, year, cfg))
    table["candidate"] = [reasons.get(d) for d in table.index]
    table["label_year"] = [d.year in cfg.label_years for d in table.index]

    if prep is not None:
        load = pd.Series(prep.observed, index=prep.index)
        load[prep.masked] = np.nan
        daily = daily_mean(load, cal.country.tz)
        truth = truth_labels(daily.reindex(table.index), table["ordinary"], cfg.truth_window_days)
        table = table.join(truth)
        table["daily_load"] = daily.reindex(table.index)
        temp = daily_mean(pd.Series(prep.temperature, index=prep.index), cal.country.tz)
        table["temperature"] = temp.reindex(table.index)
    else:
        for col in (
            "truth",
            "relative_level",
            "ref_workday",
            "ref_saturday",
            "ref_sunday",
            "daily_load",
            "temperature",
        ):
            table[col] = np.nan

    table["oracle"] = [
        t if isinstance(t, str) else w for t, w in zip(table["truth"], table["weekday_rule"], strict=True)
    ]
    table["informative"] = [
        isinstance(t, str) and t != w for t, w in zip(table["truth"], table["weekday_rule"], strict=True)
    ]
    for llm in cfg.llms:
        answers = latest_labels(cfg, llm)
        raw, used = [], []
        for d, cand, rule in zip(table.index, table["candidate"], table["weekday_rule"], strict=True):
            row = answers.get((key, d.isoformat())) if cand else None
            label = row["label"] if row and row.get("valid") else None
            raw.append(label if row else None)
            used.append(label if label else rule)
        table[f"answer:{llm}"] = raw
        table[f"asked:{llm}"] = [
            bool(answers.get((key, d.isoformat()))) and c is not None
            for d, c in zip(table.index, table["candidate"], strict=True)
        ]
        table[f"llm:{llm}"] = used
    table.index.name = "date"
    return table


def coverage(table: pd.DataFrame, llm: str) -> float:
    """Share of the candidate days in the label years that the model has answered."""
    mask = table["candidate"].notna() & table["label_year"]
    return float(table.loc[mask, f"asked:{llm}"].mean()) if mask.any() else float("nan")


def build(cfg: Config, key: str, prep: Prepared | None, cal: CountryCalendar | None = None) -> pd.DataFrame:
    cal = cal or calendars(cfg, [key])[key]
    return day_table(cfg, key, cal, prep)
