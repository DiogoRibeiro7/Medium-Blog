from datetime import date

from swaptails.calendar import (
    build_calendar,
    candidate_days,
    categorise,
    flag_rule,
    in_year_end,
    note,
    weekday_rule,
)


def test_national_regional_optional(cfg):
    de = build_calendar(cfg.countries["de"], date(2025, 1, 1), date(2025, 12, 31))
    assert de.is_national(date(2025, 5, 29))  # Christi Himmelfahrt
    assert not de.is_national(date(2025, 6, 19)) and de.is_regional(date(2025, 6, 19))  # Fronleichnam
    regions = de.get(date(2025, 6, 19)).regional["Fronleichnam"]
    assert "Bayern" in regions and "Berlin" not in regions
    pt = build_calendar(cfg.countries["pt"], date(2025, 1, 1), date(2025, 12, 31))
    assert pt.is_optional(date(2025, 3, 4))  # Carnaval
    assert not pt.is_national(date(2025, 3, 4))
    # Azores and Madeira are excluded: 18 mainland districts.
    assert pt.n_subdivisions == 18


def test_categories_and_rules(cfg):
    de = build_calendar(cfg.countries["de"], date(2025, 1, 1), date(2025, 12, 31))
    days = [
        date(2025, 5, 29),
        date(2025, 5, 30),
        date(2025, 6, 19),
        date(2025, 7, 9),
        date(2025, 7, 12),
        date(2025, 12, 29),
    ]
    cats = categorise(de, days)
    assert cats[date(2025, 5, 29)] == "national_holiday"
    assert cats[date(2025, 5, 30)] == "adjacent"  # the bridge Friday
    assert cats[date(2025, 6, 19)] == "regional_holiday"
    assert cats[date(2025, 7, 9)] == "ordinary_workday"
    assert cats[date(2025, 7, 12)] == "ordinary_weekend"
    assert cats[date(2025, 12, 29)] == "year_end"
    assert weekday_rule(date(2025, 7, 12)) == "saturday_like"
    assert flag_rule(de, date(2025, 5, 29)) == "sunday_like"
    assert flag_rule(de, date(2025, 6, 19)) == "workday"  # a national flag misses regional holidays


def test_year_end_window_wraps():
    assert in_year_end(date(2025, 12, 31), (12, 20), (1, 6))
    assert in_year_end(date(2025, 1, 3), (12, 20), (1, 6))
    assert not in_year_end(date(2025, 1, 7), (12, 20), (1, 6))


def test_candidates_are_deterministic_and_cover_bridges(cfg):
    de = build_calendar(cfg.countries["de"], date(2024, 12, 1), date(2026, 1, 31))
    a = candidate_days(de, 2025, cfg)
    b = candidate_days(de, 2025, cfg)
    assert a == b
    assert a[date(2025, 5, 29)] == "holiday"
    assert a[date(2025, 5, 30)] == "near_holiday"
    assert a[date(2025, 12, 22)] == "year_end"
    assert sum(r == "control" for r in a.values()) == cfg.control_days_per_year


def test_note_reads_neighbours_in_local_language(cfg):
    de = build_calendar(cfg.countries["de"], date(2025, 1, 1), date(2025, 12, 31))
    text = note(de, date(2025, 6, 20))
    assert "Date: Friday 20 June 2025." in text
    assert "On this date: no holiday." in text
    assert "Day before (Thursday 19 June 2025): public holiday in 6 of 16 states" in text
    assert "Fronleichnam" in text


def test_notes_cover_label_years_before_the_data(cfg):
    from swaptails.days import calendars, label_items

    items = label_items(cfg, calendars(cfg, ["de"]), [2024])
    text = dict(((d, t) for _, d, _, t in items))[date(2024, 1, 1)]
    assert "nationwide public holiday: Neujahr" in text
