from datetime import date

import numpy as np
import pandas as pd

from swaptails.data import load_prepared
from swaptails.days import calendars, day_table
from swaptails.forecasters import SeasonalNaive, SimilarDay
from swaptails.harness import hourly_codes, make_batch, origins


def test_cutoff_is_local_18h_and_target_is_the_local_day(synthetic):
    cfg = synthetic
    for key, cutoff_utc in (("pt", "2025-01-14T18:00Z"), ("de", "2025-01-14T17:00Z")):
        prep = load_prepared(cfg, key)
        items, _ = origins(cfg, prep, date(2025, 1, 15), date(2025, 1, 15), 2016)
        o = items[0]
        assert prep.index[o.cutoff] == pd.Timestamp(cutoff_utc)
        assert o.n_target == 24 and o.horizon == 30
        first = prep.index[o.cutoff + o.first_target].tz_convert(prep.country.tz)
        assert (first.date(), first.hour) == (date(2025, 1, 15), 0)


def test_forecasts_never_see_the_future(synthetic):
    """Replacing every hour after the cutoff by garbage must not change a forecast."""
    cfg = synthetic
    prep = load_prepared(cfg, "de")
    table = day_table(cfg, "de", calendars(cfg, ["de"])["de"], prep)
    items, _ = origins(cfg, prep, date(2025, 2, 3), date(2025, 2, 9), 2016)
    codes = hourly_codes(prep, table, "oracle")
    spec = cfg.forecasters["similar_day"]
    for adapter in (SeasonalNaive(cfg.forecasters["seasonal_naive"]), SimilarDay(spec)):
        for o in items:
            batch = make_batch(prep, [o], 2016, o.horizon, "oracle", codes)
            before = adapter.forecast(batch, cfg.quantiles)
            saved = prep.inputs.copy()
            prep.inputs[o.cutoff :] = 1e9
            after = adapter.forecast(make_batch(prep, [o], 2016, o.horizon, "oracle", codes), cfg.quantiles)
            prep.inputs[:] = saved
            assert np.array_equal(before, after)


def test_similar_day_uses_the_label(synthetic):
    cfg = synthetic
    prep = load_prepared(cfg, "de")
    table = day_table(cfg, "de", calendars(cfg, ["de"])["de"], prep)
    # 26 December 2024, a Thursday holiday: a week earlier was an ordinary Thursday, the day
    # before was Christmas Day.
    items, _ = origins(cfg, prep, date(2024, 12, 26), date(2024, 12, 26), 2016)
    o = items[0]
    flag = hourly_codes(prep, table, "flag")
    assert flag[o.cutoff + o.first_target] == 1
    adapter = SimilarDay(cfg.forecasters["similar_day"])
    with_flag = adapter.forecast(make_batch(prep, [o], 2016, o.horizon, "flag", flag), cfg.quantiles)[0, :, 1]
    naive = SeasonalNaive(cfg.forecasters["seasonal_naive"]).forecast(
        make_batch(prep, [o], 2016, o.horizon, "none", None), cfg.quantiles
    )[0, :, 1]
    actual = prep.observed[o.cutoff + o.first_target : o.cutoff + o.first_target + o.n_target]
    target = slice(o.first_target, o.first_target + o.n_target)
    assert np.abs(with_flag[target] - actual).sum() < np.abs(naive[target] - actual).sum()
