-- Grain: one row per origin airport and hour, keyed on the UTC timestamp.
-- The local year/month/day/hour columns repeat during the autumn DST change.
select
    origin,
    time_hour as observed_hour_ts,
    precip as precip_inches,
    visib as visibility_miles
from raw_weather
