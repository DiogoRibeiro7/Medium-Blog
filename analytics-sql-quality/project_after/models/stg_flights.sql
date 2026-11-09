-- Grain: one row per scheduled departure from JFK, LGA or EWR in 2013.
-- Columns holding the text 'NA' arrive as VARCHAR. nullif() turns 'NA' into
-- NULL and the cast fails loudly on any other unexpected value.
select
    carrier,
    flight as flight_number,
    origin,
    dest,
    month as flight_month,
    hour as sched_dep_hour,
    distance as distance_miles,
    time_hour as sched_dep_hour_ts,
    cast(nullif(dep_delay, 'NA') as integer) as dep_delay_min,
    cast(nullif(arr_delay, 'NA') as integer) as arr_delay_min,
    nullif(tailnum, 'NA') as tailnum,
    make_date(year, month, day) as flight_date,
    nullif(dep_time, 'NA') is null as is_cancelled
from raw_flights
