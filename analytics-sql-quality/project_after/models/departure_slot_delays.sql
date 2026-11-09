-- Grain: scheduled departure hour (local). dep_time is a clock reading in
-- HHMM form, so delays come from dep_delay, never from subtracting clocks.
select
    sched_dep_hour,
    count(*) as departures,
    avg(dep_delay_min) as avg_dep_delay_min
from stg_flights
where not is_cancelled
group by sched_dep_hour
