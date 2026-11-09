-- Grain: carrier x month. Departures exclude cancellations.
select
    carrier,
    flight_month,
    count(*) filter (where not is_cancelled) as departures,
    count(*) filter (where dep_delay_min > 60) as departures_over_60_min,
    cast(count(*) filter (where dep_delay_min > 60) as double)
    / nullif(count(*) filter (where not is_cancelled), 0) as share_over_60_min
from stg_flights
group by carrier, flight_month
