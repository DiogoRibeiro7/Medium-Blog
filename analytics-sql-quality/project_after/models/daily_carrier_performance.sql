-- Grain: carrier x local flight date.
-- On-time means arriving no more than 15 minutes late; the rate is taken over
-- completed flights, and cancellations are reported in their own column.
select
    carrier,
    flight_date,
    count(*) as flights_scheduled,
    count(*) filter (where is_cancelled) as flights_cancelled,
    count(arr_delay_min) as flights_completed,
    count(*) filter (where arr_delay_min <= 15) as on_time_arrivals,
    cast(count(*) filter (where arr_delay_min <= 15) as double)
    / nullif(count(arr_delay_min), 0) as on_time_rate,
    avg(arr_delay_min) as avg_arr_delay_min
from stg_flights
group by carrier, flight_date
