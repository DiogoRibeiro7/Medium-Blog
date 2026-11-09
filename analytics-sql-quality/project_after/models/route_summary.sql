-- Grain: origin x destination. Four destinations (BQN, PSE, SJU, STT) are not
-- in the airports table, so the join is a left join with a labelled fallback.
select
    flights.origin,
    flights.dest,
    coalesce(airports.airport_name, 'Unknown (' || flights.dest || ')')
        as dest_name,
    count(*) as flights,
    count(flights.arr_delay_min) as flights_completed,
    avg(flights.arr_delay_min) as avg_arr_delay_min,
    max(flights.distance_miles) as distance_miles
from stg_flights as flights
left join stg_airports as airports
    on flights.dest = airports.airport_code
group by flights.origin, flights.dest, airports.airport_name
