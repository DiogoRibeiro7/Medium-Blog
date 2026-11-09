-- Average times count, summed over slots, must give back the total delay
-- minutes of departed flights.
select 1 as failure
where abs(
    (select sum(departures * avg_dep_delay_min) from departure_slot_delays)
    - (select sum(dep_delay_min) from stg_flights where not is_cancelled)
) > 0.5
