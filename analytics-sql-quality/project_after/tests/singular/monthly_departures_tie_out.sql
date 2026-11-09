select 1 as failure
where
    (select sum(departures) from monthly_long_delays)
    is distinct from (select count(*) from stg_flights where not is_cancelled)
