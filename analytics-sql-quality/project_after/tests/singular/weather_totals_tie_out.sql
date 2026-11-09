select 1 as failure
where
    (select sum(flights) from weather_delays)
    is distinct from (select count(*) from stg_flights where not is_cancelled)
