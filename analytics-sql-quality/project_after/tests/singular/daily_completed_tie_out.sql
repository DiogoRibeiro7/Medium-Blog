select 1 as failure
where
    (select sum(flights_completed) from daily_carrier_performance)
    is distinct from (select count(arr_delay_min) from stg_flights)
