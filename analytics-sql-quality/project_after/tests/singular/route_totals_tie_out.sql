select 1 as failure
where
    (select sum(flights) from route_summary)
    is distinct from (select count(*) from stg_flights)
