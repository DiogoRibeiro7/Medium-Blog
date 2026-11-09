-- Grain: month. Registered aircraft with no departure from NYC that month.
-- not exists, not "not in": stg_flights.tailnum contains NULLs.
with months as (
    select distinct flight_month
    from stg_flights
)

select
    months.flight_month,
    count(*) as idle_aircraft
from months
cross join stg_planes as planes
where
    not exists (
        select 1
        from stg_flights as flights
        where
            flights.tailnum = planes.tailnum
            and flights.flight_month = months.flight_month
    )
group by months.flight_month
