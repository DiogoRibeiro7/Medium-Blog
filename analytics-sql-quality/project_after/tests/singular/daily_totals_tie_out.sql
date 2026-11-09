-- Every scheduled flight lands in exactly one carrier-day, on its local date.
with mart as (
    select
        flight_date,
        sum(flights_scheduled) as n
    from daily_carrier_performance
    group by flight_date
),

source as (
    select
        flight_date,
        count(*) as n
    from stg_flights
    group by flight_date
)

select coalesce(mart.flight_date, source.flight_date) as flight_date
from mart
full outer join source on mart.flight_date = source.flight_date
where mart.n is distinct from source.n
