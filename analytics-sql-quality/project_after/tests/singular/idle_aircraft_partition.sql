-- Each month, every registered aircraft is either idle or seen departing.
with active as (
    select
        flights.flight_month,
        count(distinct flights.tailnum) as active_aircraft
    from stg_flights as flights
    inner join stg_planes as planes on flights.tailnum = planes.tailnum
    group by flights.flight_month
)

select active.flight_month
from active
left join idle_aircraft on active.flight_month = idle_aircraft.flight_month
where
    coalesce(idle_aircraft.idle_aircraft, 0) + active.active_aircraft
    <> (select count(*) from stg_planes)
