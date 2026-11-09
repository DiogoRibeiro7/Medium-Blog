-- Grain: weather state at the origin in the scheduled departure hour.
select
    case
        when weather.observed_hour_ts is null then 'no observation'
        when weather.precip_inches > 0 then 'precipitation'
        else 'dry'
    end as weather_at_departure,
    count(*) as flights,
    avg(flights.dep_delay_min) as avg_dep_delay_min
from stg_flights as flights
left join stg_weather as weather
    on
        flights.origin = weather.origin
        and flights.sched_dep_hour_ts = weather.observed_hour_ts
where not flights.is_cancelled
group by weather_at_departure
