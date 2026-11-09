-- Grain: aircraft age band. Flights whose tail number is missing or absent
-- from the registry, or whose aircraft has no manufacture year, are 'unknown'.
with departed as (
    select
        flights.dep_delay_min,
        2013 - planes.manufacture_year as age_years
    from stg_flights as flights
    left join stg_planes as planes
        on flights.tailnum = planes.tailnum
    where not flights.is_cancelled
)

select
    case
        when age_years is null then 'unknown'
        when age_years < 5 then '00-04'
        when age_years < 10 then '05-09'
        when age_years < 20 then '10-19'
        else '20+'
    end as age_band,
    count(*) as flights,
    avg(dep_delay_min) as avg_dep_delay_min
from departed
group by age_band
