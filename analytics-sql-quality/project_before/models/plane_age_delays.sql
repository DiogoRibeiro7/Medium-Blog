-- does fleet age drive delays? planes table has one row per tail and year
with x as (
select f.*, f.year - try_cast(p.year as integer) as age_years
from raw_flights f join raw_planes p on f.tailnum = p.tailnum and f.year = try_cast(p.year as integer)
where f.dep_time <> 'NA')
select case when age_years < 5 then '00-04' when age_years < 10 then '05-09' when age_years < 20 then '10-19' else '20+' end as age_band,
count(*) as flights, avg(try_cast(dep_delay as integer)) as avg_dep_delay_min
from x group by 1
