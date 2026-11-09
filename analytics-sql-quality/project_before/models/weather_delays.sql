-- delays when it rains at the origin airport
select CASE WHEN w.precip > 0 THEN 'precipitation' ELSE 'dry' END as weather_at_departure,
       count(*) as flights,
       avg(try_cast(f.dep_delay as integer)) as avg_dep_delay_min
from raw_flights f
join raw_weather w on f.origin = w.origin and f.year = w.year and f.month = w.month and f.day = w.day
where f.dep_time != 'NA'
group by 1
