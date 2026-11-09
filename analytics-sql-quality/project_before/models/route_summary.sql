select f.origin, f.dest, a.name as dest_name, count(*) as flights,
    sum(case when f.arr_delay <> 'NA' then 1 else 0 end) as flights_completed,
    avg(try_cast(f.arr_delay as integer)) as avg_arr_delay_min, max(f.distance) as distance_miles
from raw_flights f join raw_airports a on f.dest = a.faa
group by f.origin, f.dest, a.name
