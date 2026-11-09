-- registered aircraft that did not fly out of NYC in a given month
select m.month as flight_month, count(*) as idle_aircraft
from (select distinct month from raw_flights) m, raw_planes p
where p.tailnum not in (select f.tailnum from raw_flights f where f.month = m.month)
group by m.month
