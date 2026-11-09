-- The shipped idle_aircraft model pointed at the typed staging tables, with
-- its NOT IN kept: what the model becomes once 'NA' is loaded as NULL.
select m.flight_month, count(*) as idle_aircraft
from (select distinct flight_month from stg_flights) m, stg_planes p
where p.tailnum not in (select f.tailnum from stg_flights f where f.flight_month = m.flight_month)
group by m.flight_month
