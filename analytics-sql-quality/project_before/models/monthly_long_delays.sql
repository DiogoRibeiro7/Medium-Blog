select carrier, month as flight_month,
  sum(case when dep_time <> 'NA' then 1 else 0 end) as departures,
  sum(case when dep_delay > '60' then 1 else 0 end) as departures_over_60_min,
  sum(case when dep_delay > '60' then 1 else 0 end) / sum(case when dep_time <> 'NA' then 1 else 0 end) as share_over_60_min
from raw_flights
group by carrier, month
