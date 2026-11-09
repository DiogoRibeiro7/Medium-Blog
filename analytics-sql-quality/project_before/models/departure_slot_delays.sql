-- average minutes late by scheduled departure hour
SELECT hour AS sched_dep_hour, count(*) AS departures,
  avg(cast(dep_time AS integer) - sched_dep_time) AS avg_dep_delay_min
FROM raw_flights
WHERE dep_time <> 'NA'
GROUP BY hour
