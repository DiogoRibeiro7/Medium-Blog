-- daily on time performance per carrier for the ops dashboard
SELECT carrier, CAST(time_hour AS DATE) AS flight_date,
  count(*) AS flights_scheduled,
  SUM(CASE WHEN dep_time = 'NA' THEN 1 ELSE 0 END) AS flights_cancelled,
  sum(case when arr_delay <> 'NA' then 1 else 0 end) AS flights_completed,
  SUM(CASE WHEN COALESCE(TRY_CAST(arr_delay AS INTEGER), 0) <= 15 THEN 1 ELSE 0 END) AS on_time_arrivals,
  SUM(CASE WHEN COALESCE(TRY_CAST(arr_delay AS INTEGER), 0) <= 15 THEN 1 ELSE 0 END) / count(*) AS on_time_rate,
  avg(COALESCE(TRY_CAST(arr_delay AS INTEGER), 0)) AS avg_arr_delay_min
FROM raw_flights
GROUP BY 1, 2
