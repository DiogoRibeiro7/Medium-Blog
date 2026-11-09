-- Grain: one row per registered aircraft. year is the year of manufacture.
select
    tailnum,
    manufacturer,
    model,
    seats,
    cast(nullif(year, 'NA') as integer) as manufacture_year
from raw_planes
