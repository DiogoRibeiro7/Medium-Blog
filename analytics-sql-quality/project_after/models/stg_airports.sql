-- Grain: one row per FAA airport code.
select
    faa as airport_code,
    name as airport_name,
    tzone as time_zone
from raw_airports
