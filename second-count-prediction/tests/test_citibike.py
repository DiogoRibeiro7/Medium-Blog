import io
import zipfile

import duckdb

from feed import citibike as C


def csv_bytes(header, rows):
    return (header + "\n" + "\n".join(rows) + "\n").encode()


def test_year_keys_new_york_only_with_monthly_flag():
    bucket = [
        {"key": "2013-citibike-tripdata.zip", "size": 1},
        {"key": "202401-citibike-tripdata.zip", "size": 2},
        {"key": "202401-citibike-tripdata.csv.zip", "size": 3},
        {"key": "JC-202401-citibike-tripdata.csv.zip", "size": 4},
        {"key": "index.html", "size": 5},
    ]
    assert [k["key"] for k in C.year_keys(bucket, 2013)] == ["2013-citibike-tripdata.zip"]
    ks = C.year_keys(bucket, 2024)
    assert [k["key"] for k in ks] == ["202401-citibike-tripdata.zip", "202401-citibike-tripdata.csv.zip"]
    assert all(k["monthly"] and k["file_month"] == "2024-01" for k in ks)


def test_walk_zip_records_every_csv_with_months_and_hash(tmp_path):
    old = csv_bytes(
        '"tripduration","starttime","stoptime"',
        ['"300","2013-06-30 23:59:00","2013-07-01 00:04:00"', '"200","2013-07-01 00:01:00","2013-07-01 00:04:00"'],
    )
    us = csv_bytes(
        "tripduration,starttime,stoptime",
        ["300,7/31/2015 23:50,8/1/2015 0:10", "10,8/1/2015 00:01:02,x", "5,,x"],
    )
    new = csv_bytes(
        "ride_id,rideable_type,started_at,ended_at",
        ["a,classic,2024-01-31 23:59:59.123,2024-02-01 00:10:00", "b,electric,2024-02-01 00:00:01,2024-02-01 00:10:00"],
    )
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("2015/201508-citibike-tripdata.csv", us)
        z.writestr("__MACOSX/2015/._201508-citibike-tripdata.csv", b"junk")
    outer = tmp_path / "x.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("201307-citibike-tripdata.csv", old)
        z.writestr("2015.zip", inner.getvalue())
        z.writestr("202402-citibike-tripdata_1.csv", new)
        z.writestr("copy/201307-citibike-tripdata.csv", old)
    files = []
    C.walk_zip(duckdb.connect(), outer, tmp_path, files)
    assert [x["path"] for x in files] == [
        "201307-citibike-tripdata.csv",
        "2015/201508-citibike-tripdata.csv",
        "202402-citibike-tripdata_1.csv",
        "copy/201307-citibike-tripdata.csv",
    ]
    assert files[0]["months"] == {"2013-06": 1, "2013-07": 1} and files[0]["sha256"] == files[3]["sha256"]
    assert files[1]["months"] == {"2015-07": 1, "2015-08": 1} and files[1]["unparsed"] == 1
    assert files[2]["months"] == {"2024-01": 1, "2024-02": 1}
