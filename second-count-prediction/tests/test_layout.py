from feed import layout as L


def f(path, months, sha=None):
    return {"path": path, "rows": sum(months.values()), "months": months, "sha256": sha or path}


def test_whole_file_and_parts_in_a_folder_are_one_copy():
    files = [
        f("2013/201308-citibike-tripdata.csv", {"2013-08": 1001958}),
        f("2013/8_August/201308-citibike-tripdata_1.csv", {"2013-08": 1000000}),
        f("2013/8_August/201308-citibike-tripdata_2.csv", {"2013-08": 1958}),
    ]
    months, dropped, flagged = L.corrected(files)
    assert months == {"2013-08": 1001958} and len(dropped) == 2 and not flagged
    reg, _ = L.registered(files)
    assert reg == {"2013-08": 2003916}  # the registered rule counts the month twice (as in the real 2013-08)


def test_equal_size_parts_are_all_counted():
    files = [f(f"202405-citibike-tripdata_{i}.csv", {"2024-05": 1000000}) for i in (1, 2, 3, 4)]
    files.append(f("202405-citibike-tripdata_5.csv", {"2024-05": 133961}))
    months, dropped, _ = L.corrected(files)
    assert months == {"2024-05": 4133961} and not dropped
    reg, reg_dropped = L.registered(files)
    assert reg == {"2024-05": 1133961} and len(reg_dropped) == 3  # the registered rule drops equal parts


def test_dash_part_naming_and_identical_files():
    files = [f(f"202604-citibike-tripdata-part{i}.csv", {"2026-04": 965093 - (i > 1)}) for i in (1, 2, 3, 4)]
    files.append(
        f("copy/202604-citibike-tripdata-part1.csv", {"2026-04": 965093}, sha="202604-citibike-tripdata-part1.csv")
    )
    months, dropped, _ = L.corrected(files)
    assert months == {"2026-04": 965093 + 3 * 965092}
    assert dropped == ["copy/202604-citibike-tripdata-part1.csv"]


def test_copies_that_differ_are_kept_and_flagged():
    files = [
        f("a/201901-citibike-tripdata.csv", {"2019-01": 10}),
        f("a/1_Jan/201901-citibike-tripdata_1.csv", {"2019-01": 9}),
    ]
    months, dropped, flagged = L.corrected(files)
    assert months == {"2019-01": 19} and flagged == ["a/1_Jan/201901-citibike-tripdata_1.csv"]


def test_stem():
    assert L.stem("201808-citibike-tripdata_12.csv") == "201808-citibike-tripdata"
    assert L.stem("202604-citibike-tripdata-part3.csv") == "202604-citibike-tripdata"
    assert L.stem("201808-citibike-tripdata.csv") == "201808-citibike-tripdata"
