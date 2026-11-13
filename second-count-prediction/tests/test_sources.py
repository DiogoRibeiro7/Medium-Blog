import io

from openpyxl import Workbook

from feed import sources as S


def test_report_month_from_each_naming_scheme():
    assert (
        S.report_month("https://s3.amazonaws.com/citibike-regunits/pdf/2015_09_September_Citi_Bike_Monthly_Report.pdf")
        == "2015-09"
    )
    assert (
        S.report_month("https://d21xlh2maitm24.cloudfront.net/nyc/January-2020-Citi-Bike-Monthly-Report-1.pdf?mtime=1")
        == "2020-01"
    )
    assert S.report_month("https://x/nyc/September-2026-Citi-Bike-Monthly-Report.pdf") == "2026-09"
    assert S.report_month("https://x/nyc/Citi-Bike-Report.pdf") is None


def test_report_links_keep_the_first_link_of_each_month():
    html = (
        '<a href="https://a/May-2024-Citi-Bike-Monthly-Report.pdf">x</a>'
        '<a href="https://b/May-2024-Citi-Bike-Monthly-Report-2.pdf">'
    )
    assert S.report_links(html) == {"2024-05": "https://a/May-2024-Citi-Bike-Monthly-Report.pdf"}


def test_report_total_patterns_in_order_and_floor():
    assert S.report_total("Ridership\nThere were 3,942,643 trips in April, with an average") == (3942643, 0)
    assert S.report_total("In September, Citi Bike riders took 1,289,709 trips and traveled") == (1289709, 1)
    assert S.report_total("There were 77,698 total bicycles rebalanced") == (None, None)
    assert S.report_total("there were 5,000 trips in March") == (None, None)  # below the floor


def test_dot_table_from_pdf_text_skips_partial_months():
    text = "Month Total Trips\n2013-05* 39,621\n2013-06 585,929\n2013-07 900,397\n2024-Q1 6,667,687"
    assert S.dot_months_from_text(text) == {"2013-06": 585929, "2013-07": 900397}


def test_dot_table_from_xlsx():
    wb = Workbook()
    ws = wb.active
    ws.append(["Citi Bike Customer Trips by Month"])
    ws.append(["Month", "Total Trips"])
    ws.append(["2013-05*", 39621])
    ws.append(["2013-06", 585929])
    ws.append(["2024-Q1", 6667687])
    buf = io.BytesIO()
    wb.save(buf)
    assert S.dot_months_from_xlsx(buf.getvalue()) == {"2013-06": 585929}


def test_month_helpers():
    assert S.month_range("2024-11", "2025-02") == ["2024-11", "2024-12", "2025-01", "2025-02"]
    assert S.next_month("2024-12") == "2025-01"


def test_chicago_trips_resumes_and_retries(tmp_path, monkeypatch):
    calls = []

    def fake(ds, m):
        calls.append((ds, m))
        if m == "2018-12" and calls.count((ds, m)) == 1:
            raise RuntimeError("no answer")
        return 10

    monkeypatch.setattr(S, "TRIPS", {"a": ("2018-11", "2019-01")})
    monkeypatch.setattr(S, "count_month", fake)
    monkeypatch.setattr(S, "count_dataset", lambda ds: (_ for _ in ()).throw(RuntimeError("no answer")))
    monkeypatch.setattr(S, "log", lambda msg: None)
    prog = tmp_path / "p.json"
    prog.write_text('{"a 2018-11": 7}')
    out = S.chicago_trips("2026-11", prog, passes=2, workers=1)
    assert out == {
        "2018-11": {"trips": 7, "datasets": ["a"]},
        "2018-12": {"trips": 10, "datasets": ["a"]},
        "2019-01": {"trips": 10, "datasets": ["a"]},
    }
    assert ("a", "2018-11") not in calls and calls.count(("a", "2018-12")) == 2


def test_chicago_trips_takes_the_grouped_answer_for_missing_months(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "TRIPS", {"a": ("2018-11", "2019-01")})
    monkeypatch.setattr(S, "count_dataset", lambda ds: {"2018-12": 5, "2019-01": 6, "2018-11": 99})
    monkeypatch.setattr(S, "count_month", lambda ds, m: (_ for _ in ()).throw(AssertionError("not needed")))
    monkeypatch.setattr(S, "log", lambda msg: None)
    prog = tmp_path / "p.json"
    prog.write_text('{"a 2018-11": 7}')
    out = S.chicago_trips("2026-11", prog)
    assert {m: v["trips"] for m, v in out.items()} == {"2018-11": 7, "2018-12": 5, "2019-01": 6}


def test_diagnose_contexts_show_the_matched_sentence():
    from diagnose import contexts

    out = contexts("Ridership\nThere were 30,827,926 trips in July, with an average of 99,445 per day", width=20)
    assert out and "30,827,926 trips in" in out[0] and out[0].startswith("pattern 0")
