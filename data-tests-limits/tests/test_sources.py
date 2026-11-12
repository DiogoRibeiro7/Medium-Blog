import pyarrow as pa
import pyarrow.parquet as pq

from feed import sources


def test_footer_reads_rows_and_schema_through_ranges(tmp_path, monkeypatch):
    p = tmp_path / "x.parquet"
    pq.write_table(
        pa.table({"a": list(range(200_000)), "b": [str(i) for i in range(200_000)]}), p, row_group_size=60_000
    )
    data = p.read_bytes()
    calls = []

    def fake_get(u, headers=None, tries=5):
        lo, hi = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        calls.append(hi - lo + 1)
        return data[lo : hi + 1]

    def fake_init(self, u):
        self.u, self.size, self.pos = u, len(data), 0

    monkeypatch.setattr(sources, "get", fake_get)
    monkeypatch.setattr(sources.RangeFile, "__init__", fake_init)
    r = sources.footer("green", "2024-01")
    assert r["rows"] == 200_000 and r["row_groups"] == 4
    assert r["schema"] == [["a", "int64"], ["b", "string"]]
    assert sum(calls) < len(data) / 10  # only the tail of the file was read


def test_months():
    assert sources.months("2019-11", "2020-02") == ["2019-11", "2019-12", "2020-01", "2020-02"]
