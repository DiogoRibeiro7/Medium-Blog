import json
from datetime import date

from swaptails.calendar import build_calendar, note
from swaptails.llm import SCHEMA, OllamaBackend, annotate, latest_labels, parse_answer, read_labels


def test_parse_answer():
    assert parse_answer('{"day_type": "sunday_like"}') == "sunday_like"
    assert parse_answer('```json\n{"day_type": "workday"}\n```') == "workday"
    assert parse_answer('Sure! {"day_type": "saturday_like"} hope that helps') == "saturday_like"
    assert parse_answer('{"day_type": "holiday"}') is None
    assert parse_answer("sunday") is None
    assert parse_answer("") is None


def test_annotate_caches_answers(cfg):
    cal = build_calendar(cfg.countries["de"], date(2025, 1, 1), date(2025, 12, 31))
    items = [("de", d, "holiday", note(cal, d)) for d in (date(2025, 5, 29), date(2025, 6, 19), date(2025, 7, 12))]
    path = annotate(cfg, "rule_eu", items)
    rows = read_labels(path)
    assert [r["label"] for r in rows] == ["sunday_like", "workday", "saturday_like"]
    annotate(cfg, "rule_eu", items)  # nothing new to ask
    assert len(read_labels(path)) == 3
    labels = latest_labels(cfg, "rule_eu")
    assert labels[("de", "2025-05-29")]["valid"]


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class _Session:
    def __init__(self):
        self.sent = None

    def post(self, url, json=None, timeout=None):
        self.sent = (url, json)
        return _Response({"message": {"content": '{"day_type": "workday"}'}})


def test_ollama_request(cfg):
    from dataclasses import replace

    spec = replace(cfg.llms["rule_eu"], backend="ollama", model="qwen3.5:9b", think=False)
    session = _Session()
    backend = OllamaBackend(spec, base_url="localhost:11434", session=session)
    assert parse_answer(backend.ask("Date: Monday 6 January 2025.", seed=7)) == "workday"
    url, body = session.sent
    assert url == "http://localhost:11434/api/chat"
    assert body["format"] == SCHEMA and body["think"] is False
    assert body["options"]["temperature"] == 0 and body["options"]["seed"] == 7
    assert json.dumps(body)  # serialisable
