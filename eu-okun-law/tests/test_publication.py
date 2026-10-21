"""Tests of the publication pipeline: every value quoted in the manuscript and the Medium
article comes from the committed results, the generated files are current, the audit
passes, the manifests hash what they list and the bibliography record is complete. Reads
the committed repository; no network."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from eu_okun.publication import (
    EXPERIMENTS,
    PLACEHOLDER,
    _sha256,
    bibliography_problems,
    change_log_entries,
    check,
    parse_bib,
    render_medium,
)
from eu_okun.publication_values import MINUS, Value, audit, collect_values, interval, num, pct

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "publication"
MANUSCRIPT = PUB / "manuscript"
SECTIONS = sorted((MANUSCRIPT / "sections").glob("*.tex"))
TEMPLATE = PUB / "medium" / "article.template.md"

# Numbers fixed by design (protocol thresholds, prior scales, horizons, lags, sampler
# settings, interval levels) or quoted from the literature. Anything else in the
# hand-written text must come from the values registry.
DESIGN_CONSTANTS = {
    "0.05", "0.1", "0.2", "0.25", "0.3", "0.5", "0.95", "1.01", "3.16",   # rules, priors
    "0", "1", "2", "4", "8", "12", "24",                                   # lags, horizons
    "5", "90", "95",                                                       # levels (%)
    "400", "1000", "3000",                                                 # sampler
    "10",                                                                  # 10 bp, F = 10
    "-0.3",                                                                # Okun (1962)
}
NUMBER = re.compile(r"(?<![\w.\-])-?\d+(?:\.\d+)?")
# Counts written as words, checked against the values they describe.
WORD_CLAIMS = [
    ("the only decided comparison", "pc_decided", 1),
    ("none of the registered", "fc_decided", 0),
    ("no registered forecast comparison is decided", "fc_decided", 0),
    ("the single decided gain", "pc_decided", 1),
    ("In the primary variant neither is decided", "cs_n_epl_decided_primary", 0),
    ("None is decided in the primary variant", "cs_n_epl_decided_primary", 0),
    ("are decided, both in the other variants", "cs_n_epl_decided", 2),
    ("only two non-primary variants are decided", "cs_n_epl_decided", 2),
    ("In the main version neither is decided", "cs_n_epl_decided_primary", 0),
    ("none is decided in the main version", "cs_n_epl_decided_primary", 0),
]


def _strip(text: str, tex: bool) -> str:
    """Text without comments, maths, macros' arguments, values, years and links."""
    if tex:
        text = re.sub(r"(?<!\\)%.*", "", text)
        text = re.sub(r"\\verb\|[^|]*\|", " ", text)
        text = re.sub(r"\\begin\{(verbatim|equation|align)\}.*?\\end\{\1\}", " ", text,
                      flags=re.S)
        text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.S)
        text = re.sub(r"\\val\{[^}]*\}", " ", text)
        text = re.sub(r"[0-9.]+\\textwidth", " ", text)
        text = re.sub(r"\\(label|ref|eqref|cite[pt]?|citetalias|input|includegraphics|"
                      r"addlinespace|setlength)(\[[^]]*\])?\{[^}]*\}", " ", text)
        text = text.replace("\\%", "%")
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = PLACEHOLDER.sub(" ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"\b(?:19|20)\d\d(?:-Q\d|-\d\d)?\b", " ", text)
    text = re.sub(r"\bEU-27\b|\b\w+-\d+\b", " ", text)
    # Names and labels that contain numbers: list numbering, headings, an index base,
    # a version and an operating system.
    text = re.sub(r"(?m)^\s*#*\s*\d+\.\s", " ", text)
    text = re.sub(r"=\d+|\bversion \d+|\bWindows \d+", " ", text)
    return text


@pytest.fixture(scope="module")
def values() -> dict[str, Value]:
    return collect_values(ROOT)


def test_every_val_in_the_manuscript_is_defined(values: dict[str, Value]) -> None:
    def text(path: Path) -> str:
        return re.sub(r"\\verb\|[^|]*\|", "", path.read_text(encoding="utf-8"))

    used = {k for path in SECTIONS for k in re.findall(r"\\val\{([^}]+)\}", text(path))}
    assert used, "the manuscript quotes no values"
    assert sorted(used - set(values)) == []


def test_every_placeholder_in_the_medium_template_is_defined(values: dict[str, Value]) -> None:
    used = set(PLACEHOLDER.findall(TEMPLATE.read_text(encoding="utf-8")))
    assert used
    assert sorted(used - set(values)) == []


@pytest.mark.parametrize("path", [*SECTIONS, TEMPLATE], ids=lambda p: p.name)
def test_no_result_is_typed_by_hand(path: Path) -> None:
    body = _strip(path.read_text(encoding="utf-8"), path.suffix == ".tex")
    found = {m.group(0) for m in NUMBER.finditer(body)}
    assert sorted(found - DESIGN_CONSTANTS) == [], "use \\val{} or {{}} for results"


def test_counts_written_as_words_match_the_values(values: dict[str, Value]) -> None:
    text = " ".join(re.sub(r"\s+", " ", p.read_text(encoding="utf-8"))
                    for p in [*SECTIONS, TEMPLATE])
    for phrase, key, expected in WORD_CLAIMS:
        if phrase in text:
            assert values[key].raw == expected, (phrase, key)


def test_strip_keeps_numbers_in_prose() -> None:
    assert NUMBER.findall(_strip("in 22 of 27 countries, $t-4$, 2016--2019", True)) == [
        "22", "27"]
    assert NUMBER.findall(_strip("gains of 3.1\\% in EU-27 (\\val{x})", True)) == ["3.1"]


def test_values_are_read_from_the_committed_results(values: dict[str, Value]) -> None:
    raw = {k: v.raw for k, v in values.items()}
    assert raw["n_countries"] == 27
    mu, tau = float(raw["by_mu"]), float(raw["by_tau"])
    assert mu < 0 < tau
    low, high = (float(x) for x in str(raw["by_mu_ci"]).strip("[]").split(","))
    assert low < mu < high
    assert int(raw["fc_decided"]) <= int(raw["fc_total"])
    assert values["claim_es_verdict"].raw in ("supported", "contradicted", "not resolved")
    for key, v in values.items():
        assert v.source, key
        assert "$" not in v.md, key


def test_value_formatting() -> None:
    v = num("k", -0.1234, 2, "src", signed=True)
    assert (v.tex, v.md) == ("$-0.12$", f"{MINUS}0.12")
    assert pct("p", 0.153, 1, "src").md == "15.3%"
    assert interval("i", -math.inf, 0.5, 2, "src").tex == "$[-\\infty, +0.50]$"


def test_the_audit_passes() -> None:
    failed = [f"{c.group}: {c.name}: {c.detail}" for c in audit(ROOT) if not c.passed]
    assert failed == []


def test_generated_files_are_current() -> None:
    assert check(ROOT) == []


def test_medium_article_is_medium_compatible() -> None:
    article = (PUB / "medium" / "article.md").read_text(encoding="utf-8")
    assert "$" not in article and "{{" not in article
    assert not re.search(r"^\s*\|.*\|\s*$", article, re.M)
    for image in re.findall(r"!\[[^]]*\]\(([^)]+)\)", article):
        assert (PUB / "medium" / image).exists(), image


def test_render_medium_refuses_what_medium_cannot_show(values: dict[str, Value]) -> None:
    with pytest.raises(KeyError):
        render_medium("{{no_such_value}}", values)
    with pytest.raises(ValueError):
        render_medium("costs $5", values)
    with pytest.raises(ValueError):
        render_medium("| a | b |\n| --- | --- |", values)
    assert render_medium("{{n_countries}} countries", values) == "27 countries"


def test_every_figure_in_the_manuscript_exists() -> None:
    for path in [MANUSCRIPT / "main.tex", *SECTIONS]:
        for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",
                               path.read_text(encoding="utf-8")):
            assert (MANUSCRIPT / "figures" / name).exists(), name


def test_every_citation_is_in_the_bibliography_and_verified() -> None:
    entries = {e["key"] for e in parse_bib((MANUSCRIPT / "references.bib")
                                          .read_text(encoding="utf-8"))}
    cited: set[str] = set()
    for path in SECTIONS:
        for group in re.findall(r"\\cite(?:t|p|talias|palias)?\{([^}]+)\}",
                                path.read_text(encoding="utf-8")):
            cited |= {k.strip() for k in group.split(",")}
    assert sorted(cited - entries) == []
    assert sorted(entries - cited) == [], "uncited entries do not appear in the article"
    record = json.loads((PUB / "manifest" / "bibliography.json").read_text(encoding="utf-8"))
    assert set(record) == entries
    assert bibliography_problems(record) == []


def test_bibliography_problems_are_detected() -> None:
    record: dict[str, dict[str, Any]] = {
              "a": {"doi": "10.1/x", "resolved": False},
              "b": {"doi": "10.1/y", "resolved": True, "checked": "title",
                    "title_matches": False},
              "c": {"doi": None, "url": "https://example.org", "http_status": 404},
              "d": {"doi": None, "url": None, "isbn": None, "note": None},
              "e": {"doi": None, "url": None, "isbn": "9781439840955", "note": None}}
    assert [p.split(":")[0] for p in bibliography_problems(record)] == ["a", "b", "c", "d"]


def test_manifests_hash_the_files_they_list() -> None:
    data = json.loads((PUB / "manifest" / "data.json").read_text(encoding="utf-8"))
    files = [f for snapshot in data["snapshots"].values() for f in snapshot]
    assert files
    for f in files:
        assert _sha256(ROOT / f["path"]) == f["sha256"], f["path"]
    experiments = json.loads((PUB / "manifest" / "experiments.json")
                             .read_text(encoding="utf-8"))["experiments"]
    assert [e["key"] for e in experiments] == [e["key"] for e in EXPERIMENTS]
    for e in experiments:
        assert _sha256(ROOT / e["protocol"]) == e["protocol_sha256"]
        assert _sha256(ROOT / e["report"]) == e["report_sha256"]
        for f in e["outputs"]:
            assert _sha256(ROOT / f["path"]) == f["sha256"], f["path"]


def test_text_hashes_ignore_line_endings(tmp_path: Path) -> None:
    lf, crlf = tmp_path / "a.csv", tmp_path / "b.csv"
    lf.write_bytes(b"x,y\n1,2\n")
    crlf.write_bytes(b"x,y\r\n1,2\r\n")
    assert _sha256(lf) == _sha256(crlf)


def test_change_log_entries(tmp_path: Path) -> None:
    protocol = tmp_path / "P.md"
    protocol.write_text("# P\n\n- not counted\n\n## Change log\n\n- one\n  - nested\n- two\n"
                        "\n## Later\n\n- not counted\n", encoding="utf-8")
    assert change_log_entries(protocol) == 2
    protocol.write_text("# P\n\n- no change log\n", encoding="utf-8")
    assert change_log_entries(protocol) == 0


def test_every_registered_command_is_documented_and_parses(
        capsys: pytest.CaptureFixture[str]) -> None:
    from eu_okun.cli import main

    readme = (PUB / "README.md").read_text(encoding="utf-8")
    for e in EXPERIMENTS:
        for command in e["commands"]:
            assert f"poetry run {command}\n" in readme, command
            name, *args = command.split()[1:]
            with pytest.raises(SystemExit):
                main([name, "--help"])
            usage = capsys.readouterr().out
            for option in (a for a in args if a.startswith("--")):
                assert option in usage, f"{name} {option}"


def test_every_protocol_is_registered() -> None:
    for e in EXPERIMENTS:
        assert (ROOT / e["protocol"]).exists()
        assert (ROOT / e["report"]).exists()
        commit, date = e["registered"].split()
        assert re.fullmatch(r"[0-9a-f]{7}", commit) and re.fullmatch(r"2026-10-\d\d", date)
