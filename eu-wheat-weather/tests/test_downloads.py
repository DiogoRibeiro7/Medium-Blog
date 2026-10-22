"""Offline tests of the download manifests, using local file URLs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eu_wheat_weather.cli import build_parser, main
from eu_wheat_weather.contract import EOBS_FILES, NUTS_VERSIONS, TARGET_NUTS, DataError
from eu_wheat_weather.downloads import (
    boundary_sources,
    fetch_all,
    file_sha256,
    verify,
    weather_sources,
)

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_records_hashes_and_verify_detects_changes(tmp_path: Path) -> None:
    source = tmp_path / "source.nc"
    source.write_bytes(b"grid" * 1000)
    manifest = tmp_path / "manifest.json"
    document = fetch_all({"a.nc": source.as_uri()}, tmp_path / "data", manifest, "test")
    entry = document["files"]["a.nc"]
    assert entry["bytes"] == 4000 and entry["sha256"] == file_sha256(source)
    assert verify(tmp_path / "data", manifest) == ["a.nc"]
    assert verify(tmp_path / "elsewhere", manifest) == []      # absent files are skipped
    (tmp_path / "data" / "a.nc").write_bytes(b"grid" * 999 + b"GRID")
    with pytest.raises(DataError, match="does not match"):
        verify(tmp_path / "data", manifest)
    assert not list((tmp_path / "data").glob("*.part"))


def test_sources_cover_the_registered_inputs() -> None:
    weather = weather_sources()
    assert set(weather) == set(EOBS_FILES.values())
    assert all(url.endswith(name) and url.startswith("https://") for name, url in weather.items())
    boundaries = boundary_sources()
    assert len(boundaries) == len(NUTS_VERSIONS)
    assert any(f"_{TARGET_NUTS}_" in name for name in boundaries)


def test_every_command_parses() -> None:
    parser = build_parser()
    for argv in (["fetch-crops", "--output-dir", "x"],
                 ["fetch-boundaries", "--manifest", "m.json"],
                 ["fetch-weather", "--manifest", "m.json"],
                 ["audit", "--crops", "x"]):
        assert parser.parse_args(argv).command == argv[0]


def test_audit_command_on_the_committed_snapshot(capsys: pytest.CaptureFixture[str]) -> None:
    snapshots = sorted((ROOT / "snapshots").glob("eurostat-crops-*"))
    if not snapshots:
        pytest.skip("No committed crop snapshot.")
    argv = ["audit", "--crops", str(snapshots[-1])]
    for manifest, flag in (("eobs-*.json", "--weather-manifest"),
                           ("gisco-nuts-*.json", "--boundaries-manifest")):
        found = sorted((ROOT / "snapshots").glob(manifest))
        if found:
            argv += [flag, str(found[-1])]
            json.loads(found[-1].read_text(encoding="utf-8"))
    assert main(argv) == 0
    assert "crops:" in capsys.readouterr().out
