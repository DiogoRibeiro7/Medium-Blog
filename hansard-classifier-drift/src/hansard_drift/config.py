"""Study configuration, read from ``config/study.toml``."""

from __future__ import annotations

import datetime as dt
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Era:
    name: str
    speaker: str
    start: dt.date
    end: dt.date
    partial: bool = False


@dataclass(frozen=True)
class Government:
    start: dt.date
    end: dt.date
    parties: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    root: Path
    raw: dict = field(repr=False)
    eras: tuple[Era, ...]
    governments: tuple[Government, ...]

    # paths -------------------------------------------------------------
    @property
    def cache_dir(self) -> Path:
        return self.root / self.raw["data"]["cache_dir"]

    @property
    def build_dir(self) -> Path:
        return self.root / self.raw["data"]["build_dir"]

    @property
    def stage1_dir(self) -> Path:
        return self.root / "results" / "stage1"

    @property
    def stage2_dir(self) -> Path:
        return self.root / "results" / "stage2"

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    # sections ------------------------------------------------------------
    @property
    def data(self) -> dict:
        return self.raw["data"]

    @property
    def sample(self) -> dict:
        return self.raw["sample"]

    @property
    def estimate(self) -> dict:
        return self.raw["estimate"]

    @property
    def transfer(self) -> dict:
        return self.raw["transfer"]

    @property
    def lexical(self) -> dict:
        return self.raw["lexical"]

    @property
    def era_names(self) -> list[str]:
        return [e.name for e in self.eras]


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    eras = tuple(
        Era(
            name=e["name"],
            speaker=e["speaker"],
            start=e["start"],
            end=e["end"],
            partial=bool(e.get("partial", False)),
        )
        for e in raw["eras"]
    )
    for a, b in zip(eras, eras[1:], strict=False):
        if a.end != b.start:
            raise ValueError(f"eras {a.name} and {b.name} are not contiguous")
    governments = tuple(
        Government(start=g["start"], end=g["end"], parties=tuple(g["parties"])) for g in raw["government"]
    )
    return Config(root=path.parent.parent, raw=raw, eras=eras, governments=governments)
