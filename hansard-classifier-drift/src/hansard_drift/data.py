"""Download the corpus and read it back in chunks."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .config import Config

COLUMNS = [
    "id",
    "speech",
    "display_as",
    "party",
    "date",
    "colnum",
    "speech_class",
    "speakername",
    "person_id",
    "parliamentary_posts",
]


def fetch(cfg: Config) -> dict:
    """Download the parquet conversion from the Hugging Face hub into the cache."""
    from huggingface_hub import HfApi, snapshot_download

    d = cfg.data
    api = HfApi()
    info = api.dataset_info(d["repo_id"], revision=d["revision"])
    path = snapshot_download(
        repo_id=d["repo_id"],
        repo_type="dataset",
        revision=d["revision"],
        allow_patterns=[d["pattern"]],
        local_dir=str(cfg.cache_dir),
    )
    files = sorted(Path(path).glob(d["pattern"]))
    if not files:
        raise RuntimeError(f"no files matching {d['pattern']} in {path}")
    record = {
        "repo_id": d["repo_id"],
        "revision": d["revision"],
        "commit": info.sha,
        "downloaded_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "files": [
            {"name": f.relative_to(cfg.cache_dir).as_posix(), "bytes": f.stat().st_size, "sha256": _sha256(f)}
            for f in files
        ],
    }
    (cfg.cache_dir / "fetch_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parquet_files(cfg: Config) -> list[Path]:
    files = sorted(cfg.cache_dir.glob(cfg.data["pattern"]))
    if not files:
        raise FileNotFoundError(f"no parquet files under {cfg.cache_dir}; run `hansard-drift fetch` first")
    return files


def fetch_manifest(cfg: Config) -> dict:
    p = cfg.cache_dir / "fetch_manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def iter_chunks(cfg: Config, columns: list[str] | None = None, batch_size: int = 200_000) -> Iterator[pd.DataFrame]:
    columns = columns or COLUMNS
    for f in parquet_files(cfg):
        pf = pq.ParquetFile(f)
        available = set(pf.schema_arrow.names)
        cols = [c for c in columns if c in available]
        for batch in pf.iter_batches(columns=cols, batch_size=batch_size):
            df = batch.to_pandas()
            for c in columns:
                if c not in df:
                    df[c] = None
            yield df
