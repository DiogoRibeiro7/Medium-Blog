"""Hand labels: a terminal labelling tool, agreement between annotators, and the label set
the estimates use.

Files in ``labels/``:

* ``sample.csv``          the blind sample (written by ``sample``)
* ``design.csv``          strata and weights (do not open it before labelling)
* ``labels_<name>.csv``   one file per annotator: id, discipline (1/0), unsure, note
* ``labels.csv``          the adjudicated set, if there is more than one annotator
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

LABEL_COLUMNS = ["id", "discipline", "unsure", "note"]


def annotator_path(cfg: Config, name: str) -> Path:
    safe = "".join(ch for ch in name.lower() if ch.isalnum() or ch in "-_")
    if not safe:
        raise ValueError("annotator name must contain letters or digits")
    return cfg.labels_dir / f"labels_{safe}.csv"


def read_annotator(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    df = pd.read_csv(path, dtype={"id": str, "note": str}, keep_default_na=False)
    return df.drop_duplicates("id", keep="last")


def annotate(cfg: Config, name: str, width: int = 100) -> int:
    """Label the unlabelled items, one at a time. Saves after every answer."""
    sample = pd.read_csv(cfg.labels_dir / "sample.csv", dtype=str, keep_default_na=False)
    path = annotator_path(cfg, name)
    done = set(read_annotator(path)["id"])
    todo = sample[~sample["id"].isin(done)]
    print(f"{len(done)} labelled, {len(todo)} to go. Codebook: labels/CODEBOOK.md")
    print("Answer 1 = disciplinary intervention, 0 = not, u = unsure, n = add a note first, q = quit.\n")
    if not path.exists():
        path.write_text(",".join(LABEL_COLUMNS) + "\n", encoding="utf-8")

    def wrap(s: str) -> str:
        return textwrap.fill(s, width=width, initial_indent="    ", subsequent_indent="    ") if s else "    (none)"

    for _, r in todo.iterrows():
        print("=" * width)
        print(f"item {r['item']}  ·  {r['date']}")
        print(f"\n  BEFORE  {r['before_speaker']}\n{wrap(r['before_text'])}")
        print(f"\n  CHAIR   {r['chair_speaker']}\n{wrap(r['chair_text'])}")
        print(f"\n  AFTER   {r['after_speaker']}\n{wrap(r['after_text'])}\n")
        note = ""
        while True:
            ans = input("label [1/0/u/n/q]: ").strip().lower()
            if ans == "q":
                return 0
            if ans == "n":
                note = input("note: ").strip().replace(",", ";")
                continue
            if ans in {"1", "0", "u"}:
                break
        disc = "" if ans == "u" else ans
        unsure = "1" if ans == "u" else "0"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"{r['id']},{disc},{unsure},{note}\n")
    print("All items labelled.")
    return 0


def annotator_files(cfg: Config) -> list[Path]:
    return sorted(cfg.labels_dir.glob("labels_*.csv"))


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    po = float(np.mean(a == b))
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss_kappa(votes: np.ndarray) -> float:
    """Fleiss' kappa for an items x categories matrix of vote counts (equal raters per item)."""
    votes = np.asarray(votes, dtype=float)
    n = votes.sum(axis=1)
    if len(votes) == 0 or not np.all(n == n[0]) or n[0] < 2:
        return np.nan
    n = n[0]
    p_j = votes.sum(axis=0) / votes.sum()
    p_i = (np.sum(votes**2, axis=1) - n) / (n * (n - 1))
    pe = float(np.sum(p_j**2))
    return (float(p_i.mean()) - pe) / (1 - pe) if pe < 1 else 1.0


def _decisive(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One column per annotator, items labelled 0/1 by every annotator only."""
    wide = None
    for name, f in frames.items():
        col = f[["id", "discipline"]].rename(columns={"discipline": name})
        wide = col if wide is None else wide.merge(col, on="id", how="outer")
    wide = wide.set_index("id")
    ok = wide.isin(["0", "1", 0, 1]).all(axis=1)
    return wide[ok].astype(int)


def agreement(cfg: Config) -> pd.DataFrame:
    files = annotator_files(cfg)
    if len(files) < 2:
        raise RuntimeError("agreement needs at least two labels_<name>.csv files")
    frames = {f.stem.removeprefix("labels_"): read_annotator(f) for f in files}
    rows, disagreements = [], []
    names = list(frames)
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            m = frames[x].merge(frames[y], on="id", suffixes=("_a", "_b"))
            m = m[(m["discipline_a"].astype(str) != "") & (m["discipline_b"].astype(str) != "")]
            a = m["discipline_a"].astype(int).to_numpy()
            b = m["discipline_b"].astype(int).to_numpy()
            rows.append(
                {
                    "annotator_a": x,
                    "annotator_b": y,
                    "items": len(m),
                    "raw_agreement": float(np.mean(a == b)) if len(m) else np.nan,
                    "kappa": cohen_kappa(a, b) if len(m) else np.nan,
                    "a_only_positive": int(((a == 1) & (b == 0)).sum()),
                    "b_only_positive": int(((a == 0) & (b == 1)).sum()),
                }
            )
            d = m[m["discipline_a"].astype(int) != m["discipline_b"].astype(int)]
            disagreements.append(d.assign(annotator_a=x, annotator_b=y))
    out = pd.DataFrame(rows)
    out.to_csv(cfg.labels_dir / "agreement.csv", index=False)
    if disagreements:
        pd.concat(disagreements).to_csv(cfg.labels_dir / "disagreements.csv", index=False)

    # Fleiss' kappa over all annotators, overall and within groups of the design. A
    # reference whose agreement falls in some eras is itself drifting.
    wide = _decisive(frames)
    design_path = cfg.labels_dir / "design.csv"
    if design_path.exists() and len(frames) >= 2:
        design = pd.read_csv(design_path, dtype={"id": str}).set_index("id")
        wide = wide.join(design[["era", "d_tag"]], how="left")
        names = list(frames)
        grp_rows = []
        groups = [("all", wide)]
        groups += [(f"era={e}", g) for e, g in wide.groupby("era", sort=False)]
        groups += [(f"d_tag={t}", g) for t, g in wide.groupby("d_tag")]
        for label, g in groups:
            ones = g[names].sum(axis=1).to_numpy()
            votes = np.column_stack([len(names) - ones, ones])
            grp_rows.append(
                {
                    "group": label,
                    "items": len(g),
                    "fleiss_kappa": fleiss_kappa(votes),
                    "unanimous": float(np.mean((ones == 0) | (ones == len(names)))) if len(g) else np.nan,
                    "share_positive_majority": float(np.mean(ones * 2 > len(names))) if len(g) else np.nan,
                }
            )
        pd.DataFrame(grp_rows).to_csv(cfg.labels_dir / "agreement_by_group.csv", index=False)
    return out


def adjudicate(cfg: Config) -> pd.DataFrame:
    """Majority vote across the annotator files into labels/labels.csv.

    An item gets the label a strict majority of all annotators gave it; with no majority
    (for example one 1, one 0 and one unsure) it is left unsure and out of the estimates.
    """
    files = annotator_files(cfg)
    if len(files) < 2:
        raise RuntimeError("adjudication needs at least two labels_<name>.csv files")
    frames = {f.stem.removeprefix("labels_"): read_annotator(f) for f in files}
    names = list(frames)
    ids = sorted(set().union(*[set(f["id"]) for f in frames.values()]))
    vote = pd.DataFrame(index=ids)
    for n, f in frames.items():
        vote[n] = f.set_index("id")["discipline"].astype(str).reindex(ids).fillna("")
    k = len(names)
    ones = (vote == "1").sum(axis=1)
    zeros = (vote == "0").sum(axis=1)
    label = np.where(ones * 2 > k, "1", np.where(zeros * 2 > k, "0", ""))
    out = pd.DataFrame(
        {
            "id": ids,
            "discipline": label,
            "unsure": np.where(label == "", 1, 0),
            "note": [f"votes 1:{a} 0:{b} of {k}" for a, b in zip(ones, zeros, strict=True)],
        }
    )
    out.to_csv(cfg.labels_dir / "labels.csv", index=False)
    return out


def final_labels(cfg: Config) -> tuple[pd.DataFrame, str]:
    """The labels the estimates use: the adjudicated file, or the only annotator's."""
    adjudicated = cfg.labels_dir / "labels.csv"
    if adjudicated.exists():
        df, source = read_annotator(adjudicated), "labels.csv"
    else:
        files = annotator_files(cfg)
        if len(files) != 1:
            raise RuntimeError(
                "no labels/labels.csv and not exactly one labels_<name>.csv; adjudicate the annotators first"
            )
        df, source = read_annotator(files[0]), files[0].name
    df = df[df["discipline"].astype(str).isin(["0", "1"])].copy()
    df["discipline"] = df["discipline"].astype(int)
    return df[["id", "discipline"]], source
