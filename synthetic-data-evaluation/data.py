"""Load the Adult census extract and split it into three disjoint real samples.

The file is the PMLB copy of the UCI Adult data (48,842 rows, Kohavi 1996).
PMLB stores every categorical column as an integer code assigned in
alphabetical order of the original labels, and codes the target as
'<=50K' = 1, '>50K' = 0. Both are decoded here, so the rest of the code
works with the original labels and with income coded as '>50K' / '<=50K'.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = (
    "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/"
    "datasets/adult/adult.tsv.gz"
)
SHA256 = "fb8bc76eafa39558b6044907f23416faf99099cefb8afd06a2cf5afbdae1b506"
HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "adult.tsv.gz"

# Labels in PMLB code order (alphabetical), from the dataset's metadata.yaml.
LABELS: dict[str, list[str]] = {
    "workclass": [
        "?", "Federal-gov", "Local-gov", "Never-worked", "Private",
        "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay",
    ],
    "education": [
        "10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th",
        "Assoc-acdm", "Assoc-voc", "Bachelors", "Doctorate", "HS-grad",
        "Masters", "Preschool", "Prof-school", "Some-college",
    ],
    "marital-status": [
        "Divorced", "Married-AF-spouse", "Married-civ-spouse",
        "Married-spouse-absent", "Never-married", "Separated", "Widowed",
    ],
    "occupation": [
        "?", "Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial",
        "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct",
        "Other-service", "Priv-house-serv", "Prof-specialty",
        "Protective-serv", "Sales", "Tech-support", "Transport-moving",
    ],
    "relationship": [
        "Husband", "Not-in-family", "Other-relative", "Own-child",
        "Unmarried", "Wife",
    ],
    "race": [
        "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White",
    ],
    "sex": ["Female", "Male"],
    "native-country": [
        "?", "Cambodia", "Canada", "China", "Columbia", "Cuba",
        "Dominican-Republic", "Ecuador", "El-Salvador", "England", "France",
        "Germany", "Greece", "Guatemala", "Haiti", "Holand-Netherlands",
        "Honduras", "Hong", "Hungary", "India", "Iran", "Ireland", "Italy",
        "Jamaica", "Japan", "Laos", "Mexico", "Nicaragua",
        "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines", "Poland",
        "Portugal", "Puerto-Rico", "Scotland", "South", "Taiwan", "Thailand",
        "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia",
    ],
}

NUMERIC = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
    "hours-per-week",
]
CATEGORICAL = [
    "workclass", "education", "marital-status", "occupation", "relationship",
    "race", "sex", "native-country", "income",
]
# Column order used by every generator. Income comes last so that sequential
# generators synthesise it conditionally on everything else.
COLUMNS = [
    "age", "sex", "race", "native-country", "education", "education-num",
    "marital-status", "relationship", "workclass", "occupation",
    "hours-per-week", "capital-gain", "capital-loss", "fnlwgt", "income",
]


def download(path: Path = RAW) -> Path:
    """Download the PMLB file once and check its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"{path} has SHA-256 {digest}, expected {SHA256}")
    return path


def load_adult() -> pd.DataFrame:
    """Return the 48,842 Adult records with the original labels."""
    raw = pd.read_csv(download(), sep="\t")
    df = pd.DataFrame(index=raw.index)
    for col in COLUMNS:
        if col == "income":
            df[col] = np.where(raw["target"].to_numpy() == 0, ">50K", "<=50K")
        elif col in LABELS:
            codes = raw[col].to_numpy().astype(int)
            df[col] = np.asarray(LABELS[col], dtype=object)[codes]
        else:
            df[col] = raw[col].to_numpy().astype(np.int64)
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    return df[COLUMNS]


def three_way_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, ...]:
    """Split into train (the confidential data), holdout and reference.

    The three parts are disjoint random thirds. Generators only ever see
    `train`. `holdout` is the test set for train-synthetic-test-real and the
    non-member set for the privacy tests. `reference` is an independent real
    sample of the same size, the benchmark an ideal generator would match.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    parts = np.array_split(idx, 3)
    return tuple(df.iloc[p].reset_index(drop=True) for p in parts)
