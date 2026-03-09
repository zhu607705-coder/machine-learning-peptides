from __future__ import annotations

import ssl
import subprocess
import urllib.request
from pathlib import Path

import pandas as pd


REAL_DATASET_URL = (
    "https://raw.githubusercontent.com/learningmatter-mit/peptimizer/master/"
    "dataset/data_synthesis/synthesis_data.csv"
)
REAL_DATASET_SOURCE = "learningmatter-mit/peptimizer"

TARGET_COLUMNS = ["first_area", "first_height", "first_width", "first_diff"]
NUMERIC_COLUMNS = [
    "coupling_strokes",
    "deprotection_strokes",
    "flow_rate",
    "temp_coupling",
    "temp_reactor_1",
]
CATEGORICAL_COLUMNS = ["coupling_agent"]

TOKEN_ALPHABET = ["<pad>"] + list("ACDEFGHIKLMNPQRSTVWY")
TOKEN_TO_INDEX = {token: index for index, token in enumerate(TOKEN_ALPHABET)}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_real_dataset(force_download: bool = False) -> Path:
    data_dir = project_root() / "data" / "real"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / "synthesis_data.csv"

    if force_download or not dataset_path.exists():
        curl_result = subprocess.run(
            ["curl", "-L", "--fail", "--max-time", "120", REAL_DATASET_URL, "-o", str(dataset_path)],
            capture_output=True,
            text=True,
        )

        if curl_result.returncode != 0:
            request = urllib.request.Request(
                REAL_DATASET_URL,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(request, context=context, timeout=120) as response:
                dataset_path.write_bytes(response.read())

    return dataset_path


def load_real_dataset(force_download: bool = False) -> pd.DataFrame:
    dataset_path = ensure_real_dataset(force_download=force_download)
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["pre-chain", "amino_acid", "serial", *TARGET_COLUMNS]).copy()
    df["pre-chain"] = df["pre-chain"].astype(str)
    df["amino_acid"] = df["amino_acid"].astype(str)
    df["serial"] = df["serial"].astype(str)

    for column in NUMERIC_COLUMNS + TARGET_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[*NUMERIC_COLUMNS, *TARGET_COLUMNS]).copy()
    df["sequence_length"] = df["pre-chain"].str.len()
    df["full_sequence"] = df["pre-chain"] + df["amino_acid"]
    return df


def encode_token_sequence(sequence: str, max_length: int) -> list[int]:
    tokens = [TOKEN_TO_INDEX.get(token, 0) for token in sequence][-max_length:]
    if len(tokens) < max_length:
        tokens = ([0] * (max_length - len(tokens))) + tokens
    return tokens
