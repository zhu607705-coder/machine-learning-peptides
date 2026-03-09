from __future__ import annotations

import json
import ssl
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DataSpec:
    name: str
    url: str
    relative_path: str
    description: str


DATA_SPECS = [
    DataSpec(
        name="peptimizer_synthesis_data",
        url="https://raw.githubusercontent.com/learningmatter-mit/peptimizer/master/dataset/data_synthesis/synthesis_data.csv",
        relative_path="data/real/synthesis_data.csv",
        description="AFPS step-level synthesis outcomes (multi-target regression).",
    ),
    DataSpec(
        name="peptimizer_cpp_predictor_dataset",
        url="https://raw.githubusercontent.com/learningmatter-mit/peptimizer/master/dataset/data_cpp/cpp_predictor_dataset.csv",
        relative_path="data/external/peptimizer_cpp_predictor_dataset.csv",
        description="CPP predictor dataset with sequence intensity labels.",
    ),
    DataSpec(
        name="cyclization_site_dataset",
        url="https://raw.githubusercontent.com/Yongboxiao/Selection-of-Cyclization-Sites/main/dataset.csv",
        relative_path="data/external/cyclization_dataset.csv",
        description="Cyclization site selection labeled sequences.",
    ),
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(request, context=context, timeout=120) as response:
        output_path.write_bytes(response.read())


def dataset_profile(path: Path) -> dict[str, object]:
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "columnNames": [str(column) for column in df.columns.tolist()],
    }


def main() -> None:
    root = project_root()
    catalog: dict[str, object] = {"datasets": []}

    for spec in DATA_SPECS:
        output_path = root / spec.relative_path
        download_file(spec.url, output_path)
        profile = dataset_profile(output_path)
        catalog["datasets"].append(
            {
                **asdict(spec),
                "absolutePath": str(output_path),
                "bytes": int(output_path.stat().st_size),
                "profile": profile,
            }
        )
        print(f"[downloaded] {spec.name}: {output_path}")

    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = artifacts_dir / "public-data-catalog.json"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote catalog: {catalog_path}")


if __name__ == "__main__":
    main()
