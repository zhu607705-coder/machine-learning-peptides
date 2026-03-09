from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from fmoc_sop_subset_analysis import (
    FEATURE_COLUMNS,
    centered_correlations,
    evaluate_grouped_ridge,
    load_rows,
    source_summary,
)
from protein_embeddings import DEFAULT_MODEL_NAME, ProteinEmbeddingExtractor, reduce_embedding_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_REPORT_PATH = PROJECT_ROOT / "artifacts" / "fmoc-sop-embedding-analysis.json"
MARKDOWN_REPORT_PATH = PROJECT_ROOT / "docs" / "fmoc-sop-embedding-analysis.md"
EMBEDDING_PREFIX = "esm2"
EMBEDDING_COMPONENTS = 8


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def attach_embeddings(
    rows: list[dict[str, Any]],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    extractor = ProteinEmbeddingExtractor(model_name=model_name, cache_dir=cache_dir or (PROJECT_ROOT / "data" / "cache" / "protein_embeddings"))
    sequences = [row["sequence_norm"] for row in rows]
    embedding_map = extractor.encode_sequences(sequences)
    hidden_size = 0
    enriched: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        vector = embedding_map[row["sequence_norm"]]
        hidden_size = max(hidden_size, int(vector.shape[0]))
        payload["embedding_vector"] = vector
        enriched.append(payload)
    metadata = {
        "modelName": extractor.model_name,
        "device": extractor.device,
        "cacheDir": str(extractor.cache_dir),
        "sequenceCount": int(len(sequences)),
        "uniqueSequenceCount": int(len(embedding_map)),
        "embeddingDimension": int(hidden_size),
    }
    return enriched, metadata


def evaluate_grouped_ridge_with_embeddings(rows: list[dict[str, Any]], target_key: str) -> dict[str, Any]:
    usable = [row for row in rows if row[target_key] is not None]
    sources = sorted({row["source_id"] for row in usable})
    predictions: list[float] = []
    truths: list[float] = []
    baseline_predictions: list[float] = []
    per_source: list[dict[str, Any]] = []

    for source in sources:
        train = [row for row in usable if row["source_id"] != source]
        test = [row for row in usable if row["source_id"] == source]
        x_train_base = np.asarray([[row[column] for column in FEATURE_COLUMNS] for row in train], dtype=float)
        x_test_base = np.asarray([[row[column] for column in FEATURE_COLUMNS] for row in test], dtype=float)
        y_train = np.asarray([row[target_key] for row in train], dtype=float)
        y_test = np.asarray([row[target_key] for row in test], dtype=float)
        raw_train_embeddings = np.stack([row["embedding_vector"] for row in train], axis=0)
        raw_test_embeddings = np.stack([row["embedding_vector"] for row in test], axis=0)

        train_embedding_features, embedding_names, reduction_meta = reduce_embedding_matrix(
            raw_train_embeddings,
            n_components=EMBEDDING_COMPONENTS,
            prefix=EMBEDDING_PREFIX,
        )

        used_components = int(reduction_meta["usedComponents"])
        centered_train = raw_train_embeddings - raw_train_embeddings.mean(axis=0, keepdims=True)
        centered_test = raw_test_embeddings - raw_train_embeddings.mean(axis=0, keepdims=True)
        right_singular_vectors = np.linalg.svd(centered_train, full_matrices=False)[2]
        projection = right_singular_vectors[:used_components].T
        test_embedding_features = centered_test @ projection

        x_train = np.concatenate([x_train_base, train_embedding_features], axis=1)
        x_test = np.concatenate([x_test_base, test_embedding_features], axis=1)
        scaler = StandardScaler().fit(x_train)
        model = Ridge(alpha=1.0).fit(scaler.transform(x_train), y_train)
        y_pred = model.predict(scaler.transform(x_test))
        source_baseline = np.full_like(y_test, float(np.mean(y_train)), dtype=float)

        predictions.extend(y_pred.tolist())
        truths.extend(y_test.tolist())
        baseline_predictions.extend(source_baseline.tolist())
        per_source.append(
            {
                "source_id": source,
                "n_test": int(len(test)),
                "embedding_components": used_components,
                "embedding_feature_names": embedding_names,
                "ridge_rmse": rmse(y_test, y_pred),
                "baseline_rmse": rmse(y_test, source_baseline),
                "ridge_mae": float(mean_absolute_error(y_test, y_pred)),
                "baseline_mae": float(mean_absolute_error(y_test, source_baseline)),
            }
        )

    y_true = np.asarray(truths, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    y_base = np.asarray(baseline_predictions, dtype=float)
    return {
        "n": int(len(y_true)),
        "sources": sources,
        "ridge": {
            "rmse": rmse(y_true, y_pred),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        },
        "mean_baseline": {
            "rmse": rmse(y_true, y_base),
            "mae": float(mean_absolute_error(y_true, y_base)),
            "r2": float(r2_score(y_true, y_base)) if len(y_true) > 1 else float("nan"),
        },
        "per_source": per_source,
    }


def build_report(rows: list[dict[str, Any]], embedding_meta: dict[str, Any]) -> dict[str, Any]:
    purity_rows = sum(row["purity_value"] is not None for row in rows)
    yield_rows = sum(row["yield_value"] is not None for row in rows)
    grouped_purity_manual = evaluate_grouped_ridge(rows, "purity_value")
    grouped_yield_manual = evaluate_grouped_ridge(rows, "yield_value")
    grouped_purity_embedding = evaluate_grouped_ridge_with_embeddings(rows, "purity_value")
    grouped_yield_embedding = evaluate_grouped_ridge_with_embeddings(rows, "yield_value")

    return {
        "dataset_summary": {
            "rows": int(len(rows)),
            "sources": sorted({row["source_id"] for row in rows}),
            "source_count": int(len({row["source_id"] for row in rows})),
            "purity_rows": int(purity_rows),
            "yield_rows": int(yield_rows),
        },
        "embedding": {
            **embedding_meta,
            "requestedComponents": EMBEDDING_COMPONENTS,
            "prefix": EMBEDDING_PREFIX,
        },
        "source_summary": source_summary(rows),
        "grouped_purity_manual": grouped_purity_manual,
        "grouped_purity_embedding": grouped_purity_embedding,
        "grouped_yield_manual": grouped_yield_manual,
        "grouped_yield_embedding": grouped_yield_embedding,
        "manual_centered_correlations": {
            "purity": centered_correlations(rows, "purity_value"),
            "yield": centered_correlations(rows, "yield_value"),
        },
        "improvements": {
            "purity_rmse_delta": float(grouped_purity_embedding["ridge"]["rmse"] - grouped_purity_manual["ridge"]["rmse"]),
            "yield_rmse_delta": float(grouped_yield_embedding["ridge"]["rmse"] - grouped_yield_manual["ridge"]["rmse"]),
            "purity_mae_delta": float(grouped_purity_embedding["ridge"]["mae"] - grouped_purity_manual["ridge"]["mae"]),
            "yield_mae_delta": float(grouped_yield_embedding["ridge"]["mae"] - grouped_yield_manual["ridge"]["mae"]),
        },
    }


def write_markdown(report: dict[str, Any]) -> None:
    summary = report["dataset_summary"]
    embedding = report["embedding"]
    lines = [
        "# Fmoc SOP Embedding Analysis",
        "",
        "## Dataset summary",
        f"- Rows: {summary['rows']}",
        f"- Sources: {summary['source_count']} -> {', '.join(summary['sources'])}",
        f"- Purity rows: {summary['purity_rows']}",
        f"- Yield rows: {summary['yield_rows']}",
        "",
        "## Embedding configuration",
        f"- Model: {embedding['modelName']}",
        f"- Device: {embedding['device']}",
        f"- Cache dir: {embedding['cacheDir']}",
        f"- Raw embedding dimension: {embedding['embeddingDimension']}",
        f"- Requested PCA components per fold: {embedding['requestedComponents']}",
        "",
        "## Grouped evaluation",
        f"- Purity manual Ridge RMSE: {report['grouped_purity_manual']['ridge']['rmse']:.3f}",
        f"- Purity manual + embedding Ridge RMSE: {report['grouped_purity_embedding']['ridge']['rmse']:.3f}",
        f"- Yield manual Ridge RMSE: {report['grouped_yield_manual']['ridge']['rmse']:.3f}",
        f"- Yield manual + embedding Ridge RMSE: {report['grouped_yield_embedding']['ridge']['rmse']:.3f}",
        "",
        "## Delta",
        f"- Purity RMSE delta (embedding - manual): {report['improvements']['purity_rmse_delta']:.3f}",
        f"- Yield RMSE delta (embedding - manual): {report['improvements']['yield_rmse_delta']:.3f}",
        f"- Purity MAE delta (embedding - manual): {report['improvements']['purity_mae_delta']:.3f}",
        f"- Yield MAE delta (embedding - manual): {report['improvements']['yield_mae_delta']:.3f}",
        "",
        "## Interpretation",
        "- The embedding features come from a pretrained ESM2 protein language model and are reduced within each grouped fold before concatenation with manual chemistry features.",
        "- A negative RMSE delta means the embedding-augmented model improves on the manual-feature baseline.",
        "- Given the small SOP-aligned literature subset, any gain should be interpreted as feature-engineering evidence rather than a claim of broad external generalization.",
    ]
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    rows_with_embeddings, embedding_meta = attach_embeddings(rows)
    report = build_report(rows_with_embeddings, embedding_meta)
    JSON_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report)


if __name__ == "__main__":
    main()
