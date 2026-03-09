from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = os.environ.get("PEPTIDE_EMBEDDING_MODEL", "facebook/esm2_t6_8M_UR50D")
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "protein_embeddings"
DEFAULT_MAX_LENGTH = 1022


@dataclass(frozen=True)
class EmbeddingMetadata:
    model_name: str
    device: str
    hidden_size: int


class ProteinEmbeddingExtractor:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_length = max_length
        self.device = device or self._select_device()
        self._tokenizer: Any = None
        self._model: Any = None

    def _select_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _ensure_backend(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer, EsmModel, logging as hf_logging
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for protein embeddings. "
                "Install it with `python3 -m pip install transformers`."
            ) from exc

        hf_logging.set_verbosity_error()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if "esm" in self.model_name.lower():
            self._model = EsmModel.from_pretrained(self.model_name, add_pooling_layer=False)
        else:
            self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def _cache_path(self, sequence: str) -> Path:
        digest = hashlib.sha1(f"{self.model_name}:{sequence}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def _encode_uncached(self, sequence: str) -> np.ndarray:
        self._ensure_backend()
        assert self._tokenizer is not None and self._model is not None
        spaced_sequence = " ".join(list(sequence))
        tokens = self._tokenizer(
            spaced_sequence,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.inference_mode():
            outputs = self._model(**tokens)
        hidden = outputs.last_hidden_state[0]
        attention_mask = tokens["attention_mask"][0].bool()
        valid_hidden = hidden[attention_mask]
        if valid_hidden.shape[0] > 2:
            valid_hidden = valid_hidden[1:-1]
        pooled = valid_hidden.mean(dim=0).detach().cpu().numpy().astype(np.float32)
        return pooled

    def encode_sequence(self, sequence: str) -> np.ndarray:
        normalized = str(sequence or "").strip().upper()
        if not normalized:
            raise ValueError("sequence must be non-empty")
        cache_path = self._cache_path(normalized)
        if cache_path.exists():
            return np.load(cache_path)
        embedding = self._encode_uncached(normalized)
        np.save(cache_path, embedding.astype(np.float32))
        return embedding

    def encode_sequences(self, sequences: list[str]) -> dict[str, np.ndarray]:
        unique_sequences = sorted({str(sequence or "").strip().upper() for sequence in sequences if str(sequence or "").strip()})
        return {sequence: self.encode_sequence(sequence) for sequence in unique_sequences}


def reduce_embedding_matrix(
    matrix: np.ndarray,
    *,
    n_components: int,
    prefix: str = "esm2",
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if matrix.ndim != 2:
        raise ValueError("embedding matrix must be 2-dimensional")
    if matrix.shape[0] == 0:
        raise ValueError("embedding matrix must contain at least one row")
    max_components = max(1, min(n_components, matrix.shape[0] - 1, matrix.shape[1]))
    if matrix.shape[0] == 1:
        max_components = 1
    pca = PCA(n_components=max_components, svd_solver="full")
    reduced = pca.fit_transform(matrix)
    feature_names = [f"{prefix}_pc{index}" for index in range(1, max_components + 1)]
    meta = {
        "requestedComponents": int(n_components),
        "usedComponents": int(max_components),
        "inputDimension": int(matrix.shape[1]),
        "explainedVarianceRatio": [float(value) for value in pca.explained_variance_ratio_.tolist()],
    }
    return reduced.astype(np.float32), feature_names, meta


def build_embedding_feature_frame(
    *,
    sequences: list[str],
    extractor: ProteinEmbeddingExtractor | None = None,
    n_components: int = 8,
    prefix: str = "esm2",
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    extractor = extractor or ProteinEmbeddingExtractor()
    mapping = extractor.encode_sequences(sequences)
    matrix = np.stack([mapping[str(sequence or "").strip().upper()] for sequence in sequences], axis=0)
    reduced, feature_names, reduction_meta = reduce_embedding_matrix(matrix, n_components=n_components, prefix=prefix)
    frame = pd.DataFrame(reduced, columns=feature_names)
    meta = {
        "sequenceCount": int(len(sequences)),
        "uniqueSequenceCount": int(len(mapping)),
        "modelName": extractor.model_name,
        "device": extractor.device,
        "reduction": reduction_meta,
    }
    return frame, feature_names, meta


class SequenceEmbeddingPCATransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        extractor: ProteinEmbeddingExtractor | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        device: str | None = None,
        n_components: int = 8,
        prefix: str = "esm2",
    ) -> None:
        self.extractor = extractor
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.n_components = n_components
        self.prefix = prefix

    def _get_extractor(self) -> ProteinEmbeddingExtractor:
        if self.extractor is not None:
            return self.extractor
        return ProteinEmbeddingExtractor(
            model_name=self.model_name,
            cache_dir=Path(self.cache_dir),
            device=self.device,
        )

    def _normalize_sequences(self, X: Any) -> list[str]:
        if isinstance(X, pd.DataFrame):
            values = X.iloc[:, 0].tolist()
        elif isinstance(X, pd.Series):
            values = X.tolist()
        else:
            array = np.asarray(X, dtype=object)
            if array.ndim == 2:
                values = array[:, 0].tolist()
            else:
                values = array.tolist()
        sequences = [str(value or "").strip().upper() for value in values]
        if any(not sequence for sequence in sequences):
            raise ValueError("sequence embedding transformer received empty sequence")
        return sequences

    def fit(self, X: Any, y: Any = None) -> "SequenceEmbeddingPCATransformer":
        extractor = self._get_extractor()
        sequences = self._normalize_sequences(X)
        mapping = extractor.encode_sequences(sequences)
        matrix = np.stack([mapping[sequence] for sequence in sequences], axis=0)
        reduced, feature_names, meta = reduce_embedding_matrix(
            matrix,
            n_components=self.n_components,
            prefix=self.prefix,
        )
        used_components = int(meta["usedComponents"])
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.extractor_ = extractor
        self.mean_ = matrix.mean(axis=0, keepdims=True)
        self.projection_ = vt[:used_components].T
        self.feature_names_ = np.asarray(feature_names, dtype=object)
        self.reduction_meta_ = meta
        self.fitted_dimension_ = reduced.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        sequences = self._normalize_sequences(X)
        mapping = self.extractor_.encode_sequences(sequences)
        matrix = np.stack([mapping[sequence] for sequence in sequences], axis=0)
        centered = matrix - self.mean_
        transformed = centered @ self.projection_
        return transformed.astype(np.float32)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return getattr(self, "feature_names_", np.asarray([], dtype=object))
