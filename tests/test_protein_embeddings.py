from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from python.protein_embeddings import (
    ProteinEmbeddingExtractor,
    SequenceEmbeddingPCATransformer,
    build_embedding_feature_frame,
    reduce_embedding_matrix,
)


class FakeExtractor(ProteinEmbeddingExtractor):
    def __init__(self, cache_dir: Path) -> None:
        super().__init__(model_name="fake/test-model", cache_dir=cache_dir, device="cpu")
        self.encode_calls = 0

    def _ensure_backend(self) -> None:
        return None

    def _encode_uncached(self, sequence: str) -> np.ndarray:
        self.encode_calls += 1
        length = len(sequence)
        ascii_sum = float(sum(ord(char) for char in sequence))
        return np.asarray(
            [
                float(length),
                ascii_sum,
                ascii_sum / max(length, 1),
                float(sequence.count("A")),
            ],
            dtype=np.float32,
        )


def test_cache_reuses_saved_embedding(tmp_path: Path) -> None:
    extractor = FakeExtractor(cache_dir=tmp_path / "cache")
    first = extractor.encode_sequence("ACDE")
    second = extractor.encode_sequence("ACDE")

    assert extractor.encode_calls == 1
    assert np.allclose(first, second)
    assert any((tmp_path / "cache").glob("*.npy"))


def test_reduce_embedding_matrix_limits_components() -> None:
    matrix = np.asarray(
        [
            [1.0, 0.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 1.0, 1.0, 4.0],
        ],
        dtype=float,
    )
    reduced, names, meta = reduce_embedding_matrix(matrix, n_components=8, prefix="esm2")

    assert reduced.shape == (3, 2)
    assert names == ["esm2_pc1", "esm2_pc2"]
    assert meta["requestedComponents"] == 8
    assert meta["usedComponents"] == 2


def test_build_embedding_feature_frame_preserves_duplicate_sequences(tmp_path: Path) -> None:
    sequences = ["AAAA", "AAAA", "CCDD"]
    extractor = FakeExtractor(cache_dir=tmp_path / "cache")
    frame, feature_names, meta = build_embedding_feature_frame(
        sequences=sequences,
        extractor=extractor,
        n_components=3,
        prefix="esm2",
    )

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape[0] == 3
    assert len(feature_names) == 2
    assert meta["sequenceCount"] == 3
    assert meta["uniqueSequenceCount"] == 2
    assert np.allclose(
        frame.iloc[0].to_numpy(dtype=float),
        frame.iloc[1].to_numpy(dtype=float),
        atol=1e-5,
    )


def test_sequence_embedding_transformer_outputs_stable_feature_names(tmp_path: Path) -> None:
    sequences = np.asarray([["AAAA"], ["AAAA"], ["CCDD"], ["WYYY"]], dtype=object)
    extractor = FakeExtractor(cache_dir=tmp_path / "cache")
    transformer = SequenceEmbeddingPCATransformer(
        extractor=extractor,
        n_components=3,
        prefix="esm2",
    )

    transformed = transformer.fit_transform(sequences)

    assert transformed.shape == (4, 3)
    assert transformer.get_feature_names_out().tolist() == ["esm2_pc1", "esm2_pc2", "esm2_pc3"]
    assert np.allclose(transformed[0], transformed[1], atol=1e-5)
