from __future__ import annotations

from peptide_core import PeptideSynthesisParams
from predictor import build_annotated_feature_row, predict


def test_build_annotated_feature_row_contains_expected_columns() -> None:
    row = build_annotated_feature_row(
        PeptideSynthesisParams(
            sequence="H-Gly-Ala-Val-Leu-Ile-OH",
            topology="Linear",
            coupling_reagent="HATU",
            solvent="DMF",
            temperature="Room Temperature",
            cleavage_time="2 hours",
        ),
        data_source="literature",
    )

    for key in [
        "length",
        "avg_hydrophobicity",
        "total_charge",
        "molecular_weight",
        "avg_volume",
        "max_coupling_difficulty",
        "bulky_ratio",
        "polar_ratio",
        "charged_ratio",
        "aromatic_ratio",
        "sulfur_ratio",
        "reagent_score",
        "solvent_score",
        "temperature_score",
        "cleavage_score",
        "data_source",
    ]:
        assert key in row
    assert row["data_source"] == "literature"


def test_predict_uses_annotated_model_family() -> None:
    result = predict(
        PeptideSynthesisParams(
            sequence="H-Gly-Ala-Val-Leu-Ile-OH",
            topology="Linear",
            coupling_reagent="HATU",
            solvent="DMF",
            temperature="Room Temperature",
            cleavage_time="2 hours",
        ),
        model_family="annotated",
        data_source_hint="literature",
    )

    assert "purity" in result
    assert "yield" in result
    assert result["training"]["modelFamily"] == "annotated"
