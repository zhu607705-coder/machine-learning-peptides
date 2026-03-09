from __future__ import annotations

import numpy as np
import pandas as pd

from python.comprehensive_ml_workflow import (
    HEAD_MAX_SOURCE_SHARE,
    HEAD_MIN_ROWS,
    HEAD_MIN_SOURCES,
    build_head_eligibility_table,
    build_purity_semantic_head,
    diagnose_head_sources,
    derive_head_decision,
    build_strict_semantic_subset,
    build_weighted_head_summary,
    remove_near_duplicates,
    remove_outliers,
    select_compatible_head_subset,
)
from python.source_head_calibration import SourceAwareShrinkageCalibrator


def make_row(
    *,
    record_id: str,
    source_id: str,
    purity_head: str,
    yield_head: str,
    purity_target: float,
    yield_target: float,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source_id": source_id,
        "sequence_norm": "ACDE",
        "topology": "linear",
        "purity_stage": purity_head,
        "yield_stage": yield_head.split("|")[0],
        "yield_basis_class": yield_head.split("|")[1],
        "purity_stage_canon": purity_head,
        "yield_stage_canon": yield_head.split("|")[0],
        "yield_basis_canon": yield_head.split("|")[1],
        "purity_semantic_head": purity_head,
        "yield_semantic_head": yield_head,
        "purity_target": purity_target,
        "yield_target": yield_target,
        "condition_summary": "default condition",
        "length": 4.0,
        "molecular_weight": 400.0,
    }


def test_build_purity_semantic_head_keeps_only_supported_values() -> None:
    assert build_purity_semantic_head("crude_hplc") == "crude_hplc"
    assert build_purity_semantic_head("purified_hplc") == "purified_hplc"
    assert build_purity_semantic_head("final_product") == "final_product"
    assert build_purity_semantic_head("unknown") == "unknown"


def test_build_head_eligibility_table_applies_minimum_rules() -> None:
    rows = []
    for source_idx, source_id in enumerate(["s1", "s2", "s3"], start=1):
        for sample_idx in range(8):
            rows.append(
                make_row(
                    record_id=f"{source_id}-{sample_idx}",
                    source_id=source_id,
                    purity_head="crude_hplc",
                    yield_head="isolated|isolated",
                    purity_target=90.0 + source_idx,
                    yield_target=20.0 + sample_idx,
                )
            )
    # source share > 70%, should be rejected
    for sample_idx in range(5):
        rows.append(
            make_row(
                record_id=f"dom-{sample_idx}",
                source_id="dom",
                purity_head="purified_hplc",
                yield_head="crude|crude",
                purity_target=80.0,
                yield_target=30.0,
            )
        )
    rows.append(
        make_row(
            record_id="rare-1",
            source_id="rare",
            purity_head="purified_hplc",
            yield_head="crude|crude",
            purity_target=81.0,
            yield_target=31.0,
        )
    )
    df = pd.DataFrame(rows)

    eligibility = build_head_eligibility_table(
        df,
        head_column="yield_semantic_head",
        min_rows=HEAD_MIN_ROWS,
        min_sources=HEAD_MIN_SOURCES,
        max_source_share=HEAD_MAX_SOURCE_SHARE,
    )
    by_head = {row["head"]: row for row in eligibility}
    assert by_head["isolated|isolated"]["eligible"] is True
    assert by_head["isolated|isolated"]["nRows"] == 24
    assert by_head["crude|crude"]["eligible"] is False
    assert any("source_share" in reason for reason in by_head["crude|crude"]["reasons"])


def test_build_strict_semantic_subset_filters_unknown_and_ineligible_heads() -> None:
    rows = []
    for source_id in ["s1", "s2", "s3"]:
        for sample_idx in range(7):
            rows.append(
                make_row(
                    record_id=f"{source_id}-ok-{sample_idx}",
                    source_id=source_id,
                    purity_head="crude_hplc",
                    yield_head="isolated|isolated",
                    purity_target=90.0,
                    yield_target=40.0 + sample_idx,
                )
            )
    rows.append(
        make_row(
            record_id="unknown-1",
            source_id="s4",
            purity_head="unknown",
            yield_head="isolated|isolated",
            purity_target=70.0,
            yield_target=10.0,
        )
    )
    rows.append(
        make_row(
            record_id="other-1",
            source_id="s4",
            purity_head="crude_hplc",
            yield_head="unknown|other",
            purity_target=71.0,
            yield_target=11.0,
        )
    )
    df = pd.DataFrame(rows)

    filtered, metadata = build_strict_semantic_subset(df)
    assert int(filtered["record_id"].nunique()) == 23
    assert "unknown-1" in set(filtered["record_id"])
    assert "other-1" in set(filtered["record_id"])
    assert metadata["purityEligibility"]
    assert metadata["yieldEligibility"]


def test_build_strict_semantic_subset_uses_target_specific_availability() -> None:
    rows = []
    for source_id in ["s1", "s2", "s3"]:
        for sample_idx in range(8):
            rows.append(
                make_row(
                    record_id=f"{source_id}-purity-{sample_idx}",
                    source_id=source_id,
                    purity_head="purified_hplc",
                    yield_head="unknown|unknown",
                    purity_target=90.0 + sample_idx,
                    yield_target=np.nan,
                )
            )
    for source_id in ["s1", "s2", "s3"]:
        for sample_idx in range(8):
            rows.append(
                make_row(
                    record_id=f"{source_id}-yield-{sample_idx}",
                    source_id=source_id,
                    purity_head="unknown",
                    yield_head="isolated|isolated",
                    purity_target=np.nan,
                    yield_target=20.0 + sample_idx,
                )
            )
    df = pd.DataFrame(rows)

    filtered, metadata = build_strict_semantic_subset(df)
    assert metadata["allowedPurityHeads"] == ["purified_hplc"]
    assert metadata["allowedYieldHeads"] == ["isolated|isolated"]
    assert metadata["outputPurityRows"] == 24
    assert metadata["outputYieldRows"] == 24
    assert int(filtered["record_id"].nunique()) == 48


def test_source_shrinkage_calibrator_falls_back_to_zero_for_unseen_source() -> None:
    calibrator = SourceAwareShrinkageCalibrator(prior_strength=3.0)
    train_sources = pd.Series(["s1", "s1", "s2", "s2"])
    residuals = np.array([2.0, 2.0, -1.0, -1.0], dtype=float)
    calibrator.fit(train_sources, residuals)
    offsets = calibrator.predict_offsets(pd.Series(["s1", "s3"]))
    assert offsets.shape == (2,)
    assert offsets[0] > 0.0
    assert offsets[1] == 0.0


def test_build_weighted_head_summary_uses_rows_times_sources() -> None:
    metrics = [
        {
            "head": "isolated|isolated",
            "target": "yield_target",
            "nRows": 30,
            "nSources": 4,
            "metrics": {"rmse": 10.0, "mae": 8.0, "r2": 0.5, "accuracyWithinTolerance": 0.6},
        },
        {
            "head": "crude_hplc",
            "target": "purity_target",
            "nRows": 20,
            "nSources": 3,
            "metrics": {"rmse": 5.0, "mae": 4.0, "r2": 0.2, "accuracyWithinTolerance": 0.7},
        },
    ]
    summary = build_weighted_head_summary(metrics)
    assert summary["totalWeightedSupport"] == (30 * 4) + (20 * 3)
    assert 0.0 < summary["weightedRmse"] < 10.0


def test_derive_head_decision_rejects_negative_r2_and_wide_intervals() -> None:
    decision = derive_head_decision(
        target="purity_target",
        n_rows=30,
        n_sources=4,
        metrics={"rmse": 20.0, "mae": 15.0, "r2": -0.1, "accuracyWithinTolerance": 0.1},
        interval_metrics={"coverage": 0.9, "meanWidth": 30.0},
    )
    assert decision["confidenceTier"] == "reject"
    assert decision["servePrediction"] is False
    assert "negative_r2" in decision["reasons"]


def test_derive_head_decision_allows_high_confidence_when_signal_is_strong() -> None:
    decision = derive_head_decision(
        target="yield_target",
        n_rows=40,
        n_sources=5,
        metrics={"rmse": 12.0, "mae": 9.0, "r2": 0.62, "accuracyWithinTolerance": 0.32},
        interval_metrics={"coverage": 0.82, "meanWidth": 18.0},
    )
    assert decision["confidenceTier"] == "high"
    assert decision["servePrediction"] is True


def test_remove_near_duplicates_keeps_same_sequence_when_context_differs() -> None:
    df = pd.DataFrame(
        [
            {
                **make_row(
                    record_id="r1",
                    source_id="hbd3",
                    purity_head="unknown",
                    yield_head="isolated|isolated",
                    purity_target=np.nan,
                    yield_target=10.0,
                ),
                "condition_summary": "wang resin route",
            },
            {
                **make_row(
                    record_id="r2",
                    source_id="hbd3",
                    purity_head="unknown",
                    yield_head="isolated|isolated",
                    purity_target=np.nan,
                    yield_target=11.0,
                ),
                "condition_summary": "chemmatrix pseudoproline route",
            },
        ]
    )
    deduped, removed = remove_near_duplicates(df, similarity_threshold=0.97)
    assert removed == 0
    assert len(deduped) == 2


def test_remove_outliers_is_source_aware_for_result_heads() -> None:
    rows = []
    for idx, value in enumerate([40.0, 42.0, 45.0, 47.0], start=1):
        rows.append(
            {
                **make_row(
                    record_id=f"s1-{idx}",
                    source_id="s1",
                    purity_head="unknown",
                    yield_head="isolated|isolated",
                    purity_target=np.nan,
                    yield_target=value,
                ),
                "length": 8.0,
                "molecular_weight": 800.0,
            }
        )
    for idx, value in enumerate([0.5, 1.0, 2.0, 3.0], start=1):
        rows.append(
            {
                **make_row(
                    record_id=f"s2-{idx}",
                    source_id="s2",
                    purity_head="unknown",
                    yield_head="isolated|isolated",
                    purity_target=np.nan,
                    yield_target=value,
                ),
                "length": 8.0,
                "molecular_weight": 800.0,
            }
        )

    filtered, removed = remove_outliers(pd.DataFrame(rows))
    assert removed == 0
    assert len(filtered) == 8


def test_diagnose_head_sources_flags_high_drift_source() -> None:
    rows = []
    for idx, value in enumerate([10.0, 11.0, 12.0, 13.0], start=1):
        rows.append(
            make_row(
                record_id=f"s1-{idx}",
                source_id="s1",
                purity_head="unknown",
                yield_head="isolated|isolated",
                purity_target=np.nan,
                yield_target=value,
            )
        )
    for idx, value in enumerate([14.0, 15.0, 16.0, 17.0], start=1):
        rows.append(
            make_row(
                record_id=f"s2-{idx}",
                source_id="s2",
                purity_head="unknown",
                yield_head="isolated|isolated",
                purity_target=np.nan,
                yield_target=value,
            )
        )
    for idx, value in enumerate([80.0, 81.0, 82.0, 83.0], start=1):
        rows.append(
            make_row(
                record_id=f"s3-{idx}",
                source_id="s3",
                purity_head="unknown",
                yield_head="isolated|isolated",
                purity_target=np.nan,
                yield_target=value,
            )
        )
    diagnostics = diagnose_head_sources(pd.DataFrame(rows), target="yield_target")
    by_source = {row["sourceId"]: row for row in diagnostics["sources"]}
    assert by_source["s3"]["isDriftOutlier"] is True
    assert by_source["s1"]["isDriftOutlier"] is False


def test_select_compatible_head_subset_drops_drift_sources_when_thresholds_hold() -> None:
    rows = []
    for source_id, values in {
        "s1": [10.0, 11.0, 12.0, 13.0],
        "s2": [14.0, 15.0, 16.0, 17.0],
        "s3": [80.0, 81.0, 82.0, 83.0],
        "s4": [18.0, 19.0, 20.0, 21.0],
        "s5": [22.0, 23.0, 24.0, 25.0],
        "s6": [26.0, 27.0, 28.0, 29.0],
    }.items():
        for idx, value in enumerate(values, start=1):
            rows.append(
                make_row(
                    record_id=f"{source_id}-{idx}",
                    source_id=source_id,
                    purity_head="unknown",
                    yield_head="isolated|isolated",
                    purity_target=np.nan,
                    yield_target=value,
                )
            )
    head_df = pd.DataFrame(rows)
    diagnostics = diagnose_head_sources(head_df, target="yield_target")
    compatible_df, compatibility = select_compatible_head_subset(
        head_df,
        diagnostics=diagnostics,
        min_rows=HEAD_MIN_ROWS,
        min_sources=HEAD_MIN_SOURCES,
    )
    assert compatibility["usedCompatibleSubset"] is True
    assert "s3" in compatibility["excludedSources"]
    assert compatible_df["source_id"].nunique() == 5
