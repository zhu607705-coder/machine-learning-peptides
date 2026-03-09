from autoresearch.prepare import summarize_report


def test_summarize_report_extracts_primary_head() -> None:
    summary = summarize_report(
        {
            "headLevelMetrics": [
                {
                    "target": "yield_target",
                    "head": "isolated|isolated",
                    "metrics": {
                        "rmse": 12.3,
                        "r2": 0.45,
                        "accuracyWithinTolerance": 0.3,
                    },
                    "decision": {"confidenceTier": "low"},
                }
            ],
            "predictionPolicy": {
                "acceptedHighConfidenceHeads": [],
                "acceptedWarningHeads": [{"head": "isolated|isolated"}],
                "rejectedHeads": [],
            },
            "weightedHeadSummary": {"weightedR2": 0.12, "weightedRmse": 9.9},
            "strictSemanticFilter": {"outputRows": 10},
            "sourceDiagnostics": [],
        },
        {"final_rows": 10},
        3.5,
    )
    assert summary["primaryHead"] == "yield_target/isolated|isolated"
    assert summary["primaryRmse"] == 12.3
    assert summary["acceptedWarningHeads"] == 1
    assert summary["weightedR2"] == 0.12
