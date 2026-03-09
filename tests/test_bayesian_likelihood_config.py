from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import bayesian_multitask_calibrator as calibrator_module
from bayesian_multitask_calibrator import BayesianMultitaskCalibrator
from comprehensive_ml_workflow import parse_args


def test_parse_args_accepts_bayesian_likelihood_options(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "comprehensive_ml_workflow.py",
            "--bayes-likelihood",
            "student_t",
            "--bayes-student-t-nu",
            "6.5",
        ],
    )
    args = parse_args()
    assert args.bayes_likelihood == "student_t"
    assert float(args.bayes_student_t_nu) == 6.5


def test_posterior_summary_tracks_selected_likelihood(monkeypatch) -> None:
    monkeypatch.setattr(calibrator_module, "pm", None)
    train_df = pd.DataFrame(
        {
            "feature_a": np.linspace(0.0, 1.0, 12),
            "feature_b": np.linspace(1.0, 2.0, 12),
            "purity_target": np.linspace(60.0, 78.0, 12),
            "yield_target": np.linspace(35.0, 61.0, 12),
            "source_id": ["s1"] * 6 + ["s2"] * 6,
            "purity_stage_canon": ["crude_hplc"] * 12,
            "yield_stage_canon": ["isolated"] * 12,
        }
    )

    calibrator = BayesianMultitaskCalibrator(
        draws=12,
        tune=12,
        chains=1,
        likelihood="student_t",
        student_t_nu=5.0,
    )
    calibrator.fit(
        train_df=train_df,
        features=["feature_a", "feature_b"],
        targets=["purity_target", "yield_target"],
        group_cols={
            "source": "source_id",
            "purity_target": "purity_stage_canon",
            "yield_target": "yield_stage_canon",
        },
    )
    summary = calibrator.posterior_summary()
    assert summary["likelihood"] == "student_t"
    assert float(summary["studentTNu"]) == 5.0
