from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 20260305


class ResidualEnsembleRegressor:
    def __init__(self, *, model: str = "voting", random_state: int = RANDOM_STATE) -> None:
        self.model = model
        self.random_state = random_state

    def _build_model(self) -> VotingRegressor:
        return VotingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(n_estimators=300, random_state=self.random_state, n_jobs=1)),
                ("et", ExtraTreesRegressor(n_estimators=300, random_state=self.random_state, n_jobs=1)),
                ("gbr", GradientBoostingRegressor(random_state=self.random_state)),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )

    def _build_preprocessor(self, x_df: pd.DataFrame) -> ColumnTransformer:
        numeric_columns = x_df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical_columns = [column for column in x_df.columns if column not in numeric_columns]
        return ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_columns,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical_columns,
                ),
            ],
            sparse_threshold=0.0,
        )

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_columns: list[str],
        residual_targets: pd.DataFrame,
    ) -> "ResidualEnsembleRegressor":
        x_train = train_df[feature_columns].copy()
        self.feature_columns_ = list(feature_columns)
        self.target_columns_ = residual_targets.columns.tolist()
        self.models_: dict[str, Pipeline] = {}
        self.importances_: dict[str, list[dict[str, float]]] = {}

        for idx, target in enumerate(self.target_columns_):
            preprocessor = self._build_preprocessor(x_train)
            model = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("regressor", self._build_model()),
                ]
            )
            model.fit(x_train, residual_targets[target].to_numpy(dtype=float))
            self.models_[target] = model

            permutation = permutation_importance(
                estimator=model,
                X=x_train,
                y=residual_targets[target].to_numpy(dtype=float),
                scoring="neg_root_mean_squared_error",
                n_repeats=12,
                random_state=self.random_state + (idx * 37),
                n_jobs=1,
            )
            ranking = sorted(
                zip(self.feature_columns_, permutation.importances_mean.tolist(), permutation.importances_std.tolist()),
                key=lambda item: item[1],
                reverse=True,
            )
            self.importances_[target] = [
                {
                    "feature": str(name),
                    "importanceMean": float(mean),
                    "importanceStd": float(std),
                }
                for name, mean, std in ranking[:20]
            ]
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        x_test = test_df[self.feature_columns_].copy()
        predictions = {
            target: model.predict(x_test)
            for target, model in self.models_.items()
        }
        return pd.DataFrame(predictions, index=test_df.index)

    def feature_importance(self) -> dict[str, Any]:
        return self.importances_
