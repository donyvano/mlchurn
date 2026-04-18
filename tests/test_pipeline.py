"""Unit tests for the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from models.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessing_pipeline,
    get_feature_names,
    split_features_target,
)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Minimal valid dataframe matching the Telco Churn schema."""
    return pd.DataFrame({
        "gender": ["Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [12, 48, 2],
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["Fiber optic", "DSL", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "No", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [70.35, 55.10, 19.90],
        "TotalCharges": [843.40, 2644.80, 39.80],
        "Churn": [1, 0, 1],
    })


class TestSplitFeaturesTarget:
    def test_splits_correctly(self, sample_df: pd.DataFrame) -> None:
        X, y = split_features_target(sample_df)
        assert TARGET_COLUMN not in X.columns
        assert len(y) == len(sample_df)
        assert set(y.unique()).issubset({0, 1})

    def test_raises_on_missing_target(self, sample_df: pd.DataFrame) -> None:
        df_no_target = sample_df.drop(columns=[TARGET_COLUMN])
        with pytest.raises(ValueError, match="Target column"):
            split_features_target(df_no_target)


class TestBuildPreprocessingPipeline:
    def test_returns_column_transformer(self, sample_df: pd.DataFrame) -> None:
        preprocessor = build_preprocessing_pipeline()
        assert isinstance(preprocessor, ColumnTransformer)

    def test_fit_transform_produces_array(self, sample_df: pd.DataFrame) -> None:
        X, _ = split_features_target(sample_df)
        preprocessor = build_preprocessing_pipeline()
        transformed = preprocessor.fit_transform(X)
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape[0] == len(sample_df)

    def test_transform_has_no_nans(self, sample_df: pd.DataFrame) -> None:
        X, _ = split_features_target(sample_df)
        preprocessor = build_preprocessing_pipeline()
        transformed = preprocessor.fit_transform(X)
        assert not np.isnan(transformed).any(), "Transformed array contains NaN values"

    def test_handles_unknown_categories(self, sample_df: pd.DataFrame) -> None:
        X, _ = split_features_target(sample_df)
        preprocessor = build_preprocessing_pipeline()
        preprocessor.fit_transform(X)

        unseen = sample_df.copy()
        unseen["Contract"] = "Lifetime"
        X_unseen, _ = split_features_target(unseen)
        result = preprocessor.transform(X_unseen)
        assert not np.isnan(result).any()

    def test_numeric_features_are_scaled(self, sample_df: pd.DataFrame) -> None:
        X, _ = split_features_target(sample_df)
        preprocessor = build_preprocessing_pipeline()
        transformed = preprocessor.fit_transform(X)
        n_numeric = len(NUMERIC_FEATURES)
        numeric_part = transformed[:, :n_numeric]
        assert numeric_part.std(axis=0).mean() < 5.0


class TestGetFeatureNames:
    def test_returns_list_of_strings(self, sample_df: pd.DataFrame) -> None:
        X, _ = split_features_target(sample_df)
        preprocessor = build_preprocessing_pipeline()
        preprocessor.fit_transform(X)
        names = get_feature_names(preprocessor)
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_numeric_features_first(self, sample_df: pd.DataFrame) -> None:
        X, _ = split_features_target(sample_df)
        preprocessor = build_preprocessing_pipeline()
        preprocessor.fit_transform(X)
        names = get_feature_names(preprocessor)
        for i, feat in enumerate(NUMERIC_FEATURES):
            assert names[i] == feat
