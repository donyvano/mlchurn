"""Scikit-learn preprocessing pipeline for the Telco Churn dataset."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

TARGET_COLUMN = "Churn"


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build the full preprocessing ColumnTransformer.

    Numeric features: median imputation → standard scaling.
    Categorical features: constant imputation → one-hot encoding (drop first).

    Returns:
        Fitted-ready ColumnTransformer instance.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into feature matrix and target vector.

    Args:
        df: Raw dataframe containing features and target.

    Returns:
        Tuple of (X, y) where X is the feature dataframe and y is the target series.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def save_pipeline(pipeline: ColumnTransformer, path: Path) -> None:
    """Serialize and save the fitted pipeline to disk.

    Args:
        pipeline: Fitted ColumnTransformer to persist.
        path: Destination file path (typically .joblib extension).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Pipeline saved to %s", path)


def load_pipeline(path: Path) -> ColumnTransformer:
    """Load a serialized preprocessing pipeline from disk.

    Args:
        path: Path to the serialized pipeline file.

    Returns:
        Deserialized ColumnTransformer.

    Raises:
        FileNotFoundError: If no pipeline file exists at the given path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No pipeline found at {path}")
    pipeline = joblib.load(path)
    logger.info("Pipeline loaded from %s", path)
    return pipeline


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract human-readable feature names after one-hot encoding.

    Args:
        preprocessor: Fitted ColumnTransformer.

    Returns:
        List of feature name strings in the same order as the transformed columns.
    """
    ohe: OneHotEncoder = preprocessor.named_transformers_["categorical"]["encoder"]
    cat_feature_names = [str(f) for f in ohe.get_feature_names_out(CATEGORICAL_FEATURES)]
    return NUMERIC_FEATURES + cat_feature_names
