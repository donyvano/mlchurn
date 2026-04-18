"""Dataset ingestion and schema validation for the Telco Churn dataset."""

import logging
import os
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

EXPECTED_COLUMNS: dict[str, type] = {
    "customerID": str,
    "gender": str,
    "SeniorCitizen": int,
    "Partner": str,
    "Dependents": str,
    "tenure": int,
    "PhoneService": str,
    "MultipleLines": str,
    "InternetService": str,
    "OnlineSecurity": str,
    "OnlineBackup": str,
    "DeviceProtection": str,
    "TechSupport": str,
    "StreamingTV": str,
    "StreamingMovies": str,
    "Contract": str,
    "PaperlessBilling": str,
    "PaymentMethod": str,
    "MonthlyCharges": float,
    "TotalCharges": str,
    "Churn": str,
}

RAW_DATA_PATH = Path(os.getenv("DATA_RAW_PATH", "data/raw/telco_churn.csv"))


def download_dataset(output_path: Path = RAW_DATA_PATH, force: bool = False) -> Path:
    """Download the Telco Customer Churn dataset from IBM's GitHub.

    Args:
        output_path: Destination file path for the CSV.
        force: Overwrite existing file if True.

    Returns:
        Path to the downloaded file.

    Raises:
        requests.HTTPError: If the download request fails.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        logger.info("Dataset already exists at %s — skipping download", output_path)
        return output_path

    logger.info("Downloading Telco Churn dataset from %s", DATASET_URL)
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    logger.info("Dataset saved to %s (%d bytes)", output_path, len(response.content))
    return output_path


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that the dataframe matches the expected schema.

    Args:
        df: Raw dataframe to validate.

    Raises:
        ValueError: If required columns are missing or data quality checks fail.
    """
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("Dataset is empty")

    missing_pct = df.isnull().mean()
    critical_cols = ["tenure", "MonthlyCharges", "Churn"]
    for col in critical_cols:
        if missing_pct[col] > 0.05:
            raise ValueError(
                f"Column '{col}' has {missing_pct[col]:.1%} missing values — "
                "exceeds 5% threshold"
            )

    valid_churn = {"Yes", "No"}
    actual_churn = set(df["Churn"].dropna().unique())
    if not actual_churn.issubset(valid_churn):
        raise ValueError(
            f"Unexpected Churn values: {actual_churn - valid_churn}"
        )

    if df["tenure"].lt(0).any():
        raise ValueError("Negative tenure values detected")

    if df["MonthlyCharges"].lt(0).any():
        raise ValueError("Negative MonthlyCharges values detected")

    logger.info(
        "Schema validation passed — %d rows, %d columns", len(df), len(df.columns)
    )


def load_raw_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load and do minimal cleaning on the raw dataset.

    Handles the TotalCharges whitespace issue present in the original dataset.

    Args:
        path: Path to the raw CSV file.

    Returns:
        Cleaned dataframe ready for the preprocessing pipeline.
    """
    df = pd.read_csv(path)
    validate_schema(df)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop(columns=["customerID"])

    n_dropped = df["TotalCharges"].isna().sum()
    if n_dropped > 0:
        logger.warning("Dropping %d rows with unparseable TotalCharges", n_dropped)
        df = df.dropna(subset=["TotalCharges"])

    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    logger.info("Loaded dataset: %d rows after cleaning", len(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    path = download_dataset()
    df = load_raw_dataset(path)
    logger.info("Dataset ready: shape=%s, churn_rate=%.2f%%", df.shape, df["Churn"].mean() * 100)
