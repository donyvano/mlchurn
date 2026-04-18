"""Simulate production data drift for monitoring dashboard demonstration."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.ingest import load_raw_dataset

logger = logging.getLogger(__name__)


def simulate_drift(
    df: pd.DataFrame,
    drift_strength: float = 0.3,
    n_samples: int = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a drifted version of the dataset to simulate production shift.

    Applies targeted perturbations to numeric features to mimic a customer
    segment shift (e.g., influx of high-tenure, low-charge customers).

    Args:
        df: Reference dataframe (training distribution).
        drift_strength: Float in [0, 1] controlling perturbation magnitude.
        n_samples: Number of synthetic drifted samples to return.
        random_seed: Reproducibility seed.

    Returns:
        Dataframe with same schema as input but with shifted distributions.
    """
    rng = np.random.default_rng(random_seed)
    sample = df.sample(n=n_samples, replace=True, random_state=random_seed).copy()

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        col_std = sample[col].std()
        shift = drift_strength * col_std
        noise = rng.normal(loc=shift, scale=col_std * 0.1, size=len(sample))
        sample[col] = (sample[col] + noise).clip(lower=0)

    if drift_strength > 0.2:
        internet_proba = np.array([0.1, 0.5, 0.4])
        sample["InternetService"] = rng.choice(
            ["No", "DSL", "Fiber optic"], size=len(sample), p=internet_proba
        )

    logger.info(
        "Generated %d drifted samples with drift_strength=%.2f", n_samples, drift_strength
    )
    return sample


def generate_drift_history(
    df: pd.DataFrame,
    n_days: int = 7,
    output_dir: Path = Path("data/processed/drift_history"),
) -> list[Path]:
    """Generate daily snapshots with progressive drift for monitoring demo.

    Args:
        df: Reference dataframe.
        n_days: Number of historical days to generate.
        output_dir: Directory where daily CSVs are saved.

    Returns:
        List of paths to the generated snapshot files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for day in range(n_days):
        drift_strength = day * 0.05
        drifted = simulate_drift(df, drift_strength=drift_strength, random_seed=day)
        path = output_dir / f"day_{day:02d}.csv"
        drifted.to_csv(path, index=False)
        paths.append(path)
        logger.info("Saved drift snapshot day=%d to %s", day, path)

    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    df = load_raw_dataset()
    generate_drift_history(df)
