"""Airflow DAG: automated 24h churn model retraining pipeline."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG  # type: ignore[import-untyped]
from airflow.operators.python import PythonOperator  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

DAG_PARAMS = {
    "n_optuna_trials": int(os.getenv("OPTUNA_N_TRIALS", "50")),
    "promotion_threshold": float(os.getenv("PROMOTION_THRESHOLD", "0.01")),
}


def task_ingest_data(**context: Any) -> str:
    """Download and cache the Telco Churn dataset.

    Args:
        context: Airflow task context (injected automatically).

    Returns:
        String path to the downloaded file.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    from data.ingest import download_dataset

    path = download_dataset(force=False)
    logger.info("Data ingestion complete: %s", path)
    return str(path)


def task_validate_schema(**context: Any) -> None:
    """Validate the raw dataset schema and reject if checks fail.

    Args:
        context: Airflow task context.

    Raises:
        ValueError: If schema validation fails.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    from data.ingest import load_raw_dataset, RAW_DATA_PATH

    df = load_raw_dataset(RAW_DATA_PATH)
    logger.info("Schema validation passed — %d rows", len(df))


def task_run_preprocessing(**context: Any) -> str:
    """Fit the preprocessing pipeline and save it to disk.

    Args:
        context: Airflow task context.

    Returns:
        Path to the saved pipeline artifact.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    from data.ingest import load_raw_dataset
    from models.pipeline import (
        build_preprocessing_pipeline,
        save_pipeline,
        split_features_target,
    )
    from pathlib import Path

    df = load_raw_dataset()
    X, _ = split_features_target(df)
    preprocessor = build_preprocessing_pipeline()
    preprocessor.fit(X)

    pipeline_path = Path("data/processed/preprocessor.joblib")
    save_pipeline(preprocessor, pipeline_path)
    logger.info("Preprocessing pipeline saved to %s", pipeline_path)
    return str(pipeline_path)


def task_train_models(**context: Any) -> dict[str, str]:
    """Run full Optuna training for XGBoost and LightGBM.

    Args:
        context: Airflow task context.

    Returns:
        Dict mapping model_type → MLflow run_id.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    params = context["params"]
    n_trials = params.get("n_optuna_trials", DAG_PARAMS["n_optuna_trials"])

    os.environ["OPTUNA_N_TRIALS"] = str(n_trials)

    from models.train import run_training_pipeline

    run_ids = run_training_pipeline()
    logger.info("Training complete: %s", run_ids)
    return run_ids


def task_evaluate(**context: Any) -> None:
    """Log evaluation summary for the latest Staging model.

    Args:
        context: Airflow task context.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    from models.registry import _get_client, get_latest_staging_auc

    client = _get_client()
    result = get_latest_staging_auc(client)
    if result:
        version, auc = result
        logger.info("Latest Staging model v%s — AUC=%.4f", version, auc)
    else:
        logger.warning("No Staging model found for evaluation step")


def task_promote_if_better(**context: Any) -> bool:
    """Promote Staging model to Production if it beats current prod by threshold.

    Args:
        context: Airflow task context.

    Returns:
        True if promotion occurred, False otherwise.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")

    params = context["params"]
    threshold = params.get("promotion_threshold", DAG_PARAMS["promotion_threshold"])

    from models.registry import run_promotion_check

    promoted = run_promotion_check(threshold=threshold)
    logger.info("Promotion result: %s", "promoted" if promoted else "no change")
    return promoted


def task_notify(**context: Any) -> None:
    """Log pipeline completion summary (extend with Slack/email integration).

    Args:
        context: Airflow task context.
    """
    promoted = context["ti"].xcom_pull(task_ids="promote_if_better")
    status_msg = "New model promoted to Production" if promoted else "No promotion — current model retained"
    logger.info("Pipeline complete — %s", status_msg)


with DAG(
    dag_id="retrain_churn_pipeline",
    description="Automated 24h retraining pipeline for churn prediction",
    default_args=DEFAULT_ARGS,
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "churn", "retraining"],
    params=DAG_PARAMS,
    doc_md="""
    ## Churn Retraining Pipeline

    Automated daily retraining pipeline that:
    1. Downloads and validates the Telco Churn dataset
    2. Fits the preprocessing pipeline
    3. Trains XGBoost and LightGBM with Optuna tuning
    4. Evaluates the best model
    5. Promotes to Production if AUC improves by `promotion_threshold`

    **Configurable params:** `n_optuna_trials`, `promotion_threshold`
    """,
) as dag:

    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=task_ingest_data,
    )

    validate = PythonOperator(
        task_id="validate_schema",
        python_callable=task_validate_schema,
    )

    preprocess = PythonOperator(
        task_id="run_preprocessing",
        python_callable=task_run_preprocessing,
    )

    train = PythonOperator(
        task_id="train_models",
        python_callable=task_train_models,
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=task_evaluate,
    )

    promote = PythonOperator(
        task_id="promote_if_better",
        python_callable=task_promote_if_better,
    )

    notify = PythonOperator(
        task_id="notify",
        python_callable=task_notify,
    )

    ingest >> validate >> preprocess >> train >> evaluate >> promote >> notify
