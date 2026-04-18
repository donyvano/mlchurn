"""Main training script: Optuna tuning + MLflow tracking for XGBoost and LightGBM."""

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from data.ingest import download_dataset, load_raw_dataset
from models.evaluate import compute_metrics, log_artifacts_to_mlflow
from models.pipeline import (
    build_preprocessing_pipeline,
    save_pipeline,
    split_features_target,
)

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-prediction")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn-classifier")
N_TRIALS = int(os.getenv("OPTUNA_N_TRIALS", "50"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
PIPELINE_ARTIFACT_PATH = Path("data/processed/preprocessor.joblib")

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _xgb_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for XGBoost hyperparameter search.

    Args:
        trial: Optuna trial object providing hyperparameter suggestions.
        X: Preprocessed feature matrix.
        y: Binary target vector.

    Returns:
        Mean cross-validated AUC-ROC score.
    """
    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    clf = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


def _lgbm_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for LightGBM hyperparameter search.

    Args:
        trial: Optuna trial object providing hyperparameter suggestions.
        X: Preprocessed feature matrix.
        y: Binary target vector.

    Returns:
        Mean cross-validated AUC-ROC score.
    """
    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    clf = LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


def _tune_and_train(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int,
) -> tuple[Any, dict[str, Any], float]:
    """Run Optuna study and return the best model with its params and AUC.

    Args:
        model_type: Either "xgboost" or "lightgbm".
        X_train: Preprocessed training features.
        y_train: Training target vector.
        n_trials: Number of Optuna trials.

    Returns:
        Tuple of (fitted_model, best_params, best_auc).
    """
    objective_fn = _xgb_objective if model_type == "xgboost" else _lgbm_objective

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(
        lambda trial: objective_fn(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_auc = study.best_value
    logger.info("%s best trial AUC=%.4f", model_type, best_auc)

    if model_type == "xgboost":
        model = XGBClassifier(
            **best_params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    else:
        model = LGBMClassifier(**best_params, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)

    model.fit(X_train, y_train)
    return model, best_params, best_auc


def train_and_log(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    preprocessor: Any,
    n_trials: int = N_TRIALS,
) -> str:
    """Train one model variant, log everything to MLflow, return the run_id.

    Args:
        model_type: "xgboost" or "lightgbm".
        X_train: Preprocessed training features.
        y_train: Training labels.
        X_test: Preprocessed test features.
        y_test: Test labels.
        feature_names: Column names matching X_train columns.
        preprocessor: Fitted ColumnTransformer (logged as artifact).
        n_trials: Number of Optuna trials.

    Returns:
        MLflow run_id string.
    """
    with mlflow.start_run(run_name=f"{model_type}-optuna") as run:
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("framework", "optuna")

        model, best_params, cv_auc = _tune_and_train(model_type, X_train, y_train, n_trials)

        mlflow.log_params(best_params)
        mlflow.log_param("n_optuna_trials", n_trials)
        mlflow.log_param("random_seed", RANDOM_SEED)

        metrics = compute_metrics(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        logger.info("%s metrics: %s", model_type, metrics)

        log_artifacts_to_mlflow(
            model=model,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            model_type=model_type,
            preprocessor=preprocessor,
        )

        if model_type == "xgboost":
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
            )
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
            )

        return run.info.run_id


def run_training_pipeline() -> dict[str, str]:
    """Orchestrate full training: data loading, preprocessing, training both models.

    Returns:
        Dict mapping model_type to MLflow run_id for each trained model.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    download_dataset()
    df = load_raw_dataset()

    from sklearn.model_selection import train_test_split

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessing_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    save_pipeline(preprocessor, PIPELINE_ARTIFACT_PATH)

    from models.pipeline import get_feature_names
    feature_names = get_feature_names(preprocessor)

    run_ids: dict[str, str] = {}
    for model_type in ["xgboost", "lightgbm"]:
        logger.info("Training %s with %d Optuna trials", model_type, N_TRIALS)
        run_id = train_and_log(
            model_type=model_type,
            X_train=X_train_transformed,
            y_train=y_train.values,
            X_test=X_test_transformed,
            y_test=y_test.values,
            feature_names=feature_names,
            preprocessor=preprocessor,
            n_trials=N_TRIALS,
        )
        run_ids[model_type] = run_id
        logger.info("Completed %s — run_id=%s", model_type, run_id)

    return run_ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    run_ids = run_training_pipeline()
    logger.info("Training complete: %s", run_ids)
