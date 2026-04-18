"""Shared pytest fixtures for the ChurnIQ test suite."""

import os

import pytest

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("MLFLOW_MODEL_NAME", "churn-classifier")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "churn-prediction")
os.environ.setdefault("OPTUNA_N_TRIALS", "5")
os.environ.setdefault("RANDOM_SEED", "42")
