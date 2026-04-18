"""MLflow Model Registry: promotion logic from Staging to Production."""

import logging
import os

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn-classifier")
PROMOTION_THRESHOLD = float(os.getenv("PROMOTION_THRESHOLD", "0.01"))


def _get_client() -> MlflowClient:
    """Return a configured MlflowClient."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def get_latest_staging_auc(client: MlflowClient) -> tuple[str, float] | None:
    """Retrieve the AUC-ROC of the latest model version in Staging.

    Args:
        client: Configured MlflowClient instance.

    Returns:
        Tuple (version_string, auc_float) or None if no Staging model exists.
    """
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    except MlflowException:
        logger.warning("Model '%s' not found in registry", MODEL_NAME)
        return None

    if not versions:
        logger.info("No model in Staging for '%s'", MODEL_NAME)
        return None

    latest = versions[0]
    if latest.run_id is None:
        logger.warning("Staging model v%s has no associated run_id", latest.version)
        return None
    run = client.get_run(latest.run_id)
    auc = run.data.metrics.get("auc_roc")

    if auc is None:
        logger.warning("Staging model v%s has no auc_roc metric", latest.version)
        return None

    return latest.version, float(auc)


def get_current_production_auc(client: MlflowClient) -> tuple[str, float] | None:
    """Retrieve the AUC-ROC of the current Production model version.

    Args:
        client: Configured MlflowClient instance.

    Returns:
        Tuple (version_string, auc_float) or None if no Production model exists.
    """
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    except MlflowException:
        return None

    if not versions:
        return None

    latest = versions[0]
    if latest.run_id is None:
        return None
    run = client.get_run(latest.run_id)
    auc = run.data.metrics.get("auc_roc")
    if auc is None:
        return None
    return latest.version, float(auc)


def promote_staging_to_production(
    client: MlflowClient, staging_version: str, archive_current: bool = True
) -> None:
    """Transition a Staging model version to Production.

    Args:
        client: Configured MlflowClient instance.
        staging_version: Version string of the Staging model to promote.
        archive_current: If True, archive the current Production model.
    """
    if archive_current:
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        for v in prod_versions:
            client.transition_model_version_stage(
                name=MODEL_NAME, version=v.version, stage="Archived"
            )
            logger.info("Archived previous Production model v%s", v.version)

    client.transition_model_version_stage(
        name=MODEL_NAME, version=staging_version, stage="Production"
    )
    logger.info("Promoted model v%s to Production", staging_version)


def run_promotion_check(threshold: float = PROMOTION_THRESHOLD) -> bool:
    """Compare Staging vs Production AUC and promote if improvement exceeds threshold.

    Args:
        threshold: Minimum relative AUC improvement required for promotion.
            A value of 0.01 means 1% relative improvement.

    Returns:
        True if promotion occurred, False otherwise.
    """
    client = _get_client()

    staging = get_latest_staging_auc(client)
    if staging is None:
        logger.info("Nothing to promote — no Staging model found")
        return False

    staging_version, staging_auc = staging
    production = get_current_production_auc(client)

    if production is None:
        logger.info(
            "No Production model exists — promoting Staging v%s (AUC=%.4f) unconditionally",
            staging_version,
            staging_auc,
        )
        promote_staging_to_production(client, staging_version)
        return True

    prod_version, prod_auc = production
    if prod_auc == 0:
        logger.warning("Production AUC is 0 — promoting Staging v%s unconditionally", staging_version)
        promote_staging_to_production(client, staging_version)
        return True
    relative_improvement = (staging_auc - prod_auc) / prod_auc

    logger.info(
        "Staging v%s AUC=%.4f vs Production v%s AUC=%.4f (Δ=%.2f%%)",
        staging_version,
        staging_auc,
        prod_version,
        prod_auc,
        relative_improvement * 100,
    )

    if relative_improvement >= threshold:
        logger.info(
            "Improvement %.2f%% exceeds threshold %.2f%% — promoting",
            relative_improvement * 100,
            threshold * 100,
        )
        promote_staging_to_production(client, staging_version)
        return True

    logger.info(
        "Improvement %.2f%% below threshold %.2f%% — keeping current Production model",
        relative_improvement * 100,
        threshold * 100,
    )
    return False


def force_promote_best_run(experiment_name: str) -> bool:
    """Find the run with highest AUC-ROC in the experiment and promote it to Production.

    Intended for use with `make promote` for manual overrides.

    Args:
        experiment_name: Name of the MLflow experiment to search.

    Returns:
        True if a model was promoted.
    """
    client = _get_client()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["metrics.auc_roc DESC"],
        max_results=1,
    )

    if not runs:
        logger.warning("No runs found in experiment '%s'", experiment_name)
        return False

    best_run = runs[0]
    best_auc = best_run.data.metrics.get("auc_roc", 0)

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    matching = [v for v in versions if v.run_id == best_run.info.run_id]

    if not matching:
        logger.warning("Best run %s has no registered model version", best_run.info.run_id)
        return False

    target_version = matching[0].version
    promote_staging_to_production(client, target_version, archive_current=True)
    logger.info(
        "Force-promoted v%s (AUC=%.4f) from run %s",
        target_version,
        best_auc,
        best_run.info.run_id,
    )
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    promoted = run_promotion_check()
    logger.info("Promotion result: %s", "promoted" if promoted else "no change")
