"""Unit tests for the MLflow model promotion logic."""

from unittest.mock import MagicMock, patch

import pytest

from models.registry import (
    get_current_production_auc,
    get_latest_staging_auc,
    promote_staging_to_production,
    run_promotion_check,
)


def _make_version(version: str, run_id: str) -> MagicMock:
    """Build a mock ModelVersion object."""
    mv = MagicMock()
    mv.version = version
    mv.run_id = run_id
    mv.creation_timestamp = 1705312800000
    return mv


def _make_run(auc: float) -> MagicMock:
    """Build a mock MLflow Run with a given auc_roc metric."""
    run = MagicMock()
    run.data.metrics = {"auc_roc": auc}
    return run


@pytest.fixture()
def mock_client() -> MagicMock:
    """Return a MagicMock that mimics MlflowClient."""
    return MagicMock()


class TestGetLatestStagingAuc:
    def test_returns_version_and_auc(self, mock_client: MagicMock) -> None:
        mv = _make_version("2", "run-abc")
        mock_client.get_latest_versions.return_value = [mv]
        mock_client.get_run.return_value = _make_run(0.874)

        result = get_latest_staging_auc(mock_client)

        assert result is not None
        version, auc = result
        assert version == "2"
        assert auc == pytest.approx(0.874)

    def test_returns_none_when_no_staging(self, mock_client: MagicMock) -> None:
        mock_client.get_latest_versions.return_value = []
        assert get_latest_staging_auc(mock_client) is None

    def test_returns_none_when_no_auc_metric(self, mock_client: MagicMock) -> None:
        mv = _make_version("1", "run-xyz")
        mock_client.get_latest_versions.return_value = [mv]
        run = MagicMock()
        run.data.metrics = {}
        mock_client.get_run.return_value = run

        assert get_latest_staging_auc(mock_client) is None

    def test_returns_none_on_mlflow_exception(self, mock_client: MagicMock) -> None:
        import mlflow
        mock_client.get_latest_versions.side_effect = mlflow.exceptions.MlflowException("err")
        assert get_latest_staging_auc(mock_client) is None


class TestGetCurrentProductionAuc:
    def test_returns_version_and_auc(self, mock_client: MagicMock) -> None:
        mv = _make_version("1", "run-prod")
        mock_client.get_latest_versions.return_value = [mv]
        mock_client.get_run.return_value = _make_run(0.860)

        result = get_current_production_auc(mock_client)
        assert result == ("1", pytest.approx(0.860))

    def test_returns_none_when_no_production(self, mock_client: MagicMock) -> None:
        mock_client.get_latest_versions.return_value = []
        assert get_current_production_auc(mock_client) is None


class TestPromoteStagingToProduction:
    def test_transitions_staging_version(self, mock_client: MagicMock) -> None:
        mock_client.get_latest_versions.return_value = []
        promote_staging_to_production(mock_client, "3", archive_current=False)

        mock_client.transition_model_version_stage.assert_called_once_with(
            name=mock_client.transition_model_version_stage.call_args[1]["name"],
            version="3",
            stage="Production",
        )

    def test_archives_current_production_before_promoting(self, mock_client: MagicMock) -> None:
        old_prod = _make_version("1", "run-old")
        mock_client.get_latest_versions.return_value = [old_prod]

        promote_staging_to_production(mock_client, "2", archive_current=True)

        calls = mock_client.transition_model_version_stage.call_args_list
        stages = [c[1]["stage"] for c in calls]
        assert "Archived" in stages
        assert "Production" in stages


class TestRunPromotionCheck:
    @patch("models.registry._get_client")
    def test_promotes_when_no_production_exists(self, mock_get_client: MagicMock) -> None:
        client = MagicMock()
        mock_get_client.return_value = client

        staging_mv = _make_version("1", "run-s")
        client.get_latest_versions.side_effect = lambda name, stages: (
            [staging_mv] if "Staging" in stages else []
        )
        client.get_run.return_value = _make_run(0.874)

        result = run_promotion_check(threshold=0.01)
        assert result is True

    @patch("models.registry._get_client")
    def test_promotes_when_improvement_exceeds_threshold(self, mock_get_client: MagicMock) -> None:
        client = MagicMock()
        mock_get_client.return_value = client

        staging_mv = _make_version("2", "run-s")
        prod_mv = _make_version("1", "run-p")

        def side_effect(name, stages):
            if "Staging" in stages:
                return [staging_mv]
            if "Production" in stages:
                return [prod_mv]
            return []

        client.get_latest_versions.side_effect = side_effect
        client.get_run.side_effect = lambda run_id: (
            _make_run(0.885) if run_id == "run-s" else _make_run(0.860)
        )

        result = run_promotion_check(threshold=0.01)
        assert result is True

    @patch("models.registry._get_client")
    def test_does_not_promote_when_below_threshold(self, mock_get_client: MagicMock) -> None:
        client = MagicMock()
        mock_get_client.return_value = client

        staging_mv = _make_version("2", "run-s")
        prod_mv = _make_version("1", "run-p")

        def side_effect(name, stages):
            if "Staging" in stages:
                return [staging_mv]
            if "Production" in stages:
                return [prod_mv]
            return []

        client.get_latest_versions.side_effect = side_effect
        client.get_run.side_effect = lambda run_id: (
            _make_run(0.862) if run_id == "run-s" else _make_run(0.860)
        )

        result = run_promotion_check(threshold=0.01)
        assert result is False

    @patch("models.registry._get_client")
    def test_returns_false_when_no_staging(self, mock_get_client: MagicMock) -> None:
        client = MagicMock()
        mock_get_client.return_value = client
        client.get_latest_versions.return_value = []

        result = run_promotion_check()
        assert result is False
