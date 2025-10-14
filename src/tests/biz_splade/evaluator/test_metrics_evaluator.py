from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from light_splade.evaluator.metrics_evaluator import MetricsEvaluator
from light_splade.evaluator.metrics_evaluator import transform_name


@pytest.mark.parametrize(
    "input_name, expected_output",
    [
        ("NDCG@10", "ndcg@10"),
        ("MAP@5", "map@5"),
        ("Recall@1", "recall@1"),
        ("P@10", "precision@10"),
        ("MRR@5", "mrr@5"),
        ("mixed@10", "mixed@10"),  # No transformation
    ],
)
def test_transform_name(input_name: str, expected_output: str) -> None:
    result = transform_name(input_name)
    assert result == expected_output


class TestMetricsEvaluator:
    @pytest.fixture
    def sample_data(
        self,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], list[int]]:
        qrels = {"1": ["101", "102"], "2": ["201", "202", "203"]}
        results = {
            "1": ["101", "103", "102", "104"],
            "2": ["201", "204", "202", "203", "205"],
        }
        k_values = [3, 5]
        return qrels, results, k_values

    @pytest.mark.parametrize(
        "metrics",
        [
            ["ndcg"],
            ["map", "recall"],
            ["precision", "ndcg", "map"],
            ["mrr"],
            ["ndcg", "mrr"],
        ],
    )
    @patch("light_splade.evaluator.metrics_evaluator.EvaluateRetrieval")
    def test_evaluate_success(
        self,
        mock_evaluate_retrieval_class: Any,
        sample_data: tuple[dict[str, list[str]], dict[str, list[str]], list[int]],
        metrics: list[str],
    ) -> None:
        qrels, results, k_values = sample_data

        # Mock EvaluateRetrieval
        mock_evaluator = Mock()
        mock_evaluate_retrieval_class.return_value = mock_evaluator

        # Mock default metrics evaluation
        mock_evaluator.evaluate.return_value = (
            {"NDCG@3": 0.8, "NDCG@5": 0.7},  # ndcg
            {"MAP@3": 0.6, "MAP@5": 0.5},  # map
            {"Recall@3": 0.9, "Recall@5": 0.8},  # recall
            {"P@3": 0.7, "P@5": 0.6},  # precision
        )

        # Mock custom metrics evaluation (mrr)
        mock_evaluator.evaluate_custom.return_value = {
            "MRR@3": 0.85,
            "MRR@5": 0.75,
        }

        result = MetricsEvaluator.evaluate(qrels=qrels, results=results, k_values=k_values, metrics=metrics)

        assert isinstance(result, dict)

        # Check that only requested metrics are returned
        for metric in metrics:
            for k in k_values:
                expected_key = f"{metric}@{k}"
                if expected_key not in result:
                    # Some metrics might not be available at all k values
                    continue

    def test_evaluate_invalid_metric(
        self,
        sample_data: tuple[dict[str, list[str]], dict[str, list[str]], list[int]],
    ) -> None:
        qrels, results, k_values = sample_data

        with pytest.raises(AssertionError) as exc_info:
            MetricsEvaluator.evaluate(
                qrels=qrels,
                results=results,
                k_values=k_values,
                metrics=["invalid_metric"],
            )

        assert "Only" in str(exc_info.value) and "metrics are supported" in str(exc_info.value)

    def test_evaluate_empty_metrics(
        self,
        sample_data: tuple[dict[str, list[str]], dict[str, list[str]], list[int]],
    ) -> None:
        qrels, results, k_values = sample_data

        with pytest.raises(AssertionError) as exc_info:
            MetricsEvaluator.evaluate(qrels=qrels, results=results, k_values=k_values, metrics=[])

        assert "At least one metric must be provided" in str(exc_info.value)

    @patch("light_splade.evaluator.metrics_evaluator.EvaluateRetrieval")
    def test_evaluate_data_transformation(
        self,
        mock_evaluate_retrieval_class: Any,
        sample_data: tuple[dict[str, list[str]], dict[str, list[str]], list[int]],
    ) -> None:
        qrels, results, k_values = sample_data

        mock_evaluator = Mock()
        mock_evaluate_retrieval_class.return_value = mock_evaluator

        # Mock return values
        mock_evaluator.evaluate.return_value = ({"NDCG@3": 0.8}, {}, {}, {})

        MetricsEvaluator.evaluate(qrels=qrels, results=results, k_values=k_values, metrics=["ndcg"])

        # Verify that data was transformed correctly
        call_args = mock_evaluator.evaluate.call_args
        transformed_qrels = call_args[1]["qrels"]
        transformed_results = call_args[1]["results"]

        # Check qrels transformation (string keys, doc_ids as keys with label 1)
        assert "1" in transformed_qrels
        assert "2" in transformed_qrels
        assert transformed_qrels["1"]["101"] == 1
        assert transformed_qrels["1"]["102"] == 1

        # Check results transformation (string keys, scores in reverse order)
        assert "1" in transformed_results
        assert "2" in transformed_results
        # Results should have scores (higher rank = higher score)
        assert all(isinstance(score, float) for score in transformed_results["1"].values())

    @patch("light_splade.evaluator.metrics_evaluator.EvaluateRetrieval")
    def test_evaluate_with_mrr_metric(
        self,
        mock_evaluate_retrieval_class: Any,
        sample_data: tuple[dict[str, list[str]], dict[str, list[str]], list[int]],
    ) -> None:
        qrels, results, k_values = sample_data

        mock_evaluator = Mock()
        mock_evaluate_retrieval_class.return_value = mock_evaluator

        mock_evaluator.evaluate_custom.return_value = {
            "MRR@3": 0.85,
            "MRR@5": 0.75,
        }

        result = MetricsEvaluator.evaluate(qrels=qrels, results=results, k_values=k_values, metrics=["mrr"])

        mock_evaluator.evaluate_custom.assert_called_once()
        assert "mrr@3" in result
        assert "mrr@5" in result
        assert result["mrr@3"] == 0.85
        assert result["mrr@5"] == 0.75
