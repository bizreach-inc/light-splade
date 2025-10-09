"""Wrapper utilities for BEIR metric evaluation.

This module adapts BEIR's :func:`EvaluateRetrieval.evaluate` to a simpler input format used by this codebase: ``qrels``
is a mapping from query id to list of positive doc ids, and ``results`` is a mapping from query id to an ordered list of
retrieved doc ids. The :class:`MetricsEvaluator` provides a convenience :meth:`evaluate` method returning a flattened
dict of metric@k scores.
"""

from beir.retrieval.evaluation import EvaluateRetrieval

SUPPORTED_METRICS = {"mrr", "ndcg", "map", "recall", "precision"}
DEFAULT_METRICS = ["map", "ndcg", "recall", "precision"]


def transform_name(metric_at_k: str) -> str:
    metric_at_k = (
        metric_at_k.replace("NDCG@", "ndcg@")
        .replace("MAP@", "map@")
        .replace("Recall@", "recall@")
        .replace("P@", "precision@")
        .replace("MRR@", "mrr@")
    )
    return metric_at_k


class MetricsEvaluator:
    @staticmethod
    def evaluate(
        qrels: dict[str, list[str]],
        results: dict[str, list[str]],
        k_values: list[int],
        metrics: list[str] = DEFAULT_METRICS,
    ) -> dict[str, float]:
        """Evaluate retrieval metrics using BEIR under a simplified interface.
        Mimic the BEIR's EvaluateRetrieval.evaluate method, but provide only positive ID list to `qrels` and ranked
        list of doc_ids to `results` instead of dicts.
        https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py#L41C9-L41C17

        Args:
            qrels (dict[str, list[str]]): List of positive IDs, Order is unspecified.
            results (dict[str, list[str]]): List of doc_ids sorted by similarity scores.
            k_values (list[int]): List of cutoffs.
            metrics (list[str], optional): List of metrics to be calculated. Defaults to ["map", "ndcg", "rec", "prec"].

        Returns:
            A mapping from metric specification strings (e.g. ``"ndcg@10"``)
            to their computed floating-point scores.
        """

        metrics = [metric.lower() for metric in metrics]
        for metric in metrics:
            assert metric in SUPPORTED_METRICS, f"Only {SUPPORTED_METRICS} metrics are supported!"
        assert len(metrics) > 0, "At least one metric must be provided!"

        # transform to BEIR's EvaluateRetrieval.evaluate input format pytrec_eval, which is called from beir, requires
        # query_ids and doc_ids in string type.
        qrels_with_label = {
            str(query_id): {str(doc_id): 1 for doc_id in doc_ids} for query_id, doc_ids in qrels.items()
        }
        results_with_score = {
            str(query_id): {str(doc_id): 0.01 * (idx + 1) for idx, doc_id in enumerate(doc_ids[::-1])}
            for query_id, doc_ids in results.items()
        }

        beir_evaluator = EvaluateRetrieval(k_values=k_values)
        scores = dict()
        if set(metrics).intersection(set(DEFAULT_METRICS)):
            scores_ = beir_evaluator.evaluate(
                qrels=qrels_with_label,
                results=results_with_score,
                k_values=k_values,
            )
            scores["ndcg"] = scores_[0]
            scores["map"] = scores_[1]
            scores["recall"] = scores_[2]
            scores["precision"] = scores_[3]

        for metric in ["mrr"]:
            if metric in metrics:
                scores[metric] = beir_evaluator.evaluate_custom(
                    qrels=qrels_with_label,
                    results=results_with_score,
                    k_values=k_values,
                    metric=metric,
                )

        final_scores = dict()
        for metric, scores_at in scores.items():
            if metric not in metrics:
                continue
            for metric_at, score in scores_at.items():
                final_scores[transform_name(metric_at)] = score

        return final_scores
