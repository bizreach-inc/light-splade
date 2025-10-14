from unittest.mock import Mock

import numpy as np
import pytest
from scipy import sparse as sps

from light_splade.evaluator.sparse_retriever import SparseRetriever


class TestSparseRetriever:
    @pytest.fixture
    def mock_sparse_indexer(self) -> Mock:
        indexer = Mock()
        indexer.doc_id_list = ["101", "102", "103", "104"]

        # Mock index_vectors to return a simple csr_matrix
        indexer.index_vectors.return_value = sps.csr_matrix([[0.5, 0.3, 0.0, 0.2]])

        # Mock get_sparse_matrix to return document sparse matrix
        doc_sparse_matrix = sps.csr_matrix(
            [
                [0.4, 0.2, 0.1, 0.0],  # doc 101
                [0.3, 0.5, 0.0, 0.1],  # doc 102
                [0.0, 0.3, 0.6, 0.2],  # doc 103
                [0.2, 0.0, 0.4, 0.3],  # doc 104
            ]
        )
        indexer.get_sparse_matrix.return_value = doc_sparse_matrix

        return indexer

    def test___init__(self, mock_sparse_indexer: Mock) -> None:
        retriever = SparseRetriever(mock_sparse_indexer)

        assert retriever.sparse_indexer == mock_sparse_indexer

    def test_retrieve_basic(self, mock_sparse_indexer: Mock) -> None:
        retriever = SparseRetriever(mock_sparse_indexer)
        query_vectors = [{"token1": 0.5, "token2": 0.3}]

        result = retriever.retrieve(query_vectors, top_k=2)

        assert len(result) == 1  # One query
        assert len(result[0]) <= 2  # Top-2 results
        assert all(isinstance(doc_id, str) for doc_id in result[0])

    def test_retrieve_with_target_doc_ids(self, mock_sparse_indexer: Mock) -> None:
        retriever = SparseRetriever(mock_sparse_indexer)
        query_vectors = [{"token1": 0.5, "token2": 0.3}]
        target_doc_ids = ["101", "103"]

        # Mock get_sparse_matrix for specific doc_ids
        target_sparse_matrix = sps.csr_matrix(
            [[0.4, 0.2, 0.1, 0.0], [0.0, 0.3, 0.6, 0.2]]  # doc 101  # doc 103
        )
        mock_sparse_indexer.get_sparse_matrix.return_value = target_sparse_matrix

        result = retriever.retrieve(query_vectors, target_doc_ids=target_doc_ids, top_k=2)

        assert len(result) == 1
        assert all(doc_id in target_doc_ids for doc_id in result[0])
        mock_sparse_indexer.get_sparse_matrix.assert_called_with(target_doc_ids)

    @pytest.mark.parametrize(
        "top_k, expected_max_results",
        [
            (0, 4),  # Should return all documents
            (2, 2),
            (10, 4),  # More than available docs
        ],
    )
    def test_retrieve_top_k(self, mock_sparse_indexer: Mock, top_k: int, expected_max_results: int) -> None:
        retriever = SparseRetriever(mock_sparse_indexer)
        query_vectors = [{"token1": 0.5, "token2": 0.3}]

        result = retriever.retrieve(query_vectors, top_k=top_k)

        assert len(result) == 1
        assert len(result[0]) <= expected_max_results

    def test_retrieve_with_threshold(self, mock_sparse_indexer: Mock) -> None:
        retriever = SparseRetriever(mock_sparse_indexer)
        query_vectors = [{"token1": 0.5, "token2": 0.3}]
        threshold = 0.5  # High threshold

        result = retriever.retrieve(query_vectors, threshold=threshold, top_k=0)

        assert len(result) == 1
        # Results should be filtered by threshold
        assert len(result[0]) <= 4

    def test_retrieve_with_scores(self, mock_sparse_indexer: Mock) -> None:
        retriever = SparseRetriever(mock_sparse_indexer)
        query_vectors = [{"token1": 0.5, "token2": 0.3}]

        result = retriever.retrieve(query_vectors, return_score=True, top_k=2)

        assert len(result) == 1
        assert len(result[0]) <= 2

        # Each result should be a tuple (doc_id, score)
        for item in result[0]:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)  # doc_id
            assert isinstance(item[1], (float, np.floating))  # score

    def test_retrieve_multiple_queries_error(self, mock_sparse_indexer: Mock) -> None:
        retriever = SparseRetriever(mock_sparse_indexer)
        query_vectors = [
            {"token1": 0.5, "token2": 0.3},
            {"token1": 0.2, "token2": 0.8},
        ]
        target_doc_ids = ["101", "102"]  # Should cause error with multiple
        # queries

        with pytest.raises(AssertionError) as exc_info:
            retriever.retrieve(query_vectors, target_doc_ids=target_doc_ids)

        assert "Multiple query vectors" in str(exc_info.value)

    def test_retrieve_multiple_queries_success(self, mock_sparse_indexer: Mock) -> None:
        retriever = SparseRetriever(mock_sparse_indexer)

        # Mock index_vectors for multiple queries
        mock_sparse_indexer.index_vectors.return_value = sps.csr_matrix(
            [[0.5, 0.3, 0.0, 0.2], [0.2, 0.8, 0.1, 0.0]]  # query 1  # query 2
        )

        query_vectors = [
            {"token1": 0.5, "token2": 0.3},
            {"token1": 0.2, "token2": 0.8},
        ]

        result = retriever.retrieve(query_vectors, top_k=2)

        assert len(result) == 2  # Two queries
        assert len(result[0]) <= 2  # Top-2 for first query
        assert len(result[1]) <= 2  # Top-2 for second query

    def test_retrieve_no_results_above_threshold(self, mock_sparse_indexer: Mock) -> None:
        retriever = SparseRetriever(mock_sparse_indexer)
        query_vectors = [{"token1": 0.1, "token2": 0.1}]  # Low scores
        threshold = 0.9  # Very high threshold

        # Mock low similarity scores
        mock_sparse_indexer.index_vectors.return_value = sps.csr_matrix([[0.1, 0.1, 0.0, 0.0]])

        result = retriever.retrieve(query_vectors, threshold=threshold, top_k=0)

        assert len(result) == 1
        assert len(result[0]) == 0  # No results above threshold

    def test_retrieve_empty_query_vector(self, mock_sparse_indexer: Mock) -> None:
        retriever = SparseRetriever(mock_sparse_indexer)
        query_vectors: list[dict[str, float]] = [{}]  # Empty query vector

        # Mock empty query sparse matrix
        mock_sparse_indexer.index_vectors.return_value = sps.csr_matrix([[0.0, 0.0, 0.0, 0.0]])

        result = retriever.retrieve(query_vectors, top_k=2)

        assert len(result) == 1
        # Should handle empty vectors gracefully
