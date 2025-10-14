from typing import Any
from unittest.mock import mock_open
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse as sps

from light_splade.evaluator.sparse_indexer import SparseIndexer


class TestSparseIndexer:
    @pytest.fixture
    def sample_vocab(self) -> list[str]:
        return ["hello", "world", "test", "##sub"]

    @pytest.fixture
    def sample_sparse_vectors(self) -> list[dict[str, float]]:
        return [
            {"hello": 0.5, "world": 0.3},
            {"test": 0.8, "hello": 0.2},
            {"world": 0.6, "##sub": 0.4},
        ]

    def test___init___basic(self, sample_vocab: list[str]) -> None:
        indexer = SparseIndexer(vocab=sample_vocab)

        assert indexer.vocab == sample_vocab
        assert len(indexer.term2index) == len(sample_vocab)
        assert indexer.dtype == np.float32
        assert indexer.max_cache_size == 1000
        assert indexer.sparse_matrix.shape == (0, len(sample_vocab))
        assert len(indexer.doc_id_list) == 0
        assert len(indexer.docid2index) == 0

    def test___init___with_custom_params(self, sample_vocab: list[str]) -> None:
        term2index = {"hello": 0, "world": 1}
        docid2index = {"doc1": 0, "doc2": 1}
        doc_id_list = ["doc1", "doc2"]
        sparse_matrix = sps.csr_matrix((2, len(sample_vocab)))

        indexer = SparseIndexer(
            vocab=sample_vocab,
            term2index=term2index,
            docid2index=docid2index,
            doc_id_list=doc_id_list,
            sparse_matrix=sparse_matrix,
            dtype=np.float64,
            max_cache_size=500,
        )

        assert indexer.term2index == term2index
        assert indexer.docid2index == docid2index
        assert indexer.doc_id_list == doc_id_list
        assert indexer.sparse_matrix is sparse_matrix
        assert indexer.dtype == np.float64
        assert indexer.max_cache_size == 500

    def test__index_vocab(self, sample_vocab: list[str]) -> None:
        indexer = SparseIndexer(vocab=sample_vocab)

        expected_term2index = {term: idx for idx, term in enumerate(sample_vocab)}
        assert indexer.term2index == expected_term2index

    def test_index_vectors_sparse(
        self,
        sample_vocab: list[str],
        sample_sparse_vectors: list[dict[str, float]],
    ) -> None:
        indexer = SparseIndexer(vocab=sample_vocab)

        result = indexer.index_vectors(sample_sparse_vectors)

        assert isinstance(result, sps.csr_matrix)
        assert result.shape == (len(sample_sparse_vectors), len(sample_vocab))
        assert result.dtype == indexer.dtype

    def test_index_vectors_dense(self, sample_vocab: list[str]) -> None:
        indexer = SparseIndexer(vocab=sample_vocab)
        dense_vectors = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])

        result = indexer.index_vectors(dense_vectors)

        assert isinstance(result, sps.csr_matrix)
        assert result.shape == dense_vectors.shape
        assert np.allclose(result.toarray(), dense_vectors)

    def test_index_docs_no_cache(
        self,
        sample_vocab: list[str],
        sample_sparse_vectors: list[dict[str, float]],
    ) -> None:
        indexer = SparseIndexer(vocab=sample_vocab)
        doc_ids = ["101", "102", "103"]

        indexer.index_docs(doc_ids, sample_sparse_vectors, use_cache=False)

        assert len(indexer.doc_id_list) == len(doc_ids)
        assert indexer.doc_id_list == doc_ids
        assert len(indexer.docid2index) == len(doc_ids)
        assert indexer.sparse_matrix.shape[0] == len(doc_ids)

    def test_index_docs_with_cache(
        self,
        sample_vocab: list[str],
        sample_sparse_vectors: list[dict[str, float]],
    ) -> None:
        indexer = SparseIndexer(vocab=sample_vocab, max_cache_size=10)
        doc_ids = ["101", "102", "103"]

        indexer.index_docs(doc_ids, sample_sparse_vectors, use_cache=True)

        # Should be in cache, not merged yet
        assert len(indexer.cache_doc_ids) == 1
        assert len(indexer.cache_sparse_matrix) == 1
        assert indexer.cache_doc_ids[0] == doc_ids

    def test_index_docs_cache_overflow(self, sample_vocab: list[str]) -> None:
        indexer = SparseIndexer(vocab=sample_vocab, max_cache_size=1)

        # First batch - should go to cache
        indexer.index_docs(["101"], [{"hello": 0.5}], use_cache=True)
        assert len(indexer.cache_doc_ids) == 1

        # Second batch - should trigger merge due to cache size limit
        indexer.index_docs(["102"], [{"world": 0.3}], use_cache=True)
        assert len(indexer.doc_id_list) == 2  # Should be merged
        assert indexer.is_cache_empty() or len(indexer.cache_doc_ids) == 1

    def test_finalize_indexing(self, sample_vocab: list[str]) -> None:
        indexer = SparseIndexer(vocab=sample_vocab)

        # Add something to cache
        indexer.index_docs(["101"], [{"hello": 0.5}], use_cache=True)
        assert not indexer.is_cache_empty()

        indexer.finalize_indexing()

        assert indexer.is_cache_empty()
        assert len(indexer.doc_id_list) == 1

    def test_is_cache_empty(self, sample_vocab: list[str]) -> None:
        indexer = SparseIndexer(vocab=sample_vocab)

        assert indexer.is_cache_empty()

        indexer.index_docs(["101"], [{"hello": 0.5}], use_cache=True)
        assert not indexer.is_cache_empty()

    @pytest.mark.parametrize(
        "doc_ids, expected_indices",
        [
            (["101", "102"], [0, 1]),
            (["102"], [1]),
            (None, slice(None)),
        ],
    )
    def test_get_sparse_matrix(
        self,
        sample_vocab: list[str],
        sample_sparse_vectors: list[dict[str, float]],
        doc_ids: Any,
        expected_indices: Any,
    ) -> None:
        indexer = SparseIndexer(vocab=sample_vocab)
        indexer.index_docs(["101", "102", "103"], sample_sparse_vectors, use_cache=False)

        result = indexer.get_sparse_matrix(doc_ids)

        assert isinstance(result, sps.csr_matrix)
        if doc_ids is None:
            assert result.shape[0] == len(sample_sparse_vectors)
        else:
            assert result.shape[0] == len(doc_ids)

    def test_get_sparse_vectors(
        self,
        sample_vocab: list[str],
        sample_sparse_vectors: list[dict[str, float]],
    ) -> None:
        indexer = SparseIndexer(vocab=sample_vocab)
        doc_ids = ["101", "102", "103"]
        indexer.index_docs(doc_ids, sample_sparse_vectors, use_cache=False)

        result = indexer.get_sparse_vectors(["101", "102"])

        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert isinstance(result[1], dict)

    def test___len__(
        self,
        sample_vocab: list[str],
        sample_sparse_vectors: list[dict[str, float]],
    ) -> None:
        indexer = SparseIndexer(vocab=sample_vocab)

        assert len(indexer) == 0

        indexer.index_docs(["101", "102", "103"], sample_sparse_vectors, use_cache=False)
        assert len(indexer) == 3

    def test_stats(self, sample_vocab: list[str]) -> None:
        indexer = SparseIndexer(vocab=sample_vocab)
        sparse_vectors = [
            {"hello": 0.5, "world": 0.3},
            {"hello": 0.2, "##sub": 0.4},
        ]
        indexer.index_docs(["101", "102"], sparse_vectors, use_cache=False)

        stats = indexer.stats()

        assert isinstance(stats, dict)
        assert "avg_nnz" in stats
        assert "avg_sparsity" in stats
        assert "num_unique_tokens" in stats
        assert "num_unique_subtokens" in stats
        assert "top_popular_tokens" in stats
        assert "most_popular_appear_in_docs" in stats
        assert stats["num_unique_tokens"] == 3
        assert stats["num_unique_subtokens"] == 1  # "##sub"
        assert stats["top_popular_tokens"][0] == ("hello", 1.0)
        assert stats["most_popular_appear_in_docs"] == 1.0
        assert stats["avg_nnz"] == 2
        assert stats["avg_sparsity"] == 0.5

    @patch("gzip.open", new_callable=mock_open)
    @patch("pickle.dump")
    def test_save(
        self,
        mock_pickle_dump: Any,
        mock_gzip_open: Any,
        sample_vocab: list[str],
    ) -> None:
        indexer = SparseIndexer(vocab=sample_vocab)
        file_path = "test.pkl.gz"

        indexer.save(file_path)

        mock_gzip_open.assert_called_once_with(file_path, "wb")
        mock_pickle_dump.assert_called_once()

        # Check the data dict that was saved
        saved_data = mock_pickle_dump.call_args[0][0]
        assert "vocab" in saved_data
        assert "term2index" in saved_data
        assert saved_data["vocab"] == sample_vocab

    @patch("gzip.open", new_callable=mock_open)
    @patch("pickle.load")
    def test_load(
        self,
        mock_pickle_load: Any,
        mock_gzip_open: Any,
        sample_vocab: list[str],
    ) -> None:
        file_path = "test.pkl.gz"
        mock_data = {
            "vocab": sample_vocab,
            "term2index": {term: idx for idx, term in enumerate(sample_vocab)},
            "docid2index": {},
            "doc_id_list": [],
            "sparse_matrix": sps.csr_matrix((0, len(sample_vocab))),
            "dtype": np.float32,
            "max_cache_size": 1000,
        }
        mock_pickle_load.return_value = mock_data

        indexer = SparseIndexer.load(file_path)

        mock_gzip_open.assert_called_once_with(file_path, "rb")
        mock_pickle_load.assert_called_once()
        assert indexer.vocab == sample_vocab
