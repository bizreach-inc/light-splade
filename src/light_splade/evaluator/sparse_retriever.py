# Copyright 2025 BizReach, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to retrieve documents from a sparse index.

This module exposes :class:`SparseRetriever`, a thin wrapper around :class:`SparseIndexer` that computes dot-product
similarities between query and document sparse vectors and returns ranked document ids (optionally with scores).
"""

from typing import Any

import numpy as np

from light_splade.schemas.types import ID_LIST
from light_splade.schemas.types import ID_WITH_SCORE_LIST
from light_splade.schemas.types import SPARSE_VECTOR_LIST

from .sparse_indexer import SparseIndexer


class SparseRetriever:
    """Retrieve top documents for sparse query vectors.

    The retriever computes similarity scores via dot-product between the supplied query sparse vectors and the document
    CSR matrix stored in the :class:`SparseIndexer` and returns ranked document ids per query.
    """

    def __init__(self, sparse_indexer: SparseIndexer) -> None:
        """Create a retriever backed by ``sparse_indexer``."""
        self.sparse_indexer = sparse_indexer

    def retrieve(
        self,
        query_vectors: SPARSE_VECTOR_LIST,
        target_doc_ids: ID_LIST | None = None,
        top_k: int = 0,
        threshold: float = 0.0,
        return_score: bool = False,
    ) -> list[ID_LIST] | list[ID_WITH_SCORE_LIST]:
        """Retrieve relevant doc ids for each query vector.

        Args:
            query_vectors (SPARSE_VECTOR_LIST): List of sparse query vectors (each a mapping term->score). If multiple
                vectors are provided, ``target_doc_ids`` must be None (search across all documents).
            target_doc_ids (ID_LIST | None): Optional list of target doc ids to restrict search to (supported only when
                a single query vector is provided).
            top_k (int): Number of ids to return per query (0 to return all).
            threshold (float): Minimum score threshold; documents below it are filtered.
            return_score (bool): If True return (doc_id, score) tuples instead of ids.

        Returns:
            A list (one entry per query) containing ordered doc ids or (doc_id, score) tuples depending on
            ``return_score``.
        """

        # Validation
        assert len(query_vectors) == 1 or target_doc_ids is None, (
            "Multiple query vectors are supported only when searching on all "
            "documents, i.e., target_doc_ids must be None"
        )

        # encode query sparse vectors into csr_matrix
        query_sparse_matrix = self.sparse_indexer.index_vectors(query_vectors)

        # Get doc sparse matrix in form of csr_matrix
        doc_sparse_matrix = self.sparse_indexer.get_sparse_matrix(target_doc_ids)

        # Compute similarity scores
        scores = query_sparse_matrix.dot(doc_sparse_matrix.T).toarray()

        # rank the list up to top_k
        if top_k == 0:
            top_k = scores.shape[1]

        sorted_indices = np.argsort(scores, axis=-1)[:, ::-1]

        # produce results
        ranked_ids: list[list[Any]] = []
        target_doc_ids_ = target_doc_ids if target_doc_ids else self.sparse_indexer.doc_id_list
        for indices, scores_ in zip(sorted_indices, scores):
            keep = scores_[indices] >= threshold
            topk_indices = indices[keep][:top_k]
            if return_score:
                ids = [(target_doc_ids_[index], score) for index, score in zip(topk_indices, scores_[topk_indices])]
            else:
                ids = [target_doc_ids_[index] for index in topk_indices]
            ranked_ids.append(ids)

        return ranked_ids
