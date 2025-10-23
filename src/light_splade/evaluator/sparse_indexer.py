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

import gzip
import pickle
from collections import Counter
from logging import getLogger
from typing import Any

import numpy as np
from scipy import sparse as sps

from light_splade.schemas.types import ID_LIST
from light_splade.schemas.types import SPARSE_VECTOR_LIST

logger = getLogger(__name__)


MAX_DOC_ON_STATS = 20000


class SparseIndexer:
    """Index and manage sparse vector representations for documents.

    This class stores sparse vectors in a CSR matrix and maintains mappings between external document ids and internal
    matrix row indices. It supports batched indexing with an optional cache to avoid frequent vstack operations and
    provides utilities to extract vectors in both CSR and dict formats.
    """

    def __init__(
        self,
        vocab: list[str],
        term2index: dict[str, int] | None = None,
        docid2index: dict[str, int] | None = None,
        doc_id_list: ID_LIST | None = None,
        sparse_matrix: sps.csr_matrix | None = None,
        dtype: type = np.float32,
        max_cache_size: int = 1000,
    ) -> None:
        """Create a SparseIndexer.

        Args:
            vocab (list[str]): List of vocabulary tokens (term -> index derived from order).
            term2index (dict[str, int] | None): Optional explicit mapping from term to index.
            docid2index (dict[str, int] | None): Optional mapping from external doc id to internal row.
            doc_id_list (ID_LIST | None): Optional list of document ids corresponding to rows.
            sparse_matrix (sps.csr_matrix | None): Optional prebuilt CSR matrix to initialize from.
            dtype (type): Numpy dtype for stored values.
            max_cache_size (int): Number of batched index operations to cache before merging into the main matrix.
        """
        self.dtype = dtype
        self.max_cache_size = max_cache_size
        self.docid2index = dict()
        self.doc_id_list = []
        self.sparse_matrix = sps.csr_matrix((0, len(vocab)), dtype=self.dtype)

        if term2index is not None:
            self.term2index = term2index

        self.vocab = vocab
        if term2index is None:
            self._index_vocab()
        else:
            self.term2index = term2index

        if docid2index is not None:
            self.docid2index = docid2index

        if doc_id_list is not None:
            self.doc_id_list = doc_id_list

        if sparse_matrix is not None:
            self.sparse_matrix = sparse_matrix

        self._init_cache()

    def _index_vocab(self) -> None:
        # Compute self.term2index from self.vocab
        self.term2index = {term: index for index, term in enumerate(self.vocab)}

    def index_vectors(self, vectors: SPARSE_VECTOR_LIST | np.ndarray) -> sps.csr_matrix:
        """Encode sparse vectors to csr_matrix.
        This function can be used to index both documents and queries.
        """
        data = []
        row_ind = []
        col_ind = []

        if not isinstance(vectors, list):  # dense vectors in form of ndarray
            return sps.csr_matrix(vectors, dtype=self.dtype)

        for i, sparse_vector in enumerate(vectors):
            for term, score in sparse_vector.items():
                col_ind.append(self.term2index[term])
                row_ind.append(i)
                data.append(score)
        return sps.csr_matrix(
            (data, (row_ind, col_ind)),
            shape=(len(vectors), len(self.vocab)),
            dtype=self.dtype,
        )

    def index_docs(
        self,
        doc_ids: ID_LIST,
        vectors: SPARSE_VECTOR_LIST | np.ndarray,
        use_cache: bool = False,
    ) -> None:
        """Receive sparse vectors and encode them in csr_matrix for fast similarity score computation. Currently,
        only dot product is supported.
        NOTE: For existing doc_ids, the new sparse vectors will be used with new indices, the old vectors will become
        zombie and will never be used. So, doc_ids should be unique over multiple calls.

        Args:
            doc_ids (ID_LIST): List of document IDs
            vectors (SPARSE_VECTOR_LIST | np.ndarray): List of documents' sparse vectors
            use_cache (bool): If True, the indexer will temporarily store the new sparse matrix and doc_ids into a
                cache. The accumulated cache is merged into ``sparse_matrix`` and ``doc_id_list`` later via
                :meth:`_merge_cache` to avoid calling ``sps.vstack`` on every batch which becomes slower as the main
                matrix grows.
        """
        assert len(doc_ids) == len(vectors)

        if not use_cache and not self.is_cache_empty():
            self._merge_cache()

        # encode the sparse vectors to csr_matrix
        new_sparse_matrix = self.index_vectors(vectors)
        if not use_cache:
            self._merge_sparse_matrix(doc_ids, new_sparse_matrix)
        else:
            self._save_cache(doc_ids, new_sparse_matrix)
            if self._cache_size() > self.max_cache_size:
                self._merge_cache()

    def finalize_indexing(self) -> None:
        if not self.is_cache_empty():
            self._merge_cache()

    def _init_cache(self) -> None:
        self.cache_doc_ids: list[ID_LIST] = []
        self.cache_sparse_matrix: list[sps.csr_matrix] = []

    def _cache_size(self) -> int:
        return len(self.cache_doc_ids)

    def is_cache_empty(self) -> bool:
        return self._cache_size() == 0

    def _save_cache(self, doc_ids: ID_LIST, sparse_matrix: sps.csr_matrix) -> None:
        self.cache_doc_ids.append(doc_ids)
        self.cache_sparse_matrix.append(sparse_matrix)

    def _merge_cache(self) -> None:
        """
        The `merge_cache` function may be slow (it may take several secs), but it should be called much less frequent
        comparing to the `index_docs` function.
        """

        # merge cache
        doc_ids = np.hstack(self.cache_doc_ids).tolist()
        new_sparse_matrix = sps.vstack(self.cache_sparse_matrix)
        self._merge_sparse_matrix(doc_ids, new_sparse_matrix)

        # clear cache
        self._init_cache()

    def _merge_sparse_matrix(self, new_doc_ids: ID_LIST, new_sparse_matrix: sps.csr_matrix) -> None:
        self.sparse_matrix = sps.vstack([self.sparse_matrix, new_sparse_matrix])

        start = len(self.docid2index)
        for i, doc_id in enumerate(new_doc_ids):
            self.docid2index[doc_id] = start + i
        self.doc_id_list.extend(new_doc_ids)

    def get_sparse_matrix(self, doc_ids: ID_LIST | None = None) -> sps.csr_matrix:
        if doc_ids is None:
            return self.sparse_matrix
        else:
            indices = [self.docid2index[doc_id] for doc_id in doc_ids]
            return self.sparse_matrix[indices]

    def get_sparse_vectors(self, doc_ids: ID_LIST) -> SPARSE_VECTOR_LIST:
        csr_enc = self.get_sparse_matrix(doc_ids)
        indptr = csr_enc.indptr
        indices = csr_enc.indices
        data = csr_enc.data

        vecs = []
        for i in range(len(doc_ids)):
            vec = dict()
            for index, score in zip(
                indices[indptr[i] : indptr[i + 1]],  # noqa
                data[indptr[i] : indptr[i + 1]],  # noqa
            ):
                term = self.vocab[index]
                vec[term] = score
            vecs.append(vec)
        return vecs

    def __len__(self) -> int:
        return int(self.sparse_matrix.shape[0])

    def stats(self) -> dict[str, Any]:
        info: dict[str, Any] = dict()

        # limit num of docs to perform statistics to limit memory usage & computation time
        sparse_matrix = self.sparse_matrix
        if self.sparse_matrix.shape[0] > MAX_DOC_ON_STATS:
            indices = np.random.choice(self.sparse_matrix.shape[0], MAX_DOC_ON_STATS, replace=False)
            sparse_matrix = sparse_matrix[indices]
            logger.info("Performing statistics on %d randomly sampled validation documents.", MAX_DOC_ON_STATS)
        else:
            logger.info("Performing statistics on all %d validation documents.", MAX_DOC_ON_STATS)

        avg_nnz = sparse_matrix.nnz / sparse_matrix.shape[0]
        nnz_y, nnz_x = sparse_matrix.todense().nonzero()
        counter = Counter(nnz_x)

        info["avg_nnz"] = int(avg_nnz)
        info["avg_sparsity"] = 1 - avg_nnz / len(self.vocab)
        info["num_unique_tokens"] = len(counter)
        info["num_unique_subtokens"] = sum([1 for idx in counter.keys() if self.vocab[idx].startswith("##")])
        info["top_popular_tokens"] = [
            (self.vocab[tup[0]], round(tup[1] / sparse_matrix.shape[0], 2)) for tup in counter.most_common(20)
        ]
        info["most_popular_appear_in_docs"] = round(counter.most_common(1)[0][1] / sparse_matrix.shape[0], 2)

        return info

    def save(self, file_path: str) -> None:
        data_dict = dict(
            vocab=self.vocab,
            term2index=self.term2index,
            docid2index=self.docid2index,
            doc_id_list=self.doc_id_list,
            sparse_matrix=self.sparse_matrix,
            dtype=self.dtype,
            max_cache_size=self.max_cache_size,
        )
        with gzip.open(file_path, "wb") as f:
            pickle.dump(data_dict, f)

    @classmethod
    def load(cls, file_path: str) -> "SparseIndexer":
        with gzip.open(file_path, "rb") as f:
            data_dict = pickle.load(f)
        return cls(**data_dict)
