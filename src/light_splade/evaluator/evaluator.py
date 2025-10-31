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

"""Evaluation utilities for SPLADE models.

This module provides the `Evaluator` class which:
    * Parses metric specifications like `ndcg@10` / `map@100`.
    * Builds a sparse index of document vectors (using a caching indexer).
    * Retrieves top-K candidates per query with a sparse retriever.
    * Computes ranking metrics through `MetricsEvaluator`.
    * Records index statistics (density, average nnz, etc.) for monitoring.

The evaluation is intentionally lightweight to allow frequent validation without full IR pipeline complexity.
"""

from logging import getLogger

import torch
from tqdm import tqdm

from light_splade.data import BaseSpladeCollator
from light_splade.data import TripletDataset
from light_splade.data import TripletDistilDataset
from light_splade.models import Splade

from .metrics_evaluator import MetricsEvaluator
from .sparse_indexer import SparseIndexer
from .sparse_retriever import SparseRetriever

logger = getLogger(__name__)


MAX_CACHE_SIZE = 1000


class Evaluator:
    """Run retrieval-style evaluation for a SPLADE model on a dataset.

    The evaluator coordinates document indexing, retrieval and metric computation for a given SPLADE model and
    evaluation dataset. It expects datasets that expose query/document masters and positive lists.
    """

    def __init__(
        self,
        eval_dataset: TripletDataset | TripletDistilDataset,
        data_collator: BaseSpladeCollator,
        model: Splade,
        batch_size: int,
        device: torch.device,
    ) -> None:
        """Initialize the evaluator.

        Args:
            eval_dataset: Dataset providing queries, documents and positives.
            data_collator: Tokenization and batching helper used for encoding.
            model: SPLADE model instance with encoding utilities.
            batch_size (int): Batch size used when encoding documents.
            device (torch.device): Torch device where model tensors should be placed.
        """
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.model = model
        self.batch_size = batch_size
        self.device = device

    def _parse_metric_specs(self, metrics: list[str]) -> tuple[list[str], list[int]]:
        """Parse metric specification strings into names and cutoff values.

        Given a list like ``["ndcg@10", "map@100"]`` this returns two lists: unique metric names and unique k cutoffs
        sorted ascending.

        Args:
            metrics (list[str]): List of metric specification strings.

        Returns:
            A tuple of (metric_names, k_values) where metric_names is a list of unique metric base names and k_values is
            a list of integer cutoffs.
        """
        metric_names: list[str] = []
        k_values: list[int] = []
        for metric in metrics:
            parts = metric.split("@")
            metric_names.append(parts[0])
            k_values.append(int(parts[1]))

        metric_names = sorted(set(metric_names))
        k_values = sorted(set(k_values))

        return metric_names, k_values

    def index_docs(self) -> SparseIndexer:
        """Vectorize and index all documents in the evaluation dataset.

        Documents are encoded in batches using the provided model and
        collator. The produced sparse/dense vectors are added to a
        :class:`SparseIndexer` which is finalized and returned for retrieval.

        Returns:
            A finalized :class:`SparseIndexer` ready for querying.
        """
        logger.info("Indexing docs...")

        doc_ids = self.eval_dataset.docs.get_id_list()
        doc_texts = [self.eval_dataset.docs[doc_id] for doc_id in doc_ids]
        vocab = [token for token, _ in self.data_collator.tokenizer.get_vocab().items()]
        indexer = SparseIndexer(vocab=vocab, max_cache_size=MAX_CACHE_SIZE)

        with torch.inference_mode():
            for start in tqdm(range(0, len(doc_ids), self.batch_size)):
                end = min(start + self.batch_size, len(doc_ids))
                batch_doc_ids = [str(doc_id) for doc_id in doc_ids[start:end]]
                batch_texts = doc_texts[start:end]
                embeddings = self.model.d_encoder.encode(
                    batch_texts,
                    batch_size=self.batch_size,
                    max_text_length=self.data_collator.max_length,
                ).numpy()
                indexer.index_docs(batch_doc_ids, embeddings, use_cache=True)

        indexer.finalize_indexing()
        return indexer

    def evaluate(self, validation_metrics: list[str]) -> dict[str, float]:
        """Compute retrieval metrics and sparse index statistics.

        The method indexes documents, retrieves top-K candidates per query,
        computes ranking metrics via :class:`MetricsEvaluator` and returns a
        mapping from the requested metric specs to numeric scores. Index
        statistics (e.g., average non-zero elements) are also included in the
        returned dictionary.

        Args:
            validation_metrics: List of metric specs like ``["ndcg@10", "map@100"]``.

        Returns:
            A dictionary mapping metric spec strings and index-stat keys to
            their numeric values.
        """
        metric_names, k_values = self._parse_metric_specs(validation_metrics)
        max_k = max(k_values)

        score_threshold = 0.0
        qids = self.eval_dataset.queries.get_id_list()
        q_texts = [self.eval_dataset.queries[qid] for qid in qids]

        # vectorize all docs in validation set,
        # and index the sparse representation
        indexer = self.index_docs()
        target_doc_ids = indexer.doc_id_list

        # loop through each query
        retriever = SparseRetriever(indexer)
        results = dict()
        qrels = dict()
        logger.info("Retrieving for each query...")

        with torch.inference_mode():
            for qid, q_text in tqdm(zip(qids, q_texts)):
                # get the ground-truth
                gt_doc_ids = self.eval_dataset.positive_list[qid]
                qrels[str(qid)] = [str(doc_id) for doc_id in gt_doc_ids]

                # vectorize the query and retrieve the top-K where K as the max from metrics
                tokens = self.data_collator.tokenize([q_text])
                sparse_vec = self.model.q_encoder.get_sparse(
                    input_ids=tokens["input_ids"].to(self.device),
                    attention_mask=tokens["attention_mask"].to(self.device),
                )[0]
                ranked_cand_ids = retriever.retrieve(
                    [sparse_vec], target_doc_ids, top_k=max_k, threshold=score_threshold
                )[0]
                results[str(qid)] = ranked_cand_ids

        # compute the metrics on the retrieved docs
        logger.info("Computing metrics...")
        eval_results = MetricsEvaluator.evaluate(
            qrels=qrels,
            results=results,  # type: ignore
            k_values=k_values,
            metrics=metric_names,
        )

        metrics = {metric: eval_results[metric.lower()] for metric in validation_metrics}

        # get stats on sparse vectors
        stats = indexer.stats()
        logger.info(f"Stats on sparse vectors: {stats}")
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                metrics[k] = v

        return metrics
