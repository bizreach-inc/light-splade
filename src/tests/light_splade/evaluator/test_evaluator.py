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

from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from light_splade.evaluator.evaluator import Evaluator


class TestEvaluator:
    @pytest.fixture
    def mock_dependencies(self) -> tuple[Mock, Mock, Mock]:
        mock_queries = MagicMock()
        mock_queries.get_id_list.return_value = [1, 2]
        mock_queries.__getitem__ = lambda self, key: {
            1: "query text 1",
            2: "query text 2",
        }[key]

        mock_docs = MagicMock()
        mock_docs.get_id_list.return_value = [101, 102, 103]
        mock_docs.__getitem__ = lambda self, key: {
            101: "doc text 1",
            102: "doc text 2",
            103: "doc text 3",
        }[key]

        # Mock eval_dataset
        eval_dataset = Mock()
        eval_dataset.docs = mock_docs
        eval_dataset.queries = mock_queries
        eval_dataset.positive_list = {1: [101, 102], 2: [102, 103]}

        # Mock data_collator
        data_collator = Mock()
        data_collator.tokenizer.get_vocab.return_value = {
            "hello": 1,
            "world": 2,
        }
        data_collator.tokenize.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        # Mock model
        model = Mock()
        model.d_encoder.return_value = torch.tensor([[0.1, 0.2, 0.3]])
        model.q_encoder.get_sparse.return_value = [{"hello": 0.5, "world": 0.3}]

        return eval_dataset, data_collator, model

    def test___init__(self, mock_dependencies: tuple[Mock, Mock, Mock]) -> None:
        eval_dataset, data_collator, model = mock_dependencies
        batch_size = 32
        device = torch.device("cpu")

        evaluator = Evaluator(
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            model=model,
            batch_size=batch_size,
            device=device,
        )

        assert evaluator.eval_dataset == eval_dataset
        assert evaluator.data_collator == data_collator
        assert evaluator.model == model
        assert evaluator.batch_size == batch_size
        assert evaluator.device == device

    @pytest.mark.parametrize(
        "metrics, expected_names, expected_k_values",
        [
            (["ndcg@10", "map@5"], ["map", "ndcg"], [5, 10]),
            (
                ["recall@1", "precision@10", "ndcg@5"],
                ["ndcg", "precision", "recall"],
                [1, 5, 10],
            ),
            (["map@10"], ["map"], [10]),
        ],
    )
    def test__parse_metric_specs(
        self,
        mock_dependencies: tuple[Mock, Mock, Mock],
        metrics: list[str],
        expected_names: list[str],
        expected_k_values: list[int],
    ) -> None:
        eval_dataset, data_collator, model = mock_dependencies

        evaluator = Evaluator(
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            model=model,
            batch_size=32,
            device=torch.device("cpu"),
        )

        metric_names, k_values = evaluator._parse_metric_specs(metrics)

        assert set(metric_names) == set(expected_names)
        assert set(k_values) == set(expected_k_values)

    @patch("light_splade.evaluator.evaluator.SparseIndexer")
    @patch("light_splade.evaluator.evaluator.tqdm")
    def test_index_docs(
        self,
        mock_tqdm: Mock,
        mock_sparse_indexer_class: Mock,
        mock_dependencies: tuple[Mock, Mock, Mock],
    ) -> None:
        eval_dataset, data_collator, model = mock_dependencies

        # Mock tqdm to return range directly
        mock_tqdm.side_effect = lambda x: x

        # Mock SparseIndexer
        mock_indexer = Mock()
        mock_sparse_indexer_class.return_value = mock_indexer

        evaluator = Evaluator(
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            model=model,
            batch_size=2,
            device=torch.device("cpu"),
        )

        result = evaluator.index_docs()

        assert result == mock_indexer
        mock_sparse_indexer_class.assert_called_once_with(vocab=["hello", "world"], max_cache_size=1000)
        mock_indexer.index_docs.assert_called()
        mock_indexer.finalize_indexing.assert_called_once()

    @patch("light_splade.evaluator.evaluator.SparseRetriever")
    @patch("light_splade.evaluator.evaluator.MetricsEvaluator")
    @patch("light_splade.evaluator.evaluator.tqdm")
    def test_evaluate(
        self,
        mock_tqdm: Mock,
        mock_metrics_evaluator_class: Mock,
        mock_sparse_retriever_class: Mock,
        mock_dependencies: tuple[Mock, Mock, Mock],
    ) -> None:
        eval_dataset, data_collator, model = mock_dependencies

        # Mock tqdm to return the input directly
        mock_tqdm.side_effect = lambda x: x

        # Mock MetricsEvaluator
        mock_metrics_evaluator_class.evaluate.return_value = {
            "ndcg@10": 0.85,
            "map@5": 0.75,
        }

        # Mock SparseRetriever
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [[101, 102]]
        mock_sparse_retriever_class.return_value = mock_retriever

        evaluator = Evaluator(
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            model=model,
            batch_size=32,
            device=torch.device("cpu"),
        )

        # Mock index_docs method
        mock_indexer = Mock()
        mock_indexer.doc_id_list = [101, 102, 103]
        mock_indexer.stats.return_value = {"avg_nnz": 5, "avg_sparsity": 0.9}
        evaluator.index_docs = Mock(return_value=mock_indexer)

        validation_metrics = ["ndcg@10", "map@5"]
        result = evaluator.evaluate(validation_metrics)

        assert "ndcg@10" in result
        assert "map@5" in result
        assert "avg_nnz" in result
        assert "avg_sparsity" in result
        assert result["ndcg@10"] == 0.85
        assert result["map@5"] == 0.75
        assert result["avg_nnz"] == 5
        assert result["avg_sparsity"] == 0.9
