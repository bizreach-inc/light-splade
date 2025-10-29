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

from typing import Any
from typing import Callable
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from transformers.models.bert import BertTokenizer
from transformers.models.bert import BertTokenizerFast

from light_splade.models.splade import Splade
from light_splade.models.splade import SpladeEncoder
from light_splade.models.splade import is_same_tokenizers


def init_tokenizer(class_name: str, vocab_file: str) -> Any:
    vocab_path = f"src/tests/data/{vocab_file}"
    if class_name == "BertTokenizer":
        tokenizer = BertTokenizer(vocab_file=vocab_path)
        return tokenizer
    elif class_name == "BertTokenizerFast":
        tokenizer_fast = BertTokenizerFast(vocab_file=vocab_path)
        return tokenizer_fast
    else:
        raise ValueError(f"Unsupported tokenizer class: {class_name}")


@pytest.mark.parametrize(
    "tokenizer1_type, tokenizer2_type, vocab_file1, vocab_file2, expected",
    [
        ("BertTokenizer", "BertTokenizer", "vocab1.txt", "vocab1.txt", True),
        (
            "BertTokenizer",
            "BertTokenizerFast",
            "vocab1.txt",
            "vocab1.txt",
            False,
        ),
        ("BertTokenizer", "BertTokenizer", "vocab1.txt", "vocab2.txt", False),
    ],
)
def test_is_same_tokenizers(
    tokenizer1_type: str,
    tokenizer2_type: str,
    vocab_file1: str,
    vocab_file2: str,
    expected: bool,
) -> None:
    tokenizer1 = init_tokenizer(tokenizer1_type, vocab_file1)
    tokenizer2 = init_tokenizer(tokenizer2_type, vocab_file2)

    result = is_same_tokenizers(tokenizer1, tokenizer2)
    assert result == expected


class TestSpladeEncoder:
    @pytest.fixture
    def mock_model_and_tokenizer(self) -> tuple[Mock, Mock]:
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.get_vocab.return_value = {
            "hello": 1,
            "world": 2,
            "splade": 4,
            "[PAD]": 0,
            "[CLS]": 101,
            "[SEP]": 102,
        }
        return mock_model, mock_tokenizer

    @pytest.mark.parametrize(
        "agg, expected_func",
        [
            ("max", torch.max),
            ("sum", torch.sum),
        ],
    )
    @patch("light_splade.models.splade.AutoModelForMaskedLM")
    @patch("light_splade.models.splade.AutoTokenizer")
    def test___init__(
        self,
        mock_auto_tokenizer: Mock,
        mock_auto_model: Mock,
        agg: str,
        expected_func: Callable,
        mock_model_and_tokenizer: tuple[Mock, Mock],
    ) -> None:
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        encoder = SpladeEncoder("test-model-path", agg=agg)

        assert encoder.transformer == mock_model
        assert encoder.tokenizer == mock_tokenizer
        assert encoder.agg_func == expected_func

    @patch("light_splade.models.splade.AutoModelForMaskedLM")
    @patch("light_splade.models.splade.AutoTokenizer")
    def test___init___invalid_agg(self, mock_auto_tokenizer: Mock, mock_auto_model: Mock) -> None:
        with pytest.raises(AssertionError):
            SpladeEncoder("test-model-path", agg="invalid")

    @patch("light_splade.models.splade.AutoModelForMaskedLM")
    @patch("light_splade.models.splade.AutoTokenizer")
    def test_from_pretrained(
        self, mock_auto_tokenizer: Mock, mock_auto_model: Mock, mock_model_and_tokenizer: tuple[Mock, Mock]
    ) -> None:
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        model_path = "test-model-path"
        encoder = SpladeEncoder(model_path)
        mock_auto_model.from_pretrained.assert_called_with(model_path, trust_remote_code=True)
        mock_auto_tokenizer.from_pretrained.assert_called_with(model_path, trust_remote_code=True)

        assert encoder.idx2token == {1: "hello", 2: "world", 4: "splade", 0: "[PAD]", 101: "[CLS]", 102: "[SEP]"}

    @patch("light_splade.models.splade.AutoModelForMaskedLM")
    @patch("light_splade.models.splade.AutoTokenizer")
    def test_forward(
        self, mock_auto_tokenizer: Mock, mock_auto_model: Mock, mock_model_and_tokenizer: tuple[Mock, Mock]
    ) -> None:
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Mock transformer output
        mock_output = Mock()
        mock_logits = torch.tensor([[[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]])
        mock_output.logits = mock_logits
        mock_model.return_value = mock_output

        encoder = SpladeEncoder("test-model-path", agg="max")

        input_ids = torch.tensor([[1, 2]])
        attention_mask = torch.tensor([[1, 1]])

        result = encoder.forward(input_ids, attention_mask)

        assert result.shape == (1, 3)  # batch_size=1, vocab_size=3

    @patch("light_splade.models.splade.AutoModelForMaskedLM")
    @patch("light_splade.models.splade.AutoTokenizer")
    def test_to_sparse(
        self,
        mock_auto_tokenizer: Mock,
        mock_auto_model: Mock,
        mock_model_and_tokenizer: tuple[Mock, Mock],
    ) -> None:
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        encoder = SpladeEncoder("test-model-path")
        encoder.tokenizer = mock_tokenizer

        # Create a dense vector with some non-zero values
        dense = torch.tensor([0.0, 2.5, 1.8, 0.0, 3.1])

        result = encoder.to_sparse(dense)[0]

        assert isinstance(result, dict)
        assert len(result) == 3  # Only non-zero values
        # Values should be sorted by weight in descending order
        values = list(result.values())
        assert values == sorted(values, reverse=True)

    @patch("light_splade.models.splade.AutoModelForMaskedLM")
    @patch("light_splade.models.splade.AutoTokenizer")
    def test_get_sparse(self, mock_auto_tokenizer: Mock, mock_auto_model: Mock) -> None:
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.get_vocab.return_value = {"hello": 1, "world": 2}
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        encoder = SpladeEncoder("test-model-path")

        # Mock forward and to_sparse methods
        encoder.forward = Mock(return_value=torch.tensor([[1.0, 2.0], [0.5, 1.5]]))
        encoder.to_sparse = Mock(side_effect=[[{"hello": 2.0}, {"world": 1.5}]])

        input_ids = torch.tensor([[1, 2], [1, 2]])
        attention_mask = torch.tensor([[1, 1], [1, 1]])

        result = encoder.get_sparse(input_ids, attention_mask)

        assert len(result) == 2
        assert result[0] == {"hello": 2.0}
        assert result[1] == {"world": 1.5}


class TestSplade:
    @pytest.fixture
    def mock_encoder(self) -> Mock:
        mock_encoder = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.vocab = {"hello": 1, "world": 2}
        mock_encoder.tokenizer = mock_tokenizer
        return mock_encoder

    @pytest.mark.parametrize(
        "d_model_path, q_model_path, freeze_d_model, agg, should_raise",
        [
            ("doc_model", None, False, "max", False),
            ("doc_model", "query_model", False, "max", False),
            ("doc_model", "query_model", True, "max", False),
            ("doc_model", None, True, "max", True),  # Should raise error
            ("doc_model", None, False, "invalid", True),  # Should raise error
        ],
    )
    @patch("light_splade.models.splade.SpladeEncoder")
    @patch("light_splade.models.splade.is_same_tokenizers")
    def test___init__(
        self,
        mock_is_same_tokenizers: Mock,
        mock_encoder_class: Mock,
        mock_encoder: Mock,
        d_model_path: str,
        q_model_path: str | None,
        freeze_d_model: bool,
        agg: str,
        should_raise: bool,
    ) -> None:
        mock_encoder_class.return_value = mock_encoder
        mock_is_same_tokenizers.return_value = True

        if should_raise:
            with pytest.raises((ValueError, AssertionError)):
                Splade(
                    d_model_path=d_model_path,
                    q_model_path=q_model_path,
                    freeze_d_model=freeze_d_model,
                    agg=agg,
                )
        else:
            splade = Splade(
                d_model_path=d_model_path,
                q_model_path=q_model_path,
                freeze_d_model=freeze_d_model,
                agg=agg,
            )

            assert splade.d_model_path == d_model_path
            assert splade.q_model_path == q_model_path
            assert splade.freeze_d_model == freeze_d_model
            assert splade.agg == agg
            assert splade.is_shared_weights == (q_model_path is None)

    @patch("light_splade.models.splade.SpladeEncoder")
    @patch("light_splade.models.splade.is_same_tokenizers")
    def test___init___different_tokenizers_raises_error(
        self,
        mock_is_same_tokenizers: Mock,
        mock_encoder_class: Mock,
        mock_encoder: Mock,
    ) -> None:
        mock_encoder_class.return_value = mock_encoder
        mock_is_same_tokenizers.return_value = False

        with pytest.raises(ValueError) as exc_info:
            Splade(d_model_path="doc_model", q_model_path="query_model")

        assert "same vocab size" in str(exc_info.value)

    @patch("light_splade.models.splade.SpladeEncoder")
    @patch("light_splade.models.splade.is_same_tokenizers")
    def test_num_trainable_params(self, mock_is_same_tokenizers: Mock, mock_encoder_class: Mock) -> None:
        mock_encoder = Mock()
        mock_encoder.tokenizer.vocab = {"hello": 1}
        mock_encoder_class.return_value = mock_encoder
        mock_is_same_tokenizers.return_value = True

        splade = Splade(d_model_path="test_model")

        # Mock parameters
        param1 = torch.tensor([1.0, 2.0])
        param1.requires_grad = True
        param2 = torch.tensor([3.0, 4.0, 5.0])
        param2.requires_grad = False
        param3 = torch.tensor([6.0])
        param3.requires_grad = True

        splade.parameters = Mock(return_value=[param1, param2, param3])

        result = splade.num_trainable_params
        assert result == 3  # Only param1 (2 elements) + param3 (1 element)

    @patch("light_splade.models.splade.SpladeEncoder")
    @patch("light_splade.models.splade.is_same_tokenizers")
    def test_num_total_params(self, mock_is_same_tokenizers: Mock, mock_encoder_class: Mock) -> None:
        mock_encoder = Mock()
        mock_encoder.tokenizer.vocab = {"hello": 1}
        mock_encoder_class.return_value = mock_encoder
        mock_is_same_tokenizers.return_value = True

        splade = Splade(d_model_path="test_model")

        # Mock parameters
        param1 = torch.tensor([1.0, 2.0])
        param2 = torch.tensor([3.0, 4.0, 5.0])

        splade.parameters = Mock(return_value=[param1, param2])

        result = splade.num_total_params
        assert result == 5  # param1 (2 elements) + param2 (3 elements)

    @patch("light_splade.models.splade.SpladeEncoder")
    @patch("light_splade.models.splade.is_same_tokenizers")
    def test_forward(self, mock_is_same_tokenizers: Mock, mock_encoder_class: Mock) -> None:
        mock_d_encoder = Mock()
        mock_q_encoder = Mock()
        mock_d_encoder.tokenizer.vocab = {"hello": 1}
        mock_q_encoder.tokenizer.vocab = {"hello": 1}

        mock_encoder_class.side_effect = [mock_d_encoder, mock_q_encoder]
        mock_is_same_tokenizers.return_value = True

        mock_d_encoder.return_value = torch.tensor([[1.0, 2.0]])
        mock_q_encoder.return_value = torch.tensor([[3.0, 4.0]])

        splade = Splade(d_model_path="doc_model", q_model_path="query_model")
        splade.d_encoder = mock_d_encoder
        splade.q_encoder = mock_q_encoder

        result = splade.forward(
            q_input_ids=torch.tensor([[1, 2]]),
            q_attention_mask=torch.tensor([[1, 1]]),
            d_input_ids=torch.tensor([[3, 4]]),
            d_attention_mask=torch.tensor([[1, 1]]),
        )

        assert "q_vector" in result
        assert "d_vector" in result
        assert torch.equal(result["q_vector"], torch.tensor([[3.0, 4.0]]))
        assert torch.equal(result["d_vector"], torch.tensor([[1.0, 2.0]]))

    @patch("light_splade.models.splade.SpladeEncoder")
    @patch("light_splade.models.splade.is_same_tokenizers")
    def test_forward_partial_inputs(self, mock_is_same_tokenizers: Mock, mock_encoder_class: Mock) -> None:
        mock_encoder = Mock()
        mock_encoder.tokenizer.vocab = {"hello": 1}
        mock_encoder_class.return_value = mock_encoder
        mock_is_same_tokenizers.return_value = True

        mock_encoder.return_value = torch.tensor([[1.0, 2.0]])

        splade = Splade(d_model_path="test_model")
        splade.d_encoder = mock_encoder
        splade.q_encoder = mock_encoder

        # Test with only document inputs
        result = splade.forward(
            d_input_ids=torch.tensor([[3, 4]]),
            d_attention_mask=torch.tensor([[1, 1]]),
        )

        assert result["q_vector"] is None
        assert result["d_vector"] is not None

    @patch("light_splade.models.splade.SpladeEncoder")
    @patch("light_splade.models.splade.is_same_tokenizers")
    def test_to_sparse(self, mock_is_same_tokenizers: Mock, mock_encoder_class: Mock) -> None:
        mock_d_encoder = Mock()
        mock_q_encoder = Mock()
        mock_d_encoder.tokenizer.vocab = {"hello": 1}
        mock_q_encoder.tokenizer.vocab = {"hello": 1}

        mock_encoder_class.side_effect = [mock_d_encoder, mock_q_encoder]
        mock_is_same_tokenizers.return_value = True

        mock_d_encoder.to_sparse.return_value = {"doc": 1.0}
        mock_q_encoder.to_sparse.return_value = {"query": 2.0}

        splade = Splade(d_model_path="doc_model", q_model_path="query_model")
        splade.d_encoder = mock_d_encoder
        splade.q_encoder = mock_q_encoder

        denses = {
            "q_vector": torch.tensor([[1.0, 2.0]]),
            "d_vector": torch.tensor([[3.0, 4.0]]),
        }

        result = splade.to_sparse(denses)

        assert result["q_vector"] == {"query": 2.0}
        assert result["d_vector"] == {"doc": 1.0}

    @patch("light_splade.models.splade.SpladeEncoder")
    @patch("light_splade.models.splade.is_same_tokenizers")
    @patch("os.path.join")
    def test_load_shared_weights(
        self,
        mock_join: Mock,
        mock_is_same_tokenizers: Mock,
        mock_encoder_class: Mock,
    ) -> None:
        mock_encoder = Mock()
        mock_encoder.tokenizer.vocab = {"hello": 1}
        mock_encoder_class.return_value = mock_encoder
        mock_is_same_tokenizers.return_value = True

        splade = Splade(d_model_path="test_model")
        splade.is_shared_weights = True
        splade.freeze_d_model = False

        splade.load("model_path")

        mock_encoder.from_pretrained.assert_called_once_with("model_path")

    @patch("light_splade.models.splade.SpladeEncoder")
    @patch("light_splade.models.splade.is_same_tokenizers")
    @patch("os.path.join")
    def test_load_separate_weights(
        self,
        mock_join: Mock,
        mock_is_same_tokenizers: Mock,
        mock_encoder_class: Mock,
    ) -> None:
        mock_d_encoder = Mock()
        mock_q_encoder = Mock()
        mock_d_encoder.tokenizer.vocab = {"hello": 1}
        mock_q_encoder.tokenizer.vocab = {"hello": 1}

        mock_encoder_class.side_effect = [mock_d_encoder, mock_q_encoder]
        mock_is_same_tokenizers.return_value = True
        mock_join.return_value = "model_path/query_model"

        splade = Splade(d_model_path="doc_model", q_model_path="query_model")
        splade.d_encoder = mock_d_encoder
        splade.q_encoder = mock_q_encoder
        splade.is_shared_weights = False
        splade.freeze_d_model = False

        splade.load("model_path")

        mock_d_encoder.from_pretrained.assert_called_with("model_path")
        mock_q_encoder.from_pretrained.assert_called_with("model_path/query_model")

    @patch("light_splade.models.splade.SpladeEncoder")
    @patch("light_splade.models.splade.is_same_tokenizers")
    @patch("os.makedirs")
    @patch("os.path.join")
    @patch("light_splade.models.splade.contiguous")
    def test_save_shared_weights(
        self,
        mock_contiguous: Mock,
        mock_join: Mock,
        mock_makedirs: Mock,
        mock_is_same_tokenizers: Mock,
        mock_encoder_class: Mock,
    ) -> None:
        mock_encoder = Mock()
        mock_encoder.tokenizer.vocab = {"hello": 1}
        mock_encoder_class.return_value = mock_encoder
        mock_is_same_tokenizers.return_value = True

        splade = Splade(d_model_path="test_model")
        splade.is_shared_weights = True

        splade.save("output_dir", save_safetensors=True)

        mock_contiguous.assert_called_once_with(mock_encoder)
        mock_encoder.transformer.save_pretrained.assert_called_once_with("output_dir", safe_serialization=True)
        mock_encoder.tokenizer.save_pretrained.assert_called_once_with("output_dir")

    @patch("light_splade.models.splade.SpladeEncoder")
    @patch("light_splade.models.splade.is_same_tokenizers")
    @patch("os.makedirs")
    @patch("os.path.join")
    @patch("light_splade.models.splade.contiguous")
    def test_save_separate_weights(
        self,
        mock_contiguous: Mock,
        mock_join: Mock,
        mock_makedirs: Mock,
        mock_is_same_tokenizers: Mock,
        mock_encoder_class: Mock,
    ) -> None:
        mock_d_encoder = Mock()
        mock_q_encoder = Mock()
        mock_d_encoder.tokenizer.vocab = {"hello": 1}
        mock_q_encoder.tokenizer.vocab = {"hello": 1}

        mock_encoder_class.side_effect = [mock_d_encoder, mock_q_encoder]
        mock_is_same_tokenizers.return_value = True
        mock_join.return_value = "output_dir/query_model"

        splade = Splade(d_model_path="doc_model", q_model_path="query_model")
        splade.d_encoder = mock_d_encoder
        splade.q_encoder = mock_q_encoder
        splade.is_shared_weights = False

        splade.save("output_dir", save_safetensors=False)

        mock_makedirs.assert_called_once_with("output_dir/query_model", exist_ok=True)
        mock_d_encoder.transformer.save_pretrained.assert_called_with("output_dir", safe_serialization=False)
        mock_q_encoder.transformer.save_pretrained.assert_called_with(
            "output_dir/query_model", safe_serialization=False
        )
