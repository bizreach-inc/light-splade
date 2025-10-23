from unittest.mock import Mock
from unittest.mock import patch

import pytest

from light_splade.data.base import BaseSpladeCollator


class TestBaseSpladeCollator:
    @patch("light_splade.data.base.AutoTokenizer")
    def test___init__(self, mock_auto_tokenizer: Mock) -> None:
        mock_tokenizer = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        collator = BaseSpladeCollator(tokenizer_path="test-tokenizer", max_length=256)

        assert collator.tokenizer == mock_tokenizer
        assert collator.max_length == 256
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("test-tokenizer", trust_remote_code=True)

    @patch("light_splade.data.base.AutoTokenizer")
    def test___init___default_max_length(self, mock_auto_tokenizer: Mock) -> None:
        mock_tokenizer = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        collator = BaseSpladeCollator(tokenizer_path="test-tokenizer")

        assert collator.max_length == 512

    @patch("light_splade.data.base.AutoTokenizer")
    def test_tokenize(self, mock_auto_tokenizer: Mock) -> None:
        expected_outputs = {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
        }
        mock_tokenizer = Mock(return_value=expected_outputs)
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        collator = BaseSpladeCollator(tokenizer_path="test-tokenizer", max_length=256)

        text_list = ["hello world", "test text"]
        result = collator.tokenize(text_list)

        assert result == expected_outputs
        mock_tokenizer.assert_called_once_with(
            text_list,
            add_special_tokens=True,
            padding="longest",
            truncation="longest_first",
            max_length=256,
            return_attention_mask=True,
            return_tensors="pt",
        )

    @patch("light_splade.data.base.AutoTokenizer")
    def test___call__(self, mock_auto_tokenizer: Mock) -> None:
        mock_tokenizer = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        collator = BaseSpladeCollator(tokenizer_path="test-tokenizer")
        collator.collate_fn = Mock()

        features = [("query", "doc1", "doc2", 0.8, 0.2)]
        collator(features)
        collator.collate_fn.assert_called_once_with(features)

    @patch("light_splade.data.base.AutoTokenizer")
    def test_collate_fn_not_implemented(self, mock_auto_tokenizer: Mock) -> None:
        mock_tokenizer = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        collator = BaseSpladeCollator(tokenizer_path="test-tokenizer")
        features = [("query", "doc1", "doc2", 0.8, 0.2)]

        with pytest.raises(NotImplementedError) as exc_info:
            collator.collate_fn(features)

        assert "collate_fn method should be implemented in the derived class" in str(exc_info.value)
