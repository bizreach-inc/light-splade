from unittest.mock import Mock
from unittest.mock import patch

import torch

from light_splade.data.triplet_distil_datacollator import TripletDistilCollator
from light_splade.schemas.model import TripletDistilBatch


class TestTripletDistilCollator:
    @patch("light_splade.data.base.AutoTokenizer")
    def test_collate_fn(self, mock_auto_tokenizer: Mock) -> None:
        mock_tokenizer = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        features = [
            ("query1", "pos_doc1", "neg_doc1", 0.9, 0.1),
            ("query2", "pos_doc2", "neg_doc2", 0.8, 0.2),
        ]
        mock_q_outputs = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }
        mock_pos_outputs = {
            "input_ids": torch.tensor([[7, 8, 9], [10, 11, 12]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }
        mock_neg_outputs = {
            "input_ids": torch.tensor([[13, 14, 15], [16, 17, 18]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

        collator = TripletDistilCollator(tokenizer_path="test-tokenizer")
        collator.tokenize = Mock(side_effect=[mock_q_outputs, mock_pos_outputs, mock_neg_outputs])

        result = collator.collate_fn(features)

        assert isinstance(result, TripletDistilBatch)
        assert torch.equal(result.q_input_ids, mock_q_outputs["input_ids"])
        assert torch.equal(result.q_attention_mask, mock_q_outputs["attention_mask"])
        assert torch.equal(result.pos_input_ids, mock_pos_outputs["input_ids"])
        assert torch.equal(result.pos_attention_mask, mock_pos_outputs["attention_mask"])
        assert torch.equal(result.neg_input_ids, mock_neg_outputs["input_ids"])
        assert torch.equal(result.neg_attention_mask, mock_neg_outputs["attention_mask"])
        assert torch.equal(result.teacher_pos_scores, torch.tensor([0.9, 0.8]))
        assert torch.equal(result.teacher_neg_scores, torch.tensor([0.1, 0.2]))

        assert collator.tokenize.call_count == 3
        collator.tokenize.assert_any_call(("query1", "query2"))
        collator.tokenize.assert_any_call(("pos_doc1", "pos_doc2"))
        collator.tokenize.assert_any_call(("neg_doc1", "neg_doc2"))
