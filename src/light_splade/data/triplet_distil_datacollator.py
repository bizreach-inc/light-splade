"""Collator for triplet distillation batches.

This collator builds a :class:`TripletDistilBatch` which includes tokenized inputs for query/positive/negative and
teacher-provided similarity scores for positive and negative examples.
"""

import torch

from light_splade.schemas.model import TripletDistilBatch

from .base import BaseSpladeCollator


class TripletDistilCollator(BaseSpladeCollator):
    """Provide collator for the :class:`TripletDistilDataset`.

    The collator expects each feature to be a tuple of (query_text, pos_text, neg_text, score_pos, score_neg) where the
    last two values are floats produced by a teacher model.
    """

    def collate_fn(self, features: list[tuple[str, str, str, float, float]]) -> TripletDistilBatch:
        q_text, pos_text, neg_text, score_pos, score_neg = zip(*features)

        q_outputs = self.tokenize(q_text)
        pos_outputs = self.tokenize(pos_text)
        neg_outputs = self.tokenize(neg_text)

        batch = TripletDistilBatch(
            q_input_ids=q_outputs["input_ids"],
            q_attention_mask=q_outputs["attention_mask"],
            pos_input_ids=pos_outputs["input_ids"],
            pos_attention_mask=pos_outputs["attention_mask"],
            neg_input_ids=neg_outputs["input_ids"],
            neg_attention_mask=neg_outputs["attention_mask"],
            teacher_pos_scores=torch.tensor(score_pos),
            teacher_neg_scores=torch.tensor(score_neg),
        )
        return batch
