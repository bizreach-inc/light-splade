"""Collator that converts triplet text triples into tokenized batches.

This collator consumes features as lists of (query_text, pos_text, neg_text) tuples and returns a :class:`TripletBatch`
with tokenized tensors ready for model ingestion.
"""

from light_splade.schemas.model import TripletBatch

from .base import BaseSpladeCollator


class TripletCollator(BaseSpladeCollator):
    """Provide collator for the :class:`TripletDataset`.

    The collator tokenizes each element of the triplet and constructs a :class:`TripletBatch` containing input ids and
    attention masks for the query, positive and negative examples.
    """

    def collate_fn(self, features: list[tuple[str, str, str]]) -> TripletBatch:
        q_text, pos_text, neg_text = zip(*features)

        q_outputs = self.tokenize(q_text)
        pos_outputs = self.tokenize(pos_text)
        neg_outputs = self.tokenize(neg_text)

        batch = TripletBatch(
            q_input_ids=q_outputs["input_ids"],
            q_attention_mask=q_outputs["attention_mask"],
            pos_input_ids=pos_outputs["input_ids"],
            pos_attention_mask=pos_outputs["attention_mask"],
            neg_input_ids=neg_outputs["input_ids"],
            neg_attention_mask=neg_outputs["attention_mask"],
        )
        return batch
