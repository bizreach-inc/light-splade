"""Triplet-only SPLADE training (no distillation).

`SpladeTripletTrainer` is a lightweight variant of `SpladeTrainer` used when training SPLADE models without teacher
supervision. It supports two loss styles:
* in_batch_negatives: Uses the cross query–doc similarity matrix (bs x bs).
* pairwise_contrastive: Classic (q, pos, neg) scoring with pointwise scores.

The implementation reuses the regularization and logging infrastructure from the base trainer but simplifies the loss
parameter map.
"""

from logging import getLogger
from typing import Any

import torch

from light_splade.losses import InBatchNegativesLoss
from light_splade.losses import PairwiseContrastiveLoss
from light_splade.models import Splade
from light_splade.schemas.config import SpladeTrainingArguments
from light_splade.schemas.model import TripletBatch
from light_splade.utils.scoring import dot_product_cross

from .splade_trainer import SpladeTrainer

logger = getLogger(__name__)


# add more splade model class here for type checking in the SpladeTripletTrainer
SUPPORTED_SPLADE_MODELS = (Splade,)
SUPPORTED_LOSSES = {loss_cls.loss_type: loss_cls for loss_cls in [InBatchNegativesLoss, PairwiseContrastiveLoss]}  # type: ignore[attr-defined]


class SpladeTripletTrainer(SpladeTrainer):
    """Trainer for SPLADE (triplet mode, no distillation)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cross_score_fn = dot_product_cross

    def _create_losses(self, args: SpladeTrainingArguments) -> list[InBatchNegativesLoss | PairwiseContrastiveLoss]:  # type: ignore
        """Instantiate exactly one supported triplet loss.

        Returned as a one-element list for consistency with the base trainer
        interface.

        Args:
            args: Training arguments with ``training_loss`` specifying the loss.

        Returns:
            A list with a single instantiated loss callable.
        """
        if args.training_loss not in SUPPORTED_LOSSES:
            raise ValueError(
                f"Unsupported training loss: {args.training_loss}. Only {list(SUPPORTED_LOSSES.keys())} is supported."
            )

        loss_cls = SUPPORTED_LOSSES[args.training_loss]
        losses = [loss_cls()]
        return losses  # type: ignore[return-value]

    def _compute_loss(
        self,
        model: Splade,
        inputs: TripletBatch,  # type: ignore[override]
        accum_loss_values: dict | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """Forward pass and loss computation for a raw triplet batch.

        - For `in_batch_negatives` the positive score matrix is (bs, bs), comparing each query to every positive doc in
        the batch (negatives are shared implicitly).
        - For `pairwise_contrastive` the positive score is (bs, 1).

        Args:
            model (Splade): model instance used for encoding.
            inputs (TripletBatch): containing tokenized tensors.
            accum_loss_values (dict | None): Optional dict accumulator for per-component losses.

        Returns:
            A tuple ``(loss, (q_vec, pos_vec, neg_vec))`` similar to the base trainer.
        """
        pos_vectors: dict = model(
            q_input_ids=inputs.q_input_ids,
            q_attention_mask=inputs.q_attention_mask,
            d_input_ids=inputs.pos_input_ids,
            d_attention_mask=inputs.pos_attention_mask,
        )
        # ignore the q_vector computation from neg_vectors
        neg_vectors: dict = model(
            d_input_ids=inputs.neg_input_ids,
            d_attention_mask=inputs.neg_attention_mask,
        )
        q_vector = pos_vectors["q_vector"]  # (bs, V)
        pos_d_vector = pos_vectors["d_vector"]  # (bs, V)
        neg_d_vector = neg_vectors["d_vector"]  # (bs, V)
        # NOTE: pos_vectors["q_vector"] is the same as neg_vectors["q_vector"]
        # so, pos_q_vector = neg_q_vector

        pos_score_fn = (
            self.cross_score_fn  # return (bs, bs)
            if self.args.training_loss == "in_batch_negatives"
            else self.score_fn  # return (bs, 1)
        )
        params = dict(
            pos_score=pos_score_fn(q_vector, pos_d_vector),  # (bs, 1) or (bs, bs) depending on the loss fn
            neg_score=self.score_fn(q_vector, neg_d_vector),  # (bs, 1)
        )

        loss_value: torch.Tensor = torch.tensor(0.0)
        for loss in self.losses:
            loss_value_ = loss(params)
            loss_value = loss_value + loss_value_
            if accum_loss_values is not None:
                accum_loss_values[f"{loss.loss_type}_loss"].append(loss_value_.detach().cpu().item())

        # regularizers
        lambda_q, lambda_d = self.get_lambdas()
        reg_loss = (
            lambda_q * self.q_regularizer(q_vector)
            + lambda_d * self.d_regularizer(pos_d_vector) * 0.5
            + lambda_d * self.d_regularizer(neg_d_vector) * 0.5
        )
        loss_value = loss_value + reg_loss
        if accum_loss_values is not None:
            accum_loss_values["regularizer"].append(reg_loss.detach().cpu().item())

        return (loss_value, (q_vector, pos_d_vector, neg_d_vector))
