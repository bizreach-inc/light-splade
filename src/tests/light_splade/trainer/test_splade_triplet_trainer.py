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

import typing
from collections import defaultdict
from typing import Any
from typing import Callable
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from light_splade.losses import InBatchNegativesLoss
from light_splade.losses import PairwiseContrastiveLoss
from light_splade.regularizer import FLOPS
from light_splade.schemas.config import SpladeTrainingArguments
from light_splade.schemas.model import TripletBatch
from light_splade.trainer.splade_triplet_trainer import SUPPORTED_LOSSES
from light_splade.trainer.splade_triplet_trainer import SpladeTripletTrainer
from light_splade.utils.scoring import dot_product
from light_splade.utils.scoring import dot_product_cross


class DummyModel:
    """Minimal stand-in for Splade two-tower interface.

    Depending on arguments passed, returns dictionaries mimicking the real model outputs.
    """

    def __init__(self, q_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vec: torch.Tensor) -> None:
        self.q_vec = q_vec
        self.pos_vec = pos_vec
        self.neg_vec = neg_vec

    def __call__(
        self,
        q_input_ids: torch.Tensor | None = None,
        q_attention_mask: torch.Tensor | None = None,
        d_input_ids: torch.Tensor | None = None,
        d_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if q_input_ids is not None and d_input_ids is not None:
            return {"q_vector": self.q_vec, "d_vector": self.pos_vec}
        return {"d_vector": self.neg_vec}


class DummyLoss:
    def __init__(self, value: float, loss_type: str) -> None:
        self.value = torch.tensor(value)
        self.loss_type = loss_type

    def __call__(self, params: dict) -> torch.Tensor:
        assert "pos_score" in params and "neg_score" in params
        return self.value


@pytest.fixture
def vectors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # batch size 3, vocab size 5
    bs, V = 3, 5
    q_vec = torch.eye(V)[:bs]
    pos_vec = torch.eye(V)[:bs]
    neg_vec = torch.flip(torch.eye(V), dims=[0])[:bs]
    return q_vec, pos_vec, neg_vec


@pytest.fixture
def batch() -> TripletBatch:
    return TripletBatch(
        q_input_ids=torch.tensor([0, 1, 2]),
        q_attention_mask=None,
        pos_input_ids=torch.tensor([3, 4, 5]),
        pos_attention_mask=None,
        neg_input_ids=torch.tensor([6, 7, 8]),
        neg_attention_mask=None,
    )


@typing.no_type_check
def build_trainer(
    training_loss: str,
    loss_impl: DummyLoss,
    lambda_q: float = 0.0,
    lambda_d: float = 0.0,
    q_reg: Callable | None = None,
    d_reg: Callable | None = None,
) -> SpladeTripletTrainer:
    trainer = SpladeTripletTrainer.__new__(SpladeTripletTrainer)
    trainer.args = SpladeTrainingArguments(training_loss=training_loss)
    trainer.losses = [loss_impl]
    trainer.score_fn = dot_product
    trainer.cross_score_fn = dot_product_cross
    trainer.get_lambdas = lambda: (lambda_q, lambda_d)
    trainer.q_regularizer = q_reg or (lambda v: torch.tensor(0.0))
    trainer.d_regularizer = d_reg or (lambda v: torch.tensor(0.0))
    return trainer


def test_supported_losses_mapping() -> None:
    # Ensure exposure of both implemented loss functions
    assert InBatchNegativesLoss.loss_type in SUPPORTED_LOSSES
    assert PairwiseContrastiveLoss.loss_type in SUPPORTED_LOSSES


def test_create_losses_success_pairwise() -> None:
    trainer = SpladeTripletTrainer.__new__(SpladeTripletTrainer)
    args = SpladeTrainingArguments(training_loss="pairwise_contrastive")
    losses = trainer._create_losses(args)
    assert len(losses) == 1
    assert isinstance(losses[0], PairwiseContrastiveLoss)


def test_create_losses_success_in_batch() -> None:
    trainer = SpladeTripletTrainer.__new__(SpladeTripletTrainer)
    args = SpladeTrainingArguments(training_loss="in_batch_negatives")
    losses = trainer._create_losses(args)
    assert len(losses) == 1
    assert isinstance(losses[0], InBatchNegativesLoss)


def test_create_losses_invalid() -> None:
    trainer = SpladeTripletTrainer.__new__(SpladeTripletTrainer)
    args = SpladeTrainingArguments(training_loss="unknown")
    with pytest.raises(ValueError) as exc_info:
        trainer._create_losses(args)
    assert "Unsupported training loss" in str(exc_info.value)


def test_compute_loss_pairwise(vectors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch: TripletBatch) -> None:
    q_vec, pos_vec, neg_vec = vectors
    model = DummyModel(q_vec, pos_vec, neg_vec)

    loss_impl = DummyLoss(1.2, "pairwise_contrastive")
    trainer = build_trainer("pairwise_contrastive", loss_impl)

    accum: defaultdict[str, list] = defaultdict(list)
    loss_value, (q_out, pos_out, neg_out) = trainer._compute_loss(model, batch, accum_loss_values=accum)

    assert torch.allclose(loss_value, torch.tensor(1.2))
    assert torch.allclose(q_out, q_vec)
    assert torch.allclose(pos_out, pos_vec)
    assert torch.allclose(neg_out, neg_vec)
    assert len(accum["pairwise_contrastive_loss"]) == 1


def test_compute_loss_in_batch_negatives_shapes(
    vectors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch: TripletBatch
) -> None:
    q_vec, pos_vec, neg_vec = vectors
    model = DummyModel(q_vec, pos_vec, neg_vec)

    class ShapeCheckingLoss(DummyLoss):
        def __call__(self, params: dict) -> torch.Tensor:
            pos = params["pos_score"]
            neg = params["neg_score"]
            assert pos.shape == (q_vec.size(0), q_vec.size(0))  # full cross
            assert neg.shape == (q_vec.size(0), 1)
            return super().__call__(params)

    loss_impl = ShapeCheckingLoss(0.9, "in_batch_negatives")
    trainer = build_trainer("in_batch_negatives", loss_impl)
    loss_value, _ = trainer._compute_loss(model, batch, accum_loss_values=None)
    assert torch.allclose(loss_value, torch.tensor(0.9))


def test_compute_loss_regularizer_added(
    vectors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch: TripletBatch
) -> None:
    q_vec, pos_vec, neg_vec = vectors
    model = DummyModel(q_vec, pos_vec, neg_vec)

    base_loss = DummyLoss(2.0, "pairwise_contrastive")

    def q_reg(_: Any) -> torch.Tensor:
        return torch.tensor(0.5)

    def d_reg(_: Any) -> torch.Tensor:
        return torch.tensor(0.25)

    trainer = build_trainer(
        "pairwise_contrastive",
        base_loss,
        lambda_q=0.1,
        lambda_d=0.2,
        q_reg=q_reg,
        d_reg=d_reg,
    )

    # reg = 0.1*0.5 + 0.2*0.25*0.5 + 0.2*0.25*0.5 = 0.05 + 0.05 = 0.10
    expected_total = 2.0 + 0.10
    accum: defaultdict[str, list] = defaultdict(list)
    loss_value, _ = trainer._compute_loss(model, batch, accum_loss_values=accum)

    assert torch.allclose(loss_value, torch.tensor(expected_total))
    assert pytest.approx(accum["regularizer"][0], rel=1e-6) == 0.10


@typing.no_type_check
def test_compute_loss_multiple_losses_accumulate(
    vectors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch: TripletBatch
) -> None:
    q_vec, pos_vec, neg_vec = vectors
    model = DummyModel(q_vec, pos_vec, neg_vec)

    loss1 = DummyLoss(0.7, "pairwise_contrastive")
    loss2 = DummyLoss(0.3, "aux")

    trainer = SpladeTripletTrainer.__new__(SpladeTripletTrainer)
    trainer.args = SpladeTrainingArguments(training_loss="pairwise_contrastive")
    trainer.losses = [loss1, loss2]
    trainer.score_fn = dot_product
    trainer.cross_score_fn = dot_product_cross
    trainer.get_lambdas = lambda: (0.0, 0.0)
    trainer.q_regularizer = FLOPS()
    trainer.d_regularizer = FLOPS()

    accum: defaultdict[str, list] = defaultdict(list)
    loss_value, _ = trainer._compute_loss(model, batch, accum_loss_values=accum)

    assert torch.allclose(loss_value, torch.tensor(1.0))
    assert len(accum["pairwise_contrastive_loss"]) == 1
    assert len(accum["aux_loss"]) == 1


def test_compute_loss_no_accumulator(
    vectors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch: TripletBatch
) -> None:
    q_vec, pos_vec, neg_vec = vectors
    model = DummyModel(q_vec, pos_vec, neg_vec)

    loss_impl = DummyLoss(0.55, "pairwise_contrastive")
    trainer = build_trainer("pairwise_contrastive", loss_impl)
    loss_value, _ = trainer._compute_loss(model, batch, accum_loss_values=None)
    assert torch.allclose(loss_value, torch.tensor(0.55))


@patch("light_splade.trainer.splade_triplet_trainer.dot_product_cross")
@patch("light_splade.trainer.splade_triplet_trainer.SpladeTrainer.__init__")
def test_init_sets_cross_score_fn(mock_parent_init: Mock, mock_cross: Mock) -> None:
    mock_parent_init.return_value = None
    trainer = SpladeTripletTrainer(
        model=Mock(),
        args=SpladeTrainingArguments(training_loss="pairwise_contrastive"),
    )
    assert trainer.cross_score_fn == mock_cross
