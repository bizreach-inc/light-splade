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

import pytest
import torch

from light_splade.losses.pairwise_contrastive import PairwiseContrastiveLoss


class TestPairwiseContrastiveLoss:
    def setup_method(self) -> None:
        self.loss_fn = PairwiseContrastiveLoss()

    def test_forward_deterministic(self) -> None:
        # Construct a deterministic small batch
        pos = torch.tensor([[2.0], [0.0], [1.0]], requires_grad=True)
        neg = torch.tensor([[0.0], [0.0], [1.0]], requires_grad=True)
        inputs = {"pos_score": pos, "neg_score": neg}
        loss = self.loss_fn(inputs)
        assert loss.shape == ()  # scalar

        # Manual computation
        logits = torch.cat([pos, neg], dim=1)  # (3,2)
        log_probs = torch.log_softmax(logits, dim=1)
        expected = -log_probs[:, 0].mean()
        assert torch.allclose(loss, expected)

        loss.backward()
        # Ensure gradients exist
        assert pos.grad is not None and neg.grad is not None

    def test_forward_identical_scores(self) -> None:
        # When pos == neg for each sample, probability for pos is 0.5 -> loss = -mean(log(0.5))
        pos = torch.zeros(4, 1, requires_grad=True)
        neg = torch.zeros(4, 1, requires_grad=True)
        inputs = {"pos_score": pos, "neg_score": neg}
        loss = self.loss_fn(inputs)
        expected = -torch.log(torch.tensor(0.5))  # scalar
        assert torch.allclose(loss, expected)

    def test_forward_extreme_difference(self) -> None:
        # Large positive margin should push loss close to 0
        pos = torch.ones(5, 1) * 50
        neg = torch.ones(5, 1) * -50
        inputs = {"pos_score": pos, "neg_score": neg}
        loss = self.loss_fn(inputs)
        assert loss < 1e-4

    @pytest.mark.parametrize(
        "invalid_inputs",
        [
            {"pos_score": torch.randn(4, 1), "neg_score": torch.randn(4, 2)},  # wrong neg shape
            {"pos_score": torch.randn(4, 2), "neg_score": torch.randn(4, 1)},  # wrong pos shape
            {"pos_score": torch.randn(5, 1), "neg_score": torch.randn(4, 1)},  # batch mismatch
            {"pos_score": torch.randn(4), "neg_score": torch.randn(4, 1)},  # missing last dim
        ],
    )
    def test_invalid_shapes(self, invalid_inputs: dict) -> None:
        with pytest.raises(ValueError):
            _ = self.loss_fn(invalid_inputs)

    def test_grad_flow_non_zero(self) -> None:
        pos = torch.randn(6, 1, requires_grad=True)
        neg = torch.randn(6, 1, requires_grad=True)
        inputs = {"pos_score": pos, "neg_score": neg}
        loss = self.loss_fn(inputs)
        loss.backward()
        # At least one gradient element should be non-zero (likely most are)
        assert torch.any(pos.grad != 0) and torch.any(neg.grad != 0)
