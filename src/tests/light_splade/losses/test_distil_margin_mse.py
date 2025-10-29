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

from light_splade.losses import DistillMarginMSELoss


class TestDistillMarginMSELoss:
    def setup_method(self) -> None:
        self.loss_fn = DistillMarginMSELoss()

    @pytest.mark.parametrize(
        "inputs",
        [
            (
                {
                    "pos_score": torch.randn(16, 1),
                    "neg_score": torch.randn(16, 1),
                    "teacher_pos_score": torch.randn(16, 1),
                    "teacher_neg_score": torch.randn(16, 1),
                }
            ),
            (
                {
                    "pos_score": torch.randn(16, 1),
                    "neg_score": torch.randn(16, 1),
                    "teacher_pos_score": torch.randn(16),
                    "teacher_neg_score": torch.randn(16),
                }
            ),
        ],
    )
    def test_call_with_valid_inputs(self, inputs: dict) -> None:
        loss = self.loss_fn(inputs)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1

        # Manually calculate the expected loss
        student_margin = inputs["pos_score"] - inputs["neg_score"]
        teacher_margin = inputs["teacher_pos_score"] - inputs["teacher_neg_score"]
        if len(teacher_margin.shape) == 1:
            teacher_margin = teacher_margin.unsqueeze(-1)
        expected_loss = torch.mean((student_margin - teacher_margin) ** 2)
        assert torch.allclose(loss, expected_loss)

    def test_call_with_invalid_inputs(self) -> None:
        bs = 16
        inputs: dict[str, torch.Tensor] = {
            "pos_score": torch.randn(bs, 2),
            "neg_score": torch.randn(bs, 2),
            "teacher_pos_score": torch.randn(bs),
            "teacher_neg_score": torch.randn(bs),
        }
        with pytest.raises(ValueError):
            _ = self.loss_fn(inputs)

    def test_call_with_zero_difference(self) -> None:
        bs = 2
        inputs: dict[str, torch.Tensor] = {
            "pos_score": torch.ones(bs, 1) * 2,
            "neg_score": torch.ones(bs, 1) * 2,
            "teacher_pos_score": torch.ones(bs, 1) * 5,
            "teacher_neg_score": torch.ones(bs, 1) * 5,
        }
        loss = self.loss_fn(inputs)
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_call_with_different_batch_sizes(self) -> None:
        bs1 = 3
        inputs1: dict[str, torch.Tensor] = {
            "pos_score": torch.randn(bs1, 1),
            "neg_score": torch.randn(bs1, 1),
            "teacher_pos_score": torch.randn(bs1, 1),
            "teacher_neg_score": torch.randn(bs1, 1),
        }
        loss1 = self.loss_fn(inputs1)
        assert isinstance(loss1, torch.Tensor)
        assert loss1.numel() == 1

        bs2 = 5
        inputs2: dict[str, torch.Tensor] = {
            "pos_score": torch.randn(bs2, 1),
            "neg_score": torch.randn(bs2, 1),
            "teacher_pos_score": torch.randn(bs2, 1),
            "teacher_neg_score": torch.randn(bs2, 1),
        }
        loss2 = self.loss_fn(inputs2)
        assert isinstance(loss2, torch.Tensor)
        assert loss2.numel() == 1

    def test_call_with_large_differences(self) -> None:
        bs = 8
        inputs: dict[str, torch.Tensor] = {
            "pos_score": torch.ones(bs, 1) * 100,
            "neg_score": torch.ones(bs, 1) * -100,
            "teacher_pos_score": torch.ones(bs, 1) * -100,
            "teacher_neg_score": torch.ones(bs, 1) * 100,
        }
        loss = self.loss_fn(inputs)
        assert loss > 0  # Expect a positive loss value
