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

import pytest
import torch

from light_splade.losses import InBatchNegativesLoss


class TestInBatchNegativesLoss:
    def setup_method(self) -> None:
        self.loss_fn = InBatchNegativesLoss()

    @pytest.mark.parametrize(
        "inputs, expected_loss",
        [
            (
                {
                    "pos_score": torch.tensor(
                        [
                            [0.9543, -0.9247, 0.3661, -0.4612],
                            [0.8875, -0.6773, -0.1575, -0.7545],
                            [-0.0651, 0.4479, -0.2165, 0.7998],
                            [-0.7042, -0.0801, 0.1492, 0.6187],
                        ],
                        requires_grad=True,
                    ),
                    "neg_score": torch.tensor([[-1.2506], [0.0392], [-0.5769], [1.0285]], requires_grad=True),
                },
                torch.tensor(1.61305),
            ),
        ],
    )
    def test_call_with_valid_inputs(self, inputs: dict[str, Any], expected_loss: torch.Tensor) -> None:
        loss = self.loss_fn(inputs)
        assert torch.allclose(loss, expected_loss)

    @pytest.mark.parametrize(
        "invalid_inputs",
        [
            ({"pos_score": torch.randn(4, 1), "neg_score": torch.randn(4, 1)}),
            ({"pos_score": torch.randn(4, 1), "neg_score": torch.randn(4)}),
            ({"pos_score": torch.randn(4, 1), "neg_score": torch.randn(5, 1)}),
            ({"pos_score": torch.randn(4), "neg_score": torch.randn(4, 1)}),
        ],
    )
    def test_call_with_invalid_inputs_dim(self, invalid_inputs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            _ = self.loss_fn(invalid_inputs)
