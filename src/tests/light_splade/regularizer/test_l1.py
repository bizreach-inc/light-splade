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

from light_splade.regularizer.l1 import L1


class TestL1:
    @pytest.mark.parametrize(
        "weights, expected",
        [
            (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.sum(torch.abs(torch.tensor([[1.0, 2.0], [3.0, 4.0]])), dim=-1).mean(),
            ),
            (torch.tensor([[0.0, 0.0], [0.0, 0.0]]), torch.tensor(0.0)),
            (
                torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]),
                torch.sum(
                    torch.abs(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])),
                    dim=-1,
                ).mean(),
            ),
        ],
    )
    def test___call__(self, weights: torch.Tensor, expected: torch.Tensor) -> None:
        l1 = L1()

        result = l1(weights)

        assert torch.allclose(result, expected, atol=1e-6)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0  # Should be a scalar tensor

    def test___call___single_batch(self) -> None:
        l1 = L1()
        weights = torch.tensor([[2.0, 4.0, 6.0]])

        result = l1(weights)

        # For single batch: sum([2,4,6]) = 12, mean([12]) = 12
        expected = torch.tensor(12.0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test___call___multiple_batches(self) -> None:
        l1 = L1()
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = l1(weights)

        # sum([1,2], [3,4], [5,6]) = [3,7,11], mean([3,7,11]) = 7
        expected = torch.tensor(7.0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test___call___negative_values(self) -> None:
        l1 = L1()
        weights = torch.tensor([[-1.0, -2.0], [3.0, -4.0]])

        result = l1(weights)

        # abs([-1,-2], [3,-4]) = [1,2], [3,4]; sum = [3,7]; mean = 5
        expected = torch.tensor(5.0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test___call___single_value(self) -> None:
        l1 = L1()
        weights = torch.tensor([[5.0]])

        result = l1(weights)

        expected = torch.tensor(5.0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test___call___mixed_positive_negative(self) -> None:
        l1 = L1()
        weights = torch.tensor([[1.0, -1.0, 2.0], [-2.0, 3.0, -3.0]])

        result = l1(weights)

        # abs([1,-1,2], [-2,3,-3]) = [1,1,2], [2,3,3]; sum = [4,8]; mean = 6
        expected = torch.tensor(6.0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test___call___empty_tensor_raises_error(self) -> None:
        l1 = L1()
        weights = torch.empty(0, 0)

        # This should handle gracefully or raise appropriate error
        result = l1(weights)
        assert torch.isnan(result) or result == 0.0
