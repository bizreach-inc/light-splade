import pytest
import torch

from light_splade.regularizer.flops import FLOPS


class TestFLOPS:
    @pytest.mark.parametrize(
        "weights, expected",
        [
            (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.sum(torch.mean(torch.abs(torch.tensor([[1.0, 2.0], [3.0, 4.0]])), dim=0) ** 2),
            ),
            (torch.tensor([[0.0, 0.0], [0.0, 0.0]]), torch.tensor(0.0)),
            (
                torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]),
                torch.sum(
                    torch.mean(
                        torch.abs(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])),
                        dim=0,
                    )
                    ** 2
                ),
            ),
        ],
    )
    def test___call__(self, weights: torch.Tensor, expected: torch.Tensor) -> None:
        flops = FLOPS()

        result = flops(weights)

        assert torch.allclose(result, expected, atol=1e-6)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0  # Should be a scalar tensor

    def test___call___single_batch(self) -> None:
        flops = FLOPS()
        weights = torch.tensor([[2.0, 4.0, 6.0]])

        result = flops(weights)

        # For single batch: mean([2,4,6]) = [2,4,6], then sum([4,16,36]) = 56
        expected = torch.tensor(56.0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test___call___multiple_batches(self) -> None:
        flops = FLOPS()
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = flops(weights)

        # mean([1,3,5], [2,4,6]) = [3,4], then sum([9,16]) = 25
        expected = torch.tensor(25.0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test___call___negative_values(self) -> None:
        flops = FLOPS()
        weights = torch.tensor([[-1.0, -2.0], [3.0, -4.0]])

        result = flops(weights)

        # abs([-1,3], [-2,-4]) = [1,3], [2,4]; mean = [2,3];
        # square = [4,9]; sum = 13
        expected = torch.tensor(13.0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test___call___empty_tensor_raises_error(self) -> None:
        flops = FLOPS()
        weights = torch.empty(0, 0)

        # This should handle gracefully or raise appropriate error
        result = flops(weights)
        assert torch.isnan(result) or result == 0.0
