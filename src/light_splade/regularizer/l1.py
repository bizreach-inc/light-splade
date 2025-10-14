"""Simple L1 regularizer.

Computes the average L1 norm across batch elements and returns a scalar penalty.
"""

import torch

from .base import BaseRegularizer


class L1(BaseRegularizer):
    """L1 regularization callable."""

    def __call__(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute the L1 regularization loss.

        Args:
            weights (torch.Tensor): Tensor shaped ``(bs, V)`` containing activations for each vocabulary term.

        Returns:
            torch.Tensor: Scalar tensor equal to the mean L1 norm across the
            batch.
        """
        return torch.sum(torch.abs(weights), dim=-1).mean()
