"""Base classes for regularizers used by SPLADE training.

This module defines the abstract :class:`BaseRegularizer` interface which all regularizers in the project implement. A
regularizer is callable and accepts a batch of model-generated weight vectors (for example, vocabulary-sized token
activations) and returns a scalar loss tensor.
"""

from abc import abstractmethod

import torch


class BaseRegularizer:
    """Abstract interface for regularizer callables.

    Subclasses must implement :meth:`__call__` and return a scalar tensor that represents the regularization penalty for
    a batch of weight vectors.
    """

    @abstractmethod
    def __call__(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute the regularization penalty for a batch of weights.

        Args:
            weights (torch.Tensor): Batch of weight vectors, typically shaped ``(batch_size, vocab_size)``.

        Returns:
            torch.Tensor: A scalar tensor containing the regularization loss.
        """
        raise NotImplementedError
