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
