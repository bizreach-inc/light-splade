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

"""FLOPS-based regularizer.

Implements the FLOPS regularization term used to encourage sparse representations, following Paria et al. The
implementation computes the mean absolute activation per vocabulary dimension and returns the squared L2 norm across
vocabulary dimensions (see Eq. (4) in the reference).
"""

import torch

from .base import BaseRegularizer


class FLOPS(BaseRegularizer):
    """FLOPS regularizer callable.

    The regularizer returns a scalar tensor computed as ``sum(mean(abs(weights), dim=0) ** 2)`` where the mean is across
    the batch dimension.
    """

    def __call__(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute FLOPS regularization loss as in Eq (4) from [1]. Note that the index N in the Eq (4) is batch_size,
        not the num of tokens.

        Args:
            weights (torch.Tensor): Tensor of shape (bs, V) containing activations for each vocabulary term.

        Returns:
            torch.Tensor: Scalar tensor representing the FLOPS penalty.
        """
        return torch.sum(torch.mean(torch.abs(weights), dim=0) ** 2)
