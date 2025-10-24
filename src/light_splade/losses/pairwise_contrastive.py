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

"""Pairwise contrastive loss module.

This implements a simple pairwise contrastive objective considering only the query's own positive and negative
documents.
"""

import torch


class PairwiseContrastiveLoss(torch.nn.Module):
    """Pairwise contrastive loss for training SPLADE v2 models.

    Expected ``inputs`` mapping contains ``pos_score`` and ``neg_score`` both of shape (bs, 1). The loss computes a
    2-way log-softmax over [pos, neg] and returns the negative log-probability of the positive item averaged across the
    batch.
    """

    loss_type = "pairwise_contrastive"

    def __init__(self) -> None:
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute contrastive loss for the batch.

        Args:
            inputs (dict[str, torch.Tensor]): Mapping with ``pos_score`` and ``neg_score`` tensors of shape (bs, 1).

        Returns:
            Scalar tensor with the average contrastive loss.
        """

        pos_scores = inputs["pos_score"]  # (bs, 1)
        neg_scores = inputs["neg_score"]  # (bs, 1)

        bs = pos_scores.shape[0]
        if pos_scores.shape != torch.Size([bs, 1]):
            raise ValueError(f"Expected pos_scores shape {(bs, 1)}, but got {pos_scores.shape}")
        if neg_scores.shape != torch.Size([bs, 1]):
            raise ValueError(f"Expected neg_scores shape {(bs, 1)}, but got {neg_scores.shape}")

        logits = torch.cat([pos_scores, neg_scores], dim=1)  # (bs, 2)
        scores = self.log_softmax(logits)
        loss: torch.Tensor = -scores[:, 0].mean()
        return loss
