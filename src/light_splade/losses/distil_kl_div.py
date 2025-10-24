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

"""KL-divergence based distillation loss.

Reference:
    Distilling Dense Representations for Ranking using Tightly-Coupled Teachers (Sheng-Chieh Lin et al.)
    - https://arxiv.org/pdf/2010.11386

The loss compares student pairwise scores against teacher pairwise scores using KL-divergence over the 2-way
distribution [pos, neg].
"""

import torch
import torch.nn.functional as F


class DistillKLDivLoss:
    """Compute a KL-divergence distillation loss for pairwise scores.

    Expected ``inputs`` is a dict containing student and teacher scores with keys: ``pos_score``, ``neg_score``,
    ``teacher_pos_score``, ``teacher_neg_score``. All tensors should have shape (bs, 1).
    """

    loss_type = "kldiv"

    def __init__(self) -> None:
        self.loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    def __call__(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the scalar loss value.

        Args:
            inputs (dict[str, torch.Tensor]): Mapping with student and teacher scores (see class docstring).

        Returns:
            Scalar tensor with the KL-divergence loss.
        """
        student_scores = torch.hstack([inputs["pos_score"], inputs["neg_score"]])  # (bs, 2)
        teacher_scores = torch.hstack([inputs["teacher_pos_score"], inputs["teacher_neg_score"]])  # (bs, 2)

        inputs_ = F.log_softmax(student_scores, dim=1)  # (bs, 2)
        target = F.log_softmax(teacher_scores, dim=1)  # (bs, 2)

        loss: torch.Tensor = self.loss(inputs_, target)
        return loss
