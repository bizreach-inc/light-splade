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

"""Margin mean-squared error distillation loss.

Reference:
    Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation (Hofstätter et al.)
    - https://arxiv.org/pdf/2010.02666

This module implements a margin-based MSE loss used for distillation. The
loss compares student and teacher margins (pos - neg) using mean-squared
error.
"""

import torch


class DistillMarginMSELoss:
    """Compute MSE between student and teacher margins.

    Expected ``inputs`` keys: ``pos_score``, ``neg_score``, ``teacher_pos_score``,
    ``teacher_neg_score``. Shapes are normalized if necessary (teacher may
    provide (bs,) instead of (bs,1)).
    """

    loss_type = "margin_mse"

    def __init__(self) -> None:
        self.loss = torch.nn.MSELoss(reduction="mean")

    def __call__(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the scalar margin-MSE loss.

        Args:
            inputs: Mapping with student and teacher scores.

        Returns:
            Scalar tensor with the MSE loss between margins.
        """
        student_margin = inputs["pos_score"] - inputs["neg_score"]
        teacher_margin = inputs["teacher_pos_score"] - inputs["teacher_neg_score"]
        if student_margin.shape != teacher_margin.shape:
            if student_margin.shape != (teacher_margin.shape[0], 1):
                raise ValueError(
                    f"Shape mismatch in DistillMarginMSELoss: student_margin {student_margin.shape} and teacher_margin "
                    f"{teacher_margin.shape} must have the same shape."
                )
            teacher_margin = teacher_margin.unsqueeze(-1)

        loss: torch.Tensor = self.loss(student_margin, teacher_margin)
        return loss
