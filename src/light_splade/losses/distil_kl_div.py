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
