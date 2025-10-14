"""In-batch negative loss utility module.

Contains :class:`InBatchNegativesLoss` which implements the in-batch negative training objective used by SPLADE v2.
"""

import torch


class InBatchNegativesLoss(torch.nn.Module):
    """In-batch negative loss for training SPLADE v2 model.

    Reference:
        SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval (Formal et al.)
        - https://arxiv.org/pdf/2109.10086
    """

    loss_type = "in_batch_negatives"

    def __init__(self) -> None:
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute in-batch negative loss.

        Args:
            inputs (dict[str, torch.Tensor]): Mapping with keys ``pos_score`` and ``neg_score``.
                ``pos_score`` must be (bs, bs) with diagonal entries representing query-to-own-positive similarities.
                ``neg_score`` must be (bs, 1) with query-to-negative similarities.

        Returns:
            Scalar tensor with the in-batch negative loss.
        """
        cross_scores = inputs["pos_score"]  # (bs, bs)
        neg_scores = inputs["neg_score"]  # (bs, 1)

        bs = cross_scores.shape[0]
        if cross_scores.shape != torch.Size([bs, bs]):
            raise ValueError(f"Expected cross_scores shape {(bs, bs)}, but got {cross_scores.shape}")
        if neg_scores.shape != torch.Size([bs, 1]):
            raise ValueError(f"Expected neg_scores shape {(bs, 1)}, but got {neg_scores.shape}")

        pos_scores = torch.diagonal(cross_scores).unsqueeze(-1)  # (bs, 1)
        off_diag_scores = cross_scores.flatten()[1:].view(bs - 1, bs + 1)[:, :-1].reshape(bs, bs - 1)  # (bs, bs-1)
        logits = torch.cat([pos_scores, neg_scores, off_diag_scores], dim=1)  # (bs, 1 + 1 + bs-1) = (bs, bs+1)
        scores = self.log_softmax(logits)
        loss = -scores[:, 0].mean()
        return loss
