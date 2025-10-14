import torch
import torch.nn.functional as F

from light_splade.losses import DistillKLDivLoss


class TestDistillKLDivLoss:
    def setup_method(self) -> None:
        self.loss_fn = DistillKLDivLoss()

    def test_loss_type(self) -> None:
        assert self.loss_fn.loss_type == "kldiv"

    def test_initialization(self) -> None:
        assert isinstance(self.loss_fn.loss, torch.nn.KLDivLoss)
        assert self.loss_fn.loss.reduction == "batchmean"
        assert self.loss_fn.loss.log_target is True

    def test_call_with_valid_inputs(self) -> None:
        bs = 16
        inputs: dict[str, torch.Tensor] = {
            "pos_score": torch.randn(bs, 1),
            "neg_score": torch.randn(bs, 1),
            "teacher_pos_score": torch.randn(bs, 1),
            "teacher_neg_score": torch.randn(bs, 1),
        }
        loss = self.loss_fn(inputs)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1

        # Manually calculate the expected loss
        student_scores = torch.hstack([inputs["pos_score"], inputs["neg_score"]])
        teacher_scores = torch.hstack([inputs["teacher_pos_score"], inputs["teacher_neg_score"]])

        log_student_probs = F.log_softmax(student_scores, dim=1)
        log_teacher_probs = F.log_softmax(teacher_scores, dim=1)

        expected_loss = F.kl_div(
            log_student_probs,
            log_teacher_probs,
            reduction="batchmean",
            log_target=True,
        )
        assert torch.allclose(loss, expected_loss)

    def test_call_with_truly_identical_distributions(self) -> None:
        bs = 2
        identical_pos_score = torch.ones(bs, 1) * 2
        identical_neg_score = torch.ones(bs, 1) * 1
        inputs: dict[str, torch.Tensor] = {
            "pos_score": identical_pos_score,
            "neg_score": identical_neg_score,
            "teacher_pos_score": identical_pos_score,
            "teacher_neg_score": identical_neg_score,
        }
        loss = self.loss_fn(inputs)
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_call_with_different_batch_sizes(self) -> None:
        bs1 = 3
        inputs1: dict[str, torch.Tensor] = {
            "pos_score": torch.randn(bs1, 1),
            "neg_score": torch.randn(bs1, 1),
            "teacher_pos_score": torch.randn(bs1, 1),
            "teacher_neg_score": torch.randn(bs1, 1),
        }
        loss1 = self.loss_fn(inputs1)
        assert isinstance(loss1, torch.Tensor)
        assert loss1.numel() == 1

        bs2 = 5
        inputs2: dict[str, torch.Tensor] = {
            "pos_score": torch.randn(bs2, 1),
            "neg_score": torch.randn(bs2, 1),
            "teacher_pos_score": torch.randn(bs2, 1),
            "teacher_neg_score": torch.randn(bs2, 1),
        }
        loss2 = self.loss_fn(inputs2)
        assert isinstance(loss2, torch.Tensor)
        assert loss2.numel() == 1

    def test_call_with_extreme_score_differences(self) -> None:
        bs = 2
        inputs: dict[str, torch.Tensor] = {
            "pos_score": torch.ones(bs, 1) * 10,
            "neg_score": torch.ones(bs, 1) * -10,
            "teacher_pos_score": torch.ones(bs, 1) * -10,
            "teacher_neg_score": torch.ones(bs, 1) * 10,
        }
        loss = self.loss_fn(inputs)
        assert loss > 0  # Expect a positive loss value
