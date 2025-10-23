from collections import defaultdict
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from hydra import compose
from hydra import initialize
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import AutoModelForMaskedLM
from transformers import BertConfig
from transformers import PreTrainedModel

from light_splade.losses import DistillKLDivLoss
from light_splade.losses import DistillMarginMSELoss
from light_splade.models import Splade
from light_splade.regularizer import FLOPS
from light_splade.regularizer import L1
from light_splade.regularizer import RegularizerScheduler
from light_splade.schemas.config import ConfigSpladeDistil
from light_splade.schemas.config import SpladeTrainingArguments
from light_splade.trainer import SpladeTrainer
from light_splade.utils.argument import instantiate


@pytest.fixture
def splade_training_args() -> SpladeTrainingArguments:
    with initialize(
        version_base="1.2",
        config_path="../../data/config",
        job_name="splade_app",
    ):
        cfg = compose(config_name="splade")
        config: ConfigSpladeDistil = instantiate(ConfigSpladeDistil, cfg)

    return config.training


@pytest.fixture
def model() -> Any:
    # Init a very tiny BERT model just for stubbing
    model_config_dict = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 128,
        "initializer_range": 0.02,
        "intermediate_size": 256,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 8,
        "num_hidden_layers": 6,
        "type_vocab_size": 2,
        "vocab_size": 3200,
    }
    model_config = BertConfig(**model_config_dict)
    model = AutoModelForMaskedLM.from_config(model_config)
    return model


class TestSpladeTrainer:
    @pytest.mark.parametrize(
        "args, kwargs, should_raise, expected_error",
        [
            ([], {"model": Mock(), "args": Mock()}, False, ""),
            (
                [Mock()],
                {"model": Mock(), "args": Mock()},
                True,
                "keyword arguments",
            ),
            ([], {"args": Mock()}, True, "`model` is not passed"),
            ([], {"model": Mock()}, True, "`args` is not passed"),
        ],
    )
    def test__validate_args(self, args: list, kwargs: dict, should_raise: bool, expected_error: str) -> None:
        trainer = SpladeTrainer.__new__(SpladeTrainer)

        if should_raise:
            with pytest.raises(ValueError) as exc_info:
                trainer._validate_args(*args, **kwargs)
            assert expected_error in str(exc_info.value)
        else:
            trainer._validate_args(*args, **kwargs)  # Should not raise

    def test__create_optimizer(
        self,
        splade_training_args: SpladeTrainingArguments,
        model: PreTrainedModel,
    ) -> None:
        args = splade_training_args

        dummy_eval_ds: Dataset = Dataset()
        trainer = SpladeTrainer(model=model, args=args, eval_dataset=dummy_eval_ds)
        result = trainer._create_optimizer(model=model, args=args)

        assert isinstance(result, AdamW)
        assert result.defaults["lr"] == args.learning_rate
        assert result.defaults["betas"] == (args.adam_beta1, args.adam_beta2)
        assert result.defaults["eps"] == args.adam_epsilon
        assert result.defaults["weight_decay"] == args.weight_decay

    @patch("light_splade.trainer.splade_trainer.get_linear_schedule_with_warmup")
    def test__create_scheduler(self, mock_get_schedule: Mock) -> None:
        trainer = SpladeTrainer.__new__(SpladeTrainer)
        mock_optimizer = Mock()
        mock_scheduler = Mock()
        mock_get_schedule.return_value = mock_scheduler

        args = Mock()
        args.warmup_steps = 1000
        args.max_steps = 10000

        result = trainer._create_scheduler(mock_optimizer, args)

        assert result == mock_scheduler
        mock_get_schedule.assert_called_once_with(
            optimizer=mock_optimizer,
            num_warmup_steps=1000,
            num_training_steps=10000,
        )

    @pytest.mark.parametrize(
        "training_loss, exp_loss_types",
        [
            ("kldiv,margin_mse", [DistillKLDivLoss, DistillMarginMSELoss]),
            ("kldiv", [DistillKLDivLoss]),
            ("margin_mse", [DistillMarginMSELoss]),
        ],
    )
    def test__create_losses(
        self,
        model: PreTrainedModel,
        splade_training_args: SpladeTrainingArguments,
        training_loss: str,
        exp_loss_types: list,
    ) -> None:
        splade_training_args.training_loss = training_loss

        trainer = SpladeTrainer(model=model, args=splade_training_args, eval_dataset=Dataset())
        result = trainer._create_losses(splade_training_args)

        assert len(result) == len(exp_loss_types)
        assert set([type(loss) for loss in result]) == set(exp_loss_types)

    @pytest.mark.parametrize(
        "reg_type, expected_type",
        [
            ("L1", L1),
            ("l1", L1),
            ("FLOPS", FLOPS),
            ("flops", FLOPS),
            ("unknown", FLOPS),  # Default to FLOPS
        ],
    )
    def test__create_regularizer(self, reg_type: str, expected_type: type) -> None:
        trainer = SpladeTrainer.__new__(SpladeTrainer)

        cfg = Mock()
        cfg.reg_type = reg_type

        result = trainer._create_regularizer(cfg)

        assert isinstance(result, expected_type)

    def test__init_regularizer(self) -> None:
        trainer = SpladeTrainer.__new__(SpladeTrainer)

        # Create mock config
        cfg = Mock()
        cfg.query.lambda_ = 0.1
        cfg.query.T = 100
        cfg.query.reg_type = "L1"
        cfg.doc.lambda_ = 0.2
        cfg.doc.T = 200
        cfg.doc.reg_type = "FLOPS"

        trainer._init_regularizer(cfg)

        assert isinstance(trainer.lambda_q_scheduler, RegularizerScheduler)
        assert isinstance(trainer.lambda_d_scheduler, RegularizerScheduler)
        assert isinstance(trainer.q_regularizer, L1)
        assert isinstance(trainer.d_regularizer, FLOPS)

    def test__reset_states(self) -> None:
        trainer = SpladeTrainer.__new__(SpladeTrainer)

        trainer._reset_states()

        assert isinstance(trainer.accum_loss_values, defaultdict)
        assert len(trainer.accum_loss_values) == 0

    def test__prepare_inputs(self) -> None:
        trainer = SpladeTrainer.__new__(SpladeTrainer)
        trainer.args = Mock()
        trainer.args.device = "cuda"

        mock_inputs = Mock()
        mock_inputs.to.return_value = "moved_inputs"

        result = trainer._prepare_inputs(mock_inputs)

        assert result == "moved_inputs"
        mock_inputs.to.assert_called_once_with("cuda")

    def test_get_lambdas(self) -> None:
        trainer = SpladeTrainer.__new__(SpladeTrainer)

        # Mock schedulers
        trainer.lambda_q_scheduler = Mock()
        trainer.lambda_d_scheduler = Mock()
        trainer.lambda_q_scheduler.get_lambda.return_value = 0.1
        trainer.lambda_d_scheduler.get_lambda.return_value = 0.2

        # Mock state
        trainer.state = Mock()
        trainer.state.global_step = 500

        lambda_q, lambda_d = trainer.get_lambdas()

        assert lambda_q == 0.1
        assert lambda_d == 0.2
        trainer.lambda_q_scheduler.set_step.assert_called_once_with(500)
        trainer.lambda_d_scheduler.set_step.assert_called_once_with(500)

    @patch("light_splade.trainer.splade_trainer.dot_product")
    def test__compute_loss(self, mock_dot_product: Mock) -> None:
        trainer = SpladeTrainer.__new__(SpladeTrainer)

        # Mock dependencies
        mock_loss_0 = Mock()
        mock_loss_0.return_value = torch.tensor(1.0)
        mock_loss_0.loss_type = "kldiv"
        mock_loss_1 = Mock()
        mock_loss_1.return_value = torch.tensor(2.0)
        mock_loss_1.loss_type = "margin_mse"
        trainer.losses = [mock_loss_0, mock_loss_1]

        trainer.get_lambdas = Mock(return_value=(0.1, 0.2))
        trainer.q_regularizer = Mock(return_value=torch.tensor(0.5))
        trainer.d_regularizer = Mock(return_value=torch.tensor(1.0))
        trainer.score_fn = Mock()

        mock_dot_product.side_effect = [
            torch.tensor([0.8]),
            torch.tensor([0.3]),
        ]

        # Mock model
        mock_model = Mock()
        pos_vectors = {
            "q_vector": torch.tensor([[1.0, 2.0]]),
            "d_vector": torch.tensor([[3.0, 4.0]]),
        }
        neg_vectors = {
            "q_vector": torch.tensor([[1.0, 2.0]]),
            "d_vector": torch.tensor([[5.0, 6.0]]),
        }
        mock_model.side_effect = [pos_vectors, neg_vectors]

        # Mock inputs
        mock_inputs = Mock()
        mock_inputs.teacher_pos_scores = torch.tensor([0.9])
        mock_inputs.teacher_neg_scores = torch.tensor([0.1])

        # Mock accum_loss_values
        accum_loss_values: dict = defaultdict(list)

        result = trainer._compute_loss(mock_model, mock_inputs, accum_loss_values)

        loss_value, outputs = result
        assert isinstance(loss_value, torch.Tensor)
        assert len(outputs) == 3  # pos_q_vector, pos_d_vector, neg_d_vector
        assert "kldiv_loss" in accum_loss_values
        assert "margin_mse_loss" in accum_loss_values
        assert "flops" in accum_loss_values

    def test_compute_loss_return_outputs_true(self) -> None:
        trainer = SpladeTrainer.__new__(SpladeTrainer)
        trainer.accum_loss_values = defaultdict(list)
        trainer._compute_loss = Mock(return_value=(torch.tensor(1.0), ("output1", "output2")))

        result = trainer.compute_loss(Mock(), Mock(), return_outputs=True)

        assert result == (torch.tensor(1.0), ("output1", "output2"))

    def test_compute_loss_return_outputs_false(self) -> None:
        trainer = SpladeTrainer.__new__(SpladeTrainer)
        trainer.accum_loss_values = defaultdict(list)
        trainer._compute_loss = Mock(return_value=(torch.tensor(1.0), ("output1", "output2")))

        result = trainer.compute_loss(Mock(), Mock(), return_outputs=False)

        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor(1.0))

    @patch("numpy.mean")
    def test_log(self, mock_np_mean: Mock) -> None:
        trainer = SpladeTrainer.__new__(SpladeTrainer)

        # Mock dependencies
        trainer.state = Mock()
        trainer.state.epoch = 1
        trainer.state.global_step = 100
        trainer.state.log_history = []
        trainer.callback_handler = Mock()
        trainer.callback_handler.on_log.return_value = Mock()
        trainer.args = Mock()
        trainer.control = Mock()

        trainer.accum_loss_values = defaultdict(list)
        trainer.accum_loss_values["kldiv_loss"] = [1.0, 1.5, 2.0]
        trainer.accum_loss_values["flops"] = [0.1, 0.2, 0.3]

        trainer.get_lambdas = Mock(return_value=(0.1, 0.2))
        mock_reset_states = Mock()
        trainer._reset_states = mock_reset_states

        mock_np_mean.side_effect = [1.5, 0.2]  # Mean values for losses

        logs = {"custom_metric": 0.95}
        trainer.log(logs)

        # Verify _reset_states was called
        mock_reset_states.assert_called_once()

        # Verify log history was updated
        assert len(trainer.state.log_history) == 1
        logged_entry = trainer.state.log_history[0]
        assert "step" in logged_entry
        assert "epoch" in logged_entry
        assert "lambda_q" in logged_entry
        assert "lambda_d" in logged_entry

    def test__load_from_checkpoint_invalid_model(self) -> None:
        trainer = SpladeTrainer.__new__(SpladeTrainer)
        trainer.model = Mock()  # Not a Splade model

        with pytest.raises(ValueError) as exc_info:
            trainer._load_from_checkpoint("checkpoint_path")

        assert "SpladeTrainer is used for" in str(exc_info.value)

    def test__load_from_checkpoint_valid_model(self) -> None:
        trainer = SpladeTrainer.__new__(SpladeTrainer)
        mock_splade_model = Mock(spec=Splade)
        trainer.model = mock_splade_model

        trainer._load_from_checkpoint("checkpoint_path")

        mock_splade_model.load.assert_called_once_with("checkpoint_path")

    @patch("os.makedirs")
    @patch("torch.save")
    def test__save_invalid_model(self, mock_torch_save: Mock, mock_makedirs: Mock) -> None:
        trainer = SpladeTrainer.__new__(SpladeTrainer)
        trainer.model = Mock()  # Not a Splade model

        with pytest.raises(ValueError) as exc_info:
            trainer._save("output_dir")

        assert "SpladeTrainer is used for" in str(exc_info.value)

    @patch("os.makedirs")
    @patch("torch.save")
    def test__save_valid_model(self, mock_torch_save: Mock, mock_makedirs: Mock) -> None:
        trainer = SpladeTrainer.__new__(SpladeTrainer)
        mock_splade_model = Mock(spec=Splade)
        trainer.model = mock_splade_model
        trainer.args = Mock()
        trainer.args.output_dir = "default_output"
        trainer.args.save_safetensors = True

        trainer._save("custom_output")

        mock_makedirs.assert_called_once_with("custom_output", exist_ok=True)
        mock_splade_model.save.assert_called_once_with("custom_output", save_safetensors=True)

    @patch("light_splade.trainer.splade_trainer.Evaluator")
    @patch("numpy.mean")
    def test_evaluate(self, mock_np_mean: Mock, mock_evaluator_class: Mock) -> None:
        trainer = SpladeTrainer.__new__(SpladeTrainer)

        # Mock dependencies
        trainer.args = Mock()
        trainer.args.per_device_eval_batch_size = 4
        trainer.args.device = "cuda"
        trainer.args.validation_metrics = ["ndcg@10"]

        trainer.eval_for_loss_dataset = Mock()
        trainer.eval_dataset = Mock()
        trainer.data_collator = Mock()
        trainer.model = Mock()
        trainer._memory_tracker = Mock()

        trainer.get_eval_dataloader = Mock(
            return_value=[
                Mock(),  # batch 1
                Mock(),  # batch 2
            ]
        )
        mock_log = Mock()
        trainer.log = mock_log
        trainer._compute_loss = Mock(side_effect=[(torch.tensor(1.0), None), (torch.tensor(1.5), None)])

        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = {"ndcg@10": 0.85}
        mock_evaluator_class.return_value = mock_evaluator

        mock_np_mean.return_value = 1.25

        result = trainer.evaluate()

        assert "eval_loss" in result
        assert "eval_ndcg@10" in result
        assert result["eval_ndcg@10"] == 0.85
        mock_log.assert_called_once()
