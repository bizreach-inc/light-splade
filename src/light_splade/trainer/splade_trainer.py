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

"""Core training logic for SPLADE (distillation variant).

This module defines `SpladeTrainer`, a thin specialization of Hugging Face's `Trainer` tailored for SPLADE-style sparse
retrievers with optional distillation losses. It provides:

* Multiple configurable distillation losses (KLDiv, Margin MSE).
* Regularization schedules (e.g., FLOPS / L1 over query and doc towers).
* A two-tower forward pass over (query, positive, negative).
* Aggregation and logging of per-loss values and regularizer contributions.
* Retrieval-style evaluation (loss + metrics via `Evaluator`).

The class intentionally preserves most standard `Trainer` semantics while restricting usage to keyword arguments only
for clarity in experiment scripts.
"""

import os
from collections import defaultdict
from logging import getLogger
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer import Trainer

from light_splade.data import TripletDistilDataset
from light_splade.evaluator import Evaluator
from light_splade.losses import DistillKLDivLoss
from light_splade.losses import DistillMarginMSELoss
from light_splade.models import Splade
from light_splade.regularizer import FLOPS
from light_splade.regularizer import L1
from light_splade.regularizer import RegularizerScheduler
from light_splade.schemas.config import RegularizerConfig
from light_splade.schemas.config import SpladeRegularizerConfig
from light_splade.schemas.config import SpladeTrainingArguments
from light_splade.schemas.model import TripletDistilBatch
from light_splade.utils.scoring import dot_product

logger = getLogger(__name__)


# add more splade model class here for type checking in the SpladeTrainer
SUPPORTED_SPLADE_MODELS = (Splade,)
SUPPORTED_LOSSES = {loss_cls.loss_type: loss_cls for loss_cls in [DistillKLDivLoss, DistillMarginMSELoss]}  # type: ignore[attr-defined]


class SpladeTrainer(Trainer):
    """Trainer for SPLADE + distillation.

    Responsibilities:
        * Enforce keyword-only construction (simplifies Hydra / script integration).
        * Build an optimizer and scheduler if not supplied.
        * Instantiate requested distillation losses from a comma-separated spec.
        * Manage regularizers with time-based (step) schedulers.
        * Compute the combined loss = sum(losses) + regularization.
        * Accumulate loss components for logging and evaluation.
        * Provide retrieval-style evaluation (loss + metrics@K) via `Evaluator`.

    Expected batches (`inputs`) are `TripletDistilBatch` instances containing
    tokenized query / positive / negative examples plus teacher scores when
    distilling.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the trainer.

        Notes:
            * This class enforces keyword-only construction; `args` must be empty.
            * See the Hugging Face `Trainer` specification for supported keyword arguments:
              https://huggingface.co/docs/transformers/v4.56.1/en/main_classes/trainer

        Args:
            *args: Must be empty (enforced); present only for API compatibility.
            **kwargs: Standard `Trainer` keyword arguments. Required keys:
                - model: A `Splade` instance.
                - args: `SpladeTrainingArguments` / HF `TrainingArguments` derivative.
                - eval_for_loss_dataset (optional): Dataset solely for loss evaluation.
                - optimizers (optional): (optimizer, scheduler) tuple to override defaults.
        """

        self._validate_args(*args, **kwargs)

        training_args = kwargs["args"]
        model: Splade = kwargs["model"]

        self.eval_for_loss_dataset = kwargs.pop("eval_for_loss_dataset", None)

        # init optimizer and scheduler
        if kwargs.get("optimizers", None) is None:
            optimizer = self._create_optimizer(model, training_args)
            scheduler = self._create_scheduler(optimizer, training_args)
            kwargs["optimizers"] = (optimizer, scheduler)

        self.losses = self._create_losses(training_args)
        self._init_regularizer(training_args.regularizers)
        self.score_fn = dot_product  # currently, fixed
        self._reset_states()

        # the kwargs contains `args: TrainingArguments` param, which is assigned to self by Trainer class
        super().__init__(*args, **kwargs)

        # to ensure that the `num_items_in_batch` param passed to compute_loss() method is always None.
        # Otherwise, the loss might be slightly incorrect when using gradient accumulation.
        self.model_accepts_loss_kwargs = False

    def _validate_args(self, *args: Any, **kwargs: Any) -> None:
        """Validate construction contract (keyword-only & required keys)."""
        if len(args) > 0:
            raise ValueError("Arguments to SpladeTrainer must be keyword arguments!")
        if kwargs.get("model") is None:
            raise ValueError("`model` is not passed to SpladeTrainer!")
        if kwargs.get("args") is None:
            raise ValueError("`args` is not passed to SpladeTrainer!")

    def _create_optimizer(self, model: nn.Module, args: SpladeTrainingArguments) -> Optimizer:
        """Create an optimizer using training arguments hyperparameters.

        Args:
            model: Model whose parameters will be optimized.
            args (SpladeTrainingArguments): Training arguments carrying optimizer hyperparameters.

        Returns:
            An :class:`torch.optim.Optimizer` instance.
        """
        return AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )

    def _create_scheduler(self, optimizer: Optimizer, args: SpladeTrainingArguments) -> LambdaLR:
        """Create linear warmup schedule matching total max steps.

        Args:
            optimizer: Optimizer to attach the scheduler to.
            args (SpladeTrainingArguments): Training arguments containing warmup and max step counts.

        Returns:
            A :class:`transformers.get_linear_schedule_with_warmup` scheduler.
        """
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_steps,
        )

    def _create_losses(self, args: SpladeTrainingArguments) -> list[DistillKLDivLoss | DistillMarginMSELoss]:
        """Instantiate requested distillation losses.

        The ``training_loss`` field may contain a comma-separated list (e.g., ``"kldiv,margin_mse"``). Order affects
        only summation order.

        Args:
            args: Training arguments containing ``training_loss`` specification.

        Returns:
            A list of instantiated loss callables.
        """
        for loss_type in args.training_loss.split(","):
            if loss_type not in SUPPORTED_LOSSES:
                raise ValueError(
                    f"Unsupported training loss: {loss_type}. Only {list(SUPPORTED_LOSSES.keys())} is supported."
                )

        losses: list[DistillKLDivLoss | DistillMarginMSELoss] = []
        if "kldiv" in args.training_loss:
            losses.append(DistillKLDivLoss())
        if "margin_mse" in args.training_loss:
            losses.append(DistillMarginMSELoss())
        return losses

    def _create_regularizer(self, cfg: RegularizerConfig) -> FLOPS | L1:
        """Factory for a regularizer based on config type (L1 or FLOPS).

        Args:
            cfg: RegularizerConfig specifying the regularizer type.

        Returns:
            An instance of :class:`light_splade.regularizer.L1` or :class:`light_splade.regularizer.FLOPS`.
        """
        if cfg.reg_type.upper() == "L1":
            return L1()
        else:
            return FLOPS()

    def _init_regularizer(self, cfg: SpladeRegularizerConfig) -> None:
        """Initialize regularizers and their schedulers for query and doc towers."""
        self.lambda_q_scheduler = RegularizerScheduler(cfg.query.lambda_, cfg.query.T)
        self.lambda_d_scheduler = RegularizerScheduler(cfg.doc.lambda_, cfg.doc.T)

        self.q_regularizer: FLOPS | L1 = self._create_regularizer(cfg.query)
        self.d_regularizer: FLOPS | L1 = self._create_regularizer(cfg.doc)

    def _reset_states(self) -> None:
        """Reset per-logging-window accumulators for losses and metrics."""
        self.accum_loss_values: dict = defaultdict(list)

    def _prepare_inputs(self, inputs: TripletDistilBatch) -> TripletDistilBatch:
        """Move the batch to the target device (delegated by HF Trainer).
        Args:
            inputs (TripletDistilBatch): containing tokenized tensors.

        Returns:
            The same batch moved to ``self.args.device``.
        """
        return inputs.to(self.args.device)

    def evaluate(
        self,
        # eval_dataset: Dataset | dict[str, Dataset] | None = None,
        eval_dataset: TripletDistilDataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Run evaluation: compute mean validation loss and retrieval metrics.

        Args:
            eval_dataset: Optional evaluation dataset. If None, ``self.eval_dataset``
                will be used.
            ignore_keys: Unused, kept for signature compatibility with HF Trainer.
            metric_key_prefix: Prefix used for returned metric keys.

        Returns:
            Dictionary mapping metric names (prefixed) to float values. Typical
            keys include ``{prefix}_loss`` and retrieval metrics like ``{prefix}_ndcg@10``.
        """
        logger.info("--- Evaluating ...")

        # run inference on validation set, evaluate validation_loss & validation_metrics, then report results into
        # metrics dict to write log
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        metrics = dict()

        if hasattr(self._memory_tracker, "start"):
            self._memory_tracker.start()
        eval_for_loss_dataloader = self.get_eval_dataloader(self.eval_for_loss_dataset)

        model = self.model
        model.eval()

        # compute evaluation loss
        logger.info("Computing evaluation loss...")
        loss_values = []
        with torch.inference_mode():
            for step, inputs in tqdm(enumerate(eval_for_loss_dataloader)):
                output = self._compute_loss(model, inputs)
                loss_value = output[0].detach().cpu().item()
                loss_values.append(loss_value)

        loss_value = float(np.mean(loss_values))
        metrics[f"{metric_key_prefix}_loss"] = loss_value
        logger.info(f"{metric_key_prefix}_loss={loss_value}")

        # run retrieval on validation set to compute metric@K
        evaluator = Evaluator(
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            model=model,
            batch_size=self.args.per_device_eval_batch_size,
            device=self.args.device,
        )
        validation_metrics = getattr(self.args, "validation_metrics", [])
        for k, v in evaluator.evaluate(validation_metrics).items():
            metrics[f"{metric_key_prefix}_{k}"] = v

        logger.info("Evaluation done.")
        self.log(metrics)
        return metrics

    def get_lambdas(self) -> tuple[float, float]:
        """Return the current (lambda_q, lambda_d) after advancing schedulers.

        The method advances both internal schedulers to the current training global step before returning values.
        """
        self.lambda_q_scheduler.set_step(self.state.global_step)
        self.lambda_d_scheduler.set_step(self.state.global_step)
        lambda_q = self.lambda_q_scheduler.get_lambda()
        lambda_d = self.lambda_d_scheduler.get_lambda()
        return lambda_q, lambda_d

    def _compute_loss(
        self,
        model: Splade,
        inputs: TripletDistilBatch,
        accum_loss_values: dict | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """Forward pass and loss computation for a distillation triplet batch.

        Args:
            model (Splade): The model instance used for encoding.
            inputs (TripletDistilBatch): containing query / pos / neg tensors.
            accum_loss_values (dict | None): Optional dict accumulator to store per-component losses.

        Returns:
            A tuple ``(total_loss, (q_vec, pos_vec, neg_vec))`` where ``total_loss``
            is a scalar tensor and the second element contains dense vectors.
        """
        pos_vectors: dict = model(
            q_input_ids=inputs.q_input_ids,
            q_attention_mask=inputs.q_attention_mask,
            d_input_ids=inputs.pos_input_ids,
            d_attention_mask=inputs.pos_attention_mask,
        )
        # ignore the q_vector computation from neg_vectors
        neg_vectors: dict = model(
            d_input_ids=inputs.neg_input_ids,
            d_attention_mask=inputs.neg_attention_mask,
        )
        q_vector = pos_vectors["q_vector"]  # (bs, V)
        pos_d_vector = pos_vectors["d_vector"]  # (bs, V)
        neg_d_vector = neg_vectors["d_vector"]  # (bs, V)
        # NOTE: pos_vectors["q_vector"] is the same as neg_vectors["q_vector"]
        # so, q_vector = pos_q_vector = neg_q_vector

        params = dict(
            pos_score=self.score_fn(q_vector, pos_d_vector),
            neg_score=self.score_fn(q_vector, neg_d_vector),
            teacher_pos_score=inputs.teacher_pos_scores,
            teacher_neg_score=inputs.teacher_neg_scores,
        )

        loss_value: torch.Tensor = torch.tensor(0.0)
        for loss in self.losses:
            loss_value_ = loss(params)
            loss_value = loss_value + loss_value_
            if accum_loss_values is not None:
                accum_loss_values[f"{loss.loss_type}_loss"].append(loss_value_.detach().cpu().item())

        # regularizers
        lambda_q, lambda_d = self.get_lambdas()
        reg_loss = (
            lambda_q * self.q_regularizer(q_vector)
            + lambda_d * self.d_regularizer(pos_d_vector) * 0.5
            + lambda_d * self.d_regularizer(neg_d_vector) * 0.5
        )
        loss_value = loss_value + reg_loss
        if accum_loss_values is not None:
            accum_loss_values["flops"].append(reg_loss.detach().cpu().item())

        return (loss_value, (q_vector, pos_d_vector, neg_d_vector))

    def compute_loss(
        self,
        model: nn.Module,
        inputs: TripletDistilBatch,
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple:
        """HF Trainer hook — delegate to `_compute_loss`.

        Args:
            model (nn.Module): Ignored; kept for HF Trainer method signature compatibility.
            inputs (TripletDistilBatch): Batch passed to :meth:`_compute_loss`.
            return_outputs (bool): If True, return ``(loss, outputs)`` otherwise just ``loss``.
            num_items_in_batch: Ignored; retained for signature compatibility.
        """
        output = self._compute_loss(model, inputs, self.accum_loss_values)
        if return_outputs:
            return output
        else:
            return output[0]

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Aggregate buffered loss components, append lambda values, and record the step.

        Args:
            logs (dict[str, float]): Mapping of metric keys to their float values to be recorded.
            start_time (float | None): Optional timestamp for the logging window; unused here.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch

        for loss_id, loss_values in self.accum_loss_values.items():
            logs[loss_id] = float(np.mean(loss_values))
        self._reset_states()

        # record lambda values.
        # NOTE: this recording also executed while evaluating.
        lambda_q, lambda_d = self.get_lambdas()
        logs["lambda_q"] = lambda_q
        logs["lambda_d"] = lambda_d

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control: Any = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _load_from_checkpoint(
        self,
        resume_from_checkpoint: str,
        model: Splade | None = None,
    ) -> None:
        """Load model weights from a SPLADE checkpoint path.

        The method expects the model to implement a ``load(path)`` method that restores internal components.
        """
        logger.info(f"Loading model from checkpoint {resume_from_checkpoint}")

        if model is None:
            model = self.model

        # Ensure that this trainer is used for Splade models only
        # NOTE: need to check whether wrapped model (DeepSpeed, DistributedDataParallel...) can be supported or not
        if not isinstance(self.model, SUPPORTED_SPLADE_MODELS):
            raise ValueError(f"The {self.__class__.__name__} is used for {SUPPORTED_SPLADE_MODELS} models only.")

        if model is not None:
            if hasattr(model, "load"):
                model.load(resume_from_checkpoint)

    def _save(
        self,
        output_dir: str | None = None,
        state_dict: dict | None = None,
    ) -> None:
        """Persist the SPLADE model and training arguments.

        Overrides `Trainer._save()` to handle the two-tower SPLADE model.
        Reference: https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/trainer.py#L4285

        Mirrors HF `Trainer` save semantics while delegating the two-tower serialization to the model's `save`
        implementation. The `state_dict` argument is ignored (the model handles its own internals).

        Args:
            output_dir (str | None): Directory to write checkpoint files. If ``None``, uses ``self.args.output_dir``.
            state_dict (dict | None): Ignored; the model handles its own serialization.
        """

        # Ensure that this trainer is used for Splade models only
        # NOTE: need to check whether wrapped model (DeepSpeed, DistributedDataParallel...) works fine or not.
        if not isinstance(self.model, SUPPORTED_SPLADE_MODELS):
            raise ValueError(f"The {self.__class__.__name__} is used for {SUPPORTED_SPLADE_MODELS} models only.")

        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if self.model is not None:
            if output_dir is not None:
                save_safetensors = getattr(self.args, "save_safetensors", False)
                self.model.save(output_dir, save_safetensors=save_safetensors)

        # Good practice: save your training arguments together with the trained model
        if output_dir is not None:
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
