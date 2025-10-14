from dataclasses import dataclass
from dataclasses import field

from transformers import TrainingArguments

from .base import JSONSerializableMixin


@dataclass
class RegularizerConfig(JSONSerializableMixin):
    reg_type: str
    lambda_: float
    T: int


@dataclass
class SpladeRegularizerConfig(JSONSerializableMixin):
    doc: RegularizerConfig
    query: RegularizerConfig


@dataclass
class SpladeTrainingArguments(JSONSerializableMixin, TrainingArguments):
    """
    NOTE: all argument objects inside SpladeTrainingArguments must enable `to_dict` method to avoid
    `not JSON serializable` error.
    """

    final_output_dir: str | None = field(
        default=None,
        metadata={"help": ("The output directory where the final trained model will be written.")},
    )

    training_loss: str = field(
        default="in_batch_negatives",
        metadata=dict(
            help="Training loss function names separated by `,`. Currently support {`kldiv`, `margin_mse`} for "
            "distillation mode, and {`in_batch_negatives`, `pairwise_contrastive`} for triplet mode."
        ),
    )

    regularizers: SpladeRegularizerConfig | None = field(
        default=None,
        metadata=dict(help="FLOPS style regularizers on `doc` & `query`"),
    )

    validation_metrics: list[str] = field(
        default_factory=lambda: ["MRR@10", "recall@10"],
        metadata=dict(help="Specify the metrics to be computed on validation data"),
    )
