from dataclasses import dataclass
from dataclasses import field

from .base import JSONSerializableMixin


@dataclass
class DataTrainingArguments(JSONSerializableMixin):
    """
    Arguments pertaining to what data we are going to input our model for
    training and eval.
    """

    train_doc_master: str = field(metadata={"help": "Path to training doc master file"})
    train_query_master: str = field(metadata={"help": "Path to training query master file"})
    train_positives: str = field(metadata={"help": "Path to training positive list file"})
    eval_loss_size: int = field(
        metadata={
            "help": "Number of samples to use for computing evaluation loss. "
            "These samples are randomly selected from training set. 0 for ignoring eval_loss computation."
        }
    )

    validation_doc_master: str = field(metadata={"help": "Path to validation doc master file"})
    validation_query_master: str = field(metadata={"help": "Path to validation query master file"})
    validation_positives: str = field(metadata={"help": "Path to validation positive list file"})


@dataclass
class DataTrainingDistilArguments(DataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    hard_negative_scores: str = field(metadata={"help": "Path to hard negative scores file"})


@dataclass
class DataTrainingTripletArguments(DataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_triplets: str = field(metadata={"help": "Path to training triplet file"})
    # `validation_triplets` does not exist because we do not need it for validation
