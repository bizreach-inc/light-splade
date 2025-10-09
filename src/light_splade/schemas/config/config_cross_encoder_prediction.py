from dataclasses import dataclass
from dataclasses import field

from .base import JSONSerializableMixin


# Config class for CrossEncoder Training
@dataclass
class ConfigCrossEncoderPrediction(JSONSerializableMixin):
    model_path: str = field(
        metadata={"help": "path to save the cross-encoder model to be used for prediction."},
    )

    train_doc_master: str = field(metadata={"help": "Path to training doc master file"})

    train_query_master: str = field(metadata={"help": "Path to training query master file"})

    validation_doc_master: str = field(metadata={"help": "Path to validation doc master file"})

    validation_query_master: str = field(metadata={"help": "Path to validation query master file"})

    hard_negative_init_scores: str = field(
        metadata={
            "help": "Path to hard negative file storing query-doc pairs with initial scores. The prediction script "
            "will use query-doc pairs in this file and predict the similarity scores for the pairs."
        }
    )

    hard_negative_cross_encoder_scores: str = field(
        metadata={
            "help": "Path to hard negative scores file. The prediction script will output query-doc pairs along with "
            "predicted similarity scores to this file."
        }
    )

    max_token_len: int = field(
        default=512,
        metadata={
            "help": "limit num of tokens for samples, including `query`, `doc` and several special tokens. "
            "Note that this limit must be less than or equal to model `max_position_embeddings`"
        },
    )

    char_per_token_ratio: float = field(
        default=1.8,
        metadata={"help": "path to save the output model and other training result."},
    )

    predict_batch_size: int = field(
        default=32,
        metadata={"help": "prediction batch size"},
    )

    max_len: int = field(
        default=0,
        metadata={
            "help": "max length of the input sequence. "
            "This is automatically calculated as `max_token_len * char_per_token_ratio`."
        },
    )

    def __post_init__(self) -> None:
        self.max_len = int(self.max_token_len * self.char_per_token_ratio)
