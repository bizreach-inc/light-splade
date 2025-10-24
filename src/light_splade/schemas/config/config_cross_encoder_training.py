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

from dataclasses import dataclass
from dataclasses import field

from .base import JSONSerializableMixin


# Config class for CrossEncoder Training
@dataclass
class ConfigCrossEncoderTraining(JSONSerializableMixin):
    input_file: str = field(
        metadata={"help": "path to input file. (e.g., `data/cross-encoder-train-18M_pairs.pkl.gz`)"},
    )

    output_path: str = field(
        metadata={"help": "path to output folder. (e.g., `model/bert-cross-encoder`)"},
    )

    final_output_path: str | None = field(
        metadata={"help": "The output directory where the final trained model will be written."},
    )

    model_path: str = field(
        metadata={"help": "path to save the output model and other training result."},
    )

    char_per_token_ratio: float = field(
        default=1.8,
        metadata={"help": "path to save the output model and other training result."},
    )

    max_train_size: int = field(
        default=0,
        metadata={"help": "limit num of training samples. 0 for no limit."},
    )

    max_eval_size: int = field(
        default=100,
        metadata={"help": "limit num of validation samples. 0 for no limit."},
    )

    max_token_len: int = field(
        default=512,
        metadata={
            "help": "limit num of tokens for samples, including `query`, `doc` and several special tokens. "
            "Note that this limit must be less than or equal to model `max_position_embeddings`"
        },
    )

    num_epochs: int = field(
        default=1,
        metadata={"help": "num of epoch to train"},
    )

    train_batch_size: int = field(
        default=32,
        metadata={"help": "training batch size"},
    )

    eval_batch_size: int = field(
        default=64,
        metadata={"help": "evaluation batch size"},
    )

    evaluation_steps: int = field(
        default=1000,
        metadata={"help": "run evaluation for every training steps"},
    )

    warmup_steps: int = field(
        default=5000,
        metadata={"help": "initial warmup steps for the learning rate"},
    )

    loss_function: str | None = field(
        default=None,
        metadata={
            "help": "loss function name. Currently support {`MSELoss`, "
            "`BCEWithLogitsLoss`, `CrossEntropyLoss`, `L1Loss`}"
        },
    )

    max_len: int = field(
        default=0,
        metadata={
            "help": "max length of the input sequence. This is automatically calculated as "
            "`max_token_len * char_per_token_ratio`."
        },
    )

    seed: int = field(
        default=42,
        metadata={"help": "random seed for reproducibility."},
    )

    def __post_init__(self) -> None:
        self.max_len = int(self.max_token_len * self.char_per_token_ratio)
