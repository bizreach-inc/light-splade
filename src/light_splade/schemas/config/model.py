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


@dataclass
class ModelArguments(JSONSerializableMixin):
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from "
            "huggingface.co/models for document encoder only or both"
        }
    )

    model_name_or_path_q: str | None = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models for query encoder. "
            "If None, the query encoder will be initialized from the document encoder. "
            "NOTE that if query encoder is different from document encoder, the two encoders must use the same "
            "tokenizer. Default is None."
        },
    )

    freeze_d_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to freeze the document encoder model. If True, the document encoder will not be trained. "
            "Default is False."
        },
    )

    agg: str = field(
        default="max",
        metadata={"help": "Aggregation function for pooling strategy. Must be `max` or `sum`. Default is `max`."},
    )

    max_length: int = field(
        default=512,
        metadata={"help": ("Maximum length of the input sequence for both document and query. Default is 512.")},
    )
