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
from logging import getLogger

from .base import JSONSerializableMixin


logger = getLogger(__name__)


@dataclass(kw_only=True)
class BaseConfigCrossEncoder:
    char_per_token_ratio: float = field(
        default=1.8,
        metadata={"help": "Average number of chars per token."},
    )

    max_token_len: int = field(
        default=512,
        metadata={
            "help": "limit num of tokens for each pair (`query`, `doc`) and several special tokens. "
            "Note that this limit must be less than or equal to model `max_position_embeddings`"
        },
    )

    max_len: int = field(
        init=False,
        metadata={
            "help": "max length of the input sequence. This is automatically calculated as "
            "`max_token_len * char_per_token_ratio`."
        },
    )

    max_query_len: float | int = field(
        default=0.33,
        metadata={
            "help": "Limits the query length when concatenating it with the document. "
            "If an integer is provided, the query length is limited by the specified number of characters. "
            "If a float is provided, the query length is limited to the given ratio of max_len (in characters). "
            "If set to 0, the query length is not limited; in this case, samples with long queries may "
            "contain little or no document text. "
        },
    )

    def __post_init__(self) -> None:
        self.max_len = int(self.max_token_len * self.char_per_token_ratio)
        if isinstance(self.max_query_len, float):
            if self.max_query_len < 0.0 or self.max_query_len >= 1.0:
                raise ValueError("If float, max_query_len must be in the range 0.0 ~ 1.0!")
            self.max_query_len = int(self.max_query_len * self.max_len)

        if self.max_query_len > self.max_len or self.max_query_len == 0:
            logger.warning(
                f"max_query_len is no limit OR max_query_len is over max_len = {self.max_len} -> adjust max_query_len to {self.max_len}. "
                "Samples with long queries may contain little or no document text."
            )
            self.max_query_len = self.max_len
