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

"""Base collator utilities for tokenization and batching.

This module provides :class:`BaseSpladeCollator`, a lightweight wrapper around
:class:`DefaultDataCollator` that initializes a tokenizer and exposes a
``tokenize`` helper used by concrete collators; the collator centralizes
arguments such as padding, truncation and the returned tensor types.
"""

from typing import Any
from typing import Sequence

from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class BaseSpladeCollator(DefaultDataCollator):
    """Provide a base collator interface used for SPLADE.

    The class initializes a tokenizer from a pretrained model and exposes a :meth:`tokenize` helper that returns a
    PyTorch-friendly :class:`BatchEncoding` with padded input ids and attention masks.

    Args:
        tokenizer_path (str): Path or identifier of the pretrained tokenizer.
        max_length (int): Maximum token length used for truncation (default: 512).
    """

    def __init__(
        self,
        tokenizer_path: str,
        max_length: int = 512,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.max_length = max_length

    # TODO: fixme (putting a tokenize method in the collator is not a good idea)
    def tokenize(self, text_list: Sequence[str]) -> BatchEncoding:
        """Tokenize a sequence of texts into a BatchEncoding.

        The returned :class:`BatchEncoding` contains ``input_ids`` and
        ``attention_mask`` tensors suitable for model input.

        Args:
            text_list (Sequence[str]): Sequence of strings to tokenize.

        Returns:
            A :class:`BatchEncoding` with tokenized tensors.
        """
        outputs: BatchEncoding = self.tokenizer(
            text_list,
            add_special_tokens=True,
            padding="longest",
            truncation="longest_first",
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return outputs

    def __call__(self, features: Any, return_tensors: Any = None) -> Any:
        """Delegate to :meth:`collate_fn` implemented by subclasses."""
        return self.collate_fn(features)

    def collate_fn(self, features: Any) -> Any:
        """Collate a list of features into a batch.

        Concrete collators must implement this method and return the appropriate batch type (often a Pydantic model or
        a dataclass used by the training loop).
        """
        raise NotImplementedError("collate_fn method should be implemented in the derived class.")
