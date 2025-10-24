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

from dataclasses import fields
from dataclasses import is_dataclass
from enum import Enum
from typing import Any

from transformers.utils import is_accelerate_available

if is_accelerate_available():
    from transformers.trainer_pt_utils import AcceleratorConfig


class JSONSerializableMixin:
    def _dict_torch_dtype_to_str(self, d: dict[str, Any]) -> None:
        """
        Copied and customized transformers' TrainingArguments.to_dict:
        https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/training_args.py#L2432

        Checks whether the passed dictionary and its nested dicts have a
        *torch_dtype* key and if it's not None, converts torch.dtype to a
        string of just the type. For example, `torch.float32` get converted
        into *"float32"* string, which can then be stored in the json format.
        """
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self._dict_torch_dtype_to_str(value)

    def to_dict(self) -> dict:
        """
        Copied and customized transformers' TrainingArguments.to_dict:
        https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/training_args.py#L2444

        Serializes this instance while replace `Enum` by their values (for JSON
        serialization support). It obfuscates the token values by removing
        their value.
        """
        if not is_dataclass(self):
            raise TypeError((f"JSONSerializableMixin is used in a dataclass, got {type(self).__name__} instead."))

        # filter out fields that are defined as field(init=False)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
            # Handle the accelerator_config if passed
            if is_accelerate_available() and isinstance(v, AcceleratorConfig):
                d[k] = v.to_dict()
            if is_dataclass(v):
                d[k] = v.to_dict()
        self._dict_torch_dtype_to_str(d)

        return d
