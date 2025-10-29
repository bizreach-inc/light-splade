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

from .data_training import DataTrainingDistilArguments
from .data_training import DataTrainingTripletArguments
from .model import ModelArguments
from .training import SpladeTrainingArguments


# Config class for Splade Training
@dataclass
class ConfigSpladeDistil:
    data: DataTrainingDistilArguments
    model: ModelArguments
    training: SpladeTrainingArguments


@dataclass
class ConfigSpladeTriplet:
    data: DataTrainingTripletArguments
    model: ModelArguments
    training: SpladeTrainingArguments
