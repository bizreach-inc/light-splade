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
