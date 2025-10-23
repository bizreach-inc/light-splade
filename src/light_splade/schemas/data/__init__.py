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

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass


@dataclass
class BaseMasterSchema(ABC):
    text: str

    @property
    @abstractmethod
    def id(self) -> int:
        raise NotImplementedError("Property id is not implemented yet!")


@dataclass
class QueryMasterSchema(BaseMasterSchema):
    qid: int

    @property
    def id(self) -> int:
        return self.qid


@dataclass
class DocumentMasterSchema(BaseMasterSchema):
    doc_id: int

    @property
    def id(self) -> int:
        return self.doc_id


@dataclass
class PositiveListSchema:
    qid: int
    positive_doc_ids: list[int]


@dataclass
class HardNegativeScoreSchema:
    qid: int
    scores: dict[int, float]


@dataclass
class TripletSchema:
    qid: int
    pos_doc_id: int
    neg_doc_id: int
