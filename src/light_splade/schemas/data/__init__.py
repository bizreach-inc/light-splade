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
