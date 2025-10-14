from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TripletBatch(dict):
    q_input_ids: torch.Tensor
    q_attention_mask: torch.Tensor
    pos_input_ids: torch.Tensor
    pos_attention_mask: torch.Tensor
    neg_input_ids: torch.Tensor
    neg_attention_mask: torch.Tensor

    def __len__(self) -> int:
        return len(self.__dict__.items())

    def to(self, device: torch.device) -> "TripletBatch":
        for key in self.__dict__:
            self.__dict__[key] = self.__dict__[key].to(device)
        return self

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]


@dataclass
class TripletDistilBatch(dict):
    q_input_ids: torch.Tensor
    q_attention_mask: torch.Tensor
    pos_input_ids: torch.Tensor
    pos_attention_mask: torch.Tensor
    neg_input_ids: torch.Tensor
    neg_attention_mask: torch.Tensor
    teacher_pos_scores: torch.Tensor
    teacher_neg_scores: torch.Tensor

    def __len__(self) -> int:
        return len(self.__dict__.items())

    def to(self, device: torch.device) -> "TripletDistilBatch":
        for key in self.__dict__:
            self.__dict__[key] = self.__dict__[key].to(device)
        return self

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]
