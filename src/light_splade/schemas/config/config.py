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
