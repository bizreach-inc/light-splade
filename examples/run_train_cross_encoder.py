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

"""
Train Cross-Encoder model for Japanese, using POSITIVE/NEGATIVE (Query, Doc) pairs dataset. This script refer to the
script `train_cross-encoder_scratch.py` in [1] from sentence-transformers.

INPUT
    - `config_file`: config file in YAML format
        (e.g., `src/config/cross_encoder_train.yml`)

OUTPUT
    - Output the trained model & other training results to the folder defined
    in config `final_output_path` key. If the `final_output_path` key is not
    defined, folder "`output_path`/final" is used instead.

REFERENCES:
[1] https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_cross-encoder_scratch.py
"""  # noqa

import gzip
import json
import os
import pickle
import random
from collections import Counter
from logging import basicConfig
from logging import getLogger
from pathlib import Path
from time import time

import transformers
import typer
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from light_splade.schemas.config import ConfigCrossEncoderTraining
from light_splade.utils.io import load_yaml
from light_splade.utils.model import contiguous
from light_splade.utils.random import set_seeds

LOSS_FUNCTION_MAP = {
    "MSELoss": nn.MSELoss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "L1Loss": nn.L1Loss,
}

basicConfig(level="INFO", format="%(asctime)s : %(levelname)s : %(name)s : %(message)s")
logger = getLogger(__name__)


def main(
    config_file: Path = typer.Option(Path("config/cross_encoder_train.yaml"), help="Path to the config file"),
) -> None:
    logger.info("=== STARTED cross-encoder training from run_train_cross_encoder.py")

    start_time = time()
    transformers.logging.set_verbosity_error()

    cfg = ConfigCrossEncoderTraining(**load_yaml(config_file))
    logger.info("cfg=" + json.dumps(cfg.to_dict(), indent=4))

    set_seeds(cfg.seed)

    final_path = cfg.final_output_path or os.path.join(cfg.output_path, "final")
    loss_function_name = cfg.loss_function or "contrastive"
    loss_fct = LOSS_FUNCTION_MAP.get(loss_function_name)

    # Load data
    logger.info("Loading training data...")
    with gzip.open(cfg.input_file, "rb") as f:
        (eval_samples, raw_train_samples) = pickle.load(f)

    # limit validation set
    if cfg.max_eval_size > 0:
        eval_samples = eval_samples[: cfg.max_eval_size]

    logger.info(f"eval_samples={len(eval_samples)}")
    logger.info(f"raw_train_samples={len(raw_train_samples)}")

    # Prepare training data
    logger.info("Filtering training samples by text length...")
    train_samples: list[InputExample] = []
    for query, passage, label in tqdm(raw_train_samples):
        data_len = len(query) + len(passage)
        if data_len > cfg.max_len:
            continue
        train_samples.append(InputExample(texts=[query, passage], label=label))
    logger.info(f"{len(train_samples)=}")

    if cfg.max_train_size > 0 and len(train_samples) >= cfg.max_train_size:
        random.shuffle(train_samples)
        train_samples = train_samples[: cfg.max_train_size]
        logger.info(f"After resampling with max_train_size={cfg.max_train_size}: {len(train_samples)=}")
    counter = Counter([sample.label for sample in train_samples])
    logger.info(f"Final: {len(train_samples)=}")
    logger.info(f"Ratios of labels: {counter}")

    # Cast to Dataset to satisfy type checker
    train_dataloader: DataLoader = DataLoader(train_samples, shuffle=True, batch_size=cfg.train_batch_size)

    evaluator = CrossEncoderRerankingEvaluator(eval_samples, name="mmarco-ja-eval", batch_size=cfg.eval_batch_size)

    model = CrossEncoder(
        cfg.model_path,
        num_labels=1,
        max_length=cfg.max_token_len,
    )
    # because we cannot control the `save_safetensors` argument in the
    # `CrossEncoder` class, rearrange the tensors to be contiguous in memory
    # which is required when saving the model with `safetensors` package.
    model = contiguous(model)

    # Train the model
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=cfg.num_epochs,
        loss_fct=loss_fct,
        evaluation_steps=cfg.evaluation_steps,
        warmup_steps=cfg.warmup_steps,
        output_path=cfg.output_path,
        use_amp=True,
    )

    # Save latest model
    # output_path: store the model which gives highest validation score
    # final_output_path: store the model trained till the final step.
    # Note that model in this folder may be worse than the model in `output_path`.
    model.save(final_path)

    logger.info("===== Finished cross-encoder training in %.4f (secs) =====", time() - start_time)


if __name__ == "__main__":
    typer.run(main)
