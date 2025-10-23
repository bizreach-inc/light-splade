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

"""Concise SPLADE triplet trainer example"""

import json
import os
from logging import basicConfig
from logging import getLogger
from pathlib import Path
from time import time

import hydra
import torch
import transformers
from omegaconf import DictConfig

from light_splade.data import TripletCollator
from light_splade.data import TripletDataset
from light_splade.models import Splade
from light_splade.schemas.config import ConfigSpladeTriplet
from light_splade.trainer import SpladeTripletTrainer
from light_splade.utils.argument import instantiate
from light_splade.utils.random import set_seeds

basicConfig(level="INFO", format="%(asctime)s : %(levelname)s : %(name)s : %(message)s")
logger = getLogger(__name__)

transformers.utils.logging.set_verbosity(transformers.logging.INFO)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(version_base="1.2", config_path="../config", config_name=None)
def main(config: DictConfig) -> None:
    start_time = time()

    cfg = instantiate(ConfigSpladeTriplet, config)

    set_seeds(cfg.training.seed)

    logger.info("cfg.data=" + json.dumps(cfg.data.to_dict(), indent=4))
    logger.info("cfg.model=" + json.dumps(cfg.model.to_dict(), indent=4))
    logger.info("cfg.training=" + json.dumps(cfg.training.to_dict(), indent=4))

    train_ds = TripletDataset(
        query_master_data_path=Path(cfg.data.train_query_master),
        doc_master_data_path=Path(cfg.data.train_doc_master),
        positive_pair_data_path=Path(cfg.data.train_positives),
        triplet_path=Path(cfg.data.train_triplets),
    )

    eval_for_loss_ds = None
    if cfg.data.eval_loss_size > 0:
        train_ds, eval_for_loss_ds = torch.utils.data.random_split(
            train_ds, lengths=[len(train_ds) - cfg.data.eval_loss_size, cfg.data.eval_loss_size]
        )

    eval_ds = TripletDataset(
        query_master_data_path=Path(cfg.data.validation_query_master),
        doc_master_data_path=Path(cfg.data.validation_doc_master),
        positive_pair_data_path=Path(cfg.data.validation_positives),
    )

    logger.info(f"{len(train_ds)=}")
    if eval_for_loss_ds is not None:
        logger.info(f"Num of triplets in eval_for_loss_ds: {len(eval_for_loss_ds)=}")
    logger.info(f"Num of queries in eval_ds: {len(eval_ds.queries)}")

    collator = TripletCollator(
        tokenizer_path=cfg.model.model_name_or_path,
        max_length=cfg.model.max_length,
    )

    model = Splade(
        d_model_path=cfg.model.model_name_or_path,
        q_model_path=cfg.model.model_name_or_path_q,
        freeze_d_model=cfg.model.freeze_d_model,
        agg=cfg.model.agg,
    )
    model = model.to(device)
    logger.info(f"{model.is_shared_weights=}")
    logger.info(f"{model.num_trainable_params=}")
    logger.info(f"{model.num_total_params=}")

    if cfg.training.resume_from_checkpoint and cfg.training.fp16:
        cfg.training.fp16 = False
        os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
        """ When resume from checkpoint, we need to set fp16=False to avoid
        the `No inf checks were recorded for this optimizer.` error
        Ref: https://discuss.pytorch.org/t/resume-training-with-mixed-precision-lead-to-no-inf-checks-were-recorded-for-this-optimizer/115828
        NOTE: turning off the mixed precision makes the training slower
        """  # noqa

    trainer = SpladeTripletTrainer(
        model=model,
        args=cfg.training,
        data_collator=collator,
        train_dataset=train_ds,
        eval_for_loss_dataset=eval_for_loss_ds,
        eval_dataset=eval_ds,
        processing_class=model.d_encoder.tokenizer,
    )
    train_result = trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)

    # Save the final model
    final_path = cfg.training.final_output_dir or cfg.training.output_dir
    trainer.save_model(final_path)
    if trainer.is_world_process_zero():
        with open(os.path.join(final_path, "training_args.json"), "w") as write_file:
            json.dump(cfg.training.to_dict(), write_file, indent=4)

    # Save training results
    trainer.args.output_dir = os.path.join(final_path, "results")
    os.makedirs(trainer.args.output_dir, exist_ok=True)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluate and save evaluation results
    if cfg.training.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_ds)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("===== Finished training Triplet-based SPLADE model in %.4f (secs) =====", time() - start_time)


if __name__ == "__main__":
    main()
