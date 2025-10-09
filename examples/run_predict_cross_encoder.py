"""Predict (query,doc) similarity logits using a Cross-Encoder"""

import gzip
import json
import warnings
from collections import defaultdict
from logging import basicConfig
from logging import getLogger
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
import typer
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm

from light_splade.data import PairScore
from light_splade.schemas.config import ConfigCrossEncoderPrediction
from light_splade.utils.io import load_yaml

warnings.filterwarnings("ignore")

basicConfig(level="INFO", format="%(asctime)s : %(levelname)s : %(name)s : %(message)s")
logger = getLogger(__name__)


def inverse_sigmoid(prob: float) -> float:
    """
    s = 1/(1 + e^(-x)) -> x = ln(s) - ln(1-s)
    """
    if prob >= 1:
        return 10**6
    if prob <= 0:
        return -(10**6)
    return float(np.log(prob) - np.log(1 - prob))


def save_scores(filepath: str, similarity_scores: dict[int, dict[int, float]]) -> None:
    filepath_ = Path(filepath)
    filepath_.parent.mkdir(parents=True, exist_ok=True)

    open_func = gzip.open if filepath.endswith(".gz") else open
    with open_func(filepath_, "wt") as f:
        for qid, scores in similarity_scores.items():
            f.write(json.dumps({"qid": qid, "scores": scores}, ensure_ascii=False) + "\n")


def load_master(train_file_path: str, valid_file_path: str) -> dict[int, str]:
    df1 = pd.read_json(train_file_path, lines=True)
    df2 = pd.read_json(valid_file_path, lines=True)
    df = pd.concat([df1, df2])
    df.columns = ["item_id", "text"]
    df.drop_duplicates(subset=["item_id"], inplace=True)

    id2item: dict[int, str] = dict()
    for item_id, query in tqdm(zip(df["item_id"], df["text"])):
        id2item[item_id] = query
    return id2item


def load_data(
    cfg: ConfigCrossEncoderPrediction,
) -> tuple[dict[int, dict[int, float]], dict[int, str], dict[int, str]]:
    logger.info("Loading data...")

    id2doc: dict[int, str] = load_master(cfg.train_doc_master, cfg.validation_doc_master)
    id2query: dict[int, str] = load_master(cfg.train_query_master, cfg.validation_query_master)

    init_scores = PairScore(Path(cfg.hard_negative_init_scores)).pair_scores

    num_pairs = sum([len(scores) for query_id, scores in init_scores.items()])
    logger.info(f"Num of docs in the master: {len(id2doc):,}")
    logger.info(f"Num of queries in the master: {len(id2query):,}")
    logger.info(f"Num of queries to compute scores={len(init_scores):,}")
    logger.info(f"Num of (query, doc) pairs to compute scores={num_pairs:,}")

    return init_scores, id2query, id2doc


def prepare_data(
    cfg: ConfigCrossEncoderPrediction, max_len: int
) -> tuple[list[tuple[int, int, int]], list[tuple[str, str]]]:
    logger.info("Preparing data for prediction...")

    init_scores, id2query, id2doc = load_data(cfg)
    sorted_qids = sorted(init_scores.keys())
    logger.info(f"{len(init_scores)=}, {len(sorted_qids)=}, {len(id2query)=}, {len(id2doc)=}")

    len_triples: list[tuple[int, int, int]] = []
    for qid in sorted_qids:
        query = id2query[qid]
        max_p_len = max(max_len - len(query), 0)
        doc_ids = init_scores[qid].keys()
        for doc_id in doc_ids:
            # NOTE: if the query_text is longer than max_len, then all of the doc_text will be empty.
            doc = id2doc[doc_id][:max_p_len]
            len_triples.append((qid, doc_id, len(query) + len(doc)))
    logger.info(f"{len(len_triples)=}")
    len_triples.sort(key=lambda x: -x[2])

    # accumulate all pairs sorted by text len (desc order)
    logger.info("Building (query, doc) pairs ordered by text len...")
    text_pairs: list[tuple[str, str]] = []
    for qid, doc_id, text_len in tqdm(len_triples):
        query = id2query[qid]
        max_p_len = max(max_len - len(query), 0)
        doc = id2doc[doc_id][:max_p_len]
        text_pairs.append((query, doc))
    logger.info(f"Num of pairs: {len(text_pairs):,}")

    logger.info("Finished preparing data for prediction!")

    return len_triples, text_pairs


def predict(text_pairs: list[tuple[str, str]], model: Any, batch_size: int) -> list[float]:
    # predict sim scores with cross-encoder
    logger.info("Predicting scores with cross-encoder...")
    pred_scores: list[float] = model.predict(text_pairs, batch_size=batch_size, show_progress_bar=True).tolist()

    # because we predict regression on label 0..1 by passing num_labels=1,
    # the CrossEncoder apply sigmoid on the logits.
    # Here, we apply inverse_sigmoid to convert the prob score back to logits.
    pred_scores = [inverse_sigmoid(score) for score in pred_scores]

    logger.info(f"Finished predicting with cross-encoder! Num of scores: {len(pred_scores)}...")
    return pred_scores


def build_similarity_scores(
    sorted_len_triples: list[tuple[int, int, int]], pred_scores: list[float]
) -> dict[int, dict[int, float]]:
    tuple_data: tuple = tuple(zip(*sorted_len_triples))
    (qids, doc_ids, lens_) = tuple_data

    # return similarity scores
    logger.info("Building similarity_scores...")
    similarity_scores: dict[int, dict[int, float]] = defaultdict(dict)
    for qid, doc_id, score in zip(qids, doc_ids, pred_scores):
        similarity_scores[qid][doc_id] = score
    return similarity_scores


def main(
    config_file: Path = typer.Option(
        Path("config/cross_encoder_predict.yaml"),
        help="Path to the config file",
    ),
) -> None:
    logger.info("=== STARTED cross-encoder prediction in run_predict_cross_encoder.py")
    start_time = time()

    # To avoid errors from transformers, like:
    # "Be aware, overflowing tokens are not returned for the setting you have
    # chosen, i.e. sequence pairs with the 'longest_first' truncation strategy."
    # So the returned list will always be empty even if some tokens have been removed.
    transformers.logging.set_verbosity_error()

    cfg = ConfigCrossEncoderPrediction(**load_yaml(config_file))
    logger.info("cfg=" + json.dumps(cfg.to_dict(), indent=4))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Initializing Cross-Encoder model...")
    model = CrossEncoder(
        cfg.model_path,
        num_labels=1,
        max_length=cfg.max_token_len,
        device=device,
    )

    sorted_len_triples, text_pairs = prepare_data(cfg=cfg, max_len=cfg.max_len)
    pred_scores = predict(text_pairs, model, cfg.predict_batch_size)
    similarity_scores = build_similarity_scores(sorted_len_triples, pred_scores)

    output_file = cfg.hard_negative_cross_encoder_scores
    logger.info(f"Writing similarity scores to {output_file}")
    save_scores(output_file, similarity_scores)

    logger.info("===== Finished cross-encoder prediction in %.4f (secs) =====", time() - start_time)


if __name__ == "__main__":
    typer.run(main)
