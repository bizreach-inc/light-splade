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

"""Convert mMARCO-ja into SPLADE triplet dataset files"""

import gzip
import json
from logging import basicConfig
from logging import getLogger
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import typer
from datasets import load_dataset
from tqdm import tqdm

from light_splade.data import TripletDataset
from light_splade.utils.io import ensure_path
from light_splade.utils.random import set_seeds

MMARCO_DS_PATH = "unicamp-dl/mmarco"
DOC_ID_FIELD = "doc_id"
QUERY_ID_FIELD = "qid"
TRIPLET_CHUNK_SIZE = 4_000_000  # chunk size for processing triplets

basicConfig(level="INFO", format="%(asctime)s : %(levelname)s : %(name)s : %(message)s")
logger = getLogger(__name__)


def sampling(values: np.ndarray, num_items: int) -> np.ndarray:
    if len(values) <= num_items or num_items == 0:
        return values
    else:
        return np.random.choice(values, num_items, replace=False)


def convert_master(
    file_path: Path,
    id2text: dict[int, str],
    target_ids: list[int],
    id_field: str,
) -> None:
    logger.info(f"Generating {file_path}...")
    master = []
    for id_ in target_ids:
        master.append({id_field: int(id_), "text": id2text[id_]})
    logger.info(f"{len(master)=}")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    open_func = gzip.open if str(file_path).endswith(".gz") else open
    with open_func(file_path, "wt") as f:
        for item in tqdm(master):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_positive_lists(file_path: Path, df_triplets: pd.DataFrame) -> None:
    logger.info(f"Generating {file_path}...")
    positive_lists = []
    for qid, df_ in df_triplets.groupby("qid"):
        positive_lists.append(
            dict(
                qid=int(qid),
                positive_doc_ids=[int(doc_id) for doc_id in df_.pos_doc_id.unique()],
            )
        )
    logger.info(f"{len(positive_lists)=}")

    open_func = gzip.open if str(file_path).endswith(".gz") else open
    with open_func(file_path, "wt") as f:
        for item in tqdm(positive_lists):
            f.write(json.dumps(item) + "\n")


def convert_triplets(file_path: Path, df_triplets: pd.DataFrame) -> None:
    """Convert triplets to ndjson format for TripletDataset.

    Args:
        file_path: Output file path
        df_triplets: DataFrame with columns [qid, pos_doc_id, neg_doc_id]
    """
    logger.info(f"Generating {file_path}...")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    open_func = gzip.open if str(file_path).endswith(".gz") else open
    with open_func(file_path, "wt") as f:
        for qid, pos_doc_id, neg_doc_id in tqdm(
            zip(df_triplets["qid"], df_triplets["pos_doc_id"], df_triplets["neg_doc_id"])
        ):
            triplet = {
                "qid": int(qid),
                "pos_doc_id": int(pos_doc_id),
                "neg_doc_id": int(neg_doc_id),
            }
            f.write(json.dumps(triplet, ensure_ascii=False) + "\n")


class MMarcoTripletConverter:
    def __init__(
        self,
        dataset_path: str | Path,
    ) -> None:
        self.dataset_path = ensure_path(dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)

    def _load(self, data_path: str, name: str, split: str) -> pd.DataFrame:
        ds = load_dataset(data_path, name, split=split, trust_remote_code=True)
        df = ds.to_pandas()
        return df

    def _load_id_triplets(self, query2id: dict[str, int], doc2id: dict[str, int]) -> pd.DataFrame:
        """
        The `unicamp-dl/mmarco` training triplets contains about 40M of text-based triplets (`query`, `positive`,
        `negative`) which requires much memory to load and process. This function loads the triplets chunk by chunk,
        and convert the texts into IDs using the provided mappings to save memory.
        """
        logger.info("Loading triplet data chunk by chunk and transforming to ids ...")

        start = 0
        is_finished = False
        query_ids = []
        pos_doc_ids = []
        neg_doc_ids = []
        while is_finished is False:
            end = start + TRIPLET_CHUNK_SIZE
            logger.info(f"\tLoading triplet chunk: {start:,} - {end:,} ...")
            ds = load_dataset(MMARCO_DS_PATH, name="japanese", split=f"train[{start}:{end}]", trust_remote_code=True)
            df = ds.to_pandas()
            if len(df) > 0:
                query_ids.extend(df["query"].map(query2id).tolist())
                pos_doc_ids.extend(df["positive"].map(doc2id).tolist())
                neg_doc_ids.extend(df["negative"].map(doc2id).tolist())
            if len(df) < TRIPLET_CHUNK_SIZE:
                is_finished = True
            start += len(df)

        logger.info("Finished loading triplets!")
        return pd.DataFrame(
            dict(
                qid=query_ids,
                pos_doc_id=pos_doc_ids,
                neg_doc_id=neg_doc_ids,
            )
        )

    def load_data(self) -> None:
        df_queries_train = self._load(MMARCO_DS_PATH, "queries-japanese", "train")
        df_queries_dev_full = self._load(MMARCO_DS_PATH, "queries-japanese", "dev.full")
        df_collection = self._load(MMARCO_DS_PATH, "collection-japanese", "collection")

        logger.info(f"{df_queries_train.shape=}")
        logger.info(f"{df_queries_dev_full.shape=}")
        logger.info(f"{df_collection.shape=}")

        df_queries = pd.concat([df_queries_train, df_queries_dev_full])
        logger.info(f"{df_queries.shape=}")
        logger.info(f"{df_queries.id.nunique()=}")

        df_collection.drop_duplicates(subset=["text"], inplace=True)
        logger.info(f"{df_collection.shape=}")

        query2id = dict(zip(df_queries.text, df_queries.id))
        id2query = dict(zip(df_queries.id, df_queries.text))
        doc2id = dict(zip(df_collection.text, df_collection.id))
        id2doc = dict(zip(df_collection.id, df_collection.text))

        # NOTE: the triplet data (name="japanese") has only `train` split
        # get triplets of IDs from original text-based triplets
        df_triplets = self._load_id_triplets(query2id, doc2id)
        logger.info(f"{df_triplets.shape=}")

        # get ids for validation set
        df_triplets_dev_full = df_triplets[df_triplets.qid.isin(df_queries_dev_full.id)]
        valid_query_ids = df_triplets_dev_full.qid.unique().tolist()
        valid_doc_ids = list(set(df_triplets_dev_full.pos_doc_id).union(df_triplets_dev_full.neg_doc_id))

        # build training set
        df_triplets_train = df_triplets[df_triplets.qid.isin(df_queries_train.id)]

        # get ids for training set
        train_query_ids = df_triplets_train.qid.unique().tolist()
        train_doc_ids = list(set(df_triplets_train.pos_doc_id).union(df_triplets_train.neg_doc_id))

        # validation
        assert len(set(valid_query_ids) & set(train_query_ids)) == 0

        logger.info(f"{len(train_query_ids)=}")
        logger.info(f"{len(train_doc_ids)=}")
        logger.info(f"{len(valid_query_ids)=}")
        logger.info(f"{len(valid_doc_ids)=}")

        self.id2query = id2query
        self.id2doc = id2doc

        self.train_query_ids = train_query_ids
        self.train_doc_ids = train_doc_ids
        self.valid_query_ids = valid_query_ids
        self.valid_doc_ids = valid_doc_ids

        self.df_triplets_train = df_triplets_train
        self.df_triplets_dev_full = df_triplets_dev_full

    def run(self) -> None:
        self.load_data()

        self.train_query_master_file = self.dataset_path / "train/query_master.ndjson"
        self.train_doc_master_file = self.dataset_path / "train/doc_master.ndjson"
        self.train_positive_list_file = self.dataset_path / "train/positive_lists.ndjson"
        self.train_triplet_file = self.dataset_path / "train/triplets.ndjson"

        self.valid_query_master_file = self.dataset_path / "validation/query_master.ndjson"
        self.valid_doc_master_file = self.dataset_path / "validation/doc_master.ndjson"
        self.valid_positive_list_file = self.dataset_path / "validation/positive_lists.ndjson"
        self.valid_triplet_file = self.dataset_path / "validation/triplets.ndjson"

        # Convert train data
        # query_master: each line is a dict including `qid` and `text` keys
        convert_master(
            self.train_query_master_file,
            self.id2query,
            self.train_query_ids,
            QUERY_ID_FIELD,
        )

        # doc_master: each line is a dict including `doc_id` and `text` keys
        convert_master(
            self.train_doc_master_file,
            self.id2doc,
            self.train_doc_ids,
            DOC_ID_FIELD,
        )

        # positive list: each line is a dict including `qid` and
        # `positive_doc_ids` keys
        convert_positive_lists(
            self.train_positive_list_file,
            self.df_triplets_train,
        )

        # triplets: each line is a dict including `qid`, `pos_doc_id`, `neg_doc_id` keys
        convert_triplets(
            self.train_triplet_file,
            self.df_triplets_train,
        )

        # Convert validation data
        convert_master(
            self.valid_query_master_file,
            self.id2query,
            self.valid_query_ids,
            QUERY_ID_FIELD,
        )

        convert_master(
            self.valid_doc_master_file,
            self.id2doc,
            self.valid_doc_ids,
            DOC_ID_FIELD,
        )

        convert_positive_lists(
            self.valid_positive_list_file,
            self.df_triplets_dev_full,
        )

        convert_triplets(
            self.valid_triplet_file,
            self.df_triplets_dev_full,
        )

        # Validate the generated data
        assert self.validate(), "Validation failed. Please check the data generation process."

    def validate(self) -> bool:
        """Simply call the corresponding dataset class to activate its
        validation"""

        try:
            logger.info("Validating the training set...")
            _ = TripletDataset(
                query_master_data_path=self.train_query_master_file,
                doc_master_data_path=self.train_doc_master_file,
                positive_pair_data_path=self.train_positive_list_file,
                triplet_path=self.train_triplet_file,
            )

            logger.info("Validating the validation set...")
            _ = TripletDataset(
                query_master_data_path=self.valid_query_master_file,
                doc_master_data_path=self.valid_doc_master_file,
                positive_pair_data_path=self.valid_positive_list_file,
                triplet_path=self.valid_triplet_file,
            )
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

        return True


def main(
    output_path: Path = typer.Option(
        Path("data/mmarco_ja_4_splade_triplet"),
        help="Path to the output dataset folder",
    ),
    seed: int = typer.Option(
        42,
        help="Random seed for reproducibility",
    ),
) -> None:
    start_time = time()
    set_seeds(seed)
    converter = MMarcoTripletConverter(output_path)
    converter.run()

    logger.info(
        "===== Finished converting mMARCO-ja data in %.4f (secs) =====",
        time() - start_time,
    )


if __name__ == "__main__":
    typer.run(main)
