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

"""Triplet dataset used for SPLADE training without distillation.

This module defines :class:`TripletDataset`, a PyTorch :class:`Dataset` that loads query and document masters, positive
lists and optional triplets to produce samples of the form (query_text, positive_text, negative_text).
"""

from logging import getLogger
from pathlib import Path

from torch.utils.data import Dataset

from light_splade.utils.io import ensure_path

from .master import DocumentMaster
from .master import QueryMaster
from .positive_list import PositiveList
from .triplet_list import TripletList

logger = getLogger(__name__)


class TripletDataset(Dataset):
    """Dataset for triplets (query, positive, negative) for SPLADE v2 training without distillation.

    One sample of this dataset is a 3-tuple:
    - Query text
    - Positive document text
    - Negative document text
    """

    def __init__(
        self,
        query_master_data_path: str | Path,
        doc_master_data_path: str | Path,
        positive_pair_data_path: str | Path,
        triplet_path: str | Path | None = None,
    ) -> None:
        """Create the dataset and load required masters and lists.

        Args:
            query_master_data_path (str | Path): folder or file of query master
            doc_master_data_path (str | Path): folder or file of document master
            positive_pair_data_path (str | Path): folder or file of positive pair data
            triplet_path (str | Path | None): optional folder or file of triplet data
        """

        query_master_data_path = ensure_path(query_master_data_path)
        doc_master_data_path = ensure_path(doc_master_data_path)
        positive_pair_data_path = ensure_path(positive_pair_data_path)
        if triplet_path is not None:
            triplet_path = ensure_path(triplet_path)

        logger.info("Loading query master...")
        self.queries = QueryMaster(query_master_data_path)
        self.qid_list = self.queries.get_id_list()

        logger.info("Loading document master...")
        self.docs = DocumentMaster(doc_master_data_path)

        logger.info("Loading positive lists...")
        self.positive_list = PositiveList(positive_pair_data_path)

        logger.info("Loading triplet data...")
        self.triplets = None
        if triplet_path is not None:
            self.triplets = TripletList(triplet_path)

        self._validate()

    def _validate(self) -> None:
        """Perform dataset validation checks.

        This method verifies consistency between the loaded masters, positive lists and optional triplet lists. Helpful,
        explicit error messages are raised to simplify debugging when data is malformed or missing.
        """

        logger.info("Checking the dataset...")

        # every qid in the positive list must appear in query master
        for qid in self.positive_list:
            if qid not in self.queries:
                raise ValueError(f"qid {qid} from the positive list does not exist in the query master")

        # every qid in the query master must appear in positive list
        for qid in self.queries.get_id_set():
            if qid not in self.positive_list:
                raise ValueError(f"qid {qid} from the query master does not exist in the positive list")

        # every query must has at least 1 positive document
        for qid in self.queries.get_id_set():
            if len(self.positive_list[qid]) == 0:
                raise ValueError(f"qid {qid} has no positive document")

        # every doc_id in the positive list must appear in doc master
        for qid in self.queries.get_id_set():
            positive_doc_ids = self.positive_list[qid]
            for doc_id in positive_doc_ids:
                if doc_id not in self.docs:
                    raise ValueError(f"doc_id {doc_id} from positive list does not exist in document master")

        # every doc_id in the triplets must appear in doc master
        if self.triplets:
            for triplet in self.triplets:
                qid, pos_doc_id, neg_doc_id = triplet
                for doc_id in [pos_doc_id, neg_doc_id]:
                    if doc_id not in self.docs:
                        raise ValueError(f"doc_id {doc_id} from triplet {triplet} does not exist in document master")

    def __len__(self) -> int:
        """Return number of triplets available (0 if none provided)."""
        if self.triplets is None:
            return 0
        return len(self.triplets)

    def __getitem__(self, index: int) -> tuple[str, str, str]:
        """Return the (query, positive, negative) texts for the index.

        Raises:
            ValueError: If triplet data was not provided during initialization.
            IndexError: If the requested index is out of range.
        """
        if self.triplets is None:
            raise ValueError("Triplet data is not provided. Cannot get item by index.")

        if index >= len(self.triplets):
            raise IndexError(f"Index {index} is out of range for triplet dataset of size {len(self.triplets)}")
        qid, pos_doc_id, neg_doc_id = self.triplets[index]
        q_text = self.queries[qid]
        pos_text = self.docs[pos_doc_id]
        neg_text = self.docs[neg_doc_id]
        return (q_text, pos_text, neg_text)
