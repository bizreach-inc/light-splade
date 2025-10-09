"""Triplet dataset with teacher similarity scores (for distillation).

This dataset augments triplet samples with teacher-provided similarity scores for the positive and negative documents.
It currently implements a ``query_based`` sampling mode where one sample corresponds to a single query (positive sampled
from available positives, negative sampled from available hard-negatives).
"""

import random
from logging import getLogger
from pathlib import Path

from torch.utils.data import Dataset

from light_splade.utils.io import ensure_path

from .master import DocumentMaster
from .master import QueryMaster
from .pair_score import PairScore
from .positive_list import PositiveList

logger = getLogger(__name__)


class TripletDistilDataset(Dataset):
    """Dataset for triplets with distillation scores.

    One sample of this dataset is a 5-tuple:
    - Query text
    - Positive document text
    - Negative document text
    - Teacher score for (query, positive)
    - Teacher score for (query, negative)
    """

    def __init__(
        self,
        query_master_data_path: str | Path,
        doc_master_data_path: str | Path,
        positive_pair_data_path: str | Path,
        hard_negative_scores_data_path: str | Path,
        sampling_mode: str = "query_based",
        random_seed: int = 42,
    ) -> None:
        """Create and initialize the distillation dataset.

        Args:
            query_master_data_path (str | Path): folder or file of query master
            doc_master_data_path (str | Path): folder or file of document master
            positive_pair_data_path (str | Path): folder or file of positive pair data
            hard_negative_scores_data_path (str | Path): folder or file of hard-negative scores
            sampling_mode (str): `query_based` or `positive_pair_based` (only
                ``query_based`` is currently supported)
            random_seed (int): Seed for random sampling (set None to avoid setting seed)
        """

        if sampling_mode not in ["query_based", "positive_pair_based"]:
            raise ValueError("`sampling_mode` must be 'query_based' or 'positive_pair_based'")
        if sampling_mode != "query_based":
            raise ValueError("Currently, only 'query_based' is supported")

        query_master_data_path = ensure_path(query_master_data_path)
        doc_master_data_path = ensure_path(doc_master_data_path)
        positive_pair_data_path = ensure_path(positive_pair_data_path)
        hard_negative_scores_data_path = ensure_path(hard_negative_scores_data_path)

        logger.info("Loading query master...")
        self.queries = QueryMaster(query_master_data_path)
        self.qid_list = self.queries.get_id_list()

        logger.info("Loading document master...")
        self.docs = DocumentMaster(doc_master_data_path)

        logger.info("Loading positive lists...")
        self.positive_list = PositiveList(positive_pair_data_path)

        logger.info("Loading hard-negative scores...")
        self.similarities = PairScore(hard_negative_scores_data_path)
        # TODO: filter qids which are not in qid_list from `similarities` to
        # save memory

        self.sampling_mode = sampling_mode
        random.seed(random_seed)

        self._validate()

    def _validate(self) -> None:
        """Perform consistency checks and raise informative errors."""

        logger.info("Validating...")

        # every qid in the positive list must appear in query master
        for qid in self.positive_list:
            if qid not in self.queries:
                raise ValueError(f"qid {qid} from the positive list does not exist in the query master")

        # every qid in the query master must appear in positive list
        for qid in self.queries.get_id_set():
            if qid not in self.positive_list:
                raise ValueError(f"qid {qid} from the query master does not exist in the positive list")

        # every doc_id must appear in doc master
        for qid in self.queries.get_id_set():
            positive_doc_ids = self.positive_list[qid]
            for doc_id in positive_doc_ids:
                if doc_id not in self.docs:
                    raise ValueError(f"doc_id {doc_id} from positive list does not exist in document master")

        # every query must has at least 1 positive document
        for qid in self.queries.get_id_set():
            if len(self.positive_list[qid]) == 0:
                raise ValueError(f"qid {qid} has no positive document")

        # every query must have at least 1 available negative document
        for qid in self.positive_list:
            pos_doc_ids = self.positive_list[qid]
            sim_scores = self.similarities[qid]
            neg_doc_ids = set(sim_scores.keys()) - set(pos_doc_ids)
            neg_doc_ids = neg_doc_ids.intersection(self.docs.get_id_set())
            if len(neg_doc_ids) == 0:
                raise ValueError(f"qid {qid} has no available negative document")

        # every (qid, positive_doc_id) pair must have teacher_pair_score
        for qid in self.positive_list:
            positive_doc_ids = self.positive_list[qid]
            sim_scores = self.similarities[qid]
            for doc_id in positive_doc_ids:
                if doc_id not in sim_scores:
                    raise ValueError(f"The pair ({qid}, {doc_id}) (qid, doc_id) has no hard-negative score")

    def __len__(self) -> int:
        """Return number of samples depending on the sampling mode.

        For ``query_based`` mode this equals the number of queries.
        """
        if self.sampling_mode != "query_based":
            raise ValueError("Currently, only 'query_based' is supported")
        return len(self.qid_list)

    def __getitem__(self, index: int) -> tuple[str, str, str, float, float]:
        """Return a single sample with teacher scores.

        The returned tuple is (q_text, pos_text, neg_text, score_pos, score_neg).
        """
        if self.sampling_mode == "query_based":
            qid = self.qid_list[index]
            q_text = self.queries[qid]
            sim_scores = self.similarities[qid]

            pos_doc_ids = self.positive_list[qid]
            neg_doc_id_set = set(sim_scores.keys()) - set(pos_doc_ids)
            neg_doc_ids = list(neg_doc_id_set.intersection(self.docs.get_id_set()))

            pos_doc_id = random.sample(pos_doc_ids, 1)[0]
            neg_doc_id = random.sample(neg_doc_ids, 1)[0]
            pos_text = self.docs[pos_doc_id]
            neg_text = self.docs[neg_doc_id]
            score_pos = sim_scores[pos_doc_id]
            score_neg = sim_scores[neg_doc_id]

            return (q_text, pos_text, neg_text, score_pos, score_neg)
        else:
            raise ValueError("Currently, only 'query_based' is supported")
