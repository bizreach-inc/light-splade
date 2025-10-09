"""Load teacher model pairwise similarity scores from NDJSON.

This module exposes :class:`PairScore`, which loads a mapping from query id to a dictionary of
document id -> similarity score produced by a teacher model (used for distillation).
"""

from pathlib import Path

from light_splade.schemas.data import HardNegativeScoreSchema

from .ndjson_loader import NdjsonLoader


class PairScore:
    """Container for pairwise scores indexed by query id.

    Args:
        data_path (Path): Path to NDJSON file(s) containing hard-negative score entries. Each line is validated against
            :class:`HardNegativeScoreSchema`.
        target_qids (list[int] | None): Optional list of qids to filter and load; if provided, only scores for those
            qids will be kept.
    """

    def __init__(self, data_path: Path, target_qids: list[int] | None = None) -> None:
        target_qids_: set[int] | None = set(target_qids) if target_qids else None
        self.load_data(data_path, target_qids_)

    def load_data(self, data_path: Path, target_qids: set[int] | None) -> None:
        """Load pairwise scores and store them in ``self.pair_scores``.

        The NDJSON may store document ids as strings; they are converted to integers before storing.
        """
        loader = NdjsonLoader(data_path)
        # map from `qid` to a dict of scores (doc_id -> score)
        pair_scores: dict[int, dict[int, float]] = dict()
        for item in loader():
            obj = HardNegativeScoreSchema(**item)
            if target_qids and obj.qid not in target_qids:
                continue
            # ndjson may save doc_id as str (e.g., '123': 0.75).
            # ensure the doc_ids are integer
            pair_scores[obj.qid] = {int(doc_id): score for doc_id, score in obj.scores.items()}
        self.pair_scores = pair_scores

    def __getitem__(self, qid: int) -> dict[int, float]:
        """Return a mapping of doc_id -> score for the given ``qid``.

        Raises:
            IndexError: If ``qid`` is not present in the loaded data.
        """
        if qid not in self.pair_scores:
            raise IndexError(f"The qid {qid} does not exist in hard-negative scores")
        return self.pair_scores[qid]
