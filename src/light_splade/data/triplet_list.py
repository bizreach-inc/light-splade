"""Utilities for managing triplet lists loaded from NDJSON.

This module defines :class:`TripletList`, a thin wrapper that validates and stores triplets of the form
(qid, pos_doc_id, neg_doc_id) loaded from NDJSON files.
"""

from collections.abc import Iterator
from pathlib import Path

from light_splade.schemas.data import TripletSchema

from .ndjson_loader import NdjsonLoader


class TripletList:
    """Manage a list of triplets (qid, pos_doc_id, neg_doc_id) from NDJSON.

    The class validates each record using :class:`TripletSchema` and exposes sequence-like access via ``__getitem__``,
    ``__len__`` and iteration.
    """

    def __init__(self, data_path: Path) -> None:
        """Load triplets from ``data_path``."""
        self.load_data(data_path)

    def load_data(self, data_path: Path) -> None:
        """Load and validate triplet records into ``self.triplets``."""
        loader = NdjsonLoader(data_path)
        triplets: list[tuple[int, int, int]] = []
        for item in loader():
            obj = TripletSchema(**item)
            triplets.append((obj.qid, obj.pos_doc_id, obj.neg_doc_id))
        self.triplets = triplets

    def __getitem__(self, index: int) -> tuple[int, int, int]:
        """Return the triplet at ``index`` or raise IndexError if out-of-range."""
        if index >= len(self.triplets):
            raise IndexError(f"Index {index} is out of range for triplet list of size {len(self.triplets)}")
        return self.triplets[index]

    def __len__(self) -> int:
        """Return the number of triplets loaded."""
        return len(self.triplets)

    def __iter__(self) -> Iterator[tuple[int, int, int]]:
        """Iterate over loaded triplets."""
        yield from self.triplets
