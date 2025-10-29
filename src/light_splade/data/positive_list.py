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

"""Positive list loader mapping queries to positive documents.

This module provides :class:`PositiveList`, a thin wrapper that loads a mapping from query id to a list of positive
document ids from NDJSON files.
"""

from collections.abc import Iterator
from pathlib import Path

from light_splade.schemas.data import PositiveListSchema

from .ndjson_loader import NdjsonLoader


class PositiveList:
    """Map from a query ID to list of positive document IDs.

    The loaded mapping is available via ``self.positives`` and supports dictionary-like access through
    :meth:`__getitem__`, existence checking and iteration over qids.
    """

    def __init__(self, data_path: Path) -> None:
        """Create and load the positive list from the provided path.

        Args:
            data_path (Path): Path to NDJSON file or directory of NDJSON files.
        """
        self.load_data(data_path)

    def load_data(self, data_path: Path) -> None:
        """Load the positives mapping and store it in ``self.positives``."""
        loader = NdjsonLoader(data_path)
        # map from `qid` to list of `positive doc_ids`
        positives: dict[int, list[int]] = dict()
        for item in loader():
            obj = PositiveListSchema(**item)
            positives[obj.qid] = obj.positive_doc_ids
        self.positives = positives

    def __getitem__(self, qid: int) -> list[int]:
        """Return the list of positive document ids for ``qid``.

        Raises:
            IndexError: If ``qid`` is not present in the mapping.
        """
        if qid not in self.positives:
            raise IndexError(f"The qid {qid} does not exist in positive list")
        return self.positives[qid]

    def __contains__(self, qid: int) -> bool:
        """Return True if ``qid`` is in the positive mapping."""
        return qid in self.positives

    def __iter__(self) -> Iterator[int]:
        """Iterate over qids present in the positive list."""
        yield from self.positives.keys()
