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

"""Master data loaders for queries and documents.

This module defines helper classes that load master data (queries or documents) from NDJSON files and expose convenient
lookup APIs. The concrete masters (`QueryMaster`, `DocumentMaster`) set a Pydantic model in ``SCHEMA_CLASS`` which is
used to validate each NDJSON record.

Example:
    master = QueryMaster(Path("/path/to/query_master.ndjson"))
    qids = master.get_id_list()
    text = master[123]

"""

from pathlib import Path
from typing import Any
from typing import Set
from typing import Type

from light_splade.schemas.data import BaseMasterSchema
from light_splade.schemas.data import DocumentMasterSchema
from light_splade.schemas.data import QueryMasterSchema

from .ndjson_loader import NdjsonLoader


class BaseMaster:
    """Load and expose master data indexed by integer id.

    Subclasses must set the `SCHEMA_CLASS` attribute to a Pydantic model (subclass of :class:`BaseMasterSchema`) which
    will be used to validate each loaded record. Records are expected to contain ``id`` and ``text`` fields.

    Attributes:
        SCHEMA_CLASS: Pydantic model class used to validate each NDJSON record.
        data (dict[int, str]): Mapping from id to text loaded from the file.
        id_list (list[int]): Ordered list of ids in the same order as the file.
        id_set (set[int]): Set of ids for O(1) membership checks.
    """

    SCHEMA_CLASS: Type[BaseMasterSchema]

    def __init__(self, data_path: Path) -> None:
        """Create a master and load data from ``data_path``.

        Args:
            data_path (Path): Path to an NDJSON file containing the master records.
        """
        self.load_data(data_path)

    def load_data(self, data_path: Path) -> None:
        """Load records from NDJSON and validate them with ``SCHEMA_CLASS``.

        The loader iterates over the NDJSON file using :class:`NdjsonLoader`,
        constructs instances of ``SCHEMA_CLASS`` for validation, and builds
        internal structures for lookup.

        Args:
            data_path (Path): Path to the NDJSON file.

        Raises:
            Exception: Any exception raised by the schema validation or loader
                is propagated to the caller.
        """
        loader = NdjsonLoader(data_path)
        id_list = []
        text_list = []
        for item in loader():
            try:
                data_obj = self.SCHEMA_CLASS(**item)
                id_list.append(data_obj.id)
                text_list.append(data_obj.text)
            except Exception as e:
                # Intentionally re-raise validation/parse errors so callers are
                # aware of malformed records.
                raise e

        self.data = {id_: text for id_, text in zip(id_list, text_list)}
        self.id_list = id_list
        self.id_set = set(id_list)  # For O(1) lookup

    def get_id_list(self) -> list[int]:
        """Return the ordered list of ids loaded from the file.

        Returns:
            A list of integer ids in the same order as the source NDJSON file.
        """
        return self.id_list

    def get_id_set(self) -> Set[int]:
        """Return the set of ids for fast existence checking.

        Returns:
            A set of integer ids.
        """
        return self.id_set

    def __len__(self) -> int:
        return len(self.id_list)

    def __getitem__(self, id: int) -> Any:
        """Return the text for the given ``id``.

        Args:
            id (int): Integer id of the record to fetch.

        Returns:
            The text associated with the id.

        Raises:
            IndexError: If the id is not present in the master data.
        """
        if id not in self.data:
            raise IndexError(f"The id {id} does not exist in the master data")
        return self.data[id]

    def __contains__(self, id: int) -> bool:
        return id in self.id_set


class QueryMaster(BaseMaster):
    """Master for query data.

    Uses :class:`QueryMasterSchema` as the validation schema for records.
    """

    SCHEMA_CLASS = QueryMasterSchema


class DocumentMaster(BaseMaster):
    """Master for document data.

    Uses :class:`DocumentMasterSchema` as the validation schema for records.
    """

    SCHEMA_CLASS = DocumentMasterSchema
