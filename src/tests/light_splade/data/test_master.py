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

from pathlib import Path
from typing import Generator
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from light_splade.data.master import BaseMaster
from light_splade.data.master import DocumentMaster
from light_splade.data.master import QueryMaster
from light_splade.schemas.data import DocumentMasterSchema
from light_splade.schemas.data import QueryMasterSchema


@pytest.fixture
def test_master() -> BaseMaster:
    class TestMaster(BaseMaster):
        def __init__(self) -> None:
            self.id_list = [1, 2, 3]
            self.id_set = {1, 2, 3}
            self.data = {1: "text1", 2: "text2", 3: "text3"}

    return TestMaster()


class TestBaseMaster:
    @patch("light_splade.data.master.NdjsonLoader")
    def test_load_data(self, mock_loader_class: Mock) -> None:
        def mock_data() -> Generator[dict, None, None]:
            yield {"qid": 1, "text": "query text 1"}
            yield {"qid": 2, "text": "query text 2"}

        mock_loader = Mock()
        mock_loader.return_value = mock_data()
        mock_loader_class.return_value = mock_loader

        class TestMaster(BaseMaster):
            SCHEMA_CLASS = QueryMasterSchema

        master = TestMaster(Path("test_path"))

        assert master.data == {1: "query text 1", 2: "query text 2"}
        assert master.id_list == [1, 2]
        assert master.id_set == {1, 2}

    @patch("light_splade.data.master.NdjsonLoader")
    def test_load_data_schema_validation_error(self, mock_loader_class: Mock) -> None:
        def mock_data() -> Generator[dict, None, None]:
            yield {"invalid": "data"}

        mock_loader = Mock()
        mock_loader.return_value = mock_data()
        mock_loader_class.return_value = mock_loader

        class TestMaster(BaseMaster):
            SCHEMA_CLASS = QueryMasterSchema

        with pytest.raises(Exception):
            TestMaster(Path("test_path"))

    @pytest.mark.parametrize(
        "id, expected_text",
        [
            (1, "text1"),
            (2, "text2"),
            (3, "text3"),
        ],
    )
    def test___getitem___success(self, test_master: BaseMaster, id: int, expected_text: str) -> None:
        assert test_master[id] == expected_text

    def test___getitem___key_error(self, test_master: BaseMaster) -> None:
        with pytest.raises(IndexError) as exc_info:
            _ = test_master[999]

        assert "The id 999 does not exist in the master data" in str(exc_info.value)

    @pytest.mark.parametrize(
        "id, expected",
        [
            (1, True),
            (2, True),
            (3, True),
            (999, False),
        ],
    )
    def test___contains__(self, test_master: BaseMaster, id: int, expected: bool) -> None:
        assert (id in test_master) == expected


class TestQueryMaster:
    @patch("light_splade.data.master.NdjsonLoader")
    def test_schema_class(self, mock_loader_class: Mock) -> None:
        def mock_data() -> Generator[dict, None, None]:
            yield {"qid": 1, "text": "query text 1"}

        mock_loader = Mock()
        mock_loader.return_value = mock_data()
        mock_loader_class.return_value = mock_loader

        query_master = QueryMaster(Path("test_path"))
        assert query_master.SCHEMA_CLASS == QueryMasterSchema


class TestDocumentMaster:
    @patch("light_splade.data.master.NdjsonLoader")
    def test_schema_class(self, mock_loader_class: Mock) -> None:
        def mock_data() -> Generator[dict, None, None]:
            yield {"doc_id": 1, "text": "document text 1"}

        mock_loader = Mock()
        mock_loader.return_value = mock_data()
        mock_loader_class.return_value = mock_loader

        doc_master = DocumentMaster(Path("test_path"))
        assert doc_master.SCHEMA_CLASS == DocumentMasterSchema
