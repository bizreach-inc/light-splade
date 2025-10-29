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

from light_splade.data.positive_list import PositiveList


class TestPositiveList:
    @patch("light_splade.data.positive_list.NdjsonLoader")
    def test_load_data(self, mock_loader_class: Mock) -> None:
        def mock_data() -> Generator[dict, None, None]:
            yield {"qid": 1, "positive_doc_ids": [101, 102]}
            yield {"qid": 2, "positive_doc_ids": [201]}

        mock_loader = Mock()
        mock_loader.return_value = mock_data()
        mock_loader_class.return_value = mock_loader

        positive_list = PositiveList(Path("test_path"))

        assert positive_list.positives == {1: [101, 102], 2: [201]}

    @pytest.mark.parametrize(
        "qid, expected_doc_ids",
        [
            (1, [101, 102, 103]),
            (2, [201, 202]),
        ],
    )
    @patch("light_splade.data.positive_list.NdjsonLoader")
    def test___getitem___success(self, mock_loader_class: Mock, qid: int, expected_doc_ids: list[int]) -> None:
        def mock_data() -> Generator[dict, None, None]:
            yield {"qid": 1, "positive_doc_ids": [101, 102, 103]}
            yield {"qid": 2, "positive_doc_ids": [201, 202]}

        mock_loader = Mock()
        mock_loader.return_value = mock_data()
        mock_loader_class.return_value = mock_loader

        positive_list = PositiveList(Path("test_path"))

        assert positive_list[qid] == expected_doc_ids

    @patch("light_splade.data.positive_list.NdjsonLoader")
    def test___getitem___key_error(self, mock_loader_class: Mock) -> None:
        def mock_data() -> Generator[dict, None, None]:
            yield {"qid": 1, "positive_doc_ids": [101, 102]}

        mock_loader = Mock()
        mock_loader.return_value = mock_data()
        mock_loader_class.return_value = mock_loader

        positive_list = PositiveList(Path("test_path"))

        with pytest.raises(IndexError) as exc_info:
            _ = positive_list[999]

        assert "The qid 999 does not exist in positive list" in str(exc_info.value)

    @pytest.mark.parametrize(
        "qid, expected",
        [
            (1, True),
            (2, True),
            (999, False),
        ],
    )
    @patch("light_splade.data.positive_list.NdjsonLoader")
    def test___contains__(self, mock_loader_class: Mock, qid: int, expected: bool) -> None:
        def mock_data() -> Generator[dict, None, None]:
            yield {"qid": 1, "positive_doc_ids": [101, 102]}
            yield {"qid": 2, "positive_doc_ids": [201, 202]}

        mock_loader = Mock()
        mock_loader.return_value = mock_data()
        mock_loader_class.return_value = mock_loader

        positive_list = PositiveList(Path("test_path"))

        assert (qid in positive_list) == expected

    @patch("light_splade.data.positive_list.NdjsonLoader")
    def test___iter__(self, mock_loader_class: Mock) -> None:
        def mock_data() -> Generator[dict, None, None]:
            yield {"qid": 1, "positive_doc_ids": [101, 102]}
            yield {"qid": 2, "positive_doc_ids": [201, 202]}
            yield {"qid": 3, "positive_doc_ids": [301]}

        mock_loader = Mock()
        mock_loader.return_value = mock_data()
        mock_loader_class.return_value = mock_loader

        positive_list = PositiveList(Path("test_path"))

        qids = list(positive_list)
        assert set(qids) == {1, 2, 3}
