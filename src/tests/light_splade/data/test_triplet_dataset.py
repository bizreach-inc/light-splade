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

from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from light_splade.data.triplet_dataset import TripletDataset


class TestTripletDataset:
    @pytest.fixture
    def mock_components(self) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
        # Query master
        mock_query_master = MagicMock()
        mock_query_master.get_id_list.return_value = [1, 2]
        mock_query_master.get_id_set.return_value = {1, 2}
        mock_query_master.__getitem__.side_effect = {1: "query_1", 2: "query_2"}.__getitem__
        mock_query_master.__contains__.side_effect = lambda x: x in {1, 2}

        # Document master
        mock_doc_master = MagicMock()
        mock_doc_master.__getitem__.side_effect = {
            101: "doc_101",
            102: "doc_102",
            201: "doc_201",
            202: "doc_202",
        }.__getitem__
        mock_doc_master.__contains__.side_effect = lambda x: x in {101, 102, 201, 202}

        # Positive list
        mock_positive_list = MagicMock()
        mock_positive_list.__iter__.return_value = iter([1, 2])
        mock_positive_list.__contains__.side_effect = lambda x: x in {1, 2}
        mock_positive_list.__getitem__.side_effect = {1: [101], 2: [201]}.__getitem__

        # Triplet list - behaves like list of tuples (qid, pos_id, neg_id)
        mock_triplet_list = MagicMock()
        mock_triplets_data = [(1, 101, 102), (2, 201, 202)]
        mock_triplet_list.__len__.return_value = len(mock_triplets_data)
        mock_triplet_list.__getitem__.side_effect = mock_triplets_data.__getitem__

        return (
            mock_query_master,
            mock_doc_master,
            mock_positive_list,
            mock_triplet_list,
        )

    @patch("light_splade.data.triplet_dataset.QueryMaster")
    @patch("light_splade.data.triplet_dataset.DocumentMaster")
    @patch("light_splade.data.triplet_dataset.PositiveList")
    @patch("light_splade.data.triplet_dataset.TripletList")
    def test___init___success(
        self,
        mock_triplet_list_cls: Mock,
        mock_positive_list_cls: Mock,
        mock_doc_master_cls: Mock,
        mock_query_master_cls: Mock,
        mock_components: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
    ) -> None:
        query_master, doc_master, positive_list, triplet_list = mock_components
        mock_query_master_cls.return_value = query_master
        mock_doc_master_cls.return_value = doc_master
        mock_positive_list_cls.return_value = positive_list
        mock_triplet_list_cls.return_value = triplet_list

        dataset = TripletDataset(
            query_master_data_path="query_path",
            doc_master_data_path="doc_path",
            positive_pair_data_path="pos_path",
            triplet_path="triplet_path",
        )

        assert dataset.queries == query_master
        assert dataset.docs == doc_master
        assert dataset.positive_list == positive_list
        assert dataset.triplets == triplet_list
        assert dataset.qid_list == [1, 2]

    @patch("light_splade.data.triplet_dataset.QueryMaster")
    @patch("light_splade.data.triplet_dataset.DocumentMaster")
    @patch("light_splade.data.triplet_dataset.PositiveList")
    @patch("light_splade.data.triplet_dataset.TripletList")
    def test___len__(
        self,
        mock_triplet_list_cls: Mock,
        mock_positive_list_cls: Mock,
        mock_doc_master_cls: Mock,
        mock_query_master_cls: Mock,
        mock_components: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
    ) -> None:
        query_master, doc_master, positive_list, triplet_list = mock_components
        mock_query_master_cls.return_value = query_master
        mock_doc_master_cls.return_value = doc_master
        mock_positive_list_cls.return_value = positive_list
        mock_triplet_list_cls.return_value = triplet_list

        dataset = TripletDataset(
            query_master_data_path="query_path",
            doc_master_data_path="doc_path",
            positive_pair_data_path="pos_path",
            triplet_path="triplet_path",
        )

        assert len(dataset) == 2

    @patch("light_splade.data.triplet_dataset.QueryMaster")
    @patch("light_splade.data.triplet_dataset.DocumentMaster")
    @patch("light_splade.data.triplet_dataset.PositiveList")
    @patch("light_splade.data.triplet_dataset.TripletList")
    def test___getitem___success(
        self,
        mock_triplet_list_cls: Mock,
        mock_positive_list_cls: Mock,
        mock_doc_master_cls: Mock,
        mock_query_master_cls: Mock,
        mock_components: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
    ) -> None:
        query_master, doc_master, positive_list, triplet_list = mock_components
        mock_query_master_cls.return_value = query_master
        mock_doc_master_cls.return_value = doc_master
        mock_positive_list_cls.return_value = positive_list
        mock_triplet_list_cls.return_value = triplet_list

        dataset = TripletDataset(
            query_master_data_path="query_path",
            doc_master_data_path="doc_path",
            positive_pair_data_path="pos_path",
            triplet_path="triplet_path",
        )

        q_text, pos_text, neg_text = dataset[0]
        assert q_text == "query_1"
        assert pos_text == "doc_101"
        assert neg_text == "doc_102"

    @patch("light_splade.data.triplet_dataset.QueryMaster")
    @patch("light_splade.data.triplet_dataset.DocumentMaster")
    @patch("light_splade.data.triplet_dataset.PositiveList")
    @patch("light_splade.data.triplet_dataset.TripletList")
    def test___getitem___out_of_range(
        self,
        mock_triplet_list_cls: Mock,
        mock_positive_list_cls: Mock,
        mock_doc_master_cls: Mock,
        mock_query_master_cls: Mock,
        mock_components: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
    ) -> None:
        query_master, doc_master, positive_list, triplet_list = mock_components
        mock_query_master_cls.return_value = query_master
        mock_doc_master_cls.return_value = doc_master
        mock_positive_list_cls.return_value = positive_list
        mock_triplet_list_cls.return_value = triplet_list

        dataset = TripletDataset(
            query_master_data_path="query_path",
            doc_master_data_path="doc_path",
            positive_pair_data_path="pos_path",
            triplet_path="triplet_path",
        )

        with pytest.raises(IndexError) as exc_info:
            _ = dataset[10]
        assert "Index 10 is out of range" in str(exc_info.value)

    # Validation error: qid from positive list not in queries
    def test__validate_qid_not_in_query_master(self) -> None:
        mock_positive_list = MagicMock()
        mock_positive_list.__iter__.return_value = iter([1, 999])
        mock_positive_list.__contains__.side_effect = lambda x: x in {1, 999}
        mock_positive_list.__getitem__.side_effect = {1: [101], 999: [102]}.__getitem__

        mock_queries = MagicMock()
        mock_queries.__contains__.side_effect = lambda x: x == 1
        mock_queries.get_id_set.return_value = {1}

        dataset = TripletDataset.__new__(TripletDataset)  # bypass __init__
        dataset.positive_list = mock_positive_list
        dataset.queries = mock_queries
        dataset.docs = MagicMock()
        dataset.triplets = Mock()

        with pytest.raises(ValueError) as exc_info:
            dataset._validate()
        assert "qid 999 from the positive list does not exist in the query master" in str(exc_info.value)

    # Validation error: qid in queries missing in positive list
    def test__validate_qid_not_in_positive_list(self) -> None:
        mock_positive_list = MagicMock()
        mock_positive_list.__iter__.return_value = iter([1])
        mock_positive_list.__contains__.side_effect = lambda x: x == 1
        mock_positive_list.__getitem__.side_effect = {1: [101]}.__getitem__

        mock_queries = MagicMock()
        mock_queries.__contains__.side_effect = lambda x: x in {1, 2}
        mock_queries.get_id_set.return_value = {1, 2}

        dataset = TripletDataset.__new__(TripletDataset)
        dataset.positive_list = mock_positive_list
        dataset.queries = mock_queries
        dataset.docs = MagicMock()
        dataset.triplets = Mock()

        with pytest.raises(ValueError) as exc_info:
            dataset._validate()
        assert "qid 2 from the query master does not exist in the positive list" in str(exc_info.value)

    # Validation error: doc id missing
    def test__validate_doc_not_in_docs(self) -> None:
        mock_positive_list = MagicMock()
        mock_positive_list.__iter__.return_value = iter([1])
        mock_positive_list.__contains__.side_effect = lambda x: x == 1
        mock_positive_list.__getitem__.side_effect = {1: [999]}.__getitem__

        mock_queries = MagicMock()
        mock_queries.get_id_set.return_value = {1}
        mock_queries.__contains__.side_effect = lambda x: x == 1

        mock_docs = MagicMock()
        mock_docs.__contains__.side_effect = lambda x: False

        dataset = TripletDataset.__new__(TripletDataset)
        dataset.positive_list = mock_positive_list
        dataset.queries = mock_queries
        dataset.docs = mock_docs
        dataset.triplets = Mock()

        with pytest.raises(ValueError) as exc_info:
            dataset._validate()
        assert "doc_id 999 from positive list does not exist in document master" in str(exc_info.value)

    # Validation error: zero positives
    def test__validate_zero_positive(self) -> None:
        mock_positive_list = MagicMock()
        mock_positive_list.__iter__.return_value = iter([1])
        mock_positive_list.__contains__.side_effect = lambda x: x == 1
        mock_positive_list.__getitem__.side_effect = {1: []}.__getitem__

        mock_queries = MagicMock()
        mock_queries.get_id_set.return_value = {1}
        mock_queries.__contains__.side_effect = lambda x: x == 1

        dataset = TripletDataset.__new__(TripletDataset)
        dataset.positive_list = mock_positive_list
        dataset.queries = mock_queries
        dataset.docs = MagicMock()
        dataset.triplets = Mock()

        with pytest.raises(ValueError) as exc_info:
            dataset._validate()
        assert "qid 1 has no positive document" in str(exc_info.value)
