from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from light_splade.data.triplet_distil_dataset import TripletDistilDataset


class TestTripletDistilDataset:
    @pytest.fixture
    def mock_masters_and_data(
        self,
    ) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
        mock_query_master = MagicMock()
        mock_query_master.get_id_list.return_value = [1, 2]
        mock_query_master.get_id_set.return_value = {1, 2}
        mock_query_master.__getitem__.side_effect = lambda x: f"query_{x}"
        mock_query_master.__contains__.side_effect = lambda x: x in {1, 2}

        mock_doc_master = MagicMock()
        mock_doc_master.get_id_set.return_value = {101, 102, 201, 202}
        mock_doc_master.__getitem__.side_effect = lambda x: f"doc_{x}"
        mock_doc_master.__contains__.side_effect = lambda x: x in {
            101,
            102,
            201,
            202,
        }

        mock_positive_list = MagicMock()
        mock_positive_list.__getitem__.side_effect = {
            1: [101],
            2: [201],
        }.__getitem__
        mock_positive_list.__iter__.return_value = iter([1, 2])
        mock_positive_list.__contains__.side_effect = lambda x: x in {1, 2}

        mock_similarities = MagicMock()
        mock_similarities.__getitem__.side_effect = {
            1: {101: 0.9, 102: 0.3},
            2: {201: 0.8, 202: 0.2},
        }.__getitem__

        return (
            mock_query_master,
            mock_doc_master,
            mock_positive_list,
            mock_similarities,
        )

    @patch("light_splade.data.triplet_distil_dataset.QueryMaster")
    @patch("light_splade.data.triplet_distil_dataset.DocumentMaster")
    @patch("light_splade.data.triplet_distil_dataset.PositiveList")
    @patch("light_splade.data.triplet_distil_dataset.PairScore")
    def test___init___success(
        self,
        mock_pair_score: MagicMock,
        mock_positive_list: MagicMock,
        mock_doc_master: MagicMock,
        mock_query_master: MagicMock,
        mock_masters_and_data: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
    ) -> None:
        query_master, doc_master, positive_list, similarities = mock_masters_and_data

        mock_query_master.return_value = query_master
        mock_doc_master.return_value = doc_master
        mock_positive_list.return_value = positive_list
        mock_pair_score.return_value = similarities

        dataset = TripletDistilDataset(
            query_master_data_path="query_path",
            doc_master_data_path="doc_path",
            positive_pair_data_path="pos_path",
            hard_negative_scores_data_path="neg_path",
            sampling_mode="query_based",
            random_seed=42,
        )

        assert dataset.queries == query_master
        assert dataset.docs == doc_master
        assert dataset.positive_list == positive_list
        assert dataset.similarities == similarities
        assert dataset.qid_list == [1, 2]
        assert dataset.sampling_mode == "query_based"

    @pytest.mark.parametrize(
        "sampling_mode, expected_error",
        [
            ("invalid_mode", "must be 'query_based' or 'positive_pair_based'"),
            ("positive_pair_based", "only 'query_based' is supported"),
        ],
    )
    @patch("light_splade.data.triplet_distil_dataset.QueryMaster")
    @patch("light_splade.data.triplet_distil_dataset.DocumentMaster")
    @patch("light_splade.data.triplet_distil_dataset.PositiveList")
    @patch("light_splade.data.triplet_distil_dataset.PairScore")
    def test___init___invalid_sampling_mode(
        self,
        mock_pair_score: MagicMock,
        mock_positive_list: MagicMock,
        mock_doc_master: MagicMock,
        mock_query_master: MagicMock,
        sampling_mode: str,
        expected_error: str,
    ) -> None:
        with pytest.raises(ValueError) as exc_info:
            TripletDistilDataset(
                query_master_data_path="query_path",
                doc_master_data_path="doc_path",
                positive_pair_data_path="pos_path",
                hard_negative_scores_data_path="neg_path",
                sampling_mode=sampling_mode,
            )

        assert expected_error in str(exc_info.value)

    @patch("light_splade.data.triplet_distil_dataset.QueryMaster")
    @patch("light_splade.data.triplet_distil_dataset.DocumentMaster")
    @patch("light_splade.data.triplet_distil_dataset.PositiveList")
    @patch("light_splade.data.triplet_distil_dataset.PairScore")
    def test___len___query_based(
        self,
        mock_pair_score: MagicMock,
        mock_positive_list: MagicMock,
        mock_doc_master: MagicMock,
        mock_query_master: MagicMock,
        mock_masters_and_data: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
    ) -> None:
        query_master, doc_master, positive_list, similarities = mock_masters_and_data

        mock_query_master.return_value = query_master
        mock_doc_master.return_value = doc_master
        mock_positive_list.return_value = positive_list
        mock_pair_score.return_value = similarities

        dataset = TripletDistilDataset(
            query_master_data_path="query_path",
            doc_master_data_path="doc_path",
            positive_pair_data_path="pos_path",
            hard_negative_scores_data_path="neg_path",
        )

        assert len(dataset) == 2

    @patch("light_splade.data.triplet_distil_dataset.QueryMaster")
    @patch("light_splade.data.triplet_distil_dataset.DocumentMaster")
    @patch("light_splade.data.triplet_distil_dataset.PositiveList")
    @patch("light_splade.data.triplet_distil_dataset.PairScore")
    @patch("light_splade.data.triplet_distil_dataset.random")
    def test___getitem___query_based(
        self,
        mock_random: MagicMock,
        mock_pair_score: MagicMock,
        mock_positive_list: MagicMock,
        mock_doc_master: MagicMock,
        mock_query_master: MagicMock,
        mock_masters_and_data: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
    ) -> None:
        query_master, doc_master, positive_list, similarities = mock_masters_and_data

        mock_query_master.return_value = query_master
        mock_doc_master.return_value = doc_master
        mock_positive_list.return_value = positive_list
        mock_pair_score.return_value = similarities

        mock_random.sample.side_effect = lambda x, n: [x[0]]

        dataset = TripletDistilDataset(
            query_master_data_path="query_path",
            doc_master_data_path="doc_path",
            positive_pair_data_path="pos_path",
            hard_negative_scores_data_path="neg_path",
        )

        result = dataset[0]

        assert len(result) == 5
        q_text, pos_text, neg_text, score_pos, score_neg = result
        assert q_text == "query_1"
        assert pos_text == "doc_101"
        assert neg_text == "doc_102"
        assert score_pos == 0.9
        assert score_neg == 0.3

    def test__validate_qid_not_in_query_master(self) -> None:
        mock_positive_list = MagicMock()
        mock_positive_list.__iter__.return_value = iter([1, 999])

        mock_queries = MagicMock()
        mock_queries.__contains__.side_effect = lambda x: x == 1

        class TestDataset(TripletDistilDataset):
            def __init__(self) -> None:
                self.positive_list = mock_positive_list
                self.queries = mock_queries

        dataset = TestDataset()

        with pytest.raises(ValueError) as exc_info:
            dataset._validate()

        assert "qid 999 from the positive list does not exist in the query master" in str(exc_info.value)

    def test__validate_qid_not_in_positive_list(self) -> None:
        mock_positive_list = MagicMock()
        mock_positive_list.__iter__.return_value = iter([1])
        mock_positive_list.__contains__.side_effect = lambda x: x == 1

        mock_queries = MagicMock()
        mock_queries.__contains__.side_effect = lambda x: x in {1, 2}
        mock_queries.get_id_set.return_value = {1, 2}

        class TestDataset(TripletDistilDataset):
            def __init__(self) -> None:
                self.positive_list = mock_positive_list
                self.queries = mock_queries

        dataset = TestDataset()

        with pytest.raises(ValueError) as exc_info:
            dataset._validate()

        assert "qid 2 from the query master does not exist in the positive list" in str(exc_info.value)
