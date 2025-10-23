from pathlib import Path
from typing import Generator
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from light_splade.data.pair_score import PairScore


class TestPairScore:
    @patch("light_splade.data.pair_score.NdjsonLoader")
    def test_load_data(self, mock_loader_class: Mock) -> None:
        def mock_data() -> Generator[dict, None, None]:
            yield {"qid": 1, "scores": {"101": 0.8, "102": 0.6}}

        mock_loader = Mock()
        mock_loader.return_value = mock_data()
        mock_loader_class.return_value = mock_loader

        pair_score = PairScore(Path("test_path"))

        assert pair_score.pair_scores == {1: {101: 0.8, 102: 0.6}}

    @patch("light_splade.data.pair_score.NdjsonLoader")
    def test_load_data_string_doc_ids_converted_to_int(self, mock_loader_class: Mock) -> None:
        def mock_data() -> Generator[dict, None, None]:
            yield {"qid": 1, "scores": {"101": 0.8, "102": 0.6}}

        mock_loader = Mock()
        mock_loader.return_value = mock_data()
        mock_loader_class.return_value = mock_loader

        pair_score = PairScore(Path("test_path"))

        assert pair_score.pair_scores == {1: {101: 0.8, 102: 0.6}}
        assert all(isinstance(doc_id, int) for doc_id in pair_score.pair_scores[1].keys())

    @pytest.mark.parametrize(
        "qid, expected_scores",
        [
            (1, {101: 0.8, 102: 0.6}),
            (2, {201: 0.9, 202: 0.7}),
        ],
    )
    @patch("light_splade.data.pair_score.NdjsonLoader")
    def test___getitem___success(
        self,
        mock_loader_class: Mock,
        qid: int,
        expected_scores: dict[int, float],
    ) -> None:
        def mock_data() -> Generator[dict, None, None]:
            yield {"qid": 1, "scores": {"101": 0.8, "102": 0.6}}
            yield {"qid": 2, "scores": {"201": 0.9, "202": 0.7}}

        mock_loader = Mock()
        mock_loader.return_value = mock_data()
        mock_loader_class.return_value = mock_loader

        pair_score = PairScore(Path("test_path"))

        assert pair_score[qid] == expected_scores

    @patch("light_splade.data.pair_score.NdjsonLoader")
    def test___getitem___key_error(self, mock_loader_class: Mock) -> None:
        def mock_data() -> Generator[dict, None, None]:
            yield {"qid": 1, "scores": {"101": 0.8, "102": 0.6}}

        mock_loader = Mock()
        mock_loader.return_value = mock_data()
        mock_loader_class.return_value = mock_loader

        pair_score = PairScore(Path("test_path"))

        with pytest.raises(IndexError) as exc_info:
            _ = pair_score[999]

        assert "The qid 999 does not exist in hard-negative scores" in str(exc_info.value)
