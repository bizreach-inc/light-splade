import json
import tempfile
from pathlib import Path

import pytest

from light_splade.data.ndjson_loader import NdjsonLoader
from light_splade.data.ndjson_loader import _get_file_list
from light_splade.data.ndjson_loader import _is_valid_filetype


@pytest.mark.parametrize(
    "file_path, exp",
    [
        (Path("d"), False),
        (Path("path/to/data/a.ndjson"), True),
        (Path("path/to/data/a.ndjson.gz"), True),
        (Path("path/to/data/a.ndjson.gzip"), False),
        (Path("path/to/data/a.txt.gz"), False),
        (Path("path/to/data/a.b.c.ndjson.gz"), True),
        (Path("path/to/data/a.b.c.txt.gz"), False),
    ],
)
def test__is_valid_filetype(file_path: Path, exp: bool) -> None:
    assert _is_valid_filetype(file_path) == exp


@pytest.mark.parametrize(
    "init_files, exp_files_list",
    [
        ([], []),
        ([Path("a.txt")], []),
        ([Path("folder"), Path("folder/a.ndjson")], [Path("folder/a.ndjson")]),
        (
            [
                Path("folder"),
                Path("folder/a.ndjson"),
                Path("folder/b.ndjson.gz"),
                Path("folder/c"),
            ],
            [Path("folder/a.ndjson"), Path("folder/b.ndjson.gz")],
        ),
    ],
)
def test__get_file_list(init_files: list[Path], exp_files_list: list[Path]) -> None:
    if len(init_files) == 0:
        return
    if len(init_files) == 1:
        init_files[0].touch()
    else:
        init_files[0].mkdir(exist_ok=False)
        for file_path in init_files[1:]:
            file_path.touch()
    file_list = _get_file_list(init_files[0])

    # tear down
    for file_path in init_files[::-1]:
        if file_path.is_dir():
            file_path.rmdir()
        else:
            file_path.unlink()

    assert file_list == exp_files_list


class TestNdjsonLoader:
    def test___init___with_valid_ndjson_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ndjson", delete=False) as f:
            test_data = [
                {"id": 1, "text": "first line"},
                {"id": 2, "text": "second line"},
            ]
            for item in test_data:
                f.write(json.dumps(item) + "\n")
            temp_path = Path(f.name)

        try:
            loader = NdjsonLoader(temp_path)
            assert len(loader._file_list) == 1
            assert loader._file_list[0] == temp_path
        finally:
            temp_path.unlink()

    def test___init___with_valid_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            file1 = temp_path / "test1.ndjson"
            file2 = temp_path / "test2.ndjson.gz"

            file1.touch()
            file2.touch()

            loader = NdjsonLoader(temp_path)

            assert len(loader._file_list) == 2
            assert set(loader._file_list) == {file1, file2}

    def test___init___invalid_path_raises_error(self) -> None:
        invalid_path = Path("/nonexistent/path")

        with pytest.raises(ValueError) as exc_info:
            NdjsonLoader(invalid_path)

        assert "is not valid" in str(exc_info.value)
        assert "must be a .ndjson or .ndjson.gz file" in str(exc_info.value)

    def test___init___empty_directory_raises_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with pytest.raises(ValueError) as exc_info:
                NdjsonLoader(temp_path)

            assert "is not valid" in str(exc_info.value)

    def test___call___with_ndjson_file(self) -> None:
        test_data = [
            {"id": 1, "text": "first line"},
            {"id": 2, "text": "second line"},
            {"id": 3, "text": "third line"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ndjson", delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
            temp_path = Path(f.name)

        try:
            loader = NdjsonLoader(temp_path)
            loaded_data = list(loader())

            assert loaded_data == test_data
        finally:
            temp_path.unlink()

    def test___call___with_multiple_files(self) -> None:
        test_data1 = [
            {"id": 1, "text": "file1 line1"},
            {"id": 2, "text": "file1 line2"},
        ]
        test_data2 = [
            {"id": 3, "text": "file2 line1"},
            {"id": 4, "text": "file2 line2"},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            file1 = temp_path / "test1.ndjson"
            file2 = temp_path / "test2.ndjson"

            with open(file1, "w") as f:
                for item in test_data1:
                    f.write(json.dumps(item) + "\n")

            with open(file2, "w") as f:
                for item in test_data2:
                    f.write(json.dumps(item) + "\n")

            loader = NdjsonLoader(temp_path)
            loaded_data = list(loader())

            # Data from both files should be loaded
            assert len(loaded_data) == 4
            assert set(json.dumps(item, sort_keys=True) for item in loaded_data) == set(
                json.dumps(item, sort_keys=True) for item in test_data1 + test_data2
            )
