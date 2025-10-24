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

"""Utilities for loading NDJSON and gzipped NDJSON files.

This module exposes :class:`NdjsonLoader`, a small helper that accepts a path to a single .ndjson/.ndjson.gz file or a
directory and yields decoded JSON objects for each line. The loader uses :mod:`tqdm` to provide a simple progress
indicator when iterating files.
"""

import gzip
import json
from pathlib import Path
from typing import Callable
from typing import Generator

from tqdm import tqdm


def _is_valid_filetype(file_path: Path) -> bool:
    """Return True if ``file_path`` refers to an NDJSON or gzipped NDJSON.

    Args:
        file_path (Path): Path to check.

    Returns:
        True if the combined suffixes end with ``.ndjson`` or ``.ndjson.gz``.
    """
    suffix = "".join(file_path.suffixes)
    return suffix.endswith(".ndjson") or suffix.endswith(".ndjson.gz")


def _get_file_list(data_path: Path) -> list[Path]:
    """Return a list of NDJSON files from ``data_path``.

    If ``data_path`` is a directory, files with extensions ``.ndjson`` and ``.ndjson.gz`` are returned. If it is a file
    and has a valid suffix, a single-element list is returned.
    """
    file_list: list[Path] = []
    if not data_path.exists():
        return file_list
    if data_path.is_dir():
        file_list = list(data_path.glob("*.ndjson")) + list(data_path.glob("*.ndjson.gz"))
    elif _is_valid_filetype(data_path):
        file_list = [data_path]
    return file_list


class NdjsonLoader:
    """Iterator that yields parsed JSON objects from NDJSON files.

    Args:
        data_path (Path): Path to either a single NDJSON (or gzipped NDJSON) file or a directory containing such files.

    Raises:
        ValueError: If ``data_path`` does not contain any supported files.
    """

    def __init__(self, data_path: Path) -> None:
        self._file_list = _get_file_list(data_path)
        if len(self._file_list) == 0:
            raise ValueError(
                f"The data_path `{str(data_path)}` is not valid "
                "(must be a .ndjson or .ndjson.gz file, or a folder which "
                "contain at least 1 such file)"
            )

    def __call__(self) -> Generator[dict, None, None]:
        """Yield dictionaries parsed from each NDJSON line.

        Yields:
            Decoded JSON objects (as Python dicts) for every line in the
            discovered NDJSON files.
        """
        for file_path in self._file_list:
            is_gzip = file_path.suffixes[-1] == ".gz"
            open_func: Callable = gzip.open if is_gzip else open
            with open_func(file_path, "rt") as f:
                for line in tqdm(f):
                    item = json.loads(line)
                    yield item
