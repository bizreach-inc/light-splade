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
from typing import Any

import yaml


def load_yaml(file_path: str | Path) -> dict[Any, Any]:
    """Load YAML content from a file path.

    Args:
        file_path (str | Path): Path to a YAML file.

    Returns:
        dict[Any, Any]: Parsed YAML content as a dictionary.
    """
    with open(file_path, encoding="utf-8") as f:
        data: dict = yaml.safe_load(f)
        return data


def ensure_path(path: str | Path) -> Path:
    """Ensure that the given path is returned as a :class:`Path`.

    Args:
        path (str | Path): The path to convert.

    Returns:
        Path: A ``pathlib.Path`` instance representing ``path``.
    """
    if not isinstance(path, Path):
        path = Path(path)
    return path
