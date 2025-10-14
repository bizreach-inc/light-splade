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
