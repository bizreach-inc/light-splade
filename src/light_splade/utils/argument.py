# NOTE: need to typing, transformers, light_splade for dynamic import.
# Do not remove them even if they are not explicitly used in this file.
import dataclasses
import re
import typing  # noqa
from types import UnionType
from typing import Any
from typing import Type

import transformers  # noqa
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig
from omegaconf import ListConfig

import light_splade  # noqa

RE_OPTIONAL = re.compile(r"typing.Optional\[(.*)\]")
RE_UNION = re.compile(r"typing.Union\[(.*)\]")


def instantiate(data_class: Type, dict_config: DictConfig) -> Any:
    """Instantiate a config object from hydra's DictConfig to avoid hydra's `ConfigValueError: Unions of containers are
    not supported:` error.

    Args:
        data_class (Type): Dataclass type to instantiate.
        dict_config (DictConfig): Hydra configuration object.

    Returns:
        Any: An instance of ``data_class`` constructed from ``dict_config``.
    """
    fields = dataclasses.fields(data_class)

    params = dict()
    for field in fields:
        if field.name not in dict_config:
            continue

        generic_types = get_generic_types(field.type)
        has_dataclass_result = has_dataclass(generic_types)
        if len(generic_types) > 1 and has_dataclass_result:
            raise ValueError(f"Not support a union of a dataclass and other types at \n{field}")

        if has_dataclass_result:
            params[field.name] = instantiate(generic_types[0], dict_config[field.name])
        else:
            value = dict_config[field.name]
            if not isinstance(value, ListConfig):
                params[field.name] = value
            else:
                params[field.name] = hydra_instantiate(value, _convert_="all")

    return data_class(**params)


def get_generic_types(type_: Any) -> list[Any]:
    """Return resolved generic types for a possibly-annotated type.

    The function supports ``typing.Union``, ``typing.Optional`` and Python 3.10 union syntax. It returns a list of
    concrete types (excluding ``None``).

    Args:
        type_: A possibly parametrized type annotation.

    Returns:
        list[Any]: Resolved type objects contained in the annotation.
    """
    generic_types = []
    type_str = str(type_)
    if isinstance(type_, UnionType):
        for type_str in str(type_).split("|"):
            type_str = type_str.strip()
            if type_str != "None":
                generic_types.append(eval(type_str))
    elif (matches := RE_OPTIONAL.search(type_str)) is not None:
        type_name = matches.group(1)
        generic_types.append(eval(type_name))
    elif (matches := RE_UNION.search(type_str)) is not None:
        type_names = matches.group(1).split(",")
        for type_name in type_names:
            if type_name != "NoneType":
                generic_types.append(eval(type_name))
    else:
        generic_types.append(type_)
    return generic_types


def has_dataclass(types: list[Any]) -> bool:
    """Return True when any of the provided types is a dataclass.

    Args:
        types: List of type objects to test.

    Returns:
        bool: True if at least one element of ``types`` is a dataclass type.
    """
    for type_ in types:
        if dataclasses.is_dataclass(type_):
            return True
    return False
