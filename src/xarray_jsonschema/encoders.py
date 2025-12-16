import re
from collections.abc import Mapping, Sequence
from enum import EnumType, IntEnum, StrEnum
from functools import singledispatch
from types import NoneType
from typing import Any, Type

import numpy as np

ENCODE_KEYWORDS = {
    'anchor': '$anchor',
    'comment': '$comment',
    'id': '$id',
    'ref': '$ref',
    'schema': '$schema',
    'vocabulary': '$vocabulary',
}

DECODE_KEYWORDS = {
    '$anchor': 'anchor',
    '$comment': 'comment',
    '$id': 'id',
    '$ref': 'ref',
    '$schema': 'schema',
    '$vocabulary': 'vocabulary',
}


def encode_keyword(keyword: str):
    """Encode a Python-formatted keyword to JSON Schema."""
    result = _snake_case_to_camel_case(keyword)
    return ENCODE_KEYWORDS.get(result, result)


@singledispatch
def encode_value(value: Any) -> Any:
    """Encode a Python value to JSON Schema."""
    return value  # Default: identity


@encode_value.register
def _(value: str) -> str:
    # ``str`` needs to override Sequence encoder...
    return value


@encode_value.register
def _(value: Sequence) -> list:
    return list(value)


@encode_value.register
def _(value: set) -> list:
    return list(value)


@encode_value.register
def _(value: StrEnum) -> str:
    return value.value


@encode_value.register
def _(value: IntEnum) -> int:
    return value.value


@encode_value.register
def _(value: EnumType) -> list:
    return [member for member in value.__members__.values()]


@encode_value.register
def _(value: np.dtype) -> str:
    # This is consistent with  ``xarray.DataArray.to_dict()``
    return str(value)


@encode_value.register
def _(value: re.Pattern) -> str:
    return value.pattern


@encode_value.register
def _(value: type) -> str:
    return _encode_type(value)


@encode_value.register
def _(value: np.ndarray) -> list:
    return value.tolist()


def _encode_type(type_: Type) -> str:
    if issubclass(type_, str):
        return 'string'
    elif issubclass(type_, bool):
        return 'boolean'
    elif issubclass(type_, int):
        return 'integer'
    elif issubclass(type_, float):
        return 'number'
    elif issubclass(type_, Mapping):
        return 'object'
    elif issubclass(type_, Sequence) and not issubclass(type_, str):
        return 'array'
    elif issubclass(type_, np.ndarray):
        return 'array'
    elif issubclass(type_, NoneType):
        return 'null'
    raise TypeError(
        f'Error encoding python type {type_!r} as JSON Schema type.'
    )


def _snake_case_to_camel_case(string: str) -> str:
    if '_' not in string:
        return string
    string = ''.join(word.title() for word in string.split('_'))
    return string[0].lower() + string[1:]
