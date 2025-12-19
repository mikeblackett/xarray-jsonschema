import re
from abc import ABC
from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from enum import EnumType, IntEnum, StrEnum
from functools import cached_property, singledispatch
from re import Pattern
from types import NoneType
from typing import Any, Type

import numpy as np

__all__ = [
    'AnyNormalizer',
    'ArrayNormalizer',
    'BooleanNormalizer',
    'ConstNormalizer',
    'EnumNormalizer',
    'IntegerNormalizer',
    'NotNormalizer',
    'NullNormalizer',
    'NumberNormalizer',
    'ObjectNormalizer',
    'Normalizer',
    'StringNormalizer',
    'TypeNormalizer',
    'as_schema',
]


KEYWORDS = {
    'anchor': '$anchor',
    'comment': '$comment',
    'id': '$id',
    'ref': '$ref',
    'schema': '$schema',
    'vocabulary': '$vocabulary',
}
"""Mapping from python strings to JSON Schema keywords."""


class Sentinels:
    ELLIPSIS = Ellipsis


def normalize_keyword(keyword: str):
    """Encode a Python-formatted keyword to JSON Schema."""
    result = _snake_case_to_camel_case(keyword)
    return KEYWORDS.get(result, result)


@singledispatch
def normalize_value(value: Any) -> Any:
    """Normalize a Python value to JSON Schema."""
    return value  # Default: identity


@normalize_value.register
def _(value: str) -> str:
    # ``str`` needs to override Sequence encoder...
    return value


@normalize_value.register
def _(value: Sequence) -> list:
    return list(normalize_value(v) for v in value)


@normalize_value.register
def _(value: set) -> list:
    return list(normalize_value(v) for v in value)


@normalize_value.register
def _(value: StrEnum) -> str:
    return value.value


@normalize_value.register
def _(value: IntEnum) -> int:
    return value.value


@normalize_value.register
def _(value: EnumType) -> list:
    return [normalize_value(v) for v in value.__members__.values()]


@normalize_value.register
def _(value: np.dtype) -> str:
    # This is consistent with  ``xarray.DataArray.to_dict()``
    return str(value)


@normalize_value.register
def _(value: re.Pattern) -> str:
    return value.pattern


@normalize_value.register
def _(value: type) -> str:
    if issubclass(value, str):
        return 'string'
    elif issubclass(value, bool):
        return 'boolean'
    elif issubclass(value, int):
        return 'integer'
    elif issubclass(value, float):
        return 'number'
    elif issubclass(value, Mapping):
        return 'object'
    elif issubclass(value, Sequence) and not issubclass(value, str):
        return 'array'
    elif issubclass(value, np.ndarray):
        return 'array'
    elif issubclass(value, NoneType):
        return 'null'
    raise TypeError(
        f'Error encoding python type {value!r} as JSON Schema type.'
    )


@normalize_value.register
def _(value: np.ndarray) -> list:
    return value.tolist()


def as_schema(obj: 'Normalizer') -> dict:
    """Return the fields of a `Normalizer` instance as a JSON Schema"""
    return asdict(obj, dict_factory=_schema_factory)


@dataclass(frozen=True, kw_only=True)
class Normalizer(ABC):
    """Abstract base class for normalizing Python objects to JSON Schema.

    Each Normalizer should correspond to a JSON Schema Data Type. The attributes
    of a `Normalizer` class should correspond to JSON Schema keywords. The values
    corresponding to the keywords can be Python primitives, or other `Normalizer`s.

    Keyword attributes should be declared in snake_case without prefixes, they
    will be converted to JSON Schema when normalizing.

    Attributes
    ----------
    title : str, optional
        A short description of the instance described by this schema.
    description : str, optional
        A description about the purpose of the instance described by this schema.
    comment : str, optional
        Notes that may be useful to future editors of a JSON schema, but should
        not be used to communicate to users of the schema.
    """

    title: str | None = None
    description: str | None = None
    comment: str | None = None

    @cached_property
    def schema(self) -> dict[str, Any]:
        """Return the fields of this instance as a JSON Schema"""
        return as_schema(self)

    def __or__(self, other) -> 'Normalizer':
        if not isinstance(other, type(self)):
            raise TypeError(
                'Normalizer objects are only unionable with their own type;'
                f' expected {type(self).__name__}, got {type(other).__name__}.'
            )
        return self.__class__(**_get_kwargs(self) | _get_kwargs(other))  # type: ignore

    def __repr__(self):
        # repr signature with only non-default arguments
        args = [
            (f.name, getattr(self, f.name))
            for f in fields(self)
            if getattr(self, f.name) != f.default and f.init
        ]
        args_string = ', '.join(f'{name}={value}' for name, value in args)
        return f'{self.__class__.__name__}({args_string})'

    @classmethod
    def from_python(cls, value: Any) -> 'Normalizer':
        match value:
            case str() | int() | float() | bool() | None:
                return ConstNormalizer(value)
            case EnumType():
                return EnumNormalizer(value)
            case re.Pattern():
                return StringNormalizer(pattern=value.pattern)
            case (v, Sentinels.ELLIPSIS):
                return ArrayNormalizer(items=Normalizer.from_python(v))
            case Sequence():
                return ArrayNormalizer(
                    prefix_items=[Normalizer.from_python(v) for v in value]
                )
            case Mapping():
                return ObjectNormalizer(
                    properties={
                        key: Normalizer.from_python(value)
                        for key, value in value.items()
                        if isinstance(key, str)
                    }
                    or None,
                    pattern_properties={
                        key.pattern: Normalizer.from_python(value)
                        for key, value in value.items()
                        if isinstance(key, re.Pattern)
                    }
                    or None,
                )
            case type():
                return TypeNormalizer(value)
            case _:
                raise NotImplementedError


@dataclass(frozen=True, kw_only=True, repr=False)
class EnumNormalizer(Normalizer):
    """Normalizer for enum type

    Attributes
    ----------
    enum : Iterable[Any]
        An iterable describing a fixed set of acceptable values.
        The iterable must contain at least one value and each value must be unique.
    """

    enum: Iterable[Any] = field(kw_only=False)


@dataclass(frozen=True, kw_only=True, repr=False)
class ObjectNormalizer(Normalizer):
    """Normalizer for mapping type

    Attributes
    ----------
    properties : Mapping[str, Normalizer] | None, default None
        A mapping where each key is the name of a property and each value is a
        `Normalizer` used to validate that property.
    pattern_properties : Mapping[str | re.Pattern, Normalizer] | None, default None
        A mapping where each key is a regular expression used to match the name
        of a property and each value is a `Normalizer` used to validate that
        property.
    additional_properties : Normalizer | bool | None, default None
        A schema that will be used to validate any properties in the instance
        that are not matched by `properties` or `patternProperties`. Boolean
        values can be used to allow/disallow any additional properties.
    required : Iterable[str] | None, default None
        An iterable of zero or more unique strings describing a list of
        required properties. Any properties not included in this list are treated
        as optional.
    required_pattern_properties: Iterable[str] | None, default None
        An iterable of zero or more regex strings describing a list of
        required pattern properties.
    max_properties : int | None, default None
        A non-negative integer used to restrict the number of properties on an
        object.
    min_properties : int | None, default None
        A non-negative integer used to restrict the number of properties on an
        object.
    """

    # TODO: (mike) replace `additionalProperties` keyword with `unevaluatedProperties`
    # TODO: (mike) add `propertyNames` keyword

    type: str = field(default='object', init=False)

    properties: Mapping[str, Normalizer] | None = None
    pattern_properties: Mapping[str | re.Pattern, Normalizer] | None = None
    property_names: Normalizer | None = None
    additional_properties: Normalizer | bool | None = None
    required: Iterable[str] | None = None
    required_pattern_properties: Iterable[str] | None = None
    max_properties: int | None = None
    min_properties: int | None = None

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Container) and _is_container_empty(value):
                object.__setattr__(self, f.name, None)


@dataclass(frozen=True, kw_only=True, repr=False)
class ArrayNormalizer(Normalizer):
    """Normalizer for sequence types.

    Attributes
    ----------
    items : Normalizer | bool | None, default None
        A single schema that will be used to validate all the items in the array.
        The empty array is always valid.
    prefix_items : Sequence[Normalizer] | None, default None
        An array, where each item is a `Normalizer` that corresponds to each
        index of the instance's array. Passing an empty array is equivalent to
        passing ``None``.
    unevaluated_items : Normalizer | None, default None
        A schema that applies to any values not evaluated by the `items`,
        `prefix_items`, or `contains` keyword.
    contains : Normalizer | None, default None
        A schema that only needs to validate against one or more items in the
        array.
    max_contains : int | None, default None
        A non-negative integer used to restrict the number of times a schema
        matches a `contains` constraint.
    min_contains : int | None, default None
        A non-negative integer used to restrict the number of times a schema
        matches a `contains` constraint.
    max_items : int | None, default None
        A non-negative integer used to restrict the number of items in an array.
    min_items : int | None, default None
        A non-negative integer used to restrict the number of items in an array.
    """

    type: str = field(default='array', init=False)

    items: Normalizer | bool | None = None
    prefix_items: Sequence[Normalizer] | None = None
    unevaluated_items: Normalizer | None = None
    contains: Normalizer | None = None
    max_contains: int | None = None
    min_contains: int | None = None
    max_items: int | None = None
    min_items: int | None = None

    def __post_init__(self):
        if self.prefix_items is not None and len(self.prefix_items) == 0:
            object.__setattr__(self, 'prefix_items', None)


@dataclass(frozen=True, kw_only=True, repr=False)
class StringNormalizer(Normalizer):
    """Normalizer for the string type.

    Attributes
    ----------
    max_length : int | None, default None
        A non-negative integer used to restrict the number of characters in a
        string.
    min_length : int | None, default None
        A non-negative integer used to restrict the number of characters in a
        string.
    pattern : str | Pattern[str] | None, default None
        A pattern used to restrict a string to a particular regular expression.
    """

    type: str = field(default='string', init=False)

    max_length: int | None = None
    min_length: int | None = None
    pattern: str | Pattern[str] | None = None


@dataclass(frozen=True, kw_only=True, repr=False)
class IntegerNormalizer(Normalizer):
    """Normalizer for the integer type.

    Attributes
    ----------
    multiple_of : int | float | None, default None
        A positive number used to restrict the instance to a multiple of a
        given number
    maximum : int | None, default None
        A number used to restrict the instance to a maximum value.
    minimum : int | None, default None
        A number used to restrict the instance to a minimum value.
    exclusive_maximum : int | None, default None
        A number used to restrict the instance to a maximum value.
    exclusive_minimum : int | None, default None
        A number used to restrict the instance to a minimum value.
    """

    type: str = field(default='integer', init=False)

    multiple_of: int | None = None
    maximum: int | None = None
    minimum: int | None = None
    exclusive_maximum: int | None = None
    exclusive_minimum: int | None = None


@dataclass(frozen=True, kw_only=True, repr=False)
class NumberNormalizer(Normalizer):
    """Normalizer for numeric type, either integers or floating point numbers.

    Attributes
    ----------
    multiple_of : int | float | None, default None
        A positive number used to restrict the instance to a multiple of a
        given number
    maximum : int | float | None, default None
        A number used to restrict the instance to a maximum value.
    minimum : int | float | None, default None
        A number used to restrict the instance to a minimum value.
    exclusive_maximum : int | float | None, default None
        A number used to restrict the instance to a maximum value.
    exclusive_minimum : int | float | None, default None
        A number used to restrict the instance to a minimum value.
    """

    type: str = field(default='number', init=False)

    multiple_of: int | float | None = None
    maximum: int | float | None = None
    minimum: int | float | None = None
    exclusive_maximum: int | float | None = None
    exclusive_minimum: int | float | None = None


@dataclass(frozen=True, repr=False, kw_only=False)
class NullNormalizer(Normalizer):
    """Normalizer for null type"""

    type: str = field(default='null', init=False)


@dataclass(frozen=True, repr=False, kw_only=False)
class BooleanNormalizer(Normalizer):
    """Normalizer for boolean type"""

    type: str = field(default='boolean', init=False)


@dataclass(frozen=True, repr=False, kw_only=True)
class ConstNormalizer(Normalizer):
    """Normalizer for constant type

    Attributes
    ----------

    const : Any
        The expected value of the instance
    """

    const: Any = field(kw_only=False)


@dataclass(frozen=True, repr=False, kw_only=True)
class AnyNormalizer(Normalizer):
    """Normalizer for any type"""

    ...


@dataclass(frozen=True, repr=False, kw_only=True)
class TypeNormalizer(Normalizer):
    """Normalizer for data type keyword"""

    type_: Type = field(kw_only=False)


@dataclass(frozen=True, repr=False, kw_only=True)
class AllOfNormalizer(Normalizer):
    all_of: Iterable[Normalizer] = field(kw_only=False)


@dataclass(frozen=True, repr=False, kw_only=True)
class AnyOfNormalizer(Normalizer):
    any_of: Iterable[Normalizer] = field(kw_only=False)


@dataclass(frozen=True, repr=False, kw_only=True)
class OneOfNormalizer(Normalizer):
    one_of: Iterable[Normalizer] = field(kw_only=False)


@dataclass(frozen=True, repr=False, kw_only=True)
class NotNormalizer(Normalizer):
    not_: Normalizer = field(kw_only=False)


def _schema_factory(data: list[tuple[str, Any]]):
    """Custom dict_factory for dataclasses.asdict method."""
    return {
        normalize_keyword(k): normalize_value(v)
        for k, v in data
        if v is not None
    }


def _get_kwargs(obj: Normalizer):
    """Return the keyword arguments from a Normalizer object."""
    return {f.name: getattr(obj, f.name) for f in fields(obj) if f.init}


def _is_container_empty(value: Container) -> bool:
    return not bool(value)


def _snake_case_to_camel_case(string: str) -> str:
    if '_' not in string:
        return string
    string = ''.join(word.title() for word in string.split('_'))
    return string[0].lower() + string[1:]
