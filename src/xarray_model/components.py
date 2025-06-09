from typing_extensions import assert_never
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Iterable

import numpy as np
from numpy.typing import DTypeLike

from xarray_model.base import BaseSchema
from xarray_model.encoders import encode_value
from xarray_model.serializers import (
    AnySerializer,
    ArraySerializer,
    ConstSerializer,
    EnumSerializer,
    IntegerSerializer,
    NullSerializer,
    ObjectSerializer,
    Serializer,
    StringSerializer,
    TypeSerializer,
)

__all__ = [
    'Attrs',
    'Attr',
    'Chunks',
    'DType',
    'Dims',
    'Name',
    'Shape',
    'Size',
]


@dataclass(frozen=True, kw_only=True, repr=False)
class _Chunk(BaseSchema):
    """A JSON Schema generator for chunk sizes.

    This model represents the tuple of block sizes for a single dimension, it
    is supposed to be used inside the Chunks schema.
    """

    shape: int | Sequence[int] = field(kw_only=False)

    @cached_property
    def serializer(self) -> Serializer:
        match self.shape:
            case -1:
                # `-1` is a dask wildcard meaning "use the full dimension size."
                return ArraySerializer(
                    items=IntegerSerializer(), min_items=1, max_items=1
                )
            case int():
                # A single integer is used to represent a uniform chunk size
                # where all chunks should be equal to `shape` except the last one,
                # which is free to vary.
                # TODO: (mike) this schema doesn't validate that the
                #  variable chunk is last in the array. That is currently not
                #  possible with JSON Schema (https://github.com/json-schema-org/json-schema-spec/issues/1060).
                #  We could opt for custom validation...
                return ArraySerializer(
                    # First chunk is equal to `shape`.
                    prefix_items=[ConstSerializer(self.shape)],
                    # Other chunks are `shape` or smaller...
                    items=IntegerSerializer(maximum=self.shape),
                    # but there is only one "other" chunk.
                    contains=IntegerSerializer(exclusive_maximum=self.shape),
                    max_contains=1,
                    min_contains=0,
                )
            case Sequence() if not isinstance(self.shape, str):
                # Expect a full match of the provided shape to the chunk size.
                # Can use -1 as a wildcard.
                prefix_items = [
                    IntegerSerializer()
                    if size == -1
                    else ConstSerializer(size)
                    for size in self.shape
                ]
                return ArraySerializer(
                    prefix_items=prefix_items or None,
                    items=False,
                )
            case _:
                raise ValueError(
                    f'Invalid shape: {self.shape}. '
                    'Expected int or sequence of ints.'
                )


@dataclass(frozen=True, kw_only=False, repr=False)
class Chunks(BaseSchema):
    """A JSON Schema generator for DataArray.chunks

    Attributes
    ----------
    chunks : bool | int | Sequence[int | Sequence[int]], default True
        The expected chunk sizes along each dimension. The validation logic depends on the argument type:
        - A boolean simply validates whether the array is chunked or not;
        - An positive integer validates that **all** dimensions have the same uniform chunk size (all but the last chunk are equal to ``chunks``);
        - A sequence of positive integers validates that each dimension has the specified uniform chunk size;
        - A sequence of sequences of positive integers validates exact chunk sizes along each dimension.
        - A value of ``-1`` can be used in place of a positive integer to validate no chunking along a dimension.
    """

    chunks: bool | int | Sequence[int | Sequence[int]] = field(default=True)

    @cached_property
    def serializer(self) -> Serializer:
        match self.chunks:
            case True:
                return ArraySerializer(
                    items=ArraySerializer(items=IntegerSerializer())
                )
            case False:
                return NullSerializer()
            case int():
                return ArraySerializer(
                    items=self.chunks.serializer
                    if isinstance(self.chunks, _Chunk)
                    else _Chunk(self.chunks).serializer
                )
            case Sequence() if not isinstance(self.chunks, str):
                prefix_items = [
                    chunk.serializer
                    if isinstance(chunk, _Chunk)
                    else _Chunk(chunk).serializer
                    for chunk in self.chunks
                ]
                return ArraySerializer(
                    prefix_items=prefix_items or None,
                    items=False,
                )
            case _:  # pragma: no cover
                assert_never(self.chunks)  # type: ignore


@dataclass(frozen=True, kw_only=True, repr=False)
class Size(BaseSchema):
    """A JSON Schema generator for dimension sizes.

    This model should be composed with the Shape model.

    Attributes
    ----------
    size : int | None, default None
        A non-negative integer specifying the expected size of the dimension.
        The default value of ``None`` will validate any size.
    multiple_of : int | None, default None
        Restrict the size of the dimension to multiples of a given number.
    maximum : int | None, default None
        A non-negative integer specifying the maximum size of the dimension.
    minimum : int | None, default None
        A non-negative integer specifying the minimum size of the dimension.
    """

    size: int | None = field(default=None, kw_only=False)
    multiple_of: int | None = None
    maximum: int | None = None
    minimum: int | None = None

    @cached_property
    def serializer(self) -> Serializer:
        match self.size:
            case None:
                return IntegerSerializer(
                    multiple_of=self.multiple_of,
                    maximum=self.maximum,
                    minimum=self.minimum,
                )
            case int():
                return ConstSerializer(self.size)
            case _:  # pragma: no cover
                assert_never(self.size)


@dataclass(frozen=True, kw_only=True, repr=False)
class Shape(BaseSchema):
    """A JSON Schema generator for DataArray.shape

    Attributes
    ----------
    shape : Sequence[int | Size] | None, default None
        Sequence of expected sizes for the dimensions of the array. The
        default value of ``None`` will match any shape.
    min_dims : int | None, default None
        Minimum expected number of dimensions i.e., the minimum sequence length.
    max_dims : int | None, default None
        Maximum expected number of dimensions i.e., the maximum sequence length.

    Examples
    --------
    >>> da = xr.DataArray(np.random.rand(5, 4), dims=['x', 'y'])

    >>> xm.Shape([5, 4]).validate(da.shape)
    >>> xm.Shape(min_items=1, max_items=2).validate(da.shape)
    >>> xm.Shape([xm.Size(5), xm.Size(4)]).validate(da.shape)
    >>> xm.Shape([xm.Size(maximum=5), xm.Size(multiple_of=2)]).validate(da.shape)
    """

    shape: Sequence[int | Size] | None = field(default=None, kw_only=False)
    min_dims: int | None = None
    max_dims: int | None = None

    @cached_property
    def serializer(self) -> Serializer:
        prefix_items = None
        min_dims = self.min_dims

        match self.shape:
            case None:
                items = IntegerSerializer()
            case Sequence():
                items = False
                prefix_items = [
                    size.serializer
                    if isinstance(size, Size)
                    else Size(size).serializer
                    for size in self.shape
                ]
                min_dims = len(prefix_items)
            case _:  # pragma: no cover
                assert_never(self.shape)

        return ArraySerializer(
            prefix_items=prefix_items,
            items=items,
            min_items=min_dims,
            max_items=self.max_dims,
        )


@dataclass(frozen=True, kw_only=True, repr=False)
class DType(BaseSchema):
    """A JSON Schema generator for DataArray.dtype.

    Attributes
    ----------
    dtype : DTypeLike | None, default None
        The expected data type of the array. This can be any dtype-like value
        accepted by `numpy.dtype`. The default value of ``None`` will match
        any dtype.
    """

    # TODO: (mike) support numpy subdytpes
    dtype: DTypeLike | None = field(default=None, kw_only=False)

    @cached_property
    def serializer(self) -> Serializer:
        if self.dtype is None:
            return StringSerializer()
        return ConstSerializer(np.dtype(self.dtype))


@dataclass(frozen=True, kw_only=True, repr=False)
class Name(BaseSchema):
    """A JSON Schema for validator DataArray.name.
    Attributes
    ----------
    name : str | Sequence[str] | None, default None
        Expected value, sequence of acceptable values, or regex pattern to
        match the name against. The default value of ``None`` will match any
        name.
    regex : bool, default False
        Flag to indicate that the ``name`` parameter should be treated as a
        regex pattern.
    min_length : int, default None
        Non-negative integer specifying the minimum length of the name.
    max_length : int, default None
        Non-negative integer specifying the maximum length of the name.

    Examples
    --------
    >>> da = xarray.DataArray(np.arange(5), dims=['x'], name='foo')
    # Validate an exact match
    >>> Name('foo').validate(da.name)
    # Validate a regex pattern
    >>> Name(r'^fo{2}$', regex=True).validate(da.name)
    # Validate a sequence of acceptable values
    >>> Name(['foo', 'bar']).validate(da.name)
    # Length constraints
    >>> Name(min_length=3).validate(da.name)
    """

    name: str | Sequence[str] | None = field(default=None, kw_only=False)
    regex: bool | None = None
    min_length: int | None = None
    max_length: int | None = None

    @cached_property
    def serializer(self) -> Serializer:
        match self.name:
            case None:
                return StringSerializer(
                    min_length=self.min_length,
                    max_length=self.max_length,
                )
            case str() if self.regex:
                return StringSerializer(
                    pattern=self.name,
                    min_length=self.min_length,
                    max_length=self.max_length,
                )
            case str():
                return ConstSerializer(self.name)
            case Sequence():
                return EnumSerializer(self.name)
            case _:  # pragma: no cover
                assert_never(self.name)


@dataclass(frozen=True, kw_only=True, repr=False)
class Dims(BaseSchema):
    """A JSON Schema generator for DataArray.dims.

    Attributes
    ----------
    dims : Sequence[str | Name] | None, default None
        A sequence of expected names for the dimensions. The
        names can either be strings or instances of `Name` models for more
        complex matching. The default value of ``None`` will match any sequence.
    contains : str | Name | None, default None
        A string or `Name` model describing a name that must be included in the
        dimensions.
    max_dims : int | None, default None
        The maximum number of dimensions.
    min_dims : int | None, default None
        The minimum number of dimensions.

    See Also
    --------
    Name : DataArray name validation model
    """

    dims: Sequence[str | Name] | None = field(kw_only=False, default=None)
    contains: str | Name | None = None
    max_dims: int | None = None
    min_dims: int | None = None

    def _coerce_to_name(self, name: str | Name) -> Name:
        return name if isinstance(name, Name) else Name(name)

    @cached_property
    def serializer(self) -> Serializer:
        prefix_items = (
            [self._coerce_to_name(name).serializer for name in self.dims]
            if self.dims
            else None
        )
        # All prefix items are required
        min_items = (
            len(prefix_items)
            if self.min_dims is None and prefix_items
            else self.min_dims
        )
        # Additional items are not allowed
        items = False if prefix_items else StringSerializer()
        contains = (
            self._coerce_to_name(self.contains).serializer
            if self.contains
            else None
        )
        return ArraySerializer(
            prefix_items=prefix_items,
            items=items,
            contains=contains,
            max_items=self.max_dims,
            min_items=min_items,
        )


@dataclass(frozen=True, kw_only=True, repr=False)
class Attr(BaseSchema):
    """A JSON Schema generator for xarray metadata key-value pairs.

    Attributes
    ----------
    key : str
        The expected name of the attribute. The name can be a regex pattern if
        the ``regex`` flag is set to ``True``.
    value : Any | None, default None
        The expected value of the attribute. The value can be a specific value
         for exact matching, or a Python type for more generic matching. The
         default value of ``None`` will match any value.
    regex : bool, default False
        A flag to indicate that the ``name`` parameter should be treated as a
        regex pattern.
    required: bool, default True
        A flag to indicate that the attribute is required. This attribute is
        silently ignored if the regex flag is set to ``True``.

    See Also
    --------
    Attrs : DataArray metadata attribute validation model
    """

    key: str = field(kw_only=False)
    regex: bool = False
    value: Any | None = None
    required: bool = True

    @cached_property
    def serializer(self) -> Serializer:
        match self.value:
            case type():
                return TypeSerializer(self.value)
            case None:
                return AnySerializer()
            case str():
                return ConstSerializer(self.value)
            case Mapping():
                # TODO: (mike) Implement nested dict validation
                warnings.warn(
                    'Nested dict attribute validation is not yet implemented.'
                )
                return ObjectSerializer()
            case Sequence():
                # TODO: (mike) Implement array validation
                warnings.warn(
                    'Array attribute validation is not yet implemented.'
                )
                return ArraySerializer()
            case _:
                return ConstSerializer(encode_value(self.value))


@dataclass(frozen=True, kw_only=True, repr=False)
class Attrs(BaseSchema):
    """A JSON Schema generator for xarray attrs.

    Attributes
    ----------
    attrs : Iterable[Attr]
        An iterable of ``Attr`` models describing the expected metadata key-value
        pairs.
    allow_extra_items : bool | None, default None
        A flag indicating whether items not described by the ``attrs`` parameter
        are allowed/disallowed. The default value of ``None`` is equivalent to
        ``True``.
    """

    attrs: Iterable[Attr] | None = field(default=None, kw_only=False)
    allow_extra_items: bool | None = None

    @cached_property
    def serializer(self) -> Serializer:
        if self.attrs is None:
            return ObjectSerializer(
                additional_properties=self.allow_extra_items
            )
        properties = {
            attr.key: attr.serializer for attr in self.attrs if not attr.regex
        }
        pattern_properties = {
            attr.key: attr.serializer for attr in self.attrs if attr.regex
        }
        required = {
            attr.key
            for attr in self.attrs
            if (attr.required and not attr.regex)
        }
        return ObjectSerializer(
            properties=properties or None,
            pattern_properties=pattern_properties or None,
            required=required or None,
            additional_properties=self.allow_extra_items,
        )
