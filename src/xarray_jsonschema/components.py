import re
import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import DTypeLike
from typing_extensions import assert_never

from xarray_jsonschema.base import XarraySchema
from xarray_jsonschema.encoders import encode_value
from xarray_jsonschema.serializers import (
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
    'AttrsSchema',
    'AttrSchema',
    'ChunksSchema',
    'DTypeSchema',
    'DimsSchema',
    'NameSchema',
    'ShapeSchema',
    'SizeSchema',
]

a: int = '1'

@dataclass(frozen=True, kw_only=True, repr=False)
class NameSchema(XarraySchema):
    """Validate the name of a xarray DataArray.

    Parameters
    ----------
    name : str | Iterable[str] | re.Pattern | None, default None
        Expected value, iterable of acceptable values, or regex pattern to
        match the name against.
    min_length : int, default None
        Non-negative integer specifying the minimum length of the name.
    max_length : int, default None
        Non-negative integer specifying the maximum length of the name.
    """

    name: str | Iterable[str] | re.Pattern | None = field(
        default=None, kw_only=False
    )
    min_length: int | None = None
    max_length: int | None = None

    @cached_property
    def serializer(self) -> Serializer:
        # TODO: (mike) support validating 'null' names?
        schema = StringSerializer(
            min_length=self.min_length,
            max_length=self.max_length,
        )
        match self.name:
            case None:
                pass
            case re.Pattern():
                schema |= StringSerializer(pattern=self.name)
            case str():
                schema = ConstSerializer(self.name)
            case Iterable():
                schema = EnumSerializer(self.name)
            case _:  # pragma: no cover
                raise TypeError(
                    f'unsupported "name" type: {type(self.name).__name__!r}'
                )
        return schema

    def validate(self, name: Hashable | None) -> None:
        return super()._validate(instance=name)


@dataclass(frozen=True, kw_only=True, repr=False)
class DimsSchema(XarraySchema):
    """Validate the dimensions of a xarray DataArray.

    Parameters
    ----------
    dims : Sequence[str | NameSchema | None] | None, default None
        A sequence of expected names for the dimensions. The
        names can either be strings or ``NameSchema`` objects for more
        complex matching.
    contains : str | NameSchema | None, default None
        A string or ``NameSchema`` object describing a name that must be included in the
        dimensions.
    max_dims : int | None, default None
        The maximum number of dimensions.
    min_dims : int | None, default None
        The minimum number of dimensions.

    See Also
    --------
    NameSchema
    ShapeSchema
    """

    dims: Sequence[str | NameSchema | None] | None = field(
        kw_only=False, default=None
    )
    contains: str | NameSchema | None = None
    max_dims: int | None = None
    min_dims: int | None = None

    @cached_property
    def serializer(self) -> Serializer:
        schema = ArraySerializer(
            items=StringSerializer(),
            max_items=self.max_dims,
            min_items=self.min_dims,
            contains=NameSchema.convert(self.contains).serializer
            if self.contains
            else None,
        )
        if self.dims is not None:
            prefix_items = [
                NameSchema.convert(name).serializer for name in self.dims
            ]
            # All prefix items are required
            min_items = (
                len(prefix_items)
                if self.min_dims is None and prefix_items
                else self.min_dims
            )
            # Additional items are not allowed
            items = False if prefix_items else StringSerializer()
            schema |= ArraySerializer(
                prefix_items=prefix_items,
                items=items,
                min_items=min_items,
            )
        return schema

    def validate(self, dims: Sequence[Hashable]) -> None:
        return super()._validate(instance=dims)


@dataclass(frozen=True, kw_only=True, repr=False)
class SizeSchema(XarraySchema):
    """Generate a JSON schema for xarray DataArray dimension sizes.

    This model should be composed with the :py:class:`~xarray_jsonschema.ShapeSchema` schema.

    Attributes
    ----------
    size : int | None, default None
        A non-negative integer specifying the expected size of the dimension.
        The default value of ``None`` will validate any size.
    maximum : int | None, default None
        A non-negative integer specifying the maximum size of the dimension.
    minimum : int | None, default None
        A non-negative integer specifying the minimum size of the dimension.
    """

    size: int | None = field(default=None, kw_only=False)
    maximum: int | None = None
    minimum: int | None = None

    @cached_property
    def serializer(self) -> Serializer:
        match self.size:
            case None:
                return IntegerSerializer(
                    maximum=self.maximum,
                    minimum=self.minimum,
                )
            case int():
                return ConstSerializer(self.size)
            case _:  # pragma: no cover
                raise TypeError(
                    f'unsupported "size" type: {type(self.name).__name__!r}'
                )

    def validate(self, _) -> None:
        raise NotImplementedError()


@dataclass(frozen=True, kw_only=True, repr=False)
class ShapeSchema(XarraySchema):
    """Generate a JSON schema for xarray DataArray shapes.

    Attributes
    ----------
    shape : Sequence[int | SizeSchema] | None, default None
        Sequence of expected sizes for the dimensions of the array. The
        default value of ``None`` will match any shape.
    min_dims : int | None, default None
        Minimum expected number of dimensions i.e., the minimum sequence length.
    max_dims : int | None, default None
        Maximum expected number of dimensions i.e., the maximum sequence length.
    """

    shape: Sequence[int | SizeSchema] | None = field(
        default=None, kw_only=False
    )
    min_dims: int | None = None
    max_dims: int | None = None

    @cached_property
    def serializer(self) -> Serializer:
        schema = ArraySerializer(
            items=IntegerSerializer(),
            min_items=self.min_dims,
            max_items=self.max_dims,
        )
        match self.shape:
            case None:
                pass
            case Sequence():
                prefix_items = [
                    SizeSchema.convert(size).serializer for size in self.shape
                ]
                schema |= ArraySerializer(
                    prefix_items=prefix_items,
                    items=False,
                    min_items=len(prefix_items),
                )
            case _:  # pragma: no cover
                assert_never(self.shape)

        return schema

    def validate(self, shape: Sequence[int]) -> None:
        return super()._validate(instance=shape)


@dataclass(frozen=True, kw_only=True, repr=False)
class DTypeSchema(XarraySchema):
    """Generate a JSON schema for xarray DataArray dtypes.

    Parameters
    ----------
    dtype : DTypeLike
        The expected data type of the array. This can be any dtype-like value
        accepted by ``numpy.dtype``.
    """

    dtype: DTypeLike = field(kw_only=False)

    @cached_property
    def serializer(self) -> Serializer:
        return ConstSerializer(np.dtype(self.dtype))

    def validate(self, dtype: np.dtype) -> None:
        return super()._validate(instance=dtype)


@dataclass(frozen=True, kw_only=True, repr=False)
class _ChunkSchema(XarraySchema):
    """Generate a JSON schema for xarray DataArray chunk block sizes.

    This model represents the tuple of block sizes for a single dimension, it
    is supposed to be used inside the ChunksSchema schema.
    """

    size: int | Sequence[int] = field(kw_only=False)

    @cached_property
    def serializer(self) -> Serializer:
        match self.size:
            case -1:
                # `-1` is a dask wildcard meaning "use the full dimension size."
                return ArraySerializer(
                    items=IntegerSerializer(), min_items=1, max_items=1
                )
            case int():
                # A single integer is used to represent a uniform chunk size
                # where all chunks should be equal to `size` except the last one,
                # which is free to vary.
                # TODO: (mike) this schema doesn't validate that the
                #  variable chunk is last in the array. That is currently not
                #  possible with JSON Schema (https://github.com/json-schema-org/json-schema-spec/issues/1060).
                #  We could opt for custom validation...
                return ArraySerializer(
                    # First chunk is equal to `shape`.
                    prefix_items=[ConstSerializer(self.size)],
                    # Other chunks are `shape` or smaller...
                    items=IntegerSerializer(maximum=self.size),
                    # but there is only one "other" chunk.
                    contains=IntegerSerializer(exclusive_maximum=self.size),
                    max_contains=1,
                    min_contains=0,
                )
            case Sequence() if not isinstance(self.size, str):
                # Expect a full match of the provided shape to the chunk size.
                # Can use -1 as a wildcard.
                prefix_items = [
                    IntegerSerializer()
                    if size == -1
                    else ConstSerializer(size)
                    for size in self.size
                ]
                return ArraySerializer(
                    prefix_items=prefix_items,
                    items=False,
                )
            case _:
                raise ValueError(
                    f'Invalid size: {self.size}. '
                    'Expected int or sequence of ints.'
                )

    def validate(self, _) -> None:
        raise NotImplementedError()


@dataclass(frozen=True, kw_only=True, repr=False)
class ChunksSchema(XarraySchema):
    """Generate a JSON schema for xarray DataArray chunks.

    Parameters
    ----------
    chunks : bool | int | Sequence[int | Sequence[int]] | None, default None
        The expected chunking along each dimension. The schema logic depends on the argument :py:type:`type`:

        - :py:type:`bool` simply validates whether the array is chunked or not;
        - :py:type:`int` validates a **uniform** chunk size along **all** dimensions;
        - :py:type:`Sequence[int]` validates a **uniform** chunk size along **each** dimension;
        - :py:type:`Sequence[Sequence[int]]` validates an exact chunk size along each dimension;
        - A value of :py:const:`-1` can be used in place of a positive integer to validate no chunking along a dimension;
        - ``None`` validates any chunked/unchunked array.
    """

    chunks: bool | int | Sequence[int | Sequence[int]] | None = field(
        default=None, kw_only=False
    )

    @cached_property
    def serializer(self) -> Serializer:
        match self.chunks:
            case None:
                return AnySerializer()
            case True:
                return ArraySerializer(
                    items=ArraySerializer(items=IntegerSerializer())
                )
            case False:
                return NullSerializer()
            case int():
                return ArraySerializer(
                    items=_ChunkSchema(self.chunks).serializer
                )
            case Sequence():
                prefix_items = [
                    _ChunkSchema(chunk).serializer for chunk in self.chunks
                ]
                return ArraySerializer(
                    prefix_items=prefix_items,
                    items=False,
                )
            case _:  # pragma: no cover
                assert_never(self.chunks)  # type: ignore

    def validate(self, chunks: Sequence[Sequence[int]] | None) -> None:
        return super()._validate(instance=chunks)


@dataclass(frozen=True, kw_only=True, repr=False)
class AttrSchema(XarraySchema):
    """Generate a JSON schema for xarray DataArray attribute key-value pairs.

    Parameters
    ----------
    value : Any | None, default None
        The expected attribute value or type. The default value of ``None`` will
        match any value.
    required: bool, default True
        A boolean flag indicating that the attribute is required. This parameter
        is silently ignored if the ``key`` argument is a regex pattern.

    See Also
    --------
    AttrsSchema
    """

    value: Any | None = field(default=None, kw_only=False)
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

    def validate(self, _) -> None:
        raise NotImplementedError()


@dataclass(frozen=True, kw_only=True, repr=False)
class AttrsSchema(XarraySchema):
    """Generate a JSON schema for xarray DataArray attrs.

    Parameters
    ----------
    attrs : Mapping[str | re.Pattern, Any]
        An iterable of ``AttrSchema`` objects describing the expected metadata key-value
        pairs.
    allow_extra_items : bool | None, default None
        A boolean flag indicating whether items not described by the ``attrs`` parameter
        are allowed/disallowed. The default value of `None` is equivalent to
        ``True``.

    See Also
    --------
    AttrSchema
    """

    attrs: Mapping[str | re.Pattern, Any] | None = field(
        default=None, kw_only=False
    )
    allow_extra_items: bool | None = None

    def __post_init__(self):
        if self.attrs is not None:
            object.__setattr__(
                self,
                'attrs',
                {k: AttrSchema.convert(v) for k, v in self.attrs.items()},
            )

    @cached_property
    def serializer(self) -> Serializer:
        if self.attrs is None:
            return ObjectSerializer(
                additional_properties=self.allow_extra_items
            )
        properties = {}
        pattern_properties = {}
        required = set()
        for k, v in self.attrs.items():
            if isinstance(k, str):
                properties |= {k: v.serializer}
                if v.required:
                    required.add(k)
            elif isinstance(k, re.Pattern):
                pattern_properties |= {k.pattern: v.serializer}
        return ObjectSerializer(
            properties=properties,
            pattern_properties=pattern_properties,
            required=required,
            additional_properties=self.allow_extra_items,
        )

    def validate(self, attrs: Mapping[str, Any]) -> None:
        return super()._validate(instance=attrs)
