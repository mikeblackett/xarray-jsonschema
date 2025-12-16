from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from typing_extensions import assert_never

from xarray_jsonschema.base import XarraySchema
from xarray_jsonschema.utilities import mapping_to_object_serializer
from xarray_jsonschema.encoders import encode_value
from xarray_jsonschema.serializers import (
    AnySerializer,
    ArraySerializer,
    ConstSerializer,
    EnumSerializer,
    IntegerSerializer,
    Serializer,
    StringSerializer,
    TypeSerializer,
)

__all__ = [
    'AttrsSchema',
    'AttrSchema',
    # 'ChunksSchema',
    'DTypeSchema',
    'DimsSchema',
    'NameSchema',
    'ShapeSchema',
    'SizeSchema',
]


class NameSchema(XarraySchema[xr.DataArray]):
    """Validate the name of a ``xarray.DataArray``.

    Attributes
    ----------

    """

    # TODO: (mike) Add support for Hashable names

    def __init__(
        self,
        name: str | Iterable[str] | None = None,
        *,
        regex: bool = False,
        min_length: int | None = None,
        max_length: int | None = None,
    ) -> None:
        """

        Parameters
        ----------
        name : str | Iterable[str] | None, default None
            Expected value or iterable of acceptable values to match the name against.
        regex : bool, default False
            A boolean flag indicating that the ``name`` parameter should be treated
            as a regex pattern. If the ``name`` is not a string, this parameter is
            silently ignored.
        min_length : int, default None
            Non-negative integer specifying the minimum length of the name.
        max_length : int, default None
            Non-negative integer specifying the maximum length of the name.
        """
        super().__init__()
        self.name = name
        self.regex = regex
        self.min_length = min_length
        self.max_length = max_length

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
            case str() if self.regex:
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

    def validate(self, obj: xr.DataArray) -> None:
        instance = obj.to_dict(data=False)['name']
        return super()._validate(instance)


class DimsSchema(XarraySchema[xr.DataArray]):
    """Validate the dimensions of a ``xarray.DataArray``.

    Notes
    -----
    This object validates the tuple of dimension names returned by
    :py:attr:`~xarray.DataArray.dims`, NOT the mapping of dimension names to
    sizes returned by :py:attr:`~xarray.Dataset.dims`.

    Parameters
    ----------
    dims : Sequence[str | NameSchema | None] | None, default None
        A sequence of expected names for the dimensions. The
        names can either be strings or ``NameSchema`` objects for more
        complex matching. Order matters.
    contains : str | NameSchema | None, default None
        A string or ``NameSchema`` object describing a name that must be included in the
        dimensions.
    max_dims : int | None, default None
        The maximum number of dimensions.
    min_dims : int | None, default None
        The minimum number of dimensions.

    See Also
    --------
    ShapeSchema
    NameSchema
    """

    # TODO: (mike) Add support for Hashable dimension names

    def __init__(
        self,
        dims: Sequence[str | NameSchema | None] | None = None,
        *,
        contains: str | NameSchema | None = None,
        min_dims: int | None = None,
        max_dims: int | None = None,
    ) -> None:
        super().__init__()
        self.dims = [NameSchema.convert(dim) for dim in dims or []]
        self.min_dims = len(self.dims) if min_dims is None else min_dims
        self.max_dims = max_dims
        self.contains = NameSchema.convert(contains) if contains else None

    @cached_property
    def serializer(self) -> Serializer:
        prefix_items = [name.serializer for name in self.dims]
        return ArraySerializer(
            items=False if prefix_items else StringSerializer(),
            prefix_items=prefix_items,
            max_items=self.max_dims,
            min_items=self.min_dims,
            contains=self.contains.serializer if self.contains else None,
        )

    def validate(self, obj: xr.DataArray) -> None:
        instance = obj.to_dict(data=False)['dims']
        return super()._validate(instance)


class AttrSchema(XarraySchema):
    """Generate a JSON schema for attribute key-value pairs.

    Parameters
    ----------
    value : Any | None, default None
        The expected attribute value or type. The default value of ``None`` will
        match any value.
    regex : bool, default False
        A boolean flag indicating that the key for this attr should be treated as a regex pattern.
    required: bool, default True
        A boolean flag indicating that the attribute is required.
    """

    def __init__(
        self,
        value: Any = None,
        *,
        regex: bool = False,
        required: bool = True,
    ) -> None:
        super().__init__()
        self.value = value
        self.regex = regex
        self.required = required

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
                raise NotImplementedError()
            case Sequence():
                # TODO: (mike) Implement array validation
                raise NotImplementedError()
            case _:
                return ConstSerializer(encode_value(self.value))

    def validate(self, _) -> None:  # type: ignore
        raise NotImplementedError()  # pragma: no cover


class AttrsSchema(XarraySchema):
    """Validate the attributes of a ``xarray.DataArray``.

    Parameters
    ----------
    attrs : Mapping[str, Any]
        A mapping of attribute names to values. The values can be
        ``AttrSchema`` instances for complex matching, or any other type for
        simple matching.
    strict : bool, default True
        A boolean flag indicating whether items not described by the ``attrs``
        parameter are allowed.

    See Also
    --------
    AttrSchema
    """

    def __init__(
        self,
        attrs: Mapping[str, Any] | None = None,
        *,
        strict: bool = False,
    ) -> None:
        self.attrs = (
            {key: AttrSchema.convert(value) for key, value in attrs.items()}
            if attrs
            else {}
        )
        self.strict = strict
        super().__init__()

    @cached_property
    def serializer(self) -> Serializer:
        return mapping_to_object_serializer(self.attrs, strict=self.strict)

    def validate(self, obj: xr.DataArray | xr.Dataset) -> None:
        instance = obj.to_dict(data=False)['attrs']
        return super()._validate(instance)


class DTypeSchema(XarraySchema[xr.DataArray]):
    """Validate the dtype of a ``xarray.DataArray``.

    Parameters
    ----------
    dtype : DTypeLike
        The expected data type of the array. This can be any dtype-like value
        accepted by ``numpy.dtype``.
    """

    def __init__(self, dtype: DTypeLike) -> None:
        super().__init__()
        self.dtype = np.dtype(dtype)

    @cached_property
    def serializer(self) -> Serializer:
        return ConstSerializer(self.dtype)

    def validate(self, obj: xr.DataArray) -> None:
        instance = obj.to_dict(data=False)['dtype']
        return super()._validate(instance)


class SizeSchema(XarraySchema[xr.DataArray]):
    """Generate a JSON schema for ``xarray.DataArray`` dimension sizes.

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

    def __init__(
        self,
        size: int | None = None,
        *,
        maximum: int | None = None,
        minimum: int | None = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.maximum = maximum
        self.minimum = minimum

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

    def validate(self, _) -> None:  # type: ignore
        raise NotImplementedError()


class ShapeSchema(XarraySchema[xr.DataArray]):
    """Validate the shape of a ``xarray.DataArray``.

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

    def __init__(
        self,
        shape: Sequence[int | SizeSchema] | None = None,
        *,
        min_dims: int | None = None,
        max_dims: int | None = None,
    ) -> None:
        super().__init__()
        self.shape = (
            [SizeSchema.convert(size) for size in shape] if shape else None
        )
        self.min_dims = min_dims
        self.max_dims = max_dims

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

    def validate(self, obj: xr.DataArray) -> None:
        instance = obj.to_dict(data=False)['shape']
        return super()._validate(instance)


# class _ChunkSchema(XarraySchema):
#     """Generate a JSON schema for ``xarray.DataArray`` chunk block sizes.
#
#     This model represents the tuple of block sizes for a single dimension, it
#     is supposed to be used inside the ChunksSchema schema.
#     """
#
#     def __init__(self, size: int | Sequence[int]) -> None:
#         super().__init__()
#         self.size = size
#
#     @cached_property
#     def serializer(self) -> Serializer:
#         match self.size:
#             case -1:
#                 # `-1` is a dask wildcard meaning "use the full dimension size."
#                 return ArraySerializer(
#                     items=IntegerSerializer(), min_items=1, max_items=1
#                 )
#             case int():
#                 # A single integer is used to represent a uniform chunk size
#                 # where all chunks should be equal to `size` except the last one,
#                 # which is free to vary.
#                 # TODO: (mike) this schema doesn't validate that the
#                 #  variable chunk is last in the array. That is currently not
#                 #  possible with JSON Schema (https://github.com/json-schema-org/json-schema-spec/issues/1060).
#                 #  We could opt for custom validation...
#                 return ArraySerializer(
#                     # First chunk is equal to `shape`.
#                     prefix_items=[ConstSerializer(self.size)],
#                     # Other chunks are `shape` or smaller...
#                     items=IntegerSerializer(maximum=self.size),
#                     # but there is only one "other" chunk.
#                     contains=IntegerSerializer(exclusive_maximum=self.size),
#                     max_contains=1,
#                     min_contains=0,
#                 )
#             case Sequence() if not isinstance(self.size, str):
#                 # Expect a full match of the provided shape to the chunk size.
#                 # Can use -1 as a wildcard.
#                 prefix_items = [
#                     IntegerSerializer()
#                     if size == -1
#                     else ConstSerializer(size)
#                     for size in self.size
#                 ]
#                 return ArraySerializer(
#                     prefix_items=prefix_items,
#                     items=False,
#                 )
#             case _:
#                 raise ValueError(
#                     f'Invalid size: {self.size}. '
#                     'Expected int or sequence of ints.'
#                 )
#
#     def validate(self, _) -> None:
#         raise NotImplementedError()
#
#
# class ChunksSchema(XarraySchema):
#     """Validate the chunking of a ``xarray.DataArray``.
#
#     Parameters
#     ----------
#     chunks : bool | int | Sequence[int | Sequence[int]] | None, default None
#         The expected chunking along each dimension. The schema logic depends on the argument :py:type:`type`:
#
#         - :py:type:`bool` simply validates whether the array is chunked or not;
#         - :py:type:`int` validates a **uniform** chunk size along **all** dimensions;
#         - :py:type:`Sequence[int]` validates a **uniform** chunk size along **each** dimension;
#         - :py:type:`Sequence[Sequence[int]]` validates an exact chunk size along each dimension;
#         - A value of :py:const:`-1` can be used in place of a positive integer to validate no chunking along a dimension;
#         - ``None`` validates any chunked/unchunked array.
#     """
#
#     def __init__(
#         self,
#         chunks: bool | int | Sequence[int | Sequence[int]] | None = None,
#     ) -> None:
#         super().__init__()
#         self.chunks = chunks
#
#     @cached_property
#     def serializer(self) -> Serializer:
#         match self.chunks:
#             case None:
#                 return AnySerializer()
#             case True:
#                 return ArraySerializer(
#                     items=ArraySerializer(items=IntegerSerializer())
#                 )
#             case False:
#                 return NullSerializer()
#             case int():
#                 return ArraySerializer(
#                     items=_ChunkSchema(self.chunks).serializer
#                 )
#             case Sequence():
#                 prefix_items = [
#                     _ChunkSchema(chunk).serializer for chunk in self.chunks
#                 ]
#                 return ArraySerializer(
#                     prefix_items=prefix_items,
#                     items=False,
#                 )
#             case _:  # pragma: no cover
#                 assert_never(self.chunks)  # type: ignore
#
#     def validate(self, chunks: Sequence[Sequence[int]] | None) -> None:
#         return super()._validate(instance=chunks)
