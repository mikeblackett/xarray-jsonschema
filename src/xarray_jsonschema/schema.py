from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from typing_extensions import assert_never

from xarray_jsonschema._normalizers import (
    AnyNormalizer,
    ArrayNormalizer,
    ConstNormalizer,
    EnumNormalizer,
    IntegerNormalizer,
    Normalizer,
    ObjectNormalizer,
    StringNormalizer,
    TypeNormalizer,
    normalize_value,
)
from xarray_jsonschema.base import XarraySchema, mapping_to_object_normalizer

__all__ = [
    'AttrsSchema',
    'AttrSchema',
    'DTypeSchema',
    'DimsSchema',
    'NameSchema',
    'ShapeSchema',
    'SizeSchema',
    'CoordsSchema',
    'DataArraySchema',
    'DatasetSchema',
    'DataVarsSchema',
]


class NameSchema(XarraySchema[xr.DataArray]):
    """A validator for ``xarray`` object names.

    Parameters
    ----------
    name : str | Iterable['str | NameSchema'] | None, default None
        An expected value or iterable of expected values.
    regex : bool, default False
        A boolean flag indicating that the ``name`` parameter is a regex pattern.
        If ``name`` is not a string, this parameter is silently ignored.
    min_length : int, default None
        A non-negative integer specifying the minimum length of the name.
    max_length : int, default None
        A non-negative integer specifying the maximum length of the name.
    """

    # TODO: (mike) Add support for Hashable names

    name: str | Iterable['str | NameSchema'] | None
    regex: bool
    min_length: int | None
    max_length: int | None

    def __init__(
        self,
        name: str | Iterable['str | NameSchema'] | None = None,
        *,
        regex: bool = False,
        min_length: int | None = None,
        max_length: int | None = None,
    ) -> None:
        super().__init__()
        self._set(
            name=name,
            regex=regex,
            min_length=min_length,
            max_length=max_length,
        )

    @cached_property
    def _normalizer(self) -> Normalizer:
        # TODO: (mike) support validating 'null' names?
        schema = StringNormalizer(
            min_length=self.min_length,
            max_length=self.max_length,
        )
        match self.name:
            case None:
                pass
            case str() if self.regex:
                schema |= StringNormalizer(pattern=self.name)
            case str():
                schema = ConstNormalizer(self.name)
            case Iterable():
                schema = EnumNormalizer(self.name)
            case _:  # pragma: no cover
                raise TypeError(
                    f'unsupported "name" type: {type(self.name).__name__!r}'
                )
        return schema

    def validate(self, obj: xr.DataArray) -> None:
        instance = obj.to_dict(data=False)['name']
        return super()._validate(instance)


class DimsSchema(XarraySchema[xr.DataArray]):
    """A validator for ``xarray`` object dimensions.

    Parameters
    ----------
    dims : Sequence[str | NameSchema | None] | None, default None
        A sequence of expected dimension names.
        Order matters.
        ``None`` will match any value.
    contains : str | NameSchema | None, default None
        A string or ``NameSchema`` object describing a required dimension.
    max_dims : int | None, default None
        The maximum number of dimensions.
    min_dims : int | None, default None
        The minimum number of dimensions.

    Notes
    -----
    This object validates the tuple of dimension names returned by
    :py:attr:`~xarray.DataArray.dims`, NOT the mapping of dimension names to
    sizes returned by :py:attr:`~xarray.Dataset.dims`.

    See Also
    --------
    ShapeSchema
    NameSchema
    """

    # TODO: (mike) Add support for Hashable dimension names

    dims: list[NameSchema]
    contains: NameSchema | None
    min_dims: int | None
    max_dims: int | None

    def __init__(
        self,
        dims: Sequence[str | NameSchema | None] | None = None,
        *,
        contains: str | NameSchema | None = None,
        min_dims: int | None = None,
        max_dims: int | None = None,
    ) -> None:
        super().__init__()
        _dims = [NameSchema.from_python(dim) for dim in dims or []]
        self._set(
            dims=_dims,
            min_dims=len(_dims) if min_dims is None else min_dims,
            max_dims=max_dims,
            contains=NameSchema.from_python(contains) if contains else None,
        )

    @cached_property
    def _normalizer(self) -> Normalizer:
        prefix_items = [name._normalizer for name in self.dims]
        return ArrayNormalizer(
            items=False if prefix_items else StringNormalizer(),
            prefix_items=prefix_items,
            max_items=self.max_dims,
            min_items=self.min_dims,
            contains=self.contains._normalizer if self.contains else None,
        )

    def validate(self, obj: xr.DataArray) -> None:
        instance = obj.to_dict(data=False)['dims']
        return super()._validate(instance)


class AttrSchema(XarraySchema):
    """A validator for ``xarray`` object attribute values.

    Parameters
    ----------
    value : Any | None, default None
        The expected value or value type.
        ``None`` will match any value.
    regex : bool, default False
        A boolean flag indicating that the key for this attr is a regex pattern.
    required: bool, default True
        A boolean flag indicating that the attribute is required.
    """

    # TODO: Support enumerated values?

    value: Any
    regex: bool
    required: bool

    def __init__(
        self,
        value: Any = None,
        *,
        regex: bool = False,
        required: bool = True,
    ) -> None:
        super().__init__()
        self._set(value=value, regex=regex, required=required)

    @cached_property
    def _normalizer(self) -> Normalizer:
        match self.value:
            case type():
                return TypeNormalizer(self.value)
            case None:
                return AnyNormalizer()
            case str():
                # Prevent str being caught by Sequence()
                return ConstNormalizer(self.value)
            case Mapping():
                return AttrsSchema(self.value)._normalizer
            case Sequence():
                prefix_items = [
                    TypeNormalizer(v)
                    if isinstance(v, type)
                    else ConstNormalizer(v)
                    for v in self.value
                ]
                return ArrayNormalizer(prefix_items=prefix_items)
            case _:
                return ConstNormalizer(normalize_value(self.value))

    def validate(self, _) -> None:  # type: ignore
        raise NotImplementedError()  # pragma: no cover


class AttrsSchema(XarraySchema):
    """A validator for ``xarray`` object attributes.

    Parameters
    ----------
    attrs : Mapping[str, AttrSchema | Any]
        A mapping of attribute names to expected values.
    strict : bool, default True
        If `True`, extra items are not allowed.

    See Also
    --------
    AttrSchema
    """

    attrs: Mapping[str, AttrSchema]
    strict: bool

    def __init__(
        self,
        attrs: Mapping[str, AttrSchema | Any] | None = None,
        *,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self._set(
            attrs=(
                {
                    key: AttrSchema.from_python(value)
                    for key, value in attrs.items()
                }
                if attrs
                else {}
            ),
            strict=strict,
        )

    @cached_property
    def _normalizer(self) -> Normalizer:
        return mapping_to_object_normalizer(self.attrs, strict=self.strict)

    def validate(self, obj: xr.DataArray | xr.Dataset) -> None:
        instance = obj.to_dict(data=False)['attrs']
        return super()._validate(instance)


class DTypeSchema(XarraySchema[xr.DataArray]):
    """A validator for ``xarray`` object data types.

    Parameters
    ----------
    dtype : DTypeLike
        The expected data type.
        Can be any dtype-like value accepted by ``numpy.dtype``.
    """

    dtype: np.dtype

    def __init__(self, dtype: DTypeLike) -> None:
        super().__init__()
        self._set(dtype=np.dtype(dtype))

    @cached_property
    def _normalizer(self) -> Normalizer:
        return ConstNormalizer(self.dtype)

    def validate(self, obj: xr.DataArray) -> None:
        instance = obj.to_dict(data=False)['dtype']
        return super()._validate(instance)


class SizeSchema(XarraySchema[xr.DataArray]):
    """A validator for ``xarray` object`` object dimension sizes.

    Parameters
    ----------
    size : int | None, default None
        A non-negative integer specifying the expected size.
        The default value of ``None`` will validate any size.
    maximum : int | None, default None
        A non-negative integer specifying the maximum size.
    minimum : int | None, default None
        A non-negative integer specifying the minimum size.
    """

    size: int | None
    maximum: int | None
    minimum: int | None

    def __init__(
        self,
        size: int | None = None,
        *,
        maximum: int | None = None,
        minimum: int | None = None,
    ) -> None:
        super().__init__()
        self._set(size=size, maximum=maximum, minimum=minimum)

    @cached_property
    def _normalizer(self) -> Normalizer:
        match self.size:
            case None:
                return IntegerNormalizer(
                    maximum=self.maximum,
                    minimum=self.minimum,
                )
            case int():
                return ConstNormalizer(self.size)
            case _:  # pragma: no cover
                raise TypeError(
                    f'unsupported "size" type: {type(self.name).__name__!r}'
                )

    def validate(self, _) -> None:  # type: ignore
        raise NotImplementedError()


class ShapeSchema(XarraySchema[xr.DataArray]):
    """A validator for ``xarray`` object shapes.

    Parameters
    ----------
    shape : Sequence[int | SizeSchema] | None, default None
        A sequence of expected sizes for the dimensions of the array.
        Order matters.
        ``None`` will match any shape.
    min_dims : int | None, default None
        The minimum expected number of dimensions.
    max_dims : int | None, default None
        The maximum expected number of dimensions.
    """

    shape: Sequence[int | SizeSchema] | None
    min_dims: int | None
    max_dims: int | None

    def __init__(
        self,
        shape: Sequence[int | SizeSchema] | None = None,
        *,
        min_dims: int | None = None,
        max_dims: int | None = None,
    ) -> None:
        super().__init__()
        self._set(
            shape=(
                [SizeSchema.from_python(size) for size in shape]
                if shape
                else None
            ),
            min_dims=min_dims,
            max_dims=max_dims,
        )

    @cached_property
    def _normalizer(self) -> Normalizer:
        schema = ArrayNormalizer(
            items=IntegerNormalizer(),
            min_items=self.min_dims,
            max_items=self.max_dims,
        )
        match self.shape:
            case None:
                pass
            case Sequence():
                prefix_items = [
                    SizeSchema.from_python(size)._normalizer
                    for size in self.shape
                ]
                schema |= ArrayNormalizer(
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


class CoordsSchema(XarraySchema[xr.DataArray]):
    """A validator for ``xarray`` object coordinates.

    Parameters
    ----------
    coords : Mapping[str, 'DataArraySchema'] | None = None
        A mapping of coordinate names to ``DataArraySchema`` objects.
    strict : bool = False
        A boolean flag indicating if extra coordinates are allowed.
    """

    coords: Mapping[str, 'DataArraySchema']
    strict: bool

    def __init__(
        self,
        coords: Mapping[str, 'DataArraySchema'] | None = None,
        strict: bool = False,
        key: str | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        super().__init__(key=key, title=title, description=description)
        self._set(coords={} if coords is None else coords, strict=strict)

    @cached_property
    def _normalizer(self) -> Normalizer:
        return mapping_to_object_normalizer(self.coords, strict=self.strict)

    def validate(self, obj: xr.DataArray | xr.Dataset) -> None:
        instance = obj.to_dict(data=False)['coords']
        return super()._validate(instance=instance)


class DataArraySchema(XarraySchema[xr.DataArray]):
    """A validator for ``xarray.DataArray`` objects.

    Parameters
    ----------
    dims: Sequence[str | NameSchema] | DimsSchema | None = None
        The expected dimensions of the array
    attrs: Mapping[str, AttrSchema | object] | AttrsSchema | None = None
        The expected attributes of the array
    dtype: DTypeLike | DTypeSchema | None = None
        The expected data type of the array
    shape: Sequence[int | SizeSchema] | ShapeSchema | None = None
        The expected shape of the array
    coords: Mapping[str, 'DataArraySchema'] | CoordsSchema | None = None
        The expected coordinates of the array
    """

    _components: ClassVar[tuple[str, ...]] = (
        'dims',
        'attrs',
        'dtype',
        'shape',
        'coords',
        'name',
    )

    dims: DimsSchema | None
    attrs: AttrsSchema | None
    dtype: DTypeSchema | None
    shape: ShapeSchema | None
    coords: CoordsSchema | None
    name: NameSchema | None

    def __init__(
        self,
        *,
        dims: Sequence[str | NameSchema] | DimsSchema | None = None,
        attrs: Mapping[str, AttrSchema | object] | AttrsSchema | None = None,
        dtype: DTypeLike | DTypeSchema | None = None,
        shape: Sequence[int | SizeSchema] | ShapeSchema | None = None,
        coords: Mapping[str, 'DataArraySchema'] | CoordsSchema | None = None,
        name: str | Iterable[str] | NameSchema | None = None,
    ) -> None:
        super().__init__()
        self._set(
            dims=DimsSchema.from_python(dims) if dims else None,
            attrs=AttrsSchema.from_python(attrs) if attrs else None,
            dtype=DTypeSchema.from_python(dtype) if dtype else None,
            shape=ShapeSchema.from_python(shape) if shape else None,
            coords=CoordsSchema.from_python(coords) if coords else None,
            name=NameSchema.from_python(name) if name else None,
        )

    @cached_property
    def _normalizer(self) -> Normalizer:
        return ObjectNormalizer(
            title=self.title,
            description=self.description,
            properties={
                key: getattr(self, key).normalizer
                for key in self._components
                if getattr(self, key) is not None
            },
        )

    def validate(self, obj: xr.DataArray) -> None:
        instance = obj.to_dict(data=False)
        return super()._validate(instance=instance)


class DataVarsSchema(XarraySchema[xr.Dataset]):
    """A validator for ``xarray`` object data variables.

    Parameters
    ----------
    data_vars : Mapping[str, DataArraySchema] | None = None
        A mapping of variable names to ``DataArraySchema`` objects.
    strict : bool = False
        A boolean flag indicating if extra data variables are allowed.
    """

    data_vars: Mapping[str, DataArraySchema]
    strict: bool

    def __init__(
        self,
        data_vars: Mapping[str, DataArraySchema] | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self._set(
            data_vars={} if data_vars is None else data_vars,
            strict=strict,
        )

    @cached_property
    def _normalizer(self) -> Normalizer:
        return mapping_to_object_normalizer(self.data_vars, strict=self.strict)

    def validate(self, obj: xr.Dataset) -> None:
        instance = obj.to_dict(data=False)['data_vars']
        return super()._validate(instance=instance)


class DatasetSchema(XarraySchema[xr.Dataset]):
    """A validator for ``xarray.Dataset`` objects.

    Parameters
    ----------
    data_vars : Mapping[str, DataArraySchema] | DataVarsSchema | None, default None
        The expected data variables of this dataset.
    dims : DimsSchema | None = None
        The expected dimensionss of this dataset.
    coords : Mapping[str, DataArraySchema] | CoordsSchema | None, default None
         The expected coordinates of this dataset.
    attrs : Mapping[str, AttrSchema | object] | AttrsSchema | None, default None
         The expected attributes of this dataset.
    """

    _components: ClassVar[tuple[str, ...]] = (
        'coords',
        'attrs',
        'data_vars',
    )

    data_vars: DataVarsSchema | None
    dims: DimsSchema | None
    attrs: AttrsSchema | None
    coords: CoordsSchema | None

    def __init__(
        self,
        *,
        coords: Mapping[str, DataArraySchema] | CoordsSchema | None = None,
        data_vars: Mapping[str, DataArraySchema]
        | DataVarsSchema
        | None = None,
        dims: DimsSchema | Sequence[str | NameSchema | None] | None = None,
        attrs: Mapping[str, AttrSchema | object] | AttrsSchema | None = None,
    ) -> None:
        super().__init__()
        self._set(
            coords=CoordsSchema.from_python(coords) if coords else None,
            attrs=AttrsSchema.from_python(attrs) if attrs else None,
            data_vars=(
                DataVarsSchema.from_python(data_vars) if data_vars else None
            ),
            dims=DimsSchema.from_python(dims),
        )

    @cached_property
    def _normalizer(self) -> Normalizer:
        return ObjectNormalizer(
            title=self.title,
            description=self.description,
            properties={
                key: getattr(self, key).normalizer
                for key in self._components
                if getattr(self, key) is not None
            },
        )

    def validate(self, obj: xr.Dataset) -> None:
        instance = obj.to_dict(data=False)
        return super()._validate(instance=instance)


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
#     def _normalizer(self) -> Normalizer:
#         match self.size:
#             case -1:
#                 # `-1` is a dask wildcard meaning "use the full dimension size."
#                 return ArrayNormalizer(
#                     items=IntegerNormalizer(), min_items=1, max_items=1
#                 )
#             case int():
#                 # A single integer is used to represent a uniform chunk size
#                 # where all chunks should be equal to `size` except the last one,
#                 # which is free to vary.
#                 # TODO: (mike) this schema doesn't validate that the
#                 #  variable chunk is last in the array. That is currently not
#                 #  possible with JSON Schema (https://github.com/json-schema-org/json-schema-spec/issues/1060).
#                 #  We could opt for custom validation...
#                 return ArrayNormalizer(
#                     # First chunk is equal to `shape`.
#                     prefix_items=[ConstNormalizer(self.size)],
#                     # Other chunks are `shape` or smaller...
#                     items=IntegerNormalizer(maximum=self.size),
#                     # but there is only one "other" chunk.
#                     contains=IntegerNormalizer(exclusive_maximum=self.size),
#                     max_contains=1,
#                     min_contains=0,
#                 )
#             case Sequence() if not isinstance(self.size, str):
#                 # Expect a full match of the provided shape to the chunk size.
#                 # Can use -1 as a wildcard.
#                 prefix_items = [
#                     IntegerNormalizer()
#                     if size == -1
#                     else ConstNormalizer(size)
#                     for size in self.size
#                 ]
#                 return ArrayNormalizer(
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
#     @property
#     def _normalizer(self) -> Normalizer:
#         match self.chunks:
#             case None:
#                 return AnyNormalizer()
#             case True:
#                 return ArrayNormalizer(
#                     items=ArrayNormalizer(items=IntegerNormalizer())
#                 )
#             case False:
#                 return NullNormalizer()
#             case int():
#                 return ArrayNormalizer(
#                     items=_ChunkSchema(self.chunks).normalizer
#                 )
#             case Sequence():
#                 prefix_items = [
#                     _ChunkSchema(chunk).normalizer for chunk in self.chunks
#                 ]
#                 return ArrayNormalizer(
#                     prefix_items=prefix_items,
#                     items=False,
#                 )
#             case _:  # pragma: no cover
#                 assert_never(self.chunks)  # type: ignore
#
#     def validate(self, chunks: Sequence[Sequence[int]] | None) -> None:
#         return super()._validate(instance=chunks)
