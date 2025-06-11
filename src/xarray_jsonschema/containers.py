from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
import re

import xarray as xr
from numpy.typing import DTypeLike

from xarray_jsonschema import ChunksSchema
from xarray_jsonschema.base import XarraySchema
from xarray_jsonschema.components import (
    AttrsSchema,
    DimsSchema,
    DTypeSchema,
    NameSchema,
    ShapeSchema,
    SizeSchema,
)
from xarray_jsonschema.serializers import (
    ObjectSerializer,
    Serializer,
)


# CoordsSchema and DataVarsSchema are components, but we define them here to avoid circular
# imports
@dataclass(frozen=True, repr=False, kw_only=True)
class CoordsSchema(XarraySchema):
    """Generate a JSON schema for xarray coordinates.

    Attributes
    ----------
    coords : Mapping[str | re.Pattern, DataArraySchema] | None, default None
        A mapping where keys are coordinate names and values are
        :py:class:`~xarray_jsonschema.DataArraySchema` objects specifying the
        properties of a specific coordinate. The default value of ``None``
        will skip validation of this property.
    require_all : bool, default True
        A boolean flag indicating if all the coordinates specified in the
        ``coords`` parameter are required or not.
    allow_extra : bool, default True
        A boolean flag indicating if coordinates not specified in the ``coords``
        parameter are allowed or not.
    """

    coords: Mapping[str | re.Pattern, 'DataArraySchema'] | None = field(
        default=None, kw_only=False
    )
    require_all: bool = True
    allow_extra: bool = True

    @cached_property
    def serializer(self) -> Serializer:
        if self.coords is None:
            return ObjectSerializer(additional_properties=self.allow_extra)
        properties = {}
        pattern_properties = {}
        required = set()
        for k, v in self.coords.items():
            if isinstance(k, str):
                properties |= {k: v.serializer}
                if self.require_all:
                    required.add(k)
            elif isinstance(k, re.Pattern):
                pattern_properties |= {k.pattern: v.serializer}
        return ObjectSerializer(
            properties=properties,
            pattern_properties=pattern_properties,
            required=required,
            additional_properties=self.allow_extra,
        )

    def validate(self, coords: Mapping[Hashable, xr.DataArray]) -> None:
        return super()._validate(instance=coords)


@dataclass(frozen=True, repr=False, kw_only=True)
class DataVarsSchema(XarraySchema):
    """Generate a JSON schema for xarray data variables.

    Attributes
    ----------
    data_vars : Mapping[str, 'DataArraySchema'] | None, default None
        A mapping where keys are variable names and values are
        :py:class:`~xarray_jsonschema.DataArraySchema` objects specifying the
        properties of a specific variable. The default value of ``None`` will
        skip validation of this property.
    require_all : bool, default True
        A boolean flag indicating if all the data variables specified in the
        ``data_vars`` parameter are required or not.
    allow_extra : bool, default True
        A boolean flag indicating if data variables not specified in the ``data_vars``
        parameter are allowed or not.
    """

    data_vars: Mapping[str, 'DataArraySchema'] = field(kw_only=False)
    require_all: bool = True
    allow_extra: bool = False

    @cached_property
    def serializer(self) -> Serializer:
        if self.data_vars is None:
            return ObjectSerializer(additional_properties=self.allow_extra)
        properties = {}
        pattern_properties = {}
        required = set()
        for k, v in self.data_vars.items():
            if isinstance(k, str):
                properties |= {k: v.serializer}
                if self.require_all:
                    required.add(k)
            elif isinstance(k, re.Pattern):
                pattern_properties |= {k.pattern: v.serializer}
        return ObjectSerializer(
            properties=properties,
            pattern_properties=pattern_properties,
            required=required,
            additional_properties=self.allow_extra,
        )

    def validate(self, data_vars: Mapping[Hashable, xr.DataArray]) -> None:
        return super()._validate(instance=data_vars)


@dataclass(frozen=True, repr=False, kw_only=True)
class DataArraySchema(XarraySchema):
    """Validate xarray DataArrays against a JSON schema.

    Parameters
    ----------
    attrs : AttrsSchema | None, default None
        Attributes definition describing the expected attributes of the DataArray.
    chunks : ChunksSchema | None, default None
        ChunksSchema definition describing the expected chunking of the DataArray dimensions.
        The default value of ``None`` will skip validation of this property.
    coords : CoordsSchema | None, default None
        Coordinates definition describing the expected coordinates of the DataArray.
        The default value of ``None`` will skip validation of this property.
    dims : DimsSchema | None, default None
        Dimensions definition describing the expected dimensions of the DataArray.
        The default value of ``None`` will skip validation of this property.
    dtype : DTypeSchema | None, default None
        Data-type definition describing the expected data type of the DataArray.
        The default value of ``None`` will skip validation of this property.
    name : NameSchema | None, default None
        NameSchema definition describing the expected name of the DataArray.
        The default value of ``None`` will skip validation of this property.
    shape : ShapeSchema | None, default None
        ShapeSchema definition describing the expected shape of the DataArray dimensions.
        The default value of ``None`` will skip validation of this property.

    Notes
    -----
    For all parameters, the default value of ``None`` will skip validation of this property.
    To specify a `null` validation constraint, consult the documentation for the relevant `XarraySchema` class.

    """

    attrs: AttrsSchema | None = None
    chunks: (
        ChunksSchema | bool | int | Sequence[int | Sequence[int]] | None
    ) = None
    coords: CoordsSchema | Mapping[str, 'DataArraySchema'] | None = None
    dims: DimsSchema | Sequence[str | NameSchema] | None = None
    dtype: DTypeSchema | DTypeLike | None = None
    name: NameSchema | str | Iterable[str] | re.Pattern | None = None
    shape: ShapeSchema | Sequence[int | SizeSchema] | None = None

    def __post_init__(self):
        if self.chunks is not None:
            object.__setattr__(
                self, 'chunks', ChunksSchema.convert(self.chunks)
            )
        if self.dims is not None:
            object.__setattr__(self, 'dims', DimsSchema.convert(self.dims))
        if self.dtype is not None:
            object.__setattr__(self, 'dtype', DTypeSchema.convert(self.dtype))
        if self.name is not None:
            object.__setattr__(self, 'name', NameSchema.convert(self.name))
        if self.shape is not None:
            object.__setattr__(self, 'shape', ShapeSchema.convert(self.shape))
        if self.coords is not None:
            object.__setattr__(
                self, 'coords', CoordsSchema.convert(self.coords)
            )

    @cached_property
    def serializer(self) -> Serializer:
        return ObjectSerializer(
            title=self.title,
            description=self.description,
            properties={
                k: getattr(self, k).serializer
                for k in (
                    'attrs',
                    'chunks',
                    'coords',
                    'dims',
                    'dtype',
                    'name',
                    'shape',
                )
                if getattr(self, k) is not None
            },
        )

    def validate(self, data_array: xr.DataArray) -> None:
        """Validate a data-array against this schema.

        Parameters
        ----------
        data_array : xr.DataArray
            The data-array instance to validate.

        Raises
        ------
        jsonschema.exceptions.ValidationError
            If the data-array instance is invalid.
        """
        instance = data_array.to_dict(data=False)
        instance |= {'chunks': data_array.chunks}
        return super()._validate(instance=instance)

    def __call__(self, data_array: xr.DataArray) -> None:
        """A wrapper for the :py:meth:`DataArraySchema.validate` method."""
        self.validate(data_array=data_array)


@dataclass(frozen=True, repr=False, kw_only=True)
class DatasetSchema(XarraySchema):
    """Validate xarray Datasets against a JSON schema.

    Attributes
    ----------
    attrs : AttrsSchema | None, default None
        An ``AttrsSchema`` object specifying the attributes of
        the DataArray. The default value of ``None`` will allow any attributes.
    coords : CoordsSchema | None, default None
        A ``CoordsSchema`` object specifying the coordinates of
        the DataArray. The default value of ``None`` will allow any coordinates.
    data_vars : DataVarsSchema | None, default None
        A ``DataVarsSchema`` object specifying the data
        variables of the DataArray. The default value of ``None`` will allow
        any data variables.
    """

    coords: (
        CoordsSchema | Mapping[str | re.Pattern, 'DataArraySchema'] | None
    ) = None
    data_vars: (
        DataVarsSchema | Mapping[str | re.Pattern, 'DataArraySchema'] | None
    ) = None
    attrs: AttrsSchema | None = None

    def __post_init__(self):
        if self.coords is not None:
            object.__setattr__(
                self, 'coords', CoordsSchema.convert(self.coords)
            )
        if self.data_vars is not None:
            object.__setattr__(
                self, 'data_vars', DataVarsSchema.convert(self.data_vars)
            )

    @cached_property
    def serializer(self) -> Serializer:
        return ObjectSerializer(
            title=self.title,
            description=self.description,
            properties={
                k: getattr(self, k).serializer
                for k in (
                    'attrs',
                    'coords',
                    'data_vars',
                )
                if getattr(self, k) is not None
            },
            additional_properties=True,  # for extra items from Dataset.to_dict()
        )

    def validate(self, dataset: xr.Dataset) -> None:
        """Validate a dataset against this schema.

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset instance to validate.

        Raises
        ------
        jsonschema.exceptions.ValidationError
            If the dataset instance is invalid.
        """
        instance = dataset.to_dict(data=False)
        return super()._validate(instance=instance)

    def __call__(self, dataset: xr.Dataset) -> None:
        """A wrapper for the :py:meth:`DatasetSchema.validate` method."""
        self.validate(dataset=dataset)
