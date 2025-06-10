from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import xarray as xr

from xarray_model import Chunks
from xarray_model.base import BaseModel, BaseSchema
from xarray_model.components import (
    Attrs,
    Dims,
    DType,
    Name,
    Shape,
)
from xarray_model.serializers import (
    ObjectSerializer,
    Serializer,
)


@dataclass(frozen=True, repr=False, kw_only=True)
class DataArrayModel(BaseModel):
    """A JSON Schema validator for :py:class:`xarray.DataArray`.

    Attributes
    ----------
    attrs : Attrs | None, default None
        An :py:class:`~xarray_model.Attrs` object specifying the attributes of
        the DataArray. The default value of ``None`` will skip validation of
        this property.
    chunks : Chunks | None, default None
        A :py:class:`~xarray_model.Chunks` object specifying the chunk sizes of the DataArray dimensions.
        The default value of ``None`` will skip validation of this property.
    coords : Coords | None, default None
        A :py:class:`~xarray_model.Coords` object specifying the coordinates of the DataArray.
        The default value of ``None`` will skip validation of this property.
    dims : Dims | None, default None
        A :py:class:`~xarray_model.Dims` object specifying the dimensions of the DataArray. The default
        value of ``None`` will skip validation of this property.
    dtype : DType | None, default None
        A :py:class:`~xarray_model.DType` object specifying the data type of the DataArray. The default
        value of ``None`` will skip validation of this property.
    name : Name | None, default None
        A :py:class:`~xarray_model.Name` object specifying the name of the DataArray. The default value
        of ``None`` will skip validation of this property.
    shape : Shape | None, default None
        A :py:class:`~xarray_model.Name` object specifying the shape of the DataArray. The default
        value of ``None`` will skip validation of this property.
    """

    attrs: Attrs | None = None
    chunks: Chunks | None = None
    coords: 'Coords | None' = None
    dims: Dims | None = None
    dtype: DType | None = None
    name: Name | None = None
    shape: Shape | None = None

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

    def validate(self, data_array: xr.DataArray | xr.Variable) -> None:
        instance = data_array.to_dict(data=False)
        instance |= {'chunks': data_array.chunks}
        return super()._validate(instance=instance)


@dataclass(frozen=True, repr=False, kw_only=True)
class Coords(BaseSchema):
    """A JSON Schema generator for :py:attr:`xarray.DataArray.coords` and
    :py:class:`xarray.Dataset.coords`.

    Attributes
    ----------
    coords : Mapping[str, DataArrayModel] | None, default None
        A mapping where keys are coordinate names and values are
        :py:class:`~xarray_model.DataArrayModel` objects specifying the
        properties of a specific coordinate. The default value of ``None``
        will skip validation of this property.
    require_all : bool, default True
        A boolean flag indicating if all the coordinates specified in the
        ``coords`` parameter are required or not.
    allow_extra : bool, default True
        A boolean flag indicating if coordinates not specified in the ``coords``
        parameter are allowed or not.
    """

    coords: Mapping[str, DataArrayModel] | None = field(
        default=None, kw_only=False
    )
    require_all: bool = True
    allow_extra: bool = True

    @cached_property
    def serializer(self) -> Serializer:
        if self.coords is None:
            return ObjectSerializer(additional_properties=self.allow_extra)
        return ObjectSerializer(
            properties={
                name: data_array.serializer
                for name, data_array in self.coords.items()
            }
            or None,
            required=list(self.coords.keys()) if self.require_all else None,
            additional_properties=self.allow_extra,
        )


@dataclass(frozen=True, repr=False, kw_only=True)
class DataVars(BaseSchema):
    """A JSON Schema generator for :py:attr:`xarray.Dataset.data_vars`.

    Attributes
    ----------
    data_vars : Mapping[str, 'DataArrayModel'] | None, default None
        A mapping where keys are variable names and values are
        :py:class:`~xarray_model.DataArrayModel` objects specifying the
        properties of a specific variable. The default value of ``None`` will
        skip validation of this property.
    require_all_keys : bool, default True
        A boolean flag indicating if all the data variables specified in the
        ``data_vars`` parameter are required or not.
    allow_extra_keys : bool, default True
        A boolean flag indicating if data variables not specified in the ``data_vars``
        parameter are allowed or not.
    """

    data_vars: Mapping[str, 'DataArrayModel'] = field(kw_only=False)
    require_all: bool = True
    allow_extra: bool = False

    @cached_property
    def serializer(self) -> Serializer:
        return ObjectSerializer(
            properties={
                name: data_array.serializer
                for name, data_array in self.data_vars.items()
            }
            or None,
            required=list(self.data_vars.keys()) if self.require_all else None,
            additional_properties=self.allow_extra,
        )


@dataclass(frozen=True, repr=False, kw_only=True)
class DatasetModel(BaseModel):
    """A JSON Schema validator for :py:class:`xarray.Dataset`.

    Attributes
    ----------
    attrs : Attrs | None, default None
        An :py:class:`~xarray_model.Attrs` object specifying the attributes of
        the DataArray. The default value of ``None`` will allow any attributes.
    coords : Coords | None, default None
        A :py:class:`~xarray_model.Coords` object specifying the coordinates of
        the DataArray. The default value of ``None`` will allow any coordinates.
    data_vars : DataVars | None, default None
        A :py:class:`~xarray_model.DataVars` object specifying the data
        variables of the DataArray. The default value of ``None`` will allow
        any data variables.
    """

    coords: Coords | None = None
    data_vars: DataVars | None = None
    attrs: Attrs | None = None

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

    def validate(self, data_array: Any) -> None:
        instance = data_array.to_dict(data=False)
        return super()._validate(instance=instance)
