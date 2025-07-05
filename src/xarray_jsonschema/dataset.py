from collections.abc import Mapping
from functools import cached_property
from typing import ClassVar

import xarray as xr

from xarray_jsonschema import (
    AttrSchema,
    XarraySchema,
)
from xarray_jsonschema import DataArraySchema
from xarray_jsonschema.utilities import mapping_to_objectserializer
from xarray_jsonschema.components import AttrsSchema
from xarray_jsonschema.data_array import CoordsSchema, DataArraySchema
from xarray_jsonschema.serializers import Serializer, ObjectSerializer


class DataVarsSchema(XarraySchema[xr.Dataset]):
    """Validate the data variables of a ``xarray.Dataset``."""

    def __init__(
        self,
        data_vars: Mapping[str, DataArraySchema] | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.data_vars = {} if data_vars is None else data_vars
        self.strict = strict

    @cached_property
    def serializer(self) -> Serializer:
        return mapping_to_objectserializer(self.data_vars, strict=self.strict)

    def validate(self, obj: xr.Dataset) -> None:
        instance = obj.to_dict(data=False)['data_vars']
        return super()._validate(instance=instance)


class DatasetSchema(XarraySchema[xr.Dataset]):
    """Validate a ``xarray.Dataset``.

    Parameters
    ----------
    data_vars : Mapping[str, DataArraySchema] | DataVarsSchema | None, default None
    coords : Mapping[str, DataArraySchema] | CoordsSchema | None, default None
    attrs : Mapping[str, AttrSchema | object] | AttrsSchema | None, default None
    title : str | None, default None
    description : str | None, default None

    Attributes
    ----------
    coords : CoordsSchema | None, default None
    attrs : AttrsSchema | None, default None
    data_vars : DataVarsSchema | None, default None
    title : str | None, default None
    description : str | None, default None
    """

    _components: ClassVar[tuple[str, ...]] = (
        'coords',
        'attrs',
        'data_vars',
    )

    def __init__(
        self,
        *,
        coords: Mapping[str, DataArraySchema] | CoordsSchema | None = None,
        data_vars: Mapping[str, DataArraySchema]
        | DataVarsSchema
        | None = None,
        attrs: Mapping[str, AttrSchema | object] | AttrsSchema | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        self.coords = CoordsSchema.convert(coords) if coords else None
        self.attrs = AttrsSchema.convert(attrs) if attrs else None
        self.data_vars = (
            DataVarsSchema.convert(data_vars) if data_vars else None
        )

        super().__init__(title=title, description=description)

    @cached_property
    def serializer(self) -> Serializer:
        return ObjectSerializer(
            title=self.title,
            description=self.description,
            properties={
                key: getattr(self, key).serializer
                for key in self._components
                if getattr(self, key) is not None
            },
        )

    def validate(self, obj: xr.Dataset) -> None:
        """Validate a ``Dataset`` against this schema.

        Parameters
        ----------
        obj : xr.Dataset
            The dataset instance to validate.

        Raises
        ------
        jsonschema.exceptions.ValidationError
            If the dataset instance is invalid.
        """
        instance = obj.to_dict(data=False)
        return super()._validate(instance=instance)
