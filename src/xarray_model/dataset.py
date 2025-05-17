from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

import xarray as xr

from xarray_model.base import (
    BaseModel,
    DeserializationError,
    SerializationError,
)
from xarray_model.components import AttrsModel
from xarray_model.data_array import CoordsModel, DataArrayModel


@dataclass(frozen=True)
class DataVariablesModel:
    _title = 'DataVars'
    _description = (
        'Dictionary of DataArray objects corresponding to data variables'
    )

    variables: Mapping[str, 'DataArrayModel']
    require_all_keys: bool = True
    allow_extra_keys: bool = True

    def serialize(self) -> dict[str, Any]:
        schema = {
            'type': 'object',
            'properties': {
                key: variable.serialize()
                for key, variable in self.variables.items()
            },
            'additionalProperties': self.allow_extra_keys,
        }
        if self.require_all_keys:
            schema['required'] = list(self.variables.keys())
        return schema

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> Self:
        try:
            variables = {
                key: DataArrayModel.convert(coord)
                for key, coord in data['properties'].items()
            }
            allow_extra_keys = data.get('additionalProperties', True)
            require_all_keys = data.get('required', True)
        except KeyError as error:
            raise DeserializationError from error
        return cls(
            variables=variables,
            allow_extra_keys=allow_extra_keys,
            require_all_keys=require_all_keys,
        )


@dataclass(frozen=True)
class DatasetModel(BaseModel):
    _title = 'Dataset'
    _description = 'A multi-dimensional, in memory, array database.'

    coords: CoordsModel | None = None
    data_vars: DataVariablesModel | None = None
    attrs: AttrsModel | None = None
    require_all_keys: bool = True
    allow_extra_keys: bool = True

    def serialize(self) -> dict[str, Any]:
        schema = {'type': 'object', 'additionalProperties': True}
        properties = {}
        try:
            if self.attrs is not None:
                properties.update(attrs=self.attrs.serialize())
            if self.data_vars is not None:
                properties.update(data_vars=self.data_vars.serialize())
            if self.coords is not None:
                properties.update(coords=self.coords.serialize())
        except Exception as error:
            raise SerializationError from error
        return schema | {'properties': properties}

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> Self:
        kwargs = {}
        properties = data.get('properties', {})
        if 'attrs' in properties:
            kwargs.update(attrs=AttrsModel.deserialize(properties['attrs']))
        if 'variables' in properties:
            kwargs.update(
                chunks=DataVariablesModel.deserialize(properties['variables'])
            )
        if 'coords' in properties:
            kwargs.update(coords=CoordsModel.deserialize(properties['coords']))
        return cls(**kwargs)

    def validate(self, dataset: xr.Dataset) -> None:
        return super()._validate(instance=dataset.to_dict())
