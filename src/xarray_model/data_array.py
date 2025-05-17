from collections.abc import Mapping
from typing import Any, Self

from dataclasses import dataclass
from numpy.typing import DTypeLike
import xarray as xr

from xarray_model.base import (
    BaseModel,
    DeserializationError,
    SerializationError,
)
from xarray_model.components import (
    AttrsModel,
    ChunksModel,
    DTypeModel,
    DimsModel,
    NameModel,
    SizesModel,
)
from xarray_model.types import (
    AttrsType,
    ChunksType,
    DimsType,
    NameType,
    SizesType,
)


@dataclass(frozen=True)
class CoordsModel(BaseModel):
    _title = 'Coords'
    _description = (
        'Mapping of DataArray objects corresponding to coordinate variables.'
    )

    coords: Mapping[str, 'DataArrayModel']
    require_all_keys: bool = True
    allow_extra_keys: bool = True

    def serialize(self) -> dict[str, Any]:
        schema = {
            'type': 'object',
            'properties': {
                key: coord.serialize() for key, coord in self.coords.items()
            },
            'additionalProperties': self.allow_extra_keys,
        }
        if self.require_all_keys:
            schema['required'] = list(self.coords.keys())
        return schema

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> Self:
        try:
            coords = {
                key: DataArrayModel.convert(coord)
                for key, coord in data['properties'].items()
            }
            allow_extra_keys = data.get('additionalProperties', True)
            require_all_keys = data.get('required', True)
        except KeyError as error:
            raise DeserializationError from error
        return cls(
            coords=coords,
            allow_extra_keys=allow_extra_keys,
            require_all_keys=require_all_keys,
        )


@dataclass(frozen=True)
class DataArrayModel(BaseModel):
    _title = 'DataArray'
    _description = (
        'N-dimensional array with labeled coordinates and dimensions.'
    )

    attrs: AttrsModel | AttrsType | None = None
    chunks: ChunksModel | ChunksType | None = None
    coords: CoordsModel | None = None
    dims: DimsModel | DimsType | None = None
    dtype: DTypeModel | DTypeLike | None = None
    name: NameModel | NameType | None = None
    sizes: SizesModel | SizesType | None = None

    def __post_init__(self) -> None:
        if self.attrs is not None:
            object.__setattr__(self, 'attrs', AttrsModel.convert(self.attrs))
        if self.chunks is not None:
            object.__setattr__(
                self, 'chunks', ChunksModel.convert(self.chunks)
            )
        if self.coords is not None:
            object.__setattr__(
                self, 'coords', CoordsModel.convert(self.coords)
            )
        if self.dims is not None:
            object.__setattr__(self, 'dims', DimsModel.convert(self.dims))
        if self.dtype is not None:
            object.__setattr__(self, 'dtype', DTypeModel.convert(self.dtype))
        if self.name is not None:
            object.__setattr__(self, 'name', NameModel.convert(self.name))
        if self.sizes is not None:
            object.__setattr__(self, 'sizes', SizesModel.convert(self.sizes))

    def serialize(self) -> dict[str, Any]:
        schema = {'type': 'object', 'additionalProperties': True}
        properties = {}
        try:
            if self.attrs is not None:
                properties.update(attrs=self.attrs.serialize())
            if self.chunks is not None:
                properties.update(chunks=self.chunks.serialize())
            if self.coords is not None:
                properties.update(coords=self.coords.serialize())
            if self.dims is not None:
                properties.update(dims=self.dims.serialize())
            if self.dtype is not None:
                properties.update(dtype=self.dtype.serialize())
            if self.name is not None:
                properties.update(name=self.name.serialize())
            if self.sizes is not None:
                properties.update(sizes=self.sizes.serialize())
        except Exception as error:
            raise SerializationError from error
        return schema | {'properties': properties}

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> Self:
        kwargs = {}
        properties = data.get('properties', {})
        if 'attrs' in properties:
            kwargs.update(attrs=AttrsModel.deserialize(properties['attrs']))
        if 'chunks' in properties:
            kwargs.update(chunks=ChunksModel.deserialize(properties['chunks']))
        if 'coords' in properties:
            kwargs.update(coords=CoordsModel.deserialize(properties['coords']))
        if 'dims' in properties:
            kwargs.update(dims=DimsModel.deserialize(properties['dims']))
        if 'dtype' in properties:
            kwargs.update(dtype=DTypeModel.deserialize(properties['dtype']))
        if 'name' in properties:
            kwargs.update(name=NameModel.deserialize(properties['name']))
        if 'sizes' in properties:
            kwargs.update(sizes=SizesModel.deserialize(properties['sizes']))
        return cls(**kwargs)

    def validate(self, data_array: xr.DataArray) -> None:
        return super()._validate(instance=data_array.to_dict())
