# TODO: (mike) Custom error messages

from xarray_jsonschema._version import version as __version__
from xarray_jsonschema.model import (
    AttrsModel,
    CoordsModel,
    DataArrayModel,
    DatasetModel,
    DataVarsModel,
    DimsModel,
    DTypeModel,
    Model,
    NameModel,
    ShapeModel,
)
from xarray_jsonschema.validator import SchemaError, ValidationError

__all__ = [
    '__version__',
    'AttrsModel',
    'CoordsModel',
    'DataArrayModel',
    'DatasetModel',
    'DataVarsModel',
    'DimsModel',
    'DTypeModel',
    'NameModel',
    'SchemaError',
    'ShapeModel',
    'ValidationError',
    'Model',
]
