# TODO: (mike) Custom error messages

from xarray_jsonschema._version import version as __version__
from xarray_jsonschema.base import XarraySchema
from xarray_jsonschema.components import (
    AttrSchema,
    AttrsSchema,
    # ChunksSchema,
    DimsSchema,
    DTypeSchema,
    NameSchema,
    ShapeSchema,
    SizeSchema,
)
from xarray_jsonschema.data_array import CoordsSchema, DataArraySchema
from xarray_jsonschema.dataset import DatasetSchema, DataVarsSchema
from xarray_jsonschema.exceptions import SchemaError, ValidationError

__all__ = [
    '__version__',
    'AttrSchema',
    'AttrsSchema',
    # 'ChunksSchema',
    'CoordsSchema',
    'DTypeSchema',
    'DataArraySchema',
    'DataVarsSchema',
    'DatasetSchema',
    'DimsSchema',
    'NameSchema',
    'SchemaError',
    'ShapeSchema',
    'SizeSchema',
    'ValidationError',
    'XarraySchema',
]
