# TODO: (mike) Custom error messages

from xarray_jsonschema._version import version as __version__
from xarray_jsonschema.base import XarraySchema
from xarray_jsonschema.exceptions import SchemaError, ValidationError
from xarray_jsonschema.schema import (
    AttrSchema,
    AttrsSchema,
    CoordsSchema,
    DataArraySchema,
    DatasetSchema,
    DataVarsSchema,
    # ChunksSchema,
    DimsSchema,
    DTypeSchema,
    NameSchema,
    ShapeSchema,
    SizeSchema,
)

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
