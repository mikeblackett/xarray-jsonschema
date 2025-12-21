import genson as gs

from xarray_jsonschema.strategies import (
    Boolean,
    Constant,
    Integer,
    Null,
    Number,
    NumpyDType,
    Pattern,
    String,
    Tuple,
)


class XarraySchemaBuilder(gs.SchemaBuilder):
    DEFAULT_URI = 'https://json-schema.org/draft/2020-12/schema'
    EXTRA_STRATEGIES = (
        Boolean,
        Constant,
        NumpyDType,
        Integer,
        Null,
        Number,
        Pattern,
        String,
        Tuple,
    )
