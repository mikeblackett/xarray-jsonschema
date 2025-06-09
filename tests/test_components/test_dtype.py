import hypothesis as hp
from jsonschema import Validator
from numpy.typing import DTypeLike

from xarray_model import DType
from xarray_model.testing import supported_dtype_likes


@hp.given(dtype_like=supported_dtype_likes())
def test_dtype_schema_is_valid(
    dtype_like: DTypeLike, validator: Validator
) -> None:
    """Should always produce a valid JSON Schema"""
    schema = DType(dtype_like).schema
    validator.check_schema(schema)
