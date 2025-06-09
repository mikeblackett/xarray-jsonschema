from collections.abc import Sequence

import hypothesis as hp
from hypothesis import strategies as st
from jsonschema import Validator

from xarray_model import Shape
from xarray_model.testing import dimension_shapes


@hp.given(
    shape=dimension_shapes(min_dims=1),
    min_dims=st.integers(min_value=0),
    max_dims=st.integers(min_value=0),
)
def test_schema_is_valid(
    shape: Sequence[int], min_dims: int, max_dims: int, validator: Validator
) -> None:
    """Should always produce a valid JSON Schema"""
    schema = Shape(shape, min_dims=min_dims, max_dims=max_dims).schema
    validator.check_schema(schema)
